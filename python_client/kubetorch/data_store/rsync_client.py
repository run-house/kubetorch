"""
Core rsync functionality for data transfer.

This module provides the low-level RsyncClient class that handles direct rsync operations.
"""

import asyncio
import fcntl
import os
import pty
import re
import select
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import urlparse

import kubetorch.provisioning.constants as provisioning_constants

from kubetorch import globals
from kubetorch.logger import get_logger
from kubetorch.resources.compute.utils import _get_rsync_exclude_options, RsyncError
from kubetorch.serving.utils import is_running_in_kubernetes
from kubetorch.utils import http_to_ws

from .websocket_tunnel import TunnelManager

logger = get_logger(__name__)

# Rsync error codes that are transient and should be retried
# 12 = Error in rsync protocol data stream (connection dropped)
# 23 = Partial transfer due to error
# 30 = Timeout in data send/receive
# 35 = Timeout waiting for daemon connection (could be transient of data store slow to respond)
RETRYABLE_RSYNC_CODES = {12, 23, 30, 35}
RSYNC_MAX_RETRIES = 3
RSYNC_RETRY_DELAY = 2  # seconds


class RsyncClient:
    """Core rsync functionality for data transfer."""

    def __init__(
        self,
        namespace: str,
        service_name: Optional[str] = None,
    ):
        """
        Initialize the rsync client.

        Args:
            namespace (str): Kubernetes namespace (user namespace for data paths).
            service_name (str, optional): Optional service name for service-specific transfers. (Default: None)
        """
        self.namespace = namespace  # Namespace for both service and data paths
        self.service_name = service_name or "store"

    def get_rsync_pod_url(self) -> str:
        """Get the data store pod service URL."""
        # Service is in the same namespace as the data
        return f"rsync://{provisioning_constants.DATA_STORE_SERVICE_NAME}.{self.namespace}.svc.cluster.local:{provisioning_constants.REMOTE_RSYNC_PORT}/data/{self.namespace}/{self.service_name}/"

    def get_base_rsync_url(self, local_port: int) -> str:
        """Get the base rsync URL for local connections."""
        return f"rsync://localhost:{local_port}/data/{self.namespace}/{self.service_name}"

    def get_websocket_info(self, local_port: Optional[int] = None) -> tuple:
        """Get websocket connection info for rsync tunnel."""
        rsync_local_port = local_port or provisioning_constants.LOCAL_NGINX_PORT
        base_url = globals.service_url()

        ws_url = f"{http_to_ws(base_url)}/rsync/{self.namespace}/"
        parsed_url = urlparse(base_url)

        # Return a starting port for the tunnel - TunnelManager caches tunnels by
        # ws_url and reuses them across calls. The underlying WebSocketRsyncTunnel
        # has robust port finding logic, so we just provide a starting point
        start_from = (parsed_url.port or rsync_local_port) + 1
        return start_from, ws_url

    def create_rsync_target_dir(self):
        """Create the subdirectory for this particular service in the data store."""
        # Use the service_name as the key - the data store will create the directory
        # at the appropriate filesystem path
        key = self.service_name

        logger.info(f"Creating directory for key '{key}' via data store API")
        from kubetorch.data_store import DataStoreClient

        client = DataStoreClient(namespace=self.namespace)
        client.mkdir(key)

    def build_rsync_command(
        self,
        source: Union[str, List[str]],
        dest: str,
        rsync_local_port: Optional[int] = None,
        contents: bool = False,
        filter_options: Optional[str] = None,
        force: bool = False,
        is_download: bool = False,
        in_cluster: bool = False,
    ) -> str:
        """
        Build the rsync command.

        Args:
            source: Source path(s)
            dest: Destination path
            rsync_local_port: Local port for rsync (only used outside cluster)
            contents: If True, copy directory contents rather than directory
            filter_options: Additional rsync filter options
            force: Force overwrite
            is_download: True if downloading from cluster
            in_cluster: True if running inside Kubernetes cluster
        """
        if in_cluster:
            return self._build_in_cluster_rsync_cmd(source, dest, contents, filter_options, force, is_download)
        else:
            return self._build_external_rsync_cmd(
                source, dest, rsync_local_port, contents, filter_options, force, is_download
            )

    def _build_external_rsync_cmd(
        self,
        source: Union[str, List[str]],
        dest: str,
        rsync_local_port: int,
        contents: bool = False,
        filter_options: Optional[str] = None,
        force: bool = False,
        is_download: bool = False,
    ) -> str:
        """Build rsync command for external (outside cluster) execution."""
        base_url = self.get_base_rsync_url(rsync_local_port)

        if is_download:
            # For downloads, source is already a full rsync URL
            if isinstance(source, list):
                source_str = " ".join(source)
            else:
                source_str = source
            remote_dest = dest
        else:
            # For uploads, build the remote destination
            if dest:
                # Handle tilde prefix - treat as relative to home/working directory
                if dest.startswith("~/"):
                    dest = dest[2:]

                # Handle absolute vs relative paths
                if dest.startswith("/"):
                    # For absolute paths, store under special __absolute__ subdirectory
                    dest_for_rsync = f"__absolute__{dest}"
                else:
                    dest_for_rsync = dest
                remote_dest = f"{base_url}/{dest_for_rsync}"
            else:
                remote_dest = base_url

            # Process source paths and determine if we're uploading a file or directory
            source = [source] if isinstance(source, str) else source
            expanded_sources = []
            source_is_file = False
            for s in source:
                path = Path(s).expanduser().absolute()
                if not path.exists():
                    raise ValueError(f"Could not locate path to sync up: {s}")

                # Check if source is a file (not a directory)
                if not contents and path.is_file():
                    source_is_file = True

                path_str = str(path)
                if contents and path.is_dir() and not str(s).endswith("/"):
                    path_str += "/"
                expanded_sources.append(path_str)

            source_str = " ".join(expanded_sources)

            # Only append "/" to remote_dest if:
            # 1. Source is a directory (not a single file), OR
            # 2. Contents flag is set (for single files, this means put file inside directory at key), OR
            # 3. Dest already ends with "/"
            # Don't append "/" if uploading a single file without contents flag (key is exact destination path)
            if not remote_dest.endswith("/"):
                if not source_is_file or contents or (dest and dest.endswith("/")):
                    remote_dest += "/"

        # Build rsync options (includes must come before excludes in rsync)
        # Parse filter_options to separate includes and excludes
        include_opts = ""
        exclude_opts = ""
        if filter_options:
            # Use shlex to properly handle quoted values
            import shlex

            parts = shlex.split(filter_options)
            i = 0
            while i < len(parts):
                if parts[i] == "--include":
                    if i + 1 < len(parts):
                        include_opts += f" --include={parts[i+1]}"
                        i += 2
                    else:
                        i += 1
                elif parts[i] == "--exclude":
                    if i + 1 < len(parts):
                        exclude_opts += f" --exclude={parts[i+1]}"
                        i += 2
                    else:
                        i += 1
                elif parts[i].startswith("--include="):
                    include_opts += f" {parts[i]}"
                    i += 1
                elif parts[i].startswith("--exclude="):
                    exclude_opts += f" {parts[i]}"
                    i += 1
                else:
                    i += 1

        # Order: includes first, then default excludes, then custom excludes
        default_excludes = _get_rsync_exclude_options()
        rsync_options = f"{include_opts} {default_excludes} {exclude_opts}".strip()
        rsync_cmd = f"rsync -avL {rsync_options}"

        if force:
            rsync_cmd += " --ignore-times"

        if is_download:
            rsync_cmd += f" {source_str} {remote_dest}"
        else:
            rsync_cmd += f" {source_str} {remote_dest}"

        return rsync_cmd

    def _build_in_cluster_rsync_cmd(
        self,
        source: Union[str, List[str]],
        dest: str,
        contents: bool = False,
        filter_options: Optional[str] = None,
        force: bool = False,
        is_download: bool = False,
    ) -> str:
        """Build rsync command for in-cluster execution."""
        base_remote = self.get_rsync_pod_url()

        if is_download:
            # For downloads, source is the remote path
            if isinstance(source, list):
                source_str = " ".join(source)
            else:
                source_str = source
            remote_dest = dest
        else:
            # For uploads, build remote destination
            source = [source] if isinstance(source, str) else source

            # Handle tilde prefix in dest
            if dest and dest.startswith("~/"):
                dest = dest[2:]

            # Determine if source is a file or directory
            # Check if any source path exists and is a file (not a directory)
            source_is_file = False
            if not contents:
                for s in source:
                    src_path = Path(s)
                    if src_path.exists() and src_path.is_file():
                        source_is_file = True
                        break

            if contents:
                source = [s if s.endswith("/") or not Path(s).is_dir() else s + "/" for s in source]

            source_str = " ".join(source)

            if dest is None:
                remote = base_remote
            elif dest.startswith("rsync://"):
                remote = dest
            else:
                remote = base_remote + dest.lstrip("/")

            # Only append "/" if:
            # 1. Source is a directory (not a single file), OR
            # 2. Dest already ends with "/", OR
            # 3. Contents flag is set (for single files, this means put file inside directory at key)
            # Don't append "/" if uploading a single file without contents flag (key is exact destination path)
            if not remote.endswith("/"):
                if not source_is_file or contents or dest.endswith("/"):
                    remote += "/"

            remote_dest = remote

        # Build command (includes must come before excludes in rsync)
        # Parse filter_options to separate includes and excludes
        include_opts = ""
        exclude_opts = ""
        if filter_options:
            # Use shlex to properly handle quoted values
            import shlex

            parts = shlex.split(filter_options)
            i = 0
            while i < len(parts):
                if parts[i] == "--include":
                    if i + 1 < len(parts):
                        include_opts += f" --include={parts[i+1]}"
                        i += 2
                    else:
                        i += 1
                elif parts[i] == "--exclude":
                    if i + 1 < len(parts):
                        exclude_opts += f" --exclude={parts[i+1]}"
                        i += 2
                    else:
                        i += 1
                elif parts[i].startswith("--include="):
                    include_opts += f" {parts[i]}"
                    i += 1
                elif parts[i].startswith("--exclude="):
                    exclude_opts += f" {parts[i]}"
                    i += 1
                else:
                    i += 1

        # Order: includes first, then default excludes, then custom excludes
        default_excludes = _get_rsync_exclude_options()
        rsync_options = f"{include_opts} {default_excludes} {exclude_opts}".strip()
        rsync_cmd = f"rsync -av {rsync_options}"

        if force:
            rsync_cmd += " --ignore-times"

        if is_download:
            rsync_cmd += f" {source_str} {remote_dest}"
        else:
            rsync_cmd += f" {source_str} {remote_dest}"

        return rsync_cmd

    def run_rsync_command(self, rsync_cmd: str, create_target_dir: bool = True):
        """Execute rsync command with retry logic for transient errors."""
        last_error = None
        for attempt in range(RSYNC_MAX_RETRIES):
            try:
                return self._run_rsync_command_once(rsync_cmd, create_target_dir)
            except RsyncError as e:
                last_error = e
                if e.returncode in RETRYABLE_RSYNC_CODES and attempt < RSYNC_MAX_RETRIES - 1:
                    logger.warning(
                        f"Rsync failed with transient error (code {e.returncode}), "
                        f"retrying in {RSYNC_RETRY_DELAY}s (attempt {attempt + 1}/{RSYNC_MAX_RETRIES})"
                    )
                    time.sleep(RSYNC_RETRY_DELAY)
                else:
                    raise
        raise last_error

    def _run_rsync_command_once(self, rsync_cmd: str, create_target_dir: bool = True):
        """Execute rsync command with proper error handling (single attempt)."""
        logger.debug(f"Executing rsync command: {rsync_cmd}")
        backup_rsync_cmd = rsync_cmd

        # Extract source paths from rsync command for user-friendly logging
        # Rsync command format: rsync [options] source1 source2 ... dest
        # For uploads: sources are local paths, dest is rsync://...
        # For downloads: source is rsync://..., dest is local path
        try:
            import shlex

            cmd_parts = shlex.split(rsync_cmd)
            source_paths = []
            # Find all non-option, non-rsync arguments
            # Sources are paths that don't start with rsync:// (for uploads) or are rsync:// (for downloads)
            # We'll identify sources as paths before the rsync:// URL (for uploads) or the rsync:// URL itself (for downloads)
            found_rsync_url = False
            for part in cmd_parts:
                if part.startswith("rsync://"):
                    found_rsync_url = True
                    # For downloads, the rsync:// URL is the source
                    # For uploads, everything before this is the source
                    break
                elif not part.startswith("-") and part != "rsync" and not part.startswith("--"):
                    # This is a local path - for uploads it's a source, for downloads it's the dest
                    # We'll collect these and use them if we don't find an rsync:// URL (download case)
                    if not found_rsync_url:
                        source_paths.append(part)

            # If we found an rsync:// URL, sources are everything before it (for uploads)
            # If we didn't find one, we might be in a download case - but we can't easily tell
            # For now, just use the paths we collected
            # Get just the directory/file names (basenames) for cleaner logging
            synced_items = [Path(p).name for p in source_paths if p and Path(p).exists()]
        except Exception:
            synced_items = []

        # Add --mkpath if needed and not already present
        if "--mkpath" not in rsync_cmd and create_target_dir:
            # Add --mkpath for rsync 3.2.0+ (creates parent directories automatically)
            rsync_cmd = rsync_cmd.replace("rsync ", "rsync --mkpath ", 1)
            logger.debug(f"Added --mkpath flag: {rsync_cmd}")

        # Run with PTY for better output handling
        logger.debug(f"Rsync command: {rsync_cmd}")

        leader, follower = pty.openpty()
        proc = subprocess.Popen(
            shlex.split(rsync_cmd),
            stdout=follower,
            stderr=follower,
            text=True,
            close_fds=True,
        )
        os.close(follower)

        # Set to non-blocking mode
        flags = fcntl.fcntl(leader, fcntl.F_GETFL)
        fcntl.fcntl(leader, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        buffer = b""
        transfer_completed = False
        error_patterns = [
            r"rsync\(\d+\): error:",
            r"rsync error:",
            r"@ERROR:",
        ]
        error_regexes = [re.compile(pattern, re.IGNORECASE) for pattern in error_patterns]

        try:
            with os.fdopen(leader, "rb", buffering=0) as stdout:
                while True:
                    rlist, _, _ = select.select([stdout], [], [], 0.1)
                    if stdout in rlist:
                        try:
                            chunk = os.read(stdout.fileno(), 1024)
                        except BlockingIOError:
                            continue

                        if not chunk:  # EOF
                            break

                        buffer += chunk
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)
                            decoded_line = line.decode(errors="replace").strip()
                            logger.debug(f"{decoded_line}")

                            # Check for --mkpath not supported error
                            if (
                                create_target_dir
                                and ("rsync: --mkpath" in decoded_line or "rsync: unrecognized option" in decoded_line)
                                and not is_running_in_kubernetes()
                            ):
                                logger.warning(
                                    "Rsync failed: --mkpath is not supported, falling back to creating target dir. "
                                    "Please upgrade rsync to 3.2.0+ to improve performance."
                                )
                                proc.terminate()
                                self.create_rsync_target_dir()
                                return self.run_rsync_command(backup_rsync_cmd, create_target_dir=False)

                            for error_regex in error_regexes:
                                if error_regex.search(decoded_line):
                                    proc.terminate()
                                    raise RsyncError(rsync_cmd, 1, decoded_line, decoded_line)

                            if "total size is" in decoded_line and "speedup is" in decoded_line:
                                transfer_completed = True
                                # Parse the total size to verify something was transferred
                                # Format: "total size is 1,234 speedup is 1.23"
                                size_match = re.search(r"total size is\s+([\d,]+)", decoded_line)
                                if size_match:
                                    size_str = size_match.group(1).replace(",", "")
                                    try:
                                        total_size = int(size_str)
                                        if total_size == 0:
                                            logger.warning(
                                                "Rsync reported 0 bytes transferred - file may not exist on source"
                                            )
                                    except ValueError:
                                        pass  # Couldn't parse size, but completion message seen

                        if transfer_completed:
                            break

                    exit_code = proc.poll()
                    if exit_code is not None:
                        if exit_code != 0:
                            # Process exited with error - wait for it to finish and raise error
                            proc.wait()
                            raise RsyncError(
                                rsync_cmd, exit_code, "", decoded_line if "decoded_line" in locals() else ""
                            )
                        # Process exited successfully (code 0)
                        break

                # Wait for process to complete if it hasn't already
                final_exit_code = proc.poll()
                if final_exit_code is None:
                    # Process still running - wait for it
                    final_exit_code = proc.wait()
        except Exception as e:
            # Only terminate if process is still running
            if proc.poll() is None:
                proc.terminate()
            raise e

        # Check if transfer completed successfully
        # If we saw the completion message, consider it successful even if exit code is non-zero
        # (some rsync versions return non-zero codes like 20 for SIGUSR1 even on success)
        if transfer_completed:
            # Transfer completed successfully - ignore exit code
            pass
        elif final_exit_code != 0:
            # No completion message and non-zero exit code - this is an error
            raise RsyncError(rsync_cmd, final_exit_code, "", decoded_line if "decoded_line" in locals() else "")
        else:
            # Exit code is 0 but no completion message - might be okay, but log
            logger.warning("Rsync completed without seeing completion message - transfer may have failed silently")

        # Log user-friendly success message
        if synced_items:
            items_str = ", ".join(synced_items)
            logger.info(f"Synced {items_str}")
        else:
            logger.debug("Rsync operation completed successfully")

    async def run_rsync_command_async(self, rsync_cmd: str, create_target_dir: bool = True):
        """Execute rsync command asynchronously."""
        # For now, run synchronously in executor
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.run_rsync_command, rsync_cmd, create_target_dir)

    def upload(
        self,
        source: Union[str, List[str]],
        dest: str,
        contents: bool = False,
        filter_options: Optional[str] = None,
        force: bool = False,
        local_port: Optional[int] = None,
    ):
        """
        Upload files to the cluster.

        Args:
            source: Local file(s) or directory(s)
            dest: Remote destination path
            contents: If True, copy directory contents
            filter_options: Additional rsync filters
            force: Force overwrite
            local_port: Local port for websocket tunnel
        """
        in_cluster = is_running_in_kubernetes()
        logger.debug(f"RsyncClient.upload: in_cluster={in_cluster}, service={self.service_name}, dest={dest}")

        if in_cluster:
            # In-cluster upload
            rsync_cmd = self.build_rsync_command(
                source=source,
                dest=dest,
                contents=contents,
                filter_options=filter_options,
                force=force,
                in_cluster=True,
            )
            self.run_rsync_command(rsync_cmd)
        else:
            # External upload via websocket tunnel (reused across calls)
            start_port, ws_url = self.get_websocket_info(local_port)
            tunnel = TunnelManager.get_tunnel(ws_url, start_port)

            rsync_cmd = self.build_rsync_command(
                source=source,
                dest=dest,
                rsync_local_port=tunnel.local_port,
                contents=contents,
                filter_options=filter_options,
                force=force,
                in_cluster=False,
            )
            self.run_rsync_command(rsync_cmd)

    async def upload_async(
        self,
        source: Union[str, List[str]],
        dest: str,
        contents: bool = False,
        filter_options: Optional[str] = None,
        force: bool = False,
        local_port: Optional[int] = None,
    ):
        """Async version of upload."""
        if is_running_in_kubernetes():
            rsync_cmd = self.build_rsync_command(
                source=source,
                dest=dest,
                contents=contents,
                filter_options=filter_options,
                force=force,
                in_cluster=True,
            )
            await self.run_rsync_command_async(rsync_cmd)
        else:
            # External upload via websocket tunnel (reused across calls)
            start_port, ws_url = self.get_websocket_info(local_port)
            tunnel = TunnelManager.get_tunnel(ws_url, start_port)

            rsync_cmd = self.build_rsync_command(
                source=source,
                dest=dest,
                rsync_local_port=tunnel.local_port,
                contents=contents,
                filter_options=filter_options,
                force=force,
                in_cluster=False,
            )
            await self.run_rsync_command_async(rsync_cmd)

    def download(
        self,
        source: str,
        dest: str,
        contents: bool = False,
        filter_options: Optional[str] = None,
        force: bool = False,
        local_port: Optional[int] = None,
    ):
        """
        Download files from the cluster.

        Args:
            source: Remote source path (full rsync URL)
            dest: Local destination path
            contents: If True, copy directory contents
            filter_options: Additional rsync filters
            force: Force overwrite
            local_port: Local port for websocket tunnel
        """
        if is_running_in_kubernetes():
            # In-cluster download
            rsync_cmd = self.build_rsync_command(
                source=source,
                dest=dest,
                contents=contents,
                filter_options=filter_options,
                force=force,
                is_download=True,
                in_cluster=True,
            )
            self.run_rsync_command(rsync_cmd, create_target_dir=False)
        else:
            # External download via websocket tunnel (reused across calls)
            start_port, ws_url = self.get_websocket_info(local_port)
            tunnel = TunnelManager.get_tunnel(ws_url, start_port)

            # Replace the rsync pod URL with localhost URL for tunneling
            base_url = self.get_base_rsync_url(tunnel.local_port)
            rsync_pod_url = self.get_rsync_pod_url()

            if isinstance(source, str):
                source = source.replace(rsync_pod_url, base_url + "/")

            rsync_cmd = self.build_rsync_command(
                source=source,
                dest=dest,
                rsync_local_port=tunnel.local_port,
                contents=contents,
                filter_options=filter_options,
                force=force,
                is_download=True,
                in_cluster=False,
            )
            self.run_rsync_command(rsync_cmd, create_target_dir=False)
