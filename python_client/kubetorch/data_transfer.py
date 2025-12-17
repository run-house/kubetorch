"""
Data transfer utilities for kubetorch.

Provides a key-value store interface for transferring data to and from the cluster.
This module contains the core rsync functionality used throughout kubetorch.
"""

import asyncio
import fcntl
import os
import pty
import re
import select
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import urlparse

from kubernetes import client, config

import kubetorch.serving.constants as serving_constants

from kubetorch import globals
from kubetorch.logger import get_logger
from kubetorch.resources.compute.utils import _get_rsync_exclude_options, find_available_port, RsyncError
from kubetorch.resources.compute.websocket import WebSocketRsyncTunnel
from kubetorch.servers.http.utils import is_running_in_kubernetes
from kubetorch.utils import http_to_ws

logger = get_logger(__name__)


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
            namespace: Kubernetes namespace
            service_name: Optional service name for service-specific transfers
        """
        self.namespace = namespace
        self.service_name = service_name or "store"
        self._core_api = None

    @property
    def core_api(self):
        """Lazy load Kubernetes core API client."""
        if self._core_api is None:
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()
            self._core_api = client.CoreV1Api()
        return self._core_api

    def get_rsync_pod_url(self) -> str:
        """Get the rsync pod service URL."""
        return f"rsync://kubetorch-rsync.{self.namespace}.svc.cluster.local:{serving_constants.REMOTE_RSYNC_PORT}/data/{self.namespace}/{self.service_name}/"

    def get_base_rsync_url(self, local_port: int) -> str:
        """Get the base rsync URL for local connections."""
        return f"rsync://localhost:{local_port}/data/{self.namespace}/{self.service_name}"

    def get_websocket_info(self, local_port: Optional[int] = None) -> tuple:
        """Get websocket connection info for rsync tunnel."""
        rsync_local_port = local_port or serving_constants.LOCAL_NGINX_PORT
        base_url = globals.service_url()

        ws_url = f"{http_to_ws(base_url)}/rsync/{self.namespace}/"
        parsed_url = urlparse(base_url)

        # Choose a local ephemeral port for the tunnel
        start_from = (parsed_url.port or rsync_local_port) + 1
        websocket_port = find_available_port(start_from, max_tries=10)
        return websocket_port, ws_url

    def create_rsync_target_dir(self):
        """Create the subdirectory for this particular service in the rsync pod."""
        subdir = f"/data/{self.namespace}/{self.service_name}"

        label_selector = f"app={serving_constants.RSYNC_SERVICE_NAME}"
        pod_name = (
            self.core_api.list_namespaced_pod(namespace=self.namespace, label_selector=label_selector)
            .items[0]
            .metadata.name
        )
        subdir_cmd = f"kubectl exec {pod_name} -n {self.namespace} -- mkdir -p {subdir}"
        logger.info(f"Creating directory on rsync pod with cmd: {subdir_cmd}")
        subprocess.run(subdir_cmd, shell=True, check=True)

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
        """Execute rsync command with proper error handling."""
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
            raise RsyncError(
                rsync_cmd, final_exit_code, "", decoded_line if "decoded_line" in locals() else ""
            )
        else:
            # Exit code is 0 but no completion message - might be okay, but log
            logger.debug("Rsync completed without seeing completion message, but exit code was 0")

        # Log user-friendly success message
        if synced_items:
            items_str = ", ".join(synced_items)
            logger.info(f"{items_str} successfully synced to cluster")
        else:
            logger.info("Rsync operation completed successfully")

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
            # External upload via websocket tunnel
            websocket_port, ws_url = self.get_websocket_info(local_port)
            logger.debug(f"Opening WebSocket tunnel on port {websocket_port} to {ws_url}")

            with WebSocketRsyncTunnel(websocket_port, ws_url) as tunnel:
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
            websocket_port, ws_url = self.get_websocket_info(local_port)
            logger.debug(f"Opening WebSocket tunnel on port {websocket_port} to {ws_url}")

            with WebSocketRsyncTunnel(websocket_port, ws_url) as tunnel:
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
            # External download via websocket tunnel
            websocket_port, ws_url = self.get_websocket_info(local_port)
            logger.debug(f"Opening WebSocket tunnel on port {websocket_port} to {ws_url}")

            with WebSocketRsyncTunnel(websocket_port, ws_url) as tunnel:
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


class DataTransferClient:
    """High-level client for key-value store interface."""

    def __init__(self, namespace: Optional[str] = None, kubeconfig_path: Optional[str] = None):
        """
        Initialize the data transfer client.

        Args:
            namespace: Kubernetes namespace (defaults to global config)
            kubeconfig_path: Path to kubeconfig file (not used directly, for compatibility)
        """
        self.namespace = namespace or globals.config.namespace

    def put(
        self,
        key: str,
        src: Union[str, Path, List[Union[str, Path]]],
        contents: bool = False,
        filter_options: Optional[str] = None,
        force: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Upload files or directories to the cluster using a key-value store interface.

        Args:
            key: Storage key (e.g., "my-service/models", "shared/dataset"). Trailing slashes are stripped.
            src: Local file(s) or directory(s) to upload
            contents: If True, copy directory contents (adds trailing slashes for rsync)
            filter_options: Additional rsync filter options
            force: Force overwrite of existing files
            verbose: Show detailed progress
        """
        # Strip trailing slashes from key - keys are abstract, not directory paths
        key = key.rstrip("/")

        # Check if we're inside a service in the cluster and should auto-prepend service name
        import os

        kt_service_name = os.getenv("KT_SERVICE_NAME")
        in_cluster = is_running_in_kubernetes()

        # Parse the key to determine service and path
        service_name = None

        # If key starts with "/", it's an absolute path - don't prepend service name
        if key.startswith("/"):
            dest_path = key[1:]  # Remove leading slash
        elif kt_service_name and in_cluster and not key.startswith("/"):
            # We're inside a service in the cluster and key is relative - auto-prepend service name
            service_name = kt_service_name
            dest_path = key
        elif "/" in key:
            parts = key.split("/", 1)
            # Check if first part could be a service name
            if not parts[0].startswith(".") and not parts[0].startswith("/"):
                if "." not in parts[0] and "-" in parts[0]:
                    # Likely a service name
                    service_name = parts[0]
                    dest_path = parts[1] if len(parts) > 1 else ""
                else:
                    dest_path = key
            else:
                dest_path = key
        elif not key.startswith(".") and not key.startswith("/"):
            # Key has no "/" - check if it could be a service name
            if "." not in key and "-" in key:
                # Likely a service name (e.g., "my-service")
                service_name = key
                dest_path = ""
            else:
                dest_path = key
        else:
            dest_path = key

        # Create rsync client with appropriate service name
        rsync_client = RsyncClient(namespace=self.namespace, service_name=service_name or "store")

        if verbose:
            logger.info(f"Uploading to key '{key}' from {src}")

        # Convert Path objects to strings
        if isinstance(src, Path):
            src = str(src)
        elif isinstance(src, list):
            src = [str(s) if isinstance(s, Path) else s for s in src]

        # Prepend "store/" to the path if no service specified
        if not service_name:
            dest_path = f"store/{dest_path}"

        # If contents=True, add trailing slash to dest_path for rsync "copy contents" behavior
        # But don't convert empty string to "/" (which would be treated as absolute path)
        if contents and dest_path:
            dest_path = dest_path.rstrip("/") + "/"
        elif contents and not dest_path:
            # Empty dest_path with contents=True means copy contents to root - leave as empty string
            # RsyncClient will handle this correctly
            pass

        try:
            # Debug logging
            logger.debug(
                f"DataTransferClient.put: in_cluster={is_running_in_kubernetes()}, service_name={service_name}, dest_path={dest_path}, contents={contents}"
            )

            rsync_client.upload(
                source=src, dest=dest_path, contents=contents, filter_options=filter_options, force=force
            )

            if verbose:
                logger.info(f"Successfully stored at key '{key}'")

        except RsyncError as e:
            logger.error(f"Failed to store at key '{key}': {e}")
            raise

    def get(
        self,
        key: str,
        dest: Optional[Union[str, Path]] = None,
        contents: bool = False,
        filter_options: Optional[str] = None,
        force: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Download files or directories from the cluster using a key-value store interface.

        Args:
            key: Storage key to retrieve. Trailing slashes are stripped.
            dest: Local destination path (defaults to current working directory). Trailing slashes are stripped unless contents=True.
            contents: If True, copy directory contents (adds trailing slashes to both source and dest for rsync)
            filter_options: Additional rsync filter options
            force: Force overwrite of existing files
            verbose: Show detailed progress
        """
        import os

        # Default to current working directory if dest not specified
        if dest is None:
            dest = os.getcwd()

        # Strip trailing slashes from key - keys are abstract, not directory paths
        key = key.rstrip("/")

        # Check if we're inside a service in the cluster and should auto-prepend service name
        kt_service_name = os.getenv("KT_SERVICE_NAME")
        in_cluster = is_running_in_kubernetes()

        # Parse the key to determine service and path
        service_name = None

        # If key starts with "/", it's an absolute path - don't prepend service name
        if key.startswith("/"):
            source_path = key[1:]  # Remove leading slash
        elif kt_service_name and in_cluster and not key.startswith("/"):
            # We're inside a service in the cluster and key is relative - auto-prepend service name
            service_name = kt_service_name
            source_path = key
        elif "/" in key:
            parts = key.split("/", 1)
            # Check if first part could be a service name
            if not parts[0].startswith(".") and not parts[0].startswith("/"):
                if "." not in parts[0] and "-" in parts[0]:
                    # Likely a service name
                    service_name = parts[0]
                    source_path = parts[1] if len(parts) > 1 else ""
                else:
                    source_path = key
            else:
                source_path = key
        elif not key.startswith(".") and not key.startswith("/"):
            # Key has no "/" - check if it could be a service name
            if "." not in key and "-" in key:
                # Likely a service name (e.g., "my-service")
                service_name = key
                source_path = ""
            else:
                source_path = key
        else:
            source_path = key

        # Create rsync client with appropriate service name
        rsync_client = RsyncClient(namespace=self.namespace, service_name=service_name or "store")

        # Convert Path to string and strip trailing slashes
        if isinstance(dest, Path):
            dest = str(dest)
        dest = dest.rstrip("/")

        if verbose:
            logger.info(f"Downloading from key '{key}' to {dest}")

        # Prepend "store/" to the path if no service specified
        if not service_name:
            source_path = f"store/{source_path}"

        # Build full rsync URL
        rsync_url = rsync_client.get_rsync_pod_url()
        if not rsync_url.endswith("/"):
            rsync_url += "/"
        remote_source = rsync_url + source_path.lstrip("/")

        # Determine if we're downloading a single file (key doesn't end with / and likely a file)
        # For single file downloads, ensure destination ends with / so rsync treats it as a directory
        # and preserves the file structure or uses basename appropriately
        is_single_file_download = not contents and "/" in key and not key.endswith("/")

        # If contents=True, add trailing slashes to both source and dest for rsync "copy contents" behavior
        # If downloading a single file to a directory, add trailing slash to dest to ensure it's treated as a directory
        if contents:
            remote_source = remote_source.rstrip("/") + "/"
            dest = dest.rstrip("/") + "/"
        elif is_single_file_download:
            # For single file downloads, ensure destination is treated as a directory
            dest = dest.rstrip("/") + "/"

        try:
            rsync_client.download(
                source=remote_source, dest=dest, contents=contents, filter_options=filter_options, force=force
            )

            if verbose:
                logger.info(f"Successfully retrieved key '{key}'")

        except RsyncError as e:
            logger.error(f"Failed to retrieve key '{key}': {e}")
            raise

    def ls(self, key: str = "", verbose: bool = False) -> List[str]:
        """
        List files and directories under a key path in the store.

        Args:
            key: Storage key path to list (empty string for root). Trailing slashes are stripped.
            verbose: Show detailed progress

        Returns:
            List of file/directory paths under the key
        """
        import subprocess

        # Strip trailing slashes from key - keys are abstract, not directory paths
        key = key.rstrip("/")

        # Check if we're inside a service in the cluster and should auto-prepend service name
        import os

        kt_service_name = os.getenv("KT_SERVICE_NAME")
        in_cluster = is_running_in_kubernetes()

        # Parse the key to determine service and path
        service_name = None

        # If key starts with "/", it's an absolute path - don't prepend service name
        if key.startswith("/"):
            list_path = key[1:]  # Remove leading slash
        elif kt_service_name and in_cluster and not key.startswith("/"):
            # We're inside a service in the cluster and key is relative - auto-prepend service name
            service_name = kt_service_name
            list_path = key
        elif "/" in key:
            parts = key.split("/", 1)
            # Check if first part could be a service name
            if not parts[0].startswith(".") and not parts[0].startswith("/"):
                if "." not in parts[0] and "-" in parts[0]:
                    # Likely a service name
                    service_name = parts[0]
                    list_path = parts[1] if len(parts) > 1 else ""
                else:
                    list_path = key
            else:
                list_path = key
        else:
            # Check if the key itself looks like a service name (contains - and no .)
            if not key.startswith(".") and "." not in key and "-" in key:
                # Likely a service name
                service_name = key
                list_path = ""
            else:
                list_path = key

        if verbose:
            logger.info(f"Listing contents of key '{key}'")

        # Build the full path in the rsync pod
        # Files are stored at /data/{namespace}/{service_name}/{path}
        if service_name:
            full_path = (
                f"/data/{self.namespace}/{service_name}/{list_path}"
                if list_path
                else f"/data/{self.namespace}/{service_name}"
            )
        elif not key or key == "":
            # Listing root - show all services and store directory at namespace level
            full_path = f"/data/{self.namespace}"
        else:
            # No service specified, use store path
            full_path = f"/data/{self.namespace}/store/{list_path}" if list_path else f"/data/{self.namespace}/store"

        # Find the rsync pod using label selector (same as create_rsync_target_dir)
        # Create a temporary RsyncClient to access core_api
        temp_rsync_client = RsyncClient(namespace=self.namespace, service_name=service_name or "store")
        label_selector = f"app={serving_constants.RSYNC_SERVICE_NAME}"
        try:
            pod_list = temp_rsync_client.core_api.list_namespaced_pod(
                namespace=self.namespace, label_selector=label_selector
            )
            if not pod_list.items:
                logger.error(f"No rsync pod found with label selector '{label_selector}'")
                return []
            pod_name = pod_list.items[0].metadata.name
        except Exception as e:
            logger.error(f"Failed to find rsync pod: {e}")
            return []

        # Build the ls command
        ls_cmd = f"ls -la {full_path}"

        try:
            if in_cluster:
                # In-cluster: use direct kubectl exec
                from kubernetes import client as k8s_client, config as k8s_config

                k8s_config.load_incluster_config()
                v1 = k8s_client.CoreV1Api()

                from kubernetes.stream import stream

                resp = stream(
                    v1.connect_get_namespaced_pod_exec,
                    pod_name,
                    self.namespace,
                    command=["sh", "-c", ls_cmd],
                    stderr=True,
                    stdin=False,
                    stdout=True,
                    tty=False,
                )

                # Parse the ls output
                lines = resp.strip().split("\n")
            else:
                # External: use kubectl exec via subprocess
                kubectl_cmd = f"kubectl exec -n {self.namespace} {pod_name} -- {ls_cmd}"
                result = subprocess.run(kubectl_cmd, shell=True, capture_output=True, text=True)

                if result.returncode != 0:
                    error_msg = result.stderr.strip() if result.stderr else result.stdout.strip()
                    logger.error(f"Failed to list key '{key}' (path: {full_path}, pod: {pod_name}): {error_msg}")
                    # If directory doesn't exist, return empty list (not an error)
                    if "No such file or directory" in error_msg or "not found" in error_msg.lower():
                        if verbose:
                            logger.info(f"Directory {full_path} does not exist")
                        return []
                    return []

                lines = result.stdout.strip().split("\n") if result.stdout.strip() else []

            # Parse ls output to extract file/directory names
            files = []
            if not lines:
                if verbose:
                    logger.info(f"No output from ls command for path {full_path}")
                return []

            for line in lines[1:]:  # Skip the "total" line
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 9:
                    # The filename is in the last column(s)
                    filename = " ".join(parts[8:])
                    if filename not in [".", ".."]:
                        # Add trailing slash for directories
                        if line.startswith("d"):
                            filename += "/"
                        files.append(filename)
                elif verbose and line.strip():
                    # Log lines that don't match expected format
                    logger.debug(f"Unexpected ls output line: {line}")

            if verbose:
                logger.info(f"Found {len(files)} items under key '{key}'")

            return files

        except Exception as e:
            logger.error(f"Failed to list key '{key}': {e}")
            return []

    def rm(
        self,
        key: str,
        recursive: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Delete a file or directory from the store.

        Args:
            key: Storage key to delete. Trailing slashes are stripped.
            recursive: If True, delete directories recursively (like rm -r)
            verbose: Show detailed progress
        """
        import subprocess

        # Strip trailing slashes from key - keys are abstract, not directory paths
        key = key.rstrip("/")

        # Check if we're inside a service in the cluster and should auto-prepend service name
        import os

        kt_service_name = os.getenv("KT_SERVICE_NAME")
        in_cluster = is_running_in_kubernetes()

        # Parse the key to determine service and path (same logic as ls)
        service_name = None

        if key.startswith("/"):
            delete_path = key[1:]  # Remove leading slash
        elif kt_service_name and in_cluster and not key.startswith("/"):
            service_name = kt_service_name
            delete_path = key
        elif "/" in key:
            parts = key.split("/", 1)
            if not parts[0].startswith(".") and not parts[0].startswith("/"):
                if "." not in parts[0] and "-" in parts[0]:
                    service_name = parts[0]
                    delete_path = parts[1] if len(parts) > 1 else ""
                else:
                    delete_path = key
            else:
                delete_path = key
        else:
            # Check if the key itself looks like a service name
            if not key.startswith(".") and "." not in key and "-" in key:
                service_name = key
                delete_path = ""
            else:
                delete_path = key

        if verbose:
            logger.info(f"Deleting key '{key}'")

        # Build the full path in the rsync pod
        if service_name:
            full_path = (
                f"/data/{self.namespace}/{service_name}/{delete_path}"
                if delete_path
                else f"/data/{self.namespace}/{service_name}"
            )
        else:
            full_path = (
                f"/data/{self.namespace}/store/{delete_path}" if delete_path else f"/data/{self.namespace}/store"
            )

        # Find the rsync pod
        temp_rsync_client = RsyncClient(namespace=self.namespace, service_name=service_name or "store")
        label_selector = f"app={serving_constants.RSYNC_SERVICE_NAME}"
        try:
            pod_list = temp_rsync_client.core_api.list_namespaced_pod(
                namespace=self.namespace, label_selector=label_selector
            )
            if not pod_list.items:
                raise ValueError(f"No rsync pod found with label selector '{label_selector}'")
            pod_name = pod_list.items[0].metadata.name
        except Exception as e:
            logger.error(f"Failed to find rsync pod: {e}")
            raise

        # Build the rm command
        rm_flag = "-rf" if recursive else "-f"
        rm_cmd = f"rm {rm_flag} {full_path}"

        try:
            if in_cluster:
                from kubernetes import client as k8s_client, config as k8s_config

                k8s_config.load_incluster_config()
                v1 = k8s_client.CoreV1Api()
                from kubernetes.stream import stream

                stream(
                    v1.connect_get_namespaced_pod_exec,
                    pod_name,
                    self.namespace,
                    command=["sh", "-c", rm_cmd],
                    stderr=True,
                    stdin=False,
                    stdout=True,
                    tty=False,
                )

                if verbose:
                    logger.info(f"Successfully deleted key '{key}'")
            else:
                kubectl_cmd = f"kubectl exec -n {self.namespace} {pod_name} -- {rm_cmd}"
                result = subprocess.run(kubectl_cmd, shell=True, capture_output=True, text=True)

                if result.returncode != 0:
                    error_msg = result.stderr.strip() if result.stderr else result.stdout.strip()
                    # If file doesn't exist, that's okay (idempotent)
                    if "No such file or directory" in error_msg or "not found" in error_msg.lower():
                        if verbose:
                            logger.info(f"Key '{key}' does not exist (already deleted)")
                        return
                    raise RuntimeError(f"Failed to delete key '{key}': {error_msg}")

                if verbose:
                    logger.info(f"Successfully deleted key '{key}'")

        except Exception as e:
            logger.error(f"Failed to delete key '{key}': {e}")
            raise


def sync_workdir_from_store(namespace: str, service_name: str):
    """
    Sync files from the rsync pod into the current working directory inside the server pod.
    
    This function is called by http_server.py during pod startup to sync files that were
    uploaded via the KV interface (kt.put or DataTransferClient.put) into the server pod's working directory.
    
    Performs two download operations (potentially in parallel):
    - Regular files (excluding __absolute__*) into the working directory
    - Absolute path files (under __absolute__/...) into their absolute destinations
    
    Uses the DataTransferClient KV interface, which allows future scalability with peer-to-peer
    transfer via a central metadata store. When called from inside a pod, empty key "" auto-prepends
    the service name to download from the service's storage area.
    """
    import concurrent.futures
    from concurrent.futures import ThreadPoolExecutor
    import os

    # Use DataTransferClient KV interface for future scalability
    dt_client = DataTransferClient(namespace=namespace)

    def sync_regular_files():
        """Sync regular files (excluding __absolute__*) to current directory."""
        # Empty key "" auto-prepends service name when called from inside pod (KT_SERVICE_NAME is set)
        # contents=True copies contents into current directory
        # filter_options excludes __absolute__* files
        try:
            dt_client.get(
                key="",
                dest=".",
                contents=True,
                filter_options="--exclude='__absolute__*'",
            )
        except RsyncError as e:
            # If the service storage area doesn't exist yet, that's okay
            error_msg = str(e).lower()
            if "no such file or directory" in error_msg or "not found" in error_msg:
                logger.debug("Service storage area does not exist yet, skipping regular files sync")
            else:
                raise
        except Exception as e:
            # Catch any other exceptions and check error message
            error_msg = str(e).lower()
            if "no such file or directory" in error_msg or "not found" in error_msg:
                logger.debug("Service storage area does not exist yet, skipping regular files sync")
            else:
                raise

    def sync_absolute_files():
        """Sync absolute path files (under __absolute__/) to root filesystem."""
        # Check if __absolute__ directory exists by trying to list it
        try:
            items = dt_client.ls(key="__absolute__")
            if items:
                # Download __absolute__ contents to / with contents=True
                dt_client.get(key="__absolute__", dest="/", contents=True)
            else:
                logger.debug("No absolute path files to sync")
        except Exception as e:
            # If __absolute__ doesn't exist, that's okay
            error_msg = str(e).lower()
            if "no such file or directory" in error_msg or "not found" in error_msg:
                logger.debug("No absolute path files to sync")
            else:
                raise

    # KT_SERVICE_NAME should already be set by http_server.py, but verify it matches
    # This ensures DataTransferClient.get() with empty key will use the correct service name
    if os.environ.get("KT_SERVICE_NAME") != service_name:
        logger.warning(
            f"KT_SERVICE_NAME environment variable ({os.environ.get('KT_SERVICE_NAME')}) "
            f"does not match service_name parameter ({service_name}). Setting it to {service_name}."
        )
        os.environ["KT_SERVICE_NAME"] = service_name

    with ThreadPoolExecutor(max_workers=2) as executor:
        regular_future = executor.submit(sync_regular_files)
        absolute_future = executor.submit(sync_absolute_files)
        futures = [regular_future, absolute_future]
        for future in concurrent.futures.as_completed(futures):
            future.result()


# Create singleton instances for convenience
_default_client = None


def put(
    key: str,
    src: Union[str, Path, List[Union[str, Path]]],
    contents: bool = False,
    filter_options: Optional[str] = None,
    force: bool = False,
    verbose: bool = False,
    namespace: Optional[str] = None,
    kubeconfig_path: Optional[str] = None,
) -> None:
    """
    Upload files or directories to the cluster using a key-value store interface.

    Args:
        key: Storage key (trailing slashes are stripped)
        src: Local file(s) or directory(s) to upload
        contents: If True, copy directory contents (adds trailing slashes for rsync)
        filter_options: Additional rsync filter options
        force: Force overwrite of existing files
        verbose: Show detailed progress
        namespace: Kubernetes namespace
        kubeconfig_path: Path to kubeconfig file (for compatibility)

    Examples:
        >>> import kubetorch as kt
        >>> kt.put(key="model-v1/weights", src="./trained_model/")
        >>> kt.put(key="datasets/train.csv", src="./data/train.csv")
        >>> kt.put(key="my-service/config", src="./configs/", contents=True)  # Copy contents of configs/
    """
    global _default_client

    if _default_client is None or namespace or kubeconfig_path:
        _default_client = DataTransferClient(namespace=namespace, kubeconfig_path=kubeconfig_path)

    _default_client.put(
        key=key, src=src, contents=contents, filter_options=filter_options, force=force, verbose=verbose
    )


def get(
    key: str,
    dest: Optional[Union[str, Path]] = None,
    contents: bool = False,
    filter_options: Optional[str] = None,
    force: bool = False,
    verbose: bool = False,
    namespace: Optional[str] = None,
    kubeconfig_path: Optional[str] = None,
) -> None:
    """
    Download files or directories from the cluster using a key-value store interface.

    Args:
        key: Storage key to retrieve (trailing slashes are stripped)
        dest: Local destination path (defaults to current working directory). Trailing slashes are stripped unless contents=True.
        contents: If True, copy directory contents (adds trailing slashes to both source and dest for rsync)
        filter_options: Additional rsync filter options
        force: Force overwrite of existing files
        verbose: Show detailed progress
        namespace: Kubernetes namespace
        kubeconfig_path: Path to kubeconfig file (for compatibility)

    Examples:
        >>> import kubetorch as kt
        >>> kt.get(key="model-v1/weights")  # Downloads to current directory
        >>> kt.get(key="model-v1/weights", dest="./local_model")  # Creates local_model/weights/
        >>> kt.get(key="model-v1/weights", dest="./local_model", contents=True)  # Copies contents into local_model/
        >>> kt.get(key="datasets/train.csv", dest="./data/train.csv")  # Downloads to specific file
        >>> kt.get(key="my-service/outputs", dest="./results", contents=True)  # Copies contents into results/
    """
    global _default_client

    if _default_client is None or namespace or kubeconfig_path:
        _default_client = DataTransferClient(namespace=namespace, kubeconfig_path=kubeconfig_path)

    _default_client.get(
        key=key, dest=dest, contents=contents, filter_options=filter_options, force=force, verbose=verbose
    )


def ls(
    key: str = "", verbose: bool = False, namespace: Optional[str] = None, kubeconfig_path: Optional[str] = None
) -> List[str]:
    """
    List files and directories under a key path in the store.

    Examples:
        >>> import kubetorch as kt
        >>> kt.ls()  # List root of store
        >>> kt.ls("my-service")  # List contents of my-service
        >>> kt.ls("my-service/models")  # List models directory

    Returns:
        List of file/directory paths (directories have trailing /)
    """
    global _default_client

    if _default_client is None or namespace or kubeconfig_path:
        _default_client = DataTransferClient(namespace=namespace, kubeconfig_path=kubeconfig_path)

    return _default_client.ls(key=key, verbose=verbose)


def rm(
    key: str,
    recursive: bool = False,
    verbose: bool = False,
    namespace: Optional[str] = None,
    kubeconfig_path: Optional[str] = None,
) -> None:
    """
    Delete a file or directory from the store.

    Args:
        key: Storage key to delete. Trailing slashes are stripped.
        recursive: If True, delete directories recursively (like rm -r)
        verbose: Show detailed progress
        namespace: Kubernetes namespace
        kubeconfig_path: Path to kubeconfig file (for compatibility)

    Examples:
        >>> import kubetorch as kt
        >>> kt.rm("my-service/old-model.pkl")  # Delete a file
        >>> kt.rm("my-service/temp-data", recursive=True)  # Delete a directory
    """
    global _default_client

    if _default_client is None or namespace or kubeconfig_path:
        _default_client = DataTransferClient(namespace=namespace, kubeconfig_path=kubeconfig_path)

    _default_client.rm(key=key, recursive=recursive, verbose=verbose)


# Legacy rsync interface for backward compatibility
def rsync(
    source: Union[str, List[str]],
    dest: str,
    namespace: str,
    service_name: str,
    contents: bool = False,
    filter_options: Optional[str] = None,
    force: bool = False,
    local_port: Optional[int] = None,
):
    """
    Legacy rsync function for backward compatibility.
    Used by Compute class and other internal components.
    """
    client = RsyncClient(namespace=namespace, service_name=service_name)
    client.upload(
        source=source,
        dest=dest,
        contents=contents,
        filter_options=filter_options,
        force=force,
        local_port=local_port,
    )


async def rsync_async(
    source: Union[str, List[str]],
    dest: str,
    namespace: str,
    service_name: str,
    contents: bool = False,
    filter_options: Optional[str] = None,
    force: bool = False,
    local_port: Optional[int] = None,
):
    """
    Legacy async rsync function for backward compatibility.
    Used by Compute class and other internal components.
    """
    client = RsyncClient(namespace=namespace, service_name=service_name)
    await client.upload_async(
        source=source,
        dest=dest,
        contents=contents,
        filter_options=filter_options,
        force=force,
        local_port=local_port,
    )
