"""
High-level client for key-value store interface.

This module provides the DataSyncClient class that provides a key-value store
interface on top of the low-level RsyncClient, with support for peer-to-peer
data transfer via a metadata server.
"""

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import kubetorch.serving.constants as serving_constants
from kubetorch import globals
from kubetorch.logger import get_logger
from kubetorch.resources.compute.utils import find_available_port, RsyncError
from kubetorch.servers.http.utils import is_running_in_kubernetes

from .key_utils import parse_key, ParsedKey
from .metadata_client import MetadataClient
from .rsync_client import RsyncClient

logger = get_logger(__name__)


class DataSyncError(Exception):
    """Exception raised for data sync operations (key-value store) errors."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


@dataclass
class SourceInfo:
    """Information about a data source for retrieval."""

    ip: Optional[str] = None
    pod_name: Optional[str] = None
    namespace: Optional[str] = None
    src_path: Optional[str] = None
    proxy_through_store: bool = False
    peer_ip: Optional[str] = None


class DataSyncClient:
    """High-level client for key-value store interface."""

    def __init__(self, namespace: Optional[str] = None, kubeconfig_path: Optional[str] = None):
        """
        Initialize the data sync client.

        Args:
            namespace: Kubernetes namespace (defaults to global config)
            kubeconfig_path: Path to kubeconfig file (not used directly, for compatibility)
        """
        self.namespace = namespace or globals.config.namespace
        self.metadata_client = MetadataClient(
            namespace=self.namespace, metadata_port=serving_constants.DATA_SYNC_METADATA_PORT
        )

    def _parse_key_for_put(self, key: str) -> ParsedKey:
        """Parse key for put operations (auto-prepends service name in-cluster)."""
        return parse_key(key, auto_prepend_service=True)

    def _parse_key_for_get(self, key: str) -> ParsedKey:
        """Parse key for get operations (does NOT auto-prepend service name)."""
        return parse_key(key, auto_prepend_service=False)

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
        parsed = self._parse_key_for_put(key)

        # Create rsync client with appropriate service name
        rsync_client = RsyncClient(namespace=self.namespace, service_name=parsed.service_name or "store")

        if verbose:
            logger.info(f"Uploading to key '{key}' from {src}")

        # Convert Path objects to strings
        if isinstance(src, Path):
            src = str(src)
        elif isinstance(src, list):
            src = [str(s) if isinstance(s, Path) else s for s in src]

        # Build destination path
        dest_path = parsed.storage_path

        # If contents=True, add trailing slash for rsync "copy contents" behavior
        if contents and dest_path:
            dest_path = dest_path.rstrip("/") + "/"

        try:
            logger.debug(
                f"DataSyncClient.put: in_cluster={is_running_in_kubernetes()}, "
                f"service_name={parsed.service_name}, dest_path={dest_path}, contents={contents}"
            )

            rsync_client.upload(
                source=src, dest=dest_path, contents=contents, filter_options=filter_options, force=force
            )

            # After successful upload, register with metadata server
            self._register_store_pod(key, verbose)

            if verbose:
                logger.info(f"Successfully stored at key '{key}'")

        except RsyncError as e:
            logger.error(f"Failed to store at key '{key}': {e}")
            raise

    def _register_store_pod(self, key: str, verbose: bool = False) -> None:
        """Register the store pod with metadata server after upload."""
        if not is_running_in_kubernetes():
            return

        try:
            rsync_client = RsyncClient(namespace=self.namespace, service_name="store")
            label_selector = f"app={serving_constants.DATA_SYNC_SERVICE_NAME}"
            pod_list = rsync_client.core_api.list_namespaced_pod(
                namespace=self.namespace, label_selector=label_selector
            )
            if pod_list.items:
                store_pod_ip = pod_list.items[0].status.pod_ip
                self.metadata_client.register_store_pod(key, store_pod_ip)
                if verbose:
                    logger.debug(f"Registered key '{key}' with metadata server (store pod IP: {store_pod_ip})")
        except Exception as e:
            logger.warning(f"Failed to register key '{key}' with metadata server: {e}")

    def get(
        self,
        key: str,
        dest: Optional[Union[str, Path]] = None,
        contents: bool = False,
        filter_options: Optional[str] = None,
        force: bool = False,
        seed_data: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Download files or directories from the cluster using a key-value store interface.

        Args:
            key: Storage key to retrieve. Trailing slashes are stripped.
            dest: Local destination path (defaults to current working directory).
            contents: If True, copy directory contents (adds trailing slashes for rsync)
            filter_options: Additional rsync filter options
            force: Force overwrite of existing files
            seed_data: If True, automatically call vput() after successful retrieval (default: True)
            verbose: Show detailed progress
        """
        # Default to current working directory if dest not specified
        if dest is None:
            dest = os.getcwd()

        dest = self._normalize_dest(dest, contents, key)
        parsed = self._parse_key_for_get(key)
        in_cluster = is_running_in_kubernetes()

        if verbose:
            logger.info(f"Downloading from key '{key}' to {dest}")

        # Get source information from metadata server
        source_info, has_store_backup = self._resolve_source(key, in_cluster, verbose)

        # Try to retrieve the data
        if in_cluster and source_info and source_info.ip:
            # In-cluster peer-to-peer transfer
            success = self._get_from_peer_in_cluster(
                key, parsed, source_info, dest, contents, filter_options, force, has_store_backup, verbose
            )
            if success:
                self._maybe_seed_data(key, dest, seed_data, verbose)
                return

        if not in_cluster and source_info and source_info.pod_name and not has_store_backup:
            # External peer-to-peer transfer (only when store doesn't have it)
            success = self._get_from_peer_external(
                key, parsed, source_info, dest, contents, filter_options, force, verbose
            )
            if success:
                return

        # Fall back to store pod
        if has_store_backup:
            self._get_from_store_pod(key, parsed, dest, contents, filter_options, force, in_cluster, verbose)
            self._maybe_seed_data(key, dest, seed_data, verbose)
        else:
            raise DataSyncError(f"Key '{key}' not found - no peer sources and no store pod backup available")

    def _normalize_dest(self, dest: Union[str, Path, List], contents: bool, key: str) -> str:
        """Normalize destination path for rsync."""
        if isinstance(dest, Path):
            dest = str(dest)
        elif isinstance(dest, list):
            dest = str(dest[0]) if dest else os.getcwd()
        elif not isinstance(dest, str):
            dest = str(dest)

        dest = dest.rstrip("/")

        # Add trailing slash for contents mode or single file downloads
        is_single_file = not contents and "/" in key and not key.endswith("/")
        if contents or is_single_file:
            dest = dest + "/"

        return dest

    def _resolve_source(self, key: str, in_cluster: bool, verbose: bool) -> tuple[Optional[SourceInfo], bool]:
        """Resolve the best source for retrieving a key."""
        source_info = None
        has_store_backup = False

        if in_cluster:
            raw_info = self.metadata_client.get_source_ip(key, external=False)
            if isinstance(raw_info, dict):
                source_info = SourceInfo(ip=raw_info.get("ip"), src_path=raw_info.get("src_path"))
            elif raw_info:
                source_info = SourceInfo(ip=raw_info)

            if verbose and source_info and source_info.ip:
                logger.info(f"Metadata server returned peer IP '{source_info.ip}' for key '{key}'")

            has_store_backup = self.metadata_client.has_store_pod(key)
        else:
            # External client - check store pod first
            has_store_backup = self.metadata_client.has_store_pod(key)

            if not has_store_backup:
                # Only check peer pods if store doesn't have it
                raw_info = self.metadata_client.get_source_ip(key, external=True)
                if raw_info and isinstance(raw_info, dict):
                    source_info = SourceInfo(
                        pod_name=raw_info.get("pod_name"),
                        namespace=raw_info.get("namespace", self.namespace),
                        src_path=raw_info.get("src_path"),
                        proxy_through_store=raw_info.get("proxy_through_store", False),
                        peer_ip=raw_info.get("peer_ip"),
                    )
                    if verbose:
                        logger.info(f"External client: checking peer pods for key '{key}': {raw_info}")

        return source_info, has_store_backup

    def _get_from_peer_in_cluster(
        self,
        key: str,
        parsed: ParsedKey,
        source_info: SourceInfo,
        dest: str,
        contents: bool,
        filter_options: Optional[str],
        force: bool,
        has_store_backup: bool,
        verbose: bool,
    ) -> bool:
        """
        Attempt peer-to-peer transfer within the cluster.

        Returns True if successful, False to fall back to store pod.
        """
        rsync_client = RsyncClient(namespace=self.namespace, service_name=parsed.service_name or "store")

        # Get the relative path for rsync
        src_path_relative = source_info.src_path
        if not src_path_relative:
            src_path_relative = key.split("/")[-1] if "/" in key else key

        if verbose and src_path_relative:
            logger.info(f"Source relative path: {src_path_relative}")

        # Build peer rsync URL
        peer_url = f"rsync://{source_info.ip}:{serving_constants.REMOTE_RSYNC_PORT}/data/"
        remote_source = peer_url + src_path_relative
        if contents:
            remote_source = remote_source.rstrip("/") + "/"

        try:
            if verbose:
                logger.info(f"Attempting peer-to-peer rsync from {source_info.ip}")

            rsync_client.download(
                source=remote_source, dest=dest, contents=contents, filter_options=filter_options, force=force
            )

            if verbose:
                logger.info(f"Successfully retrieved key '{key}' from peer {source_info.ip}")

            self.metadata_client.complete_request(key, source_info.ip)
            return True

        except RsyncError as e:
            logger.warning(f"Peer {source_info.ip} unreachable for key '{key}': {e}")
            self.metadata_client.remove_source(key, source_info.ip)
            self.metadata_client.complete_request(key, source_info.ip)

            if not has_store_backup:
                raise DataSyncError(
                    f"Peer {source_info.ip} unreachable and no store pod backup available for key '{key}'"
                )
            return False

    def _get_from_peer_external(
        self,
        key: str,
        parsed: ParsedKey,
        source_info: SourceInfo,
        dest: str,
        contents: bool,
        filter_options: Optional[str],
        force: bool,
        verbose: bool,
    ) -> bool:
        """
        Attempt peer-to-peer transfer from outside the cluster using port-forward.

        Returns True if successful, False otherwise.
        """
        if source_info.proxy_through_store or not source_info.pod_name:
            if verbose:
                logger.info("Proxying through store pod for external client")
            return False

        rsync_client = RsyncClient(namespace=self.namespace, service_name=parsed.service_name or "store")
        pod_name = source_info.pod_name
        pod_namespace = source_info.namespace or self.namespace

        try:
            # Resolve service name to actual pod name if needed
            if pod_name == serving_constants.DATA_SYNC_SERVICE_NAME:
                label_selector = f"app={serving_constants.DATA_SYNC_SERVICE_NAME}"
                pod_list = rsync_client.core_api.list_namespaced_pod(
                    namespace=pod_namespace, label_selector=label_selector
                )
                if not pod_list.items:
                    raise RuntimeError(f"No {serving_constants.DATA_SYNC_SERVICE_NAME} pod found")
                pod_name = pod_list.items[0].metadata.name

            if verbose:
                logger.info(f"Using port-forward to connect to peer pod {pod_name}")

            # Set up port-forward
            local_port = find_available_port(serving_constants.REMOTE_RSYNC_PORT + 1000)
            pf_cmd = [
                "kubectl",
                "port-forward",
                f"pod/{pod_name}",
                f"{local_port}:{serving_constants.REMOTE_RSYNC_PORT}",
                "-n",
                pod_namespace,
            ]

            pf_process = subprocess.Popen(
                pf_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )

            time.sleep(2)
            if pf_process.poll() is not None:
                stderr = pf_process.stderr.read().decode() if pf_process.stderr else ""
                raise RuntimeError(f"Port-forward failed: {stderr}")

            try:
                return self._download_via_port_forward(
                    key, source_info, local_port, dest, contents, filter_options, force, rsync_client, verbose
                )
            finally:
                try:
                    pf_process.terminate()
                    pf_process.wait(timeout=5)
                except Exception:
                    pf_process.kill()
                    pf_process.wait()

        except Exception as e:
            logger.warning(f"Failed to use port-forward to peer pod: {e}")
            return False

    def _download_via_port_forward(
        self,
        key: str,
        source_info: SourceInfo,
        local_port: int,
        dest: str,
        contents: bool,
        filter_options: Optional[str],
        force: bool,
        rsync_client: RsyncClient,
        verbose: bool,
    ) -> bool:
        """Download data via an established port-forward connection."""
        src_path_relative = source_info.src_path
        if not src_path_relative:
            src_path_relative = key.split("/")[-1] if "/" in key else key

        remote_source = f"rsync://localhost:{local_port}/data/{src_path_relative}"
        if contents:
            remote_source = remote_source.rstrip("/") + "/"

        if verbose:
            logger.info(f"Rsync source URL: {remote_source}, destination: {dest}")

        # Track files before download
        dest_path = Path(dest.rstrip("/"))
        files_before = set()
        if dest_path.is_dir():
            files_before = {f.name for f in dest_path.glob("*") if f.is_file()}

        rsync_cmd = rsync_client.build_rsync_command(
            source=remote_source,
            dest=dest,
            rsync_local_port=local_port,
            contents=contents,
            filter_options=filter_options,
            force=force,
            is_download=True,
            in_cluster=False,
        )

        rsync_client.run_rsync_command(rsync_cmd, create_target_dir=False)

        # Verify files were downloaded
        if dest_path.is_dir():
            files_after = {f.name for f in dest_path.glob("*") if f.is_file()}
            new_files = files_after - files_before
            if verbose:
                logger.info(f"New files downloaded: {sorted(new_files)}")
            if not new_files:
                raise DataSyncError(f"Rsync completed but no files were transferred from peer for key '{key}'")
        elif not dest_path.exists():
            raise DataSyncError(f"Destination does not exist after download: {dest_path}")

        if verbose:
            logger.info(f"Successfully retrieved key '{key}' from peer via port-forward")

        if source_info.peer_ip:
            self.metadata_client.complete_request(key, source_info.peer_ip)

        return True

    def _get_from_store_pod(
        self,
        key: str,
        parsed: ParsedKey,
        dest: str,
        contents: bool,
        filter_options: Optional[str],
        force: bool,
        in_cluster: bool,
        verbose: bool,
    ) -> None:
        """Download from the store pod."""
        rsync_client = RsyncClient(namespace=self.namespace, service_name=parsed.service_name or "store")

        rsync_url = rsync_client.get_rsync_pod_url()
        if not rsync_url.endswith("/"):
            rsync_url += "/"

        remote_source = rsync_url + parsed.storage_path.lstrip("/")
        if contents:
            remote_source = remote_source.rstrip("/") + "/"

        if verbose:
            logger.info(f"Downloading from store pod: {remote_source} to {dest}")

        try:
            rsync_client.download(
                source=remote_source, dest=dest, contents=contents, filter_options=filter_options, force=force
            )

            if verbose:
                logger.info(f"Successfully retrieved key '{key}'")

            # Notify completion
            if in_cluster:
                self._notify_store_completion(key)

        except RsyncError as e:
            logger.error(f"Failed to retrieve key '{key}' from store pod: {e}")
            raise

    def _notify_store_completion(self, key: str) -> None:
        """Notify metadata server that store pod request completed."""
        try:
            rsync_client = RsyncClient(namespace=self.namespace, service_name="store")
            label_selector = f"app={serving_constants.DATA_SYNC_SERVICE_NAME}"
            pod_list = rsync_client.core_api.list_namespaced_pod(
                namespace=self.namespace, label_selector=label_selector
            )
            if pod_list.items:
                store_pod_ip = pod_list.items[0].status.pod_ip
                self.metadata_client.complete_request(key, store_pod_ip)
        except Exception as e:
            logger.debug(f"Failed to notify store pod completion: {e}")

    def _maybe_seed_data(self, key: str, dest: str, seed_data: bool, verbose: bool) -> None:
        """Optionally seed data after successful retrieval."""
        if not seed_data or not is_running_in_kubernetes():
            return

        dest_path = Path(dest.rstrip("/"))
        if not dest_path.exists():
            if verbose:
                logger.debug(f"Skipping auto-seed - destination does not exist: {dest_path}")
            return

        try:
            if verbose:
                logger.info(f"Auto-seeding key '{key}' after successful retrieval")
            self.vput(key=key, src=dest_path, start_rsyncd=True, verbose=verbose)
            if verbose:
                logger.info(f"Successfully seeded key '{key}'")
        except Exception as e:
            logger.warning(f"Failed to auto-seed key '{key}': {e}")

    def ls(self, key: str = "", verbose: bool = False) -> List[dict]:
        """
        List files and directories under a key path in the store.

        Args:
            key: Storage key path to list (empty string for root).
            verbose: Show detailed progress

        Returns:
            List of dicts with item information.
        """
        parsed = parse_key(key, auto_prepend_service=True)

        if verbose:
            logger.info(f"Listing contents of key '{key}' (query: '{parsed.full_key}')")

        try:
            result = self.metadata_client.list_keys(prefix=parsed.full_key)
            items = result.get("items", [])

            if verbose:
                logger.info(f"Found {len(items)} items under key '{key}'")

            return items
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
            key: Storage key to delete.
            recursive: If True, delete directories recursively
            verbose: Show detailed progress
        """
        parsed = parse_key(key, auto_prepend_service=True)

        if verbose:
            logger.info(f"Deleting key '{key}'")

        result = self.metadata_client.delete_key(parsed.full_key, recursive=recursive)

        if not result.get("success", False):
            error = result.get("error", "Unknown error")
            raise RuntimeError(f"Failed to delete key '{key}': {error}")

        if verbose:
            deleted_meta = result.get("deleted_from_metadata", False)
            deleted_fs = result.get("deleted_from_filesystem", False)
            if deleted_meta and deleted_fs:
                logger.info(f"Deleted key '{key}' from metadata and filesystem")
            elif deleted_meta:
                logger.info(f"Deleted virtual key '{key}' from metadata")
            elif deleted_fs:
                logger.info(f"Deleted key '{key}' from filesystem")
            else:
                logger.info(f"Key '{key}' does not exist")

    def vput(
        self,
        key: str,
        src: Union[str, Path],
        start_rsyncd: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Virtual put - publish that this pod has data for the given key without copying it.

        This enables zero-copy peer-to-peer data transfer.

        Args:
            key: Storage key. Trailing slashes are stripped.
            src: Local path to the data (used for validation, not copied)
            start_rsyncd: If True, start rsync daemon to enable peer-to-peer transfers
            verbose: Show detailed progress
        """
        if not is_running_in_kubernetes():
            raise RuntimeError("vput can only be called from inside a Kubernetes pod")

        key = key.rstrip("/")
        src_path = Path(src)

        if not src_path.exists():
            raise ValueError(f"Source path does not exist: {src}")

        if start_rsyncd:
            self._ensure_rsync_daemon(src_path, verbose)

        # Get pod information
        pod_ip = os.getenv("POD_IP")
        if not pod_ip:
            raise RuntimeError("POD_IP environment variable not set")

        pod_name = os.getenv("POD_NAME")
        pod_namespace = os.getenv("POD_NAMESPACE", self.namespace)

        # Convert to relative path from working directory
        working_dir = Path.cwd().absolute()
        src_path_absolute = src_path.absolute()

        try:
            src_path_relative = str(src_path_absolute.relative_to(working_dir)).replace("\\", "/")
        except ValueError:
            raise ValueError(
                f"Source path {src_path_absolute} is not under working directory {working_dir}. "
                f"vput() can only publish files within the working directory."
            )

        if verbose:
            logger.info(f"Publishing key '{key}' from pod IP '{pod_ip}' (path: {src_path_relative})")

        success = self.metadata_client.publish_key(
            key, pod_ip, pod_name=pod_name, namespace=pod_namespace, src_path=src_path_relative
        )

        if success:
            if verbose:
                logger.info(f"Successfully published key '{key}'")
        else:
            raise RuntimeError(f"Failed to publish key '{key}' with metadata server")

    def _ensure_rsync_daemon(self, src_path: Path, verbose: bool = False) -> None:
        """Ensure rsync daemon is running for peer-to-peer transfers."""
        # Check if rsync is installed
        try:
            result = subprocess.run(["which", "rsync"], capture_output=True, text=True, check=True)
            if not result.stdout.strip():
                raise RuntimeError("rsync not found in PATH")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("rsync is not installed. Install with: apt-get install -y rsync")

        serve_path = str(Path.cwd().absolute())

        # Check if daemon is already running with correct config
        try:
            result = subprocess.run(["rsync", "rsync://localhost:873/"], capture_output=True, text=True, timeout=1)
            if result.returncode == 0:
                config_file = Path("/tmp/rsyncd.conf")
                if config_file.exists() and f"path = {serve_path}" in config_file.read_text():
                    if verbose:
                        logger.debug(f"Rsync daemon already running for {serve_path}")
                    return
                # Kill existing daemon with wrong config
                self._kill_rsync_daemon()
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Start new daemon
        self._start_rsync_daemon(serve_path, verbose)

    def _kill_rsync_daemon(self) -> None:
        """Kill existing rsync daemon."""
        pid_file = Path("/tmp/rsyncd.pid")
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, 15)  # SIGTERM
                time.sleep(0.5)
            except (ValueError, ProcessLookupError, OSError):
                pass

    def _start_rsync_daemon(self, serve_path: str, verbose: bool) -> None:
        """Start rsync daemon to serve data."""
        config_file = Path("/tmp/rsyncd.conf")
        config_content = f"""pid file = /tmp/rsyncd.pid
log file = /tmp/rsyncd.log
uid = root
gid = root
use chroot = false
max connections = 10
timeout = 600
[data]
path = {serve_path}
read only = false
"""
        config_file.write_text(config_content)

        if verbose:
            logger.info(f"Starting rsync daemon to serve {serve_path}")

        process = subprocess.Popen(
            ["rsync", "--daemon", "--no-detach", f"--config={config_file}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        time.sleep(0.5)

        if process.poll() is not None:
            stderr = process.stderr.read().decode() if process.stderr else ""
            raise RuntimeError(f"Failed to start rsync daemon: {stderr}")

        if verbose:
            logger.info(f"Rsync daemon started (PID: {process.pid})")
