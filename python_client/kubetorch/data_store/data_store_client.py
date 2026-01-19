"""
High-level client for key-value store interface.

This module provides the DataStoreClient class that provides a key-value store
interface on top of the low-level RsyncClient, with support for peer-to-peer
data transfer via a metadata server.
"""

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import kubetorch.provisioning.constants as provisioning_constants
from kubetorch import globals
from kubetorch.logger import get_logger
from kubetorch.resources.compute.utils import find_available_port, RsyncError
from kubetorch.serving.utils import is_running_in_kubernetes

from .key_utils import parse_key, ParsedKey
from .metadata_client import MetadataClient
from .rsync_client import RsyncClient
from .types import BroadcastWindow, Lifespan, Locale

logger = get_logger(__name__)


class DataStoreError(Exception):
    """Exception raised for data store operations (key-value store) errors."""

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


class DataStoreClient:
    """High-level client for key-value store interface."""

    def __init__(self, namespace: Optional[str] = None, kubeconfig_path: Optional[str] = None):
        """
        Initialize the data store client.

        Args:
            namespace (str, optional): Kubernetes namespace. (Default: global config namespace)
            kubeconfig_path (str, optional): Path to kubeconfig file (not used directly, for compatibility). (Default: None)
        """
        self.namespace = namespace or globals.config.namespace
        self.metadata_client = MetadataClient(
            namespace=self.namespace, metadata_port=provisioning_constants.DATA_STORE_METADATA_PORT
        )

    def put(
        self,
        key: Union[str, List[str]],
        src: Union[str, Path, List[Union[str, Path]]],
        locale: Locale = "store",
        lifespan: Lifespan = "cluster",
        broadcast: Optional[BroadcastWindow] = None,
        contents: bool = False,
        filter_options: Optional[str] = None,
        force: bool = False,
        verbose: bool = False,
        start_rsyncd: bool = True,
        base_path: str = "/",
    ) -> None:
        """
        Upload files or directories to the cluster using a key-value store interface.

        Args:
            key (str or List[str]): Storage key(s). Can be a single key or list of keys.
            src (str or Path or List[str or Path]): Local file(s) or directory(s) to upload.
            locale (Locale, optional): Where data is stored ("store" or "local"). (Default: "store")
            lifespan (Lifespan, optional): How long data persists ("cluster" or "resource"). (Default: "cluster")
            broadcast (BroadcastWindow, optional): Optional BroadcastWindow for coordinated transfers. (Default: None)
            contents (bool, optional): If True, copy directory contents. (Default: False)
            filter_options (str, optional): Additional rsync filter options. (Default: None)
            force (bool, optional): Force overwrite of existing files. (Default: False)
            verbose (bool, optional): Show detailed progress. (Default: False)
            start_rsyncd (bool, optional): For locale="local": Start rsync daemon. (Default: True)
            base_path (str, optional): For locale="local": Root path for rsync daemon. (Default: "/")
        """
        # Normalize keys to list
        keys = [key] if isinstance(key, str) else key

        if locale == "store":
            self._put_to_store(
                keys=keys,
                src=src,
                lifespan=lifespan,
                contents=contents,
                filter_options=filter_options,
                force=force,
                verbose=verbose,
            )
        elif locale == "local":
            self._put_local(
                keys=keys,
                src=src,
                lifespan=lifespan,
                start_rsyncd=start_rsyncd,
                base_path=base_path,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Invalid locale: {locale}. Must be 'store' or 'local'.")

        # Handle broadcast if specified
        if broadcast:
            self._handle_put_broadcast(keys, src, broadcast, verbose)

    def _put_to_store(
        self,
        keys: List[str],
        src: Union[str, Path, List[Union[str, Path]]],
        lifespan: Lifespan,
        contents: bool,
        filter_options: Optional[str],
        force: bool,
        verbose: bool,
    ) -> None:
        """Upload data to the central store pod."""
        # Convert Path objects to strings
        if isinstance(src, Path):
            src = str(src)
        elif isinstance(src, list):
            src = [str(s) if isinstance(s, Path) else s for s in src]

        for key in keys:
            parsed = parse_key(key)

            # Create rsync client with appropriate service name
            rsync_client = RsyncClient(namespace=self.namespace, service_name=parsed.service_name or "store")

            if verbose:
                logger.info(f"Uploading to key '{key}' from {src}")

            # Build destination path
            dest_path = parsed.storage_path

            # If contents=True, add trailing slash for rsync "copy contents" behavior
            if contents and dest_path:
                dest_path = dest_path.rstrip("/") + "/"

            try:
                logger.debug(
                    f"DataStoreClient.put: in_cluster={is_running_in_kubernetes()}, "
                    f"service_name={parsed.service_name}, dest_path={dest_path}, contents={contents}"
                )

                rsync_client.upload(
                    source=src, dest=dest_path, contents=contents, filter_options=filter_options, force=force
                )

                # After successful upload, register with metadata server
                self._register_store_key(key, lifespan, verbose)

                if verbose:
                    logger.info(f"Successfully stored at key '{key}'")

            except RsyncError as e:
                logger.error(f"Failed to store at key '{key}': {e}")
                raise

    def _put_local(
        self,
        keys: List[str],
        src: Union[str, Path, List[Union[str, Path]]],
        lifespan: Lifespan,
        start_rsyncd: bool,
        base_path: str,
        verbose: bool,
    ) -> None:
        """Register local data with metadata server (zero-copy mode)."""
        if not is_running_in_kubernetes():
            raise RuntimeError("locale='local' can only be used inside a Kubernetes pod")

        # Convert src to Path
        if isinstance(src, list):
            src_path = Path(src[0]) if src else Path(".")
        elif isinstance(src, str):
            src_path = Path(src)
        else:
            src_path = src

        if not src_path.exists():
            raise ValueError(f"Source path does not exist: {src}")

        if start_rsyncd:
            self._ensure_rsync_daemon(src_path, base_path, verbose)

        # Get pod information
        pod_ip = os.getenv("POD_IP")
        if not pod_ip:
            raise RuntimeError("POD_IP environment variable not set")

        pod_name = os.getenv("POD_NAME")
        pod_namespace = os.getenv("POD_NAMESPACE", self.namespace)

        # Determine service_name for lifespan="resource"
        service_name = os.getenv("KT_SERVICE_NAME") if lifespan == "resource" else None

        # Convert to path relative to base_path
        src_path_absolute = src_path.absolute()
        base_path_obj = Path(base_path).absolute()

        try:
            src_path_relative = str(src_path_absolute.relative_to(base_path_obj)).replace("\\", "/")
        except ValueError:
            raise ValueError(
                f"Source path {src_path_absolute} is not under base_path {base_path_obj}. "
                f"Either move the file under {base_path} or set base_path to a parent directory."
            )

        for key in keys:
            parsed = parse_key(key)
            normalized_key = parsed.full_key

            if verbose:
                logger.info(f"Publishing key '{key}' from pod IP '{pod_ip}' (path: {src_path_relative})")

            success = self.metadata_client.publish_key(
                normalized_key,
                pod_ip,
                pod_name=pod_name,
                namespace=pod_namespace,
                src_path=src_path_relative,
                lifespan=lifespan,
                service_name=service_name,
            )

            if success:
                if verbose:
                    logger.info(f"Successfully published key '{key}'")
            else:
                raise RuntimeError(f"Failed to publish key '{key}' with metadata server")

    def _register_store_key(self, key: str, lifespan: Lifespan, verbose: bool = False) -> None:
        """Register that a key exists in the store with metadata server."""
        if not is_running_in_kubernetes():
            return

        try:
            # Use the store service URL as the identifier (stable across pod restarts)
            # The actual rsync routing uses the service URL, not pod IP
            store_service_url = f"{provisioning_constants.DATA_STORE_SERVICE_NAME}.{self.namespace}.svc.cluster.local"

            # Determine service_name for lifespan="resource"
            service_name = os.getenv("KT_SERVICE_NAME") if lifespan == "resource" else None

            self.metadata_client.register_store_pod(
                key, store_service_url, lifespan=lifespan, service_name=service_name
            )
            if verbose:
                logger.debug(f"Registered key '{key}' with metadata server")
        except Exception as e:
            logger.warning(f"Failed to register key '{key}' with metadata server: {e}")

    def _handle_put_broadcast(
        self,
        keys: List[str],
        src: Union[str, Path, List[Union[str, Path]]],
        broadcast: BroadcastWindow,
        verbose: bool,
    ) -> None:
        """Handle broadcast window for put() - this pod is a putter."""
        if not is_running_in_kubernetes():
            logger.warning("Broadcast window ignored - not running in Kubernetes")
            return

        pod_ip = os.getenv("POD_IP")
        pod_name = os.getenv("POD_NAME")

        if verbose:
            logger.info(f"Joining broadcast quorum as putter for keys: {keys}")

        # Join the broadcast quorum as a putter
        broadcast_info = self.metadata_client.join_broadcast(
            keys=keys,
            role="putter",
            pod_ip=pod_ip,
            pod_name=pod_name,
            broadcast=broadcast,
        )

        broadcast_id = broadcast_info.get("broadcast_id")
        if not broadcast_id:
            logger.warning("Failed to join broadcast quorum")
            return

        # Poll until quorum is ready
        max_wait = broadcast.timeout or 300
        start_time = time.time()

        while time.time() - start_time < max_wait:
            status_info = self.metadata_client.get_broadcast_status(broadcast_id, pod_ip)
            status = status_info.get("status")

            if status == "ready":
                if verbose:
                    logger.info(f"Broadcast quorum ready. Getters: {status_info.get('getters', [])}")

                # Send data to all getters
                getters = status_info.get("getters", [])
                for getter in getters:
                    getter_ip = getter.get("pod_ip")
                    if getter_ip:
                        try:
                            if verbose:
                                logger.info(f"Sending data to getter {getter_ip}")
                            # Note: For now, getters pull from putters, not the other way around
                            # This is simpler and works with existing rsync daemon setup
                        except Exception as e:
                            logger.warning(f"Failed to send to getter {getter_ip}: {e}")

                # Mark complete
                self.metadata_client.complete_broadcast(broadcast_id, pod_ip)
                return
            elif status == "missed":
                logger.warning("Missed the broadcast window")
                return

            time.sleep(0.5)

        logger.warning(f"Broadcast quorum timed out after {max_wait}s")

    def get(
        self,
        key: Union[str, List[str]],
        dest: Optional[Union[str, Path]] = None,
        broadcast: Optional[BroadcastWindow] = None,
        contents: bool = False,
        filter_options: Optional[str] = None,
        force: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Download files or directories from the cluster using a key-value store interface.

        Args:
            key: Storage key(s) to retrieve.
            dest: Local destination path (defaults to current working directory).
            broadcast: Optional BroadcastWindow for coordinated transfers.
            contents: If True, copy directory contents.
            filter_options: Additional rsync filter options.
            force: Force overwrite of existing files.
            verbose: Show detailed progress.
        """
        # Normalize keys to list
        keys = [key] if isinstance(key, str) else key

        # Default to current working directory if dest not specified
        if dest is None:
            dest = os.getcwd()

        if broadcast:
            self._get_with_broadcast(
                keys=keys,
                dest=dest,
                broadcast=broadcast,
                contents=contents,
                filter_options=filter_options,
                force=force,
                verbose=verbose,
            )
        else:
            # Regular get without broadcast
            for k in keys:
                self._get_single(
                    key=k,
                    dest=dest,
                    contents=contents,
                    filter_options=filter_options,
                    force=force,
                    verbose=verbose,
                )

    def _get_with_broadcast(
        self,
        keys: List[str],
        dest: Union[str, Path],
        broadcast: BroadcastWindow,
        contents: bool,
        filter_options: Optional[str],
        force: bool,
        verbose: bool,
    ) -> None:
        """Get with broadcast window - coordinate with other getters via tree topology."""
        if not is_running_in_kubernetes():
            logger.warning("Broadcast window ignored - not running in Kubernetes. Falling back to regular get.")
            for k in keys:
                self._get_single(k, dest, contents, filter_options, force, verbose)
            return

        from kubetorch.data_store.pod_data_server import DEFAULT_TCP_PORT, PodDataServerClient, start_server_if_needed

        pod_ip = os.getenv("POD_IP")
        pod_name = os.getenv("POD_NAME")

        # Start pod data server if not running - needed for tracking completed broadcasts
        # and allowing child getters to request our local path
        start_server_if_needed()

        # Use filesystem broadcast for p2p tree-based propagation
        group_id = broadcast.group_id
        if not group_id:
            # Auto-generate group_id from keys
            group_id = "_".join(sorted(keys))

        fanout = broadcast.fanout
        timeout = broadcast.timeout or 60.0

        for k in keys:
            if verbose:
                logger.info(f"Joining filesystem broadcast for key '{k}' (group: {group_id})")

            parsed = parse_key(k)
            dest_str = self._normalize_dest(dest, contents, k)

            # Join MDS broadcast to get parent info
            result = self.metadata_client.join_fs_broadcast(
                group_id=group_id,
                key=parsed.full_key,
                pod_ip=pod_ip,
                pod_name=pod_name,
                fanout=fanout,
            )

            if result.get("status") == "error":
                logger.warning(
                    f"Filesystem broadcast join failed for key '{k}': {result.get('error')}. "
                    f"Falling back to regular get."
                )
                self._get_single(k, dest, contents, filter_options, force, verbose)
                continue

            parent_rank = result.get("parent_rank")
            parent_ip = result.get("parent_ip")
            source_path = result.get("source_path")

            if verbose:
                logger.info(
                    f"Joined broadcast: rank={result.get('rank')}, parent_rank={parent_rank}, " f"parent_ip={parent_ip}"
                )

            try:
                rsync_client = RsyncClient(namespace=self.namespace, service_name=parsed.service_name or "store")

                if parent_rank == 0:
                    # Parent is the source (store) - use source_path directly
                    rsync_path = source_path
                else:
                    # Parent is another getter - ask their pod data server for the local path
                    if verbose:
                        logger.info(f"Requesting path from parent getter at {parent_ip}:{DEFAULT_TCP_PORT}")

                    pds_client = PodDataServerClient()
                    try:
                        path_result = pds_client.fs_broadcast_get_path_remote(
                            parent_ip=parent_ip,
                            parent_port=DEFAULT_TCP_PORT,
                            group_id=group_id,
                            key=parsed.full_key,
                            timeout=timeout,
                        )
                    except Exception as conn_err:
                        logger.error(
                            f"Failed to connect to parent pod data server at {parent_ip}:{DEFAULT_TCP_PORT}: {conn_err}"
                        )
                        raise

                    if path_result.get("status") != "ok":
                        raise RuntimeError(f"Failed to get path from parent: {path_result.get('error')}")

                    rsync_path = path_result["local_path"]
                    if verbose:
                        logger.info(f"Got rsync path from parent: {rsync_path}")

                # Build rsync URL and download
                peer_url = f"rsync://{parent_ip}:{provisioning_constants.REMOTE_RSYNC_PORT}/data/"
                remote_source = peer_url + rsync_path
                if contents:
                    remote_source = remote_source.rstrip("/") + "/"

                if verbose:
                    logger.info(f"Rsyncing from parent {parent_ip}: {remote_source} -> {dest_str}")

                rsync_client.download(
                    source=remote_source,
                    dest=dest_str,
                    contents=contents,
                    filter_options=filter_options,
                    force=force,
                )

                # Notify local pod data server that download is complete
                # This allows child getters to request our local path
                # Use absolute path so rsync daemon (serving from /) can find it
                pds_client = PodDataServerClient()
                local_path_for_broadcast = str(Path(dest_str.rstrip("/")).resolve())
                pds_client.fs_broadcast_complete(
                    group_id=group_id,
                    key=parsed.full_key,
                    local_path=local_path_for_broadcast,
                )
                if verbose:
                    logger.info(
                        f"Registered broadcast completion: {group_id}/{parsed.full_key} -> {local_path_for_broadcast}"
                    )

                # Start rsync daemon so we can serve as parent for later joiners
                self._ensure_rsync_daemon(Path(local_path_for_broadcast), base_path="/", verbose=verbose)

                if verbose:
                    logger.info(
                        f"Successfully retrieved key '{k}' via filesystem broadcast "
                        f"(rank={result.get('rank')}, parent_rank={parent_rank})"
                    )

            except Exception as e:
                import traceback

                logger.error(f"Filesystem broadcast failed for key '{k}': {e}")
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                logger.warning(f"Falling back to regular get for key '{k}'")
                self._get_single(k, dest, contents, filter_options, force, verbose)

    def _get_from_putter(
        self,
        key: str,
        putter_ip: str,
        dest: Union[str, Path],
        contents: bool,
        filter_options: Optional[str],
        force: bool,
        verbose: bool,
    ) -> None:
        """Get data directly from a putter pod."""
        parsed = parse_key(key)
        rsync_client = RsyncClient(namespace=self.namespace, service_name=parsed.service_name or "store")

        dest_str = self._normalize_dest(dest, contents, key)

        # Build peer rsync URL - putter is serving via rsync daemon
        # The src_path is the key's path segment
        src_path = parsed.path if parsed.path else key.split("/")[-1]
        peer_url = f"rsync://{putter_ip}:{provisioning_constants.REMOTE_RSYNC_PORT}/data/"
        remote_source = peer_url + src_path
        if contents:
            remote_source = remote_source.rstrip("/") + "/"

        if verbose:
            logger.info(f"Downloading from putter: {remote_source} to {dest_str}")

        rsync_client.download(
            source=remote_source, dest=dest_str, contents=contents, filter_options=filter_options, force=force
        )

        if verbose:
            logger.info(f"Successfully retrieved key '{key}' from putter {putter_ip}")

    def _get_single(
        self,
        key: str,
        dest: Union[str, Path],
        contents: bool,
        filter_options: Optional[str],
        force: bool,
        verbose: bool,
    ) -> None:
        """Get a single key without broadcast coordination."""
        dest_str = self._normalize_dest(dest, contents, key)
        parsed = parse_key(key)
        in_cluster = is_running_in_kubernetes()

        if verbose:
            logger.info(f"Downloading from key '{key}' to {dest_str}")

        # Get source information from metadata server
        source_info, has_store_backup = self._resolve_source(parsed.full_key, in_cluster, verbose)

        # Try to retrieve the data
        if in_cluster and source_info and source_info.ip:
            # In-cluster peer-to-peer transfer
            success = self._get_from_peer_in_cluster(
                key, parsed, source_info, dest_str, contents, filter_options, force, has_store_backup, verbose
            )
            if success:
                return

        if not in_cluster and source_info and source_info.pod_name and not has_store_backup:
            # External peer-to-peer transfer (only when store doesn't have it)
            success = self._get_from_peer_external(
                key, parsed, source_info, dest_str, contents, filter_options, force, verbose
            )
            if success:
                return

        # Fall back to store pod
        if has_store_backup:
            self._get_from_store_pod(key, parsed, dest_str, contents, filter_options, force, verbose)
        else:
            raise DataStoreError(
                f"Key '{parsed.full_key}' not found - no peer sources and no store pod backup available"
            )

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
        peer_url = f"rsync://{source_info.ip}:{provisioning_constants.REMOTE_RSYNC_PORT}/data/"
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
                raise DataStoreError(
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
        # This function is only called when store doesn't have the data (not has_store_backup),
        # so we need to port-forward directly to the peer pod that has the data.
        if source_info.proxy_through_store or not source_info.pod_name:
            if verbose:
                logger.info("Cannot use peer transfer - no pod name available")
            return False

        rsync_client = RsyncClient(namespace=self.namespace, service_name=parsed.service_name or "store")
        pod_name = source_info.pod_name
        pod_namespace = source_info.namespace or self.namespace

        try:
            if verbose:
                logger.info(f"Using port-forward to connect to peer pod {pod_name}")

            # Set up port-forward
            local_port = find_available_port(provisioning_constants.REMOTE_RSYNC_PORT + 1000)
            pf_cmd = [
                "kubectl",
                "port-forward",
                f"pod/{pod_name}",
                f"{local_port}:{provisioning_constants.REMOTE_RSYNC_PORT}",
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
                raise DataStoreError(f"Rsync completed but no files were transferred from peer for key '{key}'")
        elif not dest_path.exists():
            raise DataStoreError(f"Destination does not exist after download: {dest_path}")

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

        except RsyncError as e:
            logger.error(f"Failed to retrieve key '{key}' from store pod: {e}")
            raise

    def ls(self, key: str = "", verbose: bool = False) -> List[dict]:
        """
        List files and directories under a key path in the store.

        Args:
            key: Storage key path to list (empty string for root).
            verbose: Show detailed progress

        Returns:
            List of dicts with item information.
        """
        parsed = parse_key(key)

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

    def mkdir(self, key: str, verbose: bool = False) -> bool:
        """
        Create a directory at the given key path in the store.

        Args:
            key: Storage key path to create.
            verbose: Show detailed progress

        Returns:
            True if successful, False otherwise.
        """
        parsed = parse_key(key)

        if verbose:
            logger.info(f"Creating directory for key '{key}'")

        try:
            result = self.metadata_client.mkdir(parsed.full_key)
            success = result.get("success", False)

            if verbose and success:
                logger.info(f"Created directory for key '{key}'")

            return success
        except Exception as e:
            logger.error(f"Failed to create directory for key '{key}': {e}")
            return False

    def rm(
        self,
        key: str,
        recursive: bool = False,
        prefix_mode: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Delete a file or directory from the store.

        Args:
            key: Storage key to delete.
            recursive: If True, delete directories recursively (directory semantics)
            prefix_mode: If True, delete all keys starting with this string prefix
            verbose: Show detailed progress
        """
        parsed = parse_key(key)

        if verbose:
            if prefix_mode:
                logger.info(f"Deleting all keys with prefix '{key}'")
            else:
                logger.info(f"Deleting key '{key}'")

        result = self.metadata_client.delete_key(parsed.full_key, recursive=recursive, prefix_mode=prefix_mode)

        if not result.get("success", False):
            error = result.get("error", "Unknown error")
            raise RuntimeError(f"Failed to delete key '{key}': {error}")

        if verbose:
            deleted_meta = result.get("deleted_from_metadata", False)
            deleted_fs = result.get("deleted_from_filesystem", False)
            deleted_count = result.get("deleted_metadata_count", 0)
            deleted_fs_count = result.get("deleted_fs_count", 0)
            if prefix_mode:
                parts = []
                if deleted_count > 0:
                    parts.append(f"{deleted_count} metadata keys")
                if deleted_fs_count > 0:
                    parts.append(f"{deleted_fs_count} filesystem entries")
                if parts:
                    logger.info(f"Deleted {', '.join(parts)} with prefix '{key}'")
                else:
                    logger.info(f"No keys found with prefix '{key}'")
            elif deleted_meta and deleted_fs:
                logger.info(f"Deleted key '{key}' from metadata and filesystem")
            elif deleted_meta:
                logger.info(f"Deleted virtual key '{key}' from metadata")
            elif deleted_fs:
                logger.info(f"Deleted key '{key}' from filesystem")
            else:
                logger.info(f"Key '{key}' does not exist")

    def _ensure_rsync_daemon(self, src_path: Path, base_path: str = "/", verbose: bool = False) -> None:
        """Ensure rsync daemon is running for peer-to-peer transfers."""
        # Check if rsync is installed
        try:
            result = subprocess.run(["which", "rsync"], capture_output=True, text=True, check=True)
            if not result.stdout.strip():
                raise RuntimeError("rsync not found in PATH")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("rsync is not installed. Install with: apt-get install -y rsync")

        # Serve from base_path (defaults to "/" to allow vput of files anywhere)
        serve_path = str(Path(base_path).absolute())

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
