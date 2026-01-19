"""
Module-level convenience functions for data store operations.

This module provides the top-level API functions (put, get, ls, rm) that users
call directly, as well as the sync_workdir_from_store function used internally.
"""
import concurrent.futures
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Union

from kubetorch.logger import get_logger
from kubetorch.resources.compute.utils import RsyncError

from .data_store_client import DataStoreClient, DataStoreError
from .types import BroadcastWindow, Lifespan, Locale

logger = get_logger(__name__)

# Create singleton instances for convenience
_default_client = None


def put(
    key: Union[str, List[str]],
    src: Optional[Union[str, Path, List[Union[str, Path]], "torch.Tensor", dict]] = None,
    locale: Locale = "store",
    lifespan: Lifespan = "cluster",
    broadcast: Optional[BroadcastWindow] = None,
    contents: bool = False,
    filter_options: Optional[str] = None,
    force: bool = False,
    verbose: bool = False,
    namespace: Optional[str] = None,
    kubeconfig_path: Optional[str] = None,
    # Parameters for locale="local" (zero-copy mode)
    start_rsyncd: bool = True,
    base_path: str = "/",
    nccl_port: int = 29500,
) -> None:
    """
    Upload data to the cluster using a key-value store interface.

    Supports two data types (auto-detected from `src`):
    - **Filesystem data**: Files/directories uploaded via rsync
    - **GPU data**: GPU tensors or state dicts broadcast via NCCL

    Args:
        key: Storage key(s). Keys should be explicit paths like "my-service/models/v1".
            Can be a single key or list of keys for batch filesystem operations.
        src: Data to upload. Can be:
            - Path(s) to local file(s) or directory(s) for filesystem transfer
            - GPU tensor for single tensor broadcast via NCCL
            - Dict of GPU tensors (state dict) for multi-tensor broadcast via NCCL
        locale: Where data is stored:
            - "store" (default): Copy to central store pod. Data is persisted and
              accessible from any pod. (Filesystem only - GPU data always uses "local")
            - "local": Zero-copy mode. Data stays on the local pod and is only
              registered with the metadata server. Other pods fetch directly from
              this pod.
        lifespan: How long data persists:
            - "cluster" (default): Data persists until explicitly deleted.
            - "resource": Data is automatically cleaned up when the service is torn down.
        broadcast: Optional BroadcastWindow for coordinated multi-party transfers.
            When specified, this put() joins as a "putter" and waits for other
            participants before transferring data.
        contents: If True, copy directory contents (adds trailing slashes for rsync). (Filesystem only)
        filter_options: Additional rsync filter options. (Filesystem only)
        force: Force overwrite of existing files. (Filesystem only)
        verbose: Show detailed progress.
        namespace: Kubernetes namespace.
        kubeconfig_path: Path to kubeconfig file (for compatibility).
        start_rsyncd: For locale="local": Start rsync daemon to serve data (default: True). (Filesystem only)
        base_path: For locale="local": Root path for rsync daemon (default: "/"). (Filesystem only)
        nccl_port: Port for NCCL communication (default: 29500). (GPU only)

    Examples:
        # Upload filesystem data to central store
        >>> import kubetorch as kt
        >>> kt.put(key="my-service/weights", src="./trained_model/")

        # Zero-copy mode (data stays local, other pods fetch directly)
        >>> kt.put(key="my-service/data", src="/app/data", locale="local")

        # Resource-scoped (auto-cleaned on service teardown)
        >>> kt.put(key="my-service/temp", src="./temp/", lifespan="resource")

        # Coordinated filesystem broadcast with timeout
        >>> kt.put(
        ...     key="my-service/weights",
        ...     src="./weights/",
        ...     locale="local",
        ...     broadcast=kt.BroadcastWindow(timeout=10.0)
        ... )

        # GPU tensor - other pods receive via NCCL broadcast
        >>> import torch
        >>> tensor = torch.randn(1000, 1000, device="cuda")
        >>> kt.put(key="model/layer1", src=tensor)

        # GPU state dict - all tensors broadcast over single NCCL process group
        >>> state_dict = model.state_dict()  # Contains CUDA tensors
        >>> kt.put(key="model/weights", src=state_dict, broadcast=kt.BroadcastWindow(world_size=4))
    """
    if src is None:
        raise ValueError("src is required. Provide a path for filesystem data or a GPU tensor/dict for GPU data.")

    from .gpu_transfer import _is_gpu_data

    # Check if this is GPU data
    if _is_gpu_data(src):
        # GPU data transfer via NCCL
        from .gpu_transfer import _get_gpu_manager

        # Handle single key only for GPU data
        if isinstance(key, list):
            raise ValueError("GPU data transfer only supports a single key, not a list of keys.")

        manager = _get_gpu_manager()
        return manager.publish(key=key, data=src, nccl_port=nccl_port, broadcast=broadcast, verbose=verbose)

    # Filesystem data transfer
    global _default_client

    if _default_client is None or namespace or kubeconfig_path:
        _default_client = DataStoreClient(namespace=namespace, kubeconfig_path=kubeconfig_path)

    _default_client.put(
        key=key,
        src=src,
        locale=locale,
        lifespan=lifespan,
        broadcast=broadcast,
        contents=contents,
        filter_options=filter_options,
        force=force,
        verbose=verbose,
        start_rsyncd=start_rsyncd,
        base_path=base_path,
    )


def get(
    key: Union[str, List[str]],
    dest: Optional[Union[str, Path, "torch.Tensor", dict]] = None,
    broadcast: Optional[BroadcastWindow] = None,
    contents: bool = False,
    filter_options: Optional[str] = None,
    force: bool = False,
    verbose: bool = False,
    namespace: Optional[str] = None,
    kubeconfig_path: Optional[str] = None,
) -> None:
    """
    Download data from the cluster using a key-value store interface.

    Supports two data types:
    - **Filesystem data**: Files/directories downloaded via rsync
    - **GPU data**: GPU tensors or state dicts received via NCCL broadcast

    The data type is auto-detected from the `dest` parameter:
    - If dest is a path (str/Path) or None: filesystem data
    - If dest is a GPU tensor or dict of GPU tensors: GPU data

    Args:
        key: Storage key(s) to retrieve. Keys should be explicit paths like
            "my-service/models/v1". Can be a single key or list of keys.
        dest: Destination for the data:
            - For filesystem: Local path (defaults to current working directory)
            - For GPU: Pre-allocated tensor or state_dict (dict of tensors) to receive into
        broadcast: Optional BroadcastWindow for coordinated multi-party transfers.
            When specified, this get() joins as a "getter" and waits for putters
            before receiving data. Use broadcast.timeout to control wait time.
        contents: If True, copy directory contents (adds trailing slashes). (Filesystem only)
        filter_options: Additional rsync filter options. (Filesystem only)
        force: Force overwrite of existing files. (Filesystem only)
        verbose: Show detailed progress.
        namespace: Kubernetes namespace.
        kubeconfig_path: Path to kubeconfig file (for compatibility).

    Examples:
        # Download from store
        >>> import kubetorch as kt
        >>> import torch
        >>>
        >>> # Filesystem data
        >>> kt.get(key="my-service/weights")  # Downloads to current directory
        >>> kt.get(key="my-service/weights", dest="./local_model/")  # Downloads to local_model/
        >>>
        >>> # Coordinated broadcast (wait for putter)
        >>> kt.get(
        ...     key="my-service/weights",
        ...     dest="./weights/",
        ...     broadcast=kt.BroadcastWindow(timeout=10.0)
        ... )
        >>>
        >>> # GPU tensor - provide pre-allocated destination
        >>> tensor = torch.empty(1000, 1000, device="cuda:0")
        >>> kt.get(key="model/layer1", dest=tensor)
        >>>
        >>> # GPU state dict with coordinated broadcast
        >>> model = MyModel().cuda()
        >>> kt.get(
        ...     key="model/weights",
        ...     dest=model.state_dict(),
        ...     broadcast=kt.BroadcastWindow(world_size=4)
        ... )
        >>> model.load_state_dict(model.state_dict())  # Already updated in-place
    """
    from .gpu_transfer import _is_gpu_data

    # Check if dest is GPU data (tensor or dict of tensors)
    if dest is not None and _is_gpu_data(dest):
        from .gpu_transfer import _get_gpu_manager

        manager = _get_gpu_manager()
        return manager.retrieve(key=key, dest=dest, broadcast=broadcast, verbose=verbose)

    # Filesystem data retrieval
    global _default_client

    if _default_client is None or namespace or kubeconfig_path:
        _default_client = DataStoreClient(namespace=namespace, kubeconfig_path=kubeconfig_path)

    _default_client.get(
        key=key,
        dest=dest,
        broadcast=broadcast,
        contents=contents,
        filter_options=filter_options,
        force=force,
        verbose=verbose,
    )


def ls(
    key: str = "", verbose: bool = False, namespace: Optional[str] = None, kubeconfig_path: Optional[str] = None
) -> List[dict]:
    """
    List files and directories under a key path in the store.
    Combines locally-published keys and filesystem contents from the central store.

    Examples:
        >>> import kubetorch as kt
        >>> kt.ls()  # List root of store
        >>> kt.ls("my-service")  # List contents of my-service
        >>> kt.ls("my-service/models")  # List models directory

    Returns:
        List of dicts with item information:
        - name: Item name (directories have trailing /)
        - is_directory: True if directory
        - locale: Where the data lives - "store" for central store, or pod name for local data
    """
    global _default_client

    if _default_client is None or namespace or kubeconfig_path:
        _default_client = DataStoreClient(namespace=namespace, kubeconfig_path=kubeconfig_path)

    return _default_client.ls(key=key, verbose=verbose)


def rm(
    key: str,
    recursive: bool = False,
    prefix: bool = False,
    verbose: bool = False,
    namespace: Optional[str] = None,
    kubeconfig_path: Optional[str] = None,
) -> None:
    """
    Delete a file or directory from the store.

    Args:
        key: Storage key to delete. Trailing slashes are stripped.
        recursive: If True, delete directories recursively (like rm -r)
        prefix: If True, delete all keys starting with this string prefix
        verbose: Show detailed progress
        namespace: Kubernetes namespace
        kubeconfig_path: Path to kubeconfig file (for compatibility)

    Examples:
        >>> import kubetorch as kt
        >>> kt.rm("my-service/old-model.pkl")  # Delete a file
        >>> kt.rm("my-service/temp-data", recursive=True)  # Delete a directory
        >>> kt.rm("gpu-test", prefix=True)  # Delete all keys starting with "gpu-test"
    """
    global _default_client

    if _default_client is None or namespace or kubeconfig_path:
        _default_client = DataStoreClient(namespace=namespace, kubeconfig_path=kubeconfig_path)

    _default_client.rm(key=key, recursive=recursive, prefix_mode=prefix, verbose=verbose)


def _sync_workdir_from_store(namespace: str, service_name: str):
    """
    Sync files from the rsync pod into the current working directory inside the server pod.

    This function is called by http_server.py during pod startup to sync files that were
    uploaded via the KV interface (kt.put or DataStoreClient.put) into the server pod's working directory.

    Performs two download operations (potentially in parallel):
    - Regular files (excluding __absolute__*) into the working directory
    - Absolute path files (under __absolute__/...) into their absolute destinations

    Uses the DataStoreClient KV interface, which allows future scalability with peer-to-peer
    transfer via a central metadata store.
    """
    logger.info(
        f"Syncing workdir from data store: namespace={namespace}, service_name={service_name}, cwd={os.getcwd()}"
    )

    # Use DataStoreClient KV interface for future scalability
    dt_client = DataStoreClient(namespace=namespace)

    # Use absolute key path (starting with /) to specify exactly which service's data to download
    # This avoids any auto-prepending of the current pod's service name
    service_key = f"/{service_name}"

    def sync_regular_files():
        """Sync regular files (excluding __absolute__*) to current directory."""
        try:
            logger.info(f"Downloading files from {service_key} to current directory")
            dt_client.get(
                key=service_key,
                dest=".",
                contents=True,
                filter_options="--exclude='__absolute__*'",
            )
            logger.info(f"Successfully synced files from {service_key}")
        except (RsyncError, DataStoreError) as e:
            # If the service storage area doesn't exist yet, that's okay
            error_msg = str(e).lower()
            if "no such file or directory" in error_msg or "not found" in error_msg:
                logger.warning(f"Service storage area {service_key} does not exist, skipping regular files sync")
            else:
                raise
        except Exception as e:
            # Catch any other exceptions and check error message
            error_msg = str(e).lower()
            if "no such file or directory" in error_msg or "not found" in error_msg:
                logger.warning(f"Service storage area {service_key} does not exist, skipping regular files sync")
            else:
                raise

    def sync_absolute_files():
        """Sync absolute path files (under __absolute__/) to root filesystem."""
        # Check if __absolute__ directory exists by trying to list it
        try:
            items = dt_client.ls(key=f"{service_key}/__absolute__")
            if items:
                # Download __absolute__ contents to / with contents=True
                dt_client.get(key=f"{service_key}/__absolute__", dest="/", contents=True)
            else:
                logger.debug("No absolute path files to sync")
        except Exception as e:
            # If __absolute__ doesn't exist, that's okay
            error_msg = str(e).lower()
            if "no such file or directory" in error_msg or "not found" in error_msg:
                logger.debug("No absolute path files to sync")
            else:
                raise

    with ThreadPoolExecutor(max_workers=2) as executor:
        regular_future = executor.submit(sync_regular_files)
        absolute_future = executor.submit(sync_absolute_files)
        futures = [regular_future, absolute_future]
        for future in concurrent.futures.as_completed(futures):
            future.result()
