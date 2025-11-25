"""
Module-level convenience functions for data store operations.

This module provides the top-level API functions (put, get, ls, rm) that users
call directly, as well as the sync_workdir_from_store function used internally.
"""

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Union

from kubetorch.logger import get_logger
from kubetorch.resources.compute.utils import RsyncError

from .data_store_client import DataStoreClient, DataStoreError

logger = get_logger(__name__)

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
        _default_client = DataStoreClient(namespace=namespace, kubeconfig_path=kubeconfig_path)

    _default_client.put(
        key=key, src=src, contents=contents, filter_options=filter_options, force=force, verbose=verbose
    )


def get(
    key: str,
    dest: Optional[Union[str, Path]] = None,
    contents: bool = False,
    filter_options: Optional[str] = None,
    force: bool = False,
    seed_data: bool = True,
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
        seed_data: If True, automatically call vput() after successful retrieval to seed the data (default: True)
        verbose: Show detailed progress
        namespace: Kubernetes namespace
        kubeconfig_path: Path to kubeconfig file (for compatibility)

    Examples:
        >>> import kubetorch as kt
        >>> kt.get(key="model-v1/weights")  # Downloads to current directory and auto-seeds
        >>> kt.get(key="model-v1/weights", dest="./local_model")  # Creates local_model/weights/
        >>> kt.get(key="model-v1/weights", dest="./local_model", contents=True)  # Copies contents into local_model/
        >>> kt.get(key="datasets/train.csv", dest="./data/train.csv", seed_data=False)  # Download without seeding
        >>> kt.get(key="my-service/outputs", dest="./results", contents=True)  # Copies contents into results/
    """
    global _default_client

    if _default_client is None or namespace or kubeconfig_path:
        _default_client = DataStoreClient(namespace=namespace, kubeconfig_path=kubeconfig_path)

    _default_client.get(
        key=key,
        dest=dest,
        contents=contents,
        filter_options=filter_options,
        force=force,
        seed_data=seed_data,
        verbose=verbose,
    )


def ls(
    key: str = "", verbose: bool = False, namespace: Optional[str] = None, kubeconfig_path: Optional[str] = None
) -> List[dict]:
    """
    List files and directories under a key path in the store.
    Combines virtual keys (vput-published) and filesystem contents.

    Examples:
        >>> import kubetorch as kt
        >>> kt.ls()  # List root of store
        >>> kt.ls("my-service")  # List contents of my-service
        >>> kt.ls("my-service/models")  # List models directory

    Returns:
        List of dicts with item information:
        - name: Item name (directories have trailing /)
        - is_virtual: True if published via vput (not in filesystem)
        - is_directory: True if directory
        - pod_name: Pod name where virtual key is stored (if virtual)
        - pod_namespace: Namespace of pod (if virtual)
    """
    global _default_client

    if _default_client is None or namespace or kubeconfig_path:
        _default_client = DataStoreClient(namespace=namespace, kubeconfig_path=kubeconfig_path)

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
        _default_client = DataStoreClient(namespace=namespace, kubeconfig_path=kubeconfig_path)

    _default_client.rm(key=key, recursive=recursive, verbose=verbose)


def vput(
    key: str,
    src: Union[str, Path],
    start_rsyncd: bool = True,
    verbose: bool = False,
    namespace: Optional[str] = None,
    kubeconfig_path: Optional[str] = None,
) -> None:
    """
    Virtual put - publish that this pod has data for the given key without copying it.
    This enables zero-copy peer-to-peer data transfer. Other pods can then rsync
    directly from this pod when requesting the key.

    Args:
        key: Storage key (e.g., "my-service/models", "shared/dataset"). Trailing slashes are stripped.
        src: Local path to the data (used for validation, not copied)
        start_rsyncd: If True, start rsync daemon to enable peer-to-peer transfers (default: True)
        verbose: Show detailed progress
        namespace: Kubernetes namespace
        kubeconfig_path: Path to kubeconfig file (for compatibility)

    Note:
        This does NOT copy data to the store pod. It only registers with the metadata
        server that this pod has the data. Use regular `put()` if you want to store
        data in the persistent store pod.

        If start_rsyncd=True, the rsync daemon will be started to serve the data.
        Rsync must be installed in the pod (raises RuntimeError if not found).

    Examples:
        >>> import kubetorch as kt
        >>> # After downloading data, publish that this pod has it
        >>> kt.get(key="model-v1/weights", dest="./weights")
        >>> kt.vput(key="model-v1/weights", src="./weights")  # Other pods can now rsync from this pod
        >>> kt.vput(key="model-v1/weights", src="./weights", start_rsyncd=False)  # Don't start daemon
    """
    global _default_client

    if _default_client is None or namespace or kubeconfig_path:
        _default_client = DataStoreClient(namespace=namespace, kubeconfig_path=kubeconfig_path)

    _default_client.vput(key=key, src=src, start_rsyncd=start_rsyncd, verbose=verbose)


def sync_workdir_from_store(namespace: str, service_name: str):
    """
    Sync files from the rsync pod into the current working directory inside the server pod.

    This function is called by http_server.py during pod startup to sync files that were
    uploaded via the KV interface (kt.put or DataStoreClient.put) into the server pod's working directory.

    Performs two download operations (potentially in parallel):
    - Regular files (excluding __absolute__*) into the working directory
    - Absolute path files (under __absolute__/...) into their absolute destinations

    Uses the DataStoreClient KV interface, which allows future scalability with peer-to-peer
    transfer via a central metadata store. When called from inside a pod, empty key "" auto-prepends
    the service name to download from the service's storage area.
    """
    import os

    # Use DataStoreClient KV interface for future scalability
    dt_client = DataStoreClient(namespace=namespace)

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
        except (RsyncError, DataStoreError) as e:
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
    # This ensures DataStoreClient.get() with empty key will use the correct service name
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
    from .rsync_client import RsyncClient

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
    from .rsync_client import RsyncClient

    client = RsyncClient(namespace=namespace, service_name=service_name)
    await client.upload_async(
        source=source,
        dest=dest,
        contents=contents,
        filter_options=filter_options,
        force=force,
        local_port=local_port,
    )
