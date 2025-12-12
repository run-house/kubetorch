"""
High-level client for key-value store interface.

This module provides the DataSyncClient class that provides a key-value store
interface on top of the low-level RsyncClient.
"""

import os
import subprocess
from pathlib import Path
from typing import List, Optional, Union

import kubetorch.serving.constants as serving_constants

from kubetorch import globals
from kubetorch.logger import get_logger
from kubetorch.resources.compute.utils import RsyncError
from kubetorch.servers.http.utils import is_running_in_kubernetes

from .rsync_client import RsyncClient

logger = get_logger(__name__)


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
                f"DataSyncClient.put: in_cluster={is_running_in_kubernetes()}, service_name={service_name}, dest_path={dest_path}, contents={contents}"
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
        elif isinstance(dest, list):
            # If dest is a list, take the first element (shouldn't happen, but handle gracefully)
            if dest:
                dest = str(dest[0])
            else:
                dest = os.getcwd()
        elif not isinstance(dest, str):
            # Convert any other type to string
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
        # Strip trailing slashes from key - keys are abstract, not directory paths
        key = key.rstrip("/")

        # Check if we're inside a service in the cluster and should auto-prepend service name
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
        # Strip trailing slashes from key - keys are abstract, not directory paths
        key = key.rstrip("/")

        # Check if we're inside a service in the cluster and should auto-prepend service name
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
