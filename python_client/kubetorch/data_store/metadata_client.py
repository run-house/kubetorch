"""
Client for communicating with the metadata server.

The metadata server tracks which pods have published data for each key,
enabling peer-to-peer data transfer and load balancing.
"""

from typing import Optional, Union
from urllib.parse import quote

import requests

from kubetorch.logger import get_logger
from kubetorch.servers.http.utils import is_running_in_kubernetes

logger = get_logger(__name__)


class MetadataClient:
    """Client for the metadata server API."""

    def __init__(self, namespace: str, metadata_port: int = 8081):
        """
        Initialize the metadata client.

        Args:
            namespace: Kubernetes namespace (same namespace as the data-store service)
            metadata_port: Port where metadata server is running
        """
        self.namespace = namespace  # Namespace where the data-store service is deployed
        self.metadata_port = metadata_port
        self._base_url = None

    @property
    def base_url(self) -> str:
        """Get the base URL for the metadata server."""
        if self._base_url is None:
            service_name = "kubetorch-data-store"
            if is_running_in_kubernetes():
                # Service is in the same namespace
                self._base_url = f"http://{service_name}.{self.namespace}.svc.cluster.local:{self.metadata_port}"
            else:
                # Outside cluster - use port-forward via globals.service_url mechanism
                from kubetorch import globals as kt_globals

                # Use the same port-forward mechanism as other services
                # Note: This will create a port-forward to the service
                try:
                    self._base_url = kt_globals.service_url(
                        service_name=service_name,
                        namespace=self.namespace,
                        remote_port=self.metadata_port,
                        health_endpoint="/health",
                    )
                except Exception as e:
                    # If port-forward fails, fall back to localhost (user may have set it up manually)
                    logger.warning(f"Failed to set up port-forward for metadata server, using localhost: {e}")
                    self._base_url = f"http://localhost:{self.metadata_port}"
        return self._base_url

    def get_source_ip(
        self, key: str, retry_with_peers: bool = True, external: bool = False
    ) -> Optional[Union[str, dict]]:
        """
        Get an IP address or pod info to rsync data from for the given key.

        Args:
            key: Storage key
            retry_with_peers: If True and server returns 503, retry with random peer selection
            external: If True, return pod name + namespace instead of IP (for external clients)

        Returns:
            If external=False: IP address (pod IP) to rsync from, or None if key doesn't exist
            If external=True: Dict with "pod_name", "namespace", and optionally "proxy_through_store" and "peer_ip", or None
        """
        try:
            # URL-encode the key to handle special characters
            encoded_key = quote(key, safe="")
            url = f"{self.base_url}/api/v1/keys/{encoded_key}/source"
            if external:
                url += "?external=true"

            response = requests.get(url, timeout=5)

            # Handle 503 (all sources at max concurrent) - retry with random peer
            if response.status_code == 503 and retry_with_peers:
                data = response.json()
                available_ips = data.get("ips", [])
                if available_ips:
                    import random

                    selected_ip = random.choice(available_ips)
                    logger.info(f"All sources at max concurrent for key '{key}', randomly selected peer: {selected_ip}")
                    if external:
                        # For external clients, we can't use random IP selection because we need pod_name for port-forward
                        # Fall back to store pod instead
                        logger.debug("External client: falling back to store pod instead of random peer selection")
                        return None
                    return selected_ip
                return None

            response.raise_for_status()
            data = response.json()

            # Check if key was found (new response format - 200 with {"found": False})
            if data.get("found") is False:
                # Key doesn't exist in metadata or filesystem
                return None

            if external:
                # Return dict with pod info
                return data
            else:
                # Return IP address and src_path if available
                ip = data.get("ip")
                src_path = data.get("src_path")
                if src_path:
                    return {"ip": ip, "src_path": src_path}
                return ip
        except requests.RequestException as e:
            logger.warning(f"Failed to get source IP for key '{key}': {e}")
            return None

    def publish_key(
        self,
        key: str,
        pod_ip: str,
        pod_name: Optional[str] = None,
        namespace: Optional[str] = None,
        src_path: Optional[str] = None,
    ) -> bool:
        """
        Publish that this pod has data for the given key.

        Args:
            key: Storage key
            pod_ip: IP address of the pod publishing the data
            pod_name: Optional pod name (for external client support)
            namespace: Optional namespace (for external client support)
            src_path: Optional relative path from working directory to the data on the pod (for peer-to-peer rsync)

        Returns:
            True if successful, False otherwise
        """
        try:
            # URL-encode the key to handle special characters
            encoded_key = quote(key, safe="")
            payload = {"ip": pod_ip}
            if pod_name:
                payload["pod_name"] = pod_name
            if namespace:
                payload["namespace"] = namespace
            if src_path:
                payload["src_path"] = src_path

            response = requests.post(
                f"{self.base_url}/api/v1/keys/{encoded_key}/publish",
                json=payload,
                timeout=5,
            )
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.warning(f"Failed to publish key '{key}' from IP '{pod_ip}': {e}")
            return False

    def complete_request(self, key: str, ip: str) -> bool:
        """
        Notify metadata server that a request to a source IP has completed.
        Decrements the concurrent request count.

        Args:
            key: Storage key
            ip: IP address that completed the request

        Returns:
            True if successful, False otherwise
        """
        try:
            # URL-encode the key to handle special characters
            encoded_key = quote(key, safe="")
            response = requests.post(
                f"{self.base_url}/api/v1/keys/{encoded_key}/source/complete",
                json={"ip": ip},
                timeout=5,
            )
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.debug(f"Failed to notify completion for key '{key}' IP '{ip}': {e}")
            return False

    def has_store_pod(self, key: str) -> bool:
        """
        Check if the store pod has data for the given key.

        The metadata server checks both its in-memory dictionary and the filesystem,
        so False means the key definitely doesn't exist.

        Args:
            key: Storage key

        Returns:
            True if store pod has the data, False if it doesn't exist (checked in metadata and filesystem)

        Raises:
            requests.exceptions.ConnectionError: If metadata server is unreachable
            requests.exceptions.ConnectTimeout: If metadata server connection times out
        """
        # URL-encode the key to handle special characters
        encoded_key = quote(key, safe="")
        response = requests.get(
            f"{self.base_url}/api/v1/keys/{encoded_key}",
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()

        # Check if key was found (new response format)
        if data.get("found") is False:
            # Key doesn't exist in metadata or filesystem
            return False

        # Key exists - check if store pod has it
        return data.get("store_pod_ip") is not None

    def remove_source(self, key: str, pod_ip: str) -> bool:
        """
        Remove a source IP from the list of available sources for a key.
        Called when a source IP is found to be unreachable.

        Args:
            key: Storage key
            pod_ip: IP address that is unreachable

        Returns:
            True if successful, False otherwise
        """
        try:
            # URL-encode the key to handle special characters
            encoded_key = quote(key, safe="")
            response = requests.delete(
                f"{self.base_url}/api/v1/keys/{encoded_key}/sources/{pod_ip}",
                timeout=5,
            )
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.warning(f"Failed to remove source IP '{pod_ip}' for key '{key}': {e}")
            return False

    def delete_key(self, key: str, recursive: bool = False) -> dict:
        """
        Delete a key from both the metadata server and filesystem.
        This removes virtual keys (vput-published) from the metadata server AND deletes files from the filesystem.

        Args:
            key: Storage key to delete
            recursive: If True, delete directories recursively

        Returns:
            dict with success status and details about what was deleted
        """
        try:
            # URL-encode the key to handle special characters
            encoded_key = quote(key, safe="")
            url = f"{self.base_url}/api/v1/keys/{encoded_key}"
            if recursive:
                url += "?recursive=true"
            response = requests.delete(url, timeout=30)  # Longer timeout for filesystem operations
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.warning(f"Failed to delete key '{key}': {e}")
            return {"success": False, "error": str(e)}

    def list_keys(self, prefix: str = "") -> dict:
        """
        List all keys matching a prefix, combining virtual keys (vput) and filesystem contents.

        Args:
            prefix: Key prefix to match (default: empty string for root)

        Returns:
            dict with "prefix" and "items" list, where each item has:
            - name: Item name (relative to prefix)
            - is_virtual: True if published via vput (not in filesystem)
            - is_directory: True if directory
            - pod_name: Pod name where virtual key is stored (if virtual)
            - pod_namespace: Namespace of pod (if virtual)
        """
        try:
            encoded_prefix = quote(prefix, safe="")
            response = requests.get(
                f"{self.base_url}/api/v1/keys/list?prefix={encoded_prefix}",
                timeout=5,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.warning(f"Failed to list keys with prefix '{prefix}': {e}")
            return {"prefix": prefix, "items": []}

    def register_store_pod(self, key: str, store_pod_ip: str) -> bool:
        """
        Register that the store pod itself has data for a key.
        Called by the store pod after storing data.

        Args:
            key: Storage key
            store_pod_ip: IP address of the store pod

        Returns:
            True if successful, False otherwise
        """
        try:
            # URL-encode the key to handle special characters
            encoded_key = quote(key, safe="")
            response = requests.post(
                f"{self.base_url}/api/v1/keys/{encoded_key}/store",
                json={"ip": store_pod_ip},
                timeout=5,
            )
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.warning(f"Failed to register store pod IP '{store_pod_ip}' for key '{key}': {e}")
            return False
