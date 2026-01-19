"""
Client for communicating with the metadata server.

The metadata server tracks which pods have published data for each key,
enabling peer-to-peer data transfer and load balancing.
"""

from typing import List, Optional, Union
from urllib.parse import quote

import httpx

from kubetorch.logger import get_logger
from kubetorch.serving.global_http_clients import get_sync_client
from kubetorch.serving.utils import is_running_in_kubernetes

from .types import BroadcastWindow, Lifespan

logger = get_logger(__name__)


class MetadataClient:
    """Client for the metadata server API."""

    def __init__(self, namespace: str, metadata_port: int = 8081):
        """
        Initialize the metadata client.

        Args:
            namespace (str): Kubernetes namespace (same namespace as the data-store service).
            metadata_port (int, optional): Port where metadata server is running. (Default: 8081)
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
            key (str): Storage key.
            retry_with_peers (bool, optional): If True and server returns 503, retry with random peer selection. (Default: True)
            external (bool, optional): If True, return pod name + namespace instead of IP (for external clients). (Default: False)

        Returns:
            If external=False: IP address (pod IP) to rsync from, or None if key doesn't exist.
            If external=True: Dict with "pod_name", "namespace", and optionally "proxy_through_store" and "peer_ip", or None.
        """
        try:
            # URL-encode the key to handle special characters
            encoded_key = quote(key, safe="")
            url = f"{self.base_url}/api/v1/keys/{encoded_key}/source"
            if external:
                url += "?external=true"

            response = get_sync_client().get(url, timeout=5)

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
        except httpx.HTTPStatusError as e:
            logger.warning(f"Failed to get source IP for key '{key}': HTTP {e.response.status_code}")
            return None
        except httpx.RequestError as e:
            logger.warning(f"Failed to get source IP for key '{key}': {e}")
            return None

    def publish_key(
        self,
        key: str,
        pod_ip: str,
        pod_name: Optional[str] = None,
        namespace: Optional[str] = None,
        src_path: Optional[str] = None,
        lifespan: Lifespan = "cluster",
        service_name: Optional[str] = None,
    ) -> bool:
        """
        Publish that this pod has data for the given key.

        Args:
            key: Storage key
            pod_ip: IP address of the pod publishing the data
            pod_name: Optional pod name (for external client support)
            namespace: Optional namespace (for external client support)
            src_path: Optional relative path from working directory to the data on the pod (for peer-to-peer rsync)
            lifespan: "cluster" for persistent, "resource" for service-scoped cleanup
            service_name: Service name for resource-scoped cleanup

        Returns:
            True if successful, False otherwise
        """
        try:
            # URL-encode the key to handle special characters
            encoded_key = quote(key, safe="")
            payload = {"ip": pod_ip, "lifespan": lifespan}
            if pod_name:
                payload["pod_name"] = pod_name
            if namespace:
                payload["namespace"] = namespace
            if src_path:
                payload["src_path"] = src_path
            if service_name:
                payload["service_name"] = service_name

            response = get_sync_client().post(
                f"{self.base_url}/api/v1/keys/{encoded_key}/publish",
                json=payload,
                timeout=5,
            )
            response.raise_for_status()
            return True
        except httpx.HTTPStatusError as e:
            logger.warning(f"Failed to publish key '{key}' from IP '{pod_ip}': HTTP {e.response.status_code}")
            return False
        except httpx.RequestError as e:
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
            response = get_sync_client().post(
                f"{self.base_url}/api/v1/keys/{encoded_key}/source/complete",
                json={"ip": ip},
                timeout=5,
            )
            response.raise_for_status()
            return True
        except httpx.HTTPStatusError as e:
            logger.debug(f"Failed to notify completion for key '{key}' IP '{ip}': HTTP {e.response.status_code}")
            return False
        except httpx.RequestError as e:
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
            httpx.ConnectError: If metadata server is unreachable
            httpx.TimeoutException: If metadata server connection times out
        """
        # URL-encode the key to handle special characters
        encoded_key = quote(key, safe="")
        response = get_sync_client().get(
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
            response = get_sync_client().delete(
                f"{self.base_url}/api/v1/keys/{encoded_key}/sources/{pod_ip}",
                timeout=5,
            )
            response.raise_for_status()
            return True
        except httpx.HTTPStatusError as e:
            logger.warning(f"Failed to remove source IP '{pod_ip}' for key '{key}': HTTP {e.response.status_code}")
            return False
        except httpx.RequestError as e:
            logger.warning(f"Failed to remove source IP '{pod_ip}' for key '{key}': {e}")
            return False

    def delete_key(self, key: str, recursive: bool = False, prefix_mode: bool = False) -> dict:
        """
        Delete a key from both the metadata server and filesystem.
        This removes virtual keys (locally-published) from the metadata server AND deletes files from the filesystem.

        Args:
            key: Storage key to delete
            recursive: If True, delete directories recursively (directory semantics - adds /)
            prefix_mode: If True, delete all keys starting with this string (no / added)

        Returns:
            dict with success status and details about what was deleted
        """
        try:
            # URL-encode the key to handle special characters
            encoded_key = quote(key, safe="")
            url = f"{self.base_url}/api/v1/keys/{encoded_key}"
            params = []
            if recursive:
                params.append("recursive=true")
            if prefix_mode:
                params.append("prefix_mode=true")
            if params:
                url += "?" + "&".join(params)
            response = get_sync_client().delete(url, timeout=30)  # Longer timeout for filesystem operations
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.warning(f"Failed to delete key '{key}': HTTP {e.response.status_code}")
            return {"success": False, "error": f"HTTP {e.response.status_code}"}
        except httpx.RequestError as e:
            logger.warning(f"Failed to delete key '{key}': {e}")
            return {"success": False, "error": str(e)}

    def mkdir(self, key: str) -> dict:
        """
        Create a directory at the given key path.

        Args:
            key: Storage key path to create

        Returns:
            dict with success status and created path
        """
        try:
            encoded_key = quote(key, safe="")
            url = f"{self.base_url}/api/v1/keys/{encoded_key}/mkdir"
            response = get_sync_client().post(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.warning(f"Failed to create directory for key '{key}': HTTP {e.response.status_code}")
            return {"success": False, "error": f"HTTP {e.response.status_code}"}
        except httpx.RequestError as e:
            logger.warning(f"Failed to create directory for key '{key}': {e}")
            return {"success": False, "error": str(e)}

    def list_keys(self, prefix: str = "") -> dict:
        """
        List all keys matching a prefix, combining locally-published keys and filesystem contents.

        Args:
            prefix: Key prefix to match (default: empty string for root)

        Returns:
            dict with "prefix" and "items" list, where each item has:
            - name: Item name (relative to prefix)
            - is_directory: True if directory
            - locale: Where the data lives - "store" for central store, or pod name for local data
        """
        try:
            encoded_prefix = quote(prefix, safe="")
            response = get_sync_client().get(
                f"{self.base_url}/api/v1/keys/list?prefix={encoded_prefix}",
                timeout=5,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.warning(f"Failed to list keys with prefix '{prefix}': HTTP {e.response.status_code}")
            return {"prefix": prefix, "items": []}
        except httpx.RequestError as e:
            logger.warning(f"Failed to list keys with prefix '{prefix}': {e}")
            return {"prefix": prefix, "items": []}

    def register_store_pod(
        self,
        key: str,
        store_pod_ip: str,
        lifespan: Lifespan = "cluster",
        service_name: Optional[str] = None,
    ) -> bool:
        """
        Register that the store pod itself has data for a key.
        Called by the store pod after storing data.

        Args:
            key: Storage key
            store_pod_ip: IP address of the store pod
            lifespan: "cluster" for persistent, "resource" for service-scoped cleanup
            service_name: Service name for resource-scoped cleanup

        Returns:
            True if successful, False otherwise
        """
        try:
            # URL-encode the key to handle special characters
            encoded_key = quote(key, safe="")
            payload = {"ip": store_pod_ip, "lifespan": lifespan}
            if service_name:
                payload["service_name"] = service_name

            response = get_sync_client().post(
                f"{self.base_url}/api/v1/keys/{encoded_key}/store",
                json=payload,
                timeout=5,
            )
            response.raise_for_status()
            return True
        except httpx.HTTPStatusError as e:
            logger.warning(
                f"Failed to register store pod IP '{store_pod_ip}' for key '{key}': HTTP {e.response.status_code}"
            )
            return False
        except httpx.RequestError as e:
            logger.warning(f"Failed to register store pod IP '{store_pod_ip}' for key '{key}': {e}")
            return False

    # ==================== Broadcast Quorum Methods ====================

    def join_broadcast(
        self,
        keys: List[str],
        role: str,
        pod_ip: str,
        pod_name: Optional[str] = None,
        broadcast: Optional[BroadcastWindow] = None,
    ) -> dict:
        """
        Join a broadcast quorum for coordinated data transfer.

        Args:
            keys: List of keys to transfer
            role: "putter" (data source) or "getter" (data destination)
            pod_ip: IP address of this pod
            pod_name: Name of this pod
            broadcast: BroadcastWindow configuration

        Returns:
            dict with broadcast_id and status
        """
        try:
            payload = {
                "keys": keys,
                "role": role,
                "pod_ip": pod_ip,
            }
            if pod_name:
                payload["pod_name"] = pod_name
            if broadcast:
                payload["timeout"] = broadcast.timeout
                payload["world_size"] = broadcast.world_size
                payload["ips"] = broadcast.ips
                payload["group_id"] = broadcast.group_id

            response = get_sync_client().post(
                f"{self.base_url}/api/v1/broadcast/join",
                json=payload,
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.warning(f"Failed to join broadcast quorum: HTTP {e.response.status_code}")
            return {}
        except httpx.RequestError as e:
            logger.warning(f"Failed to join broadcast quorum: {e}")
            return {}

    def get_broadcast_status(self, broadcast_id: str, pod_ip: str) -> dict:
        """
        Get the status of a broadcast quorum.

        Args:
            broadcast_id: ID of the broadcast quorum
            pod_ip: IP address of this pod (to get role-specific info)

        Returns:
            dict with status, putters, getters, and other info
        """
        try:
            response = get_sync_client().get(
                f"{self.base_url}/api/v1/broadcast/{broadcast_id}/status",
                params={"pod_ip": pod_ip},
                timeout=5,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.warning(f"Failed to get broadcast status for {broadcast_id}: HTTP {e.response.status_code}")
            return {"status": "error", "error": f"HTTP {e.response.status_code}"}
        except httpx.RequestError as e:
            logger.warning(f"Failed to get broadcast status for {broadcast_id}: {e}")
            return {"status": "error", "error": str(e)}

    def complete_broadcast(self, broadcast_id: str, pod_ip: str) -> bool:
        """
        Mark this participant as having completed the broadcast transfer.

        Args:
            broadcast_id: ID of the broadcast quorum
            pod_ip: IP address of this pod

        Returns:
            True if successful, False otherwise
        """
        try:
            response = get_sync_client().post(
                f"{self.base_url}/api/v1/broadcast/{broadcast_id}/complete",
                params={"pod_ip": pod_ip},
                timeout=5,
            )
            response.raise_for_status()
            return True
        except httpx.HTTPStatusError as e:
            logger.warning(f"Failed to complete broadcast {broadcast_id}: HTTP {e.response.status_code}")
            return False
        except httpx.RequestError as e:
            logger.warning(f"Failed to complete broadcast {broadcast_id}: {e}")
            return False

    def cleanup_service_keys(self, service_name: str) -> dict:
        """
        Delete all keys with lifespan='resource' for a service.
        Called when a service is torn down.

        Args:
            service_name: Name of the service to clean up

        Returns:
            dict with deleted_count and success status
        """
        try:
            response = get_sync_client().delete(
                f"{self.base_url}/api/v1/services/{quote(service_name, safe='')}/cleanup",
                params={"namespace": self.namespace},
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.warning(f"Failed to cleanup keys for service '{service_name}': HTTP {e.response.status_code}")
            return {"success": False, "error": f"HTTP {e.response.status_code}", "deleted_count": 0}
        except httpx.RequestError as e:
            logger.warning(f"Failed to cleanup keys for service '{service_name}': {e}")
            return {"success": False, "error": str(e), "deleted_count": 0}

    # ==================== Filesystem Broadcast Methods ====================

    def join_fs_broadcast(
        self,
        group_id: str,
        key: str,
        pod_ip: str,
        pod_name: Optional[str] = None,
        fanout: Optional[int] = None,
    ) -> dict:
        """
        Join a filesystem broadcast group and get parent info.

        Args:
            group_id: Broadcast group identifier
            key: Storage key to retrieve
            pod_ip: IP address of this pod
            pod_name: Name of this pod
            fanout: Tree fanout (default: 50 for filesystem)

        Returns:
            dict with status, rank, parent_ip, parent_rank, source_path, etc.
        """
        return self.join_fs_broadcast_with_callback(
            group_id=group_id,
            key=key,
            pod_ip=pod_ip,
            pod_name=pod_name,
            fanout=fanout,
            transfer_callback=None,
        )

    def join_fs_broadcast_with_callback(
        self,
        group_id: str,
        key: str,
        pod_ip: str,
        pod_name: Optional[str] = None,
        fanout: Optional[int] = None,
        transfer_callback=None,
    ) -> dict:
        """
        Join a filesystem broadcast group via WebSocket and perform transfer.

        This is a rolling participation system - the server returns immediately
        with parent info, allowing this node to start rsync right away.

        Args:
            group_id: Broadcast group identifier
            key: Storage key to retrieve
            pod_ip: IP address of this pod
            pod_name: Name of this pod
            fanout: Tree fanout (default: 50 for filesystem)
            transfer_callback: Function to call to perform the actual rsync.
                              Called with (parent_ip, parent_path, source_path) -> bool

        Returns:
            dict with status, rank, transfer result, etc.
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self._join_fs_broadcast_with_callback_async(
                            group_id, key, pod_ip, pod_name, fanout, transfer_callback
                        ),
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self._join_fs_broadcast_with_callback_async(
                        group_id, key, pod_ip, pod_name, fanout, transfer_callback
                    )
                )
        except RuntimeError:
            return asyncio.run(
                self._join_fs_broadcast_with_callback_async(group_id, key, pod_ip, pod_name, fanout, transfer_callback)
            )

    async def _join_fs_broadcast_with_callback_async(
        self,
        group_id: str,
        key: str,
        pod_ip: str,
        pod_name: Optional[str],
        fanout: Optional[int],
        transfer_callback,
    ) -> dict:
        """Async implementation of filesystem broadcast join with callback."""
        import asyncio
        import json

        try:
            from websockets.asyncio.client import connect as ws_connect
        except ImportError:
            from websockets import connect as ws_connect

        # Build WebSocket URL
        encoded_group_id = quote(group_id, safe="")
        base_url = self.base_url
        if base_url.startswith("https://"):
            ws_url = f"wss://{base_url[8:]}/ws/fs-broadcast/{encoded_group_id}"
        elif base_url.startswith("http://"):
            ws_url = f"ws://{base_url[7:]}/ws/fs-broadcast/{encoded_group_id}"
        else:
            ws_url = f"ws://{base_url}/ws/fs-broadcast/{encoded_group_id}"

        logger.debug(f"Joining filesystem broadcast group '{group_id}' for key '{key}'")

        try:
            async with ws_connect(ws_url) as websocket:
                # Send join message
                join_msg = {
                    "action": "join",
                    "key": key,
                    "pod_ip": pod_ip,
                    "pod_name": pod_name,
                }
                if fanout is not None:
                    join_msg["fanout"] = fanout

                await websocket.send(json.dumps(join_msg))

                # Wait for ready response (immediate for rolling participation)
                response = json.loads(await asyncio.wait_for(websocket.recv(), timeout=30.0))

                if response.get("event") == "error":
                    return {
                        "status": "error",
                        "error": response.get("message", "Unknown error"),
                    }

                if response.get("event") != "ready":
                    return {
                        "status": "error",
                        "error": f"Unexpected response: {response}",
                    }

                logger.debug(
                    f"Joined filesystem broadcast: rank={response.get('rank')}, "
                    f"parent_ip={response.get('parent_ip')}"
                )

                result = {
                    "status": "ready",
                    "rank": response.get("rank"),
                    "parent_ip": response.get("parent_ip"),
                    "parent_pod_name": response.get("parent_pod_name"),
                    "parent_path": response.get("parent_path"),
                    "parent_rank": response.get("parent_rank"),
                    "source_ip": response.get("source_ip"),
                    "source_pod_name": response.get("source_pod_name"),
                    "source_path": response.get("source_path"),
                }

            # Connection closed - now perform the transfer outside the websocket context
            transfer_success = False
            if transfer_callback:
                try:
                    transfer_success = transfer_callback(
                        result.get("parent_ip"),
                        result.get("parent_path"),
                        result.get("source_path"),
                    )
                except Exception as e:
                    logger.error(f"Transfer callback failed: {e}")
                    transfer_success = False

            result["transfer_success"] = transfer_success
            return result

        except asyncio.TimeoutError:
            return {"status": "error", "error": "Timeout waiting for server response"}
        except Exception as e:
            logger.error(f"Filesystem broadcast WebSocket error: {e}")
            return {"status": "error", "error": str(e)}
