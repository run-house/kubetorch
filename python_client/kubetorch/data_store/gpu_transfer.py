"""
GPU tensor transfer support for kubetorch data store.

Enables zero-copy peer-to-peer GPU tensor transfers using NCCL broadcast
with quorum-based coordination through the metadata server.

Architecture:
- Publishing process registers tensor IPC handles with a per-node GPU Data Server
- GPU Data Server handles NCCL broadcasts, isolating NCCL from application processes
- Server-to-server coordination for multi-party broadcasts

Supports:
- Single tensors
- State dicts (Dict[str, Tensor]) - common for model weights
- Nested dicts of tensors

The receiver provides the destination tensor/state_dict, so no metadata
about tensor shapes/dtypes needs to be stored or transmitted.

BroadcastWindow support:
- When a BroadcastWindow is provided, multiple putters and getters coordinate
  through a unified quorum before performing NCCL transfers
- First participant to join becomes rank 0 (NCCL master)
- All participants receive a transfer manifest specifying what to send/receive
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

from kubetorch.logger import get_logger
from kubetorch.servers.http.utils import is_running_in_kubernetes

if TYPE_CHECKING:
    from .types import BroadcastWindow

logger = get_logger(__name__)

# Constants
DEFAULT_NCCL_PORT = 29500
DEFAULT_QUORUM_TIMEOUT = 0.0  # Default: start immediately, don't wait for others
QUORUM_POLL_INTERVAL = 0.1  # 100ms
GPU_SERVER_STARTUP_TIMEOUT = 10.0  # Seconds to wait for GPU server to start


@dataclass
class GPUBroadcastInfo:
    """Information needed to participate in an NCCL broadcast."""

    broadcast_id: str
    master_addr: str
    master_port: int
    rank: int
    world_size: int
    status: str = "waiting"  # "waiting", "ready", "completed", "missed"


def _get_torch():
    """Lazily import torch to avoid import errors when torch isn't installed."""
    try:
        import torch

        return torch
    except ImportError:
        raise ImportError("PyTorch is required for GPU tensor transfers. Install it with: pip install torch")


def _get_torch_distributed():
    """Lazily import torch.distributed."""
    torch = _get_torch()
    return torch.distributed


def _is_gpu_tensor(obj) -> bool:
    """Check if object is a GPU tensor."""
    try:
        torch = _get_torch()
        return isinstance(obj, torch.Tensor) and obj.is_cuda
    except ImportError:
        return False


def _is_gpu_data(obj) -> bool:
    """Check if object is GPU data (tensor or dict of tensors)."""
    if _is_gpu_tensor(obj):
        return True

    if isinstance(obj, dict):
        # Check if any values are GPU tensors (recursively)
        for v in obj.values():
            if _is_gpu_tensor(v) or (isinstance(v, dict) and _is_gpu_data(v)):
                return True

    return False


def _flatten_state_dict(state_dict: Dict, prefix: str = "") -> Dict[str, Any]:
    """Flatten a potentially nested state dict to flat key -> tensor mapping."""
    result = {}
    for key, value in state_dict.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten_state_dict(value, full_key))
        else:
            result[full_key] = value
    return result


def _get_sorted_tensor_keys(data: Union[Any, Dict]) -> List[str]:
    """Get sorted list of tensor keys for consistent ordering between sender and receiver."""
    torch = _get_torch()

    if isinstance(data, torch.Tensor):
        return [""]  # Single tensor, empty key

    # State dict - flatten and sort keys
    flat = _flatten_state_dict(data)
    return sorted(flat.keys())


def _get_tensor_by_key(data: Union[Any, Dict], key: str):
    """Get tensor from data by flattened key."""
    if key == "":
        return data  # Single tensor

    # Navigate nested dict
    parts = key.split(".")
    current = data
    for part in parts:
        current = current[part]
    return current


class GPUTransferManager:
    """
    Manages GPU tensor transfers via NCCL broadcast.

    This class is used internally by DataStoreClient for GPU data transfers.
    It coordinates with the per-node GPU Data Server for actual NCCL operations.
    """

    def __init__(self, namespace: Optional[str] = None):
        """Initialize the GPU transfer manager."""
        from kubetorch import globals

        self.namespace = namespace or globals.config.namespace

        from kubetorch.serving.constants import DATA_STORE_METADATA_PORT

        from .metadata_client import MetadataClient

        self.metadata_client = MetadataClient(namespace=self.namespace, metadata_port=DATA_STORE_METADATA_PORT)

        # GPU data server client (lazy initialization)
        self._gpu_server_client = None

        # Legacy: Storage for pending data (kept for backward compatibility with tests)
        self._pending_data: Dict[str, Dict] = {}
        self._process_group = None

    def _get_gpu_server_client(self):
        """Get or create the GPU data server client, starting server if needed."""
        if self._gpu_server_client is None:
            from .gpu_data_server import GPUDataServerClient, start_server_if_needed

            # Start server if not running
            server_pid = start_server_if_needed()
            logger.debug(f"GPU Data Server PID: {server_pid}")

            self._gpu_server_client = GPUDataServerClient()

        return self._gpu_server_client

    def publish(
        self,
        key: str,
        data: Union[Any, Dict],
        nccl_port: int = DEFAULT_NCCL_PORT,
        broadcast: Optional["BroadcastWindow"] = None,
        verbose: bool = False,
    ) -> Optional[Dict]:
        """
        Publish GPU data (tensor or state dict) for broadcast.

        This registers the tensor with the local GPU Data Server via IPC handles,
        allowing the server to broadcast the data when getters request it.
        The calling process retains ownership of the tensor memory.

        Args:
            key: Storage key
            data: GPU tensor or dict of GPU tensors (state dict)
            nccl_port: Port for NCCL communication (used as hint for server)
            broadcast: Optional BroadcastWindow for coordinated multi-party transfer.
                When provided, this call blocks until all participants join the quorum,
                then performs the NCCL transfer as part of a unified process group.
            verbose: Show detailed progress

        Returns:
            When broadcast is provided: Dict with transfer results including rank, world_size
            When broadcast is None: None (backward compatible)
        """
        if not is_running_in_kubernetes():
            raise RuntimeError("GPU publish can only be called from inside a Kubernetes pod")

        torch = _get_torch()

        # Validate data is on GPU
        if isinstance(data, torch.Tensor):
            if not data.is_cuda:
                raise ValueError("Tensor must be on a CUDA device")
            tensors_to_register = [("", data)]
        elif isinstance(data, dict):
            flat = _flatten_state_dict(data)
            tensors_to_register = []
            for k, v in flat.items():
                if isinstance(v, torch.Tensor):
                    if not v.is_cuda:
                        raise ValueError(f"Tensor at '{k}' must be on a CUDA device")
                    tensors_to_register.append((k, v))
        else:
            raise ValueError("Data must be a torch.Tensor or dict of tensors")

        # Get pod info
        pod_ip = os.getenv("POD_IP")
        pod_name = os.getenv("POD_NAME")

        if not pod_ip:
            raise RuntimeError("POD_IP environment variable not set")
        if not pod_name:
            raise RuntimeError("POD_NAME environment variable not set")

        # Get GPU data server client (starts server if needed)
        gpu_client = self._get_gpu_server_client()

        # Register each tensor with the GPU data server
        # For state dicts, we register each tensor with a composite key
        for tensor_key, tensor in tensors_to_register:
            full_key = f"{key}/{tensor_key}" if tensor_key else key

            response = gpu_client.register_tensor(
                key=full_key,
                tensor=tensor,
            )

            if response.get("status") != "ok":
                raise RuntimeError(f"Failed to register tensor '{full_key}' with GPU server: {response.get('error')}")

        # Register with metadata server (key + source info for discovery)
        success = self._publish_to_metadata_server(
            key=key,
            pod_ip=pod_ip,
            pod_name=pod_name,
            nccl_port=nccl_port,
            is_state_dict=len(tensors_to_register) > 1
            or (len(tensors_to_register) == 1 and tensors_to_register[0][0] != ""),
            tensor_keys=[k for k, _ in tensors_to_register] if len(tensors_to_register) > 1 else None,
        )

        if not success:
            raise RuntimeError(f"Failed to publish GPU data for key '{key}'")

        # Store reference for backward compatibility with tests
        self._pending_data[key] = {
            "data": data,
            "nccl_port": nccl_port,
        }

        if verbose:
            if len(tensors_to_register) == 1 and tensors_to_register[0][0] == "":
                tensor = tensors_to_register[0][1]
                logger.info(f"Published GPU tensor '{key}': shape={list(tensor.shape)}, dtype={tensor.dtype}")
            else:
                logger.info(f"Published GPU state dict '{key}': {len(tensors_to_register)} tensors")

        # If broadcast window specified, join the coordinated quorum
        if broadcast is not None:
            tensor = tensors_to_register[0][1] if tensors_to_register else None
            return self._join_broadcast_group(
                key=key,
                role="putter",
                tensor=tensor,
                broadcast=broadcast,
                verbose=verbose,
            )

        return None

    def retrieve(
        self,
        key: str,
        dest: Union[Any, Dict],
        quorum_timeout: float = DEFAULT_QUORUM_TIMEOUT,
        broadcast: Optional["BroadcastWindow"] = None,
        verbose: bool = False,
    ) -> Optional[Dict]:
        """
        Retrieve GPU data via NCCL broadcast into a pre-allocated destination.

        This method contacts the metadata server to find the source, then uses
        the local GPU data server to perform the NCCL transfer.

        Args:
            key: Storage key
            dest: Pre-allocated destination tensor or state_dict to receive into
            quorum_timeout: How long to wait for other consumers (default 0 = immediate)
            broadcast: Optional BroadcastWindow for coordinated multi-party transfer.
                When provided, this call blocks until all participants join the quorum,
                then performs the NCCL transfer as part of a unified process group.
            verbose: Show detailed progress

        Returns:
            When broadcast is provided: Dict with transfer results including rank, world_size
            When broadcast is None: None (backward compatible)
        """
        if not is_running_in_kubernetes():
            raise RuntimeError("GPU retrieve can only be called from inside a Kubernetes pod")

        torch = _get_torch()

        # Validate destination is on GPU
        if isinstance(dest, torch.Tensor):
            if not dest.is_cuda:
                raise ValueError("Destination tensor must be on a CUDA device")
            tensors_to_receive = [("", dest)]
        elif isinstance(dest, dict):
            flat = _flatten_state_dict(dest)
            tensors_to_receive = []
            for k, v in flat.items():
                if isinstance(v, torch.Tensor):
                    if not v.is_cuda:
                        raise ValueError(f"Tensor at '{k}' must be on a CUDA device")
                    tensors_to_receive.append((k, v))
        else:
            raise ValueError("dest must be a torch.Tensor or dict of tensors")

        # If broadcast window specified, join the coordinated quorum
        if broadcast is not None:
            tensor = tensors_to_receive[0][1] if tensors_to_receive else dest
            return self._join_broadcast_group(
                key=key,
                role="getter",
                tensor=tensor,
                broadcast=broadcast,
                verbose=verbose,
            )

        # Get source info from metadata server
        source_info = self._get_gpu_source_info(key)

        if source_info is None:
            raise RuntimeError(f"No GPU data source found for key '{key}'")

        source_ip = source_info["ip"]
        source_gpu_port = source_info.get("gpu_server_port", 29400)

        if verbose:
            logger.info(f"Found GPU source for '{key}': {source_ip}:{source_gpu_port}")

        # Get GPU data server client (starts server if needed)
        gpu_client = self._get_gpu_server_client()

        # Receive each tensor
        for tensor_key, tensor in tensors_to_receive:
            full_key = f"{key}/{tensor_key}" if tensor_key else key

            if verbose:
                logger.info(f"Receiving tensor '{full_key}': shape={list(tensor.shape)}")

            response = gpu_client.receive_broadcast(
                key=full_key,
                source_ip=source_ip,
                source_gpu_port=source_gpu_port,
                dest_tensor=tensor,
            )

            if response.get("status") != "ok":
                raise RuntimeError(f"Failed to receive tensor '{full_key}': {response.get('error')}")

        if verbose:
            if len(tensors_to_receive) == 1:
                logger.info(f"Successfully received GPU tensor '{key}'")
            else:
                logger.info(f"Successfully received GPU state dict '{key}': {len(tensors_to_receive)} tensors")

        return None

    def _join_broadcast_group(
        self,
        key: str,
        role: str,  # "putter" or "getter"
        tensor: Any,  # torch.Tensor
        broadcast: "BroadcastWindow",
        verbose: bool = False,
    ) -> Dict:
        """
        Join a GPU broadcast group for coordinated multi-party transfer.

        Uses WebSocket connection to the metadata server for quorum coordination.
        The protocol is:
        1. Connect to ws://{host}/ws/broadcast/{group_id}
        2. Send join message with key, role, pod_ip, etc.
        3. Wait for 'ready' event with rank assignment and transfer manifest
        4. Perform NCCL transfer via GPU data server
        5. Send 'complete' action and close connection

        Args:
            key: Storage key for this tensor
            role: "putter" (sender) or "getter" (receiver)
            tensor: The tensor to send (putter) or receive into (getter)
            broadcast: BroadcastWindow configuration
            verbose: Show detailed progress

        Returns:
            Dict with transfer results including rank, world_size, status
        """
        import asyncio

        # Run the async implementation in an event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a task
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run, self._join_broadcast_group_async(key, role, tensor, broadcast, verbose)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(self._join_broadcast_group_async(key, role, tensor, broadcast, verbose))
        except RuntimeError:
            # No event loop exists, create one
            return asyncio.run(self._join_broadcast_group_async(key, role, tensor, broadcast, verbose))

    async def _join_broadcast_group_async(
        self,
        key: str,
        role: str,
        tensor: Any,
        broadcast: "BroadcastWindow",
        verbose: bool = False,
    ) -> Dict:
        """Async implementation of broadcast group join using WebSocket."""
        import asyncio

        try:
            from websockets.asyncio.client import connect as ws_connect
        except ImportError:
            from websockets import connect as ws_connect

        if broadcast.group_id is None:
            raise ValueError("BroadcastWindow.group_id is required for GPU broadcast")

        pod_ip = os.getenv("POD_IP")
        pod_name = os.getenv("POD_NAME")

        if not pod_ip:
            raise RuntimeError("POD_IP environment variable not set")

        # Build WebSocket URL
        # Convert http(s):// to ws(s)://
        # URL-encode the group_id since it may contain slashes
        from urllib.parse import quote

        encoded_group_id = quote(broadcast.group_id, safe="")
        base_url = self.metadata_client.base_url
        if base_url.startswith("https://"):
            ws_url = f"wss://{base_url[8:]}/ws/broadcast/{encoded_group_id}"
        elif base_url.startswith("http://"):
            ws_url = f"ws://{base_url[7:]}/ws/broadcast/{encoded_group_id}"
        else:
            ws_url = f"ws://{base_url}/ws/broadcast/{encoded_group_id}"

        if verbose:
            logger.info(f"Connecting to broadcast group '{broadcast.group_id}' via WebSocket")

        # Calculate timeout for the entire operation
        timeout = broadcast.timeout or 600.0

        async with ws_connect(ws_url) as websocket:
            # Build join message
            join_msg = {
                "action": "join",
                "key": key,
                "role": role,
                "pod_ip": pod_ip,
                "pod_name": pod_name,
                "timeout": broadcast.timeout,
                "world_size": broadcast.world_size,
            }

            # For putters, include tensor metadata
            if role == "putter" and tensor is not None:
                join_msg["tensor_shape"] = list(tensor.shape)
                join_msg["tensor_dtype"] = str(tensor.dtype)

            # For getters, include the destination tensor's IPC handle
            # The GPU server will write received data directly into this tensor
            if role == "getter" and tensor is not None:
                from .gpu_data_server import _get_ipc_handle, _serialize_ipc_handle

                dest_ipc_handle = _get_ipc_handle(tensor)
                join_msg["dest_ipc_handle"] = _serialize_ipc_handle(dest_ipc_handle)
                join_msg["tensor_shape"] = list(tensor.shape)
                join_msg["tensor_dtype"] = str(tensor.dtype)

            # Send join message
            import json

            await websocket.send(json.dumps(join_msg))

            if verbose:
                logger.info(f"Sent join request as {role} for key '{key}'")

            # Wait for queued confirmation
            response = json.loads(await asyncio.wait_for(websocket.recv(), timeout=30.0))
            if response.get("event") == "error":
                raise RuntimeError(f"Failed to join broadcast: {response.get('message')}")
            if response.get("event") != "queued":
                raise RuntimeError(f"Unexpected response: {response}")

            if verbose:
                logger.info(f"Queued in broadcast group (position: {response.get('position')})")

            # Wait for ready notification or heartbeats
            ready_data = None
            start_time = asyncio.get_event_loop().time()

            while ready_data is None:
                try:
                    msg = json.loads(await asyncio.wait_for(websocket.recv(), timeout=35.0))

                    if msg.get("event") == "ready":
                        ready_data = msg
                    elif msg.get("event") == "heartbeat":
                        # Respond to server heartbeat
                        await websocket.send(json.dumps({"action": "heartbeat"}))
                    elif msg.get("event") == "error":
                        raise RuntimeError(f"Broadcast error: {msg.get('message')}")

                except asyncio.TimeoutError:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if elapsed > timeout:
                        raise RuntimeError(f"Timeout waiting for broadcast group to form ({timeout}s)")
                    # Send heartbeat to keep connection alive
                    await websocket.send(json.dumps({"action": "heartbeat"}))

            # Extract transfer info from ready message
            rank = ready_data["rank"]
            world_size = ready_data["world_size"]
            master_addr = ready_data["master_addr"]
            master_port = ready_data["master_port"]
            ancestors = ready_data.get("ancestors", [])
            sends = ready_data.get("sends", [])
            receives = ready_data.get("receives", [])
            local_transfers = ready_data.get("local_transfers", [])
            is_coordinator = ready_data.get("is_coordinator", False)

            if verbose:
                coord_str = " (COORDINATOR)" if is_coordinator else ""
                logger.info(
                    f"Broadcast group ready: rank={rank}, world_size={world_size}, "
                    f"master={master_addr}:{master_port}{coord_str}"
                )
                if ancestors:
                    logger.info(f"  Ancestors in tree: {ancestors}")
                if sends:
                    logger.info(f"  Will send: {[s['key'] for s in sends]}")
                if receives:
                    logger.info(f"  Will receive: {[r['key'] for r in receives]}")
                if local_transfers:
                    logger.info(f"  Local transfers: {[lt['key'] for lt in local_transfers]}")

            transfer_success = True
            transfer_result = {"status": "ok"}

            if is_coordinator:
                # Coordinator: Call GPU server with consolidated manifest
                # The GPU server handles NCCL for all tensors on this pod
                gpu_client = self._get_gpu_server_client()

                transfer_result = gpu_client.execute_broadcast_group(
                    group_id=broadcast.group_id,
                    rank=rank,
                    world_size=world_size,
                    master_addr=master_addr,
                    master_port=master_port,
                    sends=sends,
                    receives=receives,
                    local_transfers=local_transfers,
                )

                transfer_success = transfer_result.get("status") == "ok"

                if verbose:
                    if transfer_success:
                        logger.info(
                            f"GPU server completed: {len(sends)} sends, "
                            f"{len(receives)} receives, {len(local_transfers)} local"
                        )
                    else:
                        logger.error(f"GPU server failed: {transfer_result.get('error')}")
            else:
                # Non-coordinator: GPU transfer handled by coordinator
                # Our destination tensor was passed via IPC handle, so data will
                # appear in it once the coordinator completes
                if verbose:
                    logger.info("Waiting for coordinator to complete transfer...")

            # Send completion message
            await websocket.send(
                json.dumps(
                    {
                        "action": "complete",
                        "success": transfer_success,
                    }
                )
            )

            # Wait for completion acknowledgment
            try:
                ack = json.loads(await asyncio.wait_for(websocket.recv(), timeout=10.0))
                if ack.get("event") != "completed":
                    logger.warning(f"Unexpected completion ack: {ack}")
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for completion acknowledgment")

            if not transfer_success:
                raise RuntimeError(f"Broadcast transfer failed: {transfer_result.get('error')}")

            if verbose:
                logger.info(f"Broadcast group '{broadcast.group_id}' transfer complete")

            return {
                "status": "ok",
                "rank": rank,
                "world_size": world_size,
                "group_id": broadcast.group_id,
                "ancestors": ancestors,
                "is_coordinator": is_coordinator,
            }

    def _get_gpu_source_info(self, key: str) -> Optional[dict]:
        """Get GPU data source info from metadata server."""
        from urllib.parse import quote

        import requests

        encoded_key = quote(key, safe="")
        url = f"{self.metadata_client.base_url}/api/v1/keys/{encoded_key}/gpu/source"

        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()
            if data.get("found") is False:
                return None
            return data
        except requests.RequestException as e:
            logger.error(f"Failed to get GPU source for key '{key}': {e}")
            return None

    def serve_broadcast(
        self,
        key: str,
        broadcast_id: str,
        world_size: int,
        verbose: bool = False,
    ) -> None:
        """
        Serve a broadcast as the source pod (rank 0).

        Called when the quorum is ready and all participants are waiting.
        """
        dist = _get_torch_distributed()

        pending = self._pending_data.get(key)
        if pending is None:
            raise RuntimeError(f"No pending data for key '{key}'")

        data = pending["data"]
        nccl_port = pending["nccl_port"]
        pod_ip = os.getenv("POD_IP")

        if verbose:
            tensor_keys = _get_sorted_tensor_keys(data)
            logger.info(f"Starting broadcast for '{key}': world_size={world_size}, " f"tensors={len(tensor_keys)}")

        # Set up NCCL
        os.environ["MASTER_ADDR"] = pod_ip
        os.environ["MASTER_PORT"] = str(nccl_port)

        try:
            # Initialize process group as rank 0
            if not dist.is_initialized():
                dist.init_process_group(
                    backend="nccl",
                    rank=0,
                    world_size=world_size,
                )
                self._process_group = dist.group.WORLD
            else:
                ranks = list(range(world_size))
                self._process_group = dist.new_group(ranks)

            # Broadcast data
            self._broadcast_data(data)

            if verbose:
                logger.info(f"Broadcast complete for '{key}'")

            # Notify completion
            self._complete_broadcast(
                key=key,
                broadcast_id=broadcast_id,
                pod_ip=pod_ip,
            )

        finally:
            self._cleanup_process_group()

    def _broadcast_data(self, data: Union[Any, Dict]) -> None:
        """Broadcast tensor or state dict (sender side)."""
        dist = _get_torch_distributed()
        torch = _get_torch()

        # Get sorted keys for consistent ordering
        tensor_keys = _get_sorted_tensor_keys(data)

        for key in tensor_keys:
            tensor = _get_tensor_by_key(data, key)
            if isinstance(tensor, torch.Tensor):
                dist.broadcast(tensor, src=0, group=self._process_group)

    def _receive_into(self, dest: Union[Any, Dict]) -> None:
        """Receive broadcast into destination tensor or state dict."""
        dist = _get_torch_distributed()
        torch = _get_torch()

        # Get sorted keys for consistent ordering (must match sender)
        tensor_keys = _get_sorted_tensor_keys(dest)

        for key in tensor_keys:
            tensor = _get_tensor_by_key(dest, key)
            if isinstance(tensor, torch.Tensor):
                dist.broadcast(tensor, src=0, group=self._process_group)

    def _cleanup_process_group(self):
        """Clean up NCCL process group."""
        dist = _get_torch_distributed()
        if self._process_group is not None:
            if self._process_group != dist.group.WORLD:
                dist.destroy_process_group(self._process_group)
            else:
                dist.destroy_process_group()
            self._process_group = None

    def _publish_to_metadata_server(
        self,
        key: str,
        pod_ip: str,
        pod_name: str,
        nccl_port: int,
        is_state_dict: bool = False,
        tensor_keys: Optional[List[str]] = None,
    ) -> bool:
        """Publish GPU data key to metadata server."""
        from urllib.parse import quote

        import requests

        from .gpu_data_server import DEFAULT_SOCKET_PATH

        encoded_key = quote(key, safe="")
        url = f"{self.metadata_client.base_url}/api/v1/keys/{encoded_key}/gpu/publish"

        try:
            payload = {
                "ip": pod_ip,
                "pod_name": pod_name,
                "namespace": self.namespace,
                "nccl_port": nccl_port,
                "gpu_server_socket": DEFAULT_SOCKET_PATH,
                "is_state_dict": is_state_dict,
            }
            if tensor_keys:
                payload["tensor_keys"] = tensor_keys

            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to publish GPU key '{key}': {e}")
            return False

    def _request_broadcast(
        self,
        key: str,
        pod_ip: str,
        pod_name: str,
        quorum_timeout: float,
    ) -> GPUBroadcastInfo:
        """Request to join GPU broadcast quorum."""
        from urllib.parse import quote

        import requests

        encoded_key = quote(key, safe="")
        url = f"{self.metadata_client.base_url}/api/v1/keys/{encoded_key}/gpu/get"

        response = requests.post(
            url,
            json={
                "pod_ip": pod_ip,
                "pod_name": pod_name,
                "namespace": self.namespace,
                "quorum_timeout": quorum_timeout,
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        return GPUBroadcastInfo(
            broadcast_id=data["broadcast_id"],
            master_addr=data["master_addr"],
            master_port=data["master_port"],
            rank=data["rank"],
            world_size=data["world_size"],
            status=data["status"],
        )

    def _poll_quorum(
        self,
        key: str,
        broadcast_id: str,
        pod_ip: str,
    ) -> GPUBroadcastInfo:
        """Poll the quorum status."""
        from urllib.parse import quote

        import requests

        encoded_key = quote(key, safe="")
        url = f"{self.metadata_client.base_url}/api/v1/keys/{encoded_key}/gpu/quorum/{broadcast_id}"

        response = requests.get(url, params={"pod_ip": pod_ip}, timeout=5)
        response.raise_for_status()
        data = response.json()

        return GPUBroadcastInfo(
            broadcast_id=data["broadcast_id"],
            master_addr=data["master_addr"],
            master_port=data["master_port"],
            rank=data["rank"],
            world_size=data["world_size"],
            status=data["status"],
        )

    def _complete_broadcast(
        self,
        key: str,
        broadcast_id: str,
        pod_ip: str,
    ) -> None:
        """Notify that broadcast is complete."""
        from urllib.parse import quote

        import requests

        encoded_key = quote(key, safe="")
        url = f"{self.metadata_client.base_url}/api/v1/keys/{encoded_key}/gpu/quorum/{broadcast_id}/complete"

        try:
            response = requests.post(url, params={"pod_ip": pod_ip}, timeout=5)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"Failed to notify broadcast completion: {e}")


# Singleton instance
_gpu_manager: Optional[GPUTransferManager] = None


def _get_gpu_manager() -> GPUTransferManager:
    """Get or create the global GPU transfer manager."""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUTransferManager()
    return _gpu_manager
