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

        # Get GPU data server client (starts server if needed)
        gpu_client = self._get_gpu_server_client()

        # Determine if this is a state dict
        is_state_dict = len(tensors_to_register) > 1 or (
            len(tensors_to_register) == 1 and tensors_to_register[0][0] != ""
        )
        tensor_keys = [k for k, _ in tensors_to_register] if len(tensors_to_register) > 1 else None

        # Use high-level put_tensor API - GPU Data Server handles registration + MDS publish
        # For broadcast, the GPU Data Server joins the broadcast group via WebSocket
        last_response = None
        for i, (tensor_key, tensor) in enumerate(tensors_to_register):
            full_key = f"{key}/{tensor_key}" if tensor_key else key
            is_last = i == len(tensors_to_register) - 1

            # Build broadcast config for the last tensor (when broadcast is specified)
            broadcast_config = None
            if is_last and broadcast is not None:
                broadcast_config = {
                    "group_id": broadcast.group_id,
                    "timeout": broadcast.timeout or 600.0,
                    "world_size": broadcast.world_size,
                }

            # For state dicts, only publish to MDS on the last tensor
            # For single tensors, publish immediately
            # Broadcast coordination happens via GPU Data Server when broadcast_config is set
            response = gpu_client.put_tensor(
                key=full_key,
                tensor=tensor,
                is_state_dict=is_state_dict if is_last else False,
                tensor_keys=tensor_keys if is_last else None,
                broadcast=broadcast_config,
            )

            if response.get("status") != "ok":
                raise RuntimeError(f"Failed to publish tensor '{full_key}': {response.get('error')}")

            if is_last:
                last_response = response

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

        # Return broadcast result if broadcast was specified
        if broadcast is not None and last_response is not None:
            return last_response

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

        This method uses the GPU Data Server's high-level API which handles
        MDS lookup and NCCL transfer automatically.

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

        # Get GPU data server client (starts server if needed)
        gpu_client = self._get_gpu_server_client()

        # If broadcast window specified, join via GPU Data Server's high-level API
        # GPU Data Server handles WebSocket coordination internally
        if broadcast is not None:
            # Build broadcast config
            broadcast_config = {
                "group_id": broadcast.group_id,
                "timeout": broadcast.timeout or 600.0,
                "world_size": broadcast.world_size,
            }

            # For broadcast, use the first tensor (consistent with previous behavior)
            tensor_key, tensor = tensors_to_receive[0]
            full_key = f"{key}/{tensor_key}" if tensor_key else key

            if verbose:
                logger.info(f"Joining broadcast group for '{full_key}': shape={list(tensor.shape)}")

            response = gpu_client.get_tensor(
                keys=full_key,
                dest_tensors=tensor,
                broadcast=broadcast_config,
            )

            if response.get("status") != "ok":
                raise RuntimeError(f"Failed to receive tensor via broadcast '{full_key}': {response.get('error')}")

            if verbose:
                logger.info(f"Broadcast receive complete for '{key}'")

            return response

        # Point-to-point: Use batch get_tensor (MDS lookup + NCCL in single session)
        keys = [f"{key}/{tensor_key}" if tensor_key else key for tensor_key, _ in tensors_to_receive]
        tensors = [tensor for _, tensor in tensors_to_receive]

        if verbose:
            logger.info(f"Receiving {len(tensors)} tensor(s) for '{key}'")

        response = gpu_client.get_tensor(keys=keys, dest_tensors=tensors)

        if response.get("status") != "ok":
            raise RuntimeError(f"Failed to receive tensors for '{key}': {response.get('error')}")

        if verbose:
            if len(tensors_to_receive) == 1:
                logger.info(f"Successfully received GPU tensor '{key}'")
            else:
                logger.info(f"Successfully received GPU state dict '{key}': {len(tensors_to_receive)} tensors")

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
