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

import httpx

from kubetorch.logger import get_logger
from kubetorch.serving.global_http_clients import get_sync_client
from kubetorch.serving.utils import is_running_in_kubernetes

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

        from kubetorch.provisioning.constants import DATA_STORE_METADATA_PORT

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
            from .pod_data_server import PodDataServerClient, start_server_if_needed

            # Start server if not running
            server_pid = start_server_if_needed()
            logger.debug(f"Pod Data Server PID: {server_pid}")

            self._gpu_server_client = PodDataServerClient()

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
                For state_dicts, all tensors are transferred in the same NCCL session.
                Use broadcast.pack=True for maximum efficiency (single packed buffer).
            verbose: Show detailed progress

        Returns:
            When broadcast is provided: Dict with transfer results including rank, world_size
            When broadcast is None: None (backward compatible)
        """
        if not is_running_in_kubernetes():
            raise RuntimeError("GPU publish can only be called from inside a Kubernetes pod")

        torch = _get_torch()

        # Validate data is on GPU and flatten if dict
        if isinstance(data, torch.Tensor):
            if not data.is_cuda:
                raise ValueError("Tensor must be on a CUDA device")
            tensors_to_register = [("", data)]
        elif isinstance(data, dict):
            flat = _flatten_state_dict(data)
            tensors_to_register = []
            for k, v in sorted(flat.items()):  # Sort for deterministic ordering
                if isinstance(v, torch.Tensor):
                    if not v.is_cuda:
                        raise ValueError(f"Tensor at '{k}' must be on a CUDA device")
                    tensors_to_register.append((k, v))
        else:
            raise ValueError("Data must be a torch.Tensor or dict of tensors")

        # Get GPU data server client (starts server if needed)
        gpu_client = self._get_gpu_server_client()

        # Handle packed mode for broadcasts
        if broadcast is not None and broadcast.pack and len(tensors_to_register) > 1:
            return self._publish_packed(
                key=key,
                tensors=tensors_to_register,
                broadcast=broadcast,
                gpu_client=gpu_client,
                verbose=verbose,
            )

        # Build list of all tensor info
        all_tensor_info = []
        for tensor_key, tensor in tensors_to_register:
            full_key = f"{key}/{tensor_key}" if tensor_key else key
            all_tensor_info.append(
                {
                    "key": full_key,
                    "tensor_key": tensor_key,
                    "tensor": tensor,
                }
            )

        # Broadcast mode: use unified put_tensors_broadcast for all cases (1 or N tensors)
        if broadcast is not None:
            broadcast_config = {
                "group_id": broadcast.group_id,
                "timeout": broadcast.timeout or 600.0,
                "world_size": broadcast.world_size,
            }

            # Single path handles both single tensor and multi-tensor broadcasts
            response = gpu_client.put_tensors_broadcast(
                keys=[info["key"] for info in all_tensor_info],
                tensors=[info["tensor"] for info in all_tensor_info],
                broadcast=broadcast_config,
            )

            if response.get("status") != "ok":
                raise RuntimeError(f"Failed to broadcast '{key}': {response.get('error')}")

            # Store reference for backward compatibility
            self._pending_data[key] = {"data": data, "nccl_port": nccl_port}

            if verbose:
                logger.info(f"Published GPU data '{key}': {len(tensors_to_register)} tensor(s) via broadcast")

            return response

        # No broadcast - batch register + publish all tensors in single call
        full_keys = [f"{key}/{tensor_key}" if tensor_key else key for tensor_key, _ in tensors_to_register]
        tensors = [tensor for _, tensor in tensors_to_register]
        response = gpu_client.put_tensor(keys=full_keys, tensors=tensors)
        if response.get("status") != "ok":
            raise RuntimeError(f"Failed to publish tensors: {response.get('error')}")

        if verbose:
            if len(tensors_to_register) == 1:
                _, tensor = tensors_to_register[0]
                logger.info(f"Published GPU tensor '{key}': shape={list(tensor.shape)}, dtype={tensor.dtype}")
            else:
                logger.info(f"Published GPU state dict '{key}': {len(tensors_to_register)} tensors")

        # Store reference for backward compatibility
        self._pending_data[key] = {"data": data, "nccl_port": nccl_port}

        return None

    def _publish_packed(
        self,
        key: str,
        tensors: List[tuple],
        broadcast: "BroadcastWindow",
        gpu_client,
        verbose: bool = False,
    ) -> Dict:
        """
        Publish state_dict using packed mode - concatenate all tensors into one buffer.

        This provides maximum efficiency by using a single NCCL broadcast for all tensors.
        Requires all participants to have identical dict structure.
        """
        torch = _get_torch()

        # Get sorted tensor keys and values
        sorted_keys = [k for k, _ in tensors]  # Already sorted in publish()
        sorted_tensors = [t for _, t in tensors]

        # Validate all tensors have same dtype (required for packing)
        dtypes = set(t.dtype for t in sorted_tensors)
        if len(dtypes) > 1:
            raise ValueError(
                f"pack=True requires all tensors to have the same dtype, "
                f"but found: {dtypes}. Use pack=False for mixed dtypes."
            )
        dtype = sorted_tensors[0].dtype

        # Pack all tensors into a single buffer
        packed = torch.cat([t.flatten().to(dtype) for t in sorted_tensors])

        if verbose:
            logger.info(
                f"Packed {len(sorted_tensors)} tensors into buffer: " f"shape={list(packed.shape)}, dtype={dtype}"
            )

        # Create packed key
        packed_key = f"{key}/__packed__"

        # Register the packed tensor
        response = gpu_client.put_tensor(key=packed_key, tensor=packed)
        if response.get("status") != "ok":
            raise RuntimeError(f"Failed to register packed tensor: {response.get('error')}")

        # Join broadcast with the packed tensor
        broadcast_config = {
            "group_id": broadcast.group_id,
            "timeout": broadcast.timeout or 600.0,
            "world_size": broadcast.world_size,
            "pack": True,
            "tensor_keys": sorted_keys,  # Send metadata about original structure
        }

        response = gpu_client.put_tensors_broadcast(
            keys=[packed_key],
            tensors=[packed],
            broadcast=broadcast_config,
        )

        if response.get("status") != "ok":
            raise RuntimeError(f"Failed to broadcast packed tensor: {response.get('error')}")

        if verbose:
            logger.info(f"Published packed GPU state dict '{key}': {len(tensors)} tensors in 1 broadcast")

        return response

    def retrieve(
        self,
        key: str,
        dest: Union[Any, Dict],
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
            broadcast: Optional BroadcastWindow for coordinated multi-party transfer.
                When provided, this call blocks until all participants join the quorum,
                then performs the NCCL transfer as part of a unified process group.
                For state_dicts, all tensors are received in the same NCCL session.
                Use broadcast.pack=True for maximum efficiency (single packed buffer).
                Use broadcast.timeout to control how long to wait for participants.
            verbose: Show detailed progress

        Returns:
            When broadcast is provided: Dict with transfer results including rank, world_size
            When broadcast is None: None (backward compatible)
        """
        if not is_running_in_kubernetes():
            raise RuntimeError("GPU retrieve can only be called from inside a Kubernetes pod")

        torch = _get_torch()

        # Validate destination is on GPU and flatten if dict
        if isinstance(dest, torch.Tensor):
            if not dest.is_cuda:
                raise ValueError("Destination tensor must be on a CUDA device")
            tensors_to_receive = [("", dest)]
        elif isinstance(dest, dict):
            flat = _flatten_state_dict(dest)
            tensors_to_receive = []
            for k, v in sorted(flat.items()):  # Sort for deterministic ordering
                if isinstance(v, torch.Tensor):
                    if not v.is_cuda:
                        raise ValueError(f"Tensor at '{k}' must be on a CUDA device")
                    tensors_to_receive.append((k, v))
        else:
            raise ValueError("dest must be a torch.Tensor or dict of tensors")

        # Get GPU data server client (starts server if needed)
        gpu_client = self._get_gpu_server_client()

        # Handle packed mode for broadcasts
        if broadcast is not None and broadcast.pack and len(tensors_to_receive) > 1:
            return self._retrieve_packed(
                key=key,
                tensors=tensors_to_receive,
                broadcast=broadcast,
                gpu_client=gpu_client,
                verbose=verbose,
            )

        # Broadcast mode: use unified get_tensors_broadcast for all cases (1 or N tensors)
        if broadcast is not None:
            broadcast_config = {
                "group_id": broadcast.group_id,
                "timeout": broadcast.timeout or 600.0,
                "world_size": broadcast.world_size,
            }

            # Build list of all tensors to receive
            all_tensor_info = []
            for tensor_key, tensor in tensors_to_receive:
                full_key = f"{key}/{tensor_key}" if tensor_key else key
                all_tensor_info.append(
                    {
                        "key": full_key,
                        "tensor_key": tensor_key,
                        "tensor": tensor,
                    }
                )

            if verbose:
                logger.info(f"Joining broadcast group for '{key}': {len(all_tensor_info)} tensor(s)")

            # Single path handles both single tensor and multi-tensor broadcasts
            response = gpu_client.get_tensors_broadcast(
                tensors=[(info["key"], info["tensor"]) for info in all_tensor_info],
                broadcast=broadcast_config,
            )

            if response.get("status") != "ok":
                raise RuntimeError(f"Failed to receive via broadcast '{key}': {response.get('error')}")

            if verbose:
                logger.info(f"Broadcast receive complete for '{key}': {len(all_tensor_info)} tensor(s)")

            return response

        # Point-to-point: Use unified get_tensor (handles both single and multiple tensors)
        keys = [f"{key}/{tensor_key}" if tensor_key else key for tensor_key, _ in tensors_to_receive]
        tensors = [tensor for _, tensor in tensors_to_receive]

        response = gpu_client.get_tensor(keys=keys, dest_tensors=tensors)
        if response.get("status") != "ok":
            raise RuntimeError(f"Failed to receive '{key}': {response.get('error')}")

        if verbose:
            if len(tensors_to_receive) == 1:
                logger.info(f"Successfully received GPU tensor '{key}'")
            else:
                logger.info(f"Successfully received GPU state dict '{key}': {len(tensors_to_receive)} tensors")

        return None

    def _retrieve_packed(
        self,
        key: str,
        tensors: List[tuple],
        broadcast: "BroadcastWindow",
        gpu_client,
        verbose: bool = False,
    ) -> Dict:
        """
        Retrieve state_dict using packed mode - receive single packed buffer and unpack.

        This provides maximum efficiency by using a single NCCL broadcast.
        Requires all participants to have identical dict structure.
        """
        torch = _get_torch()

        # Get sorted tensors (already sorted in retrieve())
        sorted_tensors = [t for _, t in tensors]

        # Validate all tensors have same dtype
        dtypes = set(t.dtype for t in sorted_tensors)
        if len(dtypes) > 1:
            raise ValueError(
                f"pack=True requires all tensors to have the same dtype, "
                f"but found: {dtypes}. Use pack=False for mixed dtypes."
            )
        dtype = sorted_tensors[0].dtype

        # Calculate total size and allocate packed buffer
        total_numel = sum(t.numel() for t in sorted_tensors)
        packed = torch.empty(total_numel, dtype=dtype, device=sorted_tensors[0].device)

        if verbose:
            logger.info(
                f"Allocated packed buffer for {len(sorted_tensors)} tensors: "
                f"total_numel={total_numel}, dtype={dtype}"
            )

        # Create packed key
        packed_key = f"{key}/__packed__"

        # Join broadcast to receive the packed tensor
        broadcast_config = {
            "group_id": broadcast.group_id,
            "timeout": broadcast.timeout or 600.0,
            "world_size": broadcast.world_size,
            "pack": True,
        }

        response = gpu_client.get_tensors_broadcast(
            tensors=[(packed_key, packed)],
            broadcast=broadcast_config,
        )

        if response.get("status") != "ok":
            raise RuntimeError(f"Failed to receive packed tensor: {response.get('error')}")

        # Unpack into destination tensors
        offset = 0
        for tensor_key, dest_tensor in tensors:
            numel = dest_tensor.numel()
            dest_tensor.copy_(packed[offset : offset + numel].view_as(dest_tensor))
            offset += numel

        # Validate we consumed exactly the right amount
        if offset != packed.numel():
            raise ValueError(
                f"State dict structure mismatch: expected {offset} elements "
                f"but packed tensor has {packed.numel()} elements. "
                f"Ensure putter and getter have identical dict structure."
            )

        if verbose:
            logger.info(f"Unpacked {len(tensors)} tensors from packed buffer")

        return response

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

        encoded_key = quote(key, safe="")
        url = f"{self.metadata_client.base_url}/api/v1/keys/{encoded_key}/gpu/quorum/{broadcast_id}/complete"

        try:
            response = get_sync_client().post(url, params={"pod_ip": pod_ip}, timeout=5)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.warning(f"Failed to notify broadcast completion: HTTP {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            logger.warning(f"Failed to notify broadcast completion: {e}")


# Singleton instance
_gpu_manager: Optional[GPUTransferManager] = None


def _get_gpu_manager() -> GPUTransferManager:
    """Get or create the global GPU transfer manager."""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUTransferManager()
    return _gpu_manager
