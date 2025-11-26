"""
Helper class for GPU tensor transfer testing on remote cluster.

This helper runs inside a Kubernetes pod with GPU and provides methods for:
- Publishing GPU tensors via kt.put(data=tensor)
- Retrieving GPU tensors via kt.get(dest=tensor)
- Verifying tensor contents

With the GPU Data Server architecture, transfers are automatic:
- kt.put() registers tensor IPC handles with local GPU server
- kt.get() triggers server-to-server NCCL transfer
"""

import os
from typing import Dict, List

import kubetorch as kt


class GPUTestHelper:
    """Helper class for GPU tensor transfer testing on remote cluster."""

    @property
    def service_name(self) -> str:
        """Get the service name from KT_SERVICE_NAME environment variable."""
        service_name = os.getenv("KT_SERVICE_NAME")
        if not service_name:
            raise RuntimeError("KT_SERVICE_NAME environment variable not set")
        return service_name

    @property
    def pod_ip(self) -> str:
        """Get the pod IP from POD_IP environment variable."""
        pod_ip = os.getenv("POD_IP")
        if not pod_ip:
            raise RuntimeError("POD_IP environment variable not set")
        return pod_ip

    @property
    def pod_name(self) -> str:
        """Get the pod name from POD_NAME environment variable."""
        pod_name = os.getenv("POD_NAME")
        if not pod_name:
            raise RuntimeError("POD_NAME environment variable not set")
        return pod_name

    def check_gpu_available(self) -> Dict:
        """Check if GPU is available on this pod."""
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            device_count = torch.cuda.device_count() if cuda_available else 0
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else None

            return {
                "cuda_available": cuda_available,
                "device_count": device_count,
                "device_name": device_name,
            }
        except ImportError:
            return {"cuda_available": False, "error": "torch not installed"}
        except Exception as e:
            return {"cuda_available": False, "error": str(e)}

    def create_test_tensor(
        self,
        shape: List[int],
        dtype: str = "float32",
        fill_value: float = 1.0,
        device: str = "cuda:0",
    ) -> Dict:
        """
        Create a test tensor on GPU.

        Returns dict with tensor info (actual tensor is kept in memory for put).
        """
        import torch

        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "float16": torch.float16,
            "int32": torch.int32,
            "int64": torch.int64,
        }

        tensor = torch.full(shape, fill_value, dtype=dtype_map.get(dtype, torch.float32), device=device)

        # Store tensor reference
        self._test_tensor = tensor

        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "sum": float(tensor.sum().item()),
            "mean": float(tensor.mean().item()),
        }

    def publish_tensor(
        self,
        key: str,
        shape: List[int],
        dtype: str = "float32",
        fill_value: float = 1.0,
        nccl_port: int = 29500,
    ) -> Dict:
        """
        Create and publish a GPU tensor via put(data=...).

        The tensor is registered with the local GPU Data Server via IPC handles.
        When a consumer calls get(), the GPU servers coordinate the NCCL transfer.

        Args:
            key: Storage key for the tensor
            shape: Tensor shape
            dtype: Tensor dtype
            fill_value: Value to fill tensor with
            nccl_port: Port for NCCL communication
        """
        import torch

        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "float16": torch.float16,
            "int32": torch.int32,
            "int64": torch.int64,
        }

        tensor = torch.full(shape, fill_value, dtype=dtype_map.get(dtype, torch.float32), device="cuda:0")

        # Keep tensor alive - it must remain valid while registered
        if not hasattr(self, "_published_tensors"):
            self._published_tensors = {}
        self._published_tensors[key] = tensor

        try:
            kt.put(key=key, data=tensor, nccl_port=nccl_port, verbose=True)
            return {
                "success": True,
                "key": key,
                "shape": shape,
                "dtype": dtype,
                "fill_value": fill_value,
                "pod_ip": self.pod_ip,
                "pod_name": self.pod_name,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_tensor(
        self,
        key: str,
        shape: List[int],
        dtype: str = "float32",
        device: str = "cuda:0",
        quorum_timeout: float = 0.0,
    ) -> Dict:
        """
        Get a GPU tensor from the store via get with pre-allocated destination.

        The local GPU Data Server contacts the source's GPU server and
        performs the NCCL transfer automatically.

        Args:
            key: Storage key for the tensor
            shape: Shape of tensor to allocate
            dtype: Dtype of tensor to allocate
            device: Device to receive tensor on
            quorum_timeout: How long to wait for other consumers (0 = immediate)
        """
        import torch

        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "float16": torch.float16,
            "int32": torch.int32,
            "int64": torch.int64,
        }

        try:
            # Pre-allocate destination tensor
            dest_tensor = torch.empty(shape, dtype=dtype_map.get(dtype, torch.float32), device=device)

            # Get data into the pre-allocated tensor
            kt.get(key=key, dest=dest_tensor, quorum_timeout=quorum_timeout, verbose=True)

            return {
                "success": True,
                "key": key,
                "shape": list(dest_tensor.shape),
                "dtype": str(dest_tensor.dtype),
                "device": str(dest_tensor.device),
                "sum": float(dest_tensor.sum().item()),
                "mean": float(dest_tensor.mean().item()),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def publish_tensors(
        self,
        keys: List[str],
        shapes: List[List[int]],
        fill_values: List[float],
        dtype: str = "float32",
        nccl_port: int = 29500,
    ) -> Dict:
        """
        Publish multiple GPU tensors via put(data=...).

        Args:
            keys: Storage keys for each tensor
            shapes: Tensor shapes for each tensor
            fill_values: Values to fill each tensor with
            dtype: Tensor dtype (same for all)
            nccl_port: Port for NCCL communication
        """
        import torch

        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "float16": torch.float16,
            "int32": torch.int32,
            "int64": torch.int64,
        }

        if not hasattr(self, "_published_tensors"):
            self._published_tensors = {}

        results = []
        for key, shape, fill_value in zip(keys, shapes, fill_values):
            tensor = torch.full(shape, fill_value, dtype=dtype_map.get(dtype, torch.float32), device="cuda:0")
            self._published_tensors[key] = tensor

            try:
                kt.put(key=key, data=tensor, nccl_port=nccl_port, verbose=True)
                results.append(
                    {
                        "success": True,
                        "key": key,
                        "shape": shape,
                        "fill_value": fill_value,
                    }
                )
            except Exception as e:
                results.append({"success": False, "key": key, "error": str(e)})

        return {
            "success": all(r["success"] for r in results),
            "results": results,
            "pod_ip": self.pod_ip,
            "pod_name": self.pod_name,
        }

    def get_tensors(
        self,
        keys: List[str],
        shapes: List[List[int]],
        dtype: str = "float32",
        device: str = "cuda:0",
        quorum_timeout: float = 0.0,
    ) -> Dict:
        """
        Get multiple GPU tensors from the store.

        Args:
            keys: Storage keys for each tensor
            shapes: Shapes of tensors to allocate
            dtype: Dtype of tensors to allocate
            device: Device to receive tensors on
            quorum_timeout: How long to wait for other consumers (0 = immediate)
        """
        import torch

        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "float16": torch.float16,
            "int32": torch.int32,
            "int64": torch.int64,
        }

        results = []
        for key, shape in zip(keys, shapes):
            try:
                dest_tensor = torch.empty(shape, dtype=dtype_map.get(dtype, torch.float32), device=device)
                kt.get(key=key, dest=dest_tensor, quorum_timeout=quorum_timeout, verbose=True)

                results.append(
                    {
                        "success": True,
                        "key": key,
                        "shape": list(dest_tensor.shape),
                        "sum": float(dest_tensor.sum().item()),
                        "mean": float(dest_tensor.mean().item()),
                    }
                )
            except Exception as e:
                results.append({"success": False, "key": key, "error": str(e)})

        return {
            "success": all(r["success"] for r in results),
            "results": results,
        }

    def publish_tensor_with_broadcast(
        self,
        keys: List[str],
        shapes: List[List[int]],
        fill_values: List[float],
        broadcast_window: Dict,
        dtype: str = "float32",
    ) -> Dict:
        """
        Publish a GPU tensor using BroadcastWindow for coordinated transfer.

        Each rank extracts its own key/shape/fill_value from the lists based on LOCAL_RANK.

        Args:
            keys: List of storage keys (one per rank)
            shapes: List of tensor shapes (one per rank)
            fill_values: List of fill values (one per rank)
            broadcast_window: BroadcastWindow configuration dict
            dtype: Tensor dtype
        """
        import torch

        from kubetorch.data_store.types import BroadcastWindow

        # Get this process's local rank
        local_rank = int(os.getenv("LOCAL_RANK", "0"))

        # Extract this rank's parameters
        key = keys[local_rank]
        shape = shapes[local_rank]
        fill_value = fill_values[local_rank]

        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "float16": torch.float16,
            "int32": torch.int32,
            "int64": torch.int64,
        }

        try:
            tensor = torch.full(shape, fill_value, dtype=dtype_map.get(dtype, torch.float32), device="cuda:0")

            # Create BroadcastWindow from dict
            bw = BroadcastWindow(
                group_id=broadcast_window.get("group_id"),
                timeout=broadcast_window.get("timeout"),
                world_size=broadcast_window.get("world_size"),
                ips=broadcast_window.get("ips"),
            )

            result = kt.put(key=key, data=tensor, broadcast=bw, verbose=True)

            return {
                "success": True,
                "key": key,
                "shape": shape,
                "fill_value": fill_value,
                "local_rank": local_rank,
                "broadcast_result": result,
            }
        except Exception as e:
            return {"success": False, "key": key, "local_rank": local_rank, "error": str(e)}

    def get_tensor_with_broadcast(
        self,
        keys: List[str],
        shapes: List[List[int]],
        broadcast_window: Dict,
        dtype: str = "float32",
        device: str = "cuda:0",
    ) -> Dict:
        """
        Get a GPU tensor using BroadcastWindow for coordinated transfer.

        Each rank extracts its own key/shape from the lists based on LOCAL_RANK.

        Args:
            keys: List of storage keys (one per rank)
            shapes: List of expected tensor shapes (one per rank)
            broadcast_window: BroadcastWindow configuration dict
            dtype: Tensor dtype
            device: Device to receive tensor on
        """
        import torch

        from kubetorch.data_store.types import BroadcastWindow

        # Get this process's local rank
        local_rank = int(os.getenv("LOCAL_RANK", "0"))

        # Extract this rank's parameters
        key = keys[local_rank]
        shape = shapes[local_rank]

        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "float16": torch.float16,
            "int32": torch.int32,
            "int64": torch.int64,
        }

        try:
            dest_tensor = torch.empty(shape, dtype=dtype_map.get(dtype, torch.float32), device=device)

            # Create BroadcastWindow from dict
            bw = BroadcastWindow(
                group_id=broadcast_window.get("group_id"),
                timeout=broadcast_window.get("timeout"),
                world_size=broadcast_window.get("world_size"),
                ips=broadcast_window.get("ips"),
            )

            result = kt.get(key=key, dest=dest_tensor, broadcast=bw, verbose=True)

            return {
                "success": True,
                "key": key,
                "shape": list(dest_tensor.shape),
                "sum": float(dest_tensor.sum().item()),
                "mean": float(dest_tensor.mean().item()),
                "local_rank": local_rank,
                "broadcast_result": result,
            }
        except Exception as e:
            return {"success": False, "key": key, "local_rank": local_rank, "error": str(e)}

    def verify_tensor_values(
        self,
        key: str,
        expected_sum: float,
        expected_shape: List[int],
        dtype: str = "float32",
        device: str = "cuda:0",
        quorum_timeout: float = 0.0,
        tolerance: float = 1.0,  # Use larger tolerance for large tensor sums due to float32 precision
    ) -> Dict:
        """
        Get a tensor and verify its values match expectations.

        Args:
            key: Storage key
            expected_sum: Expected sum of tensor values
            expected_shape: Expected tensor shape (also used to allocate destination)
            dtype: Tensor dtype
            device: Device to receive on
            quorum_timeout: How long to wait for other consumers (0 = immediate)
            tolerance: Tolerance for float comparison
        """
        result = self.get_tensor(
            key=key, shape=expected_shape, dtype=dtype, device=device, quorum_timeout=quorum_timeout
        )

        if not result["success"]:
            return result

        shape_matches = result["shape"] == expected_shape
        sum_matches = abs(result["sum"] - expected_sum) < tolerance

        return {
            "success": True,
            "shape_matches": shape_matches,
            "sum_matches": sum_matches,
            "actual_shape": result["shape"],
            "actual_sum": result["sum"],
            "expected_shape": expected_shape,
            "expected_sum": expected_sum,
            "all_correct": shape_matches and sum_matches,
        }
