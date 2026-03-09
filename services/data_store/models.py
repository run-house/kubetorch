"""
Pydantic models for the kubetorch data store server.

Request/response validation models for the metadata server API.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class DataType(str, Enum):
    """Type of data stored for a key."""

    FILESYSTEM = "filesystem"
    GPU = "gpu"
    MEMORY = "memory"


# Request models
class SourceRequest(BaseModel):
    ip: str
    pod_name: Optional[str] = None
    namespace: Optional[str] = None
    src_path: Optional[str] = None
    data_type: Optional[str] = None  # "filesystem", "gpu", "memory"
    # GPU-specific fields
    tensor_shape: Optional[List[int]] = None
    tensor_dtype: Optional[str] = None


class CompleteRequest(BaseModel):
    ip: str


class StoreRequest(BaseModel):
    ip: str


class BroadcastJoinRequest(BaseModel):
    """Request to join a broadcast quorum."""

    keys: List[str]
    role: str  # "putter" or "getter"
    pod_ip: str
    pod_name: Optional[str] = None
    namespace: Optional[str] = None
    timeout: Optional[float] = None
    world_size: Optional[int] = None
    ips: Optional[List[str]] = None
    group_id: Optional[str] = None


class GPUPublishRequest(BaseModel):
    """Request to publish GPU tensor data."""

    ip: str
    pod_name: str
    namespace: Optional[str] = None
    # For batch publishing: list of keys to publish (ignores path key if provided)
    keys: Optional[List[str]] = None
    tensor_shape: Optional[
        List[int]
    ] = None  # Optional - shape may not be known at publish time
    tensor_dtype: Optional[
        str
    ] = None  # Optional - dtype may not be known at publish time
    nccl_port: int = 29500  # NCCL_PORT_START default
    gpu_server_socket: Optional[str] = None  # Unix socket path of the GPU server
    gpu_server_port: int = (
        29400  # TCP port of the GPU server for server-to-server communication
    )
    is_state_dict: bool = False  # True if publishing a state dict (multiple tensors)
    tensor_keys: Optional[List[str]] = None  # List of tensor keys if state dict


class GPUGetRequest(BaseModel):
    """Request to get GPU tensor data with quorum support."""

    pod_ip: str
    pod_name: str
    namespace: Optional[str] = None
    quorum_timeout: float = 5.0  # DEFAULT_GPU_QUORUM_TIMEOUT default


class GPUSourcesRequest(BaseModel):
    """Request to get GPU sources for multiple keys at once."""

    keys: List[str]


# Response models
class GPUBroadcastInfo(BaseModel):
    """Information needed to join an NCCL broadcast."""

    broadcast_id: str
    master_addr: str
    master_port: int
    rank: int
    world_size: int
    tensor_shape: List[int]
    tensor_dtype: str
    status: str  # "waiting", "ready", "completed", "missed"


class GPUQuorumStatus(BaseModel):
    """Status of a GPU broadcast quorum."""

    broadcast_id: str
    key: str
    status: str
    participants: int
    world_size: Optional[int] = None
    master_addr: Optional[str] = None
    master_port: Optional[int] = None
    started_at: float
    timeout: float
