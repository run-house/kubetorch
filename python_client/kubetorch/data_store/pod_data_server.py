"""
Pod Data Server - Per-node server for data transfers (GPU tensors and filesystem broadcasts).

This server runs as a separate process on each node to handle:
- GPU tensor transfers via NCCL broadcasts
- Filesystem broadcast coordination for tree-based p2p propagation

Architecture:
- GPU: Application processes call kt.put(src=tensor) which registers the tensor
  via CUDA IPC handles with this server. The server performs NCCL broadcasts.
- Filesystem: Tracks completed filesystem broadcasts with local paths. Child getters
  request data from parent's pod data server, which blocks until parent completes
  and returns the local path for rsync.
- Server-to-server communication for coordination (no metadata server bounce)

Usage:
    # Start server (typically done automatically on first kt.put)
    python -m kubetorch.data_store.pod_data_server

    # Or programmatically
    from kubetorch.data_store.pod_data_server import start_server
    start_server()
"""

import base64
import ctypes
import json
import os
import signal
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from kubetorch.logger import get_logger

logger = get_logger(__name__)


def _setup_cuda_ipc_permissions():
    """
    Enable ptrace permissions required for CUDA IPC in PyTorch 2.5+.

    PyTorch 2.5+ uses expandable segments which require pidfd_getfd syscall
    for IPC. This needs ptrace permission, otherwise IPC reconstruction segfaults.
    See: https://github.com/pytorch/pytorch/issues/165419
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PTRACER = 0x59616D61
        PR_SET_PTRACER_ANY = ctypes.c_ulong(-1).value
        libc.prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0)
    except Exception:
        pass


# Constants
DEFAULT_SOCKET_PATH = "/tmp/kt-gpu-data-server.sock"
DEFAULT_TCP_PORT = 29400  # TCP port for server-to-server communication
DEFAULT_NCCL_PORT_RANGE_START = 29500
DEFAULT_NCCL_PORT_RANGE_END = 29600
SERVER_PID_FILE = "/tmp/kt-gpu-data-server.pid"

# Configurable via environment variables
KT_NCCL_TIMEOUT_SECONDS = int(os.environ.get("KT_NCCL_TIMEOUT_SECONDS", "60"))
KT_NCCL_MAX_FAILURES = int(os.environ.get("KT_NCCL_MAX_FAILURES", "3"))


def _get_nccl_cuda_versions() -> dict:
    """Get NCCL and CUDA version information for compatibility checking."""
    try:
        torch = _get_torch()
        versions = {
            "cuda_version": getattr(torch.version, "cuda", None),
            "torch_version": torch.__version__,
        }
        # NCCL version is only available if CUDA is available
        if torch.cuda.is_available():
            try:
                versions["nccl_version"] = torch.cuda.nccl.version()
            except Exception:
                versions["nccl_version"] = None
        else:
            versions["nccl_version"] = None
        return versions
    except Exception:
        return {"cuda_version": None, "torch_version": None, "nccl_version": None}


def _check_version_compatibility(local_versions: dict, remote_versions: dict) -> tuple[bool, str]:
    """
    Check if NCCL/CUDA versions are compatible between two nodes.

    Returns:
        (is_compatible, error_message)
    """
    # Check NCCL version (must match major.minor)
    local_nccl = local_versions.get("nccl_version")
    remote_nccl = remote_versions.get("nccl_version")

    if local_nccl and remote_nccl:
        # NCCL version is a tuple like (2, 19, 3)
        if isinstance(local_nccl, (list, tuple)) and isinstance(remote_nccl, (list, tuple)):
            if len(local_nccl) >= 2 and len(remote_nccl) >= 2:
                if local_nccl[0] != remote_nccl[0] or local_nccl[1] != remote_nccl[1]:
                    return False, (
                        f"NCCL version mismatch: local={'.'.join(map(str, local_nccl))}, "
                        f"remote={'.'.join(map(str, remote_nccl))}. "
                        "NCCL requires matching major.minor versions for communication."
                    )

    # Check CUDA version (warn but don't fail on minor mismatch)
    local_cuda = local_versions.get("cuda_version")
    remote_cuda = remote_versions.get("cuda_version")

    if local_cuda and remote_cuda and local_cuda != remote_cuda:
        # Log warning but don't fail - CUDA minor version mismatches often work
        logger.warning(f"CUDA version mismatch: local={local_cuda}, remote={remote_cuda}")

    return True, ""


def _serialize_ipc_handle(ipc_handle: Tuple) -> List:
    """
    Serialize a CUDA IPC handle tuple to a JSON-compatible format.

    The IPC handle from _share_cuda_() contains bytes objects that need
    to be base64-encoded for JSON serialization.

    Handle format: (device, handle_bytes, size, offset, ref_counter_handle_bytes,
                   ref_counter_offset, event_handle_bytes, event_sync_required)
    """
    serialized = []
    for item in ipc_handle:
        if isinstance(item, bytes):
            # Base64 encode bytes and mark with prefix
            serialized.append({"_bytes": base64.b64encode(item).decode("ascii")})
        else:
            serialized.append(item)
    return serialized


def _deserialize_ipc_handle(serialized: List) -> Tuple:
    """
    Deserialize a JSON-compatible IPC handle back to the original format.
    """
    deserialized = []
    for item in serialized:
        if isinstance(item, dict) and "_bytes" in item:
            # Decode base64 bytes
            deserialized.append(base64.b64decode(item["_bytes"]))
        else:
            deserialized.append(item)
    return tuple(deserialized)


@dataclass
class RegisteredTensor:
    """Metadata for a registered GPU tensor."""

    key: str
    ipc_handle: Tuple  # CUDA IPC handle tuple from _share_cuda_()
    shape: Tuple[int, ...]
    dtype: str
    device: int  # CUDA device index
    pid: int  # PID of the process that registered this tensor
    registered_at: float = field(default_factory=time.time)


@dataclass
class BroadcastRequest:
    """Request from a getter to participate in a broadcast."""

    key: str
    getter_ip: str
    getter_port: int
    shape: Tuple[int, ...]  # Expected shape (for validation)
    dtype: str  # Expected dtype (for validation)


def _get_torch():
    """Lazily import torch."""
    try:
        import torch

        return torch
    except ImportError:
        raise ImportError("PyTorch is required for GPU data server")


def _get_torch_distributed():
    """Lazily import torch.distributed."""
    torch = _get_torch()
    return torch.distributed


def _get_ipc_handle(tensor) -> Tuple:
    """Get CUDA IPC handle from a tensor."""
    _setup_cuda_ipc_permissions()

    if hasattr(tensor, "untyped_storage"):
        storage = tensor.untyped_storage()
    else:
        storage = tensor.storage()

    return storage._share_cuda_()


def _reconstruct_tensor_from_ipc(
    ipc_handle: Tuple,
    shape: Tuple[int, ...],
    dtype_str: str,
    device: int,
):
    """Reconstruct a CUDA tensor from an IPC handle."""
    torch = _get_torch()

    dtype_map = {
        "torch.float32": torch.float32,
        "torch.float64": torch.float64,
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.int32": torch.int32,
        "torch.int64": torch.int64,
        "torch.int16": torch.int16,
        "torch.int8": torch.int8,
        "torch.uint8": torch.uint8,
        "torch.bool": torch.bool,
    }
    dtype = dtype_map.get(dtype_str, torch.float32)

    torch.cuda._lazy_init()
    _setup_cuda_ipc_permissions()

    # IPC handle format: (device, handle, size, offset, ref_counter_handle,
    #                     ref_counter_offset, event_handle, event_sync_required)
    if hasattr(torch, "UntypedStorage"):
        storage = torch.UntypedStorage._new_shared_cuda(
            ipc_handle[0],
            ipc_handle[1],
            ipc_handle[2],
            ipc_handle[3],
            ipc_handle[4],
            ipc_handle[5],
            ipc_handle[6],
            ipc_handle[7],
        )
    elif hasattr(torch.cuda, "UntypedStorage"):
        storage = torch.cuda.UntypedStorage._new_shared_cuda(
            ipc_handle[0],
            ipc_handle[1],
            ipc_handle[2],
            ipc_handle[3],
            ipc_handle[4],
            ipc_handle[5],
            ipc_handle[6],
            ipc_handle[7],
        )
    else:
        storage = torch.cuda.ByteStorage._new_shared_cuda(
            ipc_handle[0],
            ipc_handle[1],
            ipc_handle[2],
            ipc_handle[3],
            ipc_handle[4],
            ipc_handle[5],
            ipc_handle[6],
            ipc_handle[7],
        )

    tensor = torch.empty(shape, dtype=dtype, device=f"cuda:{device}")
    tensor.set_(storage, storage_offset=0, size=shape)

    return tensor


class PodDataServer:
    """
    Per-node server for GPU tensor transfers via NCCL.

    Handles:
    - Registration of tensors from local application processes (via IPC handles)
    - NCCL broadcast coordination with other GPU servers
    - PID tracking for cleanup when processes die
    """

    def __init__(
        self,
        socket_path: str = DEFAULT_SOCKET_PATH,
        tcp_port: int = DEFAULT_TCP_PORT,
        nccl_port_start: int = DEFAULT_NCCL_PORT_RANGE_START,
        nccl_port_end: int = DEFAULT_NCCL_PORT_RANGE_END,
    ):
        self.socket_path = socket_path
        self.tcp_port = tcp_port
        self.nccl_port_start = nccl_port_start
        self.nccl_port_end = nccl_port_end
        self._next_nccl_port = nccl_port_start

        # Registered tensors: key -> RegisteredTensor
        self._registered: Dict[str, RegisteredTensor] = {}

        # PID -> list of keys (for cleanup when process dies)
        self._pid_keys: Dict[int, List[str]] = {}

        # Lock for thread safety
        self._lock = threading.Lock()

        # Server sockets
        self._unix_socket: Optional[socket.socket] = None
        self._tcp_socket: Optional[socket.socket] = None
        self._running = False

        # Active NCCL process groups (broadcast_id -> process_group)
        self._process_groups: Dict[str, Any] = {}

        # Pending receive requests: broadcast_id -> {dest_tensor, event}
        self._pending_receives: Dict[str, Dict] = {}

        # NCCL failure tracking for auto-restart
        self._consecutive_nccl_failures = 0
        self._nccl_failure_lock = threading.Lock()

        # Cache version info at startup
        self._versions = _get_nccl_cuda_versions()

        # MDS client (lazy initialization)
        self._mds_base_url: Optional[str] = None

        # Pod info from environment
        self._pod_ip = os.getenv("POD_IP")
        self._pod_name = os.getenv("POD_NAME")
        self._namespace = os.getenv("KT_NAMESPACE")

        # Internal NCCL execution coordination per broadcast group
        # Ensures NCCL runs exactly once per pod, even with multiple participants
        # group_id -> {"executing": bool, "result": dict, "event": threading.Event}
        self._broadcast_execution: Dict[str, Dict] = {}
        self._broadcast_execution_lock = threading.Lock()

        # Filesystem broadcast tracking
        # (group_id, key) -> {"local_path": str, "completed_at": float}
        self._fs_broadcasts_completed: Dict[Tuple[str, str], Dict] = {}
        # (group_id, key) -> threading.Event for waiting
        self._fs_broadcast_events: Dict[Tuple[str, str], threading.Event] = {}
        self._fs_broadcast_lock = threading.Lock()
        # TTL for completed broadcasts (10 minutes)
        self._fs_broadcast_ttl = 600

    def _record_nccl_success(self):
        """Record a successful NCCL operation, resetting failure counter."""
        with self._nccl_failure_lock:
            self._consecutive_nccl_failures = 0

    def _record_nccl_failure(self, error: str):
        """
        Record an NCCL failure. If consecutive failures exceed threshold,
        terminate the server to allow a clean restart.
        """
        with self._nccl_failure_lock:
            self._consecutive_nccl_failures += 1
            count = self._consecutive_nccl_failures

        logger.error(f"NCCL failure #{count}/{KT_NCCL_MAX_FAILURES}: {error}")

        if count >= KT_NCCL_MAX_FAILURES:
            logger.critical(
                f"Pod Data Server terminating after {count} consecutive NCCL failures. "
                "NCCL state is likely corrupted. Server will auto-restart on next request. "
                "Registered tensors will need to be re-published with kt.put()."
            )
            # Use os._exit to ensure immediate termination without cleanup
            # that might hang on corrupted NCCL state
            os._exit(1)

    def start(self):
        """Start the GPU data server."""
        # Remove stale socket file
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        # Write PID file
        with open(SERVER_PID_FILE, "w") as f:
            f.write(str(os.getpid()))

        logger.info(f"Pod Data Server starting (PID: {os.getpid()})")
        logger.info(f"Unix socket: {self.socket_path}")
        logger.info(f"TCP port: {self.tcp_port}")
        logger.info(f"PID file: {SERVER_PID_FILE}")

        # Create Unix socket for local process communication
        self._unix_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._unix_socket.bind(self.socket_path)
        self._unix_socket.listen(16)

        # Create TCP socket for server-to-server communication
        self._tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._tcp_socket.bind(("0.0.0.0", self.tcp_port))
        self._tcp_socket.listen(16)

        self._running = True

        # Start PID monitoring thread
        pid_monitor = threading.Thread(target=self._monitor_pids, daemon=True)
        pid_monitor.start()

        # Start TCP server thread for remote requests
        tcp_thread = threading.Thread(target=self._tcp_accept_loop, daemon=True)
        tcp_thread.start()

        # Handle shutdown signals
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        logger.info("Pod Data Server ready")

        # Main accept loop for Unix socket (local clients)
        while self._running:
            try:
                self._unix_socket.settimeout(1.0)
                try:
                    client_socket, _ = self._unix_socket.accept()
                except socket.timeout:
                    continue

                # Handle client in thread
                handler = threading.Thread(
                    target=self._handle_local_client,
                    args=(client_socket,),
                    daemon=True,
                )
                handler.start()

            except Exception as e:
                if self._running:
                    logger.error(f"Error in accept loop: {e}")

        self._cleanup()

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signal."""
        logger.info("Shutdown signal received")
        self._running = False

    def _cleanup(self):
        """Clean up server resources."""
        logger.info("Cleaning up Pod Data Server")

        if self._unix_socket:
            self._unix_socket.close()

        if self._tcp_socket:
            self._tcp_socket.close()

        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        if os.path.exists(SERVER_PID_FILE):
            os.unlink(SERVER_PID_FILE)

    def _tcp_accept_loop(self):
        """Accept loop for TCP connections from remote GPU servers."""
        while self._running:
            try:
                self._tcp_socket.settimeout(1.0)
                try:
                    client_socket, addr = self._tcp_socket.accept()
                except socket.timeout:
                    continue

                logger.debug(f"TCP connection from {addr}")

                # Handle remote client in thread
                handler = threading.Thread(
                    target=self._handle_remote_client,
                    args=(client_socket, addr),
                    daemon=True,
                )
                handler.start()

            except Exception as e:
                if self._running:
                    logger.error(f"Error in TCP accept loop: {e}")

    def _handle_remote_client(self, client_socket: socket.socket, addr: tuple):
        """Handle a connection from a remote GPU server."""
        try:
            # Read message length (4 bytes, big-endian)
            length_data = client_socket.recv(4)
            if not length_data:
                return

            msg_length = struct.unpack(">I", length_data)[0]

            # Read message
            data = b""
            while len(data) < msg_length:
                chunk = client_socket.recv(min(msg_length - len(data), 4096))
                if not chunk:
                    break
                data += chunk

            # Parse JSON message
            message = json.loads(data.decode("utf-8"))
            command = message.get("command")

            # Dispatch remote commands
            if command == "request_broadcast":
                # Remote getter is requesting to receive data
                response = self._handle_remote_broadcast_request(message)
            elif command == "join_broadcast":
                # Source server telling us to join as receiver
                response = self._handle_join_broadcast(message)
            elif command == "fs_broadcast_get_path":
                # Child getter requesting local path for filesystem broadcast
                response = self._handle_fs_broadcast_get_path(message)
            elif command == "ping":
                response = {"status": "ok", "pid": os.getpid(), "tcp_port": self.tcp_port}
            else:
                response = {"status": "error", "error": f"Unknown remote command: {command}"}

            # Send response
            response_data = json.dumps(response).encode("utf-8")
            client_socket.sendall(struct.pack(">I", len(response_data)))
            client_socket.sendall(response_data)

        except Exception as e:
            logger.error(f"Error handling remote client {addr}: {e}")
            try:
                error_response = json.dumps({"status": "error", "error": str(e)}).encode("utf-8")
                client_socket.sendall(struct.pack(">I", len(error_response)))
                client_socket.sendall(error_response)
            except Exception:
                pass

        finally:
            client_socket.close()

    def _handle_local_client(self, client_socket: socket.socket):
        """Handle a connection from a local process (via Unix socket)."""
        try:
            # Read message length (4 bytes, big-endian)
            length_data = client_socket.recv(4)
            if not length_data:
                return

            msg_length = struct.unpack(">I", length_data)[0]

            # Read message
            data = b""
            while len(data) < msg_length:
                chunk = client_socket.recv(min(msg_length - len(data), 4096))
                if not chunk:
                    break
                data += chunk

            # Parse JSON message
            message = json.loads(data.decode("utf-8"))
            command = message.get("command")

            # Dispatch command
            if command == "register":
                response = self._handle_register(message)
            elif command == "unregister":
                response = self._handle_unregister(message)
            elif command == "serve_broadcast":
                response = self._handle_serve_broadcast(message)
            elif command == "receive_broadcast":
                # Local getter process requesting to receive data
                response = self._handle_receive_broadcast(message)
            elif command == "list_keys":
                response = self._handle_list_keys(message)
            elif command == "execute_broadcast_group":
                response = self._handle_execute_broadcast_group(message)
            elif command == "put_tensor":
                # High-level: register + MDS publish
                response = self._handle_put_tensor(message)
            elif command == "get_tensor":
                # High-level: MDS lookup + NCCL receive
                response = self._handle_get_tensor(message)
            elif command == "put_tensors_broadcast":
                # Batch: register multiple tensors + join broadcast as putter
                response = self._handle_put_tensors_broadcast(message)
            elif command == "get_tensors_broadcast":
                # Batch: join broadcast as getter for multiple tensors
                response = self._handle_get_tensors_broadcast(message)
            elif command == "ping":
                response = {"status": "ok", "pid": os.getpid(), "tcp_port": self.tcp_port}
            elif command == "fs_broadcast_complete":
                # Local client notifying that a filesystem broadcast download is complete
                response = self._handle_fs_broadcast_complete(message)
            elif command == "fs_broadcast_get_path":
                # Local client requesting path (for testing; normally via TCP from child)
                response = self._handle_fs_broadcast_get_path(message)
            else:
                response = {"status": "error", "error": f"Unknown command: {command}"}

            # Send response
            response_data = json.dumps(response).encode("utf-8")
            client_socket.sendall(struct.pack(">I", len(response_data)))
            client_socket.sendall(response_data)

        except Exception as e:
            logger.error(f"Error handling client: {e}")
            try:
                error_response = json.dumps({"status": "error", "error": str(e)}).encode("utf-8")
                client_socket.sendall(struct.pack(">I", len(error_response)))
                client_socket.sendall(error_response)
            except Exception:
                pass

        finally:
            client_socket.close()

    def _handle_register(self, message: dict) -> dict:
        """Handle tensor registration from a local process."""
        key = message["key"]
        ipc_handle = _deserialize_ipc_handle(message["ipc_handle"])
        shape = tuple(message["shape"])
        dtype = message["dtype"]
        device = message["device"]
        pid = message["pid"]

        with self._lock:
            # Check if key already registered by same PID (update) or different PID (error)
            if key in self._registered:
                existing = self._registered[key]
                if existing.pid != pid:
                    return {
                        "status": "error",
                        "error": f"Key '{key}' already registered by PID {existing.pid}",
                    }

            # Register tensor
            self._registered[key] = RegisteredTensor(
                key=key,
                ipc_handle=ipc_handle,
                shape=shape,
                dtype=dtype,
                device=device,
                pid=pid,
            )

            # Track PID -> keys mapping
            if pid not in self._pid_keys:
                self._pid_keys[pid] = []
            if key not in self._pid_keys[pid]:
                self._pid_keys[pid].append(key)

        logger.info(f"Registered tensor '{key}' from PID {pid}: shape={shape}, dtype={dtype}, device={device}")

        return {"status": "ok", "key": key}

    def _handle_unregister(self, message: dict) -> dict:
        """Handle tensor unregistration."""
        key = message["key"]
        pid = message.get("pid")

        with self._lock:
            if key not in self._registered:
                return {"status": "ok", "message": "Key not registered"}

            registered = self._registered[key]
            if pid and registered.pid != pid:
                return {
                    "status": "error",
                    "error": f"Key '{key}' owned by PID {registered.pid}, not {pid}",
                }

            # Remove registration
            del self._registered[key]

            # Update PID tracking
            owner_pid = registered.pid
            if owner_pid in self._pid_keys:
                if key in self._pid_keys[owner_pid]:
                    self._pid_keys[owner_pid].remove(key)
                if not self._pid_keys[owner_pid]:
                    del self._pid_keys[owner_pid]

        logger.info(f"Unregistered tensor '{key}'")

        return {"status": "ok", "key": key}

    def _handle_list_keys(self, message: dict) -> dict:
        """List registered keys."""
        with self._lock:
            keys = []
            for key, reg in self._registered.items():
                keys.append(
                    {
                        "key": key,
                        "shape": reg.shape,
                        "dtype": reg.dtype,
                        "device": reg.device,
                        "pid": reg.pid,
                        "registered_at": reg.registered_at,
                    }
                )

        return {"status": "ok", "keys": keys}

    def _handle_serve_broadcast(self, message: dict) -> dict:
        """
        Handle request to serve a broadcast to remote getters.

        This is called when the metadata server has coordinated a quorum
        and directed getters to this server.
        """
        key = message["key"]
        broadcast_id = message["broadcast_id"]
        getter_endpoints = message["getter_endpoints"]  # List of {"ip": ..., "port": ...}
        nccl_port = message.get("nccl_port", self._get_next_nccl_port())

        with self._lock:
            if key not in self._registered:
                return {"status": "error", "error": f"Key '{key}' not registered"}

            registered = self._registered[key]

        # Reconstruct tensor from IPC handle
        try:
            tensor = _reconstruct_tensor_from_ipc(
                ipc_handle=registered.ipc_handle,
                shape=registered.shape,
                dtype_str=registered.dtype,
                device=registered.device,
            )
        except Exception as e:
            logger.error(f"Failed to reconstruct tensor for '{key}': {e}")
            return {
                "status": "error",
                "error": f"Failed to reconstruct tensor (original process may have freed it): {e}",
            }

        # Perform NCCL broadcast
        try:
            world_size = len(getter_endpoints) + 1  # +1 for this server (rank 0)
            self._perform_broadcast(
                tensor=tensor,
                broadcast_id=broadcast_id,
                nccl_port=nccl_port,
                getter_endpoints=getter_endpoints,
                world_size=world_size,
            )
            return {"status": "ok", "broadcast_id": broadcast_id}
        except Exception as e:
            logger.error(f"Broadcast failed for '{key}': {e}")
            return {"status": "error", "error": str(e)}

    def _perform_broadcast(
        self,
        tensor,
        broadcast_id: str,
        nccl_port: int,
        getter_endpoints: List[dict],
        world_size: int,
    ):
        """
        Perform NCCL broadcast as rank 0 (source).

        The getter GPU servers will connect as ranks 1..N.
        """
        from datetime import timedelta

        dist = _get_torch_distributed()

        pod_ip = os.getenv("POD_IP", "127.0.0.1")

        # Set up NCCL environment
        os.environ["MASTER_ADDR"] = pod_ip
        os.environ["MASTER_PORT"] = str(nccl_port)

        logger.info(
            f"Starting broadcast {broadcast_id}: world_size={world_size}, "
            f"MASTER_ADDR={pod_ip}, MASTER_PORT={nccl_port}"
        )

        process_group = None
        try:
            # Initialize process group as rank 0 with timeout
            if dist.is_initialized():
                # Create new group for this broadcast
                ranks = list(range(world_size))
                process_group = dist.new_group(ranks)
            else:
                dist.init_process_group(
                    backend="nccl",
                    rank=0,
                    world_size=world_size,
                    timeout=timedelta(seconds=KT_NCCL_TIMEOUT_SECONDS),
                )
                process_group = dist.group.WORLD

            # Broadcast the tensor
            dist.broadcast(tensor, src=0, group=process_group)

            logger.info(f"Broadcast {broadcast_id} complete")
            self._record_nccl_success()

        except Exception as e:
            logger.error(f"Broadcast {broadcast_id} failed: {e}")
            self._record_nccl_failure(str(e))
            raise

        finally:
            # Cleanup process group
            if process_group is not None:
                if process_group != dist.group.WORLD:
                    dist.destroy_process_group(process_group)
                else:
                    dist.destroy_process_group()

    def _handle_receive_broadcast(self, message: dict) -> dict:
        """
        Handle a local getter process requesting to receive GPU data.

        This server contacts the source's GPU server to initiate the broadcast.

        Args:
            message: {
                "key": storage key,
                "source_ip": IP of source pod,
                "source_gpu_port": TCP port of source GPU server,
                "dest_ipc_handle": IPC handle for destination tensor,
                "shape": tensor shape,
                "dtype": tensor dtype,
                "device": CUDA device index,
                "nccl_timeout": optional timeout override in seconds,
            }
        """
        key = message["key"]
        source_ip = message["source_ip"]
        source_gpu_port = message["source_gpu_port"]
        dest_ipc_handle = _deserialize_ipc_handle(message["dest_ipc_handle"])
        shape = tuple(message["shape"])
        dtype = message["dtype"]
        device = message["device"]
        nccl_timeout = message.get("nccl_timeout")  # Optional timeout override

        pod_ip = os.getenv("POD_IP", "127.0.0.1")

        logger.info(f"Receive broadcast request for '{key}' from {source_ip}:{source_gpu_port}")

        # Contact source GPU server to request broadcast
        try:
            response = self._send_tcp_message(
                source_ip,
                source_gpu_port,
                {
                    "command": "request_broadcast",
                    "key": key,
                    "getter_ip": pod_ip,
                    "getter_port": self.tcp_port,
                    "shape": list(shape),
                    "dtype": dtype,
                    # Include version info for compatibility checking
                    "versions": self._versions,
                },
                timeout=30.0,
            )

            if response.get("status") != "ok":
                return {"status": "error", "error": response.get("error", "Unknown error from source")}

            # Source will initiate NCCL - we need to join
            master_addr = response["master_addr"]
            master_port = response["master_port"]
            rank = response["rank"]
            world_size = response["world_size"]
            broadcast_id = response["broadcast_id"]

            # Reconstruct destination tensor from IPC handle
            dest_tensor = _reconstruct_tensor_from_ipc(
                ipc_handle=dest_ipc_handle,
                shape=shape,
                dtype_str=dtype,
                device=device,
            )

            # Join NCCL broadcast as receiver
            self._join_nccl_broadcast(
                dest_tensor=dest_tensor,
                broadcast_id=broadcast_id,
                master_addr=master_addr,
                master_port=master_port,
                rank=rank,
                world_size=world_size,
                nccl_timeout=nccl_timeout,
            )

            return {"status": "ok", "broadcast_id": broadcast_id}

        except Exception as e:
            logger.error(f"Failed to receive broadcast for '{key}': {e}")
            return {"status": "error", "error": str(e)}

    def _handle_remote_broadcast_request(self, message: dict) -> dict:
        """
        Handle a remote getter requesting data from this source server.
        Supports single key or list of keys for batch transfer.

        This server has the data registered locally. We set up NCCL and tell
        the getter to join.

        Args:
            message: {
                "keys": list of storage keys (or "key" for single),
                "getter_ip": IP of getter,
                "getter_port": TCP port of getter's GPU server,
                "shapes": list of expected tensor shapes (or "shape" for single),
                "dtypes": list of expected tensor dtypes (or "dtype" for single),
                "versions": NCCL/CUDA version info from getter,
            }
        """
        # Support both single key and list of keys
        keys = message.get("keys") or [message["key"]]
        # getter_ip and getter_port available for future multi-getter support
        _ = message["getter_ip"], message["getter_port"]

        # Check NCCL/CUDA version compatibility before attempting transfer
        remote_versions = message.get("versions", {})
        if remote_versions:
            is_compatible, error_msg = _check_version_compatibility(self._versions, remote_versions)
            if not is_compatible:
                logger.error(f"Version incompatibility with getter: {error_msg}")
                return {"status": "error", "error": error_msg}

        # Reconstruct all tensors from IPC handles
        tensors = []
        for key in keys:
            with self._lock:
                if key not in self._registered:
                    return {"status": "error", "error": f"Key '{key}' not registered on this server"}
                registered = self._registered[key]

            try:
                tensor = _reconstruct_tensor_from_ipc(
                    ipc_handle=registered.ipc_handle,
                    shape=registered.shape,
                    dtype_str=registered.dtype,
                    device=registered.device,
                )
                tensors.append(tensor)
            except Exception as e:
                logger.error(f"Failed to reconstruct tensor for '{key}': {e}")
                return {
                    "status": "error",
                    "error": f"Failed to reconstruct tensor '{key}' (original process may have freed it): {e}",
                }

        # Set up NCCL broadcast
        import uuid

        broadcast_id = str(uuid.uuid4())[:8]
        nccl_port = self._get_next_nccl_port()
        pod_ip = os.getenv("POD_IP", "127.0.0.1")
        world_size = 2  # Source + 1 getter (TODO: support multiple getters)

        # Start broadcast in background thread
        broadcast_thread = threading.Thread(
            target=self._serve_nccl_broadcast,
            args=(tensors, broadcast_id, nccl_port, world_size),
            daemon=True,
        )
        broadcast_thread.start()

        logger.info(f"Starting broadcast {broadcast_id}: {len(tensors)} tensor(s) to getter")

        # Return connection info to getter
        return {
            "status": "ok",
            "broadcast_id": broadcast_id,
            "master_addr": pod_ip,
            "master_port": nccl_port,
            "rank": 1,  # Getter is rank 1
            "world_size": world_size,
        }

    def _handle_join_broadcast(self, message: dict) -> dict:
        """
        Handle source server telling this getter to join NCCL broadcast.

        This is used for multi-getter broadcasts where the source coordinates.
        """
        # TODO: Implement for multi-getter support
        return {"status": "error", "error": "Multi-getter broadcast not yet implemented"}

    def _handle_execute_broadcast_group(self, message: dict) -> dict:
        """
        Execute a coordinated broadcast group transfer.

        This is called by the coordinator for a pod to execute ALL transfers
        for that pod. The GPU server:
        1. Performs local transfers first (same-node optimization)
        2. Joins NCCL process group and performs remote broadcasts

        For sends: Looks up source tensors from the registry by key
        For receives: Reconstructs destination tensors from IPC handles
        For local transfers: Direct GPU copy from source to destination
        """
        group_id = message["group_id"]
        rank = message["rank"]
        world_size = message["world_size"]
        master_addr = message["master_addr"]
        master_port = message["master_port"]
        sends = message.get("sends", [])
        receives = message.get("receives", [])
        local_transfers = message.get("local_transfers", [])

        dist = _get_torch_distributed()

        logger.info(
            f"Executing broadcast group {group_id}: rank={rank}/{world_size}, "
            f"master={master_addr}:{master_port}, sends={len(sends)}, "
            f"receives={len(receives)}, local={len(local_transfers)}"
        )

        try:
            # Step 1: Perform local transfers (same-node optimization)
            # These bypass NCCL entirely - direct GPU copy
            for lt in local_transfers:
                key = lt["key"]
                src_key = lt.get("src_tensor_key", key)
                dest_handle = lt.get("dest_ipc_handle")
                shape = tuple(lt.get("shape", []))
                dtype = lt.get("dtype", "torch.float32")

                # Look up source tensor from registry
                with self._lock:
                    if src_key not in self._registered:
                        return {"status": "error", "error": f"Source tensor '{src_key}' not registered"}
                    src_reg = self._registered[src_key]

                # Reconstruct source tensor from IPC
                src_tensor = _reconstruct_tensor_from_ipc(
                    ipc_handle=src_reg.ipc_handle,
                    shape=src_reg.shape,
                    dtype_str=src_reg.dtype,
                    device=src_reg.device,
                )

                # Reconstruct destination tensor from IPC handle
                if dest_handle is None:
                    return {"status": "error", "error": f"No dest_ipc_handle for local transfer '{key}'"}

                dest_ipc = _deserialize_ipc_handle(dest_handle)
                dest_tensor = _reconstruct_tensor_from_ipc(
                    ipc_handle=dest_ipc,
                    shape=shape,
                    dtype_str=dtype,
                    device=0,  # TODO: get from dest handle
                )

                # Direct GPU copy (handles cross-GPU case automatically)
                dest_tensor.copy_(src_tensor)
                logger.debug(f"Local transfer complete: {key}, shape={list(shape)}")

            # Step 2: If no remote transfers, we're done
            if not sends and not receives:
                logger.info(f"Broadcast group {group_id} complete (local only)")
                return {"status": "ok", "group_id": group_id, "rank": rank}

            # Step 3: Prepare tensors for NCCL transfers
            # For sends: look up from registry
            # For receives: reconstruct destination from IPC handle
            send_tensors: Dict[str, Any] = {}  # key -> tensor
            recv_tensors: Dict[str, Any] = {}  # key -> tensor

            for send in sends:
                key = send["key"]
                src_key = send.get("src_tensor_key", key)

                with self._lock:
                    if src_key not in self._registered:
                        return {"status": "error", "error": f"Source tensor '{src_key}' not registered for send"}
                    src_reg = self._registered[src_key]

                send_tensors[key] = _reconstruct_tensor_from_ipc(
                    ipc_handle=src_reg.ipc_handle,
                    shape=src_reg.shape,
                    dtype_str=src_reg.dtype,
                    device=src_reg.device,
                )

            for recv in receives:
                key = recv["key"]
                dest_handle = recv.get("dest_ipc_handle")
                shape = tuple(recv.get("shape", []))
                dtype = recv.get("dtype", "torch.float32")

                if dest_handle is None:
                    return {"status": "error", "error": f"No dest_ipc_handle for receive '{key}'"}

                dest_ipc = _deserialize_ipc_handle(dest_handle)
                recv_tensors[key] = _reconstruct_tensor_from_ipc(
                    ipc_handle=dest_ipc,
                    shape=shape,
                    dtype_str=dtype,
                    device=0,  # TODO: get from dest handle
                )

            # Step 4: Set up NCCL and perform broadcasts
            from datetime import timedelta

            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = str(master_port)

            process_group = None
            try:
                if dist.is_initialized():
                    dist.destroy_process_group()

                dist.init_process_group(
                    backend="nccl",
                    rank=rank,
                    world_size=world_size,
                    timeout=timedelta(seconds=KT_NCCL_TIMEOUT_SECONDS),
                )
                process_group = dist.group.WORLD

                # Build unified transfer list ordered by (src_rank, key)
                # All ranks must execute broadcasts in the same order
                all_broadcasts = []

                for send in sends:
                    key = send["key"]
                    all_broadcasts.append(
                        {
                            "src_rank": rank,
                            "key": key,
                            "tensor": send_tensors[key],
                        }
                    )

                for recv in receives:
                    key = recv["key"]
                    all_broadcasts.append(
                        {
                            "src_rank": recv["from_rank"],
                            "key": key,
                            "tensor": recv_tensors[key],
                        }
                    )

                # Sort for deterministic order across all participants
                all_broadcasts.sort(key=lambda x: (x["src_rank"], x["key"]))

                # Execute broadcasts
                for bc in all_broadcasts:
                    src_rank = bc["src_rank"]
                    key = bc["key"]
                    tensor = bc["tensor"]

                    logger.debug(f"Broadcast {key}: src={src_rank}, shape={list(tensor.shape)}")
                    dist.broadcast(tensor, src=src_rank, group=process_group)

                logger.info(f"Broadcast group {group_id} complete")
                self._record_nccl_success()
                return {"status": "ok", "group_id": group_id, "rank": rank}

            finally:
                if process_group is not None:
                    try:
                        dist.destroy_process_group()
                    except Exception as e:
                        logger.warning(f"Failed to destroy process group: {e}")

        except Exception as e:
            logger.error(f"Broadcast group {group_id} failed: {e}")
            self._record_nccl_failure(str(e))
            import traceback

            logger.error(traceback.format_exc())
            return {"status": "error", "error": str(e)}

    def _serve_nccl_broadcast(
        self,
        tensors: Union[Any, List[Any]],
        broadcast_id: str,
        nccl_port: int,
        world_size: int,
    ):
        """
        Serve NCCL broadcast as rank 0 (source) in a background thread.
        Supports single tensor or list of tensors.

        Args:
            tensors: Single tensor or list of tensors to broadcast
        """
        from datetime import timedelta

        dist = _get_torch_distributed()
        pod_ip = os.getenv("POD_IP", "127.0.0.1")

        # Normalize to list
        if not isinstance(tensors, list):
            tensors = [tensors]

        # Set up NCCL environment
        os.environ["MASTER_ADDR"] = pod_ip
        os.environ["MASTER_PORT"] = str(nccl_port)

        logger.info(
            f"Serving NCCL broadcast {broadcast_id}: world_size={world_size}, "
            f"tensors={len(tensors)}, MASTER_ADDR={pod_ip}, MASTER_PORT={nccl_port}"
        )

        process_group = None
        try:
            # Initialize process group as rank 0 with timeout
            if dist.is_initialized():
                ranks = list(range(world_size))
                process_group = dist.new_group(ranks)
            else:
                dist.init_process_group(
                    backend="nccl",
                    rank=0,
                    world_size=world_size,
                    timeout=timedelta(seconds=KT_NCCL_TIMEOUT_SECONDS),
                )
                process_group = dist.group.WORLD

            # Broadcast all tensors in same NCCL session
            for tensor in tensors:
                dist.broadcast(tensor, src=0, group=process_group)

            logger.info(f"NCCL broadcast {broadcast_id} complete: {len(tensors)} tensor(s)")
            self._record_nccl_success()

        except Exception as e:
            logger.error(f"NCCL broadcast {broadcast_id} failed: {e}")
            self._record_nccl_failure(str(e))

        finally:
            if process_group is not None:
                try:
                    if process_group != dist.group.WORLD:
                        dist.destroy_process_group(process_group)
                    else:
                        dist.destroy_process_group()
                except Exception:
                    pass

    def _join_nccl_broadcast(
        self,
        dest_tensors: Union[Any, List[Any]],
        broadcast_id: str,
        master_addr: str,
        master_port: int,
        rank: int,
        world_size: int,
        nccl_timeout: Optional[int] = None,
    ):
        """
        Join NCCL broadcast as a receiver (rank > 0).
        Supports single tensor or list of tensors.

        Args:
            dest_tensors: Single tensor or list of tensors to receive into
            nccl_timeout: Optional timeout override in seconds (for testing)
        """
        from datetime import timedelta

        dist = _get_torch_distributed()

        # Normalize to list
        if not isinstance(dest_tensors, list):
            dest_tensors = [dest_tensors]

        # Use override timeout if provided, otherwise use global setting
        timeout_seconds = nccl_timeout if nccl_timeout is not None else KT_NCCL_TIMEOUT_SECONDS

        # Set up NCCL environment
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)

        logger.info(
            f"Joining NCCL broadcast {broadcast_id}: rank={rank}, world_size={world_size}, "
            f"tensors={len(dest_tensors)}, MASTER_ADDR={master_addr}, MASTER_PORT={master_port}, timeout={timeout_seconds}s"
        )

        process_group = None
        try:
            # Initialize process group with timeout
            if dist.is_initialized():
                ranks = list(range(world_size))
                process_group = dist.new_group(ranks)
            else:
                dist.init_process_group(
                    backend="nccl",
                    rank=rank,
                    world_size=world_size,
                    timeout=timedelta(seconds=timeout_seconds),
                )
                process_group = dist.group.WORLD

            # Receive broadcast into all destination tensors
            for tensor in dest_tensors:
                dist.broadcast(tensor, src=0, group=process_group)

            logger.info(f"NCCL broadcast {broadcast_id} received {len(dest_tensors)} tensor(s) successfully")
            self._record_nccl_success()

        except Exception as e:
            logger.error(f"NCCL broadcast {broadcast_id} join failed: {e}")
            self._record_nccl_failure(str(e))
            raise

        finally:
            if process_group is not None:
                try:
                    if process_group != dist.group.WORLD:
                        dist.destroy_process_group(process_group)
                    else:
                        dist.destroy_process_group()
                except Exception:
                    pass

    def _send_tcp_message(self, host: str, port: int, message: dict, timeout: float = 30.0) -> dict:
        """Send a message to a remote GPU server via TCP."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)

        try:
            sock.connect((host, port))

            # Send message
            data = json.dumps(message).encode("utf-8")
            sock.sendall(struct.pack(">I", len(data)))
            sock.sendall(data)

            # Receive response length
            length_data = sock.recv(4)
            if not length_data:
                raise RuntimeError("Remote server closed connection")

            msg_length = struct.unpack(">I", length_data)[0]

            # Receive response
            data = b""
            while len(data) < msg_length:
                chunk = sock.recv(min(msg_length - len(data), 4096))
                if not chunk:
                    break
                data += chunk

            return json.loads(data.decode("utf-8"))

        finally:
            sock.close()

    def _get_next_nccl_port(self) -> int:
        """Get next available NCCL port."""
        port = self._next_nccl_port
        self._next_nccl_port += 1
        if self._next_nccl_port > self.nccl_port_end:
            self._next_nccl_port = self.nccl_port_start
        return port

    def _monitor_pids(self):
        """Monitor registered PIDs and clean up when they die."""
        while self._running:
            time.sleep(5.0)  # Check every 5 seconds

            with self._lock:
                dead_pids = []
                for pid in self._pid_keys.keys():
                    if not self._is_process_alive(pid):
                        dead_pids.append(pid)

                for pid in dead_pids:
                    keys = self._pid_keys.pop(pid, [])
                    for key in keys:
                        if key in self._registered:
                            del self._registered[key]
                            logger.info(f"Cleaned up tensor '{key}' (PID {pid} died)")

    def _is_process_alive(self, pid: int) -> bool:
        """Check if a process is still alive."""
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    # ==================== MDS Client Methods ====================

    def _get_mds_base_url(self) -> str:
        """Get or initialize the MDS base URL."""
        if self._mds_base_url is None:
            if not self._namespace:
                raise RuntimeError("KT_NAMESPACE environment variable not set")
            from kubetorch.serving.constants import DATA_STORE_METADATA_PORT

            service_name = "kubetorch-data-store"
            self._mds_base_url = f"http://{service_name}.{self._namespace}.svc.cluster.local:{DATA_STORE_METADATA_PORT}"
        return self._mds_base_url

    def _mds_publish_gpu(self, key: str) -> bool:
        """Publish GPU data key to metadata server."""
        from urllib.parse import quote

        import requests

        if not self._pod_ip or not self._pod_name or not self._namespace:
            logger.error("Missing POD_IP, POD_NAME, or KT_NAMESPACE environment variables")
            return False

        encoded_key = quote(key, safe="")
        url = f"{self._get_mds_base_url()}/api/v1/keys/{encoded_key}/gpu/publish"

        try:
            payload = {
                "ip": self._pod_ip,
                "pod_name": self._pod_name,
                "namespace": self._namespace,
                "nccl_port": self.nccl_port_start,
                "gpu_server_port": self.tcp_port,
                "gpu_server_socket": self.socket_path,
            }

            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to publish GPU key '{key}' to MDS: {e}")
            return False

    def _mds_get_gpu_source(self, key: str) -> Optional[dict]:
        """Get GPU data source info from metadata server."""
        from urllib.parse import quote

        import requests

        encoded_key = quote(key, safe="")
        url = f"{self._get_mds_base_url()}/api/v1/keys/{encoded_key}/gpu/source"

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

    # ==================== Broadcast WebSocket Support ====================

    def _join_broadcast_via_websocket(
        self,
        tensors: List[dict],  # List of {"key": str, "shape": list, "dtype": str, "dest_ipc_handle": optional}
        role: str,  # "putter" or "getter"
        broadcast_group_id: str,
        broadcast_timeout: float,
        broadcast_world_size: Optional[int],
    ) -> dict:
        """
        Join a broadcast group via MDS WebSocket.

        Takes a list of tensors (can be single element for single tensor broadcasts).
        Each tensor dict has: key, shape, dtype, and optionally dest_ipc_handle (for getters).

        Each client request creates its own WebSocket connection to MDS. The MDS
        aggregates participants from the same pod and assigns one coordinator per pod.
        The coordinator gets the full manifest and executes NCCL. All participants
        wait for MDS "completed" signal before returning.
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self._join_broadcast_via_websocket_async(
                            tensors,
                            role,
                            broadcast_group_id,
                            broadcast_timeout,
                            broadcast_world_size,
                        ),
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self._join_broadcast_via_websocket_async(
                        tensors,
                        role,
                        broadcast_group_id,
                        broadcast_timeout,
                        broadcast_world_size,
                    )
                )
        except RuntimeError:
            return asyncio.run(
                self._join_broadcast_via_websocket_async(
                    tensors,
                    role,
                    broadcast_group_id,
                    broadcast_timeout,
                    broadcast_world_size,
                )
            )

    async def _join_broadcast_via_websocket_async(
        self,
        tensors: List[dict],
        role: str,
        broadcast_group_id: str,
        broadcast_timeout: float,
        broadcast_world_size: Optional[int],
    ) -> dict:
        """Async implementation of broadcast group join."""
        import asyncio

        try:
            from websockets.asyncio.client import connect as ws_connect
        except ImportError:
            from websockets import connect as ws_connect

        from urllib.parse import quote

        pod_ip = self._pod_ip or os.getenv("POD_IP")
        pod_name = self._pod_name or os.getenv("POD_NAME")

        if not pod_ip:
            return {"status": "error", "error": "POD_IP not set"}

        # Build WebSocket URL
        encoded_group_id = quote(broadcast_group_id, safe="")
        base_url = self._get_mds_base_url()
        if base_url.startswith("https://"):
            ws_url = f"wss://{base_url[8:]}/ws/broadcast/{encoded_group_id}"
        elif base_url.startswith("http://"):
            ws_url = f"ws://{base_url[7:]}/ws/broadcast/{encoded_group_id}"
        else:
            ws_url = f"ws://{base_url}/ws/broadcast/{encoded_group_id}"

        tensor_keys = [t["key"] for t in tensors]
        logger.info(
            f"Joining broadcast group '{broadcast_group_id}' as {role} " f"with {len(tensors)} tensor(s): {tensor_keys}"
        )

        try:
            async with ws_connect(ws_url) as websocket:
                # Build join message with tensors list (unified format)
                join_msg = {
                    "action": "join",
                    "role": role,
                    "pod_ip": pod_ip,
                    "pod_name": pod_name,
                    "timeout": broadcast_timeout,
                    "world_size": broadcast_world_size,
                    "tensors": [
                        {
                            "key": t["key"],
                            "shape": t.get("shape", []),
                            "dtype": t.get("dtype", "torch.float32"),
                            "dest_ipc_handle": t.get("dest_ipc_handle"),  # For getters
                        }
                        for t in tensors
                    ],
                }

                await websocket.send(json.dumps(join_msg))

                # Wait for queued confirmation
                response = json.loads(await asyncio.wait_for(websocket.recv(), timeout=30.0))
                if response.get("event") == "error":
                    return {"status": "error", "error": f"Failed to join: {response.get('message')}"}
                if response.get("event") != "queued":
                    return {"status": "error", "error": f"Unexpected response: {response}"}

                logger.info(f"Queued in broadcast group (position: {response.get('position')})")

                # Wait for ready notification
                ready_data = None
                start_time = asyncio.get_event_loop().time()

                while ready_data is None:
                    try:
                        msg = json.loads(await asyncio.wait_for(websocket.recv(), timeout=35.0))

                        if msg.get("event") == "ready":
                            ready_data = msg
                        elif msg.get("event") == "heartbeat":
                            await websocket.send(json.dumps({"action": "heartbeat"}))
                        elif msg.get("event") == "error":
                            return {"status": "error", "error": f"Broadcast error: {msg.get('message')}"}

                    except asyncio.TimeoutError:
                        elapsed = asyncio.get_event_loop().time() - start_time
                        if elapsed > broadcast_timeout:
                            return {
                                "status": "error",
                                "error": f"Timeout waiting for broadcast group ({broadcast_timeout}s)",
                            }
                        await websocket.send(json.dumps({"action": "heartbeat"}))

                # Extract transfer info - all participants get the full manifest
                rank = ready_data["rank"]
                world_size = ready_data["world_size"]
                master_addr = ready_data["master_addr"]
                master_port = ready_data["master_port"]
                sends = ready_data.get("sends", [])
                receives = ready_data.get("receives", [])
                local_transfers = ready_data.get("local_transfers", [])

                logger.info(
                    f"Broadcast group ready: rank={rank}, world_size={world_size}, "
                    f"master={master_addr}:{master_port}, "
                    f"sends={len(sends)}, receives={len(receives)}, local={len(local_transfers)}"
                )

                # Internal coordination: ensure NCCL runs exactly once per pod
                # First participant to reach here executes, others wait for result
                should_execute = False
                execution_state = None

                with self._broadcast_execution_lock:
                    if broadcast_group_id not in self._broadcast_execution:
                        # First participant - we execute NCCL
                        self._broadcast_execution[broadcast_group_id] = {
                            "executing": True,
                            "result": None,
                            "event": threading.Event(),
                        }
                        should_execute = True
                    execution_state = self._broadcast_execution[broadcast_group_id]

                if should_execute:
                    # Execute NCCL transfer
                    transfer_result = self._handle_execute_broadcast_group(
                        {
                            "group_id": broadcast_group_id,
                            "rank": rank,
                            "world_size": world_size,
                            "master_addr": master_addr,
                            "master_port": master_port,
                            "sends": sends,
                            "receives": receives,
                            "local_transfers": local_transfers,
                        }
                    )

                    # Store result and signal other waiters
                    with self._broadcast_execution_lock:
                        execution_state["result"] = transfer_result
                        execution_state["executing"] = False
                        execution_state["event"].set()
                else:
                    # Wait for the executing participant to finish
                    logger.info(f"Waiting for NCCL execution to complete for group '{broadcast_group_id}'")
                    execution_state["event"].wait(timeout=broadcast_timeout)

                # Get the result (same for all participants)
                transfer_result = execution_state["result"]
                transfer_success = transfer_result and transfer_result.get("status") == "ok"
                transfer_error = transfer_result.get("error") if transfer_result else "Execution failed"

                # Send completion to MDS
                await websocket.send(json.dumps({"action": "complete", "success": transfer_success}))

                # Wait for "completed" acknowledgment from MDS
                try:
                    while True:
                        completed_msg = json.loads(await asyncio.wait_for(websocket.recv(), timeout=broadcast_timeout))
                        if completed_msg.get("event") == "completed":
                            break
                        elif completed_msg.get("event") == "error":
                            return {"status": "error", "error": f"Broadcast failed: {completed_msg.get('message')}"}
                        elif completed_msg.get("event") == "heartbeat":
                            await websocket.send(json.dumps({"action": "heartbeat"}))
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for broadcast completion ack")

                # Clean up execution state
                with self._broadcast_execution_lock:
                    if broadcast_group_id in self._broadcast_execution:
                        del self._broadcast_execution[broadcast_group_id]

                if not transfer_success:
                    return {"status": "error", "error": f"Transfer failed: {transfer_error}"}

                logger.info(f"Broadcast group '{broadcast_group_id}' transfer complete")

                return {
                    "status": "ok",
                    "rank": rank,
                    "world_size": world_size,
                    "group_id": broadcast_group_id,
                }

        except Exception as e:
            logger.error(f"Broadcast WebSocket error: {e}")
            return {"status": "error", "error": str(e)}

    # ==================== High-Level Command Handlers ====================

    def _handle_put_tensor(self, message: dict) -> dict:
        """
        Handle put_tensor command - register tensor locally + publish to MDS.

        If broadcast config is provided, joins the broadcast group via WebSocket.
        Pod Data Server is always the coordinator for its pod.
        """
        key = message["key"]
        ipc_handle = _deserialize_ipc_handle(message["ipc_handle"])
        shape = tuple(message["shape"])
        dtype = message["dtype"]
        device = message["device"]
        pid = message.get("pid", 0)
        broadcast = message.get("broadcast")  # Optional broadcast config

        # Step 1: Register tensor locally
        with self._lock:
            self._registered[key] = RegisteredTensor(
                key=key,
                ipc_handle=ipc_handle,
                shape=shape,
                dtype=dtype,
                device=device,
                pid=pid,
            )

            # Track PID for cleanup
            if pid not in self._pid_keys:
                self._pid_keys[pid] = []
            if key not in self._pid_keys[pid]:
                self._pid_keys[pid].append(key)

        # Step 2: If broadcast, join via WebSocket; otherwise publish to MDS
        if broadcast:
            # Use unified tensors list format (single tensor as list of one)
            tensors = [
                {
                    "key": key,
                    "shape": list(shape),
                    "dtype": dtype,
                }
            ]
            return self._join_broadcast_via_websocket(
                tensors=tensors,
                role="putter",
                broadcast_group_id=broadcast["group_id"],
                broadcast_timeout=broadcast.get("timeout", 600.0),
                broadcast_world_size=broadcast.get("world_size"),
            )
        else:
            # Point-to-point: just publish to MDS
            if not self._mds_publish_gpu(key):
                return {"status": "error", "error": "Failed to publish to MDS"}

            logger.info(f"put_tensor: registered and published '{key}'")
            return {"status": "ok", "key": key}

    def _handle_get_tensor(self, message: dict) -> dict:
        """
        Handle get_tensor command - lookup from MDS + receive via NCCL.
        Supports batching multiple tensors, grouping by source for efficiency.

        If broadcast config is provided, joins the broadcast group via WebSocket.
        Pod Data Server is always the coordinator for its pod.
        """
        keys = message["keys"]
        dest_ipc_handles_serialized = message["dest_ipc_handles"]
        dest_ipc_handles = [_deserialize_ipc_handle(h) for h in dest_ipc_handles_serialized]
        shapes = [tuple(s) for s in message["shapes"]]
        dtypes = message["dtypes"]
        devices = message["devices"]
        nccl_timeout = message.get("nccl_timeout")
        broadcast = message.get("broadcast")  # Optional broadcast config

        # If broadcast, join via WebSocket with all tensors
        if broadcast:
            tensors = [
                {
                    "key": keys[i],
                    "shape": list(shapes[i]),
                    "dtype": dtypes[i],
                    "dest_ipc_handle": dest_ipc_handles_serialized[i],
                }
                for i in range(len(keys))
            ]
            return self._join_broadcast_via_websocket(
                tensors=tensors,
                role="getter",
                broadcast_group_id=broadcast["group_id"],
                broadcast_timeout=broadcast.get("timeout", 600.0),
                broadcast_world_size=broadcast.get("world_size"),
            )

        # Point-to-point: Lookup source for each key and group by source
        from typing import Tuple

        source_groups: Dict[Tuple[str, int], List] = {}
        missing_keys = []

        for i, key in enumerate(keys):
            source_info = self._mds_get_gpu_source(key)
            if source_info is None:
                missing_keys.append(key)
                continue

            source_ip = source_info["ip"]
            source_gpu_port = source_info.get("gpu_server_port", DEFAULT_TCP_PORT)
            source_key = (source_ip, source_gpu_port)

            if source_key not in source_groups:
                source_groups[source_key] = []
            source_groups[source_key].append(
                {
                    "index": i,
                    "key": key,
                    "shape": shapes[i],
                    "dtype": dtypes[i],
                    "device": devices[i],
                    "ipc_handle": dest_ipc_handles[i],
                }
            )

        if missing_keys:
            return {"status": "error", "error": f"No GPU source found for keys: {missing_keys}"}

        logger.info(f"get_tensor: {len(keys)} keys from {len(source_groups)} source(s)")

        # Process each source group with its own NCCL session
        try:
            timeout_seconds = nccl_timeout if nccl_timeout is not None else KT_NCCL_TIMEOUT_SECONDS

            for (source_ip, source_gpu_port), items in source_groups.items():
                group_keys = [item["key"] for item in items]
                group_shapes = [item["shape"] for item in items]
                group_dtypes = [item["dtype"] for item in items]

                # Reconstruct destination tensors for this group
                dest_tensors = [
                    _reconstruct_tensor_from_ipc(item["ipc_handle"], item["shape"], item["dtype"], item["device"])
                    for item in items
                ]

                broadcast_id = f"get-batch-{time.time()}"
                request_msg = {
                    "command": "request_broadcast",
                    "keys": group_keys,
                    "broadcast_id": broadcast_id,
                    "getter_ip": self._pod_ip or "127.0.0.1",
                    "getter_port": self.tcp_port,
                    "shapes": [list(s) for s in group_shapes],
                    "dtypes": group_dtypes,
                    "versions": self._versions,
                }

                response = self._send_tcp_message(source_ip, source_gpu_port, request_msg, timeout=30.0)

                if response.get("status") != "ok":
                    return {"status": "error", "error": f"Source {source_ip} rejected request: {response.get('error')}"}

                self._join_nccl_broadcast(
                    dest_tensors=dest_tensors,
                    broadcast_id=broadcast_id,
                    master_addr=response["master_addr"],
                    master_port=response["master_port"],
                    rank=response["rank"],
                    world_size=response["world_size"],
                    nccl_timeout=timeout_seconds,
                )

                logger.info(f"get_tensor: received {len(group_keys)} tensors from {source_ip}")

            return {"status": "ok", "keys": keys}

        except Exception as e:
            logger.error(f"get_tensor failed: {e}")
            return {"status": "error", "error": str(e)}

    # ==================== Batch Tensor Methods for State Dict ====================

    def _handle_put_tensors_broadcast(self, message: dict) -> dict:
        """
        Handle batch put_tensors_broadcast command.

        Registers multiple tensors and joins broadcast as putter.
        All tensors are transferred in a single NCCL session.
        """
        tensor_infos = message["tensors"]
        broadcast = message["broadcast"]
        pid = message.get("pid", 0)

        # Step 1: Register all tensors locally
        for info in tensor_infos:
            key = info["key"]
            ipc_handle = _deserialize_ipc_handle(info["ipc_handle"])
            shape = tuple(info["shape"])
            dtype = info["dtype"]
            device = info["device"]

            with self._lock:
                self._registered[key] = RegisteredTensor(
                    key=key,
                    ipc_handle=ipc_handle,
                    shape=shape,
                    dtype=dtype,
                    device=device,
                    pid=pid,
                )

                # Track PID for cleanup
                if pid not in self._pid_keys:
                    self._pid_keys[pid] = []
                if key not in self._pid_keys[pid]:
                    self._pid_keys[pid].append(key)

        logger.info(f"Registered {len(tensor_infos)} tensors for broadcast")

        # Step 2: Join broadcast via WebSocket with ALL tensors (unified method)
        tensors = [
            {
                "key": t["key"],
                "shape": t.get("shape", []),
                "dtype": t.get("dtype", "torch.float32"),
            }
            for t in tensor_infos
        ]
        return self._join_broadcast_via_websocket(
            tensors=tensors,
            role="putter",
            broadcast_group_id=broadcast["group_id"],
            broadcast_timeout=broadcast.get("timeout", 600.0),
            broadcast_world_size=broadcast.get("world_size"),
        )

    def _handle_get_tensors_broadcast(self, message: dict) -> dict:
        """
        Handle batch get_tensors_broadcast command.

        Joins broadcast as getter for multiple tensors.
        All tensors are received in a single NCCL session.
        """
        tensor_infos = message["tensors"]
        broadcast = message["broadcast"]

        logger.info(f"Joining broadcast to receive {len(tensor_infos)} tensors")

        # Join broadcast via WebSocket with ALL tensors (unified method)
        tensors = [
            {
                "key": t["key"],
                "shape": t.get("shape", []),
                "dtype": t.get("dtype", "torch.float32"),
                "dest_ipc_handle": t.get("dest_ipc_handle"),  # For getters
            }
            for t in tensor_infos
        ]
        return self._join_broadcast_via_websocket(
            tensors=tensors,
            role="getter",
            broadcast_group_id=broadcast["group_id"],
            broadcast_timeout=broadcast.get("timeout", 600.0),
            broadcast_world_size=broadcast.get("world_size"),
        )

    # ==================== Filesystem Broadcast Methods ====================

    def _cleanup_expired_fs_broadcasts(self):
        """Remove filesystem broadcasts older than TTL."""
        current_time = time.time()
        with self._fs_broadcast_lock:
            expired = [
                key
                for key, data in self._fs_broadcasts_completed.items()
                if current_time - data["completed_at"] > self._fs_broadcast_ttl
            ]
            for key in expired:
                del self._fs_broadcasts_completed[key]
                if key in self._fs_broadcast_events:
                    del self._fs_broadcast_events[key]

    def _handle_fs_broadcast_complete(self, message: dict) -> dict:
        """
        Handle notification that a filesystem broadcast download is complete.

        Called by the local client after successfully downloading data.
        Signals any waiting child getters that data is available.
        """
        group_id = message.get("group_id")
        key = message.get("key")
        local_path = message.get("local_path")

        if not all([group_id, key, local_path]):
            return {"status": "error", "error": "Missing required fields: group_id, key, local_path"}

        broadcast_key = (group_id, key)

        with self._fs_broadcast_lock:
            # Store completion info
            self._fs_broadcasts_completed[broadcast_key] = {
                "local_path": local_path,
                "completed_at": time.time(),
            }

            # Signal any waiting child getters
            if broadcast_key in self._fs_broadcast_events:
                self._fs_broadcast_events[broadcast_key].set()

            logger.info(f"Filesystem broadcast complete: {group_id}/{key} -> {local_path}")

        # Cleanup expired broadcasts periodically
        self._cleanup_expired_fs_broadcasts()

        return {"status": "ok"}

    def _handle_fs_broadcast_get_path(self, message: dict) -> dict:
        """
        Handle request from child getter for parent's local path.

        Blocks until parent has completed its download, then returns the local path.
        Called via TCP from child pod's client.
        """
        group_id = message.get("group_id")
        key = message.get("key")
        timeout = message.get("timeout", 60.0)

        if not all([group_id, key]):
            return {"status": "error", "error": "Missing required fields: group_id, key"}

        broadcast_key = (group_id, key)

        # Check if already completed
        with self._fs_broadcast_lock:
            if broadcast_key in self._fs_broadcasts_completed:
                data = self._fs_broadcasts_completed[broadcast_key]
                return {
                    "status": "ok",
                    "local_path": data["local_path"],
                    "pod_ip": self._pod_ip,
                }

            # Create event if not exists
            if broadcast_key not in self._fs_broadcast_events:
                self._fs_broadcast_events[broadcast_key] = threading.Event()
            event = self._fs_broadcast_events[broadcast_key]

        # Wait for completion
        logger.debug(f"Waiting for filesystem broadcast {group_id}/{key} to complete (timeout={timeout}s)")
        completed = event.wait(timeout=timeout)

        if not completed:
            return {"status": "error", "error": f"Timeout waiting for broadcast {group_id}/{key}"}

        # Get the path
        with self._fs_broadcast_lock:
            if broadcast_key in self._fs_broadcasts_completed:
                data = self._fs_broadcasts_completed[broadcast_key]
                return {
                    "status": "ok",
                    "local_path": data["local_path"],
                    "pod_ip": self._pod_ip,
                }

        return {"status": "error", "error": "Broadcast completed but path not found"}


class PodDataServerClient:
    """Client for communicating with the Pod Data Server."""

    def __init__(self, socket_path: str = DEFAULT_SOCKET_PATH):
        self.socket_path = socket_path

    def _send_message(self, message: dict, timeout: float = 30.0) -> dict:
        """Send a message to the server and return the response."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)

        try:
            sock.connect(self.socket_path)

            # Send message
            data = json.dumps(message).encode("utf-8")
            sock.sendall(struct.pack(">I", len(data)))
            sock.sendall(data)

            # Receive response length
            length_data = sock.recv(4)
            if not length_data:
                raise RuntimeError("Server closed connection")

            msg_length = struct.unpack(">I", length_data)[0]

            # Receive response
            data = b""
            while len(data) < msg_length:
                chunk = sock.recv(min(msg_length - len(data), 4096))
                if not chunk:
                    break
                data += chunk

            return json.loads(data.decode("utf-8"))

        finally:
            sock.close()

    def ping(self) -> dict:
        """Ping the server to check if it's alive."""
        return self._send_message({"command": "ping"})

    def register_tensor(
        self,
        key: str,
        tensor,
        pid: Optional[int] = None,
    ) -> dict:
        """
        Register a CUDA tensor with the server.

        Args:
            key: Storage key
            tensor: CUDA tensor to register
            pid: PID of the registering process (defaults to current)

        Returns:
            Server response dict
        """
        _get_torch()  # Ensure torch is available

        if not tensor.is_cuda:
            raise ValueError("Tensor must be on a CUDA device")

        # Get IPC handle (version-compatible)
        ipc_handle = _get_ipc_handle(tensor)

        # Convert IPC handle to JSON-serializable format
        # IPC handle contains bytes objects that need base64 encoding
        serializable_handle = _serialize_ipc_handle(ipc_handle)

        message = {
            "command": "register",
            "key": key,
            "ipc_handle": serializable_handle,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": tensor.device.index,
            "pid": pid or os.getpid(),
        }

        return self._send_message(message)

    def unregister_tensor(self, key: str, pid: Optional[int] = None) -> dict:
        """Unregister a tensor from the server."""
        message = {
            "command": "unregister",
            "key": key,
            "pid": pid or os.getpid(),
        }
        return self._send_message(message)

    def list_keys(self) -> dict:
        """List all registered keys."""
        return self._send_message({"command": "list_keys"})

    def request_broadcast(
        self,
        key: str,
        broadcast_id: str,
        getter_endpoints: List[dict],
        nccl_port: Optional[int] = None,
    ) -> dict:
        """
        Request the server to perform a broadcast.

        Args:
            key: Storage key
            broadcast_id: Unique ID for this broadcast
            getter_endpoints: List of {"ip": ..., "port": ...} for getters
            nccl_port: NCCL port to use (optional)

        Returns:
            Server response dict
        """
        message = {
            "command": "serve_broadcast",
            "key": key,
            "broadcast_id": broadcast_id,
            "getter_endpoints": getter_endpoints,
        }
        if nccl_port:
            message["nccl_port"] = nccl_port

        return self._send_message(message, timeout=60.0)

    def receive_broadcast(
        self,
        key: str,
        source_ip: str,
        source_gpu_port: int,
        dest_tensor,
        nccl_timeout: Optional[int] = None,
    ) -> dict:
        """
        Request to receive a broadcast from a remote source.

        This tells the local GPU server to contact the source's GPU server
        and participate in the NCCL broadcast.

        Args:
            key: Storage key
            source_ip: IP of the source pod
            source_gpu_port: TCP port of the source's GPU server
            dest_tensor: Pre-allocated destination tensor to receive into
            nccl_timeout: Override NCCL timeout in seconds (for testing)

        Returns:
            Server response dict
        """
        _get_torch()  # Ensure torch is available

        if not dest_tensor.is_cuda:
            raise ValueError("Destination tensor must be on a CUDA device")

        # Get IPC handle for destination tensor (version-compatible)
        ipc_handle = _get_ipc_handle(dest_tensor)

        # Serialize IPC handle for JSON transmission
        serializable_handle = _serialize_ipc_handle(ipc_handle)

        message = {
            "command": "receive_broadcast",
            "key": key,
            "source_ip": source_ip,
            "source_gpu_port": source_gpu_port,
            "dest_ipc_handle": serializable_handle,
            "shape": list(dest_tensor.shape),
            "dtype": str(dest_tensor.dtype),
            "device": dest_tensor.device.index,
        }

        if nccl_timeout is not None:
            message["nccl_timeout"] = nccl_timeout

        # Use longer client timeout to account for NCCL timeout
        client_timeout = max(120.0, (nccl_timeout or 60) + 30)
        return self._send_message(message, timeout=client_timeout)

    def execute_broadcast_group(
        self,
        group_id: str,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
        sends: List[dict],
        receives: List[dict],
        local_transfers: Optional[List[dict]] = None,
    ) -> dict:
        """
        Execute a coordinated broadcast group transfer.

        This is called by the coordinator for a pod to execute ALL transfers
        for that pod via the shared GPU data server.

        The GPU server:
        1. Looks up source tensors for sends from its registry (by key)
        2. Reconstructs destination tensors for receives from IPC handles
        3. Performs local transfers (same-node) via direct GPU copy
        4. Joins NCCL process group and performs remote broadcasts

        Args:
            group_id: Unique identifier for this broadcast group
            rank: This pod's rank in the NCCL process group
            world_size: Total number of pods (unique ranks)
            master_addr: IP address of rank 0 (NCCL master)
            master_port: NCCL port for this group
            sends: List of tensors to broadcast:
                [{"key": str, "src_tensor_key": str, "to_ranks": [int], "shape": [...], "dtype": str}]
                The src_tensor_key is used to look up the tensor from the GPU server's registry
            receives: List of tensors to receive:
                [{"key": str, "from_rank": int, "dest_ipc_handle": [...], "shape": [...], "dtype": str}]
                The dest_ipc_handle is the serialized IPC handle of the destination tensor
            local_transfers: List of same-node transfers (optimization):
                [{"key": str, "src_tensor_key": str, "dest_ipc_handle": [...], "shape": [...], "dtype": str}]
                These bypass NCCL - direct GPU copy from source to destination

        Returns:
            Server response dict
        """
        message = {
            "command": "execute_broadcast_group",
            "group_id": group_id,
            "rank": rank,
            "world_size": world_size,
            "master_addr": master_addr,
            "master_port": master_port,
            "sends": sends,
            "receives": receives,
            "local_transfers": local_transfers or [],
        }

        # Longer timeout for broadcast operations
        return self._send_message(message, timeout=300.0)

    # ==================== High-Level API Methods ====================

    def put_tensor(
        self,
        key: str,
        tensor,
        pid: Optional[int] = None,
        broadcast: Optional[dict] = None,
    ) -> dict:
        """
        High-level put: Register tensor + publish to MDS (or join broadcast).

        Pod Data Server handles everything - just pass the tensor.

        Args:
            key: Storage key
            tensor: CUDA tensor to publish
            pid: PID of the registering process (defaults to current)
            broadcast: Optional broadcast config dict with:
                - group_id: Broadcast group identifier (required)
                - timeout: Timeout for quorum formation (default 600s)
                - world_size: Expected number of participants (optional)

        Returns:
            Server response dict
        """
        _get_torch()  # Ensure torch is available

        if not tensor.is_cuda:
            raise ValueError("Tensor must be on a CUDA device")

        # Get IPC handle
        ipc_handle = _get_ipc_handle(tensor)
        serializable_handle = _serialize_ipc_handle(ipc_handle)

        message = {
            "command": "put_tensor",
            "key": key,
            "ipc_handle": serializable_handle,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": tensor.device.index,
            "pid": pid or os.getpid(),
        }
        if broadcast:
            message["broadcast"] = broadcast

        # Longer timeout for broadcast operations
        timeout = 300.0 if broadcast else 30.0
        return self._send_message(message, timeout=timeout)

    def get_tensor(
        self,
        keys: Union[str, List[str]],
        dest_tensors: Union[Any, List[Any]],
        nccl_timeout: Optional[int] = None,
        broadcast: Optional[dict] = None,
    ) -> dict:
        """
        High-level get: MDS lookup + NCCL receive (or join broadcast).
        Supports batching multiple tensors in a single NCCL session.

        Pod Data Server handles everything - just pass the destination tensor(s).

        Args:
            keys: Storage key(s) - single string or list
            dest_tensors: Pre-allocated CUDA tensor(s) to receive into
            nccl_timeout: Override NCCL timeout in seconds (for testing)
            broadcast: Optional broadcast config dict with:
                - group_id: Broadcast group identifier (required)
                - timeout: Timeout for quorum formation (default 600s)
                - world_size: Expected number of participants (optional)

        Returns:
            Server response dict
        """
        _get_torch()  # Ensure torch is available

        # Normalize to lists
        if isinstance(keys, str):
            keys = [keys]
            dest_tensors = [dest_tensors]
        elif not isinstance(dest_tensors, list):
            dest_tensors = [dest_tensors]

        if len(keys) != len(dest_tensors):
            raise ValueError(f"Number of keys ({len(keys)}) must match dest_tensors ({len(dest_tensors)})")

        # Validate and build batch data
        dest_ipc_handles = []
        shapes = []
        dtypes = []
        devices = []

        for i, tensor in enumerate(dest_tensors):
            if not tensor.is_cuda:
                raise ValueError(f"Destination tensor at index {i} must be on a CUDA device")
            dest_ipc_handles.append(_serialize_ipc_handle(_get_ipc_handle(tensor)))
            shapes.append(list(tensor.shape))
            dtypes.append(str(tensor.dtype))
            devices.append(tensor.device.index)

        message = {
            "command": "get_tensor",
            "keys": keys,
            "dest_ipc_handles": dest_ipc_handles,
            "shapes": shapes,
            "dtypes": dtypes,
            "devices": devices,
        }

        if nccl_timeout is not None:
            message["nccl_timeout"] = nccl_timeout
        if broadcast:
            message["broadcast"] = broadcast

        # Longer timeout for broadcast or batch operations
        if broadcast:
            client_timeout = broadcast.get("timeout", 600.0) + 60.0
        else:
            client_timeout = max(120.0, (nccl_timeout or 60) + 30)
        return self._send_message(message, timeout=client_timeout)

    # ==================== Batch Tensor Methods for State Dict ====================

    def put_tensors_broadcast(
        self,
        keys: List[str],
        tensors: List,
        broadcast: dict,
        pid: Optional[int] = None,
    ) -> dict:
        """
        Batch put: Register multiple tensors and join broadcast as putter.

        All tensors are transferred in a single NCCL session for efficiency.
        Used for state_dict broadcasts where multiple tensors need to be sent together.

        Args:
            keys: List of storage keys (one per tensor)
            tensors: List of CUDA tensors to publish
            broadcast: Broadcast config dict with:
                - group_id: Broadcast group identifier (required)
                - timeout: Timeout for quorum formation (default 600s)
                - world_size: Expected number of participants (optional)
            pid: PID of the registering process (defaults to current)

        Returns:
            Server response dict with broadcast results
        """
        _get_torch()

        if len(keys) != len(tensors):
            raise ValueError(f"keys and tensors must have same length: {len(keys)} vs {len(tensors)}")

        # Build tensor info list
        tensor_infos = []
        for key, tensor in zip(keys, tensors):
            if not tensor.is_cuda:
                raise ValueError(f"Tensor for key '{key}' must be on a CUDA device")

            ipc_handle = _get_ipc_handle(tensor)
            tensor_infos.append(
                {
                    "key": key,
                    "ipc_handle": _serialize_ipc_handle(ipc_handle),
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "device": tensor.device.index,
                }
            )

        message = {
            "command": "put_tensors_broadcast",
            "tensors": tensor_infos,
            "broadcast": broadcast,
            "pid": pid or os.getpid(),
        }

        # Longer timeout for broadcast operations
        timeout = broadcast.get("timeout", 600.0) + 60.0
        return self._send_message(message, timeout=timeout)

    def get_tensors_broadcast(
        self,
        tensors: List[tuple],  # List of (key, dest_tensor)
        broadcast: dict,
    ) -> dict:
        """
        Batch get: Join broadcast as getter for multiple tensors.

        All tensors are received in a single NCCL session for efficiency.
        Used for state_dict broadcasts where multiple tensors need to be received together.

        Args:
            tensors: List of (key, dest_tensor) tuples
            broadcast: Broadcast config dict with:
                - group_id: Broadcast group identifier (required)
                - timeout: Timeout for quorum formation (default 600s)
                - world_size: Expected number of participants (optional)

        Returns:
            Server response dict with broadcast results
        """
        _get_torch()

        # Build tensor info list
        tensor_infos = []
        for key, dest_tensor in tensors:
            if not dest_tensor.is_cuda:
                raise ValueError(f"Destination tensor for key '{key}' must be on a CUDA device")

            ipc_handle = _get_ipc_handle(dest_tensor)
            tensor_infos.append(
                {
                    "key": key,
                    "dest_ipc_handle": _serialize_ipc_handle(ipc_handle),
                    "shape": list(dest_tensor.shape),
                    "dtype": str(dest_tensor.dtype),
                    "device": dest_tensor.device.index,
                }
            )

        message = {
            "command": "get_tensors_broadcast",
            "tensors": tensor_infos,
            "broadcast": broadcast,
        }

        # Longer timeout for broadcast operations
        timeout = broadcast.get("timeout", 600.0) + 60.0
        return self._send_message(message, timeout=timeout)

    # ==================== Filesystem Broadcast Methods ====================

    def fs_broadcast_complete(self, group_id: str, key: str, local_path: str) -> dict:
        """
        Notify local server that a filesystem broadcast download is complete.

        Called after successfully downloading data via rsync.

        Args:
            group_id: Broadcast group identifier
            key: Storage key
            local_path: Local path where data was downloaded

        Returns:
            Server response dict
        """
        message = {
            "command": "fs_broadcast_complete",
            "group_id": group_id,
            "key": key,
            "local_path": local_path,
        }
        return self._send_message(message)

    def fs_broadcast_get_path_remote(
        self,
        parent_ip: str,
        parent_port: int,
        group_id: str,
        key: str,
        timeout: float = 60.0,
    ) -> dict:
        """
        Request local path from parent's pod data server via TCP.

        Blocks until parent has completed its download, then returns the path.

        Args:
            parent_ip: IP address of parent pod
            parent_port: TCP port of parent's pod data server
            group_id: Broadcast group identifier
            key: Storage key
            timeout: Max time to wait for parent to complete

        Returns:
            Server response dict with 'local_path' and 'pod_ip'
        """
        message = {
            "command": "fs_broadcast_get_path",
            "group_id": group_id,
            "key": key,
            "timeout": timeout,
        }

        # Connect via TCP to parent's pod data server
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout + 10)  # Extra buffer for network

        try:
            sock.connect((parent_ip, parent_port))

            # Send message
            data = json.dumps(message).encode("utf-8")
            sock.sendall(struct.pack(">I", len(data)))
            sock.sendall(data)

            # Receive response length
            length_data = sock.recv(4)
            if not length_data:
                raise RuntimeError("Parent server closed connection")

            msg_length = struct.unpack(">I", length_data)[0]

            # Receive response
            data = b""
            while len(data) < msg_length:
                chunk = sock.recv(min(msg_length - len(data), 4096))
                if not chunk:
                    break
                data += chunk

            return json.loads(data.decode("utf-8"))

        finally:
            sock.close()


def is_server_running(socket_path: str = DEFAULT_SOCKET_PATH) -> bool:
    """Check if the GPU data server is running."""
    if not os.path.exists(socket_path):
        return False

    try:
        client = PodDataServerClient(socket_path)
        response = client.ping()
        return response.get("status") == "ok"
    except Exception:
        return False


def _forward_subprocess_output_to_log_capture(pipe, stream_name: str, source: str) -> None:
    """
    Read from subprocess pipe and forward to LogCapture.

    Runs in a daemon thread to continuously forward logs.
    """
    try:
        from kubetorch.servers.http.log_capture import get_log_capture
    except ImportError:
        # If LogCapture not available, just read and discard to prevent pipe blocking
        for line in iter(pipe.readline, b""):
            pass
        return

    for line in iter(pipe.readline, b""):
        try:
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                log_capture = get_log_capture()
                if log_capture:
                    log_capture.add_log(
                        message=text,
                        level="ERROR" if stream_name == "stderr" else "INFO",
                        extra_labels={"source": source},
                    )
        except Exception:
            pass  # Don't crash the reader thread


def start_server_if_needed(socket_path: str = DEFAULT_SOCKET_PATH) -> int:
    """
    Start the GPU data server if not already running.

    Returns:
        PID of the server (existing or newly started)
    """
    if is_server_running(socket_path):
        # Read existing PID
        if os.path.exists(SERVER_PID_FILE):
            with open(SERVER_PID_FILE) as f:
                return int(f.read().strip())
        return -1

    # Start new server process
    import subprocess
    import threading

    process = subprocess.Popen(
        [sys.executable, "-m", "kubetorch.data_store.pod_data_server"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    # Start daemon threads to forward subprocess output to LogCapture
    # These threads will also ensure pipes don't fill up and block the subprocess
    stdout_thread = threading.Thread(
        target=_forward_subprocess_output_to_log_capture,
        args=(process.stdout, "stdout", "pds"),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_forward_subprocess_output_to_log_capture,
        args=(process.stderr, "stderr", "pds"),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    # Wait for server to be ready
    for _ in range(50):  # 5 seconds timeout
        time.sleep(0.1)
        if is_server_running(socket_path):
            logger.info(f"Pod Data Server started (PID: {process.pid})")
            return process.pid

    raise RuntimeError("Failed to start Pod Data Server")


def main():
    """Main entry point for running the server."""
    import argparse

    _setup_cuda_ipc_permissions()

    parser = argparse.ArgumentParser(description="Pod Data Server for kubetorch")
    parser.add_argument(
        "--socket-path",
        default=DEFAULT_SOCKET_PATH,
        help=f"Unix socket path (default: {DEFAULT_SOCKET_PATH})",
    )
    parser.add_argument(
        "--nccl-port-start",
        type=int,
        default=DEFAULT_NCCL_PORT_RANGE_START,
        help=f"Start of NCCL port range (default: {DEFAULT_NCCL_PORT_RANGE_START})",
    )
    parser.add_argument(
        "--nccl-port-end",
        type=int,
        default=DEFAULT_NCCL_PORT_RANGE_END,
        help=f"End of NCCL port range (default: {DEFAULT_NCCL_PORT_RANGE_END})",
    )

    args = parser.parse_args()

    server = PodDataServer(
        socket_path=args.socket_path,
        nccl_port_start=args.nccl_port_start,
        nccl_port_end=args.nccl_port_end,
    )
    server.start()


if __name__ == "__main__":
    main()
