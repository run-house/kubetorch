"""
GPU Data Server - Per-node server for GPU tensor transfers via NCCL.

This server runs as a separate process on each node to handle NCCL broadcasts,
isolating NCCL process group operations from application processes.

Architecture:
- Application processes call kt.put(data=tensor) which registers the tensor
  via CUDA IPC handles with this server
- The server holds IPC handles (not tensors) - memory is owned by the original process
- When getters request data, the server reconstructs tensors and performs NCCL broadcast
- Server-to-server communication for NCCL coordination (no metadata server bounce)

Usage:
    # Start server (typically done automatically on first kt.put)
    python -m kubetorch.data_store.gpu_data_server

    # Or programmatically
    from kubetorch.data_store.gpu_data_server import start_server
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
from typing import Any, Dict, List, Optional, Tuple

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

    if hasattr(torch, "UntypedStorage"):
        storage = torch.UntypedStorage._new_shared_cuda(*ipc_handle)
    elif hasattr(torch.cuda, "UntypedStorage"):
        storage = torch.cuda.UntypedStorage._new_shared_cuda(*ipc_handle)
    else:
        storage = torch.cuda.ByteStorage._new_shared_cuda(*ipc_handle)

    tensor = torch.empty(shape, dtype=dtype, device=f"cuda:{device}")
    tensor.set_(storage, storage_offset=0, size=shape)

    return tensor


class GPUDataServer:
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

    def start(self):
        """Start the GPU data server."""
        # Remove stale socket file
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        # Write PID file
        with open(SERVER_PID_FILE, "w") as f:
            f.write(str(os.getpid()))

        logger.info(f"GPU Data Server starting (PID: {os.getpid()})")
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

        logger.info("GPU Data Server ready")

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
        logger.info("Cleaning up GPU Data Server")

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
            elif command == "ping":
                response = {"status": "ok", "pid": os.getpid(), "tcp_port": self.tcp_port}
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
            # Initialize process group as rank 0
            if dist.is_initialized():
                # Create new group for this broadcast
                ranks = list(range(world_size))
                process_group = dist.new_group(ranks)
            else:
                dist.init_process_group(
                    backend="nccl",
                    rank=0,
                    world_size=world_size,
                )
                process_group = dist.group.WORLD

            # Broadcast the tensor
            dist.broadcast(tensor, src=0, group=process_group)

            logger.info(f"Broadcast {broadcast_id} complete")

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
            }
        """
        key = message["key"]
        source_ip = message["source_ip"]
        source_gpu_port = message["source_gpu_port"]
        dest_ipc_handle = _deserialize_ipc_handle(message["dest_ipc_handle"])
        shape = tuple(message["shape"])
        dtype = message["dtype"]
        device = message["device"]

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
            )

            return {"status": "ok", "broadcast_id": broadcast_id}

        except Exception as e:
            logger.error(f"Failed to receive broadcast for '{key}': {e}")
            return {"status": "error", "error": str(e)}

    def _handle_remote_broadcast_request(self, message: dict) -> dict:
        """
        Handle a remote getter requesting data from this source server.

        This server has the data registered locally. We set up NCCL and tell
        the getter to join.

        Args:
            message: {
                "key": storage key,
                "getter_ip": IP of getter,
                "getter_port": TCP port of getter's GPU server,
                "shape": expected tensor shape,
                "dtype": expected tensor dtype,
            }
        """
        key = message["key"]
        # getter_ip and getter_port available for future multi-getter support
        _ = message["getter_ip"], message["getter_port"]

        with self._lock:
            if key not in self._registered:
                return {"status": "error", "error": f"Key '{key}' not registered on this server"}

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

        # Set up NCCL broadcast
        import uuid

        broadcast_id = str(uuid.uuid4())[:8]
        nccl_port = self._get_next_nccl_port()
        pod_ip = os.getenv("POD_IP", "127.0.0.1")
        world_size = 2  # Source + 1 getter (TODO: support multiple getters)

        # Start broadcast in background thread
        broadcast_thread = threading.Thread(
            target=self._serve_nccl_broadcast,
            args=(tensor, broadcast_id, nccl_port, world_size),
            daemon=True,
        )
        broadcast_thread.start()

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
                return {"status": "ok", "group_id": group_id, "rank": rank}

            finally:
                if process_group is not None:
                    try:
                        dist.destroy_process_group()
                    except Exception as e:
                        logger.warning(f"Failed to destroy process group: {e}")

        except Exception as e:
            logger.error(f"Broadcast group {group_id} failed: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return {"status": "error", "error": str(e)}

    def _serve_nccl_broadcast(
        self,
        tensor,
        broadcast_id: str,
        nccl_port: int,
        world_size: int,
    ):
        """
        Serve NCCL broadcast as rank 0 (source) in a background thread.
        """
        dist = _get_torch_distributed()
        pod_ip = os.getenv("POD_IP", "127.0.0.1")

        # Set up NCCL environment
        os.environ["MASTER_ADDR"] = pod_ip
        os.environ["MASTER_PORT"] = str(nccl_port)

        logger.info(
            f"Serving NCCL broadcast {broadcast_id}: world_size={world_size}, "
            f"MASTER_ADDR={pod_ip}, MASTER_PORT={nccl_port}"
        )

        process_group = None
        try:
            # Initialize process group as rank 0
            if dist.is_initialized():
                ranks = list(range(world_size))
                process_group = dist.new_group(ranks)
            else:
                dist.init_process_group(
                    backend="nccl",
                    rank=0,
                    world_size=world_size,
                )
                process_group = dist.group.WORLD

            # Broadcast the tensor
            dist.broadcast(tensor, src=0, group=process_group)

            logger.info(f"NCCL broadcast {broadcast_id} complete")

        except Exception as e:
            logger.error(f"NCCL broadcast {broadcast_id} failed: {e}")

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
        dest_tensor,
        broadcast_id: str,
        master_addr: str,
        master_port: int,
        rank: int,
        world_size: int,
    ):
        """
        Join NCCL broadcast as a receiver (rank > 0).
        """
        dist = _get_torch_distributed()

        # Set up NCCL environment
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)

        logger.info(
            f"Joining NCCL broadcast {broadcast_id}: rank={rank}, world_size={world_size}, "
            f"MASTER_ADDR={master_addr}, MASTER_PORT={master_port}"
        )

        process_group = None
        try:
            # Initialize process group
            if dist.is_initialized():
                ranks = list(range(world_size))
                process_group = dist.new_group(ranks)
            else:
                dist.init_process_group(
                    backend="nccl",
                    rank=rank,
                    world_size=world_size,
                )
                process_group = dist.group.WORLD

            # Receive broadcast into destination tensor
            dist.broadcast(dest_tensor, src=0, group=process_group)

            logger.info(f"NCCL broadcast {broadcast_id} received successfully")

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


class GPUDataServerClient:
    """Client for communicating with the GPU Data Server."""

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

        return self._send_message(message, timeout=120.0)

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


def is_server_running(socket_path: str = DEFAULT_SOCKET_PATH) -> bool:
    """Check if the GPU data server is running."""
    if not os.path.exists(socket_path):
        return False

    try:
        client = GPUDataServerClient(socket_path)
        response = client.ping()
        return response.get("status") == "ok"
    except Exception:
        return False


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

    process = subprocess.Popen(
        [sys.executable, "-m", "kubetorch.data_store.gpu_data_server"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    # Wait for server to be ready
    for _ in range(50):  # 5 seconds timeout
        time.sleep(0.1)
        if is_server_running(socket_path):
            logger.info(f"GPU Data Server started (PID: {process.pid})")
            return process.pid

    raise RuntimeError("Failed to start GPU Data Server")


def main():
    """Main entry point for running the server."""
    import argparse

    _setup_cuda_ipc_permissions()

    parser = argparse.ArgumentParser(description="GPU Data Server for kubetorch")
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

    server = GPUDataServer(
        socket_path=args.socket_path,
        nccl_port_start=args.nccl_port_start,
        nccl_port_end=args.nccl_port_end,
    )
    server.start()


if __name__ == "__main__":
    main()
