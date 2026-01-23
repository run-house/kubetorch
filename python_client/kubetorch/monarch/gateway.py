"""
Monarch Gateway - Server-side component for external Monarch access.

This class is deployed to K8s pods via kt.cls and handles Monarch operations
proxied from external clients over WebSocket. It maintains references to
HostMesh, ProcMesh, ActorMesh objects and executes operations on behalf of
the client.
"""

import logging
import os
import pickle
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

# Set Monarch/Hyperactor timeouts BEFORE importing monarch
# These must be set before any monarch imports to take effect
os.environ.setdefault("HYPERACTOR_HOST_SPAWN_READY_TIMEOUT", "300s")
os.environ.setdefault("HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT", "300s")
os.environ.setdefault("HYPERACTOR_MESH_PROC_SPAWN_MAX_IDLE", "300s")

logger = logging.getLogger(__name__)

# Default port for Monarch worker
DEFAULT_MONARCH_PORT = 26600


def _is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def _get_pod_ip() -> str:
    """Get the IP address of this pod."""
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)


class MonarchGateway:
    """
    Gateway for external Monarch access.

    Deployed as a kt.cls to K8s pods, this class:
    1. Starts the Monarch worker process if not already running
    2. Bootstraps a Monarch root client
    3. Attaches to worker pods discovered via headless service DNS
    4. Creates and manages HostMesh, ProcMesh, ActorMesh objects
    5. Executes actor method calls on behalf of external clients

    The external client (KubernetesJob) communicates with this gateway over
    WebSocket, sending operation requests and receiving results.
    """

    def __init__(self, monarch_port: int = DEFAULT_MONARCH_PORT):
        self._initialized = False
        self._host_mesh = None
        self._proc_meshes: Dict[str, Any] = {}
        self._actor_meshes: Dict[str, Any] = {}
        self._futures: Dict[str, Any] = {}
        self._worker_process: Optional[subprocess.Popen] = None
        self._monarch_port = monarch_port
        self._worker_pythonpath: set = set()  # Paths included in worker's PYTHONPATH

        # Start the Monarch worker if not already running
        self._ensure_worker_running()

    def _ensure_worker_running(self, extra_paths: Optional[List[str]] = None):
        """
        Start the Monarch worker process if not already running.

        Args:
            extra_paths: Additional paths to include in PYTHONPATH for the worker.
                         If paths are new and worker is running, it will be restarted.

        Checks if the port is in use first - if so, assumes worker is already running.
        """
        pod_ip = _get_pod_ip()
        extra_paths = extra_paths or []

        # Check if we need to restart the worker for new paths
        new_paths = set(extra_paths) - self._worker_pythonpath
        if new_paths and self._worker_process is not None:
            logger.info(f"New module paths needed: {new_paths}. Restarting Monarch worker...")
            self._stop_worker()

        if _is_port_in_use(self._monarch_port, pod_ip):
            if self._worker_process is not None:
                # Worker is ours and running
                logger.debug(f"Monarch worker already running on {pod_ip}:{self._monarch_port}")
                return
            else:
                # Port in use by something else (maybe previous worker)
                logger.warning(
                    f"Monarch worker port {self._monarch_port} already in use on {pod_ip}, "
                    "but not by us. Waiting for it to free up..."
                )
                time.sleep(3)
                if _is_port_in_use(self._monarch_port, pod_ip):
                    logger.info("Port still in use, assuming existing worker is running")
                    return

        logger.info(f"Starting Monarch worker on {pod_ip}:{self._monarch_port}")

        # Build PYTHONPATH including all needed directories
        # Start with cwd (the synced code root) and all its subdirectories
        # This ensures any module within the synced directory is importable
        cwd = os.getcwd()
        paths_to_add = {cwd}

        # Add all immediate subdirectories that might contain Python modules
        try:
            for entry in os.listdir(cwd):
                entry_path = os.path.join(cwd, entry)
                if os.path.isdir(entry_path) and not entry.startswith("."):
                    paths_to_add.add(entry_path)
        except Exception as e:
            logger.warning(f"Failed to scan cwd subdirectories: {e}")

        paths_to_add.update(extra_paths)
        self._worker_pythonpath.update(paths_to_add)
        logger.info(f"Module paths for worker: {paths_to_add}")

        worker_script = f"""
import os
import sys
import logging

# Set Monarch/Hyperactor timeouts before importing monarch
os.environ.setdefault("HYPERACTOR_HOST_SPAWN_READY_TIMEOUT", "300s")
os.environ.setdefault("HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT", "300s")
os.environ.setdefault("HYPERACTOR_MESH_PROC_SPAWN_MAX_IDLE", "300s")

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

from monarch.actor import run_worker_loop_forever

address = "tcp://{pod_ip}:{self._monarch_port}"
print(f"Starting Monarch worker at {{address}} (PYTHONPATH={{os.environ.get('PYTHONPATH', '')}})", flush=True)
run_worker_loop_forever(address=address, ca="trust_all_connections")
"""

        # Build environment with PYTHONPATH including all needed directories
        worker_env = os.environ.copy()
        current_pythonpath = worker_env.get("PYTHONPATH", "")
        pythonpath_parts = [p for p in current_pythonpath.split(":") if p]
        for path in self._worker_pythonpath:
            if path not in pythonpath_parts:
                pythonpath_parts.insert(0, path)
        worker_env["PYTHONPATH"] = ":".join(pythonpath_parts)

        logger.info(f"Worker PYTHONPATH: {worker_env['PYTHONPATH']}")

        try:
            self._worker_process = subprocess.Popen(
                [sys.executable, "-c", worker_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                env=worker_env,
            )

            # Give the worker a moment to start
            time.sleep(2)

            # Check if it's still running
            if self._worker_process.poll() is not None:
                _, stderr = self._worker_process.communicate()
                raise RuntimeError(f"Monarch worker failed to start: {stderr.decode('utf-8', errors='replace')}")

            logger.info(f"Monarch worker started with PID {self._worker_process.pid}")

        except Exception as e:
            raise RuntimeError(f"Failed to start Monarch worker: {e}. Ensure torchmonarch is installed.")

    def _stop_worker(self):
        """Stop the Monarch worker subprocess."""
        if self._worker_process is not None:
            logger.info(f"Stopping Monarch worker (PID {self._worker_process.pid})")
            try:
                self._worker_process.terminate()
                self._worker_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._worker_process.kill()
                self._worker_process.wait()
            self._worker_process = None
            # Give the port a moment to be released
            time.sleep(1)

    def _discover_worker_ips(self, headless_service_dns: str) -> List[str]:
        """Discover all worker pod IPs via headless service DNS."""
        try:
            addr_info = socket.getaddrinfo(headless_service_dns, None, socket.AF_INET)
            pod_ips = sorted(list(set([addr[4][0] for addr in addr_info])))
            logger.info(f"Discovered {len(pod_ips)} worker IPs via DNS: {pod_ips}")
            return pod_ips
        except socket.gaierror as e:
            logger.error(f"Failed to resolve headless service DNS {headless_service_dns}: {e}")
            raise

    def initialize(
        self,
        headless_service_dns: Optional[str] = None,
        worker_ips: Optional[List[str]] = None,
        monarch_port: int = 26600,
    ) -> Dict[str, Any]:
        """
        Initialize the gateway by attaching to worker pods.

        Args:
            headless_service_dns: DNS name of headless service for pod discovery.
                                  If not provided, uses KT_SERVICE_NAME-headless.
            worker_ips: Explicit list of worker IPs (alternative to DNS discovery).
            monarch_port: Port where Monarch workers are listening (default 26600).

        Returns:
            Dict with initialization status and host mesh info.
        """
        if self._initialized:
            return {
                "status": "already_initialized",
                "host_mesh_id": "hm_default",
                "num_workers": len(self._host_mesh) if self._host_mesh else 0,
            }

        # Discover workers
        if worker_ips:
            pod_ips = worker_ips
        else:
            if not headless_service_dns:
                service_name = os.environ.get("KT_SERVICE_NAME", "")
                namespace = os.environ.get("POD_NAMESPACE", "default")
                headless_service_dns = f"{service_name}-headless.{namespace}.svc.cluster.local"

            pod_ips = self._discover_worker_ips(headless_service_dns)

        if not pod_ips:
            raise RuntimeError("No worker IPs discovered")

        # Build worker addresses
        worker_addresses = [f"tcp://{ip}:{monarch_port}" for ip in pod_ips]
        logger.info(f"Attaching to {len(worker_addresses)} Monarch workers")

        # Import Monarch and attach to workers
        try:
            # Enable TCP transport explicitly (required for remote workers)
            from monarch.actor import enable_transport

            enable_transport("tcp")
            logger.info("Enabled TCP transport for Monarch")

            from monarch.actor import attach_to_workers

            self._host_mesh = attach_to_workers(
                ca="trust_all_connections",
                workers=worker_addresses,
            )

            # Wait for the host mesh to be fully initialized (connections established)
            logger.info("Waiting for host mesh to initialize...")
            self._host_mesh.initialized.get()
            logger.info("Host mesh initialized successfully")

            # Give connections a moment to fully stabilize (per SkyPilot example)
            time.sleep(5)
            logger.info("Host mesh connections stabilized")

            self._initialized = True

            logger.info(f"Successfully attached to {len(worker_addresses)} workers")
            return {
                "status": "initialized",
                "host_mesh_id": "hm_default",
                "num_workers": len(worker_addresses),
                "worker_ips": pod_ips,
            }

        except Exception as e:
            logger.error(f"Failed to attach to workers: {e}")
            raise

    def spawn_procs(
        self,
        per_host: Dict[str, int],
        name: str = "procs",
    ) -> Dict[str, Any]:
        """
        Spawn a ProcMesh on the HostMesh.

        Args:
            per_host: Dict specifying processes per host, e.g. {"gpus": 8}
            name: Name for the proc mesh

        Returns:
            Dict with proc_mesh_id and shape info
        """
        if not self._initialized:
            raise RuntimeError("Gateway not initialized. Call initialize() first.")

        proc_mesh = self._host_mesh.spawn_procs(per_host=per_host, name=name)
        proc_mesh_id = f"pm_{uuid4().hex[:8]}"
        self._proc_meshes[proc_mesh_id] = proc_mesh

        # Get shape info
        shape = dict(proc_mesh.sizes) if hasattr(proc_mesh, "sizes") else {}

        logger.info(f"Spawned proc mesh {proc_mesh_id} with shape {shape}")
        return {
            "proc_mesh_id": proc_mesh_id,
            "shape": shape,
        }

    def spawn_actors(
        self,
        proc_mesh_id: str,
        name: str,
        module_name: str,
        class_name: str,
        module_path: str = "",
        args: Optional[List] = None,
        kwargs: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Spawn an ActorMesh on a ProcMesh.

        Args:
            proc_mesh_id: ID of the proc mesh to spawn on
            name: Name for the actor mesh
            module_name: Module containing the actor class
            class_name: Name of the actor class
            module_path: Relative path from synced root to the module's directory
            args: Positional arguments for actor __init__
            kwargs: Keyword arguments for actor __init__

        Returns:
            Dict with actor_mesh_id and shape info
        """
        if proc_mesh_id not in self._proc_meshes:
            raise ValueError(f"Unknown proc_mesh_id: {proc_mesh_id}")

        proc_mesh = self._proc_meshes[proc_mesh_id]

        # Import the actor class from the synced module
        import importlib
        import sys

        # Add the module's directory to sys.path
        # module_path is relative to the synced root (which is cwd on server)
        cwd = os.getcwd()
        if module_path:
            import_path = os.path.join(cwd, module_path)
        else:
            import_path = cwd

        if import_path not in sys.path:
            sys.path.insert(0, import_path)

        try:
            module = importlib.import_module(module_name)
            actor_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to import actor class '{class_name}' from module '{module_name}': {e}. "
                f"module_path='{module_path}', import_path='{import_path}'. "
                f"Current sys.path: {sys.path[:5]}..."
            )

        args = args or []
        kwargs = kwargs or {}

        actor_mesh = proc_mesh.spawn(name, actor_class, *args, **kwargs)
        actor_mesh_id = f"am_{uuid4().hex[:8]}"
        self._actor_meshes[actor_mesh_id] = actor_mesh

        # Get shape info
        shape = dict(actor_mesh.sizes) if hasattr(actor_mesh, "sizes") else {}

        logger.info(f"Spawned actor mesh {actor_mesh_id} with shape {shape}")
        return {
            "actor_mesh_id": actor_mesh_id,
            "shape": shape,
        }

    def call_endpoint(
        self,
        actor_mesh_id: str,
        endpoint_name: str,
        args_bytes: bytes,
        kwargs_bytes: bytes,
        selection: str = "all",
    ) -> Dict[str, Any]:
        """
        Call an endpoint on an ActorMesh.

        Args:
            actor_mesh_id: ID of the actor mesh
            endpoint_name: Name of the endpoint method
            args_bytes: Pickled positional arguments
            kwargs_bytes: Pickled keyword arguments
            selection: "all" for broadcast call, "one" for single actor

        Returns:
            Dict with future_id for retrieving results
        """
        logger.info(f"call_endpoint received: {endpoint_name} on {actor_mesh_id}")

        if actor_mesh_id not in self._actor_meshes:
            raise ValueError(f"Unknown actor_mesh_id: {actor_mesh_id}")

        actor_mesh = self._actor_meshes[actor_mesh_id]
        args = pickle.loads(args_bytes) if args_bytes else []
        kwargs = pickle.loads(kwargs_bytes) if kwargs_bytes else {}

        logger.info(f"Calling {endpoint_name}.call() on actor mesh...")
        endpoint = getattr(actor_mesh, endpoint_name)

        if selection == "one":
            future = endpoint.call_one(*args, **kwargs)
        else:
            future = endpoint.call(*args, **kwargs)
        logger.info("endpoint.call() returned future")

        future_id = f"fut_{uuid4().hex[:8]}"
        self._futures[future_id] = future

        logger.info(f"Endpoint {endpoint_name} called on {actor_mesh_id}, future_id={future_id}")
        return {"future_id": future_id}

    def broadcast_endpoint(
        self,
        actor_mesh_id: str,
        endpoint_name: str,
        args_bytes: bytes,
        kwargs_bytes: bytes,
    ) -> Dict[str, Any]:
        """
        Broadcast to an endpoint (fire-and-forget, no return value).

        Args:
            actor_mesh_id: ID of the actor mesh
            endpoint_name: Name of the endpoint method
            args_bytes: Pickled positional arguments
            kwargs_bytes: Pickled keyword arguments

        Returns:
            Dict with status
        """
        if actor_mesh_id not in self._actor_meshes:
            raise ValueError(f"Unknown actor_mesh_id: {actor_mesh_id}")

        actor_mesh = self._actor_meshes[actor_mesh_id]
        args = pickle.loads(args_bytes) if args_bytes else []
        kwargs = pickle.loads(kwargs_bytes) if kwargs_bytes else {}

        endpoint = getattr(actor_mesh, endpoint_name)
        endpoint.broadcast(*args, **kwargs)

        logger.debug(f"Endpoint {endpoint_name} broadcast completed")
        return {"status": "ok"}

    def _convert_monarch_result(self, result: Any) -> Any:
        """
        Convert Monarch-specific objects to plain Python objects.

        This ensures results can be unpickled on clients without torchmonarch installed.
        """
        result_type = type(result).__name__
        result_module = getattr(type(result), "__module__", "")

        logger.debug(f"Converting result: type={result_type}, module={result_module}")

        # Check if it's a ValueMesh (result of endpoint.call())
        # ValueMesh has items() that yields (Point, value) pairs
        if hasattr(result, "items") and hasattr(result, "_labels"):
            # It's a ValueMesh - convert to dict with data and shape
            try:
                data = {}
                for point, value in result.items():
                    # Point is a named tuple-like object, convert to regular tuple
                    if hasattr(point, "_asdict"):
                        key = tuple(point._asdict().values())
                    elif hasattr(point, "__iter__"):
                        key = tuple(point)
                    else:
                        key = (point,)
                    data[key] = value

                # Get shape from labels
                shape = {}
                if hasattr(result, "_labels") and hasattr(result, "_shape"):
                    for label, size in zip(result._labels, result._shape):
                        shape[label] = size

                logger.debug(f"Converted ValueMesh: {len(data)} items, shape={shape}")
                return {"_type": "ValueMesh", "data": data, "shape": shape}
            except Exception as e:
                logger.warning(f"Failed to convert ValueMesh via items(): {e}")
                # Fall back to trying to extract values directly
                try:
                    return list(result.values())
                except Exception as e2:
                    logger.warning(f"Failed to extract values(): {e2}")

        # For other Monarch objects, try to extract the underlying data
        if "monarch" in result_module:
            logger.warning(f"Unknown Monarch type: {result_type}, attempting conversion")
            # Try common extraction methods (but not item() which requires kwargs)
            if hasattr(result, "values"):
                try:
                    return list(result.values())
                except Exception:
                    pass
            if hasattr(result, "to_dict"):
                try:
                    return result.to_dict()
                except Exception:
                    pass

        return result

    def get_future_result(
        self,
        future_id: str,
        timeout: Optional[float] = None,
    ) -> bytes:
        """
        Get the result of a future.

        Args:
            future_id: ID of the future
            timeout: Timeout in seconds (None for no timeout)

        Returns:
            Pickled result (converted to plain Python objects)
        """
        if future_id not in self._futures:
            raise ValueError(f"Unknown future_id: {future_id}")

        future = self._futures[future_id]
        logger.info(f"Getting result for future {future_id} (timeout={timeout})")
        result = future.get(timeout=timeout)
        logger.info(f"Got result for future {future_id}: {type(result)}")

        # Convert Monarch objects to plain Python for client-side unpickling
        converted_result = self._convert_monarch_result(result)
        logger.info(f"Converted result type: {type(converted_result)}")

        # Clean up the future reference
        del self._futures[future_id]

        return pickle.dumps(converted_result)

    def check_future_ready(self, future_id: str) -> Dict[str, Any]:
        """
        Check if a future is ready without blocking.

        Args:
            future_id: ID of the future

        Returns:
            Dict with ready status
        """
        if future_id not in self._futures:
            raise ValueError(f"Unknown future_id: {future_id}")

        future = self._futures[future_id]

        # Check if future is done (implementation depends on Monarch's Future API)
        # For now, assume we need to try getting with timeout=0
        try:
            result = future.get(timeout=0)
            return {"ready": True, "result": pickle.dumps(result)}
        except TimeoutError:
            return {"ready": False}
        except Exception as e:
            return {"ready": True, "error": str(e)}

    def stop_actor_mesh(self, actor_mesh_id: str) -> Dict[str, Any]:
        """Stop an actor mesh."""
        if actor_mesh_id not in self._actor_meshes:
            raise ValueError(f"Unknown actor_mesh_id: {actor_mesh_id}")

        actor_mesh = self._actor_meshes[actor_mesh_id]
        actor_mesh.stop().get()
        del self._actor_meshes[actor_mesh_id]

        logger.info(f"Stopped actor mesh {actor_mesh_id}")
        return {"status": "stopped"}

    def stop_proc_mesh(self, proc_mesh_id: str) -> Dict[str, Any]:
        """Stop a proc mesh."""
        if proc_mesh_id not in self._proc_meshes:
            raise ValueError(f"Unknown proc_mesh_id: {proc_mesh_id}")

        proc_mesh = self._proc_meshes[proc_mesh_id]
        proc_mesh.stop().get()
        del self._proc_meshes[proc_mesh_id]

        logger.info(f"Stopped proc mesh {proc_mesh_id}")
        return {"status": "stopped"}

    def shutdown(self) -> Dict[str, Any]:
        """Shutdown the gateway and all resources."""
        # Stop all actor meshes
        for actor_mesh_id in list(self._actor_meshes.keys()):
            try:
                self.stop_actor_mesh(actor_mesh_id)
            except Exception as e:
                logger.warning(f"Error stopping actor mesh {actor_mesh_id}: {e}")

        # Stop all proc meshes
        for proc_mesh_id in list(self._proc_meshes.keys()):
            try:
                self.stop_proc_mesh(proc_mesh_id)
            except Exception as e:
                logger.warning(f"Error stopping proc mesh {proc_mesh_id}: {e}")

        # Shutdown host mesh
        if self._host_mesh:
            try:
                self._host_mesh.shutdown().get()
            except Exception as e:
                logger.warning(f"Error shutting down host mesh: {e}")

        self._initialized = False
        self._host_mesh = None
        self._proc_meshes.clear()
        self._actor_meshes.clear()
        self._futures.clear()

        logger.info("Gateway shutdown complete")
        return {"status": "shutdown"}

    def get_status(self) -> Dict[str, Any]:
        """Get current gateway status."""
        # Get number of workers from host mesh if available
        num_workers = 0
        if self._host_mesh is not None:
            try:
                # HostMesh may have a size or len
                num_workers = len(self._host_mesh) if hasattr(self._host_mesh, "__len__") else 1
            except Exception:
                num_workers = 1

        return {
            "initialized": self._initialized,
            "num_workers": num_workers,
            "num_proc_meshes": len(self._proc_meshes),
            "num_actor_meshes": len(self._actor_meshes),
            "num_pending_futures": len(self._futures),
        }
