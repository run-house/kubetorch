"""
Monarch KubernetesJob - Client for external Monarch access via Kubetorch.

This module provides the KubernetesJob class that allows users to create and
interact with Monarch meshes from outside the Kubernetes cluster.
"""

import json
import logging
import threading
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    import kubetorch as kt

logger = logging.getLogger(__name__)

# Default Monarch port for worker communication
MONARCH_WORKER_PORT = 26600

# Default GPU image compatible with torchmonarch
# For CPU-only, we use kt.images.Debian() which is set in _configure_monarch_image
DEFAULT_MONARCH_IMAGE_GPU = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"


class GatewayConnection:
    """
    WebSocket connection to the MonarchGateway.

    Handles communication between the local client and the remote gateway,
    including connection management, message serialization, and reconnection.
    """

    def __init__(self, service_url: str, use_websocket: bool = True):
        """
        Initialize the gateway connection.

        Args:
            service_url: URL of the Kubetorch service (e.g., http://my-service.ns.svc:32300)
            use_websocket: If True, use WebSocket connection. If False, use HTTP.
        """
        self._service_url = service_url.rstrip("/")
        self._use_websocket = use_websocket
        self._ws = None
        self._ws_lock = threading.Lock()
        self._request_counter = 0

    def _get_ws_url(self) -> str:
        """Convert HTTP URL to WebSocket URL."""
        url = self._service_url
        if url.startswith("https://"):
            return url.replace("https://", "wss://") + "/ws/callable"
        elif url.startswith("http://"):
            return url.replace("http://", "ws://") + "/ws/callable"
        else:
            return "ws://" + url + "/ws/callable"

    def _ensure_connected(self):
        """Ensure WebSocket connection is established."""
        if self._ws is not None:
            return

        try:
            import websocket
        except ImportError:
            raise ImportError(
                "websocket-client package required for WebSocket connection. "
                "Install with: pip install websocket-client"
            )

        ws_url = self._get_ws_url()
        logger.info(f"Connecting to gateway WebSocket: {ws_url}")

        self._ws = websocket.create_connection(ws_url, timeout=30)
        logger.info("WebSocket connection established")

    def call(self, method: str, **kwargs) -> Any:
        """
        Call a method on the gateway.

        Args:
            method: The gateway method to call (e.g., "initialize", "spawn_procs")
            **kwargs: Arguments to pass to the method

        Returns:
            The method result
        """
        if self._use_websocket:
            return self._call_websocket(method, **kwargs)
        else:
            return self._call_http(method, **kwargs)

    def _call_websocket(self, method: str, **kwargs) -> Any:
        """Make a call over WebSocket."""
        with self._ws_lock:
            self._ensure_connected()

            self._request_counter += 1
            request_id = f"req_{self._request_counter}"

            # Prepare request
            request = {
                "request_id": request_id,
                "cls_or_fn_name": "MonarchGateway",
                "method_name": method,
                "params": {"kwargs": kwargs},
                "serialization": "pickle",
            }

            # Handle bytes arguments - convert to base64 for JSON
            import base64

            params_kwargs = request["params"]["kwargs"]
            for key, value in params_kwargs.items():
                if isinstance(value, bytes):
                    params_kwargs[key] = {"__bytes__": base64.b64encode(value).decode("utf-8")}

            # Send request
            self._ws.send(json.dumps(request))

            # Receive response
            response_str = self._ws.recv()
            response = json.loads(response_str)

            if response.get("error"):
                error = response["error"]
                traceback_str = error.get("traceback", "")
                error_msg = f"Gateway error ({error.get('type', 'Unknown')}): {error.get('message', 'No message')}"
                if traceback_str:
                    error_msg += f"\n\nRemote traceback:\n{traceback_str}"
                raise RuntimeError(error_msg)

            result = response.get("result")

            # Handle pickle-serialized results - server returns {"data": "base64_encoded_pickle"}
            if isinstance(result, dict) and "data" in result:
                import pickle

                pickled_data = base64.b64decode(result["data"])
                result = pickle.loads(pickled_data)

            # Handle bytes results - convert from base64
            if isinstance(result, dict) and "__bytes__" in result:
                result = base64.b64decode(result["__bytes__"])

            return result

    def _call_http(self, method: str, **kwargs) -> Any:
        """Make a call over HTTP (fallback)."""
        import base64
        import pickle

        import httpx

        url = f"{self._service_url}/MonarchGateway/{method}"

        # Handle bytes arguments - convert to base64 for JSON transport
        pickled_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, bytes):
                pickled_kwargs[key] = {"__bytes__": base64.b64encode(value).decode("utf-8")}
            else:
                pickled_kwargs[key] = value

        response = httpx.post(
            url,
            json={"kwargs": pickled_kwargs},
            headers={"X-Serialization": "pickle"},
            timeout=300,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Gateway HTTP error: {response.status_code} - {response.text}")

        result = response.json()

        # Handle pickle-serialized results - server returns {"data": "base64_encoded_pickle"}
        if isinstance(result, dict) and "data" in result:
            pickled_data = base64.b64decode(result["data"])
            result = pickle.loads(pickled_data)

        # Handle bytes results
        if isinstance(result, dict) and "__bytes__" in result:
            result = base64.b64decode(result["__bytes__"])

        return result

    def close(self):
        """Close the connection."""
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None


class KubernetesJob:
    """
    Kubetorch's Monarch job for external cluster access.

    This class allows users to create and interact with Monarch meshes from
    outside the Kubernetes cluster. It:
    1. Deploys a MonarchGateway class to K8s pods via kt.cls
    2. Establishes a WebSocket connection to the gateway
    3. Provides proxy objects that mirror Monarch's API

    Usage:
        job = KubernetesJob(compute=kt.Compute(cpu="4", gpu=8, replicas=4))
        job.apply()
        state = job.state()
        proc_mesh = state.workers.spawn_procs(per_host={"gpus": 8})
        actors = proc_mesh.spawn("trainers", TrainerActor)
        result = actors.train.call(config).get()

    For pre-allocated compute:
        job = KubernetesJob(selector={"app": "my-monarch-workers"})
    """

    def __init__(
        self,
        compute: Optional["kt.Compute"] = None,
        selector: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
        namespace: str = "default",
        monarch_port: int = 26600,
        use_websocket: bool = True,
        sync_dirs: Optional[List[str]] = None,
    ):
        """
        Initialize a KubernetesJob.

        Args:
            compute: kt.Compute object for fresh resource allocation.
                     Mutually exclusive with selector.
            selector: Label selector for pre-allocated pods.
                      Mutually exclusive with compute.
            name: Name for the job/service. Auto-generated if not provided.
            namespace: Kubernetes namespace.
            monarch_port: Port where Monarch workers listen (default 26600).
            use_websocket: Use WebSocket for communication (default True).
            sync_dirs: List of directories containing actor class definitions to sync.
                      Defaults to the git root of the current working directory.
        """
        if compute is not None and selector is not None:
            raise ValueError("Cannot specify both compute and selector")
        if compute is None and selector is None:
            raise ValueError("Must specify either compute or selector")

        self._compute = compute
        self._selector = selector
        self._name = name or f"monarch-{uuid4().hex[:8]}"
        self._namespace = namespace
        self._monarch_port = monarch_port
        self._use_websocket = use_websocket
        self._sync_dirs = sync_dirs

        self._gateway_module = None
        self._gateway_connection: Optional[GatewayConnection] = None
        self._applied = False
        self._service_url: Optional[str] = None

    def _configure_monarch_image(
        self, image: Optional["kt.Image"], has_gpus: bool, sync_dirs: Optional[List[str]] = None
    ) -> "kt.Image":
        """
        Configure the image with Monarch worker bootstrap.

        If no image is provided, creates one with the default Monarch base image.
        Uses kt.images.Debian() for CPU-only (faster to pull), GPU image for GPU workloads.
        Adds torchmonarch-nightly installation and worker startup command.

        Args:
            image: Optional existing kt.Image to extend
            has_gpus: Whether the compute has GPU resources
            sync_dirs: Directories containing actor definitions to sync

        Returns:
            Configured kt.Image with Monarch worker support
        """
        import os

        import kubetorch as kt

        if image is None:
            if has_gpus:
                # GPU workloads use PyTorch CUDA image
                image = kt.Image(name="monarch-worker").from_docker(DEFAULT_MONARCH_IMAGE_GPU)
            else:
                # CPU-only uses standard Kubetorch Debian image
                image = kt.images.Debian()
        elif image.image_id is None:
            # Image exists but has no base - set appropriate default
            if has_gpus:
                image = image.from_docker(DEFAULT_MONARCH_IMAGE_GPU)
            else:
                # Use Debian base for CPU
                image = kt.images.Debian()

        # For CPU-only, install PyTorch CPU version (smaller than CUDA version)
        if not has_gpus:
            image = image.pip_install(["torch --index-url https://download.pytorch.org/whl/cpu"])

        # Install torchmonarch (required for Monarch worker)
        image = image.pip_install(["torchmonarch"])

        # Determine directories to sync
        if sync_dirs:
            dirs_to_sync = [os.path.abspath(d) for d in sync_dirs]
        else:
            # Default to the git root (or project root) of cwd
            from kubetorch.resources.callables.utils import locate_working_dir

            working_dir, _, _ = locate_working_dir(os.getcwd())
            dirs_to_sync = [working_dir]

        # Copy each sync directory contents to the server's working directory
        # so modules can be imported by their simple names (e.g., "demo" not "subdir.demo")
        for sync_dir in dirs_to_sync:
            image = image.copy(sync_dir, ".", contents=True)

        # Note: Worker startup is handled by MonarchGateway.__init__, not in the image.
        # This allows proper subprocess management and port conflict detection.

        return image

    def apply(self):
        """
        Deploy the gateway and prepare workers.

        This is idempotent - calling multiple times is a no-op.
        """
        if self._applied:
            return

        import kubetorch as kt
        from kubetorch.monarch.gateway import MonarchGateway

        logger.info(f"Applying KubernetesJob: {self._name}")

        if self._compute is not None:
            # Fresh allocation - deploy with kt.cls
            # Check if compute has GPU resources
            has_gpus = bool(getattr(self._compute, "gpus", None))

            # Configure image with Monarch worker bootstrap and sync directories
            monarch_image = self._configure_monarch_image(self._compute.image, has_gpus, self._sync_dirs)
            self._compute.image = monarch_image

            # Enable pickle serialization (needed for actor classes and results)
            self._compute.allowed_serialization = ["json", "pickle"]

            # Get the number of workers from compute config
            replicas = getattr(self._compute, "replicas", None) or 1

            # Configure for local mode (no SPMD) and headless service for discovery
            self._compute = self._compute.distribute(
                distribution_type="local",
                workers=replicas,
            )

            self._gateway_module = kt.cls(MonarchGateway, name=self._name).to(
                self._compute,
            )
            self._service_url = self._gateway_module.base_endpoint

        else:
            # Pre-allocated - use selector to find existing pods
            # For pre-allocated pods, we assume they already have Monarch workers running
            # Create a Compute with just the selector
            self._compute = kt.Compute(selector=self._selector)
            self._compute.allowed_serialization = ["json", "pickle"]
            self._compute = self._compute.distribute(distribution_type="local")

            self._gateway_module = kt.cls(MonarchGateway, name=self._name).to(
                self._compute,
            )
            self._service_url = self._gateway_module.base_endpoint

        # Get the actual service name (may have username prefix)
        actual_service_name = self._gateway_module.service_name
        logger.info(f"Gateway deployed at: {self._service_url} (service: {actual_service_name})")

        # Establish connection to gateway
        self._gateway_connection = GatewayConnection(
            self._service_url,
            use_websocket=self._use_websocket,
        )

        # Initialize the gateway (connects to Monarch workers)
        # Use the actual service name for the headless DNS lookup
        headless_dns = f"{actual_service_name}-headless.{self._namespace}.svc.cluster.local"
        init_result = self._gateway_connection.call(
            "initialize",
            headless_service_dns=headless_dns,
            monarch_port=self._monarch_port,
        )
        logger.info(f"Gateway initialized: {init_result}")

        self._applied = True

    def state(self) -> "JobState":
        """
        Get the job state with HostMesh proxies.

        Returns:
            JobState with 'workers' attribute containing the HostMeshProxy
        """
        from kubetorch.monarch.proxy import HostMeshProxy, JobState

        self.apply()

        # Get gateway status for shape info
        status = self._gateway_connection.call("get_status")

        # Create host mesh proxy
        # For now, use a simple shape based on the number of workers
        # In the future, we could query the actual shape from the gateway
        host_mesh = HostMeshProxy(
            host_mesh_id="hm_default",
            shape={"hosts": status.get("num_workers", 1)},
            gateway=self._gateway_connection,
        )

        return JobState({"workers": host_mesh})

    def kill(self):
        """Kill the job and clean up resources."""
        if self._gateway_connection:
            try:
                self._gateway_connection.call("shutdown")
            except Exception as e:
                logger.warning(f"Error during gateway shutdown: {e}")

            self._gateway_connection.close()
            self._gateway_connection = None

        # TODO: Tear down K8s resources if we created them
        self._applied = False
        logger.info(f"KubernetesJob {self._name} killed")

    def __enter__(self):
        """Context manager entry."""
        self.apply()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.kill()
        return False

    @property
    def service_url(self) -> Optional[str]:
        """Get the service URL for the gateway."""
        return self._service_url

    @property
    def name(self) -> str:
        """Get the job name."""
        return self._name
