import asyncio
import atexit
import os
import signal
import socket
import subprocess
import threading
import time

from dataclasses import dataclass
from functools import cache
from typing import Any, Dict, List, Literal, Optional, Union

import httpx

from kubetorch.config import KubetorchConfig
from kubetorch.constants import (
    CONTROLLER_CONNECT_TIMEOUT,
    CONTROLLER_POOL_TIMEOUT,
    CONTROLLER_READ_TIMEOUT,
    CONTROLLER_WRITE_TIMEOUT,
)
from kubetorch.logger import get_logger
from kubetorch.provisioning.constants import (
    DEFAULT_NGINX_HEALTH_ENDPOINT,
    DEFAULT_NGINX_PORT,
    KUBETORCH_CONTROLLER,
    LOCAL_NGINX_PORT,
)

logger = get_logger(__name__)

# For use in `kt deploy` decorators
disable_decorators = False

config = KubetorchConfig()


@dataclass
class MetricsConfig:
    """
    Configuration for streaming metrics during a Kubetorch service call.

    Attributes:
        interval (int): Time between two consecutive metrics outputs, in seconds. (Default: 30)
        scope (str): Metrics aggregation level. Options: "pod", "resource". (Default: "resource")
    """

    interval: int = 30  # polling interval in seconds
    scope: Literal["pod", "resource"] = "resource"  # aggregation level (default to "resource")


@dataclass
class LoggingConfig:
    """Configuration for logging behavior on a Kubetorch service.

    This config is set at the Compute level and applies to all calls made to that service.
    It controls both runtime log streaming (during method calls) and startup log streaming
    (during `.to()` deployment).

    Attributes:
        stream_logs (bool): Whether log streaming is enabled for this service. When ``True``, logs
            from the remote compute are streamed back to the client during calls and
            service startup. Individual calls can override this with ``stream_logs=False``.
            If None, falls back to global config.stream_logs setting. (Default: True)
        level (str): Log level for the remote service. Controls which logs are emitted by the service and available
            for streaming. Also controls client-side filtering. Options: "debug", "info", "warning", "error".
            (Default: "info")
        include_system_logs (bool): Whether to include framework logs (e.g., uvicorn.access). (Default: False)
        include_events (bool): Whether to include Kubernetes events during service startup.
            Events include pod scheduling, image pulling, container starting, etc. (Default: True)
        grace_period (float): Seconds to continue streaming after request completes, to catch
            any final logs that arrive late. (Default: 2.0)
        include_name (bool): Whether to prepend pod/service name to each log line. (Default: True)
        poll_timeout (float): Timeout in seconds for WebSocket receive during normal streaming. (Default: 1.0)
        grace_poll_timeout (float): Timeout in seconds for WebSocket receive during grace period.
            Shorter timeout allows faster shutdown while still catching late logs. (Default: 0.5)
        shutdown_grace_period (float): Seconds to block the main thread after the HTTP call
            completes, waiting for the log streaming thread to finish. This prevents
            the Python interpreter from exiting before final logs are printed.
            Set to 0 for no blocking (default), or a few seconds (e.g., 3.0) if you
            need to ensure wrap-up logs from the remote compute are captured.
            (Default: 0)
    """

    stream_logs: bool = None
    level: Literal["debug", "info", "warning", "error"] = "info"
    include_system_logs: bool = False
    include_events: bool = True
    grace_period: float = 2.0
    include_name: bool = True
    poll_timeout: float = 1.0
    grace_poll_timeout: float = 0.5
    shutdown_grace_period: float = 0


@dataclass
class DebugConfig:
    """Configuration for debugging mode.

    Attributes:
        mode (str): Debug mode - "pdb" (WebSocket PTY) or "pdb-ui" (web-based UI). Options: "pdb", "pdb-ui".
            (Default: "pdb")
        port (int): Debug port. (Default: 5678)
    """

    mode: Literal["pdb", "pdb-ui"] = "pdb"
    port: int = 5678  # DEFAULT_DEBUG_PORT

    def to_dict(self):
        return {"mode": self.mode, "port": self.port}


@dataclass(frozen=True)
class PFHandle:
    process: subprocess.Popen
    port: int
    base_url: str  # "http://localhost:<port>"


# cache a single pf per service (currently just a single NGINX proxy)
_port_forwards: Dict[str, PFHandle] = {}
# Use both a threading lock and an asyncio lock for different contexts
_pf_lock = threading.Lock()
# Async lock must be created lazily when event loop is available
_pf_async_lock: Optional[asyncio.Lock] = None


def _kill(proc: Optional[subprocess.Popen]) -> None:
    if not proc:
        return
    try:
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=3)
    except Exception:
        pass


def _cleanup_port_forwards():
    with _pf_lock:
        for h in list(_port_forwards.values()):
            _kill(h.process)
        _port_forwards.clear()


def _ensure_pf(service_name: str, namespace: str, remote_port: int, health_endpoint: str) -> PFHandle:
    from kubetorch.provisioning.utils import wait_for_port_forward
    from kubetorch.resources.compute.utils import find_available_port

    # Cache key includes port to support multiple ports per service
    cache_key = f"{service_name}:{remote_port}"

    # Fast path: check without lock first
    h = _port_forwards.get(cache_key)
    if h and h.process.poll() is None:
        return h

    # Slow path: need to create port forward
    with _pf_lock:
        # Double-check pattern: check again inside the lock
        h = _port_forwards.get(cache_key)
        if h and h.process.poll() is None:
            return h

        # Now create the port forward while holding the lock
        # This ensures only one thread creates the port forward
        local_port = find_available_port(LOCAL_NGINX_PORT)

        cmd = [
            "kubectl",
            "port-forward",
            f"svc/{service_name}",
            f"{local_port}:{remote_port}",
            "--namespace",
            namespace,
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)

        # If it dies immediately, surface stderr (much clearer than a generic timeout)
        time.sleep(0.3)

        if proc.poll() is not None:
            err = (proc.stderr.read() or b"").decode(errors="ignore")
            raise RuntimeError(f"kubectl port-forward exited (rc={proc.returncode}): {err.strip()}")

        if health_endpoint:
            cluster_config = wait_for_port_forward(proc, local_port, health_endpoint=health_endpoint)
            if isinstance(cluster_config, dict):
                config.cluster_config = cluster_config
                config.write(values={"cluster_config": cluster_config})
        else:
            # Minimal TCP wait (no HTTP probe)
            deadline = time.time() + 10
            ok = False
            while time.time() < deadline:
                try:
                    with socket.create_connection(("127.0.0.1", local_port), timeout=0.5):
                        ok = True
                        break
                except OSError:
                    time.sleep(0.1)
            if not ok:
                raise TimeoutError("Timeout waiting for port forward to be ready")

        time.sleep(0.2)  # tiny grace

        h = PFHandle(process=proc, port=local_port, base_url=f"http://localhost:{local_port}")
        # Store in cache while still holding the lock
        _port_forwards[cache_key] = h
        return h


async def _ensure_pf_async(service_name: str, namespace: str, remote_port: int, health_endpoint: str) -> PFHandle:
    """Async version of _ensure_pf for use in async contexts."""
    from kubetorch.provisioning.utils import wait_for_port_forward
    from kubetorch.resources.compute.utils import find_available_port

    # Cache key includes port to support multiple ports per service
    cache_key = f"{service_name}:{remote_port}"

    # Fast path: check without lock first
    h = _port_forwards.get(cache_key)
    if h and h.process.poll() is None:
        return h

    # Ensure async lock is created (lazy initialization)
    global _pf_async_lock
    if _pf_async_lock is None:
        _pf_async_lock = asyncio.Lock()

    # Slow path: need to create port forward
    async with _pf_async_lock:
        # Double-check pattern: check again inside the lock
        h = _port_forwards.get(cache_key)
        if h and h.process.poll() is None:
            return h

        # Create port forward in a thread to avoid blocking the event loop
        def create_port_forward():
            local_port = find_available_port(LOCAL_NGINX_PORT)

            cmd = [
                "kubectl",
                "port-forward",
                f"svc/{service_name}",
                f"{local_port}:{remote_port}",
                "--namespace",
                namespace,
            ]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )

            # If it dies immediately, surface stderr
            time.sleep(0.3)

            if proc.poll() is not None:
                err = (proc.stderr.read() or b"").decode(errors="ignore")
                raise RuntimeError(f"kubectl port-forward exited (rc={proc.returncode}): {err.strip()}")

            if health_endpoint:
                wait_for_port_forward(proc, local_port, health_endpoint=health_endpoint)
            else:
                # Minimal TCP wait (no HTTP probe)
                deadline = time.time() + 10
                ok = False
                while time.time() < deadline:
                    try:
                        with socket.create_connection(("127.0.0.1", local_port), timeout=0.5):
                            ok = True
                            break
                    except OSError:
                        time.sleep(0.1)
                if not ok:
                    raise TimeoutError("Timeout waiting for port forward to be ready")

            time.sleep(0.2)  # tiny grace

            return PFHandle(process=proc, port=local_port, base_url=f"http://localhost:{local_port}")

        # Run the blocking operation in a thread
        loop = asyncio.get_event_loop()
        h = await loop.run_in_executor(None, create_port_forward)

        # Store in cache while still holding the lock
        _port_forwards[cache_key] = h
        return h


def service_url(
    service_name: str = KUBETORCH_CONTROLLER,
    namespace: str = config.install_namespace,
    remote_port: int = DEFAULT_NGINX_PORT,
    health_endpoint: str = DEFAULT_NGINX_HEALTH_ENDPOINT,
) -> str:
    """
    Return a URL to reach a Kubernetes Service.
    - If running in-cluster:  {scheme}://{svc}.{ns}.svc.cluster.local:{remote_port}{path}
    - Else: ensure a single kubectl port-forward (cached) and return http://localhost:<port>{path}
    """
    from kubetorch.serving.utils import is_running_in_kubernetes

    if is_running_in_kubernetes():
        return f"http://{service_name}.{namespace}.svc.cluster.local:{remote_port}"

    # Ingress URL into the cluster from outside
    if config.api_url:
        return config.api_url

    h = _ensure_pf(service_name, namespace, remote_port, health_endpoint)

    # if the process died between creation and use, recreate once
    if h.process.poll() is not None:
        cache_key = f"{service_name}:{remote_port}"
        with _pf_lock:
            _port_forwards.pop(cache_key, None)
        h = _ensure_pf(service_name, namespace, remote_port, health_endpoint)
    return h.base_url


async def service_url_async(
    service_name: str = KUBETORCH_CONTROLLER,
    namespace: str = config.install_namespace,
    remote_port: int = DEFAULT_NGINX_PORT,
    health_endpoint: str = DEFAULT_NGINX_HEALTH_ENDPOINT,
) -> str:
    """
    Async version of service_url for use in async contexts.
    Return a URL to reach a Kubernetes Service.
    - If running in-cluster:  {scheme}://{svc}.{ns}.svc.cluster.local:{remote_port}{path}
    - Else: ensure a single kubectl port-forward (cached) and return http://localhost:<port>{path}
    """
    from kubetorch.serving.utils import is_running_in_kubernetes

    if is_running_in_kubernetes():
        return f"http://{service_name}.{namespace}.svc.cluster.local:{remote_port}"

    h = await _ensure_pf_async(service_name, namespace, remote_port, health_endpoint)

    # if the process died between creation and use, recreate once
    if h.process.poll() is not None:
        cache_key = f"{service_name}:{remote_port}"
        # Ensure async lock is created
        global _pf_async_lock
        if _pf_async_lock is None:
            _pf_async_lock = asyncio.Lock()

        async with _pf_async_lock:
            _port_forwards.pop(cache_key, None)
        h = await _ensure_pf_async(service_name, namespace, remote_port, health_endpoint)
    return h.base_url


atexit.register(_cleanup_port_forwards)


# ----------------------------------------------------------------------
# Controller Client
# ----------------------------------------------------------------------
class ControllerClient:
    """
    HTTP client for Kubetorch Controller API.

    This client replaces direct Kubernetes API calls with HTTP requests to the controller.
    The controller acts as a proxy that handles authentication and routing to the K8s API.
    """

    def __init__(self, base_url: str):
        """
        Initialize controller client.

        Args:
            base_url (str): Base URL for the controller (e.g., "http://localhost:8080")
        """
        self.base_url = base_url.rstrip("/")
        # Quick operations (e.g.: list, basic get) that should use a timeout
        self._read_timeout = httpx.Timeout(
            connect=CONTROLLER_CONNECT_TIMEOUT,
            read=CONTROLLER_READ_TIMEOUT,
            write=CONTROLLER_WRITE_TIMEOUT,
            pool=CONTROLLER_POOL_TIMEOUT,
        )

        # No default timeout - long-running operations (e.g.: deploy, check-ready) should not time out
        self.session = httpx.Client(headers={"Content-Type": "application/json"}, timeout=None)

    def _request(self, method: str, path: str, ignore_not_found=False, timeout=None, **kwargs) -> httpx.Response:
        """Make HTTP request to controller.

        Retries connection errors and controller unavailability (502/503).
        The controller already retries K8s API errors (429, 500, 504).
        """
        from kubetorch import ControllerRequestError

        if timeout is not None:
            kwargs["timeout"] = timeout

        url = f"{self.base_url}{path}"

        # Retry connection errors and controller unavailability
        max_attempts = 5
        base_delay = 0.5  # seconds

        for attempt in range(1, max_attempts + 1):
            try:
                response = self.session.request(method, url, **kwargs)

                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    status = response.status_code
                    if status == 404 and ignore_not_found:
                        return None
                    if status == 204:
                        return None

                    # Try to extract detailed error message from response body
                    error_message = None
                    try:
                        import json

                        error_body = response.json()

                        # Handle nested JSON in 'detail' field (controller may wrap K8s errors)
                        if isinstance(error_body, dict) and "detail" in error_body:
                            detail = error_body["detail"]
                            # Try to parse detail if it's a JSON string
                            if isinstance(detail, str):
                                try:
                                    error_body = json.loads(detail)
                                except (json.JSONDecodeError, ValueError):
                                    error_message = detail

                        if not error_message and isinstance(error_body, dict) and "message" in error_body:
                            error_message = error_body["message"]

                    except Exception:
                        pass

                    # Fallback to response text or generic message
                    if not error_message:
                        error_message = (
                            response.text[:200]
                            if response.text
                            else f"Request failed with status {response.status_code}"
                        )

                    # Retry 502/503 errors (controller unavailable/overloaded, not K8s API errors)
                    if status in {502, 503} and attempt < max_attempts:
                        # Exponential backoff for overload scenarios
                        retry_delay = base_delay * attempt
                        time.sleep(retry_delay)
                        continue

                    # Log 404 at debug level (often expected - resource/CRD not found)
                    # Log 409 at debug level (often expected - resource/CRD already exists)
                    # Log other errors at error level
                    if status == 404 or status == 409:
                        logger.debug(f"{method} {url} returned {status}: {error_message}")
                    else:
                        logger.error(f"{method} {url} failed with status {response.status_code}: {error_message}")

                    # Don't retry other HTTP errors - controller already retried K8s API
                    raise ControllerRequestError(
                        method=method, url=url, status_code=response.status_code, message=error_message
                    ) from e

                return response

            except ControllerRequestError:
                # Don't retry HTTP errors (except 502/503 handled above)
                raise

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                # Retry connection errors (controller pod down/restarting)
                if attempt < max_attempts:
                    retry_delay = base_delay * attempt
                    logger.warning(
                        f"Controller connection error on attempt {attempt}/{max_attempts}, "
                        f"retrying in {retry_delay:.1f}s: {e}"
                    )
                    time.sleep(retry_delay)
                    continue
                logger.error(f"{method} {url} - connection failed after {max_attempts} attempts: {e}")
                raise
            except Exception as e:
                logger.error(f"{method} {url} - {e}")
                raise

    def get(self, path: str, ignore_not_found=False, **kwargs) -> Dict[str, Any]:
        """GET request to controller. Defaults to read timeout, pass timeout=None for no timeout."""
        timeout = kwargs.pop("timeout", self._read_timeout)
        response = self._request("GET", path, ignore_not_found=ignore_not_found, timeout=timeout, **kwargs)
        if response is None:
            return None
        return response.json()

    def post(self, path: str, json: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """POST request to controller. Defaults to read timeout, pass timeout=None for no timeout."""
        timeout = kwargs.pop("timeout", self._read_timeout)
        response = self._request("POST", path, json=json, timeout=timeout, **kwargs)
        return response.json()

    def delete(self, path: str, ignore_not_found=False, **kwargs) -> Dict[str, Any]:
        """DELETE request to controller. Defaults to read timeout, pass timeout=None for no timeout."""
        timeout = kwargs.pop("timeout", self._read_timeout)
        response = self._request("DELETE", path, ignore_not_found=ignore_not_found, timeout=timeout, **kwargs)
        if response is None:
            return None
        return response.json()

    def patch(self, path: str, json: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """PATCH request to controller. Defaults to read timeout, pass timeout=None for no timeout."""
        timeout = kwargs.pop("timeout", self._read_timeout)
        response = self._request("PATCH", path, json=json, timeout=timeout, **kwargs)
        return response.json()

    # PersistentVolumeClaims (Volumes)
    def create_pvc(self, namespace: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Create a PersistentVolumeClaim."""
        return self.post(f"/controller/volumes/{namespace}", json=body)

    def get_pvc(self, namespace: str, name: str, ignore_not_found=False) -> Dict[str, Any]:
        """Get a PersistentVolumeClaim."""
        return self.get(f"/controller/volumes/{namespace}/{name}", ignore_not_found=ignore_not_found)

    def delete_pvc(self, namespace: str, name: str) -> Dict[str, Any]:
        """Delete a PersistentVolumeClaim."""
        return self.delete(f"/controller/volumes/{namespace}/{name}", ignore_not_found=True)

    def list_pvcs(self, namespace: str, label_selector: Optional[str] = None) -> Dict[str, Any]:
        """List PersistentVolumeClaims."""
        params = {"label_selector": label_selector} if label_selector else {}
        return self.get(f"/controller/volumes/{namespace}", params=params)

    # Services
    def create_service(self, namespace: str, body: Dict[str, Any], params: Dict = None) -> Dict[str, Any]:
        """Create a Service"""
        return self.post(f"/controller/services/{namespace}", json=body, params=params)

    def get_service(self, namespace: str, name: str, ignore_not_found=False) -> Dict[str, Any]:
        """Get a Service"""
        return self.get(f"/controller/services/{namespace}/{name}", ignore_not_found=ignore_not_found)

    def fetch_resources_for_teardown(
        self,
        namespace: str,
        name: Optional[Union[str, dict, list]] = None,
        prefix: Optional[bool] = None,
        teardown_all: Optional[bool] = None,
        username: Optional[str] = None,
        exact_match: Optional[bool] = None,
    ):
        """Fetch K8s resources that would be deleted by a teardown request."""
        body = {
            "namespace": namespace,
            "services": name,
            "prefix": prefix,
            "teardown_all": teardown_all,
            "username": username,
            "exact_match": exact_match,
        }
        body = {k: v for k, v in body.items() if v is not None}
        return self.get("/controller/teardown/list", json=body)

    def delete_services(
        self,
        namespace: str,
        services: Optional[Union[str, dict]] = None,
        force: Optional[bool] = None,
        prefix: Optional[bool] = None,
        teardown_all: Optional[bool] = None,
        username: Optional[str] = None,
        exact_match: Optional[bool] = None,
        ignore_not_found: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Delete K8s services and associated resources."""
        body = {
            "namespace": namespace,
            "services": services,
            "force": force,
            "prefix": prefix,
            "teardown_all": teardown_all,
            "username": username,
            "exact_match": exact_match,
        }
        body = {k: v for k, v in body.items() if v is not None and v != "" and v != []}
        return self.delete("/controller/teardown", ignore_not_found=ignore_not_found, json=body)

    # Deployments
    # NOTE: create_deployment removed - use deploy() or apply() instead

    def get_deployment(self, namespace: str, name: str, ignore_not_found=False) -> Dict[str, Any]:
        """Get a Deployment"""
        return self.get(f"/controller/deployments/{namespace}/{name}", ignore_not_found=ignore_not_found)

    # Secrets
    def create_secret(self, namespace: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Create a secret."""
        return self.post(f"/controller/secrets/{namespace}", json=body)

    def get_secret(self, namespace: str, name: str, ignore_not_found=False) -> Dict[str, Any]:
        """Get a secret."""
        return self.get(f"/controller/secrets/{namespace}/{name}", ignore_not_found=ignore_not_found)

    def patch_secret(self, namespace: str, name: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Patch a secret."""
        return self.patch(f"/controller/secrets/{namespace}/{name}", json=body)

    def list_secrets(self, namespace: str, label_selector: Optional[str] = None) -> Dict[str, Any]:
        """List secrets in a namespace."""
        params = {"label_selector": label_selector} if label_selector else {}
        return self.get(f"/controller/secrets/{namespace}", params=params)

    def delete_secret(self, namespace: str, name: str) -> Dict[str, Any]:
        """Delete a secret."""
        return self.delete(f"/controller/secrets/{namespace}/{name}", ignore_not_found=True)

    def list_secrets_all_namespaces(self, label_selector: Optional[str] = None) -> Dict[str, Any]:
        """List secrets across all namespaces."""
        params = {"label_selector": label_selector} if label_selector else {}
        return self.get("/controller/secrets", params=params)

    # Pods
    def list_pods(self, namespace: str, label_selector: Optional[str] = None) -> Dict[str, Any]:
        """List pods in a namespace."""
        params = {"label_selector": label_selector} if label_selector else {}
        return self.get(f"/controller/pods/{namespace}", params=params)

    def get_pod(self, namespace: str, name: str, ignore_not_found=False) -> Dict[str, Any]:
        """Get a specific pod."""
        return self.get(f"/controller/pods/{namespace}/{name}", ignore_not_found=ignore_not_found)

    def get_pod_logs(
        self, namespace: str, name: str, container: Optional[str] = None, tail_lines: Optional[int] = None
    ) -> str:
        """Get logs from a pod."""
        params = {}
        if container:
            params["container"] = container
        if tail_lines:
            params["tailLines"] = str(tail_lines)

        url = f"{self.base_url}/controller/pods/{namespace}/{name}/logs"
        try:
            response = self.session.request("GET", url, params=params)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"GET {url} - {e}")
            raise

    # Nodes
    def list_nodes(self, label_selector: Optional[str] = None) -> Dict[str, Any]:
        """List cluster nodes."""
        params = {"label_selector": label_selector} if label_selector else {}
        return self.get("/controller/nodes", params=params)

    # StorageClasses
    def list_storage_classes(self) -> Dict[str, Any]:
        """List available storage classes."""
        return self.get("/controller/storage-classes")

    # ConfigMaps
    def list_config_maps(self, namespace: str, label_selector: Optional[str] = None) -> Dict[str, Any]:
        """List ConfigMaps"""
        params = {"label_selector": label_selector} if label_selector else {}
        return self.get(f"/controller/configmaps/{namespace}", params=params)

    def list_ingresses(self, namespace: str, label_selector: str = None):
        params = {"label_selector": label_selector} if label_selector else {}
        return self.get(f"/controller/ingresses/{namespace}", params=params)

    def register_pool(
        self,
        name: str,
        namespace: str,
        specifier: Dict[str, Any],
        service: Optional[Dict[str, Any]] = None,
        dockerfile: Optional[str] = None,
        module: Optional[Dict[str, Any]] = None,
        pool_metadata: Optional[Dict[str, Any]] = None,
        server_port: int = 32300,
        labels: Optional[Dict[str, Any]] = None,
        annotations: Optional[Dict[str, Any]] = None,
        resource_kind: Optional[str] = None,
        resource_name: Optional[str] = None,
        create_headless_service: bool = False,
    ) -> Dict[str, Any]:
        """Register a compute pool via /controller/pool.

        A pool is a logical group of pods that calls can be directed to.
        This registers the pool in the controller and creates K8s Service(s)
        for label_selector pools, but does not create pods.

        Args:
            name (str): Unique identifier for the pool
            namespace (str): Kubernetes namespace
            specifier (dict, optional): How to track pods in the pool:
                - {"type": "label_selector", "selector": {"app": "workers"}}
            service (dict, optional): Optional service configuration:
                - {"url": "..."} - user-provided URL (e.g. Knative)
                - {"selector": {...}} - custom selector for routing
                - {"name": "..."} - custom service name
            dockerfile (str, optional): Optional dockerfile instructions for rebuilding workers
            module (dict, optional): Optional application to deploy onto a pool.
            pool_metadata (dict, optional): Optional metadata (username, etc.)
            server_port (int, optional): Port for the K8s service (default: 32300)
            labels (dict, optional): Labels for the K8s service
            annotations (dict, optional): Annotations for the K8s service
            resource_kind (str, optional): K8s resource kind for teardown (e.g., "Deployment", "PyTorchJob")
            resource_name (str, optional): K8s resource name for teardown (defaults to pool name)
            create_headless_service (bool, optional): Whether to create a headless service for distributed pod discovery

        Returns:
            Pool response with status, message, and service_url
        """
        body = {
            "name": name,
            "namespace": namespace,
            "specifier": specifier,
            "server_port": server_port,
        }
        args = {
            "service": service,
            "dockerfile": dockerfile,
            "module": module,
            "pool_metadata": pool_metadata,
            "labels": labels,
            "annotations": annotations,
            "create_headless_service": create_headless_service,
            "resource_kind": resource_kind,
            "resource_name": resource_name,
        }
        filtered_args = {k: v for k, v in args.items() if v not in (None, False)}
        body.update(filtered_args)

        return self.post("/controller/pool", json=body, timeout=None)

    def get_pool(self, namespace: str, name: str) -> Dict[str, Any]:
        """Get information about a registered pool."""
        return self.get(f"/controller/pool/{namespace}/{name}", ignore_not_found=True)

    def delete_pool(self, namespace: str, name: str) -> Dict[str, Any]:
        """Delete a registered pool and its associated K8s services."""
        return self.delete(f"/controller/pool/{namespace}/{name}", ignore_not_found=True)

    def apply(
        self,
        service_name: str,
        namespace: str,
        resource_type: str,
        resource_manifest: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply a K8s compute manifest via /controller/apply.

        This creates pods/workloads in the cluster by applying the provided manifest.
        It does not create K8s Services (use register_pool for that).

        Args:
            service_name (str): Name of the service.
            namespace (str): Kubernetes namespace.
            resource_type (str): Type of resource (deployment, knative, raycluster, etc.).
            resource_manifest (Dict[str, Any]): The full K8s manifest to apply.

        Returns:
            Apply response with status, message, and created resource.
        """
        body = {
            "service_name": service_name,
            "namespace": namespace,
            "resource_type": resource_type,
            "resource_manifest": resource_manifest,
        }
        return self.post("/controller/apply", json=body, timeout=None)

    def deploy(
        self,
        service_name: str,
        namespace: str,
        resource_type: str,
        resource_manifest: Dict[str, Any],
        specifier: Dict[str, Any],
        service: Optional[Dict[str, Any]] = None,
        dockerfile: Optional[str] = None,
        module: Optional[Dict[str, Any]] = None,
        pool_metadata: Optional[Dict[str, Any]] = None,
        server_port: int = 32300,
        labels: Optional[Dict[str, Any]] = None,
        annotations: Optional[Dict[str, Any]] = None,
        create_headless_service: bool = False,
    ) -> Dict[str, Any]:
        """Deploy K8s resource and register pool.

        Args:
            service_name (str): Name of the service.
            namespace (str): Kubernetes namespace.
            resource_type (str): Type of resource (deployment, knative, raycluster, etc.).
            resource_manifest (Dict[str, Any]): The full K8s manifest to apply.
            specifier (Dict[str, Any]): How to track pods in the pool
                (e.g., {"type": "label_selector", "selector": {...}}).
            service (Optional[Dict[str, Any]]): Service configuration for routing. (Default: None)
            dockerfile (Optional[str]): Dockerfile instructions for rebuilding workers. (Default: None)
            module (Optional[Dict[str, Any]]): Application to deploy onto a pool. (Default: None)
            pool_metadata (Optional[Dict[str, Any]]): Metadata (username, etc.). (Default: None)
            server_port (int): Port for the K8s service. (Default: 32300)
            labels (Optional[Dict[str, Any]]): Labels for the K8s service. (Default: None)
            annotations (Optional[Dict[str, Any]]): Annotations for the K8s service. (Default: None)
            create_headless_service (bool): Whether to create a headless service for
                distributed pod discovery. (Default: False)

        Returns:
            Deploy response with apply_status, pool_status, service_url, and created resource.
        """
        body = {
            "service_name": service_name,
            "namespace": namespace,
            "resource_type": resource_type,
            "resource_manifest": resource_manifest,
            "specifier": specifier,
            "server_port": server_port,
        }
        args = {
            "service": service,
            "dockerfile": dockerfile,
            "module": module,
            "pool_metadata": pool_metadata,
            "labels": labels,
            "annotations": annotations,
            "create_headless_service": create_headless_service,
        }
        filtered_args = {k: v for k, v in args.items() if v not in (None, False)}
        body.update(filtered_args)

        return self.post("/controller/deploy", json=body, timeout=None)

    def list_pools(self, namespace: str) -> Dict[str, Any]:
        """List all compute pools."""
        return self.get(f"/controller/pools/{namespace}")

    def get_connections(self) -> Dict[str, Any]:
        """Get WebSocket connection debug info (connected pods for each pool)."""
        return self.get("/controller/debug/connections")

    def discover_resources(
        self,
        namespace: str,
        label_selector: Optional[str] = None,
        name_filter: Optional[str] = None,
        prefix_filter: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Discover all kubetorch-managed resources in a namespace.

        Args:
            namespace (str): Kubernetes namespace to search.
            label_selector (Optional[str]): K8s label selector for server-side filtering
                (e.g., "kubetorch.com/username=xyz"). (Default: None)
            name_filter (Optional[str]): Filter by name substring. (Default: None)
            prefix_filter (Optional[str]): Filter by name prefix. (Default: None)

        Returns:
            Dict mapping resource type to list of resources:
            {
                "knative_services": [...],
                "deployments": [...],
                "rayclusters": [...],
                "training_jobs": [...],
                "pools": [...],
            }
        """
        params = {
            "label_selector": label_selector,
            "name_filter": name_filter,
            "prefix_filter": prefix_filter,
        }
        params = {k: v for k, v in params.items() if v}
        return self.get(f"/controller/discover/{namespace}", params=params or None)


@cache
def controller_client() -> ControllerClient:
    """
    Return the global ControllerClient instance.

    The controller client automatically handles:
    1. In-cluster: Uses cluster DNS to reach controller
    2. Out-of-cluster: Auto-creates port-forward to controller
    3. Explicit API URL: Uses config.api_url if set

    Note: This function is cached (@cache decorator) to reuse the same HTTP session across all requests, using
    the same instance on subsequent calls (thread-safe by default).
    """
    # Use service_url to get the base URL (handles in-cluster vs out-of-cluster)
    base_url = service_url(
        service_name=KUBETORCH_CONTROLLER,
        namespace=config.install_namespace,
        remote_port=DEFAULT_NGINX_PORT,
        health_endpoint=DEFAULT_NGINX_HEALTH_ENDPOINT,
    )
    return ControllerClient(base_url=base_url)
