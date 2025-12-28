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
from typing import Any, Dict, Literal, Optional

import requests

from kubetorch.config import KubetorchConfig
from kubetorch.logger import get_logger
from kubetorch.serving.constants import (
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
        interval: Time between two consecutive metrics outputs, in seconds. Default: 30.
        scope: Metrics aggregation level. Options: "pod", "resource". Default: "resource"
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
        stream_logs: Whether log streaming is enabled for this service. When True, logs
            from the remote compute are streamed back to the client during calls and
            service startup. Individual calls can override this with stream_logs=False.
            If None, falls back to global config.stream_logs setting. Default: True
        level: Log level for the remote service. Controls which logs are emitted by
            the service and available for streaming. Also controls client-side filtering.
            Options: "debug", "info", "warning", "error". Default: "info"
        include_system_logs: Whether to include framework logs (e.g., uvicorn.access).
            Default: False (only show application logs)
        include_events: Whether to include Kubernetes events during service startup.
            Events include pod scheduling, image pulling, container starting, etc.
            Default: True
        grace_period: Seconds to continue streaming after request completes, to catch
            any final logs that arrive late. Default: 2.0
        include_name: Whether to prepend pod/service name to each log line.
            Default: True
        poll_timeout: Timeout in seconds for WebSocket receive during normal streaming.
            Default: 1.0
        grace_poll_timeout: Timeout in seconds for WebSocket receive during grace period.
            Shorter timeout allows faster shutdown while still catching late logs.
            Default: 0.5
        shutdown_grace_period: Seconds to block the main thread after the HTTP call
            completes, waiting for the log streaming thread to finish. This prevents
            the Python interpreter from exiting before final logs are printed.
            Set to 0 for no blocking (default), or a few seconds (e.g., 3.0) if you
            need to ensure wrap-up logs from the remote compute are captured.
            Default: 0
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
        mode: Debug mode - "pdb" (default, WebSocket PTY) or "pdb-ui" (web-based UI)
        port: Debug port (default: 5678)
    """

    mode: Literal["pdb", "pdb-ui"] = "pdb"
    port: int = 5678  # DEFAULT_DEBUG_PORT


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
    from kubetorch.resources.compute.utils import find_available_port
    from kubetorch.serving.utils import wait_for_port_forward

    # Fast path: check without lock first
    h = _port_forwards.get(service_name)
    if h and h.process.poll() is None:
        return h

    # Slow path: need to create port forward
    with _pf_lock:
        # Double-check pattern: check again inside the lock
        h = _port_forwards.get(service_name)
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
        _port_forwards[service_name] = h
        return h


async def _ensure_pf_async(service_name: str, namespace: str, remote_port: int, health_endpoint: str) -> PFHandle:
    """Async version of _ensure_pf for use in async contexts."""
    from kubetorch.resources.compute.utils import find_available_port
    from kubetorch.serving.utils import wait_for_port_forward

    # Fast path: check without lock first
    h = _port_forwards.get(service_name)
    if h and h.process.poll() is None:
        return h

    # Ensure async lock is created (lazy initialization)
    global _pf_async_lock
    if _pf_async_lock is None:
        _pf_async_lock = asyncio.Lock()

    # Slow path: need to create port forward
    async with _pf_async_lock:
        # Double-check pattern: check again inside the lock
        h = _port_forwards.get(service_name)
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
        _port_forwards[service_name] = h
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
    from kubetorch.servers.http.utils import is_running_in_kubernetes

    if is_running_in_kubernetes():
        return f"http://{service_name}.{namespace}.svc.cluster.local:{remote_port}"

    # Ingress URL into the cluster from outside
    if config.api_url:
        return config.api_url

    h = _ensure_pf(service_name, namespace, remote_port, health_endpoint)

    # if the process died between creation and use, recreate once
    if h.process.poll() is not None:
        with _pf_lock:
            _port_forwards.pop(service_name, None)
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
    from kubetorch.servers.http.utils import is_running_in_kubernetes

    if is_running_in_kubernetes():
        return f"http://{service_name}.{namespace}.svc.cluster.local:{remote_port}"

    h = await _ensure_pf_async(service_name, namespace, remote_port, health_endpoint)

    # if the process died between creation and use, recreate once
    if h.process.poll() is not None:
        # Ensure async lock is created
        global _pf_async_lock
        if _pf_async_lock is None:
            _pf_async_lock = asyncio.Lock()

        async with _pf_async_lock:
            _port_forwards.pop(service_name, None)
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
            base_url: Base URL for the controller (e.g., "http://localhost:8080")
        """
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _request(self, method: str, path: str, ignore_not_found=False, **kwargs) -> requests.Response:
        """Make HTTP request to controller.

        Retries connection errors and controller unavailability (502/503).
        The controller already retries K8s API errors (429, 500, 504).
        """
        from kubetorch import ControllerRequestError

        url = f"{self.base_url}{path}"

        # Retry connection errors and controller unavailability
        max_attempts = 5
        base_delay = 0.5  # seconds

        for attempt in range(1, max_attempts + 1):
            try:
                response = self.session.request(method, url, **kwargs)

                try:
                    response.raise_for_status()
                except requests.HTTPError as e:
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

            except (requests.ConnectionError, requests.Timeout) as e:
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
        """GET request to controller"""
        response = self._request("GET", path, ignore_not_found=ignore_not_found, **kwargs)
        if response is None:
            return None
        return response.json()

    def post(self, path: str, json: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """POST request to controller"""
        response = self._request("POST", path, json=json, **kwargs)
        return response.json()

    def delete(self, path: str, ignore_not_found=False, **kwargs) -> Dict[str, Any]:
        """DELETE request to controller"""
        response = self._request("DELETE", path, ignore_not_found=ignore_not_found, **kwargs)
        if response is None:
            return None
        return response.json()

    def patch(self, path: str, json: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """PATCH request to controller"""
        response = self._request("PATCH", path, json=json, **kwargs)
        return response.json()

    # PersistentVolumeClaims
    def create_pvc(self, namespace: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Create a PersistentVolumeClaim"""
        return self.post(f"/api/v1/namespaces/{namespace}/persistentvolumeclaims", json=body)

    def get_pvc(self, namespace: str, name: str, ignore_not_found=False) -> Dict[str, Any]:
        """Get a PersistentVolumeClaim"""
        return self.get(
            f"/api/v1/namespaces/{namespace}/persistentvolumeclaims/{name}", ignore_not_found=ignore_not_found
        )

    def delete_pvc(self, namespace: str, name: str) -> Dict[str, Any]:
        """Delete a PersistentVolumeClaim"""
        return self.delete(f"/api/v1/namespaces/{namespace}/persistentvolumeclaims/{name}", ignore_not_found=True)

    def list_pvcs(self, namespace: str, label_selector: Optional[str] = None) -> Dict[str, Any]:
        """List PersistentVolumeClaims"""
        params = {"label_selector": label_selector} if label_selector else {}
        return self.get(f"/api/v1/namespaces/{namespace}/persistentvolumeclaims", params=params)

    # Services
    def create_service(self, namespace: str, body: Dict[str, Any], params: Dict = None) -> Dict[str, Any]:
        """Create a Service"""
        return self.post(f"/api/v1/namespaces/{namespace}/services", json=body, params=params)

    def get_service(self, namespace: str, name: str, ignore_not_found=False) -> Dict[str, Any]:
        """Get a Service"""
        return self.get(f"/api/v1/namespaces/{namespace}/services/{name}", ignore_not_found=ignore_not_found)

    def delete_service(self, namespace: str, name: str, ignore_not_found=False) -> Dict[str, Any]:
        """Delete a Service"""
        return self.delete(f"/api/v1/namespaces/{namespace}/services/{name}", ignore_not_found=ignore_not_found)

    def list_services(self, namespace: str, label_selector: Optional[str] = None) -> Dict[str, Any]:
        """List Services"""
        params = {"label_selector": label_selector} if label_selector else {}
        return self.get(f"/api/v1/namespaces/{namespace}/services", params=params)

    # Deployments
    def create_deployment(self, namespace: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Create a Deployment"""
        return self.post(f"/apis/apps/v1/namespaces/{namespace}/deployments", json=body)

    def get_deployment(self, namespace: str, name: str, ignore_not_found=False) -> Dict[str, Any]:
        """Get a Deployment"""
        return self.get(f"/apis/apps/v1/namespaces/{namespace}/deployments/{name}", ignore_not_found=ignore_not_found)

    def delete_deployment(self, namespace: str, name: str) -> Dict[str, Any]:
        """Delete a Deployment"""
        return self.delete(f"/apis/apps/v1/namespaces/{namespace}/deployments/{name}", ignore_not_found=True)

    def patch_deployment(self, namespace: str, name: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Patch a Deployment"""
        return self.patch(f"/apis/apps/v1/namespaces/{namespace}/deployments/{name}", json=body)

    def list_deployments(self, namespace: str, label_selector: Optional[str] = None) -> Dict[str, Any]:
        """List Deployments"""
        params = {"label_selector": label_selector} if label_selector else {}
        return self.get(f"/apis/apps/v1/namespaces/{namespace}/deployments", params=params)

    # Endpoints
    def get_endpoints(self, namespace: str, name: str) -> Dict[str, Any]:
        """Get Endpoints for a Service"""
        return self.get(f"/api/v1/namespaces/{namespace}/endpoints/{name}")

    # Secrets
    def create_secret(self, namespace: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Create a Secret"""
        return self.post(f"/api/v1/namespaces/{namespace}/secrets", json=body)

    def get_secret(self, namespace: str, name: str, ignore_not_found=False) -> Dict[str, Any]:
        """Get a Secret"""
        return self.get(f"/api/v1/namespaces/{namespace}/secrets/{name}", ignore_not_found=ignore_not_found)

    def patch_secret(self, namespace: str, name: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Patch a Secret"""
        return self.patch(f"/api/v1/namespaces/{namespace}/secrets/{name}", json=body)

    def list_secrets(self, namespace: str, label_selector: Optional[str] = None) -> Dict[str, Any]:
        """List Secrets"""
        params = {"label_selector": label_selector} if label_selector else {}
        return self.get(f"/api/v1/namespaces/{namespace}/secrets", params=params)

    def delete_secret(self, namespace: str, name: str) -> Dict[str, Any]:
        """Delete a Secret"""
        return self.delete(f"/api/v1/namespaces/{namespace}/secrets/{name}", ignore_not_found=True)

    def list_secrets_all_namespaces(self, label_selector: Optional[str] = None) -> Dict[str, Any]:
        """List Secrets across all namespaces"""
        params = {"label_selector": label_selector} if label_selector else {}
        return self.get("/api/v1/secrets", params=params)

    # Pods
    def list_pods(self, namespace: str, label_selector: Optional[str] = None) -> Dict[str, Any]:
        """List Pods"""
        params = {"label_selector": label_selector} if label_selector else {}
        return self.get(f"/api/v1/namespaces/{namespace}/pods", params=params)

    def get_pod(self, namespace: str, name: str) -> Dict[str, Any]:
        """Get a Pod"""
        return self.get(f"/api/v1/namespaces/{namespace}/pods/{name}")

    def delete_pod(
        self,
        namespace: str,
        name: str,
        grace_period_seconds: Optional[int] = None,
        propagation_policy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delete a Pod"""
        params = {}
        if grace_period_seconds is not None:
            params["grace_period_seconds"] = str(grace_period_seconds)
        if propagation_policy is not None:
            params["propagation_policy"] = propagation_policy
        return self.delete(f"/api/v1/namespaces/{namespace}/pods/{name}", params=params, ignore_not_found=True)

    def get_pod_logs(
        self, namespace: str, name: str, container: Optional[str] = None, tail_lines: Optional[int] = None
    ) -> str:
        """Get Pod logs"""
        params = {}
        if container:
            params["container"] = container
        if tail_lines:
            params["tailLines"] = str(tail_lines)

        url = f"{self.base_url}/api/v1/namespaces/{namespace}/pods/{name}/log"
        try:
            response = self.session.request("GET", url, params=params)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"GET {url} - {e}")
            raise

    # Namespaces
    def get_namespace(self, name: str) -> Dict[str, Any]:
        """Get a Namespace"""
        return self.get(f"/api/v1/namespaces/{name}")

    def list_namespaces(self) -> Dict[str, Any]:
        """List Namespaces"""
        return self.get("/api/v1/namespaces")

    # Nodes
    def list_nodes(self, label_selector: Optional[str] = None) -> Dict[str, Any]:
        """List Nodes"""
        params = {"label_selector": label_selector} if label_selector else {}
        return self.get("/api/v1/nodes", params=params)

    def get_node(self, name: str) -> Dict[str, Any]:
        """Get a Node"""
        return self.get(f"/api/v1/nodes/{name}")

    # StorageClasses
    def list_storage_classes(self) -> Dict[str, Any]:
        """List StorageClasses"""
        return self.get("/apis/storage.k8s.io/v1/storageclasses")

    def get_storage_class(self, name: str) -> Dict[str, Any]:
        """Get a StorageClass"""
        return self.get(f"/apis/storage.k8s.io/v1/storageclasses/{name}")

    # Events
    def list_events(self, namespace: str, field_selector: Optional[str] = None) -> Dict[str, Any]:
        """List Kubernetes Events via controller."""
        params = {"field_selector": field_selector} if field_selector else {}
        return self.get(f"/api/v1/namespaces/{namespace}/events", params=params)

    # ConfigMaps
    def list_config_maps(self, namespace: str, label_selector: Optional[str] = None) -> Dict[str, Any]:
        """List ConfigMaps"""
        params = {"label_selector": label_selector} if label_selector else {}
        return self.get(f"/api/v1/namespaces/{namespace}/configmaps", params=params)

    def get_config_map(self, namespace: str, name: str) -> Dict[str, Any]:
        """Get a ConfigMap"""
        return self.get(f"/api/v1/namespaces/{namespace}/configmaps/{name}")

    def delete_config_map(
        self,
        namespace: str,
        name: str,
        grace_period_seconds: Optional[int] = None,
        propagation_policy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delete a ConfigMap"""
        params = {}
        if grace_period_seconds is not None:
            params["grace_period_seconds"] = str(grace_period_seconds)
        if propagation_policy:
            params["propagation_policy"] = propagation_policy
        return self.delete(f"/api/v1/namespaces/{namespace}/configmaps/{name}", params=params)

    # Custom Resource Definitions (CRDs)
    def create_namespaced_custom_object(
        self, group: str, version: str, namespace: str, plural: str, body: Dict[str, Any], params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create a custom resource"""
        return self.post(f"/apis/{group}/{version}/namespaces/{namespace}/{plural}", json=body, params=params)

    def get_namespaced_custom_object(
        self, group: str, version: str, namespace: str, plural: str, name: str, ignore_not_found=False
    ) -> Dict[str, Any]:
        """Get a custom resource"""
        return self.get(
            f"/apis/{group}/{version}/namespaces/{namespace}/{plural}/{name}", ignore_not_found=ignore_not_found
        )

    def patch_namespaced_custom_object(
        self, group: str, version: str, namespace: str, plural: str, name: str, body: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Patch a custom resource"""
        return self.patch(f"/apis/{group}/{version}/namespaces/{namespace}/{plural}/{name}", json=body)

    def delete_namespaced_custom_object(
        self,
        group: str,
        version: str,
        namespace: str,
        plural: str,
        name: str,
        grace_period_seconds: Optional[int] = None,
        propagation_policy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delete a custom resource"""
        params = {}
        if grace_period_seconds is not None:
            params["grace_period_seconds"] = str(grace_period_seconds)
        if propagation_policy:
            params["propagation_policy"] = propagation_policy
        return self.delete(f"/apis/{group}/{version}/namespaces/{namespace}/{plural}/{name}", params=params)

    def list_namespaced_custom_object(
        self,
        group: str,
        version: str,
        namespace: str,
        plural: str,
        label_selector: Optional[str] = None,
        ignore_not_found=False,
    ) -> Dict[str, Any]:
        """List custom resources in a namespace"""
        params = {"label_selector": label_selector} if label_selector else {}
        return self.get(
            f"/apis/{group}/{version}/namespaces/{namespace}/{plural}", params=params, ignore_not_found=ignore_not_found
        )

    def list_ingresses(self, namespace: str, label_selector: str = None):
        params = {"label_selector": label_selector} if label_selector else {}
        return self.get(f"/apis/networking.k8s.io/v1/namespaces/{namespace}/ingresses", params=params)

    # ReplicaSets
    def list_namespaced_replica_set(self, namespace: str, label_selector: Optional[str] = None) -> Dict[str, Any]:
        """List ReplicaSets in a namespace."""
        params = {"label_selector": label_selector} if label_selector else {}
        return self.get(f"/apis/apps/v1/namespaces/{namespace}/replicasets", params=params)

    def get_namespaced_replica_set(self, namespace: str, name: str) -> Dict[str, Any]:
        """Get a ReplicaSet"""
        return self.get(f"/apis/apps/v1/namespaces/{namespace}/replicasets/{name}")

    def delete_namespaced_replica_set(
        self,
        namespace: str,
        name: str,
        grace_period_seconds: Optional[int] = None,
        propagation_policy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delete a ReplicaSet"""
        params = {}
        if grace_period_seconds is not None:
            params["grace_period_seconds"] = str(grace_period_seconds)
        if propagation_policy:
            params["propagation_policy"] = propagation_policy
        return self.delete(f"/apis/apps/v1/namespaces/{namespace}/replicasets/{name}", params=params)

    def list_cluster_custom_object(
        self,
        group: str,
        version: str,
        plural: str,
        label_selector: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List cluster-scoped custom resources"""
        params = {"label_selector": label_selector} if label_selector else {}
        return self.get(f"/apis/{group}/{version}/{plural}", params=params)


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


# added a comment to trigger CI
