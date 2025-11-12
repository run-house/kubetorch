import asyncio
import atexit
import os
import signal
import socket
import subprocess
import threading
import time

from dataclasses import dataclass
from typing import Dict, Literal, Optional

from kubetorch.config import KubetorchConfig
from kubetorch.serving.constants import (
    DEFAULT_NGINX_HEALTH_ENDPOINT,
    DEFAULT_NGINX_PORT,
    LOCAL_NGINX_PORT,
    NGINX_GATEWAY_PROXY,
)

# For use in `kt deploy` decorators
disable_decorators = False

config = KubetorchConfig()


@dataclass
class MetricsConfig:
    interval: int = 30  # polling interval in seconds
    scope: Literal["pod", "resource"] = "resource"  # aggregation level (default to "resource")


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
    service_name: str = NGINX_GATEWAY_PROXY,
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
    service_name: str = NGINX_GATEWAY_PROXY,
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
