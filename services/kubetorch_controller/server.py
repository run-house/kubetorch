"""
Kubetorch Controller - Central K8s API proxy for Kubetorch operations.

This controller provides a centralized endpoint for all Kubernetes API operations,
replacing direct K8s API calls from clients. It serves as a foundation for the
platform controller to add features like auth, quotas, etc.

It also manages deployed resources and pod configurations via SQLite database,
handling the apply and deploy flow for Kubetorch services.
"""

import asyncio
import logging
import os
from functools import wraps
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool

from kubernetes import client, config, dynamic
from kubernetes.client.rest import ApiException
from kubernetes.stream import stream as k8s_stream

# Read configuration from environment variables, set via the Helm chart
VERSION = os.getenv("VERSION")  # matches image tag
WORKERS = int(os.getenv("WORKERS", "8"))
CONNECTION_POOL_SIZE = int(os.getenv("CONNECTION_POOL_SIZE", "20"))
NAMESPACE = os.getenv("NAMESPACE", "kubetorch")
AUTH_ENDPOINT = os.getenv(
    "AUTH_ENDPOINT", ""
)  # e.g., http://mgmt-controller:8000/api/v1/auth/validate

# Simple logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(filename)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


class HealthFilter(logging.Filter):
    def filter(self, record):
        try:
            msg = record.getMessage()
            return "/health" not in msg
        except Exception:
            return True


logging.getLogger("uvicorn.access").addFilter(HealthFilter())

try:
    config.load_incluster_config()
except Exception:
    config.load_kube_config()

# Configure K8s client for better connection management under high load
configuration = client.Configuration.get_default_copy()
configuration.connection_pool_maxsize = CONNECTION_POOL_SIZE
configuration.retries = 0  # We handle retries at app level with @retry_k8s_call
configuration.pool_connections = CONNECTION_POOL_SIZE
configuration.pool_maxsize = CONNECTION_POOL_SIZE

# Initialize K8s API clients with optimized configuration
api_client = client.ApiClient(configuration)
core_v1 = client.CoreV1Api(api_client)
apps_v1 = client.AppsV1Api(api_client)
storage_v1 = client.StorageV1Api(api_client)
networking_v1 = client.NetworkingV1Api(api_client)
custom_objects = client.CustomObjectsApi(api_client)
dynamic_client = dynamic.DynamicClient(api_client)

logger.info(f"Kubetorch Controller Version: {VERSION}")

# Import background tasks lifespan (manages EventWatcher, TTLController, etc.)
from background_tasks import create_lifespan

# Initialize database and K8s clients
from core import k8s
from core.database import init_db
from routes import (
    apply_router,
    configmaps_router,
    deploy_router,
    deployments_router,
    discover_router,
    ingresses_router,
    nodes_router,
    pods_router,
    pool_router,
    secrets_router,
    services_router,
    teardown_router,
    volumes_router,
    ws_pods_router,
)

# Create FastAPI app with lifespan for background tasks
app = FastAPI(title="Kubetorch Controller", version=VERSION, lifespan=create_lifespan())

init_db()
k8s.init(apps_v1, core_v1, custom_objects, dynamic_client)

app.include_router(pool_router)
app.include_router(apply_router)
app.include_router(configmaps_router)
app.include_router(deploy_router)
app.include_router(deployments_router)
app.include_router(discover_router)
app.include_router(ingresses_router)
app.include_router(nodes_router)
app.include_router(pods_router)
app.include_router(secrets_router)
app.include_router(services_router)
app.include_router(volumes_router)
app.include_router(ws_pods_router)
app.include_router(teardown_router)

# ====================================================================
# Authentication Setup
# ====================================================================
from auth.middleware import AuthMiddleware, setup_auth

# Initialize auth client (connects to mgmt_controller for validation)
auth_client = setup_auth(AUTH_ENDPOINT)

# Add auth middleware
app.add_middleware(AuthMiddleware)


# ====================================================================
# Retry Logic for K8s API Calls
# ====================================================================
def retry_k8s_call(max_retries=5, base_delay=0.5):
    """Decorator to retry K8s API calls on transient errors."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except ApiException as e:
                    # Retry on actual connection/timeout errors only
                    # NOTE: Don't use string matching - ApiException includes HTTP headers
                    # with "Connection: keep-alive" which would match everything
                    is_connection_error = isinstance(
                        e, (ConnectionError, TimeoutError, OSError)
                    ) or (
                        hasattr(e, "__cause__")
                        and isinstance(
                            e.__cause__, (ConnectionError, TimeoutError, OSError)
                        )
                    )
                    if is_connection_error and attempt < max_retries:
                        sleep = base_delay * attempt
                        logger.warning(
                            f"K8s API connection error (attempt {attempt}/{max_retries}), "
                            f"retrying in {sleep:.2f}s: {e}"
                        )
                        await asyncio.sleep(sleep)
                        continue
                    raise
            raise

        return wrapper

    return decorator


def k8s_exception_to_http(e: ApiException) -> HTTPException:
    """Convert K8s ApiException to HTTP exception with standard K8s error format"""
    try:
        error_body = e.body
        return HTTPException(status_code=e.status, detail=error_body)
    except Exception:
        logger.error(
            f"Kubernetes API exception (no body): status={e.status}, error={e}"
        )
        return HTTPException(status_code=e.status, detail=str(e))


# =============================================================================
# Health Check
# =============================================================================
@app.get("/health")
async def health():
    """Simple health check - no external dependencies to ensure fast response."""
    return {"status": "ok", "version": VERSION}


# =============================================================================
# Stream (exec)
# =============================================================================
def _create_isolated_core_v1_for_exec():
    """Create an isolated K8s CoreV1Api client for exec calls.

    Exec uses WebSocket connections which pollute the shared connection pool
    and cause "Handshake status 200 OK" errors on subsequent regular API calls.
    Using an isolated client prevents this interference.
    """
    exec_config = client.Configuration()
    try:
        config.load_incluster_config(client_configuration=exec_config)
    except config.ConfigException:
        config.load_kube_config(client_configuration=exec_config)
    exec_api_client = client.ApiClient(configuration=exec_config)
    return client.CoreV1Api(api_client=exec_api_client)


@app.post("/api/v1/namespaces/{namespace}/pods/{pod}/exec")
@retry_k8s_call()
async def exec_in_pod(
    namespace: str,
    pod: str,
    request: Request,
    command: Optional[List[str]] = Query(default=None),
    container: Optional[str] = Query(default=None),
):
    if command is None or container is None:
        try:
            body = await request.json()
        except Exception:
            body = None

        if isinstance(body, dict):
            if command is None:
                command = body.get("command")
            if container is None:
                container = body.get("container")
        elif isinstance(body, list) and command is None:
            command = body

    if not command:
        raise HTTPException(
            status_code=400,
            detail="command must be provided as query parameters (?command=...) "
            'or as JSON body (e.g. {"command": ["bash", "-lc", "..."]}).',
        )

    def _exec():
        # Use isolated client to prevent WebSocket connection pool pollution
        isolated_core_v1 = _create_isolated_core_v1_for_exec()
        kwargs = {
            "command": command,
            "stderr": True,
            "stdin": False,
            "stdout": True,
            "tty": False,
        }
        if container:
            kwargs["container"] = container

        return k8s_stream(
            isolated_core_v1.connect_get_namespaced_pod_exec,
            pod,
            namespace,
            **kwargs,
        )

    try:
        resp = await run_in_threadpool(_exec)
        return {"output": resp}
    except ApiException as e:
        raise k8s_exception_to_http(e)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8081"))
    uvicorn.run(app, host="0.0.0.0", port=port)
