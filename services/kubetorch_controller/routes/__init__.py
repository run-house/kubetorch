"""Routes package for Kubetorch Controller."""

from routes.apply import router as apply_router
from routes.configmaps import router as configmaps_router
from routes.deploy import router as deploy_router
from routes.deployments import router as deployments_router
from routes.discover import router as discover_router
from routes.ingresses import router as ingresses_router
from routes.nodes import router as nodes_router
from routes.pods import router as pods_router
from routes.pool import router as pool_router
from routes.secrets import router as secrets_router
from routes.services import router as services_router
from routes.teardown import router as teardown_router
from routes.volumes import router as volumes_router
from routes.ws_pods import router as ws_pods_router

__all__ = [
    "pool_router",
    "apply_router",
    "configmaps_router",
    "deploy_router",
    "deployments_router",
    "discover_router",
    "ingresses_router",
    "nodes_router",
    "pods_router",
    "secrets_router",
    "services_router",
    "volumes_router",
    "ws_pods_router",
    "teardown_router",
]
