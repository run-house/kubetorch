"""ConfigMaps route for managing K8s configmaps."""

import logging
from typing import Optional

from core import k8s

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from kubernetes import client
from kubernetes.client.rest import ApiException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/controller", tags=["configmaps"])


@router.get("/configmaps/{namespace}")
async def list_configmaps(namespace: str, label_selector: Optional[str] = None):
    """List ConfigMaps in a namespace.

    Args:
        namespace (str): Kubernetes namespace.
        label_selector (str, optional): Label selector to filter configmaps.

    Returns:
        K8s ConfigMapList response.
    """
    try:
        result = await run_in_threadpool(
            k8s.core_v1.list_namespaced_config_map,
            namespace=namespace,
            label_selector=label_selector,
        )
        return JSONResponse(
            content=client.ApiClient().sanitize_for_serialization(result)
        )
    except ApiException as e:
        logger.error(f"Failed to list configmaps in {namespace}: {e}")
        from server import k8s_exception_to_http

        raise k8s_exception_to_http(e)
