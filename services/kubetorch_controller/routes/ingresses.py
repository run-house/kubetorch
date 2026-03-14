"""Ingresses route for managing K8s ingresses."""

import logging
from typing import Optional

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from kubernetes import client
from kubernetes.client.rest import ApiException

from server import networking_v1

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/controller", tags=["ingresses"])


@router.get("/ingresses/{namespace}")
async def list_ingresses(namespace: str, label_selector: Optional[str] = None):
    """List Ingresses in a namespace.

    Args:
        namespace (str): Kubernetes namespace.
        label_selector (str, optional): Label selector to filter ingresses.

    Returns:
        K8s IngressList response.
    """
    try:
        result = await run_in_threadpool(
            networking_v1.list_namespaced_ingress,
            namespace=namespace,
            label_selector=label_selector,
        )
        return JSONResponse(
            content=client.ApiClient().sanitize_for_serialization(result)
        )
    except ApiException as e:
        logger.error(f"Failed to list ingresses in {namespace}: {e}")
        from server import k8s_exception_to_http

        raise k8s_exception_to_http(e)
