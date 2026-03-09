"""Nodes route for listing cluster nodes."""

import logging
from typing import Optional

from core import k8s

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from kubernetes import client
from kubernetes.client.rest import ApiException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/controller", tags=["nodes"])


@router.get("/nodes")
async def list_nodes(label_selector: Optional[str] = None):
    """List cluster nodes.

    Args:
        label_selector (str, optional): Label selector to filter nodes (e.g., 'gpu=true').

    Returns:
        K8s NodeList response.
    """
    try:
        result = await run_in_threadpool(
            k8s.core_v1.list_node,
            label_selector=label_selector,
        )
        return JSONResponse(
            content=client.ApiClient().sanitize_for_serialization(result)
        )
    except ApiException as e:
        logger.error(f"Failed to list nodes: {e}")
        from server import k8s_exception_to_http

        raise k8s_exception_to_http(e)
