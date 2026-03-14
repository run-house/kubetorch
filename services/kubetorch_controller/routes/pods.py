"""Pods route for listing, getting, and fetching logs from pods."""

import logging
from typing import Optional

from core import k8s

from fastapi import APIRouter, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, PlainTextResponse
from kubernetes import client
from kubernetes.client.rest import ApiException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/controller", tags=["pods"])


@router.get("/pods/{namespace}")
async def list_pods(namespace: str, label_selector: Optional[str] = None):
    """List pods in a namespace.

    Args:
        namespace (str): Kubernetes namespace.
        label_selector (str, optional): Label selector to filter pods.

    Returns:
        K8s PodList response.
    """
    try:
        result = await run_in_threadpool(
            k8s.core_v1.list_namespaced_pod,
            namespace=namespace,
            label_selector=label_selector,
        )
        return JSONResponse(
            content=client.ApiClient().sanitize_for_serialization(result)
        )
    except ApiException as e:
        logger.error(f"Failed to list pods in {namespace}: {e}")
        from server import k8s_exception_to_http

        raise k8s_exception_to_http(e)


@router.get("/pods/{namespace}/{name}")
async def get_pod(namespace: str, name: str):
    """Get a specific pod.

    Args:
        namespace (str): Kubernetes namespace.
        name (str): Pod name.

    Returns:
        K8s Pod response.
    """
    try:
        result = await run_in_threadpool(
            k8s.core_v1.read_namespaced_pod,
            name=name,
            namespace=namespace,
        )
        return JSONResponse(
            content=client.ApiClient().sanitize_for_serialization(result)
        )
    except ApiException as e:
        logger.error(f"Failed to get pod {name} in {namespace}: {e}")
        from server import k8s_exception_to_http

        raise k8s_exception_to_http(e)


@router.get("/pods/{namespace}/{name}/logs")
async def get_pod_logs(
    namespace: str,
    name: str,
    container: Optional[str] = Query(None, description="Container name"),
    tail_lines: Optional[int] = Query(
        None, alias="tailLines", description="Number of lines from end"
    ),
):
    """Get logs from a pod.

    Args:
        namespace (str): Kubernetes namespace.
        name (str): Pod name.
        container (str, optional): Container name (required for multi-container pods).
        tail_lines (int, optional): Number of lines to return from the end of the logs.

    Returns:
        Plain text logs.
    """
    try:
        result = await run_in_threadpool(
            k8s.core_v1.read_namespaced_pod_log,
            name=name,
            namespace=namespace,
            container=container,
            tail_lines=tail_lines,
        )
        return PlainTextResponse(content=result)
    except ApiException as e:
        logger.error(f"Failed to get logs for pod {name} in {namespace}: {e}")
        from server import k8s_exception_to_http

        raise k8s_exception_to_http(e)
