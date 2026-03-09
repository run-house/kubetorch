"""Deployments route for managing K8s deployments."""

import logging

from core import k8s

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from kubernetes import client
from kubernetes.client.rest import ApiException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/controller", tags=["deployments"])


@router.get("/deployments/{namespace}/{name}")
async def get_deployment(namespace: str, name: str):
    """Get a specific Kubernetes Deployment.

    Args:
        namespace (str): Kubernetes namespace.
        name (str): Deployment name.

    Returns:
        K8s Deployment response.
    """
    try:
        result = await run_in_threadpool(
            k8s.apps_v1.read_namespaced_deployment,
            name=name,
            namespace=namespace,
        )
        return JSONResponse(
            content=client.ApiClient().sanitize_for_serialization(result)
        )
    except ApiException as e:
        logger.error(f"Failed to get deployment {name} in {namespace}: {e}")
        from server import k8s_exception_to_http

        raise k8s_exception_to_http(e)
