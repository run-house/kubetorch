"""Services route for managing K8s services."""

import logging

from core import k8s

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from kubernetes import client
from kubernetes.client.rest import ApiException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/controller", tags=["services"])


@router.post("/services/{namespace}")
async def create_service(namespace: str, request: Request):
    """Create a Kubernetes Service.

    Args:
        namespace (str): Kubernetes namespace.
        request: Request body containing the service manifest.

    Returns:
        K8s Service response.
    """
    try:
        body = await request.json()
        raw_params = dict(request.query_params)
        result = await run_in_threadpool(
            k8s.core_v1.create_namespaced_service,
            namespace=namespace,
            body=body,
            **raw_params,
        )
        return JSONResponse(
            content=client.ApiClient().sanitize_for_serialization(result)
        )
    except ApiException as e:
        logger.error(f"Failed to create service in {namespace}: {e}")
        from server import k8s_exception_to_http

        raise k8s_exception_to_http(e)


@router.get("/services/{namespace}/{name}")
async def get_service(namespace: str, name: str):
    """Get a specific Kubernetes Service.

    Args:
        namespace (str): Kubernetes namespace.
        name (str): Service name.

    Returns:
        K8s Service response.
    """
    try:
        result = await run_in_threadpool(
            k8s.core_v1.read_namespaced_service,
            name=name,
            namespace=namespace,
        )
        return JSONResponse(
            content=client.ApiClient().sanitize_for_serialization(result)
        )
    except ApiException as e:
        logger.error(f"Failed to get service {name} in {namespace}: {e}")
        from server import k8s_exception_to_http

        raise k8s_exception_to_http(e)
