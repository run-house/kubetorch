"""Secrets route for managing Kubernetes secrets."""

import asyncio
import logging
import os
from typing import Optional

from core import k8s

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from kubernetes import client
from kubernetes.client import V1ListMeta, V1SecretList
from kubernetes.client.rest import ApiException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/controller", tags=["secrets"])


@router.post("/secrets/{namespace}")
async def create_secret(namespace: str, request: Request):
    """Create a secret.

    Args:
        namespace (str): Kubernetes namespace.
        request: Request body containing the secret manifest.

    Returns:
        K8s Secret response.
    """
    try:
        body = await request.json()
        result = await run_in_threadpool(
            k8s.core_v1.create_namespaced_secret,
            namespace=namespace,
            body=body,
        )
        return JSONResponse(
            content=client.ApiClient().sanitize_for_serialization(result)
        )
    except ApiException as e:
        logger.error(f"Failed to create secret in {namespace}: {e}")
        from server import k8s_exception_to_http

        raise k8s_exception_to_http(e)


@router.get("/secrets/{namespace}/{name}")
async def get_secret(namespace: str, name: str):
    """Get a specific secret.

    Args:
        namespace (str): Kubernetes namespace.
        name (str): Secret name.

    Returns:
        K8s Secret response.
    """
    try:
        result = await run_in_threadpool(
            k8s.core_v1.read_namespaced_secret,
            name=name,
            namespace=namespace,
        )
        return JSONResponse(
            content=client.ApiClient().sanitize_for_serialization(result)
        )
    except ApiException as e:
        logger.error(f"Failed to get secret {name} in {namespace}: {e}")
        from server import k8s_exception_to_http

        raise k8s_exception_to_http(e)


@router.patch("/secrets/{namespace}/{name}")
async def patch_secret(namespace: str, name: str, request: Request):
    """Patch a secret.

    Args:
        namespace (str): Kubernetes namespace.
        name (str): Secret name.
        request: Request body containing the patch data.

    Returns:
        K8s Secret response.
    """
    try:
        body = await request.json()
        result = await run_in_threadpool(
            k8s.core_v1.patch_namespaced_secret,
            name=name,
            namespace=namespace,
            body=body,
        )
        return JSONResponse(
            content=client.ApiClient().sanitize_for_serialization(result)
        )
    except ApiException as e:
        logger.error(f"Failed to patch secret {name} in {namespace}: {e}")
        from server import k8s_exception_to_http

        raise k8s_exception_to_http(e)


@router.get("/secrets/{namespace}")
async def list_secrets(namespace: str, label_selector: Optional[str] = None):
    """List secrets in a namespace.

    Args:
        namespace (str): Kubernetes namespace.
        label_selector (str, optional): Label selector to filter secrets.

    Returns:
        K8s SecretList response.
    """
    try:
        result = await run_in_threadpool(
            k8s.core_v1.list_namespaced_secret,
            namespace=namespace,
            label_selector=label_selector,
        )
        return JSONResponse(
            content=client.ApiClient().sanitize_for_serialization(result)
        )
    except ApiException as e:
        logger.error(f"Failed to list secrets in {namespace}: {e}")
        from server import k8s_exception_to_http

        raise k8s_exception_to_http(e)


@router.delete("/secrets/{namespace}/{name}")
async def delete_secret(namespace: str, name: str):
    """Delete a secret.

    Args:
        namespace (str): Kubernetes namespace.
        name (str): Secret name.

    Returns:
        K8s Status response.
    """
    try:
        result = await run_in_threadpool(
            k8s.core_v1.delete_namespaced_secret,
            name=name,
            namespace=namespace,
        )
        return JSONResponse(
            content=client.ApiClient().sanitize_for_serialization(result)
        )
    except ApiException as e:
        logger.error(f"Failed to delete secret {name} in {namespace}: {e}")
        from server import k8s_exception_to_http

        raise k8s_exception_to_http(e)


@router.get("/secrets")
async def list_secrets_all_namespaces(label_selector: Optional[str] = None):
    """List secrets across all namespaces.

    Args:
        label_selector (str, optional): Label selector to filter secrets.

    Returns:
        K8s SecretList response.
    """
    try:
        deployment_namespaces = [
            ns for ns in os.environ.get("WATCH_NAMESPACES", "").split(",") if ns
        ] or ["default"]

        results = await asyncio.gather(
            *[
                run_in_threadpool(
                    k8s.core_v1.list_namespaced_secret,
                    namespace=ns,
                    label_selector=label_selector,
                )
                for ns in deployment_namespaces
            ]
        )

        all_secrets = V1SecretList(
            api_version="v1",
            kind="SecretList",
            metadata=V1ListMeta(),
            items=[secret for result in results for secret in result.items],
        )
        return JSONResponse(
            content=client.ApiClient().sanitize_for_serialization(all_secrets)
        )
    except ApiException as e:
        logger.error(f"Failed to list secrets across all namespaces: {e}")
        from server import k8s_exception_to_http

        raise k8s_exception_to_http(e)
