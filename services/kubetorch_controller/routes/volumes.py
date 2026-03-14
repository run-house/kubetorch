"""Volumes route for managing PVCs and storage classes."""

import asyncio
import logging
import os
from typing import Optional

from core import k8s

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from kubernetes import client
from kubernetes.client import V1ListMeta, V1PersistentVolumeClaimList
from kubernetes.client.rest import ApiException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/controller", tags=["volumes"])


@router.post("/volumes/{namespace}")
async def create_pvc(namespace: str, request: Request):
    """Create a PersistentVolumeClaim.

    Args:
        namespace (str): Kubernetes namespace.
        request: Request body containing the PVC manifest.

    Returns:
        K8s PersistentVolumeClaim response.
    """
    try:
        body = await request.json()
        result = await run_in_threadpool(
            k8s.core_v1.create_namespaced_persistent_volume_claim,
            namespace=namespace,
            body=body,
        )
        return JSONResponse(
            content=client.ApiClient().sanitize_for_serialization(result)
        )
    except ApiException as e:
        logger.error(f"Failed to create PVC in {namespace}: {e}")
        from server import k8s_exception_to_http

        raise k8s_exception_to_http(e)


@router.get("/volumes/{namespace}/{name}")
async def get_pvc(namespace: str, name: str):
    """Get a specific PersistentVolumeClaim.

    Args:
        namespace (str): Kubernetes namespace.
        name (str): PVC name.

    Returns:
        K8s PersistentVolumeClaim response.
    """
    try:
        result = await run_in_threadpool(
            k8s.core_v1.read_namespaced_persistent_volume_claim,
            name=name,
            namespace=namespace,
        )
        return JSONResponse(
            content=client.ApiClient().sanitize_for_serialization(result)
        )
    except ApiException as e:
        logger.error(f"Failed to get PVC {name} in {namespace}: {e}")
        from server import k8s_exception_to_http

        raise k8s_exception_to_http(e)


@router.delete("/volumes/{namespace}/{name}")
async def delete_pvc(namespace: str, name: str):
    """Delete a PersistentVolumeClaim.

    Args:
        namespace (str): Kubernetes namespace.
        name (str): PVC name.

    Returns:
        K8s Status response.
    """
    try:
        result = await run_in_threadpool(
            k8s.core_v1.delete_namespaced_persistent_volume_claim,
            name=name,
            namespace=namespace,
        )
        return JSONResponse(
            content=client.ApiClient().sanitize_for_serialization(result)
        )
    except ApiException as e:
        logger.error(f"Failed to delete PVC {name} in {namespace}: {e}")
        from server import k8s_exception_to_http

        raise k8s_exception_to_http(e)


@router.get("/volumes/{namespace}")
async def list_pvcs(namespace: str, label_selector: Optional[str] = None):
    """List PersistentVolumeClaims in a namespace.

    Args:
        namespace (str): Kubernetes namespace.
        label_selector (str, optional): Label selector to filter PVCs.

    Returns:
        K8s PersistentVolumeClaimList response.
    """
    try:
        result = await run_in_threadpool(
            k8s.core_v1.list_namespaced_persistent_volume_claim,
            namespace=namespace,
            label_selector=label_selector,
        )
        return JSONResponse(
            content=client.ApiClient().sanitize_for_serialization(result)
        )
    except ApiException as e:
        logger.error(f"Failed to list PVCs in {namespace}: {e}")
        from server import k8s_exception_to_http

        raise k8s_exception_to_http(e)


@router.get("/volumes")
async def list_volumes_all_namespaces(label_selector: Optional[str] = None):
    """List volumes across all namespaces.

    Args:
        label_selector (str, optional): Label selector to filter volumes.

    Returns:
        K8s PersistentVolumeClaimList response.
    """
    try:
        deployment_namespaces = [
            ns for ns in os.environ.get("WATCH_NAMESPACES", "").split(",") if ns
        ] or ["default"]

        results = await asyncio.gather(
            *[
                run_in_threadpool(
                    k8s.core_v1.list_namespaced_persistent_volume_claim,
                    namespace=ns,
                    label_selector=label_selector,
                )
                for ns in deployment_namespaces
            ]
        )

        all_pvcs = V1PersistentVolumeClaimList(
            api_version="v1",
            kind="PersistentVolumeClaimList",
            metadata=V1ListMeta(),
            items=[volume for result in results for volume in result.items],
        )
        return JSONResponse(
            content=client.ApiClient().sanitize_for_serialization(all_pvcs)
        )
    except ApiException as e:
        logger.error(f"Failed to list volumes across all namespaces: {e}")
        from server import k8s_exception_to_http

        raise k8s_exception_to_http(e)


@router.get("/storage-classes")
async def list_storage_classes():
    """List available storage classes.

    Returns:
        K8s StorageClassList response.
    """
    try:
        # Need to use the storage_v1 client which is initialized in server.py
        from server import storage_v1

        result = await run_in_threadpool(storage_v1.list_storage_class)
        return JSONResponse(
            content=client.ApiClient().sanitize_for_serialization(result)
        )
    except ApiException as e:
        logger.error(f"Failed to list storage classes: {e}")
        from server import k8s_exception_to_http

        raise k8s_exception_to_http(e)
