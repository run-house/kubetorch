import logging
from typing import Optional

from auth.middleware import check_namespace_access, get_current_user
from auth.models import AuthenticatedUser
from core.models import ServiceTeardownRequest

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from helpers.delete_helpers import (
    fetch_k8_resources_for_teardown,
    fetch_pools_for_teardown,
    teardown_services_by_name,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/controller", tags=["teardown"])


@router.get("/teardown/list")
async def list_resources_for_teardown(request: ServiceTeardownRequest):
    """Fetch K8s resources that would be deleted by a teardown request.
    Used in `kt teardown` if --force or --yes flags ARE NOT provided, for previewing services before executing teardown,

    Args:
        request (ServiceTeardownRequest): Request containing namespace, services, prefix, etc.

    Returns:
        Dict with 'resources' list of K8s resources (name, kind, api_version).
    """
    namespace = request.namespace
    service_name = request.services if request else None
    prefix = request.prefix if request else None
    teardown_all = request.teardown_all if request else False
    username = request.username if request else None
    exact_match = request.exact_match if request else False

    resources = await fetch_k8_resources_for_teardown(
        namespace=namespace,
        service_name=service_name,
        prefix=prefix,
        teardown_all=teardown_all,
        username=username,
        exact_match=exact_match,
    )

    return {"resources": resources}


@router.delete("/teardown")
async def teardown_services(
    request: ServiceTeardownRequest,
    user: Optional[AuthenticatedUser] = Depends(get_current_user),
):
    """Delete K8s services and associated resources.

    Performs complete cleanup including deployments, pods, configmaps,
    Kubernetes Services, pools, and datastore cache.

    Args:
        request (ServiceTeardownRequest): Teardown configuration
        user (AuthenticatedUser, optional): Authenticated user for namespace access

    Returns:
        Dict with deleted_resources and byo_deleted_services lists
    """
    namespace = request.namespace
    services = request.services if request else None
    force = request.force if request else False
    prefix = request.prefix if request else None
    teardown_all = request.teardown_all if request else False
    username = request.username if request else None
    exact_match = request.exact_match if request else False

    check_namespace_access(user, namespace)

    # If services is a dict, resources were already fetched client-side
    # Otherwise (str, list, or using prefix/teardown_all), fetch from K8s
    if isinstance(services, dict):
        resources_to_delete = services.get("resources", [])
    else:
        resources_to_delete = await fetch_k8_resources_for_teardown(
            namespace=namespace,
            service_name=services,
            prefix=prefix,
            teardown_all=teardown_all,
            username=username,
            exact_match=exact_match,
        )

    # Extract unique service names for cleanup
    service_names = list({r["name"] for r in resources_to_delete})

    # Fetch pools to check for BYO services (needed for response)
    pools_to_delete = await fetch_pools_for_teardown(
        namespace=namespace,
        service_name=service_names,
        exact_match=True,
    )

    if not resources_to_delete and not pools_to_delete:
        msg = (
            f"Service {services} not found"
            if isinstance(services, str)
            else "No services found"
        )
        return JSONResponse(status_code=200, content=msg)

    # Perform the teardown using the shared function
    result = await teardown_services_by_name(
        namespace=namespace,
        service_names=service_names,
        force=force,
    )

    if not result.success:
        # Return first error as HTTP error
        error_msg = result.errors[0] if result.errors else "Teardown failed"
        raise HTTPException(status_code=520, detail=error_msg)

    byo_services = [p["name"] for p in pools_to_delete if p.get("is_byo")]

    return {
        "deleted_resources": result.deleted_resources,
        "byo_deleted_services": byo_services,
    }
