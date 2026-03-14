import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Query
from fastapi.concurrency import run_in_threadpool
from helpers.discover_helpers import discover_k8_resources, discover_pools

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/controller", tags=["discover"])


@router.get("/discover/{namespace}")
async def discover_resources(
    namespace: str,
    label_selector: Optional[str] = Query(
        None,
        description="Additional label selector for K8s filtering (e.g., 'kubetorch.com/username=xyz')",
    ),
    name_filter: Optional[str] = Query(
        None, description="Filter resources by name substring (post-fetch)"
    ),
    prefix_filter: Optional[str] = Query(
        None, description="Filter resources by name prefix (post-fetch)"
    ),
) -> Dict[str, Any]:
    """
    Discover all kubetorch-managed resources in a namespace (k8 resources + pools).

    Args:
        namespace: Kubernetes namespace to search
        label_selector: K8s label selector for server-side filtering (e.g., "kubetorch.com/username=xyz")
        name_filter: Filter by name substring (post-fetch)
        prefix_filter: Filter by name prefix (post-fetch)

    Returns:
        Dict mapping resource type to list of resources:
        {
            "knative_services": [...],
            "deployments": [...],
            "rayclusters": [...],
            "training_jobs": [...],
            "pools": [...],
        }
    """
    k8_resources = await discover_k8_resources(
        namespace=namespace,
        label_selector=label_selector,
        name_filter=name_filter,
        prefix_filter=prefix_filter,
    )

    pools = await run_in_threadpool(
        discover_pools,
        namespace=namespace,
        label_selector=label_selector,
        name_filter=name_filter,
        prefix_filter=prefix_filter,
    )

    discover_result = k8_resources
    discover_result["pools"] = pools
    return discover_result
