"""Helper functions for resource discovery."""

import asyncio
import json
import logging
from typing import Dict, List, Optional

from core import k8s
from core.database import get_db, Pool
from fastapi.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

SUPPORTED_TRAINING_JOBS = ["pytorchjob", "tfjob", "mxjob", "xgboostjob"]


def build_label_selector(
    template_value: str, extra_selector: Optional[str] = None
) -> str:
    """Build label selector combining template label with optional extra selector."""
    base = f"kubetorch.com/template={template_value}"
    if extra_selector:
        return f"{base},{extra_selector}"
    return base


async def list_custom_objects(
    group: str, version: str, namespace: str, plural: str, label_selector: str
) -> List[Dict]:
    """List custom objects with error handling for missing CRDs."""
    try:
        result = await run_in_threadpool(
            k8s.custom_objects.list_namespaced_custom_object,
            group=group,
            version=version,
            namespace=namespace,
            plural=plural,
            label_selector=label_selector,
        )
        return result.get("items", [])
    except Exception as e:
        logger.debug(f"Could not list {plural}: {e}")
        return []


async def list_deployments(namespace: str, label_selector: str) -> List[Dict]:
    """List deployments."""
    try:
        from kubernetes import client

        result = await run_in_threadpool(
            k8s.apps_v1.list_namespaced_deployment,
            namespace=namespace,
            label_selector=label_selector,
        )
        return [client.ApiClient().sanitize_for_serialization(d) for d in result.items]
    except Exception as e:
        logger.debug(f"Could not list deployments: {e}")
        return []


def get_name(resource: Dict) -> str:
    """Get name from a resource dict (handles both K8s and pool formats)."""
    if "metadata" in resource:
        return resource.get("metadata", {}).get("name", "")
    return resource.get("name", "")


def filter_resources(
    resources: List[Dict],
    name_filter: Optional[str] = None,
    prefix_filter: Optional[str] = None,
) -> List[Dict]:
    """Filter resources by name substring or prefix."""
    result = resources
    if name_filter:
        result = [r for r in result if name_filter in get_name(r)]
    if prefix_filter:
        result = [r for r in result if get_name(r).startswith(prefix_filter)]
    return result


async def discover_k8_resources(
    namespace: str,
    label_selector: Optional[str] = None,
    name_filter: Optional[str] = None,
    prefix_filter: Optional[str] = None,
):
    """
    Fetches K8s resources that match the provided filter conditions.

    Args:
        namespace (str): Kubernetes namespace to search
        label_selector (str, optional): K8s label selector for server-side filtering
        name_filter (str, optional): Filter by name substring (post-fetch)
        prefix_filter (str, optional): Filter by name prefix (post-fetch)

    Returns:
        Dict mapping resource type to list of resources:
        {
            "knative_services": [...],
            "deployments": [...],
            "rayclusters": [...],
            "training_jobs": [...],
        }
    """
    knative_task = list_custom_objects(
        "serving.knative.dev",
        "v1",
        namespace,
        "services",
        build_label_selector("ksvc", label_selector),
    )
    deployments_task = list_deployments(
        namespace, build_label_selector("deployment", label_selector)
    )
    rayclusters_task = list_custom_objects(
        "ray.io",
        "v1",
        namespace,
        "rayclusters",
        build_label_selector("raycluster", label_selector),
    )

    training_job_tasks = []
    for job_type in SUPPORTED_TRAINING_JOBS:
        task = list_custom_objects(
            "kubeflow.org",
            "v1",
            namespace,
            f"{job_type}s",
            build_label_selector(job_type, label_selector),
        )
        training_job_tasks.append(task)

    knative, deployments, rayclusters, *training_job_lists = await asyncio.gather(
        knative_task, deployments_task, rayclusters_task, *training_job_tasks
    )

    training_jobs = [
        job_name for job_list in training_job_lists for job_name in job_list
    ]

    return {
        "knative_services": filter_resources(
            knative, name_filter=name_filter, prefix_filter=prefix_filter
        ),
        "deployments": filter_resources(
            deployments, name_filter=name_filter, prefix_filter=prefix_filter
        ),
        "rayclusters": filter_resources(
            rayclusters, name_filter=name_filter, prefix_filter=prefix_filter
        ),
        "training_jobs": filter_resources(
            training_jobs, name_filter=name_filter, prefix_filter=prefix_filter
        ),
    }


def discover_resource_from_pods(namespace: str, selector: dict):
    """Discover the owning resource (Deployment, StatefulSet, etc.) from pods matching a selector.

    Args:
        namespace (str): Kubernetes namespace
        selector (dict): Label selector to find pods

    Returns:
        Tuple of (resource_kind, resource_name) or (None, None) if not found
    """
    try:
        label_selector = ",".join(f"{k}={v}" for k, v in selector.items())
        pods = k8s.core_v1.list_namespaced_pod(
            namespace=namespace, label_selector=label_selector
        )
        if not pods.items:
            return None, None

        owner_refs = pods.items[0].metadata.owner_references
        if not owner_refs:
            return None, None

        owner = owner_refs[0]
        resource_kind = owner.kind
        resource_name = owner.name

        # ReplicaSet is owned by Deployment - follow the chain
        if resource_kind == "ReplicaSet":
            resource_kind = "Deployment"
            resource_name = "-".join(resource_name.rsplit("-", 1)[:-1])

        return resource_kind, resource_name

    except Exception as e:
        logger.warning(f"Could not discover resource type from pods: {e}")
        return None, None


def discover_pools(
    namespace: str,
    label_selector: Optional[str] = None,
    name_filter: Optional[str] = None,
    prefix_filter: Optional[str] = None,
):
    """
    Fetches pools from the database that match the provided filter conditions.

    Args:
        namespace (str): Kubernetes namespace to search
        label_selector (str, optional): Label selector for filtering
        name_filter (str, optional): Filter by name substring
        prefix_filter (str, optional): Filter by name prefix

    Returns:
        List of pool dicts with name, namespace, specifier, metadata, etc.
    """
    pools = []
    session = get_db()
    try:
        query = session.query(Pool).filter(Pool.namespace == namespace)
        db_pools = query.all()

        for p in db_pools:
            if label_selector:
                pool_metadata = p.pool_metadata or {}
                pool_metadata = (
                    json.loads(pool_metadata)
                    if isinstance(pool_metadata, str)
                    else pool_metadata
                )
                skip = False
                for selector_part in label_selector.split(","):
                    if "=" in selector_part:
                        key, value = selector_part.split("=", 1)
                        if key == "kubetorch.com/username":
                            if pool_metadata.get("username") != value:
                                skip = True
                                break
                        if (
                            key == "kubetorch.com/service"
                            and label_selector.count(",") == 0
                        ):
                            if p.name != value:
                                skip = True
                                break
                if skip:
                    continue

            resource_kind = p.resource_kind
            pool_labels = p.labels or {}
            is_kt_managed = "kubetorch.com/template" in pool_labels
            is_byo = not (resource_kind and is_kt_managed)

            pools.append(
                {
                    "name": p.name,
                    "namespace": p.namespace,
                    "specifier": p.specifier,
                    "pool_metadata": p.pool_metadata,
                    "resource_kind": p.resource_kind,
                    "resource_name": p.resource_name,
                    "created_at": p.created_at.isoformat() if p.created_at else None,
                    "is_byo": is_byo,
                }
            )
    finally:
        session.close()

    if name_filter:
        pools = [p for p in pools if name_filter in p["name"]]
    if prefix_filter:
        pools = [p for p in pools if p["name"].startswith(prefix_filter)]

    return pools
