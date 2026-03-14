import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from urllib.parse import quote

import httpx
from fastapi.concurrency import run_in_threadpool
from helpers.discover_helpers import discover_k8_resources, discover_pools

from kubernetes.client.rest import ApiException
from kubernetes.dynamic.exceptions import NotFoundError

logger = logging.getLogger(__name__)


@dataclass
class TeardownResult:
    """Result of a teardown operation."""

    success: bool
    deleted_resources: List[Dict] = field(default_factory=list)
    deleted_pools: List[Dict] = field(default_factory=list)
    deleted_service_names: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# Mapping of K8s kinds to their API versions
KIND_TO_API_VERSION = {
    "Deployment": "apps/v1",
    "PyTorchJob": "kubeflow.org/v1",
    "TFJob": "kubeflow.org/v1",
    "MXJob": "kubeflow.org/v1",
    "XGBoostJob": "kubeflow.org/v1",
    "RayCluster": "ray.io/v1",
    "Service": "v1",
}

# Mapping from discover_k8_resources keys to K8s kind and api_version
RESOURCE_TYPE_INFO = {
    "knative_services": {"kind": "Service", "api_version": "serving.knative.dev/v1"},
    "deployments": {"kind": "Deployment", "api_version": "apps/v1"},
    "rayclusters": {"kind": "RayCluster", "api_version": "ray.io/v1"},
}


def delete_resource_sync(
    api_version: str,
    kind: str,
    name: str,
    namespace: Optional[str] = None,
    grace_period_seconds: Optional[int] = None,
    propagation_policy: Optional[str] = None,
) -> bool:
    """
    Delete a resource using the dynamic client.

    Args:
        api_version (str): API version (e.g., "apps/v1")
        kind (str): Resource kind (e.g., "Deployment")
        name (str): Resource name
        namespace (str, optional): Namespace for namespaced resources
        grace_period_seconds (int, optional): Grace period for deletion
        propagation_policy (str, optional): Propagation policy

    Returns:
        bool: True if deleted or not found, False on error
    """
    from core import k8s

    dyn = k8s.dynamic
    api = dyn.resources.get(api_version=api_version, kind=kind)

    delete_options = {}
    if grace_period_seconds is not None:
        delete_options["grace_period_seconds"] = grace_period_seconds
    if propagation_policy is not None:
        delete_options["propagation_policy"] = propagation_policy

    try:
        api.delete(name=name, namespace=namespace, **delete_options)
        return True
    except NotFoundError:
        logger.debug(f"Resource {kind}/{name} not found, already deleted")
        return True
    except Exception as e:
        logger.error(f"Failed to delete resource {kind}/{name}: {e}")
        return False


async def teardown_kt_services(
    resources_to_delete: List[Dict],
    namespace: str,
    grace_period_seconds: Optional[int] = None,
    propagation_policy: Optional[str] = None,
) -> bool:
    """
    Delete a list of K8s resources.

    Args:
        resources_to_delete (list): List of dicts with name, kind, api_version
        namespace (str): Kubernetes namespace
        grace_period_seconds (int, optional): Grace period for deletion
        propagation_policy (str, optional): Propagation policy

    Returns:
        bool: True if all deletions succeeded
    """
    success = True
    for resource in resources_to_delete:
        deleted = await run_in_threadpool(
            delete_resource_sync,
            api_version=resource["api_version"],
            kind=resource["kind"],
            name=resource["name"],
            namespace=namespace,
            grace_period_seconds=grace_period_seconds,
            propagation_policy=propagation_policy,
        )
        if not deleted:
            logger.error(f"Failed to delete {resource['kind']}/{resource['name']}")
            success = False
    return success


async def parse_resources_to_delete(
    namespace: str, label_selector: str, prefix: Optional[str] = None
) -> List[Dict]:
    """
    Parse discovered k8s resources into a flat list for deletion.

    Args:
        namespace (str): Kubernetes namespace
        label_selector (str): Label selector for filtering resources
        prefix (str, optional): Prefix filter for resource names

    Returns:
        List of dicts with name, kind, api_version for each resource to delete.
    """
    discovered = await discover_k8_resources(
        namespace=namespace, label_selector=label_selector, prefix_filter=prefix
    )

    resources = []
    for resource_type, items in discovered.items():
        if resource_type == "training_jobs":
            # Training jobs have their kind in the resource itself
            for item in items:
                kind = item.get("kind")
                resources.append(
                    {
                        "name": item.get("metadata", {}).get("name"),
                        "kind": kind,
                        "api_version": KIND_TO_API_VERSION.get(kind, "kubeflow.org/v1"),
                    }
                )
        elif resource_type in RESOURCE_TYPE_INFO:
            info = RESOURCE_TYPE_INFO[resource_type]
            for item in items:
                name = item.get("metadata", {}).get("name")
                resources.append(
                    {
                        "name": name,
                        "kind": info["kind"],
                        "api_version": info["api_version"],
                    }
                )
                # RayClusters have an associated headless service
                if resource_type == "rayclusters":
                    resources.append(
                        {
                            "name": f"{name}-headless",
                            "kind": "Service",
                            "api_version": "v1",
                        }
                    )

    return resources


async def fetch_k8_resources_for_teardown(
    namespace: str,
    service_name: Optional[Union[str, list]] = None,
    prefix: Optional[str] = None,
    teardown_all: Optional[bool] = False,
    username: Optional[str] = None,
    exact_match: Optional[bool] = False,
) -> List[Dict]:
    """
    Fetch k8s resources that need to be torn down.

    Args:
        namespace (str): Kubernetes namespace
        service_name (str, list, optional): Service name(s) to tear down
        prefix (str, optional): Prefix filter for resource names
        teardown_all (bool, optional): Whether to tear down all resources
        username (str, optional): Filter by username
        exact_match (bool, optional): Whether to match service name exactly

    Returns:
        List of dicts with name, kind, api_version for each resource to delete.
    """
    resources = []

    is_name_str = service_name and isinstance(service_name, str)
    service_names_to_list = [service_name] if is_name_str else service_name

    if service_names_to_list:
        for name in service_names_to_list:
            label_selector = f"kubetorch.com/service={name}"
            if exact_match:
                prefix = None
            found = await parse_resources_to_delete(
                namespace=namespace, label_selector=label_selector, prefix=prefix
            )
            resources.extend(found)

    elif prefix or teardown_all:
        label_selector = "kubetorch.com/service"
        if username:
            label_selector += f",kubetorch.com/username={username}"
        resources = await parse_resources_to_delete(
            namespace=namespace, label_selector=label_selector, prefix=prefix
        )

    return resources


async def fetch_pools_for_teardown(
    namespace: str,
    service_name: Optional[Union[str, list]] = None,
    prefix: Optional[str] = None,
    teardown_all: Optional[bool] = False,
    username: Optional[str] = None,
    exact_match: Optional[bool] = False,
) -> List[Dict]:
    """
    Fetch pools that need to be torn down.

    Args:
        namespace (str): Kubernetes namespace
        service_name (str, list, optional): Service name(s) to tear down
        prefix (str, optional): Prefix filter for resource names
        teardown_all (bool, optional): Whether to tear down all resources
        username (str, optional): Filter by username
        exact_match (bool, optional): Whether to match service name exactly

    Returns:
        List of pool dicts to delete.
    """
    pools = []

    is_name_str = service_name and isinstance(service_name, str)
    service_names_to_list = [service_name] if is_name_str else service_name

    if service_names_to_list:
        for name in service_names_to_list:
            label_selector = f"kubetorch.com/service={name}"
            if exact_match:
                prefix = None
            found = await run_in_threadpool(
                discover_pools,
                namespace=namespace,
                label_selector=label_selector,
                prefix_filter=prefix,
            )
            pools.extend(found)

    elif prefix or teardown_all:
        label_selector = "kubetorch.com/service"
        if username:
            label_selector += f",kubetorch.com/username={username}"
        pools = await run_in_threadpool(
            discover_pools,
            namespace=namespace,
            label_selector=label_selector,
            prefix_filter=prefix,
        )

    return pools


async def list_service_pods(service_name: str, namespace: str) -> List[str]:
    """List pods for a service."""
    from server import core_v1

    label_selector = f"kubetorch.com/service={service_name}"
    result = await run_in_threadpool(
        core_v1.list_namespaced_pod,
        namespace=namespace,
        label_selector=label_selector,
    )
    return [pod.metadata.name for pod in result.items]


async def delete_service_pods(
    namespace: str,
    service_names: List[str],
    grace_period_seconds: Optional[int] = None,
    propagation_policy: Optional[str] = None,
) -> bool:
    """Delete pods for given services."""
    from server import k8s_exception_to_http

    success = True
    for service_name in service_names:
        try:
            pods = await list_service_pods(service_name, namespace)
            for pod in pods:
                deleted = await run_in_threadpool(
                    delete_resource_sync,
                    api_version="v1",
                    kind="Pod",
                    name=pod,
                    namespace=namespace,
                    grace_period_seconds=grace_period_seconds,
                    propagation_policy=propagation_policy,
                )
                if not deleted:
                    logger.info(f"Failed to delete {service_name}'s pod: {pod}")
                    success = False
        except ApiException as e:
            raise k8s_exception_to_http(e)

    return success


async def delete_service_configmaps(
    namespace: str,
    service_names: List[str],
    grace_period_seconds: Optional[int] = None,
    propagation_policy: Optional[str] = None,
) -> bool:
    """Delete configmaps for given services."""
    from server import core_v1

    success = True
    for name in service_names:
        try:
            result = await run_in_threadpool(
                core_v1.list_namespaced_config_map,
                namespace=namespace,
                label_selector=f"kubetorch.com/service={name}",
            )
            for cm in result.items:
                deleted = await run_in_threadpool(
                    delete_resource_sync,
                    api_version="v1",
                    kind="ConfigMap",
                    name=cm.metadata.name,
                    namespace=namespace,
                    grace_period_seconds=grace_period_seconds,
                    propagation_policy=propagation_policy,
                )
                if not deleted:
                    logger.info(
                        f"Failed to delete {name}'s configmap {cm.metadata.name}"
                    )
                    success = False
        except Exception as e:
            logger.info(f"Failed to delete {name}'s configmaps: {e}")
            success = False

    return success


async def delete_kubetorch_services(
    namespace: str,
    service_names: List[str],
    grace_period_seconds: Optional[int] = None,
    propagation_policy: Optional[str] = None,
) -> bool:
    """Delete Kubernetes Service resources for given kubetorch services.

    Args:
        namespace (str): Kubernetes namespace
        service_names (list): List of service names
        grace_period_seconds (int, optional): Grace period for deletion
        propagation_policy (str, optional): Propagation policy

    Returns:
        bool: True if all deletions succeeded
    """
    from server import core_v1

    success = True
    for name in service_names:
        try:
            deleted = await run_in_threadpool(
                core_v1.delete_collection_namespaced_service,
                namespace=namespace,
                label_selector=f"kubetorch.com/service={name}",
                grace_period_seconds=grace_period_seconds,
                propagation_policy=propagation_policy,
            )
            if not deleted:
                logger.error(
                    f"Failed to delete services for {name} in namespace {namespace}"
                )
                success = False
        except Exception as e:
            logger.error(
                f"Failed to delete services for {name} in namespace {namespace}: {e}"
            )
            success = False

    return success


async def cleanup_service_keys(namespace: str, service_name: str) -> dict:
    """
    Clean up all keys for a service from the data store.

    Deletes all keys under the service name path (e.g., {service_name}/...).

    Args:
        namespace (str): Kubernetes namespace
        service_name (str): Service name to clean up

    Returns:
        dict with success status and deleted_count
    """
    base_url = f"http://kubetorch-data-store.{namespace}.svc.cluster.local:8081"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.delete(
                f"{base_url}/api/v1/keys/{quote(service_name, safe='')}",
                params={"recursive": "true"},
            )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        # 404 is OK - key doesn't exist
        if e.response.status_code == 404:
            return {"success": True, "deleted_count": 0}
        logger.warning(
            f"Failed to cleanup keys for service '{service_name}': HTTP {e.response.status_code}"
        )
        return {"success": False, "error": str(e), "deleted_count": 0}
    except httpx.RequestError as e:
        logger.warning(f"Failed to cleanup keys for service '{service_name}': {e}")
        return {"success": False, "error": str(e), "deleted_count": 0}


async def delete_cache_from_data_store(namespace: str, service_names: List[str]):
    """
    Clean up service keys from datastore.

    Deletes all keys under each service name path (e.g., {service_name}/...).

    Args:
        namespace (str): Namespace of the deleted resources
        service_names (list): List of service names to clean up

    Returns:
        Tuple of (success, deleted_count)
    """
    success = True
    deleted_count = 0

    for name in service_names:
        result = await cleanup_service_keys(namespace, name)
        if result.get("success", True):
            count = result.get("deleted_metadata_count", 0)
            deleted_count += count
            logger.info(f"Cleaned up {count} keys for service {name}")
        else:
            logger.error(f"Failed to cleanup keys for service {name}")
            success = False

    return success, deleted_count


async def teardown_services_by_name(
    namespace: str,
    service_names: List[str],
    force: bool = False,
) -> TeardownResult:
    """
    Complete teardown of services by name.

    Deletes K8s resources, orphan pods, configmaps, pools from DB, and datastore cache.

    Args:
        namespace: Kubernetes namespace
        service_names: List of service names to tear down
        force: If True, use grace_period=0 and Background propagation for faster deletion

    Returns:
        TeardownResult with deleted resources, pools, errors, and success status
    """
    from helpers.pool_helpers import delete_pools_batch

    grace_period_seconds = 0 if force else None
    propagation_policy = "Background" if force else None

    result = TeardownResult(
        success=True,
        deleted_service_names=list(service_names),
    )

    # 1. Fetch K8s resources for the given service names
    resources_to_delete = await fetch_k8_resources_for_teardown(
        namespace=namespace,
        service_name=service_names,
        exact_match=True,
    )

    # 2. Fetch pools for the given service names
    pools_to_delete = await fetch_pools_for_teardown(
        namespace=namespace,
        service_name=service_names,
        exact_match=True,
    )

    if not resources_to_delete and not pools_to_delete:
        logger.info(f"No resources or pools found for services {service_names}")
        return result

    # 3. Delete K8s resources (Knative services, Deployments, RayClusters, etc.)
    if resources_to_delete:
        if not await teardown_kt_services(
            resources_to_delete=resources_to_delete,
            namespace=namespace,
            grace_period_seconds=grace_period_seconds,
            propagation_policy=propagation_policy,
        ):
            result.errors.append("Failed to delete some K8s resources")
            result.success = False
        else:
            result.deleted_resources = resources_to_delete

    # 4. Delete orphan pods
    if not await delete_service_pods(
        namespace=namespace,
        service_names=service_names,
        grace_period_seconds=grace_period_seconds,
        propagation_policy=propagation_policy,
    ):
        result.errors.append("Failed to delete some pods")
        result.success = False

    # 5. Delete configmaps
    if not await delete_service_configmaps(
        namespace=namespace,
        service_names=service_names,
        grace_period_seconds=grace_period_seconds,
        propagation_policy=propagation_policy,
    ):
        result.errors.append("Failed to delete some configmaps")
        result.success = False

    # 6. Delete kubetorch services
    if not await delete_kubetorch_services(
        namespace=namespace,
        service_names=service_names,
        grace_period_seconds=grace_period_seconds,
        propagation_policy=propagation_policy,
    ):
        result.errors.append("Failed to delete services")
        result.success = False

    # 7. Delete pools from DB
    if pools_to_delete:
        pool_result = await run_in_threadpool(delete_pools_batch, pools_to_delete)
        if pool_result is False:
            result.errors.append("Failed to delete pools from database")
            result.success = False
        else:
            result.deleted_pools = pools_to_delete

    # 8. Cleanup datastore cache
    cache_success, _ = await delete_cache_from_data_store(namespace, service_names)
    if not cache_success:
        result.errors.append("Failed to cleanup datastore cache")
        result.success = False

    return result
