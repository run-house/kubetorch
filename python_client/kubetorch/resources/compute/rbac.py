from typing import Optional

from kubernetes import client, config

from kubetorch import globals
from kubetorch.logger import get_logger

from kubetorch.serving.constants import DEFAULT_SERVICE_ACCOUNT_NAME

logger = get_logger(__name__)


def find_service_account_in_namespace(
    namespace: str, service_account_name: str, core_v1: client.CoreV1Api
) -> Optional[client.V1ServiceAccount]:
    """Find a ServiceAccount in a given namespace."""
    try:
        return core_v1.read_namespaced_service_account(
            name=service_account_name, namespace=namespace
        )
    except client.exceptions.ApiException as e:
        if e.status != 404:
            raise e
        return None


def cleanup_namespace_and_service_account(namespace: str) -> None:
    """
    Cleanup a ServiceAccount and its associated ClusterRoleBindings in a given namespace.
    """
    if namespace == globals.config.install_namespace:
        raise ValueError(
            f"Cannot cleanup ${globals.config.install_namespace} ServiceAccount"
        )

    # Load kube config (works for both local and in-cluster)
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()

    core_v1 = client.CoreV1Api()
    rbac_v1 = client.RbacAuthorizationV1Api()

    service_account = find_service_account_in_namespace(
        namespace, DEFAULT_SERVICE_ACCOUNT_NAME, core_v1
    )
    if not service_account:
        return

    # Delete the ServiceAccount
    core_v1.delete_namespaced_service_account(
        name=DEFAULT_SERVICE_ACCOUNT_NAME, namespace=namespace
    )
    logger.info(f"Deleted ServiceAccount {namespace}/{DEFAULT_SERVICE_ACCOUNT_NAME}")

    # Delete all ClusterRoleBindings that reference the ServiceAccount
    bindings = rbac_v1.list_namespaced_role_binding(
        namespace=namespace,
        label_selector=f"kubetorch.com/service-account={DEFAULT_SERVICE_ACCOUNT_NAME},kubetorch.com/namespace={namespace}",
    )
    for binding in bindings.items:
        rbac_v1.delete_namespaced_role_binding(
            name=binding.metadata.name, namespace=namespace
        )
        logger.info(f"Deleted RoleBinding {binding.metadata.name}")

    logger.info(
        f"Cleanup completed for ServiceAccount {namespace}/{DEFAULT_SERVICE_ACCOUNT_NAME}"
    )

    # Delete the namespace
    core_v1.delete_namespace(name=namespace)
    logger.info(f"Deleted namespace {namespace}")
