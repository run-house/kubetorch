import copy
import os
import socket
import time
import warnings

from pathlib import Path

import httpx
from kubernetes.client import ApiException, CoreV1Api, V1Pod

from kubetorch import globals
from kubetorch.logger import get_logger
from kubetorch.servers.http.utils import is_running_in_kubernetes
from kubetorch.serving.constants import LOKI_GATEWAY_SERVICE_NAME, PROMETHEUS_SERVICE_NAME
from kubetorch.utils import load_kubeconfig

logger = get_logger(__name__)


class KubernetesCredentialsError(Exception):
    pass


def has_k8s_credentials():
    """
    Fast check for K8s credentials - works both in-cluster and external.
    No network calls, no imports needed.
    """
    # Check 1: In-cluster service account
    if (
        Path("/var/run/secrets/kubernetes.io/serviceaccount/token").exists()
        and Path("/var/run/secrets/kubernetes.io/serviceaccount/ca.crt").exists()
    ):
        return True

    # Check 2: Kubeconfig file
    kubeconfig_path = os.environ.get("KUBECONFIG", os.path.expanduser("~/.kube/config"))
    return Path(kubeconfig_path).exists()


def check_kubetorch_versions(response):
    from kubetorch import __version__ as python_client_version, VersionMismatchError

    try:
        data = response.json()
    except ValueError:
        # older nginx proxy versions won't return a JSON
        return

    helm_installed_version = data.get("version")
    if not helm_installed_version:
        logger.debug("No 'version' found in health check response")
        return

    if python_client_version != helm_installed_version:
        msg = (
            f"client={python_client_version}, cluster={helm_installed_version}. "
            "To suppress this error, set the environment variable "
            "`KUBETORCH_IGNORE_VERSION_MISMATCH=1`."
        )
        if not os.getenv("KUBETORCH_IGNORE_VERSION_MISMATCH"):
            raise VersionMismatchError(msg)

        warnings.warn(f"Kubetorch version mismatch: {msg}")


def extract_config_from_nginx_health_check(response):
    """Extract the config from the nginx health check response."""
    try:
        data = response.json()
    except ValueError:
        return
    config = data.get("config", {})
    return config


def wait_for_port_forward(
    process,
    local_port,
    timeout=30,
    health_endpoint: str = None,
    validate_kubetorch_versions: bool = True,
):
    from kubetorch import VersionMismatchError

    start_time = time.time()
    while time.time() - start_time < timeout:
        if process.poll() is not None:
            stderr = process.stderr.read().decode()
            raise Exception(f"Port forward failed: {stderr}")

        try:
            # Check if socket is open
            with socket.create_connection(("localhost", local_port), timeout=1):
                if not health_endpoint:
                    # If we are not checking HTTP (ex: rsync)
                    return True
        except OSError:
            time.sleep(0.2)
            continue

        if health_endpoint:
            url = f"http://localhost:{local_port}" + health_endpoint
            try:
                # Check if HTTP endpoint is ready
                resp = httpx.get(url, timeout=2)
                if resp.status_code == 200:
                    if validate_kubetorch_versions:
                        check_kubetorch_versions(resp)
                    # Extract config to set outside of function scope
                    config = extract_config_from_nginx_health_check(resp)
                    return config
            except VersionMismatchError as e:
                raise e
            except Exception as e:
                logger.debug(f"Waiting for HTTP endpoint to be ready: {e}")

        time.sleep(0.2)

    raise TimeoutError("Timeout waiting for port forward to be ready")


def pod_is_running(pod: V1Pod):
    return pod.status.phase == "Running" and pod.metadata.deletion_timestamp is None


def check_loki_enabled(core_api: CoreV1Api = None) -> bool:
    """Check if loki is enabled"""
    if core_api is None:
        load_kubeconfig()
        core_api = CoreV1Api()

    kt_namespace = globals.config.install_namespace

    try:
        # Check if loki-gateway service exists in the namespace
        core_api.read_namespaced_service(name=LOKI_GATEWAY_SERVICE_NAME, namespace=kt_namespace)
        logger.debug(f"Loki gateway service found in namespace {kt_namespace}")
    except ApiException as e:
        if e.status == 404:
            logger.debug(f"Loki gateway service not found in namespace {kt_namespace}")
            return False

        # Additional permission-proof check: try to ping the internal Loki gateway URL
        # Needed if running in kubernetes without full kubeconfig permissions
        if is_running_in_kubernetes():
            try:
                loki_url = f"http://loki-gateway.{kt_namespace}.svc.cluster.local/loki/api/v1/labels"
                response = httpx.get(loki_url, timeout=2)
                if response.status_code == 200:
                    logger.debug("Loki gateway is reachable")
                else:
                    logger.debug(f"Loki gateway returned status {response.status_code}")
                    return False
            except Exception as e:
                logger.debug(f"Loki gateway is not reachable: {e}")
                return False

    return True


def check_prometheus_enabled(core_api: CoreV1Api = None) -> bool:
    """Check if prometheus is enabled"""
    if core_api is None:
        load_kubeconfig()
        core_api = CoreV1Api()

    kt_namespace = globals.config.install_namespace

    try:
        # Check if prometheus service exists in the namespace
        core_api.read_namespaced_service(name=PROMETHEUS_SERVICE_NAME, namespace=kt_namespace)
        logger.debug(f"Metrics service found in namespace {kt_namespace}")
    except ApiException as e:
        if e.status == 404:
            logger.debug(f"Metrics service not found in namespace {kt_namespace}")
            return False

        # If running inside the cluster, try hitting the service directly
        if is_running_in_kubernetes():
            try:
                prom_url = f"http://{PROMETHEUS_SERVICE_NAME}.{kt_namespace}.svc.cluster.local/api/v1/labels"
                response = httpx.get(prom_url, timeout=2)
                if response.status_code == 200:
                    logger.debug("Metrics service is reachable")
                else:
                    logger.debug(f"Metrics service returned status {response.status_code}")
                    return False
            except Exception as e:
                logger.debug(f"Metrics service is not reachable: {e}")
                return False

    return True


def nested_override(original_dict, override_dict):
    for key, value in override_dict.items():
        if key in original_dict:
            if isinstance(original_dict[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                nested_override(original_dict[key], value)
            else:
                original_dict[key] = value  # Custom wins
        else:
            original_dict[key] = value


def nested_merge(user_dict, kt_dict):
    """
    Merge kubetorch dict into user dict, preserving user values.

    Strategy:
    - For simple values (str, int, bool, etc.): user value takes precedence
    - For dicts: recursively merge, user values take precedence
    - For lists:
      - If list items are dicts with a "name" key, merge by name (user items win)
      - If list items are dicts with a "key" key (e.g., tolerations), merge by key
      - Otherwise, append kt items that don't already exist in user list

    Args:
        user_dict: The user's dict (values take precedence)
        kt_dict: The kubetorch dict (values are added where user doesn't have them)

    Returns:
        The merged dict (modifies user_dict in place)
    """
    for key, kt_value in kt_dict.items():
        if key not in user_dict:
            user_dict[key] = copy.deepcopy(kt_value)
        else:
            user_value = user_dict[key]

            # Both have the key - merge based on type
            if isinstance(user_value, dict) and isinstance(kt_value, dict):  # Recursively merge dicts
                nested_merge(user_value, kt_value)
            elif isinstance(user_value, list) and isinstance(kt_value, list):  # Merge lists
                # Check if we're dealing with lists of dicts by checking first item of either list
                first_user_item = user_value[0] if user_value else None
                first_kt_item = kt_value[0] if kt_value else None

                if (first_user_item and isinstance(first_user_item, dict)) or (
                    first_kt_item and isinstance(first_kt_item, dict)
                ):
                    merge_key = None
                    sample_item = first_user_item or first_kt_item
                    if sample_item:
                        if "name" in sample_item:
                            merge_key = "name"
                        elif "key" in sample_item:
                            merge_key = "key"

                    if merge_key:
                        # Merge by the identified key
                        user_dict_by_key = {item.get(merge_key): item for item in user_value if item.get(merge_key)}
                        # Add kt items that don't conflict with user items
                        for kt_item in kt_value:
                            kt_key_value = kt_item.get(merge_key)
                            if kt_key_value and kt_key_value not in user_dict_by_key:
                                # Add kt item if user doesn't have one with this key
                                user_dict_by_key[kt_key_value] = copy.deepcopy(kt_item)
                        # Preserve items without the merge key from both lists
                        user_items_without_key = [item for item in user_value if not item.get(merge_key)]
                        kt_items_without_key = [copy.deepcopy(item) for item in kt_value if not item.get(merge_key)]
                        # Rebuild list: items with key (merged) + user items without key + kt items without key
                        user_dict[key] = list(user_dict_by_key.values()) + user_items_without_key + kt_items_without_key
                    else:
                        # List of dicts without a known merge key - append items that don't exist
                        user_set = set(str(item) for item in user_value)
                        for kt_item in kt_value:
                            if str(kt_item) not in user_set:
                                user_value.append(copy.deepcopy(kt_item))
                else:
                    # Simple list - append items that don't exist
                    user_set = set(str(item) for item in user_value)
                    for kt_item in kt_value:
                        if str(kt_item) not in user_set:
                            user_value.append(copy.deepcopy(kt_item))
