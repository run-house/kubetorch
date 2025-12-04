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


def pod_is_running(pod: dict) -> bool:
    """Check if pod is running. Pod must be a dict from ControllerClient."""
    status = pod.get("status", {})
    phase = status.get("phase")
    metadata = pod.get("metadata", {})
    deletion_timestamp = metadata.get("deletionTimestamp")
    return phase == "Running" and deletion_timestamp is None


def check_loki_enabled() -> bool:
    """Check if Loki is enabled using the centralized controller."""
    controller = globals.controller_client()
    kt_namespace = globals.config.install_namespace

    try:
        controller.get_service(namespace=kt_namespace, name=LOKI_GATEWAY_SERVICE_NAME)
        logger.debug(f"Loki gateway service found in namespace {kt_namespace}")
        return True

    except Exception as e:
        # controller wraps K8s 404 properly
        if "404" in str(e):
            logger.debug(f"Loki gateway service not found in namespace {kt_namespace}")
            return False

        # fallback check: inside cluster, try resolving service DNS
        if is_running_in_kubernetes():
            loki_url = f"http://{LOKI_GATEWAY_SERVICE_NAME}.{kt_namespace}.svc.cluster.local/loki/api/v1/labels"
            try:
                resp = httpx.get(loki_url, timeout=2)
                if resp.status_code == 200:
                    return True
            except Exception:
                pass

        return False


def check_prometheus_enabled() -> bool:
    """
    Check if Prometheus is enabled using the centralized controller.
    """
    controller = globals.controller_client()
    kt_namespace = globals.config.install_namespace

    try:
        controller.get_service(namespace=kt_namespace, name=PROMETHEUS_SERVICE_NAME)
        logger.debug(f"Prometheus service found in namespace {kt_namespace}")
        return True

    except Exception as e:
        if "404" in str(e):
            logger.debug(f"Prometheus service not found in namespace {kt_namespace}")
            return False

        # DNS fallback for in-cluster execution
        if is_running_in_kubernetes():
            prom_url = f"http://{PROMETHEUS_SERVICE_NAME}.{kt_namespace}.svc.cluster.local/api/v1/labels"
            try:
                resp = httpx.get(prom_url, timeout=2)
                if resp.status_code == 200:
                    return True
            except Exception:
                pass

        return False


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
