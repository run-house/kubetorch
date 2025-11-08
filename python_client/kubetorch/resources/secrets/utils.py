import os
import time
from pathlib import Path
from typing import List, Optional

import yaml
from kubernetes import client, config
from kubernetes.client import V1Pod
from kubernetes.stream import stream

from kubetorch.constants import DEFAULT_KUBECONFIG_PATH
from kubetorch.globals import config as kt_config

from kubetorch.logger import get_logger
from kubetorch.servers.http.utils import is_running_in_kubernetes

logger = get_logger(__name__)


def get_k8s_identity_name() -> Optional[str]:
    """Get Kubernetes user identity from kubeconfig file.

    Returns:
        User identity string (e.g., "user-{name}", "role-{name}", "sa-{name}") or None
    """
    try:
        if is_running_in_kubernetes():
            # Get the service account name
            service_account_name = os.environ.get("SERVICE_ACCOUNT_NAME")
            if service_account_name:
                return "sa-" + service_account_name.lower()
            return None

        # Read from kubeconfig
        kubeconfig_path = os.getenv("KUBECONFIG") or DEFAULT_KUBECONFIG_PATH
        kubeconfig_file = Path(kubeconfig_path).expanduser()
        if not kubeconfig_file.exists():
            return None

        with open(kubeconfig_file, "r") as f:
            kubeconfig = yaml.safe_load(f)

        current_context = kubeconfig.get("current-context")
        if not current_context:
            return None

        # Find current context's user
        for context in kubeconfig.get("contexts", []):
            if context.get("name") == current_context:
                user_name = context.get("context", {}).get("user")
                if not user_name:
                    return None

                # Parse AWS ARN format (EKS IAM users/roles)
                if "assumed-role" in user_name:
                    parts = user_name.split("/")
                    if len(parts) >= 2:
                        return "role-" + parts[-2].lower()
                elif "/" in user_name and (".amazonaws.com" in user_name or "arn:aws" in user_name):
                    parts = user_name.split("/")
                    return "user-" + parts[-1].lower()

                # Check for exec-based auth with AWS role
                for user in kubeconfig.get("users", []):
                    if user.get("name") == user_name:
                        exec_config = user.get("user", {}).get("exec", {})
                        for env_var in exec_config.get("env", []):
                            if env_var.get("name") == "AWS_ROLE_ARN":
                                role_arn = env_var.get("value", "")
                                if "/" in role_arn:
                                    return "role-" + role_arn.split("/")[-1].lower()
                        break

                # Default: use user name as-is
                return "user-" + user_name.lower()

    except Exception as e:
        logger.debug(f"Failed to get Kubernetes identity name: {e}")

    return None


def read_files_as_secrets_dict(path: str, filenames: List[str]):
    values = {}
    cred_path = os.path.expanduser(path)

    for filename in filenames:
        file_path = os.path.join(cred_path, filename)
        # Read the files
        content = _read_file_if_exists(file_path)
        if content:
            values[filename] = content
        # # Base64 encode the content
        # encoded = base64.b64encode(content).decode("utf-8")
        # values[filename] = encoded

    return values


def _read_file_if_exists(file_path: str) -> Optional[str]:
    try:
        with open(file_path, "r") as f:  # "rb" if you encode above.
            return f.read()
    except FileNotFoundError:
        logger.error(f"Warning: {file_path} not found, using empty content")
        return None


# ------------------------------------------------------------------------------------------------
# Secret testing utils
# ------------------------------------------------------------------------------------------------


def check_path_on_kubernetes_pods(path: str, service_name: str, namespace: str = None) -> bool:
    """
    Check if a path exists on a specific Knative service's pods
    """
    namespace = namespace or kt_config.namespace
    # Load Kubernetes configuration
    config.load_kube_config()
    # Initialize API clients
    core_v1_api = client.CoreV1Api()

    pods = _fetch_pods_for_kubernetes_service(service_name, namespace, core_v1_api)
    if not pods:
        logger.error(f"No pods found for service {service_name} in namespace {namespace}")
        return False

    path_found = True
    for pod in pods:
        pod_name = pod.metadata.name
        command = ["/bin/bash", "-c", f"[ -f {path} ] && echo yes || echo no"]
        try:
            resp = stream(
                core_v1_api.connect_get_namespaced_pod_exec,
                name=pod_name,
                namespace=namespace,
                command=command,
                container="kubetorch",
                stderr=True,
                stdout=True,
            )
            if "yes" in resp:
                continue
        except client.exceptions.ApiException as e:
            logger.error(f"Error executing command on pod {pod_name}: {e}")

        path_found = False

    return path_found


def check_env_vars_on_kubernetes_pods(env_vars: list, service_name: str, namespace: str = None) -> dict:
    """
    Check if an AWS role is assumed on a specific Knative service's pods

    :param namespace: Kubernetes namespace
    :param service_name: Name of the Knative service
    :return: Dictionary with role assumption details
    """
    namespace = namespace or kt_config.namespace
    # Load Kubernetes configuration
    config.load_kube_config()
    # Initialize API clients
    core_v1_api = client.CoreV1Api()

    pods = _fetch_pods_for_kubernetes_service(service_name, namespace, core_v1_api)
    if not pods:
        logger.error(f"No pods found for service {service_name} in namespace {namespace}")
        return {}

    found_env_vars = {}

    for pod in pods:
        for env_var in env_vars:
            if found_env_vars.get(env_var):
                # Skip if already found on another pod
                continue
            pod_name = pod.metadata.name
            command = ["/bin/bash", "-c", f"echo ${env_var}"]
            try:
                resp = stream(
                    core_v1_api.connect_get_namespaced_pod_exec,
                    name=pod_name,
                    namespace=namespace,
                    command=command,
                    container="kubetorch",
                    stderr=True,
                    stdout=True,
                )
                if len(resp.strip()) > 0:
                    found_env_vars[env_var] = resp.strip()
            except client.exceptions.ApiException as e:
                logger.error(f"Error executing command: {e}")

        if set(found_env_vars.keys()) == set(env_vars):
            # Found all env vars: skip the remaining pods
            break

    return found_env_vars


def _fetch_pods_for_kubernetes_service(service_name: str, namespace: str, client_api: client.CoreV1Api) -> List[V1Pod]:
    """
    Fetch pods for a specific Knative service with timeout
    """
    start_time = time.time()
    while time.time() - start_time < 30:
        try:
            # List pods matching the service
            pods = client_api.list_namespaced_pod(
                namespace=namespace,
                label_selector=f"kubetorch.com/service={service_name}",
            )
            ready_pods = [pod for pod in pods.items if pod.status.phase == "Running"]
            if ready_pods:
                return ready_pods
        except Exception as e:
            logger.error(f"Error fetching pods for service {service_name} in namespace {namespace}: {e}")
        time.sleep(1)

    return []
