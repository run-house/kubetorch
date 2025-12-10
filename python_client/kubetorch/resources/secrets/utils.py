import os
from pathlib import Path
from typing import List, Optional

import yaml

from kubetorch.constants import DEFAULT_KUBECONFIG_PATH
from kubetorch.globals import config as kt_config

from kubetorch.logger import get_logger
from kubetorch.servers.http.utils import is_running_in_kubernetes
from kubetorch.serving.utils import pod_is_running

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
    Check if a path exists on a specific service's pods using ControllerClient.
    """
    from kubetorch import globals

    namespace = namespace or kt_config.namespace
    controller_client = globals.controller_client()

    # Get pods using ControllerClient
    pods_result = controller_client.list_pods(
        namespace=namespace,
        label_selector=f"kubetorch.com/service={service_name}",
    )
    pods = pods_result.get("items", [])

    # Filter to running pods
    running_pods = [pod for pod in pods if pod_is_running(pod)]

    if not running_pods:
        logger.error(f"No running pods found for service {service_name} in namespace {namespace}")
        return False

    path_found = True
    for pod in running_pods:
        pod_name = pod.get("metadata", {}).get("name")
        if not pod_name:
            continue

        command = ["/bin/bash", "-c", f"[ -f {path} ] && echo yes || echo no"]
        try:
            resp = controller_client.post(
                f"/api/v1/namespaces/{namespace}/pods/{pod_name}/exec",
                json={
                    "command": command,
                    "container": "kubetorch",
                },
            )
            output = resp.get("output", "")
            if "yes" in output:
                continue
        except Exception as e:
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
    from kubetorch import globals

    namespace = namespace or kt_config.namespace
    controller_client = globals.controller_client()

    pods_result = controller_client.list_pods(
        namespace=namespace,
        label_selector=f"kubetorch.com/service={service_name}",
    )
    pods = pods_result.get("items", [])

    running_pods = [pod for pod in pods if pod_is_running(pod)]

    if not running_pods:
        logger.error(f"No running pods found for service {service_name} in namespace {namespace}")
        return {}

    found_env_vars = {}

    for pod in running_pods:
        for env_var in env_vars:
            if found_env_vars.get(env_var):
                # Skip if already found on another pod
                continue
            pod_name = pod.get("metadata", {}).get("name")
            if not pod_name:
                continue

            command = ["/bin/bash", "-c", f"echo ${env_var}"]
            try:
                resp = controller_client.post(
                    f"/api/v1/namespaces/{namespace}/pods/{pod_name}/exec",
                    json={
                        "command": command,
                        "container": "kubetorch",
                    },
                )
                output = resp.get("output", "").strip()
                if len(output) > 0:
                    found_env_vars[env_var] = output
            except Exception as e:
                logger.error(f"Error executing command: {e}")

        if set(found_env_vars.keys()) == set(env_vars):
            # Found all env vars: skip the remaining pods
            break

    return found_env_vars
