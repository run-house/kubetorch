import os
import re
import time
from typing import List, Optional

from kubernetes import client, config
from kubernetes.client import V1Pod, V1TokenReview, V1TokenReviewSpec
from kubernetes.stream import stream

from kubetorch.globals import config as kt_config

from kubetorch.logger import get_logger
from kubetorch.servers.http.utils import is_running_in_kubernetes

logger = get_logger(__name__)


def get_k8s_identity_name() -> Optional[str]:
    try:
        if is_running_in_kubernetes():
            config.load_incluster_config()
        else:
            config.load_kube_config()
        configuration = client.Configuration.get_default_copy()

        token = configuration.api_key.get("authorization", "")
        token = re.sub(r"^Bearer\s", "", token)

        api = client.AuthenticationV1Api()
        token_review = V1TokenReview(spec=V1TokenReviewSpec(token=token))

        result = api.create_token_review(token_review)

        if result.status.authenticated:
            user = result.status.user
            # For EKS IAM users/roles
            if hasattr(user, "username") and user.username.endswith("amazonaws.com"):
                # ARN format: arn:aws:iam::ACCOUNT_ID:user/USERNAME or
                # arn:aws:sts::ACCOUNT_ID:assumed-role/ROLE_NAME/SESSION_NAME
                arn_parts = user.username.split("/")
                if "assumed-role" in user.username:
                    return "role-" + arn_parts[-2].lower()  # Returns ROLE_NAME
                return "user-" + arn_parts[-1].lower()  # Returns USERNAME for IAM users
            # For Kubernetes service accounts (works for both GKE and EKS)
            elif (
                hasattr(user, "username") and "system:serviceaccount:" in user.username
            ):
                return (
                    "sa-" + user.username.split(":")[-1].lower()
                )  # Returns service account name

    except Exception as e:
        logger.info(f"Failed to get identity name: {e}")

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


def check_path_on_kubernetes_pods(
    path: str, service_name: str, namespace: str = None
) -> bool:
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
        logger.error(
            f"No pods found for service {service_name} in namespace {namespace}"
        )
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


def check_env_vars_on_kubernetes_pods(
    env_vars: list, service_name: str, namespace: str = None
) -> dict:
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
        logger.error(
            f"No pods found for service {service_name} in namespace {namespace}"
        )
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


def _fetch_pods_for_kubernetes_service(
    service_name: str, namespace: str, client_api: client.CoreV1Api
) -> List[V1Pod]:
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
            logger.error(
                f"Error fetching pods for service {service_name} in namespace {namespace}: {e}"
            )
        time.sleep(1)

    return []
