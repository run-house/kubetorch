import importlib
import inspect
import os
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import kubetorch.globals
from kubetorch.logger import get_logger
from kubetorch.resources.callables.utils import get_local_install_path, locate_working_dir
from kubetorch.resources.secrets.kubernetes_secrets_client import KubernetesSecretsClient
from kubetorch.serving.utils import StartupError

logger = get_logger(__name__)


class KnativeServiceError(Exception):
    """Base exception for Knative service errors."""

    pass


class ImagePullError(KnativeServiceError):
    """Raised when container image pull fails."""

    pass


class ResourceNotAvailableError(Exception):
    """Raised when required compute resources (GPU, memory, etc.) are not available in the cluster."""

    pass


class ServiceHealthError(KnativeServiceError):
    """Raised when service health checks fail."""

    pass


class ServiceTimeoutError(KnativeServiceError):
    """Raised when service fails to become ready within timeout period."""

    pass


class KnativeServiceConflictError(Exception):
    """Raised when a conflicting non-Knative Kubernetes Service prevents Knative service creation."""

    pass


class PodContainerError(Exception):
    """Raised when pod container is in a terminated or waiting state."""

    pass


class VersionMismatchError(Exception):
    """Raised when the Kubetorch client version is incompatible with the version running on the target cluster"""

    pass


class SecretNotFound(Exception):
    """Raised when trying to update kubetorch secret the does not exist"""

    def __init__(self, secret_name: str, namespace: str):
        super().__init__(f"kubetorch secret {secret_name} was not found in {namespace} namespace")


class ControllerRequestError(Exception):
    """Raised when a request to the kubetorch controller fails."""

    def __init__(self, method: str, url: str, status_code: int, message: str):
        self.method = method
        self.url = url
        self.status_code = status_code
        self.status = status_code
        super().__init__(f"{method} {url} failed with status {status_code}: {message}")


class RsyncError(Exception):
    def __init__(self, cmd: str, returncode: int, stdout: str, stderr: str):
        self.cmd = cmd
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(f"Rsync failed (code={returncode}): {stderr.strip()}")


TERMINATE_EARLY_ERRORS = {
    "ContainerMissing": ImagePullError,
    "ImagePullBackOff": ImagePullError,
    "ErrImagePull": ImagePullError,
    "CrashLoopBackOff": ServiceHealthError,
    "BackOff": ServiceHealthError,
    "StartupError": StartupError,
    "FailedMount": StartupError,
}


def _run_bash(
    commands: Union[str, List[str]],
    pod_names: List[str],
    namespace: str,
    container: str = None,
):
    if isinstance(commands, str):
        commands = [commands]
    commands = [f'{command}; echo "::EXIT_CODE::$?"' for command in commands]

    if isinstance(pod_names, str):
        pod_names = [pod_names]

    controller_client = kubetorch.globals.controller_client()

    ret_codes = []
    for exec_command in commands:
        for pod_name in pod_names:
            if not container:
                pod = controller_client.get_pod(namespace=namespace, name=pod_name)
                # ControllerClient returns dicts
                if isinstance(pod, dict):
                    containers = pod.get("spec", {}).get("containers", [])
                    if not containers:
                        raise Exception(f"No containers found in pod {pod_name}")
                    container = containers[0].get("name")
                else:
                    # Fallback if someone passes a raw k8s object for some reason
                    spec = getattr(pod, "spec", None)
                    pod_containers = getattr(spec, "containers", None) if spec else None
                    if not pod_containers:
                        raise Exception(f"No containers found in pod {pod_name}")
                    container = pod_containers[0].name

            try:
                resp = controller_client.post(
                    f"/api/v1/namespaces/{namespace}/pods/{pod_name}/exec",
                    json={
                        "command": ["/bin/sh", "-c", exec_command],
                        "container": container,
                    },
                )

                raw_output = resp.get("output", "")
                lines = raw_output.splitlines()
                exit_code = 0

                for i, line in enumerate(lines):
                    if "::EXIT_CODE::" in line:
                        try:
                            exit_code = int(line.split("::EXIT_CODE::")[-1].strip())
                            lines.pop(i)
                            break
                        except ValueError:
                            # If parsing fails, just ignore and leave exit_code = 0
                            pass

                stdout_text = "\n".join(lines)

                if exit_code == 0:
                    ret_codes.append([exit_code, stdout_text, ""])
                else:
                    # On non-zero exit we stuff output into the "stderr" slot,
                    # matching the original behavior.
                    ret_codes.append([exit_code, "", stdout_text])

            except Exception as e:
                raise Exception(f"Failed to execute command {exec_command} on pod {pod_name}: {str(e)}")

    return ret_codes


def _get_rsync_exclude_options() -> str:
    """Get rsync exclude options using .gitignore and/or .ktignore if available."""
    from pathlib import Path

    # Allow users to hard override all of our settings
    if os.environ.get("KT_RSYNC_FILTERS"):
        logger.debug(
            f"KT_RSYNC_FILTERS environment variable set, using rsync filters: {os.environ['KT_RSYNC_FILTERS']}"
        )
        return os.environ["KT_RSYNC_FILTERS"]

    repo_root, _, _ = locate_working_dir(os.getcwd())
    gitignore_path = os.path.join(repo_root, ".gitignore")
    kt_ignore_path = os.path.join(repo_root, ".ktignore")

    exclude_args = ""
    if Path(kt_ignore_path).exists():
        exclude_args += f" --exclude-from='{kt_ignore_path}'"
    if Path(gitignore_path).exists():
        exclude_args += f" --exclude-from='{gitignore_path}'"
    # Add some reasonable default exclusions
    exclude_args += " --exclude='*.pyc' --exclude='__pycache__' --exclude='.venv' --exclude='.git'"

    return exclude_args.strip()


def is_pod_terminated(pod: dict) -> bool:
    """Check if pod is terminated. Pod must be a dict from ControllerClient."""
    # Check if pod is marked for deletion
    deletion_timestamp = pod.get("metadata", {}).get("deletionTimestamp")
    if deletion_timestamp is not None:
        return True

    # Check pod phase
    phase = pod.get("status", {}).get("phase")
    if phase in ["Succeeded", "Failed"]:
        return True

    # Check container statuses
    container_statuses = pod.get("status", {}).get("containerStatuses", [])
    for container in container_statuses:
        state = container.get("state", {})
        if state.get("terminated"):
            return True

    return False


# ----------------- ConfigMap utils ----------------- #
def load_configmaps(
    service_name: str,
    namespace: str,
    console: "Console" = None,
) -> List[str]:
    """List configmaps that start with a given service name."""
    controller_client = kubetorch.globals.controller_client()
    try:
        configmaps = controller_client.list_config_maps(
            namespace=namespace,
            label_selector=f"kubetorch.com/service={service_name}",
        )
        # Handle dict response from ControllerClient
        if isinstance(configmaps, dict):
            return [cm["metadata"]["name"] for cm in configmaps.get("items", [])]
        # Handle object response from CoreV1Api (legacy)
        return [cm.metadata.name for cm in configmaps.items]
    except Exception as e:
        if console:
            console.print(f"[yellow]Warning:[/yellow] Failed to list configmaps: {e}")
        return []


def delete_resources_for_services(
    services: Union[str, List],
    namespace: str = None,
    force: bool = False,
    prefix: Optional[bool] = None,
    teardown_all: Optional[bool] = None,
    username: Optional[str] = None,
    exact_match: Optional[bool] = None,
):
    """Delete the relevant k8s resource(s) based on service type.

    Uses the same teardown path as the Python API (module.teardown() -> service_manager.teardown_service()).
    """

    from kubetorch import globals

    controller_client = globals.controller_client()
    delete_result = controller_client.delete_services(
        namespace=namespace,
        services=services,
        force=force,
        prefix=prefix,
        teardown_all=teardown_all,
        username=username,
        exact_match=exact_match,
    )
    return delete_result


def print_byo_deletion_warning(byo_deleted_services: list, console=None):
    byo_resources_teardown_msg = (
        f"Resources for {','.join(byo_deleted_services)} were created outside Kubetorch. You are responsible for "
        f"deleting the Kubernetes resources (pods, deployments, services, etc.)."
    )

    if console:
        console.print(f"[bold yellow]{byo_resources_teardown_msg}[/bold yellow]")
    else:
        logger.warning(byo_resources_teardown_msg)


def handle_controller_delete_error(service_name: str, controller_error: str, console=None):
    if "404" in controller_error:
        if service_name:
            error_msg = f"Service {service_name.lower()} not found"
        else:
            error_msg = "No services found"
    else:
        error_msg = controller_error.split(":")[-1]

    if console:
        console.print(error_msg)
    else:
        logger.error(error_msg)


def _collect_modules(target_str):
    from kubetorch.resources.callables.module import Module

    to_deploy = []

    if ":" in target_str:
        target_module_or_path, target_fn_or_class = target_str.split(":")
    else:
        target_module_or_path, target_fn_or_class = target_str, None

    if target_module_or_path.endswith(".py"):
        abs_path = Path(target_module_or_path).resolve()
        python_module_name = inspect.getmodulename(str(abs_path))

        sys.path.insert(0, str(abs_path.parent))
    else:
        python_module_name = target_module_or_path
        sys.path.append(".")

    module = importlib.import_module(python_module_name)

    if target_fn_or_class:
        if not hasattr(module, target_fn_or_class):
            raise ValueError(f"Function or class {target_fn_or_class} not found in {target_module_or_path}.")
        to_deploy = [getattr(module, target_fn_or_class)]
        if not isinstance(to_deploy[0], Module):
            raise ValueError(
                f"Function or class {target_fn_or_class} in {target_module_or_path} is not decorated with @kt.compute."
            )
    else:
        # Get all functions and classes to deploy
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, Module):
                to_deploy.append(obj)
        if not to_deploy:
            raise ValueError(f"No functions or classes decorated with @kt.compute found in {target_module_or_path}.")

    return to_deploy, target_fn_or_class


def fetch_resources_for_teardown(
    namespace: str,
    target: str,
    prefix: Optional[str] = None,
    teardown_all: bool = False,
    username: Optional[str] = None,
    exact_match: bool = False,
) -> Dict[str, Any]:
    """Fetches K8s resources that would be deleted for a given service.

    Returns:
        Dict with 'resources' key containing list of resource dicts, each with:
            - name: Resource name
            - kind: K8s kind (e.g., "Deployment", "Service", "RayCluster")
            - api_version: K8s API version (e.g., "apps/v1", "serving.knative.dev/v1")
    """
    from kubetorch.resources.callables.module import Module

    services = None

    if prefix in ["kt", "kubetorch", "knative"]:
        raise ValueError(f"Invalid prefix: {prefix} is reserved. Please delete these individually.")
    if prefix and username:
        raise ValueError("Cannot use both prefix and username flags together.")

    if not (target or teardown_all or prefix):
        raise ValueError("Please provide a service name or use the --all or --prefix flags")

    if target:
        # Case when service_name is a module or file path (i.e. the `kt deploy` usage path)
        if ":" in target or ".py" in target or "." in target:
            to_down, _ = _collect_modules(target)
            services = [mod.service_name for mod in to_down if isinstance(mod, Module)]
        else:
            services = [target]
            # if the target is not prefixed with the username, add the username prefix
            username = kubetorch.globals.config.username
            if username and not exact_match and not target.startswith(username + "-"):
                services.append(username + "-" + target)

    controller_client = kubetorch.globals.controller_client()
    try:
        return controller_client.fetch_resources_for_teardown(
            namespace=namespace,
            name=services,
            prefix=prefix,
            teardown_all=teardown_all,
            username=username,
            exact_match=exact_match,
        )
    except Exception as e:
        logger.warning(f"Failed to fetch resources for teardown: {e}")
        return {"resources": []}


# ----------------- Image Builder Utils ----------------- #
def _get_sync_package_paths(
    package: str,
):
    if "/" in package or "~" in package:
        package_path = (
            Path(package).expanduser()
            if Path(package).expanduser().is_absolute()
            else Path(locate_working_dir()[0]) / package
        )
        dest_dir = str(package_path.name)
    else:
        package_path = get_local_install_path(package)
        dest_dir = package

    if not (package_path and Path(package_path).exists()):
        raise ValueError(f"Could not locate local package {package}")

    full_path = Path(package_path).expanduser().resolve()
    return str(full_path), dest_dir


def is_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) != 0


def find_available_port(start_port: int, max_tries: int = 10) -> int:
    for i in range(max_tries):
        port = start_port + i
        if is_port_available(port):
            return port
    raise RuntimeError(f"Could not find available port starting from {start_port}")


# --------------- Secrets utils ---------------------------


def get_parsed_secret(secret):
    """Parse secret from either dict (ControllerClient) or V1Secret object (CoreV1Api)"""
    # Handle dict response from ControllerClient
    if isinstance(secret, dict):
        metadata = secret.get("metadata", {})
        labels = metadata.get("labels", {})
        return {
            "name": metadata.get("name"),
            "username": labels.get("kubetorch.com/username") if labels else None,
            "namespace": metadata.get("namespace"),
            "user_defined_name": labels.get("kubetorch.com/secret-name") if labels else None,
            "labels": labels,
            "annotations": metadata.get("annotations"),
            "type": secret.get("type"),
            "data": secret.get("data"),
        }
    # Handle object response from CoreV1Api (legacy)
    else:
        labels = secret.metadata.labels
        return {
            "name": secret.metadata.name,
            "username": labels.get("kubetorch.com/username", None) if labels else None,
            "namespace": secret.metadata.namespace,
            "user_defined_name": labels.get("kubetorch.com/secret-name", None) if labels else None,
            "labels": labels,
            "annotations": secret.metadata.annotations,
            "type": secret.type,
            "data": secret.data,
        }


def list_secrets(
    namespace: str = "default",
    prefix: str = None,
    all_namespaces: bool = False,
    filter_by_creator: bool = True,
    console: "Console" = None,
):
    controller_client = kubetorch.globals.controller_client()
    try:
        if all_namespaces:
            try:
                secrets_result = controller_client.list_secrets_all_namespaces()
            except ControllerRequestError as e:
                if e.status_code == 403:
                    msg = (
                        "Cross-namespace secret listing requires additional RBAC permissions. "
                        f"Falling back to namespace-scoped listing for namespace: '{namespace}'"
                    )
                    if console:
                        console.print(f"[yellow]{msg}[/yellow]\n")
                    secrets_result = controller_client.list_secrets(namespace=namespace)
                else:
                    raise
        else:
            secrets_result = controller_client.list_secrets(namespace=namespace)

        # Handle dict response from ControllerClient
        if isinstance(secrets_result, dict):
            secret_items = secrets_result.get("items", [])
            if not secret_items:
                return None
        else:
            # Handle object response from CoreV1Api (legacy)
            if not secrets_result or not secrets_result.items:
                return None
            secret_items = secrets_result.items

        filtered_secrets = []
        for secret in secret_items:
            parsed_secret = get_parsed_secret(secret)
            user_defined_secret_name = parsed_secret.get("user_defined_name")
            if user_defined_secret_name:  # filter secrets that was created by kt api, by the username set in kt.config.
                if prefix and filter_by_creator:  # filter secrets by prefix + creator
                    if (
                        parsed_secret.get("user_defined_name").startswith(prefix)
                        and parsed_secret.get("username") == kubetorch.globals.config.username
                    ):
                        filtered_secrets.append(parsed_secret)
                elif prefix:  # filter secrets by prefix
                    if parsed_secret.get("user_defined_name").startswith(prefix):
                        filtered_secrets.append(parsed_secret)
                elif filter_by_creator:  # filter secrets by creator
                    if parsed_secret.get("username") == kubetorch.globals.config.username:
                        filtered_secrets.append(parsed_secret)
                else:  # No additional filters required
                    filtered_secrets.append(parsed_secret)
        return filtered_secrets

    except Exception as e:
        console.print(f"[red]Failed to load secrets: {e}[/red]")
        return None


def delete_secrets(
    secrets: List[str],
    secrets_client: KubernetesSecretsClient,
    console: "Console" = None,
):
    """Delete the given list of secrets."""
    for secret in secrets:
        secrets_client.delete_secret(secret, console=console)
