import importlib
import inspect
import os
import socket
import sys
from pathlib import Path
from typing import List, Optional, Union

import kubetorch.globals
from kubetorch.logger import get_logger
from kubetorch.provisioning.constants import KT_SERVICE_LABEL, KT_USERNAME_LABEL
from kubetorch.resources.callables.utils import get_local_install_path, locate_working_dir
from kubetorch.resources.secrets.kubernetes_secrets_client import KubernetesSecretsClient
from kubetorch.serving.utils import StartupError
from kubetorch.utils import http_not_found

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


# ----------------- Resource Deletion Utils ----------------- #
def delete_configmaps(
    configmaps: List[str],
    namespace: str,
    console: "Console" = None,
    force: bool = False,
):
    """Delete the given list of configmaps."""

    grace_period_seconds, propagation_policy = None, None
    if force:
        grace_period_seconds = 0
        propagation_policy = "Foreground"

    controller_client = kubetorch.globals.controller_client()
    for cm in configmaps:
        try:
            controller_client.delete_config_map(
                namespace=namespace,
                name=cm,
                grace_period_seconds=grace_period_seconds,
                propagation_policy=propagation_policy,
            )
            if console:
                console.print(f"✓ Deleted configmap [blue]{cm}[/blue]")
        except Exception as e:
            # Handle both ApiException (legacy) and HTTP errors from controller
            if http_not_found(e):
                if console:
                    console.print(f"[yellow]Warning:[/yellow] ConfigMap {cm} not found")
            else:
                if console:
                    console.print(f"[red]Error:[/red] Failed to delete configmap {cm}: {e}")


def delete_knative_service(
    name: str,
    namespace,
    console: "Console" = None,
    force: bool = False,
):
    """Delete a Knative service."""

    grace_period_seconds, propagation_policy = None, None
    if force:
        grace_period_seconds = 0
        propagation_policy = "Foreground"

    try:
        kubetorch.globals.controller_client().delete_namespaced_custom_object(
            group="serving.knative.dev",
            version="v1",
            namespace=namespace,
            plural="services",
            name=name,
            grace_period_seconds=grace_period_seconds,
            propagation_policy=propagation_policy,
        )
        if console:
            console.print(f"✓ Deleted service [blue]{name}[/blue]")
    except Exception as e:
        if http_not_found(e):
            if console:
                console.print(f"[yellow]Note:[/yellow] Service {name} not found or already deleted")
        else:
            if console:
                console.print(f"[red]Error:[/red] Failed to delete service {name}: {e}")


def delete_resources_for_service(
    configmaps: List[str],
    name: str,
    service_type: str = "knative",
    namespace: str = None,
    console: "Console" = None,
    force: bool = False,
):
    """Delete the relevant k8s resource(s) based on service type.

    Uses the same teardown path as the Python API (module.teardown() -> service_manager.teardown_service()).
    """
    from kubetorch.provisioning.service_manager import ServiceManager
    from kubetorch.provisioning.utils import SUPPORTED_TRAINING_JOBS

    if service_type == "selector":
        # BYO (selector-based) compute mode:
        # The user applied the Kubernetes manifest themselves (e.g., via kubectl, Helm, or ArgoCD).
        # Kubetorch did not create or own the K8s resources, so teardown only removes
        # Kubetorch controller state and associated metadata — not the underlying pods/deployments/services
        msg = (
            f"Resources for {name} were created outside Kubetorch. You are responsible for deleting "
            "the actual Kubernetes resources (pods, deployments, services, etc.)."
        )
        # For selector-based pools, just delete the controller pool (no K8s resource to delete)
        service_manager = ServiceManager(resource_type="selector", namespace=namespace)
        service_manager.teardown_service(service_name=name, console=console, force=force)
        if console:
            console.print(f"[yellow]{msg}[/yellow]")
        else:
            logger.warning(msg)
    else:
        # manifest applied via kubetorch
        supported_types = ["deployment", "raycluster", "knative"] + [k.lower() for k in SUPPORTED_TRAINING_JOBS]
        if service_type in supported_types:
            service_manager = ServiceManager(resource_type=service_type, namespace=namespace)
        else:
            msg = f"Unknown service type: {service_type}, skipping teardown"
            if console:
                console.print(f"[yellow]{msg}[/yellow]")
            else:
                logger.warning(msg)
            return

        # Use the same teardown path as the Python API to tear down
        service_manager.teardown_service(service_name=name, console=console, force=force)

    # Delete configmaps
    if configmaps:
        delete_configmaps(
            configmaps=configmaps,
            namespace=namespace,
            console=console,
            force=force,
        )

    delete_cached_service_data(service_name=name, namespace=namespace, console=console)


def delete_cached_service_data(
    service_name: str,
    namespace: str,
    console: "Console" = None,
):
    """Delete service data from the data store (both filesystem and metadata)."""
    try:
        from kubetorch.data_store import DataStoreClient

        client = DataStoreClient(namespace=namespace)
        client.rm(key=service_name, recursive=True)

        if console:
            console.print(f"✓ Deleted cached data for [blue]{service_name}[/blue]")

    except Exception as e:
        if console:
            console.print(f"[red]Failed to clean up cached service data: {e}[/red]")
        else:
            logger.debug(f"Failed to clean up cached data: {e}")


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
    username: Optional[str] = None,
    exact_match: bool = False,
) -> dict:
    """Fetches the resources for a given service.

    Returns a dictionary with the following keys:
    - services: {
        [service_name]: {
            "configmaps": List[str],
            "pods": List[str],
            "type": str,
        }
    }
    """
    from kubetorch.resources.callables.module import Module

    resources = {"services": {}}
    services = []

    if prefix in ["kt", "kubetorch", "knative"]:
        raise ValueError(f"Invalid prefix: {prefix} is reserved. Please delete these individually.")
    if prefix and username:
        raise ValueError("Cannot use both prefix and username flags together.")

    # Initialize controller client
    controller_client = kubetorch.globals.controller_client()

    if username or prefix:
        label_selector = f"{KT_USERNAME_LABEL}={username}" if username else None
        prefix_filter = prefix if prefix else None

        try:
            discovered = controller_client.discover_resources(
                namespace=namespace,
                label_selector=label_selector,
                prefix_filter=prefix_filter,
            )

            TYPE_MAP = {
                "knative_services": "knative",
                "deployments": "deployment",
                "rayclusters": "raycluster",
            }
            for resource_type, svc_type in TYPE_MAP.items():
                for item in discovered.get(resource_type, []):
                    name = item.get("metadata", {}).get("name", "")
                    if name:
                        services.append((name, svc_type, None))

            # Training jobs - get type from kind
            for item in discovered.get("training_jobs", []):
                name = item.get("metadata", {}).get("name", "")
                kind = item.get("kind", "").lower()
                if name:
                    services.append((name, kind, "kubeflow.org"))

            # Pools need lookup to determine type
            for pool in discovered.get("pools", []):
                name = pool.get("name", "")
                if name:
                    services.append((name, None, None))

        except Exception as e:
            logger.warning(f"Failed to discover resources: {e}")

    else:
        if not target:
            raise ValueError("Please provide a service name or use the --all or --prefix flags")

        # Case when service_name is a module or file path (i.e. the `kt deploy` usage path)
        if ":" in target or ".py" in target or "." in target:
            to_down, _ = _collect_modules(target)
            services = [(mod.service_name, None, None) for mod in to_down if isinstance(mod, Module)]
        else:
            services = [(target, None, None)]
            # if the target is not prefixed with the username, add the username prefix
            username = kubetorch.globals.config.username
            if username and not exact_match and not target.startswith(username + "-"):
                services.append((username + "-" + target, None, None))

    for service_name, type_hint, group_hint in services:
        service_type = type_hint
        service_group = group_hint
        pool_selector = None

        # Pools are handled via controller - we delete the pool from DB, controller handles K8s cleanup
        # Skip pool lookup if we already have type info from discovery
        if not service_type:  # Not found yet
            try:
                pool_info = controller_client.get_pool(namespace=namespace, name=service_name)
                if pool_info:
                    specifier = pool_info.get("specifier") or {}
                    # Determine service type based on resource_kind AND whether it's KT-managed
                    # KT-managed pools have the kubetorch.com/template label (applied via /apply)
                    # Selector-only pools may have resource_kind (discovered from pods) but no template label
                    resource_kind = pool_info.get("resource_kind")
                    pool_labels = pool_info.get("labels") or {}
                    is_kt_managed = serving_constants.KT_TEMPLATE_LABEL in pool_labels

                    if resource_kind and is_kt_managed:
                        # KT-managed resource: delete the K8s resource
                        # Map resource_kind to service_type (lowercase)
                        # Most kinds just need lowercasing: Deployment->deployment, RayCluster->raycluster
                        # Training jobs: PyTorchJob->pytorchjob, TFJob->tfjob, etc.
                        service_type = resource_kind.lower()
                        # Handle Knative special case: KnativeService -> knative
                        if service_type == "knativeservice":
                            service_type = "knative"
                    else:
                        # Selector-only: user created K8s resource, only delete controller state
                        service_type = "selector"
                    pool_selector = specifier.get("selector")
            except Exception as e:
                logger.debug(f"Pool lookup for {service_name} failed: {e}")

        # Check if it's a Knative service
        if not service_type:
            try:
                service = controller_client.get_namespaced_custom_object(
                    group="serving.knative.dev",
                    version="v1",
                    namespace=namespace,
                    plural="services",
                    name=service_name,
                    ignore_not_found=True,
                )
                if (
                    isinstance(service, dict)
                    and service.get("kind") == "Service"
                    and service.get("metadata", {}).get("name") == service_name
                ):
                    service_type = "knative"
            except Exception:
                pass

        # Check if it's a Deployment (if not found as Knative service)
        if not service_type:
            try:
                deployment = controller_client.get_deployment(
                    namespace=namespace, name=service_name, ignore_not_found=True
                )
                if isinstance(deployment, dict) and deployment.get("kind") == "Deployment":
                    service_type = "deployment"
            except Exception:
                pass

        # Check if it's a RayCluster (if not found as Knative or Deployment)
        if not service_type:
            try:
                raycluster = controller_client.get_namespaced_custom_object(
                    group="ray.io",
                    version="v1",
                    namespace=namespace,
                    plural="rayclusters",
                    name=service_name,
                    ignore_not_found=True,
                )
                if (
                    isinstance(raycluster, dict)
                    and raycluster.get("kind") == "RayCluster"
                    and raycluster.get("metadata", {}).get("name") == service_name
                ):
                    service_type = "raycluster"
            except Exception:
                pass

        # Check if it's a custom training job (PyTorchJob, TFJob, MXJob, XGBoostJob) if not found as other types
        if not service_type:
            from kubetorch.provisioning.utils import SUPPORTED_TRAINING_JOBS

            for job_kind in SUPPORTED_TRAINING_JOBS:
                try:
                    plural = job_kind.lower() + "s"
                    job_resource = controller_client.get_namespaced_custom_object(
                        group="kubeflow.org",
                        version="v1",
                        namespace=namespace,
                        plural=plural,
                        name=service_name,
                        ignore_not_found=True,
                    )
                    if job_resource:
                        service_type = job_kind.lower()
                        service_group = "kubeflow.org"
                        break
                except Exception:
                    continue

        # Get associated resources if service exists
        configmaps = load_configmaps(service_name, namespace)
        pods = []
        try:
            # For selector-based pools, use the pool selector to find pods
            if pool_selector:
                label_selector = ",".join(f"{k}={v}" for k, v in pool_selector.items())
            else:
                label_selector = f"{KT_SERVICE_LABEL}={service_name}"

            pods_result = controller_client.list_pods(namespace=namespace, label_selector=label_selector)
            # Handle dict response from ControllerClient
            if pods_result:
                pods = [pod["metadata"]["name"] for pod in pods_result.get("items", [])]
        except Exception:
            pass

        # Only add the service to the resources if it has configmaps, pods, or we found the service
        if service_type or configmaps or pods:
            resources["services"][service_name] = {
                "configmaps": configmaps,
                "pods": pods,
                "type": service_type or "unknown",
                "group": service_group,
            }

    return resources


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
