import importlib
import inspect
import os
import socket
import sys
from pathlib import Path
from typing import List, Optional, Union

import kubetorch.globals
from kubetorch.logger import get_logger
from kubetorch.resources.callables.utils import get_local_install_path, locate_working_dir
from kubetorch.resources.secrets.kubernetes_secrets_client import KubernetesSecretsClient
from kubetorch.servers.http.utils import StartupError
from kubetorch.serving import constants as serving_constants
from kubetorch.serving.constants import KT_SERVICE_LABEL, KT_USERNAME_LABEL
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
    """Raised when a request to the Kubetorch controller fails."""

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


def delete_deployment(
    name: str,
    namespace: str,
    console: "Console" = None,
):
    """Delete a Deployment and its associated service."""
    controller_client = kubetorch.globals.controller_client()
    try:
        # Delete the Deployment
        controller_client.delete_deployment(
            name=name,
            namespace=namespace,
        )
        if console:
            console.print(f"✓ Deleted deployment [blue]{name}[/blue]")
    except Exception as e:
        if http_not_found(e):
            if console:
                console.print(f"[yellow]Note:[/yellow] Deployment {name} not found or already deleted")
        else:
            if console:
                console.print(f"[red]Error:[/red] Failed to delete deployment {name}: {e}")

    # Delete the associated service (regular service, not headless)
    try:
        controller_client.delete_service(
            namespace=namespace,
            name=name,
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

    try:
        headless = controller_client.get_service(namespace=namespace, name=f"{name}-headless", ignore_not_found=True)
    except Exception:
        headless = None

    if headless:
        try:
            controller_client.delete_service(
                namespace=namespace,
                name=f"{name}-headless",
            )
            if console:
                console.print(f"✓ Deleted headless service [blue]{name}-headless[/blue]")
        except Exception as e:
            if not http_not_found(e) and console:
                console.print(f"[red]Error:[/red] Failed to delete headless service {name}-headless: {e}")


def delete_raycluster(
    name: str,
    namespace: str,
    console: "Console" = None,
    force: bool = False,
):
    """Delete a RayCluster and its associated service."""

    grace_period_seconds, propagation_policy = None, None
    if force:
        grace_period_seconds = 0
        propagation_policy = "Foreground"

    try:
        # Delete the RayCluster
        kubetorch.globals.controller_client().delete_namespaced_custom_object(
            group="ray.io",
            version="v1",
            namespace=namespace,
            plural="rayclusters",
            name=name,
            grace_period_seconds=grace_period_seconds,
            propagation_policy=propagation_policy,
        )
        if console:
            console.print(f"✓ Deleted RayCluster [blue]{name}[/blue]")
    except Exception as e:
        if http_not_found(e):
            if console:
                console.print(f"[yellow]Note:[/yellow] RayCluster {name} not found or already deleted")
        else:
            if console:
                console.print(f"[red]Error:[/red] Failed to delete RayCluster {name}: {e}")

    # Delete the associated service (created alongside RayCluster)
    try:
        kubetorch.globals.controller_client().delete_service(
            namespace=namespace,
            name=name,
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

    # Delete the headless service for Ray pod discovery
    try:
        kubetorch.globals.controller_client().delete_service(
            namespace=namespace,
            name=f"{name}-headless",
        )
        if console:
            console.print(f"✓ Deleted headless service [blue]{name}-headless[/blue]")
    except Exception as e:
        if http_not_found(e):
            # This is normal for older Ray clusters without headless services
            pass
        else:
            if console:
                console.print(f"[red]Error:[/red] Failed to delete headless service {name}-headless: {e}")


def delete_trainjob(
    name: str,
    namespace: str,
    group: str,
    plural: str,
    console: "Console" = None,
    force: bool = False,
):
    """Delete a manifest and its associated service."""

    grace_period_seconds, propagation_policy = None, None
    if force:
        grace_period_seconds = 0
        propagation_policy = "Foreground"

    manifest_type = plural[:-1]

    try:
        # Delete the manifest
        kubetorch.globals.controller_client().delete_namespaced_custom_object(
            group=group,
            version="v1",
            namespace=namespace,
            plural=plural,
            name=name,
            grace_period_seconds=grace_period_seconds,
            propagation_policy=propagation_policy,
        )
        if console:
            console.print(f"✓ Deleted {manifest_type} [blue]{name}[/blue]")
    except Exception as e:
        if http_not_found(e):
            if console:
                console.print(f"[yellow]Note:[/yellow] {manifest_type} {name} not found or already deleted")
        else:
            if console:
                console.print(f"[red]Error:[/red] Failed to delete {manifest_type} {name}: {e}")

    # Delete the associated services (created alongside the trainjob)
    associated_services = kubetorch.globals.controller_client().list_services(
        namespace=namespace, label_selector=f"kubetorch.com/service={name}"
    )
    associated_services = associated_services.get("items", [])
    if len(associated_services) > 0:
        if console:
            console.print(f"Deleting services associated with [reset]{name}")
        for service in associated_services:
            associated_service_name = service["metadata"]["name"]
            try:
                kubetorch.globals.controller_client().delete_service(
                    namespace=namespace,
                    name=associated_service_name,
                )
                if console:
                    console.print(f"✓ Deleted service [blue]{associated_service_name}[/blue]")
            except Exception as e:
                if http_not_found(e):
                    pass
                else:
                    if console:
                        console.print(f"[red]Error:[/red] Failed to delete {associated_service_name}: {e}")


def delete_resources_for_service(
    configmaps: List[str],
    name: str,
    service_type: str = "knative",
    namespace: str = None,
    console: "Console" = None,
    force: bool = False,
    group: str = None,
):
    """Delete service resources based on service type."""
    # Delete the main service (Knative, Deployment, or RayCluster)
    if service_type == "deployment":
        delete_deployment(
            name=name,
            namespace=namespace,
            console=console,
        )
    elif service_type == "raycluster":
        delete_raycluster(
            name=name,
            namespace=namespace,
            console=console,
            force=force,
        )
    elif service_type == "knative":
        delete_knative_service(
            name=name,
            namespace=namespace,
            console=console,
            force=force,
        )
    elif group:  # service is a training job
        delete_trainjob(
            name=name, namespace=namespace, console=console, force=force, group=group, plural=f"{service_type}s"
        )

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

    # Initialize controller client
    controller_client = kubetorch.globals.controller_client()

    if username or prefix:
        # Search Knative services
        try:
            # Build label selector for Knative services - use template label to identify kubetorch services
            knative_label_selector = f"{serving_constants.KT_TEMPLATE_LABEL}=ksvc"
            if username:
                knative_label_selector += f",{KT_USERNAME_LABEL}={username}"

            try:
                response = controller_client.list_namespaced_custom_object(
                    group="serving.knative.dev",
                    version="v1",
                    namespace=namespace,
                    plural="services",
                    label_selector=knative_label_selector,
                    ignore_not_found=True,
                )
                items = response.get("items", []) if response else []
                knative_services = [
                    item["metadata"]["name"]
                    for item in items
                    if (username or item["metadata"]["name"].startswith(prefix))
                ]
                services.extend(knative_services)

            except Exception as e:
                logger.warning(f"Could not load knative services: {str(e)}")

        except Exception as e:
            if not http_not_found(e):
                logger.warning(f"Failed to list Knative services: {e}")

        # Search Deployments
        try:
            # Build label selector for deployments - use KT_TEMPLATE_LABEL to identify kubetorch deployments
            deployment_label_selector = f"{serving_constants.KT_TEMPLATE_LABEL}=deployment"
            if username:
                deployment_label_selector += f",{KT_USERNAME_LABEL}={username}"

            deployments_response = controller_client.list_deployments(
                namespace=namespace,
                label_selector=deployment_label_selector,
            )
            items = deployments_response.get("items", []) if deployments_response else []
            deployment_services = [
                item["metadata"]["name"] for item in items if (username or item["metadata"]["name"].startswith(prefix))
            ]
            services.extend(deployment_services)
        except Exception as e:
            # Catch all errors from controller client (ControllerRequestError) or legacy K8s client (ApiException)
            logger.warning(f"Failed to list Deployments: {e}")

        # Search RayClusters
        try:
            # Build label selector for rayclusters - use template label to identify kubetorch rayclusters
            raycluster_label_selector = f"{serving_constants.KT_TEMPLATE_LABEL}=raycluster"
            if username:
                raycluster_label_selector += f",{KT_USERNAME_LABEL}={username}"

            try:
                response = controller_client.list_namespaced_custom_object(
                    group="ray.io",
                    version="v1",
                    namespace=namespace,
                    plural="rayclusters",
                    label_selector=raycluster_label_selector,
                    ignore_not_found=True,
                )
                items = response.get("items", []) if response else []
                raycluster_services = [
                    item["metadata"]["name"]
                    for item in items
                    if (username or item["metadata"]["name"].startswith(prefix))
                ]
                services.extend(raycluster_services)

            except Exception as e:
                logger.warning(f"Could not load raycluster services: {str(e)}")

        except Exception as e:
            if not http_not_found(e):
                logger.warning(f"Failed to list RayClusters: {e}")

        from kubetorch.serving.trainjob_service_manager import TrainJobServiceManager

        for job_kind in TrainJobServiceManager.SUPPORTED_KINDS:
            try:
                plural = job_kind.lower() + "s"
                if username:
                    label_selector = f"{KT_USERNAME_LABEL}={username}"
                else:
                    label_selector = None

                if label_selector:
                    response = controller_client.list_namespaced_custom_object(
                        group="kubeflow.org",
                        version="v1",
                        namespace=namespace,
                        plural=plural,
                        label_selector=label_selector,
                        ignore_not_found=True,
                    )
                else:
                    # Search all jobs when no username filter
                    response = controller_client.list_namespaced_custom_object(
                        group="kubeflow.org",
                        version="v1",
                        namespace=namespace,
                        plural=plural,
                        ignore_not_found=True,
                    )

                items = response.get("items", []) if response else []
                # Filter by prefix if provided, and ensure it's a kubetorch service (has template label)
                job_services = []
                for item in items:
                    item_name = item["metadata"]["name"]
                    labels = item.get("metadata", {}).get("labels", {})
                    template_label = labels.get(serving_constants.KT_TEMPLATE_LABEL)
                    # Check if it's a kubetorch service (has template label with value matching job kind or "generic")
                    if template_label in (job_kind.lower(), "generic"):
                        # If prefix is provided, check if name starts with prefix
                        if prefix and item_name.startswith(prefix):
                            job_services.append(item_name)
                        elif username and labels.get(KT_USERNAME_LABEL) == username:
                            job_services.append(item_name)
                services.extend(job_services)
            except Exception as e:
                if http_not_found(e):  # Ignore if Kubeflow Training Operator is not installed
                    logger.warning(f"Failed to list {job_kind}s: {e}")

    else:
        if not target:
            raise ValueError("Please provide a service name or use the --all or --prefix flags")

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

    for service_name in services:
        service_type = None
        service_found = False
        service_group = None

        # Check if it's a Knative service
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
                service_found = True

        except Exception:
            pass

        # Check if it's a Deployment (if not found as Knative service)
        if not service_found:
            try:
                deployment = controller_client.get_deployment(
                    namespace=namespace, name=service_name, ignore_not_found=True
                )
                if isinstance(deployment, dict) and deployment.get("kind") == "Deployment":
                    service_type = "deployment"
                    service_found = True
            except Exception:
                pass

        # Check if it's a RayCluster (if not found as Knative or Deployment)
        if not service_found:
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
                    service_found = True
            except Exception:
                pass

        # Check if it's a custom training job (PyTorchJob, TFJob, MXJob, XGBoostJob) if not found as other types
        if not service_found:
            from kubetorch.serving.trainjob_service_manager import TrainJobServiceManager

            for job_kind in TrainJobServiceManager.SUPPORTED_KINDS:
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
                        service_found = True
                        service_group = "kubeflow.org"
                        break
                except Exception:
                    continue

        # Get associated resources if service exists
        configmaps = load_configmaps(service_name, namespace)
        pods = []
        try:
            pods_result = controller_client.list_pods(
                namespace=namespace, label_selector=f"{KT_SERVICE_LABEL}={service_name}"
            )
            # Handle dict response from ControllerClient
            if pods_result:
                pods = [pod["metadata"]["name"] for pod in pods_result.get("items", [])]
        except Exception:
            pass

        # Only add the service to the resources if it has configmaps, pods, or we found the service
        if service_found or configmaps or pods:
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


# ----------------- Error Handling Utils ----------------- #
def check_pod_status_for_errors(pod):
    """Check pod status for errors"""

    def _get(obj, attr, default=None):
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)

    status = _get(pod, "status", {})
    conditions = _get(status, "conditions", []) or []

    for condition in conditions:
        ctype = condition.get("type")
        cstatus = condition.get("status")
        creason = condition.get("reason")
        cmessage = condition.get("message", "")

        if ctype == "PodScheduled" and cstatus == "False" and creason == "Unschedulable":

            # Same logic as before…
            msg = cmessage.lower()

            has_autoscaler_taints = any(x in cmessage for x in ["scheduling.cast.ai/node-template"])

            if not has_autoscaler_taints and any(
                x in msg
                for x in [
                    "node selector not matched",
                    "node affinity mismatch",
                    "unsupported gpu type",
                    "unknown instance type",
                ]
            ):
                raise ResourceNotAvailableError(f"Required compute resources are not configured: {cmessage}")

    # Check container status errors
    container_statuses = status.get("containerStatuses", []) or []
    for cs in container_statuses:
        state = cs.get("state", {})
        waiting = state.get("waiting")
        if waiting:
            reason = waiting.get("reason")
            message = waiting.get("message", "")

            # For BackOff/CrashLoopBackOff, get the actual crash reason from lastState
            if reason in ("BackOff", "CrashLoopBackOff"):
                last_state = cs.get("lastState", {})
                terminated = last_state.get("terminated", {})
                if terminated:
                    exit_code = terminated.get("exitCode", "unknown")
                    term_reason = terminated.get("reason", "")
                    term_message = terminated.get("message", "")
                    # Build more informative error message
                    message = f"{message} (exit code: {exit_code}"
                    if term_reason:
                        message += f", reason: {term_reason}"
                    if term_message:
                        message += f", error: {term_message}"
                    message += ")"

            if reason in TERMINATE_EARLY_ERRORS:
                metadata = pod.get("metadata", {})
                raise TERMINATE_EARLY_ERRORS[reason](f"Pod {metadata.get('name')}: {message}")


def check_pod_events_for_errors(pod, namespace: str):
    """Check pod events for scheduling errors"""
    meta = pod.metadata if hasattr(pod, "metadata") else pod.get("metadata", {})
    name = meta.get("name")

    if not name:
        logger.warning("Pod missing metadata.name")
        return

    controller_client = kubetorch.globals.controller_client()
    try:
        events_obj = controller_client.list_events(
            namespace=namespace,
            field_selector=f"involvedObject.name={name}",
        )

        events = events_obj.get("items", [])
        for event in events:
            # event is dict also → normalize
            reason = event.get("reason", "")
            source = event.get("source", {})
            message = event.get("message", "")

            if (
                reason == "FailedScheduling"
                and source.get("component") == "karpenter"
                and "no instance type has enough resources" in message
            ):
                raise ResourceNotAvailableError(f"Pod {name} failed to schedule: {message}")

    except Exception as e:
        logger.warning(f"Error fetching events for pod {name}: {e}")


def check_replicaset_events_for_errors(
    namespace: str,
    service_name: str,
):
    """Check ReplicaSet events for creation errors like missing PriorityClass."""
    controller_client = kubetorch.globals.controller_client()
    try:
        # Get ReplicaSets associated with this Deployment
        resp = controller_client.list_namespaced_replica_set(
            namespace=namespace,
            label_selector=f"kubetorch.com/service={service_name}",
        )

        replicasets = resp.get("items", [])

        for rs in replicasets:
            rs_name = rs.get("metadata", {}).get("name")
            if not rs_name:
                continue

            # Get events for this ReplicaSet
            events_obj = controller_client.list_events(
                namespace=namespace,
                field_selector=f"involvedObject.name={rs_name}",
            )

            events = events_obj.get("items", [])

            for event in events:
                reason = event.get("reason", "")
                etype = event.get("type", "")
                message = event.get("message", "") or ""

                if reason == "FailedCreate" and etype == "Warning" and "forbidden" in message.lower():
                    # PriorityClass-specific error
                    if "priorityclass" in message.lower():
                        raise ResourceNotAvailableError(
                            f"ReplicaSet {rs_name} failed to create pods: "
                            f"{message}. Please ensure the required PriorityClass exists."
                        )

                    # Other forbidden errors
                    if any(err in message.lower() for err in ["forbidden", "no priorityclass", "priority class"]):
                        raise ResourceNotAvailableError(
                            f"ReplicaSet {rs_name} failed to create pods: "
                            f"{message}. Please check cluster configuration and permissions."
                        )

    except ResourceNotAvailableError:
        raise
    except Exception as e:
        logger.warning(f"Error checking ReplicaSet events for {service_name}: {e}")


def check_revision_for_errors(revision_name: str, namespace: str):
    """Check revision for errors"""
    try:
        revision = kubetorch.globals.controller_client().get_namespaced_custom_object(
            group="serving.knative.dev",
            version="v1",
            namespace=namespace,
            plural="revisions",
            name=revision_name,
        )
        for cond in revision.get("status", {}).get("conditions", []):
            if cond["status"] == "False":
                reason = cond.get("reason")
                message = cond.get("message", f"Revision failed with reason: {reason}")
                if reason in TERMINATE_EARLY_ERRORS:
                    raise TERMINATE_EARLY_ERRORS[reason](f"Revision {revision_name}: {message}")
    except Exception as e:
        logger.warning(f"Error checking revision: {e}")


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
