import importlib
import inspect
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Union

from kubernetes import client
from kubernetes.client.rest import ApiException
from kubernetes.stream import stream

import kubetorch.globals
from kubetorch.logger import get_logger
from kubetorch.resources.callables.utils import (
    get_local_install_path,
    locate_working_dir,
)
from kubetorch.resources.secrets.kubernetes_secrets_client import (
    KubernetesSecretsClient,
)
from kubetorch.servers.http.utils import is_running_in_kubernetes, StartupError
from kubetorch.serving import constants as serving_constants
from kubetorch.serving.constants import KT_SERVICE_LABEL, KT_USERNAME_LABEL

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


class QueueUnschedulableError(KnativeServiceError):
    """Raised when the service pod is unschedulable in the requested queue."""

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
        super().__init__(
            f"kubetorch secret {secret_name} was not found in {namespace} namespace"
        )


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
    core_api: "CoreV1Api",
    pod_names: List[str],
    namespace: str,
    container: str = None,
):
    if isinstance(commands, str):
        commands = [commands]
    commands = [
        ["/bin/sh", "-c", f'{command}; echo "::EXIT_CODE::$?"'] for command in commands
    ]

    if isinstance(pod_names, str):
        pod_names = [pod_names]

    ret_codes = []
    for exec_command in commands:
        for pod_name in pod_names:
            if not container:
                pod = core_api.read_namespaced_pod(name=pod_name, namespace=namespace)
                if not pod.spec.containers:
                    raise Exception(f"No containers found in pod {pod_name}")
                container = pod.spec.containers[0].name
            try:
                resp = stream(
                    core_api.connect_get_namespaced_pod_exec,
                    pod_name,
                    namespace,
                    container=container,
                    command=exec_command,
                    stderr=True,
                    stdin=False,
                    stdout=True,
                )

                resp = resp.splitlines()
                exit_code = 0

                for line in resp:
                    if "::EXIT_CODE::" in line:
                        try:
                            exit_code = int(line.split("::EXIT_CODE::")[-1].strip())
                            resp.remove(line)
                            break
                        except ValueError:
                            pass

                stdout = "\n".join(resp)

                if exit_code == 0:
                    ret_codes.append([exit_code, stdout, ""])
                else:
                    ret_codes.append([exit_code, "", stdout])

            except Exception as e:
                raise Exception(
                    f"Failed to execute command {exec_command} on pod {pod_name}: {str(e)}"
                )
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

    repo_root, _ = locate_working_dir(os.getcwd())
    gitignore_path = os.path.join(repo_root, ".gitignore")
    kt_ignore_path = os.path.join(repo_root, ".ktignore")

    exclude_args = ""
    if Path(kt_ignore_path).exists():
        exclude_args += f" --exclude-from='{kt_ignore_path}'"
    if Path(gitignore_path).exists():
        exclude_args += f" --exclude-from='{gitignore_path}'"
    # Add some reasonable default exclusions
    exclude_args += (
        " --exclude='*.pyc' --exclude='__pycache__' --exclude='.venv' --exclude='.git'"
    )

    return exclude_args.strip()


def is_pod_terminated(pod: client.V1Pod) -> bool:
    # Check if pod is marked for deletion
    if pod.metadata.deletion_timestamp is not None:
        return True

    # Check pod phase
    if pod.status.phase in ["Succeeded", "Failed"]:
        return True

    # Check container statuses
    if pod.status.container_statuses:
        for container in pod.status.container_statuses:
            if container.state.terminated:
                return True

    return False


# ----------------- ConfigMap utils ----------------- #
def load_configmaps(
    core_api: client.CoreV1Api,
    service_name: str,
    namespace: str,
    console: "Console" = None,
) -> List[str]:
    """List configmaps that start with a given service name."""
    try:
        configmaps = core_api.list_namespaced_config_map(
            namespace=namespace,
            label_selector=f"kubetorch.com/service={service_name}",
        )
        return [cm.metadata.name for cm in configmaps.items]
    except ApiException as e:
        if console:
            console.print(f"[yellow]Warning:[/yellow] Failed to list configmaps: {e}")
        return []


# ----------------- Resource Deletion Utils ----------------- #
def delete_configmaps(
    core_api: client.CoreV1Api,
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

    for cm in configmaps:
        try:
            core_api.delete_namespaced_config_map(
                name=cm,
                namespace=namespace,
                grace_period_seconds=grace_period_seconds,
                propagation_policy=propagation_policy,
            )
            if console:
                console.print(f"✓ Deleted configmap [blue]{cm}[/blue]")
        except ApiException as e:
            if e.status == 404:
                if console:
                    console.print(f"[yellow]Warning:[/yellow] ConfigMap {cm} not found")
            else:
                if console:
                    console.print(
                        f"[red]Error:[/red] Failed to delete configmap {cm}: {e}"
                    )


def delete_service(
    custom_api: client.CustomObjectsApi,
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
        custom_api.delete_namespaced_custom_object(
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
        if e.status == 404:
            if console:
                console.print(
                    f"[yellow]Note:[/yellow] Service {name} not found or already deleted"
                )
        else:
            if console:
                console.print(f"[red]Error:[/red] Failed to delete service {name}: {e}")


def delete_deployment(
    apps_v1_api: client.AppsV1Api,
    core_api: client.CoreV1Api,
    name: str,
    namespace: str,
    console: "Console" = None,
    force: bool = False,
):
    """Delete a Deployment and its associated service."""
    grace_period_seconds, propagation_policy = None, None
    if force:
        grace_period_seconds = 0
        propagation_policy = "Foreground"
    try:
        # Delete the Deployment
        apps_v1_api.delete_namespaced_deployment(
            name=name,
            namespace=namespace,
            grace_period_seconds=grace_period_seconds,
            propagation_policy=propagation_policy,
        )
        if console:
            console.print(f"✓ Deleted deployment [blue]{name}[/blue]")
    except ApiException as e:
        if e.status == 404:
            if console:
                console.print(
                    f"[yellow]Note:[/yellow] Deployment {name} not found or already deleted"
                )
        else:
            if console:
                console.print(
                    f"[red]Error:[/red] Failed to delete deployment {name}: {e}"
                )

    # Delete the associated service (regular service, not headless)
    try:
        core_api.delete_namespaced_service(
            name=name,
            namespace=namespace,
            grace_period_seconds=grace_period_seconds,
            propagation_policy=propagation_policy,
        )
        if console:
            console.print(f"✓ Deleted service [blue]{name}[/blue]")
    except ApiException as e:
        if e.status == 404:
            if console:
                console.print(
                    f"[yellow]Note:[/yellow] Service {name} not found or already deleted"
                )
        else:
            if console:
                console.print(f"[red]Error:[/red] Failed to delete service {name}: {e}")

    # Also try to delete the headless service for distributed deployments
    try:
        core_api.delete_namespaced_service(
            name=f"{name}-headless",
            namespace=namespace,
            grace_period_seconds=grace_period_seconds,
            propagation_policy=propagation_policy,
        )
        if console:
            console.print(f"✓ Deleted headless service [blue]{name}-headless[/blue]")
    except ApiException as e:
        if e.status == 404:
            # This is normal for non-distributed deployments
            pass
        else:
            if console:
                console.print(
                    f"[red]Error:[/red] Failed to delete headless service {name}-headless: {e}"
                )


def delete_raycluster(
    custom_api: client.CustomObjectsApi,
    core_api: client.CoreV1Api,
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
        custom_api.delete_namespaced_custom_object(
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
    except ApiException as e:
        if e.status == 404:
            if console:
                console.print(
                    f"[yellow]Note:[/yellow] RayCluster {name} not found or already deleted"
                )
        else:
            if console:
                console.print(
                    f"[red]Error:[/red] Failed to delete RayCluster {name}: {e}"
                )

    # Delete the associated service (created alongside RayCluster)
    try:
        core_api.delete_namespaced_service(
            name=name,
            namespace=namespace,
            grace_period_seconds=grace_period_seconds,
            propagation_policy=propagation_policy,
        )
        if console:
            console.print(f"✓ Deleted service [blue]{name}[/blue]")
    except ApiException as e:
        if e.status == 404:
            if console:
                console.print(
                    f"[yellow]Note:[/yellow] Service {name} not found or already deleted"
                )
        else:
            if console:
                console.print(f"[red]Error:[/red] Failed to delete service {name}: {e}")

    # Delete the headless service for Ray pod discovery
    try:
        core_api.delete_namespaced_service(
            name=f"{name}-headless",
            namespace=namespace,
            grace_period_seconds=grace_period_seconds,
            propagation_policy=propagation_policy,
        )
        if console:
            console.print(f"✓ Deleted headless service [blue]{name}-headless[/blue]")
    except ApiException as e:
        if e.status == 404:
            # This is normal for older Ray clusters without headless services
            pass
        else:
            if console:
                console.print(
                    f"[red]Error:[/red] Failed to delete headless service {name}-headless: {e}"
                )


def delete_resources_for_service(
    core_api: client.CoreV1Api,
    custom_api: client.CustomObjectsApi,
    configmaps: List[str],
    name: str,
    service_type: str = "knative",
    namespace: str = None,
    console: "Console" = None,
    force: bool = False,
):
    """Delete service resources based on service type."""
    # Delete the main service (Knative, Deployment, or RayCluster)
    if service_type == "deployment":
        apps_v1_api = client.AppsV1Api()
        delete_deployment(
            apps_v1_api=apps_v1_api,
            core_api=core_api,
            name=name,
            namespace=namespace,
            console=console,
            force=force,
        )
    elif service_type == "raycluster":
        delete_raycluster(
            custom_api=custom_api,
            core_api=core_api,
            name=name,
            namespace=namespace,
            console=console,
            force=force,
        )
    else:  # knative or unknown - try deleting as Knative service
        delete_service(
            custom_api=custom_api,
            name=name,
            namespace=namespace,
            console=console,
            force=force,
        )

    # Delete configmaps
    if configmaps:
        delete_configmaps(
            core_api=core_api,
            configmaps=configmaps,
            namespace=namespace,
            console=console,
            force=force,
        )

    delete_cached_service_data(
        core_api=core_api, service_name=name, namespace=namespace, console=console
    )


def delete_cached_service_data(
    core_api: client.CoreV1Api,
    service_name: str,
    namespace: str,
    console: "Console" = None,
):
    """Delete service data from the rsync pod."""
    try:
        # Find the rsync pod name in the provided namespace
        pods = core_api.list_namespaced_pod(
            namespace=namespace, label_selector="app=kubetorch-rsync"
        )

        if not pods.items:
            if console:
                console.print(
                    f"[yellow] No rsync pod found in namespace {namespace}[/yellow]"
                )
            return

        pod_name = pods.items[0].metadata.name
        service_path = f"/data/{namespace}/{service_name}"

        shell_cmd = (
            f"if [ -d '{service_path}' ]; then rm -rf '{service_path}' && echo 'Deleted {service_path}'; "
            f"else echo 'Path {service_path} not found'; fi"
        )

        # Execute command based on environment
        if is_running_in_kubernetes():
            response = stream(
                core_api.connect_get_namespaced_pod_exec,
                name=pod_name,
                namespace=namespace,
                command=["sh", "-c", shell_cmd],
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False,
            )
            output = response.strip()

        else:
            cmd = [
                "kubectl",
                "exec",
                "-n",
                namespace,
                pod_name,
                "--",
                "sh",
                "-c",
                shell_cmd,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode != 0:
                if console:
                    console.print(
                        f"[red]Error cleaning up cached data: {result.stderr}[/red]"
                    )
                return
            output = result.stdout.strip()

        if console:
            if "Deleted" in output:
                console.print(f"✓ Deleted cached data for [blue]{service_name}[/blue]")

    except subprocess.TimeoutExpired:
        if console:
            console.print("[red]Timeout while cleaning up cached service data[/red]")
        else:
            logger.debug("Timeout while cleaning up cached data")

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
            raise ValueError(
                f"Function or class {target_fn_or_class} not found in {target_module_or_path}."
            )
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
            raise ValueError(
                f"No functions or classes decorated with @kt.compute found in {target_module_or_path}."
            )

    return to_deploy, target_fn_or_class


def fetch_resources_for_teardown(
    namespace: str,
    target: str,
    core_api: client.CoreV1Api,
    custom_api: client.CustomObjectsApi,
    prefix: Optional[str] = None,
    username: Optional[str] = None,
    exact_match: bool = False,
) -> dict:
    """Fetchs the resources for a given service.

    Returns a dictionary with the following keys:
    - services: {
        [service_name]: {
            "configmaps": List[str],
            "pods": List[str],
            "type": str,  # "knative" or "deployment"
        }
    }
    """
    from kubetorch.resources.callables.module import Module

    resources = {"services": {}}
    services = []

    if prefix in ["kt", "kubetorch", "knative"]:
        raise ValueError(
            f"Invalid prefix: {prefix} is reserved. Please delete these individually."
        )

    # Initialize apps API for deployments
    apps_v1_api = client.AppsV1Api()

    if username or prefix:
        # Search Knative services
        try:
            # Build label selector for Knative services - use template label to identify kubetorch services
            knative_label_selector = f"{serving_constants.KT_TEMPLATE_LABEL}=ksvc"
            if username:
                knative_label_selector += f",{KT_USERNAME_LABEL}={username}"

            response = custom_api.list_namespaced_custom_object(
                group="serving.knative.dev",
                version="v1",
                namespace=namespace,
                plural="services",
                label_selector=knative_label_selector,
            )
            items = response.get("items", [])
            knative_services = [
                item["metadata"]["name"]
                for item in items
                if (username or item["metadata"]["name"].startswith(prefix))
            ]
            services.extend(knative_services)
        except client.exceptions.ApiException as e:
            if e.status != 404:  # Ignore if Knative is not installed
                logger.warning(f"Failed to list Knative services: {e}")

        # Search Deployments
        try:
            # Build label selector for deployments - use KT_TEMPLATE_LABEL to identify kubetorch deployments
            deployment_label_selector = (
                f"{serving_constants.KT_TEMPLATE_LABEL}=deployment"
            )
            if username:
                deployment_label_selector += f",{KT_USERNAME_LABEL}={username}"

            deployments_response = apps_v1_api.list_namespaced_deployment(
                namespace=namespace,
                label_selector=deployment_label_selector,
            )
            deployment_services = [
                deployment.metadata.name
                for deployment in deployments_response.items
                if (username or deployment.metadata.name.startswith(prefix))
            ]
            services.extend(deployment_services)
        except client.exceptions.ApiException as e:
            logger.warning(f"Failed to list Deployments: {e}")

        # Search RayClusters
        try:
            # Build label selector for rayclusters - use template label to identify kubetorch rayclusters
            raycluster_label_selector = (
                f"{serving_constants.KT_TEMPLATE_LABEL}=raycluster"
            )
            if username:
                raycluster_label_selector += f",{KT_USERNAME_LABEL}={username}"

            response = custom_api.list_namespaced_custom_object(
                group="ray.io",
                version="v1",
                namespace=namespace,
                plural="rayclusters",
                label_selector=raycluster_label_selector,
            )
            items = response.get("items", [])
            raycluster_services = [
                item["metadata"]["name"]
                for item in items
                if (username or item["metadata"]["name"].startswith(prefix))
            ]
            services.extend(raycluster_services)
        except client.exceptions.ApiException as e:
            if e.status != 404:  # Ignore if Ray operator is not installed
                logger.warning(f"Failed to list RayClusters: {e}")

    else:
        if not target:
            raise ValueError(
                "Please provide a service name or use the --all or --prefix flags"
            )

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

        # Check if it's a Knative service
        try:
            service = custom_api.get_namespaced_custom_object(
                group="serving.knative.dev",
                version="v1",
                namespace=namespace,
                plural="services",
                name=service_name,
            )
            if service:
                service_type = "knative"
                service_found = True
        except client.exceptions.ApiException:
            pass

        # Check if it's a Deployment (if not found as Knative service)
        if not service_found:
            try:
                deployment = apps_v1_api.read_namespaced_deployment(
                    name=service_name, namespace=namespace
                )
                # Only consider it if it has kubetorch template label
                if (
                    deployment.metadata.labels
                    and deployment.metadata.labels.get(
                        serving_constants.KT_TEMPLATE_LABEL
                    )
                    == "deployment"
                ):
                    service_type = "deployment"
                    service_found = True
            except client.exceptions.ApiException:
                pass

        # Check if it's a RayCluster (if not found as Knative or Deployment)
        if not service_found:
            try:
                raycluster = custom_api.get_namespaced_custom_object(
                    group="ray.io",
                    version="v1",
                    namespace=namespace,
                    plural="rayclusters",
                    name=service_name,
                )
                if raycluster:
                    service_type = "raycluster"
                    service_found = True
            except client.exceptions.ApiException:
                pass

        # Get associated resources if service exists
        configmaps = load_configmaps(core_api, service_name, namespace)
        pods = core_api.list_namespaced_pod(
            namespace=namespace, label_selector=f"{KT_SERVICE_LABEL}={service_name}"
        )
        pods = [pod.metadata.name for pod in pods.items]

        # Only add the service to the resources if it has configmaps, pods, or we found the service
        if service_found or configmaps or pods:
            resources["services"][service_name] = {
                "configmaps": configmaps,
                "pods": pods,
                "type": service_type or "unknown",
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
def check_pod_status_for_errors(
    pod: client.V1Pod, queue_name: str = None, scheduler_name: str = None
):
    """Check pod status for errors"""
    # Check for scheduling issues
    for condition in pod.status.conditions or []:
        if (
            condition.type == "PodScheduled"
            and condition.status == "False"
            and condition.reason == "Unschedulable"
        ):
            msg = condition.message.lower()

            # Check if the pod is scheduled in the correct queue and scheduler
            if queue_name and scheduler_name:
                scheduler = pod.metadata.annotations.get("schedulerName", "")
                queue_label = pod.metadata.labels.get("kai.scheduler/queue")
                if queue_label == queue_name and scheduler == scheduler_name:
                    raise QueueUnschedulableError(
                        f"Pod {pod.metadata.name} could not be scheduled: {condition.message}"
                    )

            # Check for specific node selector/affinity/GPU type mismatches
            # without matching temporary resource exhaustion messages
            if any(
                x in msg
                for x in [
                    "node selector not matched",
                    "node affinity mismatch",
                    "unsupported gpu type",
                    "unknown instance type",
                ]
            ):
                raise ResourceNotAvailableError(
                    f"Required compute resources are not configured in the cluster: {condition.message}"
                )

    # Check for container status errors
    if pod.status.container_statuses:
        for container_status in pod.status.container_statuses:
            if container_status.state and container_status.state.waiting:
                reason = container_status.state.waiting.reason
                message = container_status.state.waiting.message or ""
                if reason in TERMINATE_EARLY_ERRORS:
                    raise TERMINATE_EARLY_ERRORS[reason](
                        f"Pod {pod.metadata.name}: {message}"
                    )


def check_pod_events_for_errors(
    pod: client.V1Pod, namespace: str, core_api: client.CoreV1Api
):
    """Check pod events for scheduling errors"""
    try:
        events = core_api.list_namespaced_event(
            namespace=namespace,
            field_selector=f"involvedObject.name={pod.metadata.name}",
        ).items
        for event in events:
            # Check for Karpenter scheduling errors
            if (
                event.reason == "FailedScheduling"
                and event.source.component == "karpenter"
                and "no instance type has enough resources" in event.message
            ):
                raise ResourceNotAvailableError(
                    f"Pod {pod.metadata.name} failed to schedule: {event.message}"
                )
    except client.exceptions.ApiException as e:
        logger.warning(f"Error fetching events for pod {pod.metadata.name}: {e}")


def check_replicaset_events_for_errors(
    namespace: str,
    service_name: str,
    apps_v1_api: client.AppsV1Api,
    core_api: client.CoreV1Api,
):
    """Check ReplicaSet events for creation errors like missing PriorityClass.

    Args:
        service_name: Name of the service
        core_api: Core API instance

    Raises:
        ResourceNotAvailableError: If ReplicaSet creation fails due to missing resources
    """
    try:
        # Get ReplicaSets associated with this Deployment
        replicasets = apps_v1_api.list_namespaced_replica_set(
            namespace=namespace,
            label_selector=f"kubetorch.com/service={service_name}",
        ).items

        for replicaset in replicasets:
            # Check ReplicaSet events for FailedCreate errors
            events = core_api.list_namespaced_event(
                namespace=namespace,
                field_selector=f"involvedObject.name={replicaset.metadata.name}",
            ).items

            for event in events:
                if (
                    event.reason == "FailedCreate"
                    and event.type == "Warning"
                    and "forbidden" in event.message.lower()
                ):
                    # Check for specific PriorityClass errors
                    if "priorityclass" in event.message.lower():
                        raise ResourceNotAvailableError(
                            f"ReplicaSet {replicaset.metadata.name} failed to create pods: "
                            f"{event.message}. Please ensure the required PriorityClass exists in the cluster."
                        )
                    # Check for other forbidden errors
                    elif any(
                        error_type in event.message.lower()
                        for error_type in [
                            "forbidden",
                            "no priorityclass",
                            "priority class",
                        ]
                    ):
                        raise ResourceNotAvailableError(
                            f"ReplicaSet {replicaset.metadata.name} failed to create pods: "
                            f"{event.message}. Please check cluster configuration and permissions."
                        )

    except client.exceptions.ApiException as e:
        logger.warning(f"Error checking ReplicaSet events for {service_name}: {e}")
    except ResourceNotAvailableError:
        # Re-raise ResourceNotAvailableError to stop the readiness check
        raise


def check_revision_for_errors(
    revision_name: str, namespace: str, objects_api: client.CustomObjectsApi
):
    """Check revision for errors"""
    try:
        revision = objects_api.get_namespaced_custom_object(
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
                    raise TERMINATE_EARLY_ERRORS[reason](
                        f"Revision {revision_name}: {message}"
                    )
    except client.exceptions.ApiException as e:
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


def get_parsed_secret(secret: client.V1Secret):
    labels = secret.metadata.labels
    secret = {
        "name": secret.metadata.name,
        "username": labels.get("kubetorch.com/username", None) if labels else None,
        "namespace": secret.metadata.namespace,
        "user_defined_name": labels.get("kubetorch.com/secret-name", None)
        if labels
        else None,
        "labels": labels,
        "annotations": secret.metadata.annotations,
        "type": secret.type,
        "data": secret.data,
    }
    return secret


def list_secrets(
    core_api: client.CoreV1Api,
    namespace: str = "default",
    prefix: str = None,
    all_namespaces: bool = False,
    filter_by_creator: bool = True,
    console: "Console" = None,
):
    try:
        if all_namespaces:
            secrets: client.V1SecretList = core_api.list_secret_for_all_namespaces()
        else:
            secrets: client.V1SecretList = core_api.list_namespaced_secret(
                namespace=namespace
            )
        if not secrets:
            return None
        filtered_secrets = []
        for secret in secrets.items:
            parsed_secret = get_parsed_secret(secret)
            user_defined_secret_name = parsed_secret.get("user_defined_name")
            if (
                user_defined_secret_name
            ):  # filter secrets that was created by kt api, by the username set in kt.config.
                if prefix and filter_by_creator:  # filter secrets by prefix + creator
                    if (
                        parsed_secret.get("user_defined_name").startswith(prefix)
                        and parsed_secret.get("username")
                        == kubetorch.globals.config.username
                    ):
                        filtered_secrets.append(parsed_secret)
                elif prefix:  # filter secrets by prefix
                    if parsed_secret.get("user_defined_name").startswith(prefix):
                        filtered_secrets.append(parsed_secret)
                elif filter_by_creator:  # filter secrets by creator
                    if (
                        parsed_secret.get("username")
                        == kubetorch.globals.config.username
                    ):
                        filtered_secrets.append(parsed_secret)
                else:  # No additional filters required
                    filtered_secrets.append(parsed_secret)
        return filtered_secrets

    except client.rest.ApiException as e:
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
