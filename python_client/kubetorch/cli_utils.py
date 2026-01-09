import asyncio
import json
import os
import signal

import subprocess
import sys
import threading
import time
import urllib.parse
import warnings
import webbrowser
from contextlib import contextmanager
from enum import Enum
from typing import List

import typer
from rich import box
from rich.console import Console
from rich.style import Style
from rich.table import Table
from websocket import create_connection

import kubetorch.provisioning.constants as provisioning_constants

from kubetorch import globals
from kubetorch.config import KubetorchConfig
from kubetorch.constants import MAX_PORT_TRIES
from kubetorch.provisioning.utils import wait_for_port_forward

from kubetorch.resources.compute.utils import is_port_available
from kubetorch.serving.utils import stream_logs_websocket_helper, StreamType
from kubetorch.utils import hours_to_ns, http_not_found

from .logger import get_logger

console = Console()

logger = get_logger(__name__)


# ------------------ Generic helpers--------------------
class VolumeAction(str, Enum):
    list = "list"
    create = "create"
    delete = "delete"
    ssh = "ssh"


class SecretAction(str, Enum):
    list = "list"
    create = "create"
    delete = "delete"
    describe = "describe"


def default_typer_values(*args):
    """Convert typer model arguments to their default values or types, so the CLI commands can be also imported and
    called in Python if desired."""
    new_args = []
    for arg in args:
        if isinstance(arg, typer.models.OptionInfo):
            # Replace the typer model with its value
            arg = arg.default if arg.default is not None else arg.type
        elif isinstance(arg, typer.models.ArgumentInfo):
            # Replace the typer model with its value
            arg = arg.default if arg.default is not None else arg.type
        new_args.append(arg)
    return new_args


def validate_config_key(key: str = None):
    if key is None:
        return

    valid_keys = {name for name, attr in vars(KubetorchConfig).items() if isinstance(attr, property)}
    if key not in valid_keys:
        raise typer.BadParameter(f"Valid keys are: {', '.join(sorted(valid_keys))}")
    return key


def get_pods_for_service_cli(name: str, namespace: str):
    """Get pods for a service using unified label selector."""
    # Use unified service label - works for all deployment modes
    controller_client = globals.controller_client()
    label_selector = f"kubetorch.com/service={name}"
    try:
        return controller_client.list_pods(
            namespace=namespace,
            label_selector=label_selector,
        )
    except Exception as e:
        if http_not_found(e):
            return {"items": []}
        raise


def service_name_argument(*args, required: bool = True, **kwargs):
    def _lowercase(value: str) -> str:
        return value.lower() if value else value

    default = ... if required else ""
    return typer.Argument(default, callback=_lowercase, *args, **kwargs)


def get_deployment_mode(name: str, namespace: str) -> str:
    """Validate service exists and return deployment mode."""
    try:
        original_name = name
        deployment_mode = detect_deployment_mode(name, namespace)
        # If service not found and not already prefixed with username, try with username prefix
        if not deployment_mode and globals.config.username and not name.startswith(globals.config.username + "-"):
            name = f"{globals.config.username}-{name}"
            deployment_mode = detect_deployment_mode(name, namespace)

        if not deployment_mode:
            console.print(f"[red]Failed to load service [bold]{original_name}[/bold] in namespace {namespace}[/red]")
            raise typer.Exit(1)
        console.print(f"Found [green]{deployment_mode}[/green] service [blue]{name}[/blue]")
        return name, deployment_mode

    except Exception as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


def validate_pods_exist(name: str, namespace: str) -> list:
    """Validate pods exist for service and return pod list."""
    result = get_pods_for_service_cli(name, namespace)
    pods = result.get("items", [])
    if not pods:
        console.print(f"\n[red]No pods found for service {name} in namespace {namespace}[/red]")
        console.print(f"You can view the service's status using:\n [yellow]  kt status {name}[/yellow]")
        raise typer.Exit(1)
    return pods


@contextmanager
def port_forward_to_pod(
    pod_name,
    namespace: str = None,
    local_port: int = 8080,
    remote_port: int = provisioning_constants.DEFAULT_NGINX_PORT,
    health_endpoint: str = None,
):
    for attempt in range(MAX_PORT_TRIES):
        candidate_port = local_port + attempt
        if not is_port_available(candidate_port):
            logger.debug(f"Local port {candidate_port} is already in use, trying again...")
            continue

        cmd = [
            "kubectl",
            "port-forward",
            f"pod/{pod_name}",
            f"{candidate_port}:{remote_port}",
            "--namespace",
            namespace,
        ]
        logger.debug(f"Running port-forward command: {' '.join(cmd)}")

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)

        try:
            wait_for_port_forward(
                process,
                candidate_port,
                health_endpoint=health_endpoint,
                validate_kubetorch_versions=False,
            )
            time.sleep(2)
            yield candidate_port
            return

        finally:
            if process:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    process.wait()
                except (ProcessLookupError, OSError):
                    # Process may have already terminated
                    pass

    raise RuntimeError(f"Could not bind available port after {MAX_PORT_TRIES} attempts")


def get_last_updated(pod):
    conditions = pod["status"].get("conditions", [])
    latest = max(
        (c.get("lastTransitionTime") for c in conditions if c.get("lastTransitionTime")),
        default="",
    )
    return latest


def _get_logs_from_loki_worker(uri: str, print_pod_name: bool, timeout: float = 2.0):
    """Worker function for getting logs from Loki - runs in a separate thread."""
    ws = None
    try:
        # Set timeout on websocket connection to fail fast if no logs available
        ws = create_connection(uri, timeout=timeout)
        message = ws.recv()
        if not message:
            return None
        data = json.loads(message)
        logs = []
        if data.get("streams"):
            for stream in data["streams"]:
                stream_labels = stream.get("stream", {})
                pod_name_value = stream_labels.get("pod") or stream_labels.get("k8s_pod_name")
                pod_name = f"({pod_name_value}) " if print_pod_name and pod_name_value else ""
                for value in stream.get("values"):
                    try:
                        log_line = json.loads(value[1])
                        log_name = log_line.get("name")
                        if log_name == "print_redirect":
                            logs.append(f'{pod_name}{log_line.get("message")}')
                        elif log_name != "uvicorn.access":
                            formatted_log = (
                                f"{pod_name}{log_line.get('asctime')} | {log_line.get('levelname')} | "
                                f"{log_line.get('message')}\n"
                            )
                            logs.append(formatted_log)
                    except Exception:
                        logs.append(f"{pod_name}{value[1]}")
        return logs
    finally:
        if ws:
            try:
                ws.close()
            except Exception:
                pass


def load_logs_for_pod(
    query: str = None,
    uri: str = None,
    print_pod_name: bool = False,
    timeout: float = 2.0,
    namespace: str = None,
):
    """Get logs from Loki with fail-fast approach to avoid hanging.

    Args:
        query (str): LogQL query string. (Default: None)
        uri (str): Direct URI to use. Skips cluster checks. (Default: None)
        print_pod_name (bool): Whether to print pod name with each log. (Default: False)
        timeout (float): Connection timeout. (Default: 2.0)
        namespace (str): Namespace to query logs from (required if uri not provided). (Default: None)
    """
    try:
        # If URI is provided, use it directly (skip cluster checks)
        if uri:
            return _get_logs_from_loki_worker(uri, print_pod_name, timeout)

        import urllib.parse

        # Now safe to proceed with service URL setup
        from kubetorch.utils import http_to_ws

        # Wrap service_url call in daemon thread with timeout
        url_result = [None]

        def get_url():
            try:
                url_result[0] = globals.service_url()
            except Exception:
                # Silence exceptions in daemon thread
                pass

        url_thread = threading.Thread(target=get_url, daemon=True)
        url_thread.start()
        url_thread.join(timeout=5.0)

        if url_thread.is_alive():
            logger.debug("Timeout getting service URL")
            return None

        base_url = url_result[0]
        if not base_url:
            return None

        if not namespace:
            logger.debug("Namespace required for Loki query")
            return None

        start_ns = hours_to_ns()
        # Namespace-aware Loki URL - routes to data store in the target namespace
        target_uri = f"{http_to_ws(base_url)}/loki/{namespace}/api/v1/tail?query={urllib.parse.quote_plus(query)}&start={start_ns}"

        # Use daemon thread so Python exits even if websocket hangs
        result = [None]

        def worker():
            try:
                result[0] = _get_logs_from_loki_worker(target_uri, print_pod_name, timeout)
            except Exception:
                # Silence exceptions in daemon thread
                pass

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout=timeout + 1.0)

        return result[0] if not thread.is_alive() else None

    except Exception as e:
        logger.debug(f"Error getting logs from Loki: {e}")
        return None


def stream_logs_websocket(uri, stop_event, print_pod_name: bool = False):
    """Stream logs using Loki's websocket tail endpoint"""
    # Create and run event loop in a separate thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Suppress asyncio warnings during cleanup
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*coroutine.*was never awaited")
    warnings.filterwarnings("ignore", message=".*asynchronous generator.*")

    try:
        loop.run_until_complete(
            stream_logs_websocket_helper(
                uri=uri,
                stop_event=stop_event,
                stream_type=StreamType.CLI,
                print_pod_name=print_pod_name,
            )
        )
    except KeyboardInterrupt:
        # Set stop event to signal graceful shutdown
        stop_event.set()
    finally:
        # Suppress stderr during cleanup
        import os

        stderr_fd = sys.stderr.fileno()
        old_stderr = os.dup(stderr_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)

        try:
            # Redirect stderr to /dev/null during cleanup
            os.dup2(devnull, stderr_fd)

            # Cancel any remaining tasks
            pending = asyncio.all_tasks(loop)
            if pending:
                for task in pending:
                    task.cancel()

                # Give tasks a very short time to handle cancellation
                try:
                    loop.run_until_complete(
                        asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=0.1)
                    )
                except:
                    pass

            # Close the loop without shutting down async generators
            # (which causes race conditions)
            loop.close()
        except:
            pass
        finally:
            # Restore stderr
            try:
                os.dup2(old_stderr, stderr_fd)
                os.close(old_stderr)
                os.close(devnull)
            except:
                pass

            # Ensure stop event is set
            stop_event.set()


def generate_logs_query(name: str, namespace: str, selected_pod: str, deployment_mode):
    from kubetorch.provisioning.utils import SUPPORTED_TRAINING_JOBS

    if not selected_pod:
        if deployment_mode in ["knative", "deployment"] + SUPPORTED_TRAINING_JOBS:
            # Query by service name and namespace (labels set by LogCapture)
            return f'{{service="{name}", namespace="{namespace}"}}'
        else:
            console.print(f"[red]Logs does not support deployment mode: {deployment_mode}[/red]")
            return None
    else:
        # Query by specific pod name
        return f'{{pod="{selected_pod}", namespace="{namespace}"}}'


def follow_logs_in_cli(
    name: str,
    namespace: str,
    selected_pod: str,
    deployment_mode,
    print_pod_name: bool = False,
):
    """Stream logs when triggered by the CLI."""
    from kubetorch.utils import http_to_ws

    stop_event = threading.Event()

    # Set up signal handler to cleanly stop on Ctrl+C
    def signal_handler(signum, frame):
        stop_event.set()
        raise KeyboardInterrupt()

    original_handler = signal.signal(signal.SIGINT, signal_handler)

    # setting up the query
    query = generate_logs_query(name, namespace, selected_pod, deployment_mode)
    if not query:
        return

    encoded_query = urllib.parse.quote_plus(query)

    start_ns = hours_to_ns()
    base_url = globals.service_url()
    # Namespace-aware Loki URL - routes to data store in the target namespace
    uri = f"{http_to_ws(base_url)}/loki/{namespace}/api/v1/tail?query={encoded_query}&start={start_ns}"

    try:
        stream_logs_websocket(
            uri=uri,
            stop_event=stop_event,
            print_pod_name=print_pod_name,
        )
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_handler)


def is_ingress_vpc_only(annotations: dict):
    # Check for internal LoadBalancer annotations
    internal_checks = [
        annotations.get("service.beta.kubernetes.io/aws-load-balancer-internal") == "true",
        annotations.get("networking.gke.io/load-balancer-type") == "Internal",
        annotations.get("service.beta.kubernetes.io/oci-load-balancer-internal") == "true",
    ]

    vpc_only = any(internal_checks)
    return vpc_only


def load_ingress(namespace: str = globals.config.install_namespace):
    controller = globals.controller_client()

    try:
        data = controller.list_ingresses(namespace=namespace)
        items = data.get("items", [])
        for ing in items:
            if ing["metadata"]["name"] == "kubetorch-controller-ingress":
                return ing

        return None

    except Exception as e:
        if not http_not_found(e):
            logger.error(f"Failed to load ingress: {e}")
        return None


def get_ingress_host(ingress):
    """Get the configured host from the kubetorch ingress."""
    try:
        return ingress.spec.rules[0].host
    except Exception:
        return None


def detect_deployment_mode(name: str, namespace: str):
    """Detect if a service is deployed as Knative, Deployment, or RayCluster."""
    controller_client = globals.controller_client()

    # First try Deployment
    try:
        obj = controller_client.get_deployment(name=name, namespace=namespace)
        if isinstance(obj, dict) and obj.get("kind") == "Deployment":
            return "deployment"
    except Exception as e:
        if not http_not_found(e):
            raise

    # Then try Knative
    try:
        obj = controller_client.get_namespaced_custom_object(
            group="serving.knative.dev",
            version="v1",
            namespace=namespace,
            plural="services",
            name=name,
        )
        if isinstance(obj, dict) and obj.get("kind") == "Service":
            return "knative"
    except Exception as e:
        if not http_not_found(e):
            raise

    # Then try RayCluster
    try:
        obj = controller_client.get_namespaced_custom_object(
            group="ray.io",
            version="v1",
            namespace=namespace,
            plural="rayclusters",
            name=name,
        )
        return "raycluster"
    except Exception as e:
        if not http_not_found(e):
            raise

    # Then try TrainJobs
    from kubetorch.provisioning.utils import SUPPORTED_TRAINING_JOBS

    for kind in SUPPORTED_TRAINING_JOBS:
        try:
            controller_client.get_namespaced_custom_object(
                group="kubeflow.org",
                version="v1",
                namespace=namespace,
                plural=kind.lower() + "s",
                name=name,
            )
            return kind.lower()
        except Exception as e:
            if not http_not_found(e):
                raise

    return None


def load_selected_pod(service_name, provided_pod, service_pods):
    if provided_pod is None:
        return provided_pod

    if provided_pod.isnumeric():
        pod = int(provided_pod)
        if pod < 0 or pod >= len(service_pods):
            console.print(f"[red]Pod index {pod} is out of range[/red]")
            raise typer.Exit(1)
        pod_name = service_pods[pod]["metadata"]["name"]

    # case when the user provides pod name
    else:
        pod_names = [pod["metadata"]["name"] for pod in service_pods]
        if provided_pod not in pod_names:
            console.print(f"[red]{service_name} does not have an associated pod called {provided_pod}[/red]")
            raise typer.Exit(1)
        else:
            pod_name = provided_pod

    return pod_name


def load_kubetorch_volumes_from_pods(pods: list) -> List[str]:
    """Extract volume information from service definition.

    Note: pods should be dicts from controller_client.list_pods().
    """
    volumes = set()

    if not pods:
        return []

    for pod in pods:
        if not isinstance(pod, dict):
            raise TypeError(f"Expected pod to be a dict, got {type(pod)}")

        for vol in pod.get("spec", {}).get("volumes", []) or []:
            pvc = vol.get("persistentVolumeClaim")
            if pvc:
                volumes.add(vol["name"])

    return list(volumes)


def create_table_for_output(columns: List[set], no_wrap_columns_names: list = None, header_style: dict = None):
    table = Table(box=box.SQUARE, header_style=Style(**header_style))
    for name, style in columns:
        if name in no_wrap_columns_names:
            # always make service name fully visible
            table.add_column(name, style=style, no_wrap=True)
        else:
            table.add_column(name, style=style)

    return table


def notebook_placeholder():
    """Placeholder function to launch notebook service"""
    import time

    time.sleep(3600)  # Keep alive for port forwarding


def load_runhouse_dashboard(
    namespace: str,
    local_port=3000,
    local_server=False,
):
    process = []

    if not local_server:
        # Check if the service is available
        cmd = f"kubectl get svc/{provisioning_constants.KUBETORCH_UI_SERVICE_NAME} -n {namespace}"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        if result.returncode != 0:
            console.print("[red]Dashboard UI service is not available[/red]")
            raise typer.Exit(1)

        cmd = f"kubectl port-forward -n {namespace} svc/{provisioning_constants.KUBETORCH_UI_SERVICE_NAME} {local_port}:3000 -n {namespace}"
        process.append(subprocess.Popen(cmd.split()))

    # Check if the controller API service is available
    cmd = f"kubectl get svc/kubetorch-mgmt-controller-api -n {namespace}"
    result = subprocess.run(cmd.split(), capture_output=True, text=True)
    if result.returncode != 0:
        console.print("[red]Control plane API service is not available[/red]")
        raise typer.Exit(1)

    # Add kubectl port-forward -n kubetorch svc/kubetorch-mgmt-controller-api 8000:8000
    cmd = f"kubectl port-forward -n {namespace} svc/kubetorch-mgmt-controller-api 8000:8000"
    process.append(subprocess.Popen(cmd.split()))

    dashboard_url = f"http://localhost:{local_port}"

    time.sleep(1)

    webbrowser.open(dashboard_url)

    return process
