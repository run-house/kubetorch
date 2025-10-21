import asyncio
import base64
import hashlib
import json
import os
import signal

import subprocess
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import List, Optional

import httpx
import typer
import yaml
from kubernetes import client
from kubernetes.client.rest import ApiException
from pydantic import BaseModel
from rich import box
from rich.console import Console
from rich.style import Style
from rich.table import Table
from websocket import create_connection

import kubetorch.serving.constants as serving_constants

from kubetorch import globals
from kubetorch.config import KubetorchConfig
from kubetorch.constants import MAX_PORT_TRIES

from kubetorch.resources.compute.utils import is_port_available
from kubetorch.servers.http.utils import stream_logs_websocket_helper, StreamType
from kubetorch.serving.utils import wait_for_port_forward
from kubetorch.utils import load_kubeconfig

from .constants import BULLET_UNICODE, CPU_RATE, DOUBLE_SPACE_UNICODE, GPU_RATE

from .logger import get_logger

console = Console()

logger = get_logger(__name__)

OTEL_ERROR_MSG = (
    "[red]Grafana setup failed. Is `kubetorch-otel` installed? See "
    "https://www.run.house/kubetorch/advanced-installation/#kubetorch-telemetry-helm-chart for more info.[/red]"
)


# ------------------ Billing helpers--------------------
class UsageData(BaseModel):
    date_start: str
    date_end: str
    cpu_hours: float
    gpu_hours: float


class BillingTotals(BaseModel):
    cpu: float
    gpu: float


class BillingCosts(BaseModel):
    cpu: float
    gpu: float


class BillingRequest(BaseModel):
    license_key: str
    signature: str
    file_name: str
    username: Optional[str] = None
    usage_data: UsageData
    totals: BillingTotals
    costs: BillingCosts


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

    valid_keys = {
        name
        for name, attr in vars(KubetorchConfig).items()
        if isinstance(attr, property)
    }
    if key not in valid_keys:
        raise typer.BadParameter(f"Valid keys are: {', '.join(sorted(valid_keys))}")
    return key


def get_pods_for_service_cli(name: str, namespace: str, core_api):
    """Get pods for a service using unified label selector."""
    # Use unified service label - works for all deployment modes
    label_selector = f"kubetorch.com/service={name}"
    return core_api.list_namespaced_pod(
        namespace=namespace,
        label_selector=label_selector,
    )


def service_name_argument(*args, required: bool = True, **kwargs):
    def _lowercase(value: str) -> str:
        return value.lower() if value else value

    default = ... if required else ""
    return typer.Argument(default, callback=_lowercase, *args, **kwargs)


def get_deployment_mode(name: str, namespace: str, custom_api, apps_v1_api) -> str:
    """Validate service exists and return deployment mode."""
    try:
        original_name = name
        deployment_mode = detect_deployment_mode(
            name, namespace, custom_api, apps_v1_api
        )
        # If service not found and not already prefixed with username, try with username prefix
        if (
            not deployment_mode
            and globals.config.username
            and not name.startswith(globals.config.username + "-")
        ):
            name = f"{globals.config.username}-{name}"
            deployment_mode = detect_deployment_mode(
                name, namespace, custom_api, apps_v1_api
            )

        if not deployment_mode:
            console.print(
                f"[red]Failed to load service [bold]{original_name}[/bold] in namespace {namespace}[/red]"
            )
            raise typer.Exit(1)
        console.print(
            f"Found [green]{deployment_mode}[/green] service [blue]{name}[/blue]"
        )
        return name, deployment_mode

    except ApiException as e:
        console.print(f"[red]Kubernetes API error: {e}[/red]")
        raise typer.Exit(1)


def validate_pods_exist(name: str, namespace: str, core_api) -> list:
    """Validate pods exist for service and return pod list."""
    pods = get_pods_for_service_cli(name, namespace, core_api)
    if not pods.items:
        console.print(
            f"\n[red]No pods found for service {name} in namespace {namespace}[/red]"
        )
        console.print(
            f"You can view the service's status using:\n [yellow]  kt status {name}[/yellow]"
        )
        raise typer.Exit(1)
    return pods.items


@contextmanager
def port_forward_to_pod(
    pod_name,
    namespace: str = None,
    local_port: int = 8080,
    remote_port: int = serving_constants.DEFAULT_NGINX_PORT,
    health_endpoint: str = None,
):

    load_kubeconfig()
    for attempt in range(MAX_PORT_TRIES):
        candidate_port = local_port + attempt
        if not is_port_available(candidate_port):
            logger.debug(
                f"Local port {candidate_port} is already in use, trying again..."
            )
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

        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True
        )

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
        (
            c.get("lastTransitionTime")
            for c in conditions
            if c.get("lastTransitionTime")
        ),
        default="",
    )
    return latest


# ------------------ Reporting helpers--------------------
def upload_report(
    usage_data: dict,
    signature: str,
    costs: BillingCosts,
    totals: BillingTotals,
    file_name: str,
    license_key: str,
    username: str = None,
):
    billing_request = BillingRequest(
        license_key=license_key,
        signature=signature,
        file_name=file_name,
        username=username,
        usage_data=UsageData(**usage_data),
        costs=costs,
        totals=totals,
    )

    url = "https://auth.run.house/v1/billing/report"
    resp = httpx.post(url, json=billing_request.model_dump())
    if resp.status_code != 200:
        console.print("[red]Failed to send billing report[/red]")
        raise typer.Exit(1)


def export_report_pdf(report_data, filename):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        console.print(
            "[red]ReportLab is required for downloading the report. Please install it "
            "with `pip install reportlab`.[/red]"
        )
        raise typer.Exit(1)

    usage_data: dict = report_data["usage_report"]
    report_str = json.dumps(report_data, sort_keys=True)
    signature = base64.b64encode(hashlib.sha256(report_str.encode()).digest()).decode()

    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    # Sidebar
    sidebar_color = colors.HexColor("#4B9CD3")
    c.setFillColor(sidebar_color)
    c.roundRect(0, 0, 18, height, 0, fill=1, stroke=0)
    c.setFillColor(colors.black)

    y = height - 60

    # Header Title
    c.setFont("Helvetica-Bold", 26)
    c.setFillColor(sidebar_color)
    c.drawCentredString(width / 2, y, "Kubetorch Usage Report")
    c.setFillColor(colors.black)
    y -= 30

    # Header Bar
    c.setStrokeColor(sidebar_color)
    c.setLineWidth(2)
    c.line(40, y, width - 40, y)
    y -= 20

    # Info Box
    c.setFillColor(colors.whitesmoke)
    c.roundRect(40, y - 60, width - 80, 60, 8, fill=1, stroke=0)
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(55, y - 20, "Username:")
    c.drawString(55, y - 35, "Cluster:")
    c.setFont("Helvetica", 12)
    c.drawString(130, y - 20, report_data["username"])
    c.drawString(130, y - 35, report_data.get("cluster_name", "N/A"))
    y -= 100

    # Usage Summary Section
    c.setFont("Helvetica-Bold", 15)
    c.setFillColor(sidebar_color)
    c.drawString(40, y, "Usage Summary")
    c.setFillColor(colors.black)
    y -= 25

    # Table Outline (dashed)
    table_left = 40
    table_width = width - 80
    row_height = 18
    num_rows = 2  # header + data
    table_height = row_height * num_rows

    # Table Header (centered text)
    header_height = row_height
    c.setFillColor(sidebar_color)
    c.roundRect(
        table_left, y - header_height, table_width, header_height, 4, fill=1, stroke=0
    )
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(colors.white)
    header_y = y - header_height + 5
    c.drawString(table_left + 10, header_y, "Start Date")
    c.drawString(table_left + 90, header_y, "End Date")
    c.drawString(table_left + 200, header_y, "vCPU Hours")
    c.drawString(table_left + 300, header_y, "GPU Hours")
    c.setFillColor(colors.black)

    # Dashed outline (starts at header, not above)
    c.setStrokeColor(sidebar_color)
    c.setDash(4, 4)
    c.roundRect(
        table_left, y - table_height, table_width, table_height, 6, fill=0, stroke=1
    )
    c.setDash()
    y -= header_height

    # Table Rows
    c.setFont("Helvetica", 10)
    y -= row_height
    c.drawString(table_left + 10, y + 5, usage_data["date_start"])
    c.drawString(table_left + 90, y + 5, usage_data["date_end"])
    c.drawRightString(table_left + 270, y + 5, f"{usage_data['cpu_hours']:.2f}")
    c.drawRightString(table_left + 370, y + 5, f"{usage_data['gpu_hours']:.2f}")

    y -= 30

    # Invoice Calculation
    total_cpu = usage_data["cpu_hours"]
    total_gpu = usage_data["gpu_hours"]
    cpu_cost = total_cpu * CPU_RATE
    gpu_cost = total_gpu * GPU_RATE
    total_cost = cpu_cost + gpu_cost

    y -= 20
    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(sidebar_color)
    c.drawString(40, y, "Invoice Summary")
    c.setFillColor(colors.black)
    y -= 18

    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Total vCPU Hours: {total_cpu:.2f} @ ${CPU_RATE:.2f}/hr")
    c.drawRightString(width - 50, y, f"${cpu_cost:.2f}")
    y -= 15
    c.drawString(50, y, f"Total GPU Hours: {total_gpu:} @ ${GPU_RATE:.2f}/hr")
    c.drawRightString(width - 50, y, f"${gpu_cost:.2f}")
    y -= 15

    line_left = 50
    line_right = width - 50
    c.setStrokeColor(sidebar_color)
    c.setLineWidth(1.5)
    c.line(line_left, y, line_right, y)
    y -= 15

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Total Due:")
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(colors.HexColor("#008000"))
    c.drawRightString(width - 50, y, f"${total_cost:.2f}")
    c.setFillColor(colors.black)
    y -= 30

    # Signature and footer
    sig_y = 80
    sig_val_y = sig_y - 15
    footer_y = sig_val_y - 40

    # Signature at the bottom
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(colors.black)
    c.drawString(40, sig_y, "Signature:")
    c.setFont("Courier-Oblique", 8)
    c.setFillColor(colors.HexColor("#888888"))
    c.drawString(40, sig_val_y, signature)
    c.setFillColor(colors.black)

    # Footer
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColor(colors.HexColor("#888888"))
    c.drawString(
        40, footer_y, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    c.setFillColor(colors.black)

    c.save()
    return signature


def print_usage_table(usage_data, cluster_name):
    table = Table(title="Usage Summary")
    table.add_column("Start Date")
    table.add_column("End Date")
    table.add_column("vCPU Hours")
    table.add_column("GPU Hours")
    table.add_row(
        usage_data["date_start"],
        usage_data["date_end"],
        str(usage_data["cpu_hours"]),
        str(usage_data["gpu_hours"]),
    )
    console.print(table)
    console.print(f"[dim]Cluster: {str(cluster_name)}[/dim]")


def get_last_n_calendar_weeks(n_weeks):
    """Return a list of (week_start, week_end) tuples for the last n full calendar weeks (Mon–Sun),
    not including this week."""
    today = datetime.utcnow().date()
    # Find the most recent Monday before today (not including today if today is Monday)
    if today.weekday() == 0:
        last_monday = today - timedelta(days=7)
    else:
        last_monday = today - timedelta(days=today.weekday())

    weeks = []
    for i in range(n_weeks):
        week_start = last_monday - timedelta(weeks=i)
        week_end = week_start + timedelta(days=6)  # Monday + 6 = Sunday
        weeks.append((week_start, week_end))

    weeks.reverse()  # So oldest week is first
    return weeks


def get_usage_data(prom, weeks):
    from datetime import datetime, timedelta

    days = weeks * 7
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)

    # sum of CPU-seconds used by all cores for that container (ex: 2 cores for 1 second = 2 seconds)
    cpu_query = (
        f'increase(container_cpu_usage_seconds_total{{container="kubetorch"}}[{days}d])'
    )
    cpu_result = prom.custom_query(cpu_query)

    total_cpu_seconds = 0
    if cpu_result:
        for series in cpu_result:
            cpu_val = float(series["value"][1])
            total_cpu_seconds += cpu_val

    cpu_hours = round(total_cpu_seconds / 3600, 2)

    # requested GPUs × time they were running
    gpu_query = f'sum_over_time(kube_pod_container_resource_requests{{resource="nvidia_com_gpu", container="kubetorch"}}[{days}d])'
    gpu_result = prom.custom_query(gpu_query)

    # Convert to "GPU hours" over the period
    total_gpu_seconds = sum(float(s["value"][1]) for s in gpu_result or [])
    gpu_hours = total_gpu_seconds / 3600

    usage = {
        "date_start": start_time.strftime("%Y-%m-%d"),
        "date_end": end_time.strftime("%Y-%m-%d"),
        "cpu_hours": round(cpu_hours, 2),
        "gpu_hours": round(gpu_hours, 2),
    }

    return usage


# ------------------ Monitoring helpers--------------------
def get_service_metrics(prom, pod_name: str, pod_node: str, running_on_gpu: bool):
    """Get CPU, GPU (if relevant) and memory metrics for a pod"""

    def extract_prometheus_metric_value(query) -> float:
        result = prom.custom_query(query=query)
        return float(result[0].get("value")[1]) if result else 0

    # --- CPU metrics --- #
    cpu_query_time_window = "30s"
    used_cpu_query = (
        f"sum(rate(container_cpu_usage_seconds_total{{pod='{pod_name}', "
        f"container='kubetorch'}}[{cpu_query_time_window}]))"
    )
    requested_cpu_query = (
        f"sum(kube_pod_container_resource_requests{{pod='{pod_name}', "
        f"resource='cpu', container='kubetorch'}})"
    )

    used_cpu_result = extract_prometheus_metric_value(used_cpu_query)

    requested_cpu_result = extract_prometheus_metric_value(requested_cpu_query)

    cpu_util = (
        round((100 * (used_cpu_result / requested_cpu_result)), 3)
        if used_cpu_result and requested_cpu_result
        else 0
    )

    memory_usage_query = f"container_memory_usage_bytes{{pod='{pod_name}', container='kubetorch'}} / 1073741824"
    memory_usage = round(extract_prometheus_metric_value(memory_usage_query), 3)

    machine_mem_query = (
        f"machine_memory_bytes{{node='{pod_node}'}} / 1073741824"  # convert to GB
    )
    machine_mem_result = extract_prometheus_metric_value(machine_mem_query)

    cpu_mem_percent = (
        round((memory_usage / machine_mem_result) * 100, 3) if machine_mem_result else 0
    )
    collected_metrics = {
        "cpu_util": cpu_util,
        "used_cpu": round(used_cpu_result, 4),
        "requested_cpu": round(requested_cpu_result, 4),
        "cpu_memory_usage": memory_usage,
        "cpu_memory_total": round(machine_mem_result, 3),
        "cpu_memory_usage_percent": cpu_mem_percent,
    }

    # --- GPU metrics --- #
    if running_on_gpu:
        gpu_util_query = f"DCGM_FI_DEV_GPU_UTIL{{exported_pod='{pod_name}', exported_container='kubetorch'}}"
        gpu_mem_used_query = (
            f"DCGM_FI_DEV_FB_USED{{exported_pod='{pod_name}', "
            f"exported_container='kubetorch'}} * 1.048576 / 1000"
        )  # convert MiB to MB to GB
        gpu_mem_free_query = (
            f"DCGM_FI_DEV_FB_FREE{{exported_pod='{pod_name}', "
            f"exported_container='kubetorch'}} * 1.048576 / 1000"
        )  # convert MiB to MB to GB

        gpu_util = extract_prometheus_metric_value(gpu_util_query)
        gpu_mem_used = round(extract_prometheus_metric_value(gpu_mem_used_query), 3)
        gpu_mem_free = extract_prometheus_metric_value(gpu_mem_free_query)
        gpu_mem_total = gpu_mem_free + gpu_mem_used
        gpu_mem_percent = (
            round(100 * (gpu_mem_used / gpu_mem_total), 2) if gpu_mem_used else 0
        )  # raw approximation, because total_allocated_gpu_memory is not collected

        gpu_metrics = {
            "gpu_util": gpu_util,
            "gpu_memory_usage": gpu_mem_used,
            "gpu_memory_total": round(gpu_mem_total, 3),
            "gpu_memory_usage_percent": gpu_mem_percent,
        }

        collected_metrics.update(gpu_metrics)

    return collected_metrics


def get_current_cluster_name():
    try:
        from kubernetes import config as k8s_config

        k8s_config.load_incluster_config()
        # In-cluster: return a generic name or the service host
        return os.environ.get("CLUSTER_NAME", "in-cluster")
    except Exception:
        pass

    # Fallback to kubeconfig file
    kubeconfig_path = os.getenv("KUBECONFIG") or str(Path.home() / ".kube" / "config")
    if not os.path.exists(kubeconfig_path):
        return None

    with open(kubeconfig_path, "r") as f:
        kubeconfig = yaml.safe_load(f)
    current_context = kubeconfig.get("current-context")
    for context in kubeconfig.get("contexts", []):
        if context["name"] == current_context:
            return context["context"]["cluster"]
    return None


def print_pod_info(pod_name, pod_idx, is_gpu, metrics=None, queue_name=None):
    """Print pod info with metrics if available"""
    queue_msg = f" | [bold]Queue Name[/bold]: {queue_name}"
    base_msg = (
        f"{BULLET_UNICODE} [reset][bold cyan]{pod_name}[/bold cyan] (idx: {pod_idx})"
    )
    if queue_name:
        base_msg += queue_msg
    console.print(base_msg)
    if metrics:
        console.print(
            f"{DOUBLE_SPACE_UNICODE}[bold]CPU[/bold]: [reset]{metrics['cpu_util']}% "
            f"({metrics['used_cpu']} / {metrics['requested_cpu']}) | "
            f"[bold]Memory[/bold]: {metrics['cpu_memory_usage']} / {metrics['cpu_memory_total']} "
            f"[bold]GB[/bold] ({metrics['cpu_memory_usage_percent']}%)"
        )
        if is_gpu:
            console.print(
                f"{DOUBLE_SPACE_UNICODE}GPU: [reset]{metrics['gpu_util']}% | "
                f"Memory: {metrics['gpu_memory_usage']} / {metrics['gpu_memory_total']} "
                f"GB ({metrics['gpu_memory_usage_percent']}%)"
            )
    else:
        console.print(f"{DOUBLE_SPACE_UNICODE}[yellow]Metrics unavailable[/yellow]")


def _get_logs_from_loki_worker(uri: str, print_pod_name: bool):
    """Worker function for getting logs from Loki - runs in a separate thread."""
    ws = None
    try:
        ws = create_connection(uri)
        message = ws.recv()
        if not message:
            return None
        data = json.loads(message)
        logs = []
        if data.get("streams"):
            for stream in data["streams"]:
                pod_name = (
                    f'({stream.get("stream").get("pod")}) ' if print_pod_name else ""
                )
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
                        logs.append(value[1])
        return logs
    finally:
        if ws:
            try:
                ws.close()
            except Exception:
                pass


def get_logs_from_loki(
    query: str = None,
    uri: str = None,
    print_pod_name: bool = False,
    timeout: float = 5.0,
):
    """Get logs from Loki with fail-fast approach to avoid hanging."""
    try:
        # If URI is provided, use it directly (skip cluster checks)
        if uri:
            return _get_logs_from_loki_worker(uri, print_pod_name)

        import urllib.parse

        # Now safe to proceed with service URL setup
        from kubetorch import globals
        from kubetorch.utils import http_to_ws

        base_url = globals.service_url()
        target_uri = f"{http_to_ws(base_url)}/loki/api/v1/tail?query={urllib.parse.quote_plus(query)}"

        # Use thread timeout for websocket worker since websocket timeouts don't work reliably
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(
                _get_logs_from_loki_worker, target_uri, print_pod_name
            )
            try:
                result = future.result(timeout=timeout)
                return result
            except TimeoutError:
                logger.debug(f"Loki websocket connection timed out after {timeout}s")
                return None
            except Exception as e:
                logger.debug(f"Error in Loki websocket worker: {e}")
                return None
        finally:
            # Don't wait for stuck threads to complete
            executor.shutdown(wait=False)

    except Exception as e:
        logger.debug(f"Error getting logs from Loki: {e}")
        return None


def stream_logs_websocket(uri, stop_event, service_name, print_pod_name: bool = False):
    """Stream logs using Loki's websocket tail endpoint"""

    console.print(f"\nFollowing logs of [reset]{service_name}\n")

    # Create and run event loop in a separate thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            stream_logs_websocket_helper(
                uri=uri,
                stop_event=stop_event,
                stream_type=StreamType.CLI,
                print_pod_name=print_pod_name,
            )
        )
    finally:
        loop.close()
        # Signal the log thread to stop
        stop_event.set()
        # Don't wait for the log thread - it will handle its own cleanup


def get_logs_query(name: str, namespace: str, selected_pod: str, deployment_mode):
    if not selected_pod:
        if deployment_mode in ["knative", "deployment"]:
            # we need to get the pod names first since Loki doesn't have a service_name label
            core_api = client.CoreV1Api()
            pods = validate_pods_exist(name, namespace, core_api)
            pod_names = [pod.metadata.name for pod in pods]
            return f'{{k8s_pod_name=~"{"|".join(pod_names)}",k8s_container_name="kubetorch"}} | json'
        else:
            console.print(
                f"[red]Logs does not support deployment mode: {deployment_mode}[/red]"
            )
            return None
    else:
        return (
            f'{{k8s_pod_name=~"{selected_pod}",k8s_container_name="kubetorch"}} | json'
        )


def follow_logs_in_cli(
    name: str,
    namespace: str,
    selected_pod: str,
    deployment_mode,
    print_pod_name: bool = False,
):
    """Stream logs when triggerd by the CLI command."""
    from kubetorch.utils import http_to_ws

    stop_event = threading.Event()

    # Set up signal handler to cleanly stop on Ctrl+C
    def signal_handler(signum, frame):
        stop_event.set()
        raise KeyboardInterrupt()

    original_handler = signal.signal(signal.SIGINT, signal_handler)

    # setting up the query
    query = get_logs_query(name, namespace, selected_pod, deployment_mode)
    if not query:
        return
    encoded_query = urllib.parse.quote_plus(query)

    base_url = globals.service_url()
    uri = f"{http_to_ws(base_url)}/loki/api/v1/tail?query={encoded_query}"

    try:
        stream_logs_websocket(
            uri=uri,
            stop_event=stop_event,
            service_name=name,
            print_pod_name=print_pod_name,
        )
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_handler)


def is_ingress_vpc_only(annotations: dict):
    # Check for internal LoadBalancer annotations
    internal_checks = [
        annotations.get("service.beta.kubernetes.io/aws-load-balancer-internal")
        == "true",
        annotations.get("networking.gke.io/load-balancer-type") == "Internal",
        annotations.get("service.beta.kubernetes.io/oci-load-balancer-internal")
        == "true",
    ]

    vpc_only = any(internal_checks)
    return vpc_only


def load_ingress(namespace: str = globals.config.install_namespace):
    networking_v1_api = client.NetworkingV1Api()
    ingresses = networking_v1_api.list_namespaced_ingress(namespace=namespace)

    for ingress in ingresses.items:
        if ingress.metadata.name == "kubetorch-proxy-ingress":
            return ingress


def get_ingress_host(ingress):
    """Get the configured host from the kubetorch ingress."""
    try:
        return ingress.spec.rules[0].host
    except Exception:
        return None


def list_all_queues():
    try:
        custom_api = client.CustomObjectsApi()
        queues = custom_api.list_cluster_custom_object(
            group="scheduling.run.ai",
            version="v2",
            plural="queues",
        )["items"]

        if not queues:
            console.print("[yellow]No queues found in the cluster[/yellow]")
            return

        # Insert "default" queue if missing
        if not any(q["metadata"]["name"] == "default" for q in queues):
            default_children = [
                q["metadata"]["name"]
                for q in queues
                if q.get("spec", {}).get("parentQueue") == "default"
            ]
            queues.insert(
                0,
                {
                    "metadata": {"name": "default"},
                    "spec": {
                        "parentQueue": "-",
                        "children": default_children,
                        "resources": {
                            "cpu": {"quota": "-", "overQuotaWeight": "-"},
                            "gpu": {"quota": "-", "overQuotaWeight": "-"},
                            "memory": {"quota": "-", "overQuotaWeight": "-"},
                        },
                        "priority": "-",
                    },
                },
            )

        queue_table = Table(title="Available Queues", header_style=Style(bold=True))
        queue_table.add_column("QUEUE NAME", style="cyan")
        queue_table.add_column("PRIORITY", style="magenta")
        queue_table.add_column("PARENT", style="green")
        queue_table.add_column("CHILDREN", style="yellow")
        queue_table.add_column("CPU QUOTA", style="white")
        queue_table.add_column("GPU QUOTA", style="white")
        queue_table.add_column("MEMORY QUOTA", style="white")
        queue_table.add_column("OVERQUOTA WEIGHT", style="blue")

        for q in queues:
            spec = q.get("spec", {})
            resources = spec.get("resources", {})
            cpu = resources.get("cpu", {})
            gpu = resources.get("gpu", {})
            memory = resources.get("memory", {})

            queue_table.add_row(
                q["metadata"]["name"],
                str(spec.get("priority", "-")),
                spec.get("parentQueue", "-"),
                ", ".join(spec.get("children", [])) or "-",
                str(cpu.get("quota", "-")),
                str(gpu.get("quota", "-")),
                str(memory.get("quota", "-")),
                str(
                    cpu.get("overQuotaWeight", "-")
                ),  # use CPU's overQuotaWeight as example
            )

        console.print(queue_table)
        return

    except client.exceptions.ApiException as e:
        console.print(f"[red]Failed to list queues: {e}[/red]")
        raise typer.Exit(1)


def detect_deployment_mode(name: str, namespace: str, custom_api, apps_v1_api):
    """Detect if a service is deployed as Knative, Deployment, or RayCluster."""
    # First try Deployment
    try:
        apps_v1_api.read_namespaced_deployment(name=name, namespace=namespace)
        return "deployment"
    except ApiException:
        pass

    # Then try Knative
    try:
        custom_api.get_namespaced_custom_object(
            group="serving.knative.dev",
            version="v1",
            namespace=namespace,
            plural="services",
            name=name,
        )
        return "knative"
    except ApiException:
        pass

    # Then try RayCluster
    try:
        custom_api.get_namespaced_custom_object(
            group="ray.io",
            version="v1",
            namespace=namespace,
            plural="rayclusters",
            name=name,
        )
        return "raycluster"
    except ApiException:
        pass

    return None


def validate_provided_pod(service_name, provided_pod, service_pods):
    if provided_pod is None:
        return provided_pod

    if provided_pod.isnumeric():
        pod = int(provided_pod)
        if pod < 0 or pod >= len(service_pods):
            console.print(f"[red]Pod index {pod} is out of range[/red]")
            raise typer.Exit(1)
        pod_name = service_pods[pod].metadata.name

    # case when the user provides pod name
    else:
        pod_names = [pod.metadata.name for pod in service_pods]
        if provided_pod not in pod_names:
            console.print(
                f"[red]{service_name} does not have an associated pod called {provided_pod}[/red]"
            )
            raise typer.Exit(1)
        else:
            pod_name = provided_pod

    return pod_name


def load_kubetorch_volumes_for_service(namespace, service_name, core_v1) -> List[str]:
    """Extract volume information from service definition"""
    volumes = []

    try:
        pods = core_v1.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"kubetorch.com/service={service_name}",
        )
        if pods.items:
            pod = pods.items[0]
            for v in pod.spec.volumes or []:
                if v.persistent_volume_claim:
                    volumes.append(v.name)
        return volumes

    except Exception as e:
        logger.warning(f"Failed to extract volumes from service: {e}")

    return volumes


def create_table_for_output(
    columns: List[set], no_wrap_columns_names: list = None, header_style: dict = None
):
    table = Table(box=box.SQUARE, header_style=Style(**header_style))
    for name, style in columns:
        if name in no_wrap_columns_names:
            # always make service name fully visible
            table.add_column(name, style=style, no_wrap=True)
        else:
            table.add_column(name, style=style)

    return table
