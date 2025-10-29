import base64
import importlib
import inspect
import os
import signal
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import List
from urllib.parse import urlparse

import httpx
from kubernetes.client.rest import ApiException
from rich.syntax import Syntax

from kubetorch.servers.http.utils import is_running_in_kubernetes

from .cli_utils import (
    create_table_for_output,
    default_typer_values,
    get_deployment_mode,
    get_ingress_host,
    get_last_updated,
    get_logs_from_loki,
    is_ingress_vpc_only,
    load_ingress,
    load_kubetorch_volumes_for_service,
    port_forward_to_pod,
    SecretAction,
    service_name_argument,
    validate_config_key,
    validate_pods_exist,
    validate_provided_pod,
    VolumeAction,
)

from .utils import initialize_k8s_clients

try:
    import typer

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except ImportError:
    raise ImportError("Please install the required CLI dependencies: `pip install 'kubetorch[client] @ <install_url>'`")


import kubetorch.serving.constants as serving_constants

from kubetorch import globals
from kubetorch.config import ENV_MAPPINGS
from kubetorch.servers.http.utils import DEFAULT_DEBUG_PORT

from .constants import BULLET_UNICODE, KT_MOUNT_FOLDER

try:
    from .internal.cli import register_internal_commands

    _INTERNAL_COMMANDS_AVAILABLE = True
except ImportError:
    _INTERNAL_COMMANDS_AVAILABLE = False

from .logger import get_logger

app = typer.Typer(add_completion=False)
console = Console()

# Register internal CLI commands if available
if _INTERNAL_COMMANDS_AVAILABLE:
    register_internal_commands(app)


logger = get_logger(__name__)


@app.command("check")
def kt_check(
    name: str = service_name_argument(help="Service name"),
    namespace: str = typer.Option(
        globals.config.namespace,
        "-n",
        "--namespace",
    ),
):
    """
    Run a comprehensive health check for a deployed service.

    Checks:

    - Deployment pod comes up and becomes ready (if not scaled to zero)

    - Rsync has succeeded

    - Service is marked as ready and service pod(s) are ready to serve traffic

    - GPU support configured (if applicable)

    - Log streaming configuration (if applicable)

    If a step fails, will dump ``kubectl describe`` and pod logs for relevant pods.
    """
    core_api, custom_api, apps_v1_api = initialize_k8s_clients()

    def dump_pod_debug(pod_name):
        try:
            describe_proc = subprocess.run(
                ["kubectl", "describe", "pod", pod_name, "-n", namespace],
                check=False,
                capture_output=True,
                text=True,
            )
            describe_output = describe_proc.stdout or describe_proc.stderr or "<no output>"

            logs_proc = subprocess.run(
                ["kubectl", "logs", pod_name, "-n", namespace, "-c", "kubetorch"],
                check=False,
                capture_output=True,
                text=True,
            )
            logs_output = logs_proc.stdout or logs_proc.stderr or "<no output>"

            console.print(
                Panel(
                    describe_output,
                    title=f"POD DESCRIPTION ({pod_name})",
                    border_style="yellow",
                    expand=False,
                )
            )
            console.print(
                Panel(
                    logs_output,
                    title=f"POD LOGS ({pod_name})",
                    border_style="yellow",
                    expand=False,
                )
            )
        except Exception as e:
            console.print(f"[red]Failed to dump pod info: {e}[/red]")

    def fail(msg, pod_names=None):
        console.print(f"[red]{msg}[/red]")
        if pod_names:
            for pod_name in pod_names:
                dump_pod_debug(pod_name)
        raise typer.Exit(1)

    try:
        # Validate service exists and get deployment mode
        name, deployment_mode = get_deployment_mode(name, namespace, custom_api, apps_v1_api)

        console.print(f"[bold blue]Checking {deployment_mode} service...[/bold blue]")

        # 1. Deployment pod check
        console.print("[bold blue]Checking deployment pod...[/bold blue]")
        deploy_pods = validate_pods_exist(name, namespace, core_api)

        if not deploy_pods:
            if deployment_mode == "knative":
                try:
                    # Check if the Knative service is marked as ready (e.g. scaled to zero)
                    service = custom_api.get_namespaced_custom_object(
                        group="serving.knative.dev",
                        version="v1",
                        namespace=namespace,
                        plural="services",
                        name=name,
                    )
                    conditions = service.get("status", {}).get("conditions", [])
                    ready = any(c.get("type") == "Ready" and c.get("status") == "True" for c in conditions)
                    if ready:
                        console.print(
                            f"[yellow]No deployment pods found. Service [bold]{name}[/bold] is scaled to zero but marked as 'READY'. "
                            "It will scale up on demand.[/yellow]"
                        )
                        return
                    else:
                        fail("Deployment pod not found and service is not READY.")

                except Exception as e:
                    fail(f"Failed to check Knative service status: {e}")
            else:
                fail("No Deployment pods found.")

        deploy_pod = next(
            (p for p in deploy_pods if p.status.phase == "Running" and not p.metadata.deletion_timestamp),
            None,
        )
        if not deploy_pod:
            fail(
                "No deployment pod in 'Running' state found.",
                [p.metadata.name for p in deploy_pods],
            )

        deploy_pod_name = deploy_pod.metadata.name
        if deploy_pod.status.phase != "Running":
            fail(
                f"Deployment pod not running (status: {deploy_pod.status.phase})",
                [deploy_pod_name],
            )

        # 2. Rsync check
        console.print("[bold blue]Checking rsync...[/bold blue]")
        current_working_dir = "."
        check_cmd = [
            "kubectl",
            "exec",
            deploy_pod_name,
            "-n",
            namespace,
            "--",
            "ls",
            "-l",
            current_working_dir,
        ]
        try:
            result = subprocess.run(check_cmd, capture_output=True, text=True, check=True)
            lines = result.stdout.splitlines()
            entries = [line for line in lines if not line.startswith("total")]
            if not entries:
                fail("Rsync directory exists but is empty.", [deploy_pod_name])
        except subprocess.CalledProcessError as e:
            fail(
                f"Rsync directory check failed: {e.stderr or e.stdout}",
                [deploy_pod_name],
            )

        # 3. Service call check
        console.print("[bold blue]Checking service call...[/bold blue]")
        try:
            with port_forward_to_pod(
                pod_name=deploy_pod_name,
                namespace=namespace,
                local_port=32300,
                remote_port=32300,
            ) as local_port:
                url = f"http://localhost:{local_port}/health"
                resp = httpx.get(url, timeout=10)
                if not resp.is_success:
                    fail(
                        f"Service call failed: {resp.status_code} {resp.text}",
                        [deploy_pod_name],
                    )
        except Exception as e:
            fail(f"Service call check failed: {e}", [deploy_pod_name])

        # 5. GPU + autoscaler test (if GPU requested)
        gpu_requested = any(
            c.resources.limits and "nvidia.com/gpu" in c.resources.limits for c in deploy_pod.spec.containers
        )
        if gpu_requested:
            gpus_configured = False
            console.print("[bold blue]Checking GPU plugin support...[/bold blue]")
            nodes = core_api.list_node().items
            for node in nodes:
                gpus = node.status.capacity.get("nvidia.com/gpu")
                if gpus and int(gpus) > 0:
                    gpus_configured = True
                    break

            if not gpus_configured:
                console.print(
                    "[yellow]No GPU nodes currently configured on the cluster, is autoscaling configured?[/yellow]"
                )

            dcgm_exporter = True
            dcgm_namespace = globals.config.install_namespace

            pods = core_api.list_namespaced_pod(
                namespace=dcgm_namespace,
                label_selector="app.kubernetes.io/name=dcgm-exporter",
            ).items
            if not pods:
                dcgm_exporter = False

            if not dcgm_exporter:
                console.print(f"[yellow]DCGM exporter not found in namespace {dcgm_namespace}[/yellow]")

        # 6. Check logs
        if globals.config.stream_logs:
            try:
                streaming_enabled = core_api.read_namespaced_service(
                    name=serving_constants.LOKI_GATEWAY_SERVICE_NAME,
                    namespace=globals.config.install_namespace,
                )
            except ApiException:
                streaming_enabled = False

            if streaming_enabled:

                console.print("[bold blue]Checking log streaming...[/bold blue]")
                query = f'{{k8s_pod_name="{deploy_pod_name}", k8s_container_name="kubetorch"}}'
                try:
                    logs = get_logs_from_loki(query=query, print_pod_name=False, timeout=5.0)
                    if logs is None:
                        fail("No logs found for service", [deploy_pod_name])

                except Exception as e:
                    fail(f"Logs check failed: {e}", [deploy_pod_name])

        console.print("[bold green]✓ All service checks passed[/bold green]")

    except typer.Exit:
        # Just re-raise, don't print
        raise


@app.command("config")
def kt_config(
    action: str = typer.Argument(default="", help="Action to perform (set, unset, get, list)"),
    key: str = typer.Argument(None, help="Config key (e.g., 'username')", callback=validate_config_key),
    value: str = typer.Argument(None, help="Value to set"),
):
    """Manage Kubetorch configuration settings.

    Examples:

    .. code-block:: bash

        $ kt config set username johndoe

        $ kt config set volumes "volume_name_one, volume_name_two"

        $ kt config set volumes volume_name_one

        $ kt config unset username

        $ kt config get username

        $ kt config list
    """
    from kubetorch import config

    if action == "set":
        if not key or not value:
            console.print("[red]Both key and value are required for 'set'[/red]")
            raise typer.Exit(1)

        try:
            value = config.set(key, value)  # validate value
            config.write({key: value})
            console.print(f"[green]{key} set to:[/green] [blue]{value}[/blue]")
        except ValueError as e:
            console.print(f"[red]Error setting {key}:[/red] {str(e)}")
            raise typer.Exit(1)

    elif action == "unset":
        if not key:
            console.print("[red]Key is required for 'unset'[/red]")
            raise typer.Exit(1)

        try:
            config.set(key, None)
            config.write({key: None})
            console.print(f"[green]{key.capitalize()} unset[/green]")
        except ValueError as e:
            console.print(f"[red]Error unsetting {key}:[/red] {str(e)}")
            raise typer.Exit(1)

    elif action == "get":
        if not key:
            # Error panel
            console.print("[red]Key is required for 'get'[/red]")
            raise typer.Exit(1)

        if key in ENV_MAPPINGS:
            value = config.get(key)
            if value:
                console.print(f"[blue]{value}[/blue]")
            else:
                console.print(f"[yellow]{key.capitalize()} not set[/yellow]")
        else:
            console.print(f"[red]Unknown config key:[/red] [bold]{key}[/bold]")
            raise typer.Exit(1)

    elif action == "list" or not action:
        console.print(dict(config))

    else:
        console.print(f"[red]Unknown action:[/red] [bold]{action}[/bold]")
        console.print("\nValid actions are: set, get, list")
        raise typer.Exit(1)


@app.command("debug")
def kt_debug(
    pod: str = typer.Argument(..., help="Pod name"),
    namespace: str = typer.Option(
        globals.config.namespace,
        "-n",
        "--namespace",
    ),
    port: int = typer.Option(DEFAULT_DEBUG_PORT, help="Debug port used for remote debug server"),
):
    """Start an interactive debugging session on the pod, which will connect to the debug server inside the service.
    Before running this command, you must call a method on the service with pdb=True or add a
    kt.deep_breakpoint() call into your code to enable debugging.
    """
    import webbrowser

    if is_running_in_kubernetes():
        console.print(
            "[red]Debugging is not supported when running inside Kubernetes. Please run this command locally.[/red]"
        )
        raise typer.Exit(1)

    # Use the base path of web-pdb server as health endpoint because we're port-forwarding straight into the pod
    with port_forward_to_pod(
        namespace=namespace,
        pod_name=pod,
        local_port=port,
        remote_port=port,
        health_endpoint="/",
    ):
        debug_ui_url = f"http://localhost:{port}"
        console.print(f"Opening debug UI at [blue]{debug_ui_url}[/blue]")
        webbrowser.open(debug_ui_url)
        # Wait for the user to finish debugging
        console.print("[yellow]Press Ctrl+C to stop the debugging session and close the UI.[/yellow]")
        # Wait for a Ctrl+C to exit the debug session
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Debugging session ended.[/yellow]")
            raise typer.Exit(0)


@app.command("deploy")
def kt_deploy(
    target: str = typer.Argument(
        ...,
        help="Python module or file to deploy, optionally followed by a "
        "single function or class to deploy. e.e. `my_module:my_cls`, or "
        "`my_file.py`.",
    ),
):
    """Deploy a Python file or module to Kubetorch. This will deploy all functions and modules decorated with
    @kt.compute in the file or module."""
    from kubetorch.resources.compute.utils import _collect_modules

    os.environ["KT_CLI_DEPLOY_MODE"] = "1"
    to_deploy, target_fn_or_class = _collect_modules(target)

    if not target_fn_or_class:
        console.print(f"Found the following functions and classes to deploy in {target}:")
        for module in to_deploy:
            console.print(f"{BULLET_UNICODE} {module.name}")

    import asyncio

    async def deploy_all_async():
        tasks = []
        for module in to_deploy:
            console.print(f"Deploying {module.name}...")
            tasks.append(module.deploy_async())

        try:
            await asyncio.gather(*tasks)
            for module in to_deploy:
                console.print(f"Successfully deployed {module.name}.")
        except Exception as e:
            console.print(f"Failed to deploy one or more modules: {e}")
            raise e

    asyncio.run(deploy_all_async())

    if not target_fn_or_class:
        console.print(f"Successfully deployed functions and modules from {target}.")


@app.command("describe")
def kt_describe(
    name: str = service_name_argument(help="Service name"),
    namespace: str = typer.Option(
        globals.config.namespace,
        "-n",
        "--namespace",
    ),
):
    """
    Show basic info for calling the service depending on whether an ingress is configured.
    """

    core_api, custom_api, apps_v1_api = initialize_k8s_clients()

    endpoint_placeholder = "METHOD_OR_CLS_NAME"
    args_placeholder = []

    try:
        name, deployment_mode = get_deployment_mode(name, namespace, custom_api, apps_v1_api)
    except ApiException:
        console.print(f"[red] Failed to load service '{name}' in namespace '{namespace}'[/red]")
        raise typer.Exit(1)

    try:
        console.print()
        base_url = globals.config.api_url

        ingress = load_ingress()
        host = get_ingress_host(ingress) if ingress else f"{name}.{namespace}.svc.cluster.local"

        if not base_url:
            if not ingress:
                console.print("[yellow]No ingress found. Service is only accessible from inside the cluster.[/yellow]")
                base_url = f"http://{name}.{namespace}.svc.cluster.local"
            else:
                lb_ing = (
                    ingress.status.load_balancer.ingress[0]
                    if (ingress.status and ingress.status.load_balancer and ingress.status.load_balancer.ingress)
                    else None
                )

                address = lb_ing.hostname or lb_ing.ip if lb_ing else None
                if address:
                    base_url = f"http://{address}"
                else:
                    console.print("[yellow]Ingress found but no address, falling back to cluster-local.[/yellow]")
                    base_url = f"http://{name}.{namespace}.svc.cluster.local"
        else:
            parsed = urlparse(base_url)
            if not parsed.scheme:
                base_url = f"http://{base_url}"

        if ingress:
            console.print(f"[bold]Host:[/bold] [green]{name}[/green]")

            vpc_only = is_ingress_vpc_only(ingress.metadata.annotations)
            if vpc_only:
                console.print()
                console.print("[yellow]Note: This is a VPC-only ingress (internal access only)[/yellow]")

        console.print()

        if ingress:
            console.print("[bold]Calling the service using an ingress:[/bold]\n")
            # With ingress, use the full path structure
            service_path = f"/{namespace}/{name}/{endpoint_placeholder}"
        else:
            console.print("[bold]Calling the service from inside the cluster:[/bold]\n")
            service_path = f"/{endpoint_placeholder}"

        curl_code = textwrap.dedent(
            f"""\
            curl -X POST \\
              -H "Content-Type: application/json" \\
              -d '{{"args": {args_placeholder}, "kwargs": {{}}}}' \\
              {base_url}{service_path}
        """
        )
        # Only add Host header if we have ingress
        if ingress:
            curl_code = curl_code.replace(
                '-H "Content-Type: application/json"',
                f'-H "Host: {host}" \\\n  -H "Content-Type: application/json"',
            )

        console.print(Panel(Syntax(curl_code, "bash"), title="With Curl", border_style="green"))
        console.print()

        python_code = textwrap.dedent(
            f"""\
            import requests

            url = "{base_url}{service_path}"
            headers = {{
                "Content-Type": "application/json"
            }}
            data = {{
                "args": {args_placeholder},
                "kwargs": {{}}
            }}

            response = requests.post(url, headers=headers, json=data)
            print(response.json())
        """
        )
        if ingress:
            python_code = python_code.replace(
                '"Content-Type": "application/json"',
                f'"Host": "{host}",\n    "Content-Type": "application/json"',
            )
        console.print(
            Panel(
                Syntax(python_code, "python"),
                title="With Python",
                border_style="green",
            )
        )
    except Exception as e:
        console.print(
            f"[red]Failed to describe service {name} in namespace {namespace}: {e}[/red]",
        )
        raise typer.Exit(1)


@app.command("list")
def kt_list(
    namespace: str = typer.Option(
        globals.config.namespace,
        "-n",
        "--namespace",
    ),
    sort_by_updated: bool = typer.Option(False, "-s", "--sort", help="Sort by last update time"),
    tag: str = typer.Option(
        None,
        "-t",
        "--tag",
        help="Service tag or prefix (ex: 'myusername', 'some-git-branch').",
    ),
):
    """List all Kubetorch services.

    Examples:

    .. code-block:: bash

      $ kt list

      $ kt list -t dev-branch
    """
    core_api, custom_api, _ = initialize_k8s_clients()

    # Import here to avoid circular imports
    from kubetorch.serving.service_manager import BaseServiceManager

    try:
        # Use unified service discovery
        unified_services = BaseServiceManager.discover_services_static(namespace=namespace, name_filter=tag)

        if not unified_services:
            console.print(f"[yellow]No services found in {namespace} namespace[/yellow]")
            return

        # Optional second-level tag filtering
        if tag:
            unified_services = [
                svc
                for svc in unified_services
                if tag in svc["name"]
                or tag in " ".join(str(v) for v in svc["resource"].get("metadata", {}).get("labels", {}).values())
            ]
            if not unified_services:
                console.print(f"[yellow]No services found in {namespace} namespace[/yellow]")
                return

        if sort_by_updated:

            def get_update_time(svc):
                # If not a ksvc, use creation timestamp as proxy for update time
                return (
                    get_last_updated(svc["resource"]) if svc["template_type"] == "ksvc" else svc["creation_timestamp"]
                )

            unified_services.sort(key=get_update_time, reverse=True)

        # Get pod maps
        pod_map = {
            svc["name"]: BaseServiceManager.get_pods_for_service_static(svc["name"], namespace, core_api)
            for svc in unified_services
        }

        # Create table
        table_columns = [
            ("SERVICE", "cyan"),
            ("TYPE", "magenta"),
            ("STATUS", "green"),
            ("# OF PODS", "yellow"),
            ("POD NAMES", "red"),
            ("VOLUMES", "blue"),
            ("LAST STATUS CHANGE", "yellow"),
            ("TTL", "yellow"),
            ("CREATOR", "yellow"),
            ("QUEUE", "yellow"),
            ("CPUs", "yellow"),
            ("MEMORY", "yellow"),
            ("GPUs", "yellow"),
        ]
        table = create_table_for_output(
            columns=table_columns,
            no_wrap_columns_names=["SERVICE"],
            header_style={"bold": False},
        )

        for svc in unified_services:
            name = svc["name"]
            kind = svc["template_type"]
            res = svc["resource"]
            meta = res.get("metadata", {})
            labels = meta.get("labels", {})
            annotations = meta.get("annotations", {})
            status_data = res.get("status", {})

            # Get pods
            pods = pod_map.get(name, [])

            creation_ts = meta.get("creationTimestamp", None)
            timestamp = (
                datetime.fromisoformat(creation_ts.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S")
                if creation_ts
                else "Unknown"
            )
            ttl = annotations.get(serving_constants.INACTIVITY_TTL_ANNOTATION, "None")
            creator = labels.get(serving_constants.KT_USERNAME_LABEL, "—")

            volumes_display = load_kubetorch_volumes_for_service(namespace, name, core_api)

            # Get resources from revision
            cpu = memory = gpu = None
            if kind == "ksvc":
                cond = status_data.get("conditions", [{}])[0]
                status = cond.get("status")
                display_status = {
                    "True": "[green]Ready[/green]",
                    "Unknown": "[yellow]Creating[/yellow]",
                }.get(status, "[red]Failed[/red]")
                rev_name = status_data.get("latestCreatedRevisionName")
                if rev_name:
                    try:
                        rev = custom_api.get_namespaced_custom_object(
                            group="serving.knative.dev",
                            version="v1",
                            namespace=namespace,
                            plural="revisions",
                            name=rev_name,
                        )
                        container = rev["spec"]["containers"][0]
                        reqs = container.get("resources", {}).get("requests", {})
                        cpu = reqs.get("cpu")
                        memory = reqs.get("memory")
                        gpu = reqs.get("nvidia.com/gpu") or reqs.get("gpu")
                    except Exception as e:
                        logger.warning(f"Could not get revision for {name}: {e}")
            else:
                # Process Deployment - now using consistent dict access
                ready = res.get("status", {}).get("readyReplicas", 0) or 0
                desired = res.get("spec", {}).get("replicas", 0) or 0
                if kind == "raycluster":
                    state = status_data.get("state", "").lower()
                    conditions = {c["type"]: c["status"] for c in status_data.get("conditions", [])}
                    if (
                        state == "ready"
                        and conditions.get("HeadPodReady") == "True"
                        and conditions.get("RayClusterProvisioned") == "True"
                    ):
                        display_status = "[green]Ready[/green]"
                    elif state in ("creating", "upscaling", "restarting", "updating"):
                        display_status = "[yellow]Scaling[/yellow]"
                    else:
                        display_status = "[red]Failed[/red]"
                else:
                    display_status = (
                        "[green]Ready[/green]"
                        if ready == desired and desired > 0
                        else "[yellow]Scaling[/yellow]"
                        if ready < desired
                        else "[red]Failed[/red]"
                    )
                try:
                    container = res.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [{}])[0]
                    reqs = container.get("resources", {}).get("requests", {})
                    cpu = reqs.get("cpu")
                    memory = reqs.get("memory")
                    gpu = reqs.get("nvidia.com/gpu") or reqs.get("gpu")
                except Exception as e:
                    logger.warning(f"Failed to get resources for {name} in namespace {namespace}: {e}")

            # Common pod processing
            pod_lines = []
            queue = "—"
            for pod in pods:
                pod_status = pod.status.phase
                ready = all(c.ready for c in (pod.status.container_statuses or []))
                if ready and pod_status == "Running":
                    color = "green"
                elif "Creating" in display_status or "Scaling" in display_status:
                    color = "yellow"
                else:
                    color = "red"
                pod_lines.append(f"[{color}]{pod.metadata.name}[/{color}]")
                queue = pod.metadata.labels.get(serving_constants.KAI_SCHEDULER_LABEL, queue)

                # Update service status if pod is pending
                if pod_status == "Pending":
                    display_status = "[yellow]Pending[/yellow]"

            table.add_row(
                name,
                f"[magenta]{kind}[/magenta]",
                display_status,
                str(len(pods)),
                "\n".join(pod_lines),
                "\n".join(volumes_display) or "-",
                timestamp,
                ttl,
                creator,
                queue,
                cpu or "—",
                memory or "—",
                gpu or "—",
            )

        table.pad_bottom = 1
        console.print(table)

    except ApiException as e:
        console.print(f"[red]Kubernetes API error: {e}[/red]")
        raise typer.Exit(1)


@app.command("port-forward")
def kt_port_forward(
    name: str = service_name_argument(help="Service name"),
    local_port: int = typer.Argument(default=serving_constants.DEFAULT_KT_SERVER_PORT, help="Local port to bind to"),
    remote_port: int = typer.Argument(
        default=serving_constants.DEFAULT_KT_SERVER_PORT,
        help="Remote port to forward to",
    ),
    namespace: str = typer.Option(
        globals.config.namespace,
        "-n",
        "--namespace",
    ),
    pod: str = typer.Option(
        None,
        "-p",
        "--pod",
        help="Name or index of a specific pod to load logs from (0-based)",
    ),
):
    """
    Port forward a local port to the specified Kubetorch service.

    Examples:

    .. code-block:: bash


        $ kt port-forward my-service

        $ kt port-forward my-service 32300

        $ kt port-forward my-service -n custom-namespace

        $ kt port-forward my-service -p my-pod

    This allows you to access the service locally using `curl http://localhost:<port>`.
    """

    from kubetorch.resources.compute.utils import is_port_available

    if not is_port_available(local_port):
        console.print(f"\n[red]Local port {local_port} is already in use.[/red]")
        raise typer.Exit(1)

    core_api, custom_api, apps_v1_api = initialize_k8s_clients()

    name, _ = get_deployment_mode(name, namespace, custom_api, apps_v1_api)
    pods = validate_pods_exist(name, namespace, core_api)
    sorted_by_time = sorted(pods, key=lambda pod: pod.metadata.creation_timestamp)

    if pod:  # case when the user provides a pod
        pod_name = validate_provided_pod(service_name=name, provided_pod=pod, service_pods=sorted_by_time)
    else:  # if user does not provide pod, port-forward to the first pod by default
        pod_name = sorted_by_time[0].metadata.name

    process = None

    def cleanup_process():
        # Clean up the port forward process
        if process:
            process.kill()

    def signal_handler(signum, frame):
        """Handle interrupt signals for graceful shutdown."""
        console.print(f"\nReceived signal {signum}, cleaning up port forward...")
        cleanup_process()
        console.print("Port forward stopped.")
        raise typer.Exit(0)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    from kubetorch.serving.utils import wait_for_port_forward

    cmd = [
        "kubectl",
        "port-forward",
        f"pod/{pod_name}",
        f"{local_port}:{remote_port}",
        "--namespace",
        namespace,
    ]

    port_forward_msg = f"Starting port forward to {name} in namespace {namespace}"

    if pod:
        port_forward_msg = port_forward_msg + f", pod: [reset]{pod}"
    console.print(port_forward_msg)

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)

    try:
        wait_for_port_forward(process, local_port)
        time.sleep(2)
    except Exception as e:
        logger.info(f"Failed to establish port forward on port {local_port}: {e}")
        if process:
            cleanup_process()
            process = None
        return

    console.print(f"[green]✓ Port forward active on localhost:{local_port} -> {pod_name}:{remote_port}[/green]")
    console.print(f"[cyan]You can now run: curl http://localhost:{local_port}[/cyan]")
    console.print("[dim]Press Ctrl+C to stop the port forward[/dim]")

    # Keep the port forward running until interrupted
    try:
        while True:
            if process.poll() is not None:
                # Process has terminated
                console.print("[red]Port forward process has terminated unexpectedly[/red]")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        # This should be handled by the signal handler, but just in case
        pass

    except typer.Exit:
        # Re-raise typer.Exit to maintain proper CLI behavior
        raise
    except Exception as e:
        console.print(f"[red]Error during port forwarding: {e}[/red]")
        raise typer.Exit(1)
    finally:
        cleanup_process()


@app.command("run", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def kt_run(
    ctx: typer.Context,
    name: str = typer.Option(None, "--name", help="Name for the run"),
    run_async: bool = typer.Option(False, "--async", help="Whether to run async and not stream logs live"),
    file: int = typer.Option(None, "--file", help="File where the app is defined in"),
):
    """
    Build and deploy a kubetorch app that runs the provided CLI command. In order for the app
    to be deployed, the file being run must be a Python file specifying a `kt.app` construction
    at the top of the file.

    Examples:

    .. code-block:: bash

        $ kt run python train.py --epochs 5
        $ kt run fastapi run my_app.py --name fastapi-app
    """
    from kubetorch import App

    cli_cmd = " ".join(ctx.args)
    if not cli_cmd:
        raise typer.BadParameter("You must provide a command to run.")
    elif cli_cmd.split()[0].endswith(".py"):
        raise typer.BadParameter(
            "You must provide a full command to run, the Python file should not be the first argument. "
            "(e.g. `kt run python train.py`)"
        )

    python_file = file
    if not python_file:
        for arg in cli_cmd.split():
            if arg.endswith("py") and Path(arg).exists():
                python_file = arg
                break

        if not python_file:
            console.print(
                f"[red]Could not detect python file with `kt.app` in {cli_cmd}. Pass it in with `--file`.[/red]"
            )
            raise typer.Exit(1)

    # Set env vars for construction of app instance
    os.environ["KT_RUN"] = "1"
    os.environ["KT_RUN_CMD"] = cli_cmd
    os.environ["KT_RUN_FILE"] = python_file
    if name:
        os.environ["KT_RUN_NAME"] = name
    if run_async:
        os.environ["KT_RUN_ASYNC"] = "1"

    # Extract the app instance from the python file
    module_name = Path(python_file).stem
    python_file_dir = Path(python_file).resolve().parent

    # Add the directory containing the Python file to sys.path to support relative imports
    if str(python_file_dir) not in sys.path:
        sys.path.insert(0, str(python_file_dir))

    spec = importlib.util.spec_from_file_location(module_name, python_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    app_instance = None
    for _, obj in inspect.getmembers(module):
        if isinstance(obj, App):
            app_instance = obj
            break
    if not app_instance:
        console.print(f"[red]Could not find kt.app definition in {python_file} [/red]")
        raise typer.Exit(1)

    app_instance.deploy()


@app.command("secrets")
def kt_secrets(
    action: SecretAction = typer.Argument(
        SecretAction.list,
        help="Action to perform: list, create, update, delete, describe",
    ),
    name: str = typer.Argument(None, help="Secret name (for create or delete actions)"),
    prefix: str = typer.Option(
        None,
        "--prefix",
        "-x",
    ),
    namespace: str = typer.Option(
        "default",
        "-n",
        "--namespace",
    ),
    all_namespaces: bool = typer.Option(
        False,
        "--all-namespaces",
        "-A",
    ),
    yes: bool = typer.Option(False, "-y", "--yes", help="Deletion confirmation"),
    path: str = typer.Option(None, "--path", "-p", help="Path where the secret values are held"),
    provider: str = typer.Option(
        None,
        "--provider",
        "-c",
        help="Provider corresponding to the secret (e.g. 'aws', 'gcp'). "
        "If not specified, secrets are loaded from the default provider path.",
    ),
    env_vars: List[str] = typer.Option(
        None,
        "--env-vars",
        "-v",
        help="Environment variable(s) key(s) whose value(s) will hold the secret value(s)",
    ),
    show_values: bool = typer.Option(False, "-s", "--show", help="Show secrets values in the describe output"),
):
    """Manage secrets used in Kubetorch services.

    Examples:

    .. code-block:: bash

        $ kt secrets  # list secrets in the default namespace

        $ kt secrets list -n my_namespace  # list secrets in `my_namespace` namespace

        $ kt secrets -A  # list secrets in all namespaces

        $ kt secrets create --provider aws  # create a secret with the aws credentials in `default` namespace

        $ kt secrets create my_secret -v ENV_VAR_1 -v ENV_VAR_2 -n my_namespace  # create a secret using env vars

        $ kt secrets delete my_secret -n my_namespace  # delete a secret called `my_secret` from `my_namespace` namespace

        $ kt secrets delete aws   # delete a secret called `aws` from `default` namespace
    """
    import kubetorch as kt
    from kubetorch.resources.compute.utils import delete_secrets, list_secrets
    from kubetorch.resources.secrets.kubernetes_secrets_client import KubernetesSecretsClient

    secrets_client = KubernetesSecretsClient(namespace=namespace)

    core_api, custom_api, apps_v1_api = initialize_k8s_clients()

    if action == SecretAction.list:
        secrets = list_secrets(
            core_api=core_api,
            namespace=namespace,
            prefix=prefix,
            all_namespaces=all_namespaces,
            console=console,
            filter_by_creator=False,
        )

        table_columns = [
            ("SECRET", "blue"),
            ("CREATOR", "cyan"),
            ("NAMESPACE", "yellow"),
        ]
        table = create_table_for_output(
            columns=table_columns,
            no_wrap_columns_names=["SECRET"],
            header_style={"bold": True},
        )

        if not secrets:
            msg = "No secrets found"
            if not all_namespaces:
                if prefix:
                    msg += f" with prefix: {prefix}"
                msg += f" in namespace: {namespace}"
            console.print(f"[yellow]{msg}[/yellow]")
            raise typer.Exit(0)

        for secret in secrets:
            secret_name = secret.get(
                "user_defined_name"
            )  # TODO: maybe display the kt name? so it'll match kubectl get secrets
            creator = secret.get("username")
            namespace = secret.get("namespace")
            table.add_row(secret_name, creator, namespace)

        table.pad_bottom = 1
        console.print(table)

    elif action == SecretAction.create:
        if not (name or provider):
            console.print("[red]Cannot create secret: name or provider must be specified.[/red]")
            typer.Exit(1)
        env_vars_dict = {key: key for key in env_vars} if env_vars else {}

        try:
            new_secret = kt.secret(name=name, provider=provider, path=path, env_vars=env_vars_dict)
            secrets_client.create_secret(secret=new_secret, console=console)
        except Exception as e:
            console.print(f"[red]Failed to create the secret: {e}[/red]")
            raise typer.Exit(0)

    elif action == SecretAction.delete:
        prefix = name if name else prefix
        all_namespaces = False if name else all_namespaces
        secrets_to_delete = list_secrets(
            core_api=core_api,
            namespace=namespace,
            prefix=prefix,
            all_namespaces=all_namespaces,
            console=console,
        )

        username = globals.config.username
        secrets_to_delete_by_namespace: dict[str, list[str]] = {}
        for secret in secrets_to_delete:
            ns = secret.get("namespace")
            name = secret.get("name")

            if all_namespaces:
                if secret.get("username") != username:
                    continue  # skip secrets not owned by user

            secrets_to_delete_by_namespace.setdefault(ns, []).append(name)

        # Flatten names for display
        secrets_names = [name for names in secrets_to_delete_by_namespace.values() for name in names]

        if not secrets_names:
            console.print(f"[yellow]No secrets to delete for username: {username}[/yellow]")
            raise typer.Exit(0)

        secrets_word = "secret" if len(secrets_names) == 1 else "secrets"
        console.print(f"\nDeleting {len(secrets_names)} {secrets_word}...")

        for secret in secrets_names:
            console.print(f"  - [blue]{secret}[/blue]")

        if not yes:
            confirm = typer.confirm("\nDo you want to proceed?")
            if not confirm:
                console.print("[yellow]Operation cancelled[/yellow]")
                raise typer.Exit(0)

        for ns, secrets in secrets_to_delete_by_namespace.items():
            if secrets:
                client = KubernetesSecretsClient(namespace=ns)
                delete_secrets(
                    secrets=secrets,
                    console=console,
                    secrets_client=client,
                )

    elif action == SecretAction.describe:
        prefix = name if name else prefix
        all_namespaces = False if name else all_namespaces
        secrets_to_describe = list_secrets(
            core_api=core_api,
            namespace=namespace,
            prefix=name or prefix,
            all_namespaces=all_namespaces,
            filter_by_creator=False,
            console=console,
        )
        if not secrets_to_describe:
            console.print("[yellow] No secrets found[/yellow]")
            raise typer.Exit(0)

        for secret in secrets_to_describe:
            k8_name = secret.get("name")
            kt_name = secret.get("user_defined_name")
            console.print(f"[bold cyan]{kt_name}[/bold cyan]")
            console.print(f"  K8 Name: [reset]{k8_name}")
            console.print(f'  Namespace: {secret.get("namespace")}')
            console.print(f'  Labels: [reset]{secret.get("labels")}')
            console.print(f'  Type: {secret.get("type")}')
            secret_data = secret.get("data")
            if show_values:
                console.print("  Data:")
                for k, v in secret_data.items():
                    try:
                        decoded_value = base64.b64decode(v).decode("utf-8")
                    except Exception:
                        decoded_value = "<binary data>"
                    indented_value = textwrap.indent(decoded_value, "     ")
                    indented_value = indented_value.replace("\n\n", "\n")
                    console.print(f"   {k}:{indented_value}\n")


@app.command("ssh")
def kt_ssh(
    name: str = service_name_argument(help="Service name"),
    namespace: str = typer.Option(
        globals.config.namespace,
        "-n",
        "--namespace",
    ),
    pod: str = typer.Option(
        None,
        "-p",
        "--pod",
        help="Name or index of a specific pod to load logs from (0-based)",
    ),
):
    """SSH into a remote service. By default, will SSH into the first pod.

    Examples:

    .. code-block:: bash

        $ kt ssh my_service
    """
    core_api, custom_api, apps_v1_api = initialize_k8s_clients()

    try:
        # Validate service exists and get deployment mode
        name, deployment_mode = get_deployment_mode(name, namespace, custom_api, apps_v1_api)

        # Get and validate pods
        pods = validate_pods_exist(name, namespace, core_api)

        sorted_by_time = sorted(pods, key=lambda pod: pod.metadata.creation_timestamp)

        # case when the user provides a specific pod to ssh into
        if pod:
            pod_name = validate_provided_pod(service_name=name, provided_pod=pod, service_pods=sorted_by_time)
        # if pod is not provided, ssh into the first pod.
        else:
            pod_name = sorted_by_time[0].metadata.name

        console.print(f"[green]Found pod:[/green] [blue]{pod_name}[/blue] ({deployment_mode})")
        console.print("[yellow]Connecting to pod...[/yellow]")

        # Still need subprocess for the interactive shell
        subprocess.run(
            ["kubectl", "exec", "-it", pod_name, "-n", namespace, "--", "/bin/bash"],
            check=True,
        )

    except ApiException as e:
        console.print(f"[red]Kubernetes API error: {e}[/red]")
        raise typer.Exit(1)


@app.command("teardown")
def kt_teardown(
    name: str = service_name_argument(help="Service name", required=False),
    yes: bool = typer.Option(False, "-y", "--yes", help="Deletion confirmation"),
    teardown_all: bool = typer.Option(False, "-a", "--all", help="Deletes all services for the current user"),
    prefix: str = typer.Option("", "-p", "--prefix", help="Tear down all services with given prefix"),
    namespace: str = typer.Option(
        globals.config.namespace,
        "-n",
        "--namespace",
    ),
    force: bool = typer.Option(False, "-f", "--force", help="Force deletion without graceful shutdown"),
    exact_match: bool = typer.Option(
        False,
        "-e",
        "--exact-match",
        help="Only delete the exact service name, not the prefixed version",
    ),
):
    """Delete a service and all its associated resources (deployments, configmaps, etc).


    Examples:

    .. code-block:: bash

        $ kt teardown my-service -y  # force teardown resources corresponding to service

        $ kt teardown --all          # teardown all resources corresponding to username

        $ kt teardown --prefix test  # teardown resources with prefix "test"
    """
    from kubetorch import config
    from kubetorch.resources.compute.utils import delete_resources_for_service, fetch_resources_for_teardown

    name, yes, teardown_all, namespace, prefix = default_typer_values(name, yes, teardown_all, namespace, prefix)

    core_api, custom_api, _ = initialize_k8s_clients()

    if teardown_all:
        if not config.username:
            console.print(
                "[red]Username is not found, can't delete all services. Please set up a username, provide a service "
                "name or use the --prefix flag[/red]"
            )
            raise typer.Exit(1)

        console.print(f"Deleting all services for username [blue]{config.username}[/blue]...")

    elif prefix:
        console.print(
            f"Deleting all services with prefix [blue]{prefix}[/blue] in [blue]{namespace}[/blue] namespace..."
        )
    else:
        if not name:
            console.print("[red]Please provide a service name or use the --all or --prefix flags[/red]")
            raise typer.Exit(1)

        console.print(f"Finding resources for service [blue]{name}[/blue] in [blue]{namespace}[/blue] namespace...")

    resources = fetch_resources_for_teardown(
        namespace=namespace,
        target=name,
        core_api=core_api,
        custom_api=custom_api,
        prefix=prefix,
        username=config.username if teardown_all else None,
        exact_match=exact_match,
    )

    services = list(resources["services"].keys())
    service_count = len(services)

    if teardown_all or prefix:
        service_word = "service" if service_count == 1 else "services"
        if not services:
            console.print("[yellow]No services are found[/yellow]")
            raise typer.Exit(0)
        else:
            console.print(f"[yellow]Found [bold]{service_count}[/bold] {service_word} to delete.[/yellow]")

    if name and not services:
        console.print(f"[red]Service [bold]{name}[/bold] not found[/red]")
        raise typer.Exit(1)

    # Confirmation prompt for multiple services
    if not yes and service_count > 1:
        for service_name in services:
            console.print(f" • {service_name}")

    # List out resources to be deleted for each service
    console.print("\n[yellow]The following resources will be deleted:[/yellow]")
    for name in services:
        service_info = resources["services"][name]
        configmaps = service_info["configmaps"]
        service_type = service_info.get("type", "unknown")

        if service_type == "deployment":
            console.print(f"• Deployment: [blue]{name}[/blue]")
            console.print(f"• Service: [blue]{name}[/blue]")
        elif service_type == "knative":
            console.print(f"• Knative Service: [blue]{name}[/blue]")
        elif service_type == "raycluster":
            console.print(f"• RayCluster: [blue]{name}[/blue]")
            console.print(f"• Service: [blue]{name}[/blue]")
        else:
            console.print(f"• Service: [blue]{name}[/blue]")

        if configmaps:
            console.print("• ConfigMaps:")
            for cm in configmaps:
                console.print(f"  - [blue]{cm}[/blue]")

    # Confirmation prompt for single service
    if not yes and not force:  # if --force is provided, we don't need additional confirmation
        confirm = typer.confirm("\nDo you want to proceed?")
        if not confirm:
            console.print("[yellow]Teardown cancelled[/yellow]")
            raise typer.Exit(0)

    # Delete resources
    if force:
        console.print("\n[yellow]Force deleting resources...[/yellow]")
    else:
        console.print("\n[yellow]Deleting resources...[/yellow]")

    service_types = set()
    for name in services:
        service_info = resources["services"][name]
        configmaps = service_info["configmaps"]
        service_type = service_info.get("type", "knative")
        service_types.add(service_type)

        delete_resources_for_service(
            core_api=core_api,
            custom_api=custom_api,
            configmaps=configmaps,
            name=name,
            service_type=service_type,
            namespace=namespace,
            console=console,
            force=force,
        )

    # Force delete any remaining pods if --force flag is set
    if force:
        # Build list of service names to check for pods
        # Include both found services and the original target name (in case service was already deleted)
        service_names_to_check = list(services)
        if name and name not in service_names_to_check:
            service_names_to_check.append(name)

        if service_names_to_check:
            console.print("\n[yellow]Force deleting any remaining pods...[/yellow]")
            for service_name in service_names_to_check:
                try:
                    # Get pods matching the service
                    pods = core_api.list_namespaced_pod(
                        namespace=namespace,
                        label_selector=f"kubetorch.com/service={service_name}",
                    ).items

                    if pods:
                        for pod in pods:
                            try:
                                core_api.delete_namespaced_pod(
                                    name=pod.metadata.name,
                                    namespace=namespace,
                                    grace_period_seconds=0,
                                    propagation_policy="Background",
                                )
                                console.print(f"✓ Force deleted pod [blue]{pod.metadata.name}[/blue]")
                            except ApiException as e:
                                if e.status != 404:  # Ignore if already deleted
                                    console.print(f"[red]Failed to delete pod {pod.metadata.name}: {e}[/red]")
                except Exception as e:
                    console.print(f"[red]Failed to list pods for service {service_name}: {e}[/red]")

        # Also check for any orphaned pods with kubetorch labels if using --all or --prefix
        if teardown_all or prefix:
            try:
                label_selector = "kubetorch.com/service"
                if teardown_all and config.username:
                    label_selector += f",kubetorch.com/username={config.username}"

                all_pods = core_api.list_namespaced_pod(namespace=namespace, label_selector=label_selector).items

                # Filter by prefix if specified
                if prefix:
                    all_pods = [
                        p for p in all_pods if p.metadata.labels.get("kubetorch.com/service", "").startswith(prefix)
                    ]

                # Delete any remaining pods not already handled
                for pod in all_pods:
                    if pod.metadata.name not in [
                        p.metadata.name
                        for s in service_names_to_check
                        for p in core_api.list_namespaced_pod(
                            namespace=namespace,
                            label_selector=f"kubetorch.com/service={s}",
                        ).items
                    ]:
                        try:
                            core_api.delete_namespaced_pod(
                                name=pod.metadata.name,
                                namespace=namespace,
                                grace_period_seconds=0,
                                propagation_policy="Background",
                            )
                            console.print(f"✓ Force deleted orphaned pod [blue]{pod.metadata.name}[/blue]")
                        except ApiException as e:
                            if e.status != 404:
                                console.print(f"[red]Failed to delete orphaned pod {pod.metadata.name}: {e}[/red]")
            except Exception as e:
                console.print(f"[red]Failed to list orphaned pods: {e}[/red]")

    console.print("\n[green]Teardown completed successfully[/green]")


@app.command("volumes")
def kt_volumes(
    action: VolumeAction = typer.Argument(VolumeAction.list, help="Action to perform"),
    name: str = typer.Argument(None, help="Volume name (for create action)"),
    storage_class: str = typer.Option(None, "--storage-class", "-c", help="Storage class"),
    size: str = typer.Option("10Gi", "--size", "-s", help="Volume size (default: 10Gi)"),
    access_mode: str = typer.Option("ReadWriteMany", "--access-mode", "-a", help="Access mode"),
    mount_path: str = typer.Option(
        None,
        "--mount-path",
        "-m",
        help=f"Mount path (default: /{KT_MOUNT_FOLDER}/{{name}})",
    ),
    namespace: str = typer.Option(
        globals.config.namespace,
        "-n",
        "--namespace",
    ),
    all_namespaces: bool = typer.Option(
        False,
        "--all-namespaces",
        "-A",
        help="List volumes across all namespaces",
    ),
):
    """Manage volumes used in Kubetorch services.

    Examples:

    .. code-block:: bash

        $ kt volumes

        $ kt volumes -A

        $ kt volumes create my-vol

        $ kt volumes create my-vol -c gp3-csi -s 20Gi

        $ kt volumes delete my-vol

        $ kt volumes ssh my-vol
    """
    from kubernetes import client

    from kubetorch import Volume
    from kubetorch.utils import load_kubeconfig

    load_kubeconfig()
    core_v1 = client.CoreV1Api()

    target_namespace = None
    if not all_namespaces:
        target_namespace = namespace or globals.config.namespace

    if action == VolumeAction.list:
        try:
            if all_namespaces:
                pvcs = core_v1.list_persistent_volume_claim_for_all_namespaces()
                title = "Kubetorch Volumes (All Namespaces)"
            else:
                pvcs = core_v1.list_namespaced_persistent_volume_claim(namespace=target_namespace)
                title = f"Kubetorch Volumes (Namespace: {target_namespace})"

            # List all Kubetorch PVCs
            kubetorch_pvcs = [
                pvc for pvc in pvcs.items if (pvc.metadata.annotations or {}).get("kubetorch.com/mount-path")
            ]

            if not kubetorch_pvcs:
                if all_namespaces:
                    console.print("[yellow]No volumes found in all namespaces[/yellow]")
                else:
                    console.print(f"[yellow]No volumes found in namespace {target_namespace}[/yellow]")
                return

            table = Table(title=title)
            if all_namespaces:
                table.add_column("Namespace", style="green")
            table.add_column("Name", style="cyan")
            table.add_column("PVC Name", style="blue")
            table.add_column("Status", style="green")
            table.add_column("Size", style="yellow")
            table.add_column("Storage Class", style="magenta")
            table.add_column("Access Mode", style="white")
            table.add_column("Mount Path", style="dim")

            for pvc in kubetorch_pvcs:
                # Extract volume name from PVC name
                volume_name = pvc.metadata.name
                status = pvc.status.phase
                size = pvc.spec.resources.requests.get("storage", "Unknown")
                storage_class = pvc.spec.storage_class_name or "Default"
                access_mode = pvc.spec.access_modes[0] if pvc.spec.access_modes else "Unknown"

                # Get mount path from annotations
                annotations = pvc.metadata.annotations or {}
                mount_path_display = annotations.get("kubetorch.com/mount-path", f"/{KT_MOUNT_FOLDER}/{volume_name}")

                status_color = "green" if status == "Bound" else "yellow" if status == "Pending" else "red"

                row_data = []
                if all_namespaces:
                    row_data.append(pvc.metadata.namespace)

                row_data.extend(
                    [
                        volume_name,
                        pvc.metadata.name,
                        f"[{status_color}]{status}[/{status_color}]",
                        size,
                        storage_class,
                        access_mode,
                        mount_path_display,
                    ]
                )

                table.add_row(*row_data)

            console.print(table)

        except Exception as e:
            console.print(f"[red]Failed to list volumes: {e}[/red]")
            raise typer.Exit(1)

    elif action == VolumeAction.ssh:
        if not name:
            console.print("[red]Volume name is required[/red]")
            raise typer.Exit(1)

        volume = Volume.from_name(name=name, namespace=namespace, core_v1=core_v1)
        volume.ssh()

    elif action == VolumeAction.create:
        if not name:
            console.print("[red]Volume name is required[/red]")
            raise typer.Exit(1)

        if all_namespaces:
            console.print("[red]Cannot create volume with --all-namespaces. Specify a namespace.[/red]")
            raise typer.Exit(1)

        try:
            volume = Volume(
                name=name,
                storage_class=storage_class,
                mount_path=mount_path,
                size=size,
                access_mode=access_mode,
                namespace=namespace,
            )

            if volume.exists():
                console.print(
                    f"[yellow]Volume {name} (PVC: {volume.pvc_name}) already exists in "
                    f"namespace {namespace}[/yellow]"
                )
                return

            console.print(f"Creating volume [blue]{name}[/blue]...")
            volume.create()

            console.print(f"[green]✓[/green] Successfully created volume [blue]{name}[/blue]")
            config = volume.config()
            for k, v in config.items():
                console.print(f"[bold]• {k}[/bold]: {v}")

        except Exception as e:
            console.print(f"[red]Failed to create volume {name}: {e}[/red]")
            raise typer.Exit(1)

    elif action == VolumeAction.delete:
        if not name:
            console.print("[red]Volume name is required[/red]")
            raise typer.Exit(1)

        if all_namespaces:
            console.print("[red]Cannot delete volume with --all-namespaces. Specify a namespace.[/red]")
            raise typer.Exit(1)

        try:
            volume = Volume.from_name(name=name, namespace=namespace, core_v1=core_v1)

            console.print(f"Deleting volume [blue]{name}[/blue]...")
            volume.delete()

            console.print(f"[green]✓[/green] Successfully deleted volume [blue]{name}[/blue]")

        except ValueError:
            console.print(f"[red]Volume {name} not found in namespace {namespace}[/red]")
            raise typer.Exit(1)

        except Exception as e:
            console.print(f"[red]Failed to delete volume {name}: {e}[/red]")
            raise typer.Exit(1)


@app.callback(invoke_without_command=True, help="Kubetorch CLI")
def main(
    ctx: typer.Context,
    version: bool = typer.Option(None, "--version", "-v", help="Show the version and exit."),
):
    if version:
        from kubetorch import __version__

        print(f"{__version__}")
    elif ctx.invoked_subcommand is None:
        subprocess.run("kubetorch --help", shell=True)
