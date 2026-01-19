import base64
import importlib
import inspect
import json
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

from kubetorch.resources.compute.utils import (
    ControllerRequestError,
    handle_controller_delete_error,
    print_byo_deletion_warning,
)

from kubetorch.serving.global_http_clients import get_sync_client

from kubetorch.serving.utils import is_running_in_kubernetes

from .cli_utils import (
    create_table_for_output,
    default_typer_values,
    follow_logs_in_cli,
    generate_logs_query,
    get_deployment_mode,
    get_ingress_host,
    get_last_updated,
    is_ingress_vpc_only,
    load_ingress,
    load_kubetorch_volumes_from_pods,
    load_logs_for_pod,
    load_runhouse_dashboard,
    load_selected_pod,
    notebook_placeholder,
    port_forward_to_pod,
    SecretAction,
    service_name_argument,
    validate_config_key,
    validate_pods_exist,
    VolumeAction,
)

from .utils import load_head_node_pod

try:
    import typer

    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
except ImportError:
    raise ImportError("Please install the required CLI dependencies: `pip install 'kubetorch[client]'`")

import kubetorch.provisioning.constants as provisioning_constants

from kubetorch import globals
from kubetorch.config import ENV_MAPPINGS
from kubetorch.serving.utils import DEFAULT_DEBUG_PORT

from .constants import BULLET_UNICODE, DEFAULT_TAIL_LENGTH, KT_MOUNT_FOLDER

try:
    from .internal.cli import register_internal_commands

    _INTERNAL_COMMANDS_AVAILABLE = True
except ImportError:
    _INTERNAL_COMMANDS_AVAILABLE = False

from .logger import get_logger

app = typer.Typer(add_completion=False)
server_app = typer.Typer(help="Kubetorch server commands.")
app.add_typer(server_app, name="server")
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

    controller = globals.controller_client()

    def fail(msg, pods=None):
        console.print(f"[red]{msg}[/red]")
        if pods:
            for pn in pods:
                try:
                    pod_info = controller.get_pod(namespace, pn)
                    pod_logs = controller.get_pod_logs(namespace, pn, container="kubetorch")
                    # Format pod info similar to kubectl describe
                    pod_status = pod_info.get("status", {})
                    pod_phase = pod_status.get("phase", "Unknown")
                    conditions = pod_status.get("conditions", [])
                    container_statuses = pod_status.get("containerStatuses", [])
                    describe_output = f"Phase: {pod_phase}\n"
                    describe_output += "Conditions:\n"
                    for c in conditions:
                        describe_output += f"  {c.get('type')}: {c.get('status')} ({c.get('reason', '')})\n"
                    describe_output += "Container Statuses:\n"
                    for cs in container_statuses:
                        state = cs.get("state", {})
                        state_name = list(state.keys())[0] if state else "unknown"
                        describe_output += f"  {cs.get('name')}: {state_name}\n"
                        if state_name == "waiting":
                            describe_output += f"    Reason: {state.get('waiting', {}).get('reason', '')}\n"
                            describe_output += f"    Message: {state.get('waiting', {}).get('message', '')}\n"
                    console.print(Panel(describe_output, title=f"POD STATUS {pn}"))
                    console.print(Panel(pod_logs or "(no logs)", title=f"LOGS {pn}"))
                except Exception:
                    pass
        raise typer.Exit(1)

    # --------------------------------------------------
    # 1. Determine mode
    # --------------------------------------------------
    name, deployment_mode = get_deployment_mode(name, namespace)

    console.print(f"[bold blue]Checking {deployment_mode} service...[/bold blue]")

    # --------------------------------------------------
    # 2. Get pods
    # --------------------------------------------------
    console.print("[bold blue]Checking deployment pod...[/bold blue]")

    pods = validate_pods_exist(name, namespace)  # returns dicts now, we assume it

    if not pods:
        if deployment_mode == "knative":
            # Knative service exists (from get_deployment_mode) but has no pods = scaled to zero
            console.print(f"[yellow]Knative service {name} is scaled to zero (no pods running).[/yellow]")
            return
        fail("No deployment pods found.")

    # --------------------------------------------------
    # 3. Pick running pod
    # --------------------------------------------------
    running_pod = None
    for p in pods:
        phase = p.get("status", {}).get("phase")
        del_ts = p.get("metadata", {}).get("deletionTimestamp")
        if phase == "Running" and not del_ts:
            running_pod = p
            break

    if not running_pod:
        fail("No running deployment pod found.", [p["metadata"]["name"] for p in pods])

    pod_name = running_pod["metadata"]["name"]

    # --------------------------------------------------
    # 4. Rsync check
    # --------------------------------------------------
    console.print("[bold blue]Checking rsync...[/bold blue]")

    try:
        from kubetorch.data_store import DataStoreClient

        data_store = DataStoreClient(namespace=namespace)
        items = data_store.ls(key=name)
        if not items:
            fail("Rsync directory exists but is empty.", [pod_name])
    except Exception as e:
        fail(f"Rsync check failed: {e}", [pod_name])

    # --------------------------------------------------
    # 5. Service health check
    # --------------------------------------------------
    console.print("[bold blue]Checking service call...[/bold blue]")

    try:
        with port_forward_to_pod(
            pod_name=pod_name,
            namespace=namespace,
            local_port=32300,
            remote_port=32300,
        ) as lp:
            url = f"http://localhost:{lp}/health"
            r = get_sync_client().get(url, timeout=10)
            if not r.is_success:
                fail(f"Service returned {r.status_code}: {r.text}", [pod_name])
    except Exception as e:
        fail(f"Service call failed: {e}", [pod_name])

    # --------------------------------------------------
    # 6. GPU check (dict only)
    # --------------------------------------------------
    containers = running_pod.get("spec", {}).get("containers", [])
    gpu_requested = False

    for c in containers:
        limits = c.get("resources", {}).get("limits", {})
        if "nvidia.com/gpu" in limits:
            gpu_requested = True
            break

    if gpu_requested:
        console.print("[bold blue]Checking GPU nodes...[/bold blue]")
        nodes = controller.list_nodes().get("items", [])
        gpu_nodes = [n for n in nodes if int(n.get("status", {}).get("capacity", {}).get("nvidia.com/gpu", "0")) > 0]
        if not gpu_nodes:
            console.print("[yellow]No GPU nodes currently active.[/yellow]")

        # DCGM
        dcgm_ns = globals.config.install_namespace
        dcgm_pods = controller.list_pods(
            namespace=dcgm_ns,
            label_selector="app.kubernetes.io/name=dcgm-exporter",
        ).get("items", [])
        if not dcgm_pods:
            console.print(f"[yellow]No DCGM exporter found in {dcgm_ns}[/yellow]")

    # --------------------------------------------------
    # 7. Log streaming check (dict only)
    # --------------------------------------------------
    if globals.config.stream_logs:
        console.print("[bold blue]Checking log streaming...[/bold blue]")

        try:
            # Check that data store exists in the namespace (Loki is embedded in data store)
            controller.get_service(
                name="kubetorch-data-store",
                namespace=namespace,
            )
            q = f'{{pod="{pod_name}", namespace="{namespace}"}}'
            logs = load_logs_for_pod(query=q, print_pod_name=False, timeout=2.0, namespace=namespace)
            if logs is None:
                fail("No logs found for pod", [pod_name])
        except Exception as e:
            fail(f"Log streaming check failed: {e}", [pod_name])

    console.print("[bold green]✓ All service checks passed[/bold green]")


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


def _connect_pdb_websocket(namespace: str, pod: str, port: int, pod_ip: str = None):
    """Connect to a PDB WebSocket PTY server running in a pod."""
    import asyncio

    import websockets

    # Check if we're running inside the cluster
    in_cluster = is_running_in_kubernetes()

    async def run_websocket_pty():
        """Run the WebSocket PTY client."""
        # Check if running in the same pod we're debugging
        current_pod = os.environ.get("POD_NAME")
        same_pod = in_cluster and current_pod == pod

        # Determine WebSocket URL based on whether we're in-cluster or not
        if same_pod:
            # Running in the same pod - connect to localhost
            ws_url = f"ws://localhost:{port}"
            console.print(f"[blue]Connecting to local debug server on port {port}...[/blue]")
        elif in_cluster and pod_ip:
            # Direct connection to pod IP (no port forward needed)
            ws_url = f"ws://{pod_ip}:{port}"
            console.print(f"[blue]Connecting directly to pod {pod} ({pod_ip}) in cluster...[/blue]")
        elif not in_cluster:
            # Use port forwarding for external connections
            ws_url = f"ws://localhost:{port}"
            console.print(f"[blue]Setting up port forward to {pod}:{port}...[/blue]")
        else:
            # In cluster but no pod IP provided
            console.print("[red]Error: Running in cluster but no --pod-ip provided.[/red]")
            console.print("[yellow]Please copy the full kt debug command from the breakpoint output.[/yellow]")
            raise typer.Exit(1)

        async def connect_and_run():
            """Connect to WebSocket and run the PDB session."""
            console.print(f"[green]Connecting to PDB session at {ws_url}...[/green]")

            try:
                async with websockets.connect(ws_url) as websocket:
                    console.print("[green]Connected! PDB session active. Press Ctrl+D or type 'q' to exit.[/green]")

                    async def send_input():
                        """Read lines from stdin and send to WebSocket."""
                        loop = asyncio.get_event_loop()
                        try:
                            while True:
                                # Read a line from stdin in a thread to not block the event loop
                                line = await loop.run_in_executor(None, sys.stdin.readline)
                                if not line:
                                    # EOF (Ctrl+D)
                                    break
                                await websocket.send(line)
                        except asyncio.CancelledError:
                            pass
                        except Exception:
                            pass

                    async def receive_output():
                        """Receive output from the WebSocket and write to stdout."""
                        try:
                            async for message in websocket:
                                if message:
                                    sys.stdout.write(message)
                                    sys.stdout.flush()
                        except websockets.exceptions.ConnectionClosed:
                            pass
                        except asyncio.CancelledError:
                            pass

                    # Run both tasks concurrently
                    send_task = asyncio.create_task(send_input())
                    receive_task = asyncio.create_task(receive_output())

                    try:
                        # Wait for either task to complete
                        done, pending = await asyncio.wait(
                            [send_task, receive_task],
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        # Cancel the other task
                        for task in pending:
                            task.cancel()
                            try:
                                await task
                            except asyncio.CancelledError:
                                pass
                    except asyncio.CancelledError:
                        send_task.cancel()
                        receive_task.cancel()

            except websockets.exceptions.WebSocketException as e:
                console.print(f"[red]WebSocket error: {e}[/red]")
                console.print("[yellow]Make sure the debug server is running in the pod.[/yellow]")
                return
            except KeyboardInterrupt:
                pass

            console.print("\n[yellow]PDB session ended.[/yellow]")

        # If not in cluster and not same pod, wrap with port forward context manager
        if not in_cluster and not same_pod:
            with port_forward_to_pod(
                namespace=namespace,
                pod_name=pod,
                local_port=port,
                remote_port=port,
                health_endpoint=None,  # No health check needed
            ):
                await connect_and_run()
        else:
            # Direct connection (either in-cluster or same pod), no port forward needed
            await connect_and_run()

    # Run the async function
    asyncio.run(run_websocket_pty())


@app.command("debug")
def kt_debug(
    pod: str = typer.Argument(..., help="Pod name"),
    namespace: str = typer.Option(
        globals.config.namespace,
        "-n",
        "--namespace",
    ),
    port: int = typer.Option(DEFAULT_DEBUG_PORT, help="Debug port used for remote debug server"),
    mode: str = typer.Option("pdb", "--mode", help="Debug mode: 'pdb' (default) or 'pdb-ui'"),
    pod_ip: str = typer.Option(None, "--pod-ip", help="Pod IP address for in-cluster connections"),
):
    """Start an interactive debugging session on the pod, which will connect to the debug server inside the service.
    Before running this command, you must call a method on the service with debug=True or add a
    breakpoint() call into your code to enable debugging.

    Debug modes:
    - "pdb" (default): Standard PDB over WebSocket PTY (works over SSH and inside cluster)
    - "pdb-ui": Web-based PDB UI (requires running locally)
    """
    import webbrowser

    debug_mode = mode.lower()

    # Check if running in Kubernetes - only block for pdb-ui mode
    if is_running_in_kubernetes() and debug_mode == "pdb-ui":
        console.print("[red]The pdb-ui debug mode requires running locally (cannot open browser in cluster).[/red]")
        console.print(
            "[yellow]Try using pdb mode instead (which must be changed via KT_DEBUG_MODE or DebugConfig), which works inside the cluster.[/yellow]"
        )
        raise typer.Exit(1)

    if debug_mode == "pdb-ui":
        # Use web-based PDB UI
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
            console.print("[yellow]Press Ctrl+C to stop the debugging session and close the UI.[/yellow]")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                console.print("\n[yellow]Debugging session ended.[/yellow]")
                raise typer.Exit(0)
    elif debug_mode == "pdb":
        # Use standard PDB over WebSocket PTY
        _connect_pdb_websocket(namespace, pod, port, pod_ip)
    else:
        console.print(f"[red]Unknown debug mode: {debug_mode}. Use 'pdb' or 'pdb-ui'.[/red]")
        raise typer.Exit(1)


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
        console.print(f"Successfully deployed all decorated functions and modules from {target}.")


@app.command("dashboard", hidden=True)
def kt_dashboard(
    local: bool = typer.Option(False, "-l", "--local", help="Use a local server"),
    namespace: str = typer.Option(
        globals.config.install_namespace,
        "-n",
        "--namespace",
    ),
):
    """Open Runhouse Dashboard"""
    processes = []

    # Connect to Runhouse Dashboard
    runhouse_process = load_runhouse_dashboard(
        namespace=namespace,
        local_server=local,
    )

    if runhouse_process:
        processes += runhouse_process

    console.print(f"[green]Runhouse dashboard running in namespace {namespace}[/green]")

    console.print("[yellow]Both services running. Press Ctrl+C to stop...[/yellow]")

    try:
        # Main thread waits here
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        for process in processes:
            process.terminate()


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
    endpoint_placeholder = "METHOD_OR_CLS_NAME"
    args_placeholder = []

    try:
        name, deployment_mode = get_deployment_mode(name, namespace)
    except Exception:
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
    """List all Kubetorch resources.

    Examples:

    .. code-block:: bash

      $ kt list

      $ kt list -t dev-branch
    """

    # Import here to avoid circular imports
    from kubetorch.provisioning.service_manager import ServiceManager

    try:
        # Use unified service discovery
        unified_services = ServiceManager.discover_services(namespace=namespace, name_filter=tag)

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

        try:
            pods_result = globals.controller_client().list_pods(
                namespace=namespace, label_selector=f"{provisioning_constants.KT_SERVICE_LABEL}"
            )
            pods = pods_result.get("items", [])
        except Exception as e:
            logger.warning(f"Failed to list pods for all services in namespace {namespace}: {e}")
            return

        # Build pod map - for selector-based pools, use _pods from the resource (found via selector)
        pod_map = {}
        for svc in unified_services:
            if svc["template_type"] == "selector":
                # For selector-based pools, use actual pods from K8s (found via selector)
                pod_map[svc["name"]] = svc["resource"].get("_pods", [])
            else:
                pod_map[svc["name"]] = [
                    pod
                    for pod in pods
                    if pod.get("metadata", {}).get("labels", {}).get(provisioning_constants.KT_SERVICE_LABEL)
                    == svc["name"]
                ]

        # Create table
        table_columns = [
            ("RESOURCE", "cyan"),
            ("TYPE", "magenta"),
            ("STATUS", "green"),
            ("# OF PODS", "yellow"),
            ("POD NAMES", "red"),
            ("VOLUMES", "blue"),
            ("LAST STATUS CHANGE", "yellow"),
            ("TTL", "yellow"),
            ("CREATOR", "yellow"),
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
            ttl = annotations.get(provisioning_constants.INACTIVITY_TTL_ANNOTATION, "None")
            creator = labels.get(provisioning_constants.KT_USERNAME_LABEL, "—")

            volumes_display = load_kubetorch_volumes_from_pods(pods)

            # Get resources from pods
            cpu = memory = gpu = None
            if kind == "ksvc":
                cond = status_data.get("conditions", [{}])[0]
                status = cond.get("status")
                display_status = {
                    "True": "[green]Ready[/green]",
                    "Unknown": "[yellow]Creating[/yellow]",
                }.get(status, "[red]Failed[/red]")
                # Get resources from pod spec instead of revision
                if pods:
                    try:
                        container = pods[0].get("spec", {}).get("containers", [{}])[0]
                        reqs = container.get("resources", {}).get("requests", {})
                        cpu = reqs.get("cpu")
                        memory = reqs.get("memory")
                        gpu = reqs.get("nvidia.com/gpu") or reqs.get("gpu")
                    except Exception as e:
                        logger.warning(f"Could not get resources from pod for {name}: {e}")
            elif kind == "selector":
                # Selector-based pools: status based on actual pods found via selector
                num_pods = len(pods)
                has_selector = bool(res.get("_selector"))
                if num_pods > 0:
                    # Check if pods are running
                    running_pods = [p for p in pods if p.get("status", {}).get("phase") == "Running"]
                    if len(running_pods) == num_pods:
                        display_status = "[green]Ready[/green]"
                    elif len(running_pods) > 0:
                        display_status = "[yellow]Scaling[/yellow]"
                    else:
                        display_status = "[yellow]Pending[/yellow]"

                    # Infer resource type from pod's ownerReferences
                    try:
                        owner_refs = pods[0].get("metadata", {}).get("ownerReferences", [])
                        if owner_refs:
                            owner_kind = owner_refs[0].get("kind", "").lower()
                            # ReplicaSet is owned by Deployment, so show "deployment"
                            if owner_kind == "replicaset":
                                kind = "deployment"
                            elif owner_kind:
                                kind = owner_kind
                    except Exception:
                        pass  # Keep "selector" if we can't infer

                    # Extract resources from first pod's container
                    try:
                        container = pods[0].get("spec", {}).get("containers", [{}])[0]
                        reqs = container.get("resources", {}).get("requests", {})
                        cpu = reqs.get("cpu")
                        memory = reqs.get("memory")
                        gpu = reqs.get("nvidia.com/gpu") or reqs.get("gpu")
                    except Exception as e:
                        logger.warning(f"Failed to get resources for selector pool {name}: {e}")
                elif has_selector:
                    display_status = "[yellow]No pods[/yellow]"
                else:
                    display_status = "[yellow]Waiting[/yellow]"
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
                elif kind in ("pytorchjob", "tfjob", "mxjob", "xgboostjob"):
                    # Training jobs use conditions to track status
                    conditions = {c["type"]: c["status"] for c in status_data.get("conditions", [])}
                    if conditions.get("Succeeded") == "True":
                        display_status = "[green]Succeeded[/green]"
                    elif conditions.get("Running") == "True":
                        display_status = "[green]Running[/green]"
                    elif conditions.get("Created") == "True":
                        display_status = "[yellow]Created[/yellow]"
                    elif conditions.get("Failed") == "True":
                        display_status = "[red]Failed[/red]"
                    else:
                        display_status = "[yellow]Pending[/yellow]"
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
            for pod in pods:
                pod_status = pod.get("status", {}).get("phase", "Unknown")
                container_statuses = pod.get("status", {}).get("containerStatuses", []) or []
                ready = all(cs.get("ready", False) for cs in container_statuses)
                if ready and pod_status == "Running":
                    color = "green"
                elif "Creating" in display_status or "Scaling" in display_status:
                    color = "yellow"
                else:
                    color = "red"

                pod_name = pod.get("metadata", {}).get("name", "unknown")
                pod_lines.append(f"[{color}]{pod_name}[/{color}]")

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
                cpu or "—",
                memory or "—",
                gpu or "—",
            )

        table.pad_bottom = 1
        console.print(table)

    except Exception as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


@app.command("port-forward")
def kt_port_forward(
    name: str = service_name_argument(help="Service name"),
    local_port: int = typer.Argument(
        default=provisioning_constants.DEFAULT_KT_SERVER_PORT, help="Local port to bind to"
    ),
    remote_port: int = typer.Argument(
        default=provisioning_constants.DEFAULT_KT_SERVER_PORT,
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

    name, _ = get_deployment_mode(name, namespace)
    pods = validate_pods_exist(name, namespace)

    if not pods:
        console.print(f"[red]No pods found for service {name}[/red]")
        raise typer.Exit(1)

    try:
        sorted_by_time = sorted(pods, key=lambda p: p.get("metadata", {}).get("creationTimestamp", "9999"))
    except Exception:
        sorted_by_time = pods  # fallback if timestamps missing

    if pod:
        # If pod is an index
        if pod.isdigit():
            idx = int(pod)
            if idx < 0 or idx >= len(sorted_by_time):
                console.print(f"[red]Pod index {idx} out of range[/red]")
                raise typer.Exit(1)
            chosen = sorted_by_time[idx]
        else:
            # Match by pod name
            matches = [p for p in sorted_by_time if p.get("metadata", {}).get("name") == pod]
            if not matches:
                console.print(f"[red]Pod '{pod}' not found[/red]")
                raise typer.Exit(1)
            chosen = matches[0]
    else:
        # Default: oldest pod = first element
        chosen = sorted_by_time[0]

    pod_name = chosen.get("metadata", {}).get("name")
    if not pod_name:
        console.print("[red]Pod missing metadata.name[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Forwarding to pod {pod_name}[/blue]")

    cmd = [
        "kubectl",
        "port-forward",
        f"pod/{pod_name}",
        f"{local_port}:{remote_port}",
        "-n",
        namespace,
    ]

    console.print(f"[green]Running: {' '.join(cmd)}[/green]")
    subprocess.run(cmd)


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
        globals.config.namespace,
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

        $ kt secrets -A  # list secrets in all namespaces (note: requires cluster-wide RBAC)

        $ kt secrets create --provider aws  # create a secret with the aws credentials in `default` namespace

        $ kt secrets create my_secret -v ENV_VAR_1 -v ENV_VAR_2 -n my_namespace  # create a secret using env vars

        $ kt secrets delete my_secret -n my_namespace  # delete a secret called `my_secret` from `my_namespace` namespace

        $ kt secrets delete aws   # delete a secret called `aws` from `default` namespace
    """
    import kubetorch as kt
    from kubetorch.resources.compute.utils import delete_secrets, list_secrets
    from kubetorch.resources.secrets.kubernetes_secrets_client import KubernetesSecretsClient

    secrets_client = KubernetesSecretsClient(namespace=namespace)

    if action == SecretAction.list:
        secrets = list_secrets(
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
            namespace=namespace,
            prefix=prefix,
            all_namespaces=all_namespaces,
            console=console,
        )

        if not secrets_to_delete:
            console.print("[yellow] No secrets found[/yellow]")
            raise typer.Exit(0)

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
    """SSH into a remote service. By default, will SSH into the first running pod.
    For Ray clusters, prioritizes the head node.

    Examples:

    .. code-block:: bash

        $ kt ssh my_service
    """
    from kubetorch.provisioning.utils import pod_is_running

    try:
        # Validate service exists and get deployment mode
        name, deployment_mode = get_deployment_mode(name, namespace)

        # Get and validate pods
        pods = validate_pods_exist(name, namespace)

        # case when the user provides a specific pod to ssh into
        if pod:
            pod_name = load_selected_pod(service_name=name, provided_pod=pod, service_pods=pods)
        else:
            # select based on deployment mode
            running_pods = [p for p in pods if pod_is_running(p)]

            if not running_pods:
                console.print(f"[red]No running pods found for service {name}[/red]")
                raise typer.Exit(1)

            pod_name = load_head_node_pod(running_pods, deployment_mode=deployment_mode)

        console.print(f"[green]Found pod:[/green] [blue]{pod_name}[/blue] ({deployment_mode})")
        console.print("[yellow]Connecting to pod...[/yellow]")

        # Still need subprocess for the interactive shell
        subprocess.run(
            ["kubectl", "exec", "-it", pod_name, "-n", namespace, "--", "/bin/bash"],
            check=True,
        )

    except Exception as e:
        console.print(f"[red]{e}[/red]")
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
    from kubetorch.resources.callables.module import Module
    from kubetorch.resources.compute.utils import (
        _collect_modules,
        delete_resources_for_services,
        fetch_resources_for_teardown,
    )

    name, yes, teardown_all, namespace, prefix, force, exact_match = default_typer_values(
        name, yes, teardown_all, namespace, prefix, force, exact_match
    )

    force_deleting_prefix_msg = "Force deleting" if force else "Deleting"
    if teardown_all:
        if not config.username:
            console.print(
                "[red]Username is not found, can't delete all services. Please set up a username, provide a service "
                "name or use the --prefix flag[/red]"
            )
            raise typer.Exit(1)

        console.print(f"{force_deleting_prefix_msg} all services for username [blue]{config.username}[/blue]...")

    elif prefix:
        console.print(
            f"{force_deleting_prefix_msg} all services with prefix [blue]{prefix}[/blue] in [blue]{namespace}[/blue] namespace..."
        )
    else:
        if not name:
            console.print("[red]Please provide a service name or use the --all or --prefix flags[/red]")
            raise typer.Exit(1)

        console.print(
            f"{force_deleting_prefix_msg} resources for service [blue]{name}[/blue] in [blue]{namespace}[/blue] namespace..."
        )

    # Case 1: confirmation or force flags are not provided. The flow will be as follows:
    #   1. List the services that will be deleted
    #   2. Ask for confirmation
    #   3. Delete if confirmed
    if not yes and not force:
        # if --force is provided, we don't need additional confirmation
        teardown_result = fetch_resources_for_teardown(
            namespace=namespace,
            target=name,
            prefix=prefix,
            teardown_all=teardown_all,
            username=config.username if teardown_all else None,
            exact_match=exact_match,
        )

        # Extract resources list from the response dict
        resource_list = teardown_result.get("resources", [])
        service_names = list({r["name"] for r in resource_list})
        service_count = len(service_names)

        if teardown_all or prefix:
            service_word = "service" if service_count == 1 else "services"
            if not resource_list:
                console.print("[red]No services found[/red]")
                raise typer.Exit(0)
            else:
                console.print(f"[yellow]Found [bold]{service_count}[/bold] {service_word} to delete.[/yellow]")

        if name and not resource_list:
            console.print(f"[red]Service [bold]{name}[/bold] not found[/red]")
            raise typer.Exit(1)

        # Confirmation prompt
        if not yes and service_count >= 1:
            console.print("The following resources will be deleted:")
            for svc_name in service_names:
                console.print(f" • [reset]{svc_name}")

        if not yes and not force:
            confirm = typer.confirm("\nDo you want to proceed?")
            if not confirm:
                console.print("[yellow]Teardown cancelled[/yellow]")
                raise typer.Exit(0)

        # Pass the dict directly (controller uses resources, fetches pools internally)
        services = teardown_result
        prefix = None
        teardown_all = None

    else:
        # Case when service_name is a module or file path (i.e. the `kt deploy` usage path)
        if ":" in name or ".py" in name or "." in name:
            to_down, _ = _collect_modules(name)
            name = [mod.service_name for mod in to_down if isinstance(mod, Module)]

        services = name
        # Case 2: confirmation or force flags are provided.
        # List the services to delete based provided name, prefix flag, or --all flag
    try:
        username = config.username if teardown_all else None
        delete_result = delete_resources_for_services(
            services=services,
            namespace=namespace,
            force=force,
            prefix=prefix,
            teardown_all=teardown_all,
            username=username,
            exact_match=exact_match,
        )

        if isinstance(delete_result, str):
            console.print(delete_result)
        else:
            deleted_resources = delete_result.get("deleted_resources", [])
            byo_deleted_services = delete_result.get("byo_deleted_services", [])

            # Case where no services were found
            if not deleted_resources and not byo_deleted_services:
                if prefix or teardown_all:
                    console.print("No services found")
                elif name:
                    console.print(f"Service {name} not found")
                return

            if byo_deleted_services:
                print_byo_deletion_warning(byo_deleted_services, console)

            force_deleted_msg_prefix = "✓ Force deleted" if force else "✓ Deleted"

            for resource in deleted_resources:
                kind = resource.get("kind", "resource").lower()
                resource_name = resource.get("name")
                console.print(f"{force_deleted_msg_prefix} [reset]{kind} [blue]{resource_name}[/blue]")

            console.print("\n[green]Teardown completed successfully[/green]")

    except Exception as e:
        if isinstance(e, ControllerRequestError):
            handle_controller_delete_error(service_name=name, controller_error=str(e), console=console)
        else:
            console.print(f"[red] Failed to run `kt teardown`: {e}[/red]")
            raise typer.Exit(1)


@app.command("volumes")
def kt_volumes(
    action: VolumeAction = typer.Argument(VolumeAction.list, help="Action to perform"),
    name: str = typer.Argument(None, help="Volume name (for create action)"),
    storage_class: str = typer.Option(None, "--storage-class", "-c", help="Storage class"),
    mount_path: str = typer.Option(None, "--mount-path", "-m", help="Mount path"),
    size: str = typer.Option("10Gi", "--size", "-s", help="Volume size (default: 10Gi)"),
    access_mode: str = typer.Option("ReadWriteMany", "--access-mode", "-a", help="Access mode"),
    pv: str = typer.Option(None, "--pv", help="Bind to an existing PersistentVolume by name"),
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

        $ kt volumes create my-vol --pv existing-pv-name

        $ kt volumes delete my-vol

        $ kt volumes ssh my-vol
    """
    from kubetorch import globals, Volume

    controller_client = globals.controller_client()

    target_namespace = None
    if not all_namespaces:
        target_namespace = namespace or globals.config.namespace

    if action == VolumeAction.list:
        try:
            if all_namespaces:
                # Controller doesn't have all-namespaces endpoint, so we need to list from each namespace
                # Get list of allowed namespaces from config (or use common defaults)
                allowed_namespaces = globals.config.deployment_namespaces or ["default", "kubetorch"]
                all_pvcs = []
                for ns in allowed_namespaces:
                    try:
                        result = controller_client.list_pvcs(ns)
                        all_pvcs.extend(result.get("items", []))
                    except Exception:
                        # Skip namespaces that don't exist or we don't have access to
                        pass
                pvcs_items = all_pvcs
                title = "Kubetorch Volumes (All Namespaces)"
            else:
                result = controller_client.list_pvcs(target_namespace)
                pvcs_items = result.get("items", [])
                title = f"Kubetorch Volumes (Namespace: {target_namespace})"

            # List all Kubetorch PVCs
            kubetorch_pvcs = [
                pvc
                for pvc in pvcs_items
                if (pvc.get("metadata", {}).get("annotations") or {}).get("kubetorch.com/mount-path")
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
                volume_name = pvc["metadata"]["name"]
                status = pvc["status"]["phase"]
                size = pvc["spec"]["resources"]["requests"].get("storage", "Unknown")
                storage_class = pvc["spec"].get("storageClassName") or "Default"
                access_mode = pvc["spec"]["accessModes"][0] if pvc["spec"].get("accessModes") else "Unknown"

                # Get mount path from annotations
                annotations = pvc["metadata"].get("annotations") or {}
                mount_path_display = annotations.get("kubetorch.com/mount-path", f"/{KT_MOUNT_FOLDER}/{volume_name}")

                status_color = "green" if status == "Bound" else "yellow" if status == "Pending" else "red"

                row_data = []
                if all_namespaces:
                    row_data.append(pvc["metadata"]["namespace"])

                row_data.extend(
                    [
                        volume_name,
                        pvc["metadata"]["name"],
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

        volume = Volume.from_name(name=name, namespace=namespace)
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
                volume_name=pv,
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
            volume = Volume.from_name(name=name, namespace=namespace)

            console.print(f"Deleting volume [blue]{name}[/blue]...")
            volume.delete()

            console.print(f"[green]✓[/green] Successfully deleted volume [blue]{name}[/blue]")

        except ValueError:
            console.print(f"[red]Volume {name} not found in namespace {namespace}[/red]")
            raise typer.Exit(1)

        except Exception as e:
            console.print(f"[red]Failed to delete volume {name}: {e}[/red]")
            raise typer.Exit(1)


@app.command("notebook")
def kt_notebook(
    name: str = typer.Argument(None, help="Service name"),
    cpus: str = typer.Option(None, "--cpus", help="CPU resources (e.g., '2', '500m')"),
    memory: str = typer.Option(None, "--memory", "-m", help="Memory resources (e.g., '4Gi', '512Mi')"),
    gpus: str = typer.Option(None, "--gpus", help="Number of GPUs"),
    image: str = typer.Option(None, "--image", "-i", help="Container image to use"),
    namespace: str = typer.Option(
        globals.config.namespace,
        "-n",
        "--namespace",
    ),
    local_port: int = typer.Option(8888, "--port", "-p", help="Local port for notebook access"),
    inactivity_ttl: str = typer.Option(None, "--ttl", help="Inactivity TTL (e.g., '1h', '30m')"),
    restart_kernels: bool = typer.Option(
        True,
        "--restart/--no-restart",
        help="Restart notebook kernel sessions upon reconnect",
    ),
):
    """
    Launch a JupyterLab notebook server on a new or existing Kubetorch service. The notebook service will continue
    running after you exit, and you can reconnect to it until the service is torn down.

    Examples:

    .. code-block:: bash

        $ kt notebook tune-hpo # Launch notebook into new or existing service with name "tune-hpo"

        $ kt notebook --cpus 4 --memory 8Gi # Launch with specific resources

        $ kt notebook --gpus 1 --cpus 8 --memory 16Gi --image nvcr.io/nvidia/pytorch:23.10-py3  # Launch with GPU and custom image

        $ kt notebook --gpus 1 --cpus 8 --memory 16Gi --no-restart # Don't restart kernels on reconnect
    """
    import webbrowser

    import kubetorch as kt

    if is_running_in_kubernetes():
        console.print(
            "[red]Notebook command is not supported when running inside Kubernetes. "
            "Please run this command locally.[/red]"
        )
        raise typer.Exit(1)

    # Build compute configuration
    compute_kwargs = {
        "namespace": namespace,
        "cpus": cpus,
        "memory": memory,
        "gpus": gpus,
        "inactivity_ttl": inactivity_ttl,
    }

    if image:
        compute_kwargs["image"] = kt.Image(image_id=image)
    else:
        if gpus:
            console.print(
                "[yellow]Launching with GPUs without a CUDA-enabled image may limit GPU usability. "
                "Specify an appropriate image, for example: "
                "[bold]`kt notebook --gpus 1 --image nvcr.io/nvidia/pytorch:23.10-py3`[/bold].[/yellow]"
            )
            return

        compute_kwargs["image"] = kt.Image()

    compute = kt.Compute(**compute_kwargs)

    # Generate service name
    service_name = name or "kt-notebook"

    # Check if local port is available
    from kubetorch.resources.compute.utils import find_available_port

    original_port = local_port
    try:
        local_port = find_available_port(local_port, max_tries=5)
        if local_port != original_port:
            console.print(f"[yellow]Port {original_port} already in use, using port {local_port} instead.[/yellow]")
    except RuntimeError:
        console.print(f"\n[red]Ports {original_port}-{original_port + 4} are all in use.[/red]")
        raise typer.Exit(1)

    console.print("[cyan]Setting up notebook...[/cyan]")

    try:
        # If the service already exists -> load it, then compare to what was requested
        # If the service doesn't exist -> deploy with requested parameters
        remote_fn = kt.fn(notebook_placeholder, name=service_name).to(compute, stream_logs=False, get_if_exists=True)
        compute = remote_fn.compute

        # Check if requested parameters match the existing compute
        mismatches = []
        expected_params = {
            "cpus": cpus,
            "memory": memory,
            "gpus": gpus,
            "image": image,
        }

        for key, requested_value in expected_params.items():
            if requested_value is None:
                # Skip unset CLI options
                continue

            existing_value = getattr(compute, key, None)
            if key == "image":
                # compare image_ids
                existing_value = getattr(existing_value, "image_id", existing_value)

            if existing_value != requested_value:
                mismatches.append((key, existing_value, requested_value))

        if mismatches:
            console.print("[yellow]Cannot reuse existing notebook due to mismatched parameters:[/yellow]")
            for key, existing, requested in mismatches:
                display_existing = existing if existing is not None else "<default>"
                console.print(f"  - [bold]{key}[/bold]: existing = '{display_existing}', requested = '{requested}'")
            console.print(
                f"\n[yellow]Delete the existing notebook service ([bold]`kt teardown {service_name}`[/bold]) "
                "or create a new one with a different name.[/yellow]"
            )
            return

        # Ensure jupyter lab is installed
        compute.pip_install(["jupyterlab"])

        # Get pod information
        pods_result = globals.controller_client().list_pods(
            namespace=namespace, label_selector=f"kubetorch.com/service={remote_fn.service_name}"
        )
        pods = pods_result.get("items", [])
        if not pods:
            console.print(f"[red]No pods found for service {service_name}[/red]")
            raise typer.Exit(1)

        # Sort by creation timestamp and get the first pod
        sorted_pods = sorted(pods, key=lambda p: p.get("metadata", {}).get("creationTimestamp", ""))
        pod_name = sorted_pods[0].get("metadata", {}).get("name")
        console.print(f"[green]Service is up (pod: {pod_name})[/green]")

        # Start jupyter in background
        jupyter_cmd = (
            'bash -c "nohup jupyter lab --ip=0.0.0.0 --port=8888 --no-browser '
            "--allow-root --ServerApp.token='' --ServerApp.password='' "
            "--NotebookApp.token='' --NotebookApp.password='' "
            '> /tmp/jupyter.log 2>&1 &"'
        )
        if restart_kernels:
            start_cmd_result = compute.run_bash(jupyter_cmd)
            if start_cmd_result and start_cmd_result[0][0] != 0:
                console.print("[red]Error starting Jupyter Lab[/red]", start_cmd_result)
                raise typer.Exit(1)

        # Wait for jupyter to start
        for i in range(5):
            check_cmd = "tail -20 /tmp/jupyter.log"
            result = compute.run_bash(check_cmd)
            if result and result[0][0] == 0:
                output = result[0][1]
                if ("Jupyter Server" in output and "is running" in output) or ("Connecting to kernel" in output):
                    break
                else:
                    console.print("[cyan]Waiting for Jupyter to start...[/cyan]")

            time.sleep(5)

        else:
            if not restart_kernels:
                console.print("[yellow] Jupyter may have failed to start, you may need to set the restart flag to True")

        console.print(f"[cyan]Setting up port forward to localhost:{local_port}...[/cyan]")
        cmd = [
            "kubectl",
            "port-forward",
            f"pod/{pod_name}",
            f"{local_port}:8888",
            "--namespace",
            namespace,
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)

        from kubetorch.provisioning.utils import wait_for_port_forward

        try:
            wait_for_port_forward(
                process,
                local_port,
                health_endpoint=None,
                validate_kubetorch_versions=False,
            )
            time.sleep(2)
        except Exception as e:
            console.print(f"[red]Failed to establish port forward: {e}[/red]")
            if process:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    process.wait()
                except (ProcessLookupError, OSError):
                    pass
            raise typer.Exit(1)

        # Open in browser
        notebook_url = f"http://localhost:{local_port}"
        console.print(f"\n[green]✓ Notebook is ready on URL: {notebook_url}[/green]")
        console.print(
            f"[yellow]Service '{remote_fn.service_name}' will stay alive after exit; reconnecting will restart "
            f"all kernel sessions[/yellow]"
        )
        console.print(f"\n[dim]To tear down: kt teardown {remote_fn.service_name}[/dim]")
        console.print("[dim]Press Ctrl+C to stop port forwarding[/dim]\n")
        if not os.getenv("KT_NO_BROWSER"):
            webbrowser.open(notebook_url)

        # Keep running
        try:
            while True:
                if process.poll() is not None:
                    console.print("\n[yellow]Port forward process terminated[/yellow]")
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping port forward...[/yellow]")
        finally:
            # Clean up port forward process only
            if process:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    process.wait()
                except (ProcessLookupError, OSError):
                    pass

            console.print(
                f"\n[yellow]Service '{remote_fn.service_name}' is still running in namespace '{namespace}'[/yellow]"
            )
            console.print(f"[dim]To tear down: kt teardown {remote_fn.service_name}[/dim]")

    except Exception as e:
        console.print(f"[red]Error setting up notebook: {e}[/red]")
        raise typer.Exit(1)


@app.command("logs")
def kt_logs(
    name: str = service_name_argument(help="Service name"),
    follow: bool = typer.Option(False, "-f", "--follow", help="Follow the logs"),
    tail: int = typer.Option(None, "-t", "--tail", help=f"Number of lines to show (default: {DEFAULT_TAIL_LENGTH})"),
    pod: str = typer.Option(
        None,
        "-p",
        "--pod",
        help="Name or index of a specific pod to load logs from (0-based)",
    ),
    namespace: str = typer.Option(
        globals.config.namespace,
        "-n",
        "--namespace",
    ),
):
    """Load logs for a Kubetorch service.

    Examples:

    .. code-block:: bash

        $ kt logs my-service         # logs from all pods

        $ kt logs my-service -p 1    # logs only from a particular pod index

        $ kt logs my-service -f      # follow logs

        $ kt logs my-service -t 50   # tail last 50 lines
    """

    console.print(f"Looking for service [blue]{name}[/blue]...")

    # Validate service exists and get deployment mode
    name, deployment_mode = get_deployment_mode(name, namespace)

    try:
        # Get pods using the correct label selector for the deployment mode
        pods = validate_pods_exist(name, namespace)
        sorted_by_time = sorted(pods, key=lambda p: p.get("metadata", {}).get("creationTimestamp", "9999"))

        if pod:
            # specific pod is requested
            selected_pod = load_selected_pod(service_name=name, provided_pod=pod, service_pods=sorted_by_time)
        else:
            # display logs from all pods by default
            selected_pod = None
    except typer.Exit as e:
        raise e
    except Exception as e:
        console.print(f"[red]Failed to load pods for service {name} in namespace {namespace}: {str(e)}[/red]")
        raise typer.Exit(1)

    try:
        # Only show pod name when there are multiple pods and no specific pod was selected
        print_pod_name: bool = not selected_pod and len(pods) > 1

        try:
            if follow:
                console.print("[dim]Press Ctrl+C to quit[/dim]\n")
                follow_logs_in_cli(name, namespace, selected_pod, deployment_mode, print_pod_name)
            else:
                query = generate_logs_query(name, namespace, selected_pod, deployment_mode)
                if not query:
                    return

                logs = load_logs_for_pod(query=query, print_pod_name=print_pod_name, timeout=5.0, namespace=namespace)
                if logs is None:
                    console.print("[red]No logs found for service[/red]")
                    return

                tail_length = tail if tail else DEFAULT_TAIL_LENGTH
                if len(logs) > tail_length:
                    logs = logs[-tail_length:]

                for log_line in logs:
                    print(log_line.rstrip("\n"))

        except KeyboardInterrupt:
            return

    except Exception as e:
        console.print(f"[red]Failed to load logs for service {name} in namespace {namespace}[/red]\n\n {str(e)}")
        raise typer.Exit(1)


@app.command("put")
def kt_put(
    key: str = typer.Argument(..., help="Storage key (e.g., 'my-service/models', 'datasets/train')"),
    src: List[str] = typer.Option(..., "--src", "-s", help="Local file(s) or directory(s) to upload"),
    force: bool = typer.Option(False, "--force", "-f", help="Force overwrite of existing files"),
    exclude: str = typer.Option(None, "--exclude", help="Exclude patterns (rsync format, e.g., '*.pyc')"),
    include: str = typer.Option(
        None, "--include", help="Include patterns (rsync format, e.g., '*.pkl') to override .gitignore exclusions"
    ),
    contents: bool = typer.Option(
        False,
        "--contents",
        "-c",
        help="Copy directory contents (adds trailing slashes for rsync 'copy contents' behavior)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed progress"),
    namespace: str = typer.Option(globals.config.namespace, "-n", "--namespace", help="Kubernetes namespace"),
):
    """Store files or directories in the cluster using a key-value interface"""
    from kubetorch.data_store import put

    try:
        # Build filter options if exclude or include is provided
        filter_options = None
        if exclude and include:
            # If both are provided, include must come before exclude in rsync
            filter_options = f"--include='{include}' --exclude='{exclude}'"
        elif include:
            filter_options = f"--include='{include}'"
        elif exclude:
            filter_options = f"--exclude='{exclude}'"

        # Handle multiple sources
        src_list = list(src) if len(src) > 1 else src[0]

        put(
            key=key,
            src=src_list,
            contents=contents,
            filter_options=filter_options,
            force=force,
            verbose=verbose,
            namespace=namespace,
        )

        if not verbose:
            console.print(f"[green]✓[/green] Stored at key '{key}'")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("get")
def kt_get(
    key: str = typer.Argument(..., help="Storage key to retrieve (e.g., 'my-service/models', 'datasets/train')"),
    dest: str = typer.Option(
        None,
        "--dest",
        "-d",
        help="Local destination path where files will be downloaded (defaults to current working directory)",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force overwrite of existing files"),
    exclude: str = typer.Option(None, "--exclude", help="Exclude patterns (rsync format, e.g., '*.pyc')"),
    include: str = typer.Option(
        None, "--include", help="Include patterns (rsync format, e.g., '*.pkl') to override .gitignore exclusions"
    ),
    contents: bool = typer.Option(
        False,
        "--contents",
        "-c",
        help="Copy directory contents (adds trailing slashes for rsync 'copy contents' behavior)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed progress"),
    namespace: str = typer.Option(globals.config.namespace, "-n", "--namespace", help="Kubernetes namespace"),
):
    """Retrieve files or directories from the cluster using a key-value interface"""
    from kubetorch.data_store import get

    try:
        # Build filter options if exclude or include is provided
        filter_options = None
        if exclude and include:
            # If both are provided, include must come before exclude in rsync
            filter_options = f"--include='{include}' --exclude='{exclude}'"
        elif include:
            filter_options = f"--include='{include}'"
        elif exclude:
            filter_options = f"--exclude='{exclude}'"

        # Default to current working directory if dest not specified
        if dest is None:
            import os

            dest = os.getcwd()

        get(
            key=key,
            dest=dest,
            contents=contents,
            filter_options=filter_options,
            force=force,
            verbose=verbose,
            namespace=namespace,
        )

        if not verbose:
            console.print(f"[green]✓[/green] Retrieved key '{key}'")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("ls")
def kt_ls(
    key: str = typer.Argument(
        "", help="Storage key path to list (e.g., 'my-service/models', 'datasets'). Empty for root."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed progress"),
    namespace: str = typer.Option(globals.config.namespace, "-n", "--namespace", help="Kubernetes namespace"),
):
    """List files and directories in the cluster store"""
    from kubetorch.data_store import ls

    try:
        # List the contents
        items = ls(key=key, verbose=verbose, namespace=namespace)

        if not items:
            if key:
                console.print(f"[yellow]No items found under key '{key}'[/yellow]")
            else:
                console.print("[yellow]Store is empty[/yellow]")
        else:
            # Display the items
            if key:
                console.print(f"\n[bold]Contents of '{key}':[/bold]")
            else:
                console.print("\n[bold]Contents of store root:[/bold]")

            # Separate directories and files
            dirs = [item for item in items if item.get("is_directory", False)]
            files = [item for item in items if not item.get("is_directory", False)]

            # Display directories first
            for dir_item in sorted(dirs, key=lambda x: x["name"].lower()):
                dir_name = dir_item["name"]
                if not dir_name.endswith("/"):
                    dir_name += "/"

                locale = dir_item.get("locale", "store")
                if locale != "store":
                    locale_info = f" [dim](locale: {locale})[/dim]"
                    console.print(f"  📁 [blue]{dir_name}[/blue]{locale_info}")
                else:
                    console.print(f"  📁 [blue]{dir_name}[/blue]")

            # Display files
            for file_item in sorted(files, key=lambda x: x["name"].lower()):
                file_name = file_item["name"]

                locale = file_item.get("locale", "store")
                if locale != "store":
                    locale_info = f" [dim](locale: {locale})[/dim]"
                    console.print(f"  📄 {file_name}{locale_info}")
                else:
                    console.print(f"  📄 {file_name}")

            local_count = sum(1 for item in items if item.get("locale", "store") != "store")
            console.print(f"\n[green]Total: {len(dirs)} directories, {len(files)} files[/green]")
            if local_count > 0:
                console.print(f"[dim]  ({local_count} item(s) stored locally on pods)[/dim]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("rm")
def kt_rm(
    key: str = typer.Argument(..., help="Storage key to delete (e.g., 'my-service/models', 'datasets/train.csv')"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Delete directories recursively"),
    prefix: bool = typer.Option(False, "--prefix", "-p", help="Delete all keys starting with this string prefix"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed progress"),
    namespace: str = typer.Option(globals.config.namespace, "-n", "--namespace", help="Kubernetes namespace"),
):
    """Delete files or directories from the cluster store"""
    from kubetorch.data_store import rm

    try:
        rm(key=key, recursive=recursive, prefix=prefix, verbose=verbose, namespace=namespace)

        if not verbose:
            if prefix:
                console.print(f"[green]✓[/green] Deleted all keys with prefix '{key}'")
            else:
                console.print(f"[green]✓[/green] Deleted key '{key}'")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("pool", hidden=True)
def kt_pool(
    namespace: str = typer.Option(
        globals.config.namespace,
        "-n",
        "--namespace",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed module, specifier, labels, and annotations",
    ),
    connections: bool = typer.Option(
        False,
        "--connections",
        "-c",
        help="Show connected pods via WebSocket",
    ),
):
    """
    List registered pools.
    """
    import kubetorch as kt

    controller = kt.globals.controller_client()

    # Show WebSocket connection debug info if requested
    if connections:
        connection_data = controller.get_connections()
        if not connection_data:
            console.print("[yellow]No pools found[/yellow]")
            raise typer.Exit()

        table = Table(
            show_header=True,
            border_style="bright_black",
            expand=True,
        )
        table.add_column("Pool Name", style="bold magenta", no_wrap=True)
        table.add_column("Namespace", style="cyan", no_wrap=True)
        table.add_column("Pods", style="green", no_wrap=True)
        table.add_column("Pod IPs", style="yellow", overflow="fold")

        for pool_name, info in connection_data.items():
            connected_pods = info.get("connected_pods", [])
            pod_count = info.get("pod_count", len(connected_pods))
            ips = [p["ip"] for p in connected_pods if p.get("ip")]
            table.add_row(
                pool_name,
                info.get("namespace", "-"),
                str(pod_count),
                ", ".join(ips) if ips else "-",
            )

        console.print(table)
        raise typer.Exit()

    def fmt(ts):
        if not ts:
            return "-"
        try:
            return datetime.fromisoformat(ts).strftime("%m-%d %H:%M")
        except Exception:
            return ts

    resp = controller.list_pools(namespace=namespace)
    pools = resp.get("pools", [])

    if not pools:
        console.print(f"[yellow]No pools found in {namespace} namespace[/yellow]")
        raise typer.Exit()

    table = Table(
        show_header=True,
        border_style="bright_black",
    )

    table.add_column("Pool Name", style="bold magenta", no_wrap=True)
    table.add_column("User", style="green", no_wrap=True)
    table.add_column("Resource", style="cyan", no_wrap=True)
    table.add_column("Last Deployed", style="white", no_wrap=True)

    for p in pools:
        metadata = p.get("pool_metadata") or {}
        user = metadata.get("username", "-")
        resource_kind = p.get("resource_kind") or "-"

        table.add_row(
            p.get("name", "-"),
            user,
            resource_kind,
            fmt(p.get("last_deployed_at", "-")),
        )

    console.print(table)

    # Show detailed JSON for each pool only in verbose mode
    if verbose:
        console.print()
        console.rule("[bold]Pool Details[/bold]", style="bright_black")
        for p in pools:
            pool_name = p.get("name", "-")
            module = p.get("module") or {}
            specifier = p.get("specifier", {})
            labels = p.get("labels") or {}
            annotations = p.get("annotations") or {}

            console.print()
            console.print(
                Panel.fit(
                    f"[bold magenta]{pool_name}[/bold magenta]",
                    border_style="magenta",
                )
            )
            if module:
                console.print("  [white]Module:[/white]")
                console.print(Syntax(json.dumps(module, indent=2), "json", theme="monokai", line_numbers=False))
            if specifier:
                console.print("  [white]Specifier:[/white]")
                console.print(Syntax(json.dumps(specifier, indent=2), "json", theme="monokai", line_numbers=False))
            if labels:
                console.print("  [white]Labels:[/white]")
                console.print(Syntax(json.dumps(labels, indent=2), "json", theme="monokai", line_numbers=False))
            if annotations:
                console.print("  [white]Annotations:[/white]")
                console.print(Syntax(json.dumps(annotations, indent=2), "json", theme="monokai", line_numbers=False))


@server_app.command("start", hidden=True)
def kt_server_start(
    port: int = typer.Option(
        int(os.getenv("KT_SERVER_PORT", 32300)),
        "--port",
        "-p",
        help="Port to run the HTTP server on",
    ),
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        "-h",
        help="Host to bind the HTTP server to",
    ),
    pool_name: str = typer.Option(
        os.getenv("KT_SERVICE"),
        "--pool",
        "-n",
        help="Pool/service name for WebSocket registration with controller",
    ),
    controller_url: str = typer.Option(
        os.getenv("KT_CONTROLLER_URL"),
        "--controller-url",
        "-u",
        help="Controller URL",
    ),
):
    """Start the Kubetorch server.

    Used in BYO-compute deployments where the server must be launched inside
    a user-provided pod. Handles remote execution and optional self-registration
    with the Kubetorch controller.

    Examples:

    .. code-block:: bash

        $ kubetorch server start --pool my-workers --controller-url http://kubetorch-controller:8080

        $ export KT_SERVICE=my-workers

        $ export KT_CONTROLLER_URL=http://kubetorch-controller.kubetorch.svc.cluster.local:8080

        $ kubetorch server start
    """
    try:
        import uvicorn
    except ImportError:
        console.print(r'[red]uvicorn is not installed. Install with: `pip install "kubetorch\[server]"`[/red]')
        raise typer.Exit(1)

    if not is_running_in_kubernetes():
        console.print(
            "[yellow]`kubetorch server start` is typically used inside Kubernetes pods. "
            "It's not recommended to run this command directly on your local machine.[/yellow]"
        )

    if pool_name:
        os.environ["KT_SERVICE"] = pool_name

    if controller_url:
        os.environ["KT_CONTROLLER_URL"] = controller_url

    elif pool_name and not os.getenv("KT_CONTROLLER_URL"):
        # Default controller URL if not provided but pool name is set
        namespace = os.getenv("POD_NAMESPACE", "kubetorch")
        default_url = f"http://kubetorch-controller.{namespace}.svc.cluster.local:8080"
        os.environ["KT_CONTROLLER_URL"] = default_url
        console.print(f"[dim]Using default controller URL: {default_url}[/dim]")

    console.print(f"[green]Starting kubetorch HTTP server on {host}:{port}[/green]")

    from kubetorch.serving.http_server import app as http_app

    uvicorn.run(http_app, host=host, port=port)


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
