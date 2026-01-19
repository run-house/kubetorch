import asyncio
import base64
import contextvars
import importlib
import importlib.util
import inspect
import json
import logging.config
import os
import pickle
import subprocess
import sys
import threading
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Dict, Optional, Union

try:
    import httpx
except:
    pass

from fastapi import Body, FastAPI, Header, HTTPException, Query, Request

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

try:
    from global_http_clients import close_clients
    from log_capture import init_log_capture, stop_log_capture
    from metrics_push import init_metrics_pusher, stop_metrics_pusher
    from server_metrics import get_inactivity_ttl_annotation
    from utils import (
        clear_debugging_sessions,
        deep_breakpoint,
        DEFAULT_ALLOWED_SERIALIZATION,
        ensure_structured_logging,
        is_running_in_kubernetes,
        LOG_CONFIG,
        request_id_ctx_var,
        wait_for_app_start,
    )
except ImportError:
    from .global_http_clients import close_clients
    from .log_capture import init_log_capture, stop_log_capture
    from .metrics_push import init_metrics_pusher, stop_metrics_pusher
    from .server_metrics import get_inactivity_ttl_annotation
    from .utils import (
        clear_debugging_sessions,
        deep_breakpoint,
        DEFAULT_ALLOWED_SERIALIZATION,
        ensure_structured_logging,
        is_running_in_kubernetes,
        LOG_CONFIG,
        request_id_ctx_var,
        wait_for_app_start,
    )

from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import StreamingResponse

logging.config.dictConfig(LOG_CONFIG)

# Set up our structured JSON logging
ensure_structured_logging()

logger = logging.getLogger(__name__)
# Set log level based on environment variable
kt_log_level = os.getenv("KT_LOG_LEVEL")
if kt_log_level:
    kt_log_level = kt_log_level.upper()
    logger.setLevel(getattr(logging, kt_log_level, logging.INFO))

_CACHED_CALLABLES = {}
_CACHED_IMAGE = []
SUPERVISOR = None
APP_PROCESS = None
_CALLABLE_LOAD_LOCK = threading.Lock()  # Lock for thread-safe callable loading

# Log streaming and metrics collection - enabled by default in Kubernetes
KT_LOG_STREAMING_ENABLED = os.environ.get("KT_LOG_STREAMING_ENABLED", "True").lower() == "true"
KT_METRICS_ENABLED = os.environ.get("KT_METRICS_ENABLED", "True").lower() == "true"

# Global termination event that can be checked by running requests
TERMINATION_EVENT = threading.Event()

# Controller WebSocket - for receiving metadata from controller
_METADATA_RECEIVED = threading.Event()
_CONTROLLER_WS = None  # Will hold the ControllerWebSocket instance

# Create a client for FastAPI service

# Set the python breakpoint to kt.deep_breakpoint
os.environ["PYTHONBREAKPOINT"] = "kubetorch.deep_breakpoint"

request_id_ctx_var.set(os.getenv("KT_LAUNCH_ID", "-"))

#####################################
########### Proxy Helpers ###########
#####################################
if os.getenv("KT_CALLABLE_TYPE") == "app" and os.getenv("KT_APP_PORT"):
    port = os.getenv("KT_APP_PORT")
    logger.info(f"Creating /http reverse proxy to: http://localhost:{port}/")
    proxy_client = httpx.AsyncClient(base_url=f"http://localhost:{port}/", timeout=None)
else:
    proxy_client = None


async def _http_reverse_proxy(request: Request):
    """Reverse proxy for /http/* routes to FastAPI service on its port"""
    # Extract the endpoint name from the path
    # request.path_params["path"] will contain everything after /http/
    endpoint_path = request.path_params["path"]

    # Build the URL for the FastAPI service
    url = httpx.URL(path=f"/{endpoint_path}", query=request.url.query.encode("utf-8"))

    # Build the request to forward to FastAPI
    rp_req = proxy_client.build_request(request.method, url, headers=request.headers.raw, content=await request.body())

    # Send the request and get streaming response
    rp_resp = await proxy_client.send(rp_req, stream=True)

    # Return streaming response
    return StreamingResponse(
        rp_resp.aiter_raw(),
        status_code=rp_resp.status_code,
        headers=rp_resp.headers,
        background=BackgroundTask(rp_resp.aclose),
    )


##########################################
########### Controller WebSocket #########
##########################################


def _get_pod_name() -> str:
    """Get pod name - from hostname (K8s sets hostname to pod name)."""
    # Try env var first for backwards compatibility
    if os.environ.get("POD_NAME"):
        return os.environ["POD_NAME"]
    # In K8s, hostname is set to pod name
    import socket

    return socket.gethostname()


def _get_pod_namespace() -> str:
    """Get pod namespace - from service account mount or env var."""
    # Try env var first for backwards compatibility
    if os.environ.get("POD_NAMESPACE"):
        return os.environ["POD_NAMESPACE"]
    # K8s mounts namespace in service account directory
    namespace_file = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
    try:
        with open(namespace_file) as f:
            return f.read().strip()
    except (FileNotFoundError, PermissionError):
        return "default"


def _get_pod_ip() -> str:
    """Get pod IP - from resolving hostname or env var."""
    # Try env var first for backwards compatibility
    if os.environ.get("POD_IP"):
        return os.environ["POD_IP"]
    # Resolve hostname to get pod IP
    import socket

    try:
        return socket.gethostbyname(socket.gethostname())
    except socket.gaierror:
        return ""


def _populate_pod_env_vars():
    """Populate POD_NAME, POD_NAMESPACE, POD_IP env vars at startup.

    These were previously set via Kubernetes Downward API in the pod template.
    Now we derive them locally and set them as env vars so they're available
    throughout the codebase and in user code.
    """
    if not os.environ.get("POD_NAME"):
        os.environ["POD_NAME"] = _get_pod_name()
    if not os.environ.get("POD_NAMESPACE"):
        os.environ["POD_NAMESPACE"] = _get_pod_namespace()
    if not os.environ.get("POD_IP"):
        pod_ip = _get_pod_ip()
        if pod_ip:
            os.environ["POD_IP"] = pod_ip


# Populate pod env vars at module load time
_populate_pod_env_vars()


class ControllerWebSocket:
    """WebSocket client for receiving metadata from the kubetorch controller.

    On startup, pods connect to the controller via WebSocket to:
    1. Register themselves (pod_name, namespace, service_name)
    2. Request their metadata (module pointers, init_args, etc.)
    3. Receive updates when /pool is called (redeployments)

    This replaces the static env var approach where all metadata was baked
    into the pod manifest at creation time. It also replaces the push-based
    approach where the controller connected to pods - this pull-based approach
    scales better and supports autoscaling pods requesting metadata on connect.
    """

    def __init__(self):
        self._ws = None
        self._running = False
        self._task = None
        self._reconnect_delay = 1.0  # Start with 1 second, exponential backoff

    def _get_controller_url(self) -> Optional[str]:
        """Get the WebSocket URL for the controller."""
        controller_url = os.environ.get("KT_CONTROLLER_URL")
        if not controller_url:
            # Fall back to constructing from install namespace
            install_namespace = os.environ.get("KT_INSTALL_NAMESPACE", "kubetorch")
            controller_url = f"http://kubetorch-controller.{install_namespace}.svc.cluster.local:8080"

        # Convert HTTP to WS
        ws_url = controller_url.replace("http://", "ws://").replace("https://", "wss://")
        return f"{ws_url}/controller/ws/pods"

    def _get_registration_message(self) -> dict:
        """Build the registration message to send to controller."""
        return {
            "action": "register",
            "pod_name": _get_pod_name(),
            "pod_ip": _get_pod_ip(),
            "namespace": _get_pod_namespace(),
            "service_name": os.environ.get("KT_SERVICE", ""),
            "request_metadata": True,  # Always request metadata on connect
        }

    def _apply_metadata(self, metadata: dict):
        """Apply received metadata by setting environment variables.

        This maintains backwards compatibility with existing code that reads
        from env vars. The supervisor and load_callable will use these env vars.
        """
        global _METADATA_RECEIVED

        module_info = metadata.get("module", {})
        service_name = metadata.get("service_name")
        namespace = metadata.get("namespace")

        # Set module pointer env vars
        if module_info.get("module_name"):
            os.environ["KT_MODULE_NAME"] = module_info["module_name"]
        if module_info.get("cls_or_fn_name"):
            os.environ["KT_CLS_OR_FN_NAME"] = module_info["cls_or_fn_name"]
        if module_info.get("file_path"):
            os.environ["KT_FILE_PATH"] = module_info["file_path"]
        if module_info.get("project_root"):
            os.environ["KT_PROJECT_ROOT"] = module_info["project_root"]
        if module_info.get("callable_type"):
            os.environ["KT_CALLABLE_TYPE"] = module_info["callable_type"]

        # Set init args (always set, default to "null" if not provided)
        init_args = module_info.get("init_args")
        os.environ["KT_INIT_ARGS"] = json.dumps(init_args) if init_args else "null"

        # Set distributed config (always set, default to "null" if not provided)
        distributed_config = module_info.get("distributed_config")
        os.environ["KT_DISTRIBUTED_CONFIG"] = json.dumps(distributed_config) if distributed_config else "null"

        # Set service metadata
        if service_name:
            os.environ["KT_SERVICE_NAME"] = service_name
            os.environ["KT_SERVICE"] = service_name
            # Update LogCapture labels for log streaming
            try:
                from log_capture import get_log_capture
            except ImportError:
                from .log_capture import get_log_capture
            log_capture = get_log_capture()
            if log_capture and log_capture.labels.get("service") == "unknown":
                log_capture.labels["service"] = service_name

        if namespace:
            os.environ["POD_NAMESPACE"] = namespace
            try:
                from log_capture import get_log_capture
            except ImportError:
                from .log_capture import get_log_capture
            log_capture = get_log_capture()
            if log_capture:
                log_capture.labels["namespace"] = namespace

        if metadata.get("service_dns"):
            os.environ["KT_SERVICE_DNS"] = metadata["service_dns"]
        if metadata.get("deployment_mode"):
            os.environ["KT_DEPLOYMENT_MODE"] = metadata["deployment_mode"]
        if metadata.get("username"):
            os.environ["KT_USERNAME"] = metadata["username"]

        # Apply runtime config - these can change between deploys
        runtime_config = metadata.get("runtime_config", {})
        if runtime_config.get("log_streaming_enabled") is not None:
            os.environ["KT_LOG_STREAMING_ENABLED"] = str(runtime_config["log_streaming_enabled"])
        if runtime_config.get("metrics_enabled") is not None:
            os.environ["KT_METRICS_ENABLED"] = str(runtime_config["metrics_enabled"])
        if runtime_config.get("inactivity_ttl"):
            os.environ["KT_INACTIVITY_TTL"] = runtime_config["inactivity_ttl"]
        if runtime_config.get("log_level"):
            os.environ["KT_LOG_LEVEL"] = runtime_config["log_level"]
        if runtime_config.get("allowed_serialization"):
            os.environ["KT_ALLOWED_SERIALIZATION"] = runtime_config["allowed_serialization"]

        logger.info(
            f"Applied metadata from controller: module={module_info.get('module_name')}, "
            f"callable={module_info.get('cls_or_fn_name')}"
        )

        # Signal that metadata has been received
        _METADATA_RECEIVED.set()

    async def _handle_reload(self, metadata: dict):
        """Handle a reload message from controller.

        This is called when /pool is called and the controller pushes
        updated metadata to all pods. Sends an acknowledgment back to
        the controller so it can wait for all pods to process the reload.

        Blocking operations (run_image_setup, load_callable) are run in a thread pool
        to avoid blocking the event loop and dropping the WebSocket connection.
        """
        global SUPERVISOR, _CACHED_CALLABLES

        logger.info("Received reload message from controller")

        try:
            # Apply the new metadata (sets env vars - fast, ok to run in event loop)
            self._apply_metadata(metadata)

            # Run image setup for the reload - use thread pool to avoid blocking event loop
            await asyncio.to_thread(run_image_setup)

            # Clear caches
            _CACHED_CALLABLES.clear()

            # Cleanup existing supervisor
            if SUPERVISOR:
                try:
                    SUPERVISOR.cleanup()
                except Exception as e:
                    logger.warning(f"Error during supervisor cleanup on reload: {e}")
                SUPERVISOR = None

            # Recreate supervisor in thread pool so it's ready for the next request
            # This prevents race conditions where the request arrives before the supervisor is ready
            if os.environ.get("KT_CLS_OR_FN_NAME"):
                logger.info("Recreating supervisor during reload")
                clear_cache()
                await asyncio.to_thread(load_callable)
                logger.info("Supervisor recreated successfully")

            # Send acknowledgment to controller
            if self._ws:
                await self._ws.send(json.dumps({"action": "reload_ack", "status": "ok"}))
                logger.debug("Sent reload acknowledgment to controller")

        except Exception as e:
            logger.error(f"Error handling reload: {e}")
            # Send error acknowledgment
            if self._ws:
                await self._ws.send(json.dumps({"action": "reload_ack", "status": "error", "message": str(e)}))

    async def _run(self):
        """Main WebSocket connection loop with automatic reconnection."""
        try:
            import websockets
        except ImportError:
            logger.warning("websockets package not installed - controller WebSocket disabled")
            _METADATA_RECEIVED.set()
            return

        ws_url = self._get_controller_url()
        if not ws_url:
            logger.warning("No controller URL configured - metadata must be in env vars")
            _METADATA_RECEIVED.set()
            return

        logger.info(f"Connecting to controller WebSocket: {ws_url}")

        while self._running:
            try:
                async with websockets.connect(ws_url, close_timeout=10) as ws:
                    self._ws = ws
                    self._reconnect_delay = 1.0  # Reset backoff on successful connect

                    # Send registration message
                    reg_msg = self._get_registration_message()
                    await ws.send(json.dumps(reg_msg))
                    logger.info(f"Registered with controller as {reg_msg['pod_name']}")

                    # Listen for messages
                    async for message in ws:
                        try:
                            data = json.loads(message)
                            action = data.get("action")

                            if action == "metadata":
                                # Initial metadata response
                                self._apply_metadata(data)
                            elif action == "reload":
                                # Reload triggered by /pool call
                                await self._handle_reload(data)
                            elif action == "error":
                                logger.error(f"Controller error: {data.get('message')}")
                            else:
                                logger.debug(f"Unknown message action: {action}")

                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON from controller: {e}")

            except Exception as e:
                if self._running:
                    logger.warning(
                        f"Controller WebSocket connection failed: {e}. " f"Reconnecting in {self._reconnect_delay}s..."
                    )
                    await asyncio.sleep(self._reconnect_delay)
                    # Exponential backoff with max of 30 seconds
                    self._reconnect_delay = min(self._reconnect_delay * 2, 30.0)

        self._ws = None

    async def start(self):
        """Start the WebSocket connection in background."""
        if not is_running_in_kubernetes():
            # Not in K8s, metadata comes from env vars
            logger.debug("Not running in Kubernetes - skipping controller WebSocket")
            _METADATA_RECEIVED.set()
            return

        # Check if module env vars are already set (backwards compatibility)
        if os.environ.get("KT_MODULE_NAME") and os.environ.get("KT_CLS_OR_FN_NAME"):
            logger.debug("Module env vars already set - skipping controller WebSocket")
            _METADATA_RECEIVED.set()
            return

        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        """Stop the WebSocket connection."""
        self._running = False
        if self._ws:
            await self._ws.close()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


#####################################
########### Cache Helpers ###########
#####################################
def clear_cache():
    global _CACHED_CALLABLES

    logger.debug("Clearing callables cache.")
    _CACHED_CALLABLES.clear()


def cached_image_setup():
    logger.debug("Starting cached image setup.")
    global _CACHED_IMAGE
    global APP_PROCESS

    dockerfile_path = kt_directory() / "image.dockerfile"
    with open(dockerfile_path, "r") as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]

    # find first line where image differs from cache and update cache
    cache_mismatch_index = -1
    cmd_mismatch = False
    for i, (new_line, cached_line) in enumerate(zip(lines, _CACHED_IMAGE)):
        if new_line.startswith("CMD"):
            cmd_mismatch = True

        if new_line != cached_line or "# override" in new_line or cmd_mismatch:
            cache_mismatch_index = i
            break
    if cache_mismatch_index == -1:
        if len(lines) != len(_CACHED_IMAGE):
            cache_mismatch_index = min(len(lines), len(_CACHED_IMAGE))
        else:
            cache_mismatch_index = len(lines)
    _CACHED_IMAGE = lines

    if cache_mismatch_index == len(lines):
        return

    if not (cache_mismatch_index == len(lines) - 1 and cmd_mismatch):
        logger.debug("Running image setup.")
    else:
        logger.debug("Skipping image setup steps, no changes detected.")

    # Grab the current list of installed dependencies with pip freeze to check if anything changes (we need to send a
    # SIGHUP to restart the server if so)
    start_deps = None

    try:
        res = subprocess.run(
            ["pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
        start_deps = res.stdout.splitlines()
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run pip freeze: {e}")

    # only run image setup steps starting from cache mismatch point
    kt_pip_cmd = None
    for line in lines[cache_mismatch_index:]:
        command = ""
        if line.strip().startswith("#"):
            continue  # ignore comments
        if line.startswith("RUN") or line.startswith("CMD"):
            command = line[len("RUN ") :]

            if command.startswith("$KT_PIP_INSTALL_CMD"):
                kt_pip_cmd = kt_pip_cmd or _get_kt_pip_install_cmd() or "pip install"
                command = command.replace("$KT_PIP_INSTALL_CMD", kt_pip_cmd)
        elif line.startswith("COPY"):
            _, source, dest = line.split()
            # COPY instructions are essentially no-ops since rsync_file_updates()
            # already placed files in their correct locations.
            # But we verify the files exist and log the absolute paths for clarity.

            # Determine the actual absolute destination path
            if dest and dest.startswith("/"):
                # Already absolute
                dest_path = Path(dest)
            elif dest and dest.startswith("~/"):
                # Tilde prefix - strip it and treat as relative to cwd
                dest_path = Path.cwd() / dest[2:]
            else:
                # Relative to working directory (including explicit basenames)
                dest_path = Path.cwd() / dest

            # Verify the destination exists (it should have been rsync'd)
            if dest_path.exists():
                logger.info(f"Copied {source} to {dest_path.absolute()}")
            else:
                raise FileNotFoundError(
                    f"COPY {source} {dest} failed: destination {dest_path.absolute()} does not exist. "
                    f"This likely means the rsync operation failed to sync the files correctly."
                )
        elif line.startswith("ENV"):
            # Need to handle the case where the env var is being set to "" (empty string)
            line_vals = line.split(" ", 2)
            if len(line_vals) < 2:  # ENV line must have at least key
                raise ValueError("ENV line cannot be empty")
            if len(line_vals) == 2:  # ENV line with just key
                key = line_vals[1]
                val = ""
            elif len(line_vals) == 3:  # ENV line with key and value
                key, val = line_vals[1], line_vals[2]

            # Expand environment variables in the value
            # This supports patterns like $VAR, ${VAR}, and $VAR:default_value
            expanded_val = os.path.expandvars(val)

            if key not in [
                "KT_FILE_PATH",
                "KT_MODULE_NAME",
                "KT_CLS_OR_FN_NAME",
                "KT_INIT_ARGS",
                "KT_CALLABLE_TYPE",
                "KT_DISTRIBUTED_CONFIG",
            ]:
                logger.info(f"Setting env var {key}")
            os.environ[key] = expanded_val
            # If the env var is specifically KT_LOG_LEVEL, we need to update the logger level
            if key == "KT_LOG_LEVEL":
                global kt_log_level
                kt_log_level = expanded_val.upper()
                logger.setLevel(kt_log_level)
                logger.info(f"Updated log level to {kt_log_level}")
        elif line.startswith("FROM"):
            continue
        elif line:
            raise ValueError(f"Unrecognized image setup instruction {line}")

        if command:
            is_app_cmd = line.startswith("CMD")
            if is_app_cmd:
                logger.info(f"Running app command: {command}")
            else:
                logger.info(f"Running: {command}")

            try:
                # Use subprocess.Popen to capture output and redirect through StreamToLogger
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"

                if is_app_cmd and os.getenv("KT_CALLABLE_TYPE") == "app":
                    if APP_PROCESS and APP_PROCESS.poll() is None:
                        APP_PROCESS.kill()

                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    bufsize=1,
                    env=env,
                )

                if is_app_cmd and os.getenv("KT_CALLABLE_TYPE") == "app":
                    APP_PROCESS = process

                # Collect stderr for potential error logging
                import threading

                stderr_lines = []
                stderr_lock = threading.Lock()

                # Stream stdout and stderr in real-time
                # We need to do all this so the stdout and stderr are prints with the correct formatting
                # for our queries. Without it they just flow straight to system stdout and stderr without any

                def stream_output(pipe, log_func, request_id, collect_stderr=False):
                    request_id_ctx_var.set(request_id)
                    for line in iter(pipe.readline, ""):
                        if line:
                            stripped_line = line.rstrip()
                            log_func(stripped_line)

                            # Collect stderr lines for potential error logging
                            if collect_stderr:
                                with stderr_lock:
                                    stderr_lines.append(stripped_line.lstrip("ERROR: "))
                    pipe.close()

                # Start streaming threads
                current_request_id = request_id_ctx_var.get("-")

                stderr_log_func = logger.error if is_app_cmd else logger.debug
                stdout_thread = threading.Thread(
                    target=stream_output,
                    args=(process.stdout, logger.info, current_request_id),
                )
                stderr_thread = threading.Thread(
                    target=stream_output,
                    args=(
                        process.stderr,
                        stderr_log_func,
                        current_request_id,
                        not is_app_cmd,
                    ),
                )

                stdout_thread.daemon = True
                stderr_thread.daemon = True
                stdout_thread.start()
                stderr_thread.start()

                if is_app_cmd:
                    # App commands run indefinitely - never block waiting for completion
                    if os.getenv("KT_APP_PORT"):
                        # Wait for internal app to be healthy/ready on specified port
                        try:
                            port = os.getenv("KT_APP_PORT")
                            logger.debug(f"Waiting for internal app on port {port} to start:")
                            wait_for_app_start(
                                port=port,
                                health_check=os.getenv("KT_APP_HEALTHCHECK"),
                                process=process,
                            )
                            logger.info(f"App on port {port} is ready.")
                        except Exception as e:
                            logger.error(f"Caught exception waiting for app to start: {e}")
                    else:
                        # No port specified - app runs in background, just check for immediate failures
                        time.sleep(0.5)
                        poll_result = process.poll()
                        if poll_result is not None and poll_result != 0:
                            stdout_thread.join(timeout=1)
                            stderr_thread.join(timeout=1)
                            with stderr_lock:
                                if stderr_lines:
                                    logger.error("App command failed immediately:")
                                    for stderr_line in stderr_lines:
                                        logger.error(stderr_line)
                        else:
                            logger.info(f"App command running in background (PID: {process.pid})")
                elif command.rstrip().endswith("&"):
                    # Background command (ends with &) - don't wait for completion
                    time.sleep(0.5)
                    poll_result = process.poll()
                    if poll_result is not None and poll_result != 0:
                        stdout_thread.join(timeout=1)
                        stderr_thread.join(timeout=1)
                        with stderr_lock:
                            if stderr_lines:
                                logger.error("Background command failed immediately:")
                                for stderr_line in stderr_lines:
                                    logger.error(stderr_line)
                    else:
                        logger.info(f"Background process started successfully (PID: {process.pid})")
                else:
                    # Regular RUN command - wait for completion
                    return_code = process.wait()
                    stdout_thread.join()
                    stderr_thread.join()

                    if return_code != 0:
                        with stderr_lock:
                            if stderr_lines:
                                logger.error(f"Failed to run command '{command}' with stderr:")
                                for stderr_line in stderr_lines:
                                    logger.error(stderr_line)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to run command '{command}' with error: {e}")
                with stderr_lock:
                    if stderr_lines:
                        logger.error("Stderr:")
                        for stderr_line in stderr_lines:
                            logger.error(stderr_line)
    # Check if any dependencies changed and if so reload them inside the server process
    if start_deps:
        try:
            # Run pip freeze and capture the output
            res = subprocess.run(
                ["pip", "freeze"],
                capture_output=True,
                text=True,
                check=True,
            )
            end_deps = res.stdout.splitlines()
            # We only need to look at the deps which were already installed (i.e. lines in start_deps),
            # new ones can't be "stale" inside the current server process
            # We also only use lines with exact pypi versions (has "=="), no editable
            changed_deps = [line.split("==")[0] for line in start_deps if "==" in line and line not in end_deps]
            imported_changed_deps = [
                dep for dep in changed_deps if dep in sys.modules
            ]  # Only reload deps which are already imported
            if imported_changed_deps:
                logger.debug(f"New dependencies found: {imported_changed_deps}, forcing reload")

                # Don't clear the callable cache here - let load_callable_from_env handle it to preserve __kt_cached_state__
                if SUPERVISOR:
                    SUPERVISOR.cleanup()

                # Remove changed modules from sys.modules to override fresh imports
                modules_to_remove = []
                for module_name in sys.modules:
                    for dep in imported_changed_deps:
                        if module_name == dep or module_name.startswith(dep + "."):
                            modules_to_remove.append(module_name)
                            break

                for module_name in modules_to_remove:
                    try:
                        del sys.modules[module_name]
                        logger.debug(f"Removed module {module_name} from sys.modules")
                    except KeyError:
                        pass
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to run pip freeze: {e}")


def run_image_setup():
    """Run image setup (rsync files, run dockerfile instructions).

    With push-based reloads, files are rsynced before the pool is registered,
    so the dockerfile should already be present when this is called.
    """
    if os.environ.get("KT_FREEZE", "False") == "True" or not is_running_in_kubernetes():
        return

    rsync_file_updates()

    dockerfile_path = kt_directory() / "image.dockerfile"
    if not dockerfile_path.exists():
        # BYO image case - no dockerfile to process because user built their own image.
        # Code has been rsynced above, callable will be reloaded on next request via load_callable().
        logger.info("No dockerfile found, skipping cached image setup (BYO image mode)")
        return

    cached_image_setup()

    if not os.getenv("KT_CALLABLE_TYPE") == "app":
        logger.debug("Completed cached image setup.")


#####################################
######## Generic Helpers ############
#####################################
class SerializationError(Exception):
    pass


def kt_directory():
    if "KT_DIRECTORY" in os.environ:
        return Path(os.environ["KT_DIRECTORY"]).expanduser()
    else:
        return Path.cwd() / ".kt"


def _get_kt_pip_install_cmd() -> Optional[str]:
    """Get the actual KT_PIP_INSTALL_CMD value for command expansion."""
    kt_pip_cmd = os.getenv("KT_PIP_INSTALL_CMD")
    if not kt_pip_cmd:  # Fallback to reading from file
        try:
            with open(kt_directory() / "kt_pip_install_cmd", "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            return None
    return kt_pip_cmd


def is_running_in_container():
    # Check for .dockerenv file which exists in Docker containers
    return Path("/.dockerenv").exists()


async def run_in_executor_with_context(executor, func, *args, **kwargs):
    """
    Helper to run a function in an executor while preserving context variables.

    Uses contextvars.copy_context() to copy all context variables (including request_id)
    to the executor thread. This ensures log capture and other context-dependent code
    works correctly in thread pool threads.
    """
    ctx = contextvars.copy_context()
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, lambda: ctx.run(func, *args, **kwargs))


def load_callable(
    distributed_subprocess: bool = False,
    reload_cleanup_fn: [Callable, None] = None,
):
    """Load the callable from environment.

    This function is called:
    1. From _handle_reload() when a reload is pushed via WebSocket
    2. From subprocesses to load the callable in the worker

    With push-based reloads, this is always called fresh on reload - no need to
    check timestamps since the reload is triggered externally.
    """
    callable_name = os.environ["KT_CLS_OR_FN_NAME"]

    callable_obj = _CACHED_CALLABLES.get(callable_name, None)
    if callable_obj:
        # Return cached callable for subprocess calls
        if distributed_subprocess:
            logger.debug("Returning cached callable.")
            return callable_obj

    # Slow path: need to load or reload - use lock for thread safety
    with _CALLABLE_LOAD_LOCK:
        # Double-check within lock (another thread might have loaded it)
        callable_obj = _CACHED_CALLABLES.get(callable_name, None)
        if callable_obj and distributed_subprocess:
            logger.debug("Returning cached callable (found after acquiring lock).")
            return callable_obj
        # Proceed with loading/reloading
        return _load_callable_internal(distributed_subprocess, reload_cleanup_fn, callable_obj)


def _load_callable_internal(
    distributed_subprocess: bool = False,
    reload_cleanup_fn: [Callable, None] = None,
    callable_obj=None,
):
    """Internal callable loading logic - should be called within lock for thread safety."""
    callable_name = os.environ["KT_CLS_OR_FN_NAME"]

    if not callable_obj:
        logger.debug("Callable not found in cache, loading from environment.")
    else:
        logger.debug("Reloading callable.")

    # If a reload cleanup function is provided, call it before reloading
    if reload_cleanup_fn:
        reload_cleanup_fn()

    if not distributed_subprocess:
        # We don't reload the image in distributed subprocess/es, as we already did it in the
        # main process and we don't want to do it multiple times (in each subprocess).
        logger.info("Running image setup and loading callable.")
        run_image_setup()

    distributed_config = os.environ.get("KT_DISTRIBUTED_CONFIG", "null")
    deployment_mode = os.environ.get("KT_DEPLOYMENT_MODE", "deployment")

    # For RayCluster deployments, we need to start Ray even if there's no explicit distributed config.
    # This ensures workers can connect to the head node's GCS.
    if distributed_config in ["null", "None"] and deployment_mode == "raycluster" and not distributed_subprocess:
        logger.info("RayCluster deployment detected without distributed config, starting Ray for worker connectivity")
        distributed_config = json.dumps({"distribution_type": "ray"})
        os.environ["KT_DISTRIBUTED_CONFIG"] = distributed_config

    # Default to local supervisor for non-distributed mode (subprocess isolation)
    # This provides clean module isolation and simple reload semantics (terminate and recreate subprocess)
    if distributed_config in ["null", "None"] and not distributed_subprocess:
        logger.debug("Using local supervisor for subprocess isolation")
        distributed_config = json.dumps({"distribution_type": "local"})
        os.environ["KT_DISTRIBUTED_CONFIG"] = distributed_config

    if distributed_config not in ["null", "None"] and not distributed_subprocess:
        logger.debug(f"Loading supervisor: {distributed_config}")
        callable_obj = load_supervisor()
        logger.debug("Supervisor loaded successfully.")
    else:
        # Only called from subprocesses now
        logger.debug(f"Loading callable from environment: {callable_name}")
        callable_obj = load_callable_from_env()
        logger.debug("Callable loaded successfully.")

    _CACHED_CALLABLES[callable_name] = callable_obj

    return callable_obj


def load_supervisor():
    global SUPERVISOR

    if os.environ["KT_FILE_PATH"] not in sys.path:
        sys.path.insert(0, os.environ["KT_FILE_PATH"])

    distributed_config = os.environ["KT_DISTRIBUTED_CONFIG"]

    # If this is the main process of a distributed call, we don't load the callable directly,
    # we create a new supervisor if it doesn't exist or if the config has changed.
    # We don't create a supervisor if this is a distributed subprocess.
    config_hash = hash(str(distributed_config))
    if SUPERVISOR is None or config_hash != SUPERVISOR.config_hash:
        from kubetorch.serving.supervisor_factory import supervisor_factory

        logger.debug(f"Loading distributed supervisor with config: {distributed_config}")
        distributed_config = json.loads(distributed_config)
        # If we already have some distributed processes, we need to clean them up before creating a new supervisor.
        if SUPERVISOR:
            SUPERVISOR.cleanup()
        SUPERVISOR = supervisor_factory(**distributed_config)
        SUPERVISOR.config_hash = config_hash
    try:
        # If there are any errors during setup, we catch and log them, and then undo the setup
        # so that the distributed supervisor is not left in a broken state (and otherwise can still fail
        # when we call SUPERVISOR.cleanup() in lifespan).
        SUPERVISOR.setup()
    except Exception as e:
        logger.error(f"Failed to set up distributed supervisor with config {distributed_config}: {e}")
        SUPERVISOR = None
        raise e
    return SUPERVISOR


def patch_sys_path():
    abs_path = str(Path(os.environ["KT_FILE_PATH"]).expanduser().resolve())
    if os.environ["KT_FILE_PATH"] not in sys.path:
        sys.path.insert(0, abs_path)
        logger.debug(f"Added {abs_path} to sys.path")

    # Add project root to sys.path to enable imports from sibling directories
    # This allows scripts in subdirectories (e.g., experimental/script.py) to import
    # from sibling packages at the project root (e.g., from utils.module import func)
    project_root = os.environ.get("KT_PROJECT_ROOT")
    abs_project_root = str(Path(project_root).expanduser().resolve()) if project_root else None
    if abs_project_root and abs_project_root not in sys.path:
        sys.path.insert(0, abs_project_root)
        logger.debug(f"Added project root {abs_project_root} to sys.path")

    # Maybe needed for subprocesses (e.g. distributed) to find the callable's module
    # Needed for distributed subprocesses to find the file path
    existing_path = os.environ.get("PYTHONPATH", "")
    existing_path_list = existing_path.split(os.pathsep) if existing_path else []
    paths_to_add = []

    if abs_path not in existing_path_list:
        paths_to_add.append(abs_path)
    if abs_project_root and abs_project_root not in existing_path_list:
        paths_to_add.append(abs_project_root)

    if paths_to_add:
        new_paths = os.pathsep.join(paths_to_add)
        os.environ["PYTHONPATH"] = f"{new_paths}{os.pathsep}{existing_path}" if existing_path else new_paths
        logger.debug(f"Set PYTHONPATH to {os.environ['PYTHONPATH']}")


def load_callable_from_env():
    """Load callable from environment variables.

    This function is called from subprocesses (via ProcessWorker) to load the user's
    callable. Since subprocesses are terminated and recreated on redeployment,
    we don't need complex module reload logic - just a fresh import.
    """
    cls_or_fn_name = os.environ["KT_CLS_OR_FN_NAME"]
    module_name = os.environ["KT_MODULE_NAME"]

    patch_sys_path()

    # Load the module
    try:
        logger.debug(f"Importing module {module_name}")
        module = importlib.import_module(module_name)
        logger.debug(f"Module {module_name} loaded")

        # Ensure our structured logging is in place after user module import
        # (in case the user's module configured its own logging via dictConfig)
        ensure_structured_logging()

        callable_obj = getattr(module, cls_or_fn_name)
        logger.debug(f"Callable {cls_or_fn_name} loaded")
    except (ImportError, ValueError) as original_error:
        # Fall back to file-based import if package import fails
        try:
            module = import_from_file(os.environ["KT_FILE_PATH"], module_name)
            # Ensure structured logging after file-based import
            ensure_structured_logging()
            callable_obj = getattr(module, cls_or_fn_name)
        except (ImportError, ValueError):
            # Raise the original error if file import also fails, because the errors which are raised here are
            # more opaque and less useful than the original ImportError or ValueError.
            raise original_error
    except AttributeError as e:
        # If the callable is not found in the module, raise an error
        raise HTTPException(
            status_code=404,
            detail=f"Callable '{cls_or_fn_name}' not found in module '{module_name}'",
        ) from e

    # Unwrap to remove any kt deploy decorators (e.g. @kt.compute)
    if hasattr(callable_obj, "__wrapped__"):
        callable_obj = callable_obj.__wrapped__

    if isinstance(callable_obj, type):
        # Prepare init arguments
        init_kwargs = {}

        # Add user-provided init_args
        if os.environ["KT_INIT_ARGS"] not in ["null", "None"]:
            init_kwargs = json.loads(os.environ["KT_INIT_ARGS"])
            logger.info(f"Setting init_args {init_kwargs}")

        # Instantiate with arguments
        if init_kwargs:
            callable_obj = callable_obj(**init_kwargs)
        else:
            callable_obj = callable_obj()

    return callable_obj


def import_from_file(file_path: str, module_name: str):
    """Import a module from file path."""
    module_parts = module_name.split(".")
    depth = max(0, len(module_parts) - 1)

    # Convert file_path to absolute path if it's not already (note, .resolve will append the current working directory
    # if file_path is relative)
    abs_path = Path(file_path).expanduser().resolve()
    # Ensure depth doesn't exceed available parent directories
    max_available_depth = len(abs_path.parents) - 1

    if max_available_depth < 0:
        # File has no parent directories, use the file's directory itself
        parent_path = str(abs_path.parent)
    else:
        # Clamp depth to available range to avoid IndexError
        depth = min(depth, max_available_depth)
        parent_path = str(abs_path.parents[depth])

    if parent_path not in sys.path:
        sys.path.insert(0, parent_path)

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for module {module_name} from {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#####################################
########## Rsync Helpers ############
#####################################
def rsync_file_updates():
    """Sync files from rsync pod into the server pod using centralized data_store helper."""
    from kubetorch import data_store as _dt

    service_name = os.getenv("KT_SERVICE_NAME")
    namespace = os.getenv("POD_NAMESPACE")

    if not service_name or not namespace:
        logger.warning(f"Skipping rsync - KT_SERVICE_NAME={service_name}, POD_NAMESPACE={namespace}")
        return

    logger.info(f"Starting rsync for service {service_name} in namespace {namespace}")
    _dt._sync_workdir_from_store(namespace=namespace, service_name=service_name)


#####################################
########### App setup ###############
#####################################
class HealthCheckFilter(logging.Filter):
    def filter(self, record):
        return not (
            isinstance(record.args, tuple)
            and len(record.args) >= 3
            and ("/health" in record.args[2] or record.args[2] == "/")
        )


class ScrapeMetricsLogsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "/metrics" not in msg


class RequestContextFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_ctx_var.get("-")
        record.pod = os.getenv("POD_NAME", "unknown-pod")
        return True


class TerminationCheckMiddleware(BaseHTTPMiddleware):
    """Monitor for termination while request is running and return error if detected."""

    async def dispatch(self, request: Request, call_next):
        # Skip health checks and metrics endpoints
        if request.url.path in ["/health", "/", "/metrics"]:
            return await call_next(request)

        # Run the actual request in the background

        request_task = asyncio.create_task(call_next(request))

        # Monitor for termination while request is running
        while not request_task.done():
            # Check if we're terminating
            if TERMINATION_EVENT.is_set() or (
                hasattr(request.app.state, "terminating") and request.app.state.terminating
            ):
                # Cancel the request task
                request_task.cancel()

                # Return PodTerminatedError
                from kubetorch import PodTerminatedError
                from kubetorch.serving.http_server import package_exception

                pod_name = os.environ.get("POD_NAME", "unknown")
                exc = PodTerminatedError(
                    pod_name=pod_name,
                    reason="SIGTERM",
                    status_code=503,
                    events=[
                        {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "reason": "Terminating",
                            "message": "Pod received SIGTERM signal and is shutting down gracefully",
                        }
                    ],
                )

                return package_exception(exc)

            # Wait a bit before checking again or for request to complete
            try:
                result = await asyncio.wait_for(asyncio.shield(request_task), timeout=0.5)
                return result
            except asyncio.TimeoutError:
                # Request still running after 0.5s, continue loop to check termination again
                continue

        # Request completed normally
        return await request_task


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", "-")
        token = request_id_ctx_var.set(request_id)

        try:
            response = await call_next(request)
            return response
        finally:
            # Reset the context variable to its default value
            request_id_ctx_var.reset(token)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    import signal
    import threading

    # Initialize log capture early (before other logging happens)
    # This captures ALL stdout/stderr from the pod and pushes to log store
    if KT_LOG_STREAMING_ENABLED:
        log_capture = init_log_capture()
        if log_capture:
            logger.info("Log streaming enabled")
            app.state.log_capture = log_capture

    # Startup - TTL tracking via metrics
    ttl = get_inactivity_ttl_annotation()
    if ttl and KT_METRICS_ENABLED:
        logger.info(f"TTL={ttl}s enabled with metrics tracking")
    elif ttl:
        logger.warning("TTL annotation found but metrics collection is disabled - TTL will not work")
    else:
        logger.debug("No TTL annotation found")

    # Initialize metrics collection
    if KT_METRICS_ENABLED:
        metrics_pusher = init_metrics_pusher(ttl_seconds=ttl)
        if metrics_pusher:
            logger.info("Metrics collection enabled")
            app.state.metrics_pusher = metrics_pusher

    # Only register signal handlers if we're in the main thread
    # This allows tests to run without signal handling
    if threading.current_thread() is threading.main_thread():
        # Save any existing SIGTERM handler
        original_sigterm_handler = signal.getsignal(signal.SIGTERM)

        def handle_sigterm(signum, frame):
            """Handle SIGTERM for graceful shutdown."""
            logger.info("Received SIGTERM, initiating graceful shutdown...")

            # Mark that we're terminating and interrupt existing requests IMMEDIATELY
            app.state.terminating = True
            TERMINATION_EVENT.set()

            # Clean up distributed supervisor to ensure child processes are terminated
            # This is important because SIGTERM is not propagated to child processes automatically
            # This runs synchronously and may take 1-2 seconds, but existing requests are already interrupted
            global SUPERVISOR
            if SUPERVISOR:
                logger.info("Cleaning up distributed supervisor and child processes...")
                try:
                    SUPERVISOR.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up distributed supervisor: {e}")

            # Call the original handler if it exists and isn't the default
            if original_sigterm_handler and original_sigterm_handler not in (
                signal.SIG_DFL,
                signal.SIG_IGN,
            ):
                original_sigterm_handler(signum, frame)

        # Register SIGTERM handler
        signal.signal(signal.SIGTERM, handle_sigterm)
    app.state.terminating = False

    # Start controller WebSocket to receive metadata (if not already set via env vars)
    global _CONTROLLER_WS
    _CONTROLLER_WS = ControllerWebSocket()
    await _CONTROLLER_WS.start()

    # Wait for metadata to be received (with timeout)
    # This blocks startup until we have the module info needed to load the callable
    # Use asyncio-compatible wait to avoid blocking the event loop
    metadata_timeout = 30  # seconds
    start_time = time.time()
    while not _METADATA_RECEIVED.is_set():
        if time.time() - start_time > metadata_timeout:
            logger.warning(f"Timeout waiting for metadata from controller after {metadata_timeout}s")
            break
        await asyncio.sleep(0.1)  # Yield to event loop so WebSocket task can run

    try:
        if os.getenv("KT_CALLABLE_TYPE") == "app":
            run_image_setup()
        elif os.getenv("KT_CLS_OR_FN_NAME"):
            # Only load callable if one is configured (not in selector-only mode)
            load_callable()
        else:
            # Selector-only mode: server starts without a callable, waiting for deployment
            logger.info("Starting in selector-only mode (no callable configured)")

        logger.info("Kubetorch Server started.")

        # Flush logs immediately so launch logs are pushed before request_id is reset
        log_capture = getattr(app.state, "log_capture", None)
        if log_capture:
            log_capture.flush()

        request_id_ctx_var.set("-")  # Reset request_id after launch sequence
        yield

    except Exception:
        # We don't want to raise errors like ImportError during startup, as it will cause the server to crash and the
        # user won't be able to see the error in the logs to debug (e.g. quickly add dependencies or reorganize
        # imports). Instead, we log it (and a stack trace) and continue, so it will be surfaced to the user when they
        # call the service.

        # However if this service is frozen, it should just fail because the user isn't debugging the service and there is no
        # way for the dependencies to be added at runtime.
        logger.error(traceback.format_exc())

        # Flush logs immediately so launch logs are pushed before request_id is reset
        log_capture = getattr(app.state, "log_capture", None)
        if log_capture:
            log_capture.flush()

        request_id_ctx_var.set("-")
        yield

    finally:
        # Shutdown - stop controller WebSocket
        if _CONTROLLER_WS:
            await _CONTROLLER_WS.stop()
            logger.info("Controller WebSocket stopped")

        # Shutdown - stop log capture and metrics collection
        log_capture = getattr(app.state, "log_capture", None)
        if log_capture:
            stop_log_capture()
            logger.info("Log streaming stopped")

        metrics_pusher = getattr(app.state, "metrics_pusher", None)
        if metrics_pusher:
            stop_metrics_pusher()
            logger.info("Metrics collection stopped")

        # Clean up during normal shutdown so we don't leave any hanging processes, which can cause pods to hang
        # indefinitely. Skip if already cleaned up by SIGTERM handler.
        if SUPERVISOR and not getattr(app.state, "terminating", False):
            SUPERVISOR.cleanup()

        # Clear any remaining debugging sessions
        clear_debugging_sessions()

        # Close global HTTP clients
        close_clients()

        # Close proxy client if it exists
        if proxy_client is not None:
            await proxy_client.aclose()


# Add the filter to uvicorn's access logger
logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())
logging.getLogger("uvicorn.access").addFilter(ScrapeMetricsLogsFilter())
root_logger = logging.getLogger()
root_logger.addFilter(RequestContextFilter())
for handler in root_logger.handlers:
    handler.addFilter(RequestContextFilter())

app = FastAPI(lifespan=lifespan)
app.add_middleware(TerminationCheckMiddleware)  # Check termination first
app.add_middleware(RequestIDMiddleware)


# Add metrics tracking middleware
@app.middleware("http")
async def track_requests_metrics(request: Request, call_next):
    """Middleware to track active requests and record metrics."""
    metrics_pusher = getattr(request.app.state, "metrics_pusher", None)
    if metrics_pusher and request.url.path not in ["/metrics", "/health", "/"]:
        metrics_pusher.request_started()
        start_time = time.time()
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            metrics_pusher.record_request(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code,
                duration=duration,
            )
            return response
        except Exception:
            duration = time.time() - start_time
            metrics_pusher.record_request(
                method=request.method,
                endpoint=request.url.path,
                status=500,
                duration=duration,
            )
            raise
        finally:
            metrics_pusher.request_finished()
    else:
        return await call_next(request)


# add route for fastapi app
if os.getenv("KT_CALLABLE_TYPE") == "app" and os.getenv("KT_APP_PORT"):
    logger.debug("Adding route for path /http")
    app.add_route(
        "/http/{path:path}",
        _http_reverse_proxy,
        ["GET", "POST", "PUT", "DELETE", "PATCH"],
    )


#####################################
########## Error Handling ###########
#####################################
class ErrorResponse(BaseModel):
    error_type: str
    message: str
    traceback: str
    pod_name: str = "unknown"  # Default for pods deployed outside kubetorch
    state: Optional[dict] = None  # Optional serialized exception state


# Factor out the exception packaging so we can use it in the handler below and also inside distributed subprocesses
def package_exception(exc: Exception):
    import concurrent

    error_type = exc.__class__.__name__
    trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    # Check if the exception has a status_code attribute (e.g. PodTerminatedError)
    if hasattr(exc, "status_code"):
        status_code = exc.status_code
    elif isinstance(exc, (RequestValidationError, TypeError, AssertionError)):
        status_code = 422
    elif isinstance(exc, (ValueError, UnicodeError, json.JSONDecodeError)):
        status_code = 400
    elif isinstance(exc, (KeyError, FileNotFoundError)):
        status_code = 404
    elif isinstance(exc, PermissionError):
        status_code = 403
    elif isinstance(exc, (StarletteHTTPException, HTTPException)):
        status_code = exc.status_code
    elif isinstance(exc, (MemoryError, OSError)):
        status_code = 500
    elif isinstance(exc, NotImplementedError):
        status_code = 501
    elif isinstance(exc, asyncio.TimeoutError):
        status_code = 504
    elif isinstance(exc, concurrent.futures.TimeoutError):
        status_code = 504
    else:
        status_code = 500

    # Try to serialize exception state if it has __getstate__
    state = None
    if hasattr(exc, "__getstate__"):
        try:
            state = exc.__getstate__()
            json.dumps(state)
        except Exception as e:
            logger.debug(f"Could not serialize exception state for {error_type}: {e}")
            state = None

    error_response = ErrorResponse(
        error_type=error_type,
        message=str(exc),
        traceback=trace,
        pod_name=os.getenv("POD_NAME", "unknown"),
        state=state,
    )

    return JSONResponse(status_code=status_code, content=error_response.model_dump())


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return package_exception(exc)


def _apply_metadata_from_dict(metadata: dict):
    """Apply metadata from a dict (used by test endpoint).

    Similar to ControllerWebSocket._apply_metadata() but takes a dict directly.
    """
    module = metadata.get("module", {})
    runtime_config = metadata.get("runtime_config", {})

    # Set module env vars
    if module.get("module_name"):
        os.environ["KT_MODULE_NAME"] = module["module_name"]
    if module.get("cls_or_fn_name"):
        os.environ["KT_CLS_OR_FN_NAME"] = module["cls_or_fn_name"]
    if module.get("file_path"):
        os.environ["KT_FILE_PATH"] = module["file_path"]

    init_args = module.get("init_args")
    if init_args is not None:
        os.environ["KT_INIT_ARGS"] = json.dumps(init_args) if init_args else "None"
    else:
        os.environ["KT_INIT_ARGS"] = "None"

    if module.get("callable_type"):
        os.environ["KT_CALLABLE_TYPE"] = module["callable_type"]

    distributed_config = module.get("distributed_config")
    if distributed_config is not None:
        os.environ["KT_DISTRIBUTED_CONFIG"] = json.dumps(distributed_config) if distributed_config else "None"
    else:
        os.environ["KT_DISTRIBUTED_CONFIG"] = "None"

    # Set runtime config env vars
    if runtime_config.get("log_streaming_enabled") is not None:
        os.environ["KT_LOG_STREAMING_ENABLED"] = str(runtime_config["log_streaming_enabled"]).lower()
    if runtime_config.get("metrics_enabled") is not None:
        os.environ["KT_METRICS_ENABLED"] = str(runtime_config["metrics_enabled"]).lower()

    # Set other metadata
    if metadata.get("service_name"):
        os.environ["KT_SERVICE_NAME"] = metadata["service_name"]
    if metadata.get("namespace"):
        os.environ["POD_NAMESPACE"] = metadata["namespace"]
    if metadata.get("deployment_mode"):
        os.environ["KT_DEPLOYMENT_MODE"] = metadata["deployment_mode"]
    if metadata.get("username"):
        os.environ["KT_USERNAME"] = metadata["username"]


@app.post("/_test_reload", include_in_schema=False)
async def test_reload(request: Request, metadata: Dict = Body(...)):
    """Test endpoint to trigger reload with new metadata.

    This endpoint simulates the WebSocket push-based reload for testing purposes.
    It applies metadata, runs image setup, and recreates the supervisor.

    Example metadata:
    {
        "module": {
            "module_name": "my_module",
            "cls_or_fn_name": "my_function",
            "file_path": "/path/to/module",
            "init_args": null,
            "callable_type": "fn"
        },
        "runtime_config": {}
    }
    """
    global SUPERVISOR, _CACHED_CALLABLES

    try:
        # Apply the new metadata (sets env vars)
        _apply_metadata_from_dict(metadata)

        # Run image setup - use thread pool to avoid blocking event loop
        await asyncio.to_thread(run_image_setup)

        # Clear caches
        _CACHED_CALLABLES.clear()

        # Cleanup existing supervisor
        if SUPERVISOR:
            try:
                SUPERVISOR.cleanup()
            except Exception as e:
                logger.warning(f"Error during supervisor cleanup on test reload: {e}")
            SUPERVISOR = None

        # Recreate supervisor
        if os.environ.get("KT_CLS_OR_FN_NAME"):
            logger.info("Recreating supervisor during test reload")
            clear_cache()
            await asyncio.to_thread(load_callable)
            logger.info("Supervisor recreated successfully")

        return {"status": "ok", "message": "Reload completed successfully"}

    except Exception as e:
        logger.error(f"Error in test reload: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


# Health and readiness endpoints (note: must be defined before catch-all routes)
@app.get("/health", include_in_schema=False)
@app.get("/", include_in_schema=False)
async def health():
    return {"status": "healthy"}


@app.get("/metrics", include_in_schema=False)
async def metrics(request: Request):
    """Expose Prometheus-formatted metrics for scraping."""
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    except ImportError:
        return Response(content="# prometheus_client not installed\n", media_type="text/plain")

    metrics_pusher = getattr(request.app.state, "metrics_pusher", None)
    if metrics_pusher and hasattr(metrics_pusher, "registry"):
        # Use MetricsPusher's registry
        content = generate_latest(metrics_pusher.registry).decode("utf-8")
    else:
        # Fall back to default registry
        content = generate_latest().decode("utf-8")

    return Response(content=content, media_type=CONTENT_TYPE_LATEST)


@app.get("/ready", include_in_schema=False)
async def ready():
    """Readiness check - returns 200 only when callable is loaded and ready to serve."""
    callable_name = os.getenv("KT_CLS_OR_FN_NAME")
    if not callable_name:
        raise HTTPException(
            status_code=503,
            detail="Callable not loaded yet",
        )
    return {"status": "ready", "callable": callable_name}


@app.get("/app/status", include_in_schema=False)
async def app_status():
    """Check the status of the app process (for kt run).

    Returns:
        - {"running": True, "pid": int} if app process is running
        - {"running": False, "exit_code": int} if app process has exited
        - {"running": null} if this is not an app deployment
    """
    if APP_PROCESS is None:
        return {"running": None}

    exit_code = APP_PROCESS.poll()
    if exit_code is None:
        return {"running": True, "pid": APP_PROCESS.pid}
    else:
        return {"running": False, "exit_code": exit_code}


# Catch-all routes for callable invocation - must be defined AFTER specific routes
@app.post("/{cls_or_fn_name}", response_class=JSONResponse)
@app.post("/{cls_or_fn_name}/{method_name}", response_class=JSONResponse)
def run_callable(
    request: Request,
    cls_or_fn_name: str,
    method_name: Optional[str] = None,
    distributed_subcall: bool = Query(False),
    params: Optional[Union[Dict, str]] = Body(default=None),
    serialization: str = Header("json", alias="X-Serialization"),
):
    """Execute a callable through the supervisor.

    All calls are routed through the supervisor which executes user code in
    isolated subprocesses. This provides clean module isolation and simple
    reload semantics (terminate and recreate subprocess on redeployment).

    This is a sync endpoint - Starlette automatically runs it in a thread pool,
    which handles context variable propagation.
    """
    configured_callable = os.getenv("KT_CLS_OR_FN_NAME")
    if not configured_callable:
        raise HTTPException(
            status_code=503,
            detail="Server is starting up or no callable has been deployed yet. Please ensure the function is deployed before calling.",
        )
    if cls_or_fn_name != configured_callable:
        raise HTTPException(
            status_code=404,
            detail=f"Callable '{cls_or_fn_name}' not found in metadata configuration. Found '{configured_callable}' instead",
        )
    if SUPERVISOR is None:
        raise HTTPException(
            status_code=503,
            detail="Server is loading the callable. Please retry in a moment.",
        )

    # Route call through supervisor to subprocess
    result = SUPERVISOR.call(
        request,
        cls_or_fn_name,
        method_name,
        params,
        distributed_subcall,
    )
    clear_debugging_sessions()
    return result


def _parse_callable_params(
    callable_obj: Callable,
    cls_or_fn_name: str,
    method_name: Optional[str],
    params: Optional[Union[Dict, str]],
    serialization: str,
):
    """Parse and validate callable parameters. Returns (user_method, args, kwargs, debug_port, debug_mode, is_async)."""
    # Check if serialization is allowed
    allowed_serialization = os.getenv("KT_ALLOWED_SERIALIZATION", DEFAULT_ALLOWED_SERIALIZATION).split(",")
    if serialization not in allowed_serialization:
        raise HTTPException(
            status_code=400,
            detail=f"Serialization format '{serialization}' not allowed. Allowed formats: {allowed_serialization}",
        )

    # Process the call
    args = []
    kwargs = {}
    debug_port, debug_mode = None, None

    if params:
        if serialization == "pickle":
            # Handle pickle serialization - extract data from dictionary wrapper
            if isinstance(params, dict) and "data" in params:
                encoded_data = params.pop("data")
                pickled_data = base64.b64decode(encoded_data.encode("utf-8"))
                param_args = pickle.loads(pickled_data)
                # data is unpickled in the format {"args": args, "kwargs": kwargs}
                params.update(param_args)
            elif isinstance(params, str):
                # Fallback for direct string
                pickled_data = base64.b64decode(params.encode("utf-8"))
                params = pickle.loads(pickled_data)

        # Default JSON handling
        args = params.get("args", [])
        kwargs = params.get("kwargs", {})
        debugger: dict = params.get("debugger", None) if params else None
        if debugger:
            debug_mode = debugger.get("mode")
            debug_port = debugger.get("port")

    if method_name:
        if not hasattr(callable_obj, method_name):
            raise HTTPException(
                status_code=404,
                detail=f"Method '{method_name}' not found in class '{cls_or_fn_name}'",
            )
        user_method = getattr(callable_obj, method_name)
    else:
        user_method = callable_obj

    is_async_method = inspect.iscoroutinefunction(user_method)
    return user_method, args, kwargs, debug_port, debug_mode, is_async_method


def _serialize_result(result, serialization: str):
    """Serialize the result based on the format."""
    if serialization == "pickle":
        try:
            pickled_result = pickle.dumps(result)
            encoded_result = base64.b64encode(pickled_result).decode("utf-8")
            return {"data": encoded_result}
        except Exception as e:
            logger.error(f"Failed to pickle result: {str(e)}")
            raise SerializationError(f"Result could not be serialized with pickle: {str(e)}")
    else:
        # Default JSON serialization
        try:
            json.dumps(result)
        except (TypeError, ValueError) as e:
            logger.error(f"Result is not JSON serializable: {str(e)}")
            raise SerializationError(f"Result could not be serialized to JSON: {str(e)}")
    return result


async def execute_callable_async(
    callable_obj: Callable,
    cls_or_fn_name: str,
    method_name: Optional[str] = None,
    params: Optional[Union[Dict, str]] = None,
    serialization: str = "json",
    executor=None,
):
    """Execute a callable asynchronously. Used by ProcessWorker's event loop.

    Matches FastAPI's concurrency model:
    - Async callables: awaited directly on the event loop (true async concurrency)
    - Sync callables: run in thread pool via run_in_executor() (doesn't block event loop)

    This allows async callables to benefit from cooperative multitasking - many can
    run concurrently on a single thread. Sync callables are offloaded to threads.

    Args:
        executor: ThreadPoolExecutor for running sync callables. If None, uses default executor.
    """
    user_method, args, kwargs, debug_port, debug_mode, is_async = _parse_callable_params(
        callable_obj, cls_or_fn_name, method_name, params, serialization
    )

    callable_name = f"{cls_or_fn_name}.{method_name}" if method_name else cls_or_fn_name
    if debug_port:
        logger.info(f"Debugging remote callable {callable_name} on port {debug_port}")
        deep_breakpoint(debug_port, debug_mode)

    if is_async:
        # Async callable: await directly on the event loop
        # This enables true async concurrency - many async calls can run concurrently
        logger.debug(f"Calling async callable {callable_name}")
        result = await user_method(*args, **kwargs)
    else:
        # Sync callable: run in thread pool to avoid blocking the event loop
        # This matches FastAPI's behavior for sync route handlers
        logger.debug(f"Calling sync callable {callable_name} in thread pool")
        result = await run_in_executor_with_context(executor, user_method, *args, **kwargs)

    # Handle case where method returns an awaitable
    if isinstance(result, Awaitable):
        result = await result

    result = _serialize_result(result, serialization)
    clear_debugging_sessions()
    return result


#####################################
######## GPU Transfer Support #######
#####################################
# Storage for pending GPU tensors that have been published via gpu_put
_GPU_PENDING_TENSORS = {}


@app.post("/_gpu/publish")
async def gpu_publish(
    request: Request,
    key: str = Body(...),
    nccl_port: int = Body(29500),
):
    """
    Internal endpoint called after gpu_put to register the tensor.
    The actual tensor is stored in the callable's memory.
    """
    # This is called internally by gpu_put - tensor is already stored in GPUTransferClient
    return {"success": True, "key": key}


@app.post("/_gpu/serve_broadcast")
async def gpu_serve_broadcast(
    request: Request,
    key: str = Body(...),
    broadcast_id: str = Body(...),
    world_size: int = Body(...),
):
    """
    Serve a GPU tensor broadcast as the source pod (rank 0).

    Called when the quorum is ready and all participants are waiting for the broadcast.
    """
    try:
        from kubetorch.data_store.gpu_transfer import _get_gpu_client

        client = _get_gpu_client()
        await run_in_executor_with_context(
            None,
            client.serve_broadcast,
            key,
            broadcast_id,
            world_size,
            True,  # verbose
        )
        return {"success": True, "key": key, "broadcast_id": broadcast_id}
    except Exception as e:
        logger.error(f"Failed to serve GPU broadcast for key '{key}': {e}")
        return {"success": False, "error": str(e)}


@app.get("/_gpu/pending")
async def gpu_pending_keys():
    """List keys with pending GPU tensors for broadcast (debugging endpoint)."""
    try:
        from kubetorch.data_store.gpu_transfer import _get_gpu_client

        client = _get_gpu_client()
        pending = getattr(client, "_pending_broadcasts", {})
        return {
            "keys": list(pending.keys()),
            "details": {
                k: {"shape": list(v["tensor"].shape), "dtype": str(v["tensor"].dtype)} for k, v in pending.items()
            },
        }
    except Exception as e:
        return {"keys": [], "error": str(e)}


if __name__ == "__main__" and not is_running_in_container():
    # NOTE: this will only run in local development, otherwise we start the uvicorn server in the pod template setup
    import uvicorn
    from dotenv import load_dotenv

    load_dotenv()

    logger.info("Starting HTTP server")
    uvicorn.run(app, host="0.0.0.0", port=os.environ.get("KT_SERVER_PORT", 32300))
