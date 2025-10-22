import base64
import importlib
import importlib.util
import inspect
import json
import logging.config
import os
import pickle
import random
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

from fastapi import Body, FastAPI, Header, HTTPException, Request

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

try:
    from server_metrics import (
        get_inactivity_ttl_annotation,
        HeartbeatManager,
        setup_otel_metrics,
    )
    from utils import (
        clear_debugging_sessions,
        deep_breakpoint,
        DEFAULT_ALLOWED_SERIALIZATION,
        ensure_structured_logging,
        is_running_in_kubernetes,
        LOG_CONFIG,
        request_id_ctx_var,
        RSYNC_PORT,
        wait_for_app_start,
    )
except ImportError:
    from .server_metrics import (
        get_inactivity_ttl_annotation,
        HeartbeatManager,
        setup_otel_metrics,
    )
    from .utils import (
        clear_debugging_sessions,
        deep_breakpoint,
        DEFAULT_ALLOWED_SERIALIZATION,
        ensure_structured_logging,
        is_running_in_kubernetes,
        LOG_CONFIG,
        request_id_ctx_var,
        RSYNC_PORT,
        wait_for_app_start,
    )

from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import StreamingResponse

logging.config.dictConfig(LOG_CONFIG)

# Set up our structured JSON logging
ensure_structured_logging()

# Create the print logger AFTER ensure_structured_logging so it inherits handlers
print_logger = logging.getLogger("print_redirect")

logger = logging.getLogger(__name__)
# Set log level based on environment variable
# Don't default the log_level
kt_log_level = os.getenv("KT_LOG_LEVEL")
if kt_log_level:
    kt_log_level = kt_log_level.upper()
    logger.setLevel(getattr(logging, kt_log_level, logging.INFO))

_CACHED_CALLABLES = {}
_LAST_DEPLOYED = 0
_CACHED_IMAGE = []
DISTRIBUTED_SUPERVISOR = None
APP_PROCESS = None
_CALLABLE_LOAD_LOCK = threading.Lock()  # Lock for thread-safe callable loading
LOKI_HOST = os.environ.get("LOKI_HOST", "loki-gateway.kubetorch.svc.cluster.local")
LOKI_PORT = int(os.environ.get("LOKI_PORT", 80))  # Default Loki port
KT_OTEL_ENABLED = os.environ.get("KT_OTEL_ENABLED", "False").lower() == "true"
KT_TRACING_ENABLED = (
    os.environ.get("KT_TRACING_ENABLED", "").lower() != "false"
)  # Defaults to True

# Global termination event that can be checked by running requests
TERMINATION_EVENT = threading.Event()
# Create a client for FastAPI service

# Set the python breakpoint to kt.deep_breakpoint
os.environ["PYTHONBREAKPOINT"] = "kubetorch.deep_breakpoint"

request_id_ctx_var.set(os.getenv("KT_LAUNCH_ID", "-"))

#####################################
######### Instrument Traces #########
#####################################
instrument_traces = KT_TRACING_ENABLED
if instrument_traces:
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.logging import LoggingInstrumentor
        from opentelemetry.instrumentation.requests import RequestsInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        instrument_traces = False

if instrument_traces:
    logger.info("Configuring OTLP exporter to instrument traces")
    trace.set_tracer_provider(
        TracerProvider(
            resource=Resource.create(
                {
                    "service.name": os.environ.get("OTEL_SERVICE_NAME"),
                    "service.instance.id": os.environ.get("POD_NAME"),
                }
            )
        )
    )
    span_processor = BatchSpanProcessor(
        OTLPSpanExporter(
            endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"),
            insecure=True,
        )
    )
    trace.get_tracer_provider().add_span_processor(span_processor)
    RequestsInstrumentor().instrument()
    LoggingInstrumentor().instrument()

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
    rp_req = proxy_client.build_request(
        request.method, url, headers=request.headers.raw, content=await request.body()
    )

    # Send the request and get streaming response
    rp_resp = await proxy_client.send(rp_req, stream=True)

    # Return streaming response
    return StreamingResponse(
        rp_resp.aiter_raw(),
        status_code=rp_resp.status_code,
        headers=rp_resp.headers,
        background=BackgroundTask(rp_resp.aclose),
    )


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
        logger.info("Running image setup.")
    else:
        logger.debug("Skipping image setup steps, no changes detected.")

    # Grab the current list of installed dependencies with pip freeze to check if anything changes (we need to send a
    # SIGHUP to restart the server if so)
    start_deps = None
    import subprocess

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
                kt_pip_cmd = kt_pip_cmd or _get_kt_pip_install_cmd()
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
                logger.info(f"Running image setup step: {command}")

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

                if is_app_cmd and os.getenv("KT_APP_PORT"):
                    # wait for internal app to be healthy/ready if run port is provided
                    try:
                        port = os.getenv("KT_APP_PORT")
                        logger.debug(
                            f"Waiting for internal app on port {port} to start:"
                        )
                        wait_for_app_start(
                            port=port,
                            health_check=os.getenv("KT_APP_HEALTHCHECK"),
                            process=process,
                        )
                        logger.info(f"App on port {port} is ready.")
                    except Exception as e:
                        logger.error(f"Caught exception waiting for app to start: {e}")
                else:
                    # Check if this is a background command (ends with &)
                    is_background = command.rstrip().endswith("&")

                    if is_background:
                        # For background processes, give it a moment to start and check for immediate failures
                        import time

                        time.sleep(0.5)  # Brief pause to catch immediate errors

                        # Check if process failed immediately
                        poll_result = process.poll()
                        if poll_result is not None and poll_result != 0:
                            # Process exited with error
                            stdout_thread.join(timeout=1)
                            stderr_thread.join(timeout=1)
                            return_code = poll_result
                        else:
                            # Process is running in background successfully
                            logger.info(
                                f"Background process started successfully (PID: {process.pid})"
                            )
                            return_code = 0  # Indicate success for background start
                    else:
                        # Wait for process to complete
                        return_code = process.wait()

                        # Wait for streaming threads to finish
                        stdout_thread.join()
                        stderr_thread.join()

                    if return_code != 0 and not is_app_cmd:
                        with stderr_lock:
                            if stderr_lines:
                                logger.error(
                                    f"Failed to run command '{command}' with stderr:"
                                )
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
            changed_deps = [
                line.split("==")[0]
                for line in start_deps
                if "==" in line and line not in end_deps
            ]
            imported_changed_deps = [
                dep for dep in changed_deps if dep in sys.modules
            ]  # Only reload deps which are already imported
            if imported_changed_deps:
                logger.debug(
                    f"New dependencies found: {imported_changed_deps}, forcing reload"
                )

                # Don't clear the callable cache here - let load_callable_from_env handle it to preserve __kt_cached_state__
                if DISTRIBUTED_SUPERVISOR:
                    DISTRIBUTED_SUPERVISOR.cleanup()

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


def run_image_setup(deployed_time: Optional[float] = None):
    if os.environ["KT_FREEZE"] == "True" or not is_running_in_kubernetes():
        return

    rsync_file_updates()

    dockerfile_path = kt_directory() / "image.dockerfile"
    if not dockerfile_path.exists():
        raise FileNotFoundError(
            f"No image and metadata configuration found in path: {str(dockerfile_path)}"
        )
    while (
        # May need to give the dockerfile time to rsync over, so wait until the dockerfile timestamp is later than
        # when we started the deployment (recorded in .to and passed here as deployed_time). We also should only
        # wait if _LAST_DEPLOYED is not zero, as the first time the server is deployed the image is written before
        # the server starts so we don't need to wait.
        _LAST_DEPLOYED
        and dockerfile_path.stat().st_mtime < deployed_time
        and datetime.now(timezone.utc).timestamp() - deployed_time < 5
    ):
        time.sleep(0.1)

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


async def run_in_executor_with_context(executor, func, *args):
    """
    Helper to run a function in an executor while preserving the request_id context.

    This wrapper captures the current request_id from the context before running
    the function in a thread pool executor, then sets it in the new thread.
    """
    import asyncio

    # Capture the current request_id before switching threads
    current_request_id = request_id_ctx_var.get("-")

    def wrapper(*args):
        # Set the request_id in the executor thread
        token = None
        if current_request_id != "-":
            token = request_id_ctx_var.set(current_request_id)
        try:
            return func(*args)
        finally:
            # Clean up the context to avoid leaking between requests
            if token is not None:
                request_id_ctx_var.reset(token)

    return await asyncio.get_event_loop().run_in_executor(executor, wrapper, *args)


def should_reload(deployed_as_of: Optional[str] = None) -> bool:
    """
    Determine if the server should reload based on the deployment timestamp.
    If deployed_as_of is provided, it checks against the last deployed time.
    If not provided, it defaults to False.
    """
    if deployed_as_of in [None, "null", "None"]:
        return False

    try:
        deployed_time = datetime.fromisoformat(deployed_as_of).timestamp()
        return deployed_time > _LAST_DEPLOYED
    except ValueError as e:
        logger.error(f"Invalid deployed_as_of format: {deployed_as_of}. Error: {e}")
        return True


def load_callable(
    deployed_as_of: Optional[str] = None,
    distributed_subprocess: bool = False,
    reload_cleanup_fn: [Callable, None] = None,
):
    global _LAST_DEPLOYED

    callable_name = os.environ["KT_CLS_OR_FN_NAME"]

    callable_obj = _CACHED_CALLABLES.get(callable_name, None)
    if callable_obj and not should_reload(deployed_as_of):
        # If the callable is cached and doesn't need reload, return it immediately
        logger.debug("Returning cached callable.")
        return callable_obj

    # Slow path: need to load or reload - use lock for thread safety
    with _CALLABLE_LOAD_LOCK:
        # Double-check within lock (another thread might have loaded it)
        callable_obj = _CACHED_CALLABLES.get(callable_name, None)
        if callable_obj and not should_reload(deployed_as_of):
            logger.debug("Returning cached callable (found after acquiring lock).")
            return callable_obj
        # Proceed with loading/reloading
        return _load_callable_internal(
            deployed_as_of, distributed_subprocess, reload_cleanup_fn, callable_obj
        )


def _load_callable_internal(
    deployed_as_of: Optional[str] = None,
    distributed_subprocess: bool = False,
    reload_cleanup_fn: [Callable, None] = None,
    callable_obj=None,
):
    """Internal callable loading logic - should be called within lock for thread safety."""
    global _LAST_DEPLOYED

    callable_name = os.environ["KT_CLS_OR_FN_NAME"]

    if not callable_obj:
        logger.debug("Callable not found in cache, loading from environment.")
    else:
        logger.debug(
            f"Callable found in cache, but reloading because deployed_as_of {deployed_as_of} is newer than last deployed time {_LAST_DEPLOYED}"
        )

    # If not in cache or we have a more recent deployment timestamp, update metadata and reload
    if reload_cleanup_fn and _LAST_DEPLOYED:
        # If a reload cleanup function is provided and we've already deployed at least once, call it before
        # reloading the callable
        reload_cleanup_fn()

    deployed_time = (
        datetime.fromisoformat(deployed_as_of).timestamp()
        if deployed_as_of
        else datetime.now(timezone.utc).timestamp()
    )
    if not distributed_subprocess:
        # We don't reload the image in distributed subprocess/es, as we already did it in the
        # main process and we don't want to do it multiple times (in each subprocess).
        if _LAST_DEPLOYED:
            logger.info("Patching image and code updates and reloading callable.")
        else:
            logger.info("Setting up image and loading callable.")
        run_image_setup(deployed_time)

    distributed_config = os.environ["KT_DISTRIBUTED_CONFIG"]
    if distributed_config not in ["null", "None"] and not distributed_subprocess:
        logger.debug(f"Loading distributed supervisor: {distributed_config}")
        callable_obj = load_distributed_supervisor(deployed_as_of=deployed_as_of)
        logger.debug("Distributed supervisor loaded successfully.")
    else:
        logger.debug(f"Loading callable from environment: {callable_name}")
        callable_obj = load_callable_from_env()
        logger.debug("Callable loaded successfully.")

    _LAST_DEPLOYED = deployed_time
    _CACHED_CALLABLES[callable_name] = callable_obj

    return callable_obj


def load_distributed_supervisor(deployed_as_of: Optional[str] = None):
    global DISTRIBUTED_SUPERVISOR

    if os.environ["KT_FILE_PATH"] not in sys.path:
        sys.path.insert(0, os.environ["KT_FILE_PATH"])

    distributed_config = os.environ["KT_DISTRIBUTED_CONFIG"]

    # If this is the main process of a distributed call, we don't load the callable directly,
    # we create a new supervisor if it doesn't exist or if the config has changed.
    # We don't create a supervisor if this is a distributed subprocess.
    config_hash = hash(str(distributed_config))
    if (
        DISTRIBUTED_SUPERVISOR is None
        or config_hash != DISTRIBUTED_SUPERVISOR.config_hash
    ):
        from .distributed_utils import distributed_supervisor_factory

        logger.info(f"Loading distributed supervisor with config: {distributed_config}")
        distributed_config = json.loads(distributed_config)
        # If we already have some distributed processes, we need to clean them up before creating a new supervisor.
        if DISTRIBUTED_SUPERVISOR:
            DISTRIBUTED_SUPERVISOR.cleanup()
        DISTRIBUTED_SUPERVISOR = distributed_supervisor_factory(**distributed_config)
        DISTRIBUTED_SUPERVISOR.config_hash = config_hash
    try:
        # If there are any errors during setup, we catch and log them, and then undo the setup
        # so that the distributed supervisor is not left in a broken state (and otherwise can still fail
        # when we call DISTRIBUTED_SUPERVISOR.cleanup() in lifespan).
        DISTRIBUTED_SUPERVISOR.setup(deployed_as_of=deployed_as_of)
    except Exception as e:
        logger.error(
            f"Failed to set up distributed supervisor with config {distributed_config}: {e}"
        )
        DISTRIBUTED_SUPERVISOR = None
        raise e
    return DISTRIBUTED_SUPERVISOR


def patch_sys_path():
    abs_path = str(Path(os.environ["KT_FILE_PATH"]).expanduser().resolve())
    if os.environ["KT_FILE_PATH"] not in sys.path:
        sys.path.insert(0, abs_path)
        logger.debug(f"Added {abs_path} to sys.path")

    # Maybe needed for subprocesses (e.g. distributed) to find the callable's module
    # Needed for distributed subprocesses to find the file path
    existing_path = os.environ.get("PYTHONPATH", "")
    if os.environ["KT_FILE_PATH"] not in existing_path:
        os.environ["PYTHONPATH"] = (
            f"{abs_path}{os.pathsep}{existing_path}" if existing_path else abs_path
        )
        logger.debug(f"Set PYTHONPATH to {os.environ['PYTHONPATH']}")


def load_callable_from_env():
    """Load and cache callable objects from env, preserving state if __kt_cached_state__ is available."""
    cls_or_fn_name = os.environ["KT_CLS_OR_FN_NAME"]
    module_name = os.environ["KT_MODULE_NAME"]

    # Check if we have an existing cached callable and extract state if available
    cached_state = None
    existing_callable = _CACHED_CALLABLES.get(cls_or_fn_name, None)

    if existing_callable and hasattr(existing_callable, "__kt_cached_state__"):
        try:
            logger.info(
                f"Extracting cached state from {cls_or_fn_name} via __kt_cached_state__"
            )
            cached_state = existing_callable.__kt_cached_state__()
            if cached_state is not None and not isinstance(cached_state, dict):
                logger.warning(
                    f"__kt_cached_state__ returned non-dict type: {type(cached_state)}. Ignoring cached state."
                )
                cached_state = None
        except Exception as e:
            # This could happen if modules were removed from sys.modules during image setup
            # and the callable's __kt_cached_state__ method depends on them
            logger.warning(
                f"Failed to extract cached state from {cls_or_fn_name} (possibly due to module reloading): {e}. "
                f"Proceeding without cached state."
            )
            cached_state = None

    # Now that we have the state, clean up the old callable to free memory
    if existing_callable:
        logger.debug(f"Deleting existing callable: {cls_or_fn_name}")
        _CACHED_CALLABLES.pop(cls_or_fn_name, None)
        del existing_callable
        # Garbage collect to ensure everything cleaned up (especially GPU memory)
        import gc

        gc.collect()

    patch_sys_path()

    # If we're inside a distributed subprocess or the main process of a non-distributed call,
    # we load and instantiate the callable.
    try:
        # Try regular package import first
        if module_name in sys.modules:
            # We make this logs to info because some imports are slow and we want the user to know that it's not our fault
            # and not hanging
            logger.info(f"Reimporting module {module_name}")
            # Clear any existing debugging sessions when reloading modules
            clear_debugging_sessions()
            module = importlib.reload(sys.modules[module_name])
        else:
            logger.debug(f"Importing module {module_name}")
            module = importlib.import_module(module_name)
        logger.debug(f"Module {module_name} loaded")

        # Ensure our structured logging is in place after user module import
        # (in case the user's module configured its own logging)
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

        # Add cached state if available
        # Allow user to manually set "kt_cached_state" to override/disable cache
        if cached_state is not None and "kt_cached_state" not in init_kwargs:
            # Check if the class's __init__ accepts kt_cached_state parameter
            sig = inspect.signature(callable_obj.__init__)
            if "kt_cached_state" in sig.parameters:
                logger.info(f"Passing cached state to {cls_or_fn_name}.__init__")
                init_kwargs["kt_cached_state"] = cached_state
            else:
                raise ValueError(
                    f"Class {cls_or_fn_name} has __kt_cached_state__ method but __init__ does not accept "
                    f"'kt_cached_state' parameter. Please add 'kt_cached_state=None' to __init__ signature."
                )

        # Instantiate with combined arguments
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
        raise ImportError(
            f"Could not load spec for module {module_name} from {file_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#####################################
########## Rsync Helpers ############
#####################################
def generate_rsync_command(subdir: str = ".", exclude_absolute: bool = True):
    """Generate rsync command for syncing from jump pod.

    Args:
        subdir: Directory to sync to (default current directory)
        exclude_absolute: Whether to exclude __absolute__ directory (default True)
    """
    service_name = os.getenv("KT_SERVICE_NAME")
    namespace = os.getenv("POD_NAMESPACE")

    exclude_opt = "--exclude='__absolute__*' " if exclude_absolute else ""
    logger.debug("Syncing code from rsync pod to local directory")
    return f"rsync -av {exclude_opt}rsync://kubetorch-rsync.{namespace}.svc.cluster.local:{RSYNC_PORT}/data/{namespace}/{service_name}/ {subdir}"


def rsync_file_updates():
    """Rsync files from the jump pod to the worker pod.

    Performs two rsync operations in parallel:
    1. Regular files (excluding __absolute__*) to the working directory
    2. Absolute path files (under __absolute__/) to their absolute destinations
    """
    import concurrent.futures
    from concurrent.futures import ThreadPoolExecutor

    service_name = os.getenv("KT_SERVICE_NAME")
    namespace = os.getenv("POD_NAMESPACE")

    # Build base rsync URL
    rsync_base = f"rsync://kubetorch-rsync.{namespace}.svc.cluster.local:{RSYNC_PORT}/data/{namespace}/{service_name}/"

    max_retries = 5
    base_delay = 1  # seconds
    max_delay = 30  # seconds

    def run_rsync_with_retries(rsync_cmd, description):
        """Helper to run rsync with exponential backoff retries."""
        for attempt in range(max_retries):
            resp = subprocess.run(
                rsync_cmd,
                shell=True,
                capture_output=True,
                text=True,
            )

            if resp.returncode == 0:
                logger.debug(f"Successfully rsync'd {description}")
                return  # Success!

            # Check if it's a retryable error
            retryable_errors = [
                "max connections",
                "Temporary failure in name resolution",
                "Name or service not known",
                "Connection refused",
                "No route to host",
            ]

            is_retryable = any(error in resp.stderr for error in retryable_errors)

            if is_retryable and attempt < max_retries - 1:
                # Calculate exponential backoff with jitter
                delay = min(
                    base_delay * (2**attempt) + random.uniform(0, 1), max_delay
                )
                logger.warning(
                    f"Rsync {description} failed with retryable error: {resp.stderr.strip()}. "
                    f"Retrying in {delay:.1f} seconds (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(delay)
            else:
                # For non-retryable errors or final attempt, raise immediately
                if attempt == max_retries - 1:
                    logger.error(
                        f"Rsync {description} failed after {max_retries} attempts. Last error: {resp.stderr}"
                    )
                raise RuntimeError(
                    f"Rsync {description} failed with error: {resp.stderr}"
                )

        # If we exhausted all retries
        raise RuntimeError(
            f"Rsync {description} failed after {max_retries} attempts. Last error: {resp.stderr}"
        )

    def rsync_regular_files():
        """Rsync regular files (excluding __absolute__*) to working directory."""
        rsync_cmd_regular = f"rsync -avL --exclude='__absolute__*' {rsync_base} ."
        logger.debug(f"Rsyncing regular files with command: {rsync_cmd_regular}")
        run_rsync_with_retries(rsync_cmd_regular, "regular files")

    def rsync_absolute_files():
        """Rsync absolute path files to their absolute destinations."""
        # First, do a dry-run to see if __absolute__ directory exists
        check_cmd = f"rsync --list-only {rsync_base}__absolute__/"
        check_resp = subprocess.run(
            check_cmd, shell=True, capture_output=True, text=True
        )

        if check_resp.returncode == 0 and check_resp.stdout.strip():
            # __absolute__ directory exists, sync its contents to root
            # The trick is to sync from __absolute__/ to / which places files in their absolute paths
            rsync_cmd_absolute = f"rsync -avL {rsync_base}__absolute__/ /"
            logger.debug(
                f"Rsyncing absolute path files with command: {rsync_cmd_absolute}"
            )
            run_rsync_with_retries(rsync_cmd_absolute, "absolute path files")
        else:
            logger.debug("No absolute path files to sync")

    # Run both rsync operations in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both tasks
        regular_future = executor.submit(rsync_regular_files)
        absolute_future = executor.submit(rsync_absolute_files)

        # Wait for both to complete and handle any exceptions
        futures = [regular_future, absolute_future]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # This will raise any exception that occurred
            except Exception as e:
                # Cancel remaining futures if one fails
                for f in futures:
                    f.cancel()
                raise e

    logger.debug("Completed rsync of all files")


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


class RequestContextFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_ctx_var.get("-")
        record.pod = os.getenv("POD_NAME", "unknown-pod")

        if instrument_traces:
            from opentelemetry.trace import format_trace_id, get_current_span

            # Add trace_id and span_id for log correlation
            current_span = get_current_span()
            if current_span and current_span.get_span_context().is_valid:
                record.trace_id = format_trace_id(
                    current_span.get_span_context().trace_id
                )
                record.span_id = format_trace_id(
                    current_span.get_span_context().span_id
                )
            else:
                record.trace_id = "-"
                record.span_id = "-"

        return True


class TerminationCheckMiddleware(BaseHTTPMiddleware):
    """Monitor for termination while request is running and return error if detected."""

    async def dispatch(self, request: Request, call_next):
        # Skip health checks and metrics endpoints
        if request.url.path in ["/health", "/", "/metrics"]:
            return await call_next(request)

        # Run the actual request in the background
        import asyncio

        request_task = asyncio.create_task(call_next(request))

        # Monitor for termination while request is running
        while not request_task.done():
            # Check if we're terminating
            if TERMINATION_EVENT.is_set() or (
                hasattr(request.app.state, "terminating")
                and request.app.state.terminating
            ):
                # Cancel the request task
                request_task.cancel()

                # Return PodTerminatedError
                from kubetorch import PodTerminatedError
                from kubetorch.servers.http.http_server import package_exception

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
                result = await asyncio.wait_for(
                    asyncio.shield(request_task), timeout=0.5
                )
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

        if instrument_traces and request_id != "-":
            span_attributes = {
                "request_id": request_id,
                "http.method": request.method,
                "http.url": str(request.url),
                "service.name": os.environ.get("OTEL_SERVICE_NAME"),
                "service.instance.id": os.environ.get("POD_NAME"),
            }
            # of the pod crashes (e.g., due to OOM) during execution of run_callable, we'll still have at least
            # this heartbeat span recorded
            tracer = trace.get_tracer("heartbeat")
            try:
                with tracer.start_as_current_span(
                    "heartbeat.request", attributes=span_attributes
                ):
                    tracer_provider = trace.get_tracer_provider()
                    if isinstance(tracer_provider, TracerProvider):
                        tracer_provider.force_flush()
            except Exception as e:
                logger.warning(f"Heartbeat span flush failed: {e}")

        try:
            response = await call_next(request)
            return response
        finally:
            # Reset the context variable to its default value
            request_id_ctx_var.reset(token)


class TraceFlushMiddleware(BaseHTTPMiddleware):
    """Flush traces after each HTTP Request so we don't lose trace data if the pod is killed"""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        tracer_provider = trace.get_tracer_provider()
        if isinstance(tracer_provider, TracerProvider):
            tracer_provider.force_flush()
        return response


class StreamToLogger:
    def __init__(self, logger, log_level=logging.INFO, original_stream=None):
        self.logger = logger
        self.log_level = log_level
        self.original_stream = original_stream
        self.linebuf = ""

    def _is_from_logging(self):
        """Check if the current write call is coming from the logging system"""
        frame = sys._getframe()
        while frame:
            if frame.f_globals.get("__name__", "").startswith("logging"):
                return True
            frame = frame.f_back
        return False

    def write(self, buf):
        # Check if this is from logging system
        is_from_logging = self._is_from_logging()

        # Always write to original stream first
        if self.original_stream:
            self.original_stream.write(buf)
            self.original_stream.flush()

        # Skip logging if this is from the logging system to prevent infinite loops
        if self.logger.name == "print_redirect" and is_from_logging:
            return

        # Buffer and log complete lines
        temp_linebuf = self.linebuf + buf
        self.linebuf = ""

        # Split on newlines but keep carriage returns
        lines = []
        current_line = ""
        for char in temp_linebuf:
            if char == "\n":
                lines.append(current_line)
                current_line = ""
            else:
                current_line += char

        # Add any remaining content to linebuf
        if current_line:
            self.linebuf = current_line

        # Log complete lines
        for line in lines:
            if line:
                self.logger.log(self.log_level, line)

    def flush(self):
        if self.original_stream:
            self.original_stream.flush()
        if self.linebuf != "":
            self.logger.log(self.log_level, self.linebuf)
            self.linebuf = ""

    def isatty(self):
        # Delegate to the original stream if it exists, else return False
        if self.original_stream and hasattr(self.original_stream, "isatty"):
            return self.original_stream.isatty()
        return False

    def fileno(self):
        if self.original_stream and hasattr(self.original_stream, "fileno"):
            return self.original_stream.fileno()
        raise OSError("Stream does not support fileno()")

    @property
    def encoding(self):
        # Return the encoding of the original stream if available, else UTF-8
        if self.original_stream and hasattr(self.original_stream, "encoding"):
            return self.original_stream.encoding
        return "utf-8"


# Save original streams before redirection
_original_stdout = sys.stdout
_original_stderr = sys.stderr

# Redirect stdout and stderr to our logger while preserving original streams
sys.stdout = StreamToLogger(print_logger, logging.INFO, _original_stdout)
sys.stderr = StreamToLogger(print_logger, logging.ERROR, _original_stderr)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    import signal
    import threading

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
            global DISTRIBUTED_SUPERVISOR
            if DISTRIBUTED_SUPERVISOR:
                logger.info("Cleaning up distributed supervisor and child processes...")
                try:
                    DISTRIBUTED_SUPERVISOR.cleanup()
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

    # Startup
    ttl = get_inactivity_ttl_annotation()
    if ttl and KT_OTEL_ENABLED is True:
        app.state.heartbeat_manager = HeartbeatManager(ttl_seconds=ttl)
        if app.state.heartbeat_manager:
            await app.state.heartbeat_manager.start()
            logger.debug(f"Heartbeat manager started with TTL={ttl}s")
    elif ttl:
        logger.warning(
            "TTL annotation found, but OTEL is not enabled, heartbeat disabled"
        )
    else:
        logger.debug("No TTL annotation found, heartbeat disabled")

    try:
        if os.getenv("KT_CALLABLE_TYPE") == "app":
            cached_image_setup()
        else:
            load_callable()

        logger.info("Kubetorch Server started.")
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
        request_id_ctx_var.set("-")
        yield

    finally:
        # Flush OpenTelemetry traces before shutdown
        if instrument_traces:
            from opentelemetry.sdk.trace import TracerProvider

            tracer_provider = trace.get_tracer_provider()
            if isinstance(tracer_provider, TracerProvider):
                logger.info("Forcing OpenTelemetry span flush before shutdown")
                tracer_provider.force_flush()

        # Shutdown
        manager = getattr(app.state, "heartbeat_manager", None)
        if manager:
            await manager.stop()
            logger.info("Heartbeat manager stopped")

        # Clean up during normal shutdown so we don't leave any hanging processes, which can cause pods to hang
        # indefinitely. Skip if already cleaned up by SIGTERM handler.
        if DISTRIBUTED_SUPERVISOR and not getattr(app.state, "terminating", False):
            DISTRIBUTED_SUPERVISOR.cleanup()

        # Clear any remaining debugging sessions
        clear_debugging_sessions()


# Add the filter to uvicorn's access logger
logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())
root_logger = logging.getLogger()
root_logger.addFilter(RequestContextFilter())
for handler in root_logger.handlers:
    handler.addFilter(RequestContextFilter())
print_logger.addFilter(RequestContextFilter())

app = FastAPI(lifespan=lifespan)
app.add_middleware(TerminationCheckMiddleware)  # Check termination first
app.add_middleware(RequestIDMiddleware)

# Configure the FastAPI app for metrics first
# Method will return None for meter_provider if otel is not enabled
app, meter_provider = (
    setup_otel_metrics(app) if KT_OTEL_ENABLED is True else (app, None)
)

# Now instrument for traces and metrics together
if instrument_traces:
    logger.info("Instrumenting FastAPI app for traces and metrics")
    FastAPIInstrumentor.instrument_app(
        app,
        meter_provider=meter_provider,
        excluded_urls="/metrics,/health",
    )
    logger.info("Adding TraceFlushMiddleware to flush traces")
    app.add_middleware(TraceFlushMiddleware)
elif meter_provider is not None:
    try:
        # Skipped if instrument_traces is False, need to reimplement if we want to use metrics only
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        logger.info("Instrumenting FastAPI app for metrics only")
        FastAPIInstrumentor.instrument_app(
            app,
            meter_provider=meter_provider,
            excluded_urls="/,/metrics,/health",
        )
    except ImportError:
        logger.info(
            "OpenTelemetry instrumentation not enabled, skipping metrics instrumentation"
        )

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
    pod_name: str
    state: Optional[dict] = None  # Optional serialized exception state


# Factor out the exception packaging so we can use it in the handler below and also inside distributed subprocesses
def package_exception(exc: Exception):
    import asyncio
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
        except Exception as e:
            logger.debug(f"Could not serialize exception state for {error_type}: {e}")

    error_response = ErrorResponse(
        error_type=error_type,
        message=str(exc),
        traceback=trace,
        pod_name=os.getenv("POD_NAME"),
        state=state,
    )

    return JSONResponse(status_code=status_code, content=error_response.model_dump())


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return package_exception(exc)


@app.post("/_reload_image", response_class=JSONResponse)
def _reload_image(
    request: Request,
    deployed_as_of: Optional[str] = Header(None, alias="X-Deployed-As-Of"),
):
    """
    Endpoint to reload the image and metadata configuration.
    This is used to reload the image in cases where we're not calling the callable directly,
    e.g. kt.app and Ray workers.
    """
    global _LAST_DEPLOYED
    deployed_time = (
        datetime.fromisoformat(deployed_as_of).timestamp()
        if deployed_as_of
        else datetime.now(timezone.utc).timestamp()
    )
    run_image_setup(deployed_time)
    _LAST_DEPLOYED = deployed_time
    return JSONResponse(
        status_code=200,
        content={"message": "Image and metadata reloaded successfully."},
    )


@app.post("/{cls_or_fn_name}", response_class=JSONResponse)
@app.post("/{cls_or_fn_name}/{method_name}", response_class=JSONResponse)
async def run_callable(
    request: Request,
    cls_or_fn_name: str,
    method_name: Optional[str] = None,
    distributed_subcall=False,
    debug_port: Optional[int] = None,
    params: Optional[Union[Dict, str]] = Body(default=None),
    deployed_as_of: Optional[str] = Header(None, alias="X-Deployed-As-Of"),
    serialization: str = Header("json", alias="X-Serialization"),
):
    if cls_or_fn_name != os.environ["KT_CLS_OR_FN_NAME"]:
        raise HTTPException(
            status_code=404,
            detail=f"Callable '{cls_or_fn_name}' not found in metadata configuration. Found '{os.environ['KT_CLS_OR_FN_NAME']}' instead",
        )

    # NOTE: The distributed replica processes (e.g. PyTorchProcess:run) rely on this running here even though
    # they will reconstruct the callable themselves, because they skip image reloading as a performance optimization.
    # Run load_callable in executor since it may do file I/O and other blocking operations
    callable_obj = await run_in_executor_with_context(
        None, load_callable, deployed_as_of
    )

    # If this is a distributed call (and not a subcall from a different distributed replica),
    # and the type of distribution which requires a special call method (e.g. SIMD), use the
    # distributed supervisor to handle the call
    if DISTRIBUTED_SUPERVISOR and DISTRIBUTED_SUPERVISOR.intercept_call():
        # Run the blocking distributed call in executor to avoid blocking the event loop
        result = await run_in_executor_with_context(
            None,
            DISTRIBUTED_SUPERVISOR.call_distributed,
            request,
            cls_or_fn_name,
            method_name,
            params,
            distributed_subcall,
            debug_port,
            deployed_as_of,
        )
        clear_debugging_sessions()
        return result

    # If this is not a distributed call, or the distribution type does not require special handling,
    # run the callable directly
    result = await run_callable_internal(
        callable_obj=callable_obj,
        cls_or_fn_name=cls_or_fn_name,
        method_name=method_name,
        params=params,
        serialization=serialization,
        debug_port=debug_port,
    )
    return result


async def run_callable_internal(
    callable_obj: Callable,
    cls_or_fn_name: str,
    method_name: Optional[str] = None,
    params: Optional[Union[Dict, str]] = Body(default=None),
    serialization: str = "json",
    debug_port: Optional[int] = None,
):
    # Check if serialization is allowed
    allowed_serialization = os.getenv(
        "KT_ALLOWED_SERIALIZATION", DEFAULT_ALLOWED_SERIALIZATION
    ).split(",")
    if serialization not in allowed_serialization:
        raise HTTPException(
            status_code=400,
            detail=f"Serialization format '{serialization}' not allowed. Allowed formats: {allowed_serialization}",
        )

    # Process the call
    args = []
    kwargs = {}
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

    if method_name:
        if not hasattr(callable_obj, method_name):
            raise HTTPException(
                status_code=404,
                detail=f"Method '{method_name}' not found in class '{cls_or_fn_name}'",
            )
        user_method = getattr(callable_obj, method_name)
    else:
        user_method = callable_obj

    import inspect

    # Check if the user method is async
    is_async_method = inspect.iscoroutinefunction(user_method)

    if debug_port:
        logger.info(
            f"Debugging remote callable {cls_or_fn_name}.{method_name} on port {debug_port}"
        )
        deep_breakpoint(debug_port)
        # If using the debugger, step in here ("s") to enter your function/class method.
        if is_async_method:
            result = await user_method(*args, **kwargs)
        else:
            # Run sync method in thread pool to avoid blocking
            # Use lambda to properly pass both args and kwargs
            result = await run_in_executor_with_context(
                None, lambda: user_method(*args, **kwargs)
            )
    else:
        logger.debug(f"Calling remote callable {cls_or_fn_name}.{method_name}")
        if is_async_method:
            result = await user_method(*args, **kwargs)
        else:
            # Run sync method in thread pool to avoid blocking
            # Use lambda to properly pass both args and kwargs
            result = await run_in_executor_with_context(
                None, lambda: user_method(*args, **kwargs)
            )

    # Handle case where sync method returns an awaitable (e.g., from an async framework)
    # This is less common but can happen with some async libraries
    if isinstance(result, Awaitable):
        result = await result

    # Serialize response based on format
    if serialization == "pickle":
        try:
            pickled_result = pickle.dumps(result)
            encoded_result = base64.b64encode(pickled_result).decode("utf-8")
            result = {"data": encoded_result}
        except Exception as e:
            logger.error(f"Failed to pickle result: {str(e)}")
            raise SerializationError(
                f"Result could not be serialized with pickle: {str(e)}"
            )
    else:
        # Default JSON serialization
        try:
            json.dumps(result)
        except (TypeError, ValueError) as e:
            logger.error(f"Result is not JSON serializable: {str(e)}")
            raise SerializationError(
                f"Result could not be serialized to JSON: {str(e)}"
            )

    clear_debugging_sessions()

    return result


def run_callable_internal_sync(
    callable_obj: Callable,
    cls_or_fn_name: str,
    method_name: Optional[str] = None,
    params: Optional[Union[Dict, str]] = None,
    serialization: str = "json",
    debug_port: Optional[int] = None,
):
    """Synchronous wrapper for run_callable_internal, used by distributed subprocesses."""
    import asyncio
    import inspect

    # Check if serialization is allowed
    allowed_serialization = os.getenv(
        "KT_ALLOWED_SERIALIZATION", DEFAULT_ALLOWED_SERIALIZATION
    ).split(",")
    if serialization not in allowed_serialization:
        raise HTTPException(
            status_code=400,
            detail=f"Serialization format '{serialization}' not allowed. Allowed formats: {allowed_serialization}",
        )

    # Process the call
    args = []
    kwargs = {}
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

    if method_name:
        if not hasattr(callable_obj, method_name):
            raise HTTPException(
                status_code=404,
                detail=f"Method '{method_name}' not found in class '{cls_or_fn_name}'",
            )
        user_method = getattr(callable_obj, method_name)
    else:
        user_method = callable_obj

    # Check if the user method is async
    is_async_method = inspect.iscoroutinefunction(user_method)

    if debug_port:
        logger.info(
            f"Debugging remote callable {cls_or_fn_name}.{method_name} on port {debug_port}"
        )
        deep_breakpoint(debug_port)
        # If using the debugger, step in here ("s") to enter your function/class method.
        if is_async_method:
            # For async methods in sync context, we need to run them in a new event loop
            result = asyncio.run(user_method(*args, **kwargs))
        else:
            result = user_method(*args, **kwargs)
    else:
        logger.debug(f"Calling remote callable {cls_or_fn_name}.{method_name}")
        if is_async_method:
            # For async methods in sync context, we need to run them in a new event loop
            result = asyncio.run(user_method(*args, **kwargs))
        else:
            result = user_method(*args, **kwargs)

    # Handle case where sync method returns an awaitable
    if isinstance(result, Awaitable):
        result = asyncio.run(result)

    # Serialize response based on format
    if serialization == "pickle":
        try:
            pickled_result = pickle.dumps(result)
            encoded_result = base64.b64encode(pickled_result).decode("utf-8")
            result = {"data": encoded_result}
        except Exception as e:
            logger.error(f"Failed to pickle result: {str(e)}")
            raise SerializationError(
                f"Result could not be serialized with pickle: {str(e)}"
            )
    else:
        # Default JSON serialization
        try:
            json.dumps(result)
        except (TypeError, ValueError) as e:
            logger.error(f"Result is not JSON serializable: {str(e)}")
            raise SerializationError(
                f"Result could not be serialized to JSON: {str(e)}"
            )

    clear_debugging_sessions()

    return result


@app.get("/health", include_in_schema=False)
@app.get("/", include_in_schema=False)
def health():
    return {"status": "healthy"}


if __name__ == "__main__" and not is_running_in_container():
    # NOTE: this will only run in local development, otherwise we start the uvicorn server in the pod template setup
    import uvicorn
    from dotenv import load_dotenv

    load_dotenv()

    logger.info("Starting HTTP server")
    uvicorn.run(app, host="0.0.0.0", port=os.environ.get("KT_SERVER_PORT", 32300))
