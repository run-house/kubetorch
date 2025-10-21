import asyncio
import atexit
import base64
import enum
import hashlib
import json
import os
import pickle
import re
import socket
import subprocess
import sys
import time
from contextvars import ContextVar
from typing import List

import httpx

import jinja2
import websockets
import yaml

from kubetorch.constants import LOCALHOST
from kubetorch.logger import get_logger
from kubetorch.serving.constants import DEFAULT_DEBUG_PORT
from kubetorch.utils import ServerLogsFormatter

logger = get_logger(__name__)

RSYNC_PORT = 873

DEFAULT_ALLOWED_SERIALIZATION = "json"

MAGIC_CALL_KWARGS = ["workers", "restart_procs"]

LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {},
    "handlers": {},
    "root": {"level": "INFO", "handlers": []},
    "loggers": {
        "uvicorn": {"level": "INFO", "handlers": [], "propagate": True},
        "uvicorn.access": {"level": "INFO", "handlers": [], "propagate": True},
        "kubetorch": {"level": "INFO", "handlers": [], "propagate": True},
    },
}


def ensure_structured_logging():
    """Add our structured JSON handler to all loggers without removing user's handlers. We do this both when we
    set up the HTTP server and also after re-importing user code, as their modules might include logging setup
    of their own."""
    import logging
    import logging.handlers
    import os
    import sys

    from pythonjsonlogger import jsonlogger

    # First ensure logging is initialized - this is crucial!
    # If no handlers exist, we need to initialize the logging system
    root_logger = logging.getLogger()

    # Create our JSON formatter
    json_formatter = jsonlogger.JsonFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(request_id)s - %(pod)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create our structured handler (we keep using sys.stdout so user and kt logs
    # both appear in pod logs; our stdout wrapper will mirror to the original stream)
    structured_handler = logging.StreamHandler(sys.stdout)
    structured_handler.setFormatter(json_formatter)
    structured_handler.name = "kubetorch_structured"  # Name it so we can identify it

    # Set root logger level based on KT_LOG_LEVEL if it's set
    kt_log_level = os.getenv("KT_LOG_LEVEL")
    if kt_log_level:
        kt_log_level = kt_log_level.upper()
        root_logger.setLevel(getattr(logging, kt_log_level, logging.INFO))

    # Check if our handler is already there (to avoid adding duplicates)
    existing_structured = None
    for h in root_logger.handlers:
        if getattr(h, "name", None) == "kubetorch_structured":
            existing_structured = h
            break

    if not existing_structured:
        # Add our structured handler alongside any user-installed handlers
        # so both formats are emitted to pod logs
        root_logger.addHandler(structured_handler)

    # Ensure request context fields are attached to all records even if the user
    # reconfigured logging and removed our filters. Do this idempotently.
    class _ContextFieldsFilter(logging.Filter):
        def filter(self, record):
            if not hasattr(record, "request_id") or record.request_id in (None, "-"):
                try:
                    record.request_id = request_id_ctx_var.get("-")
                except Exception:
                    record.request_id = "-"
            if not hasattr(record, "pod") or record.pod in (None, ""):
                record.pod = os.getenv("POD_NAME", "unknown-pod")
            return True

    # Attach the filter to root and all of its handlers (idempotent: duplicate adds are ignored)
    context_filter = _ContextFieldsFilter()
    try:
        root_logger.addFilter(context_filter)
    except Exception:
        pass
    for h in root_logger.handlers:
        try:
            h.addFilter(context_filter)
        except Exception:
            pass

    # Ensure print_redirect logger also has proper configuration
    # This is important for the StreamToLogger output
    print_logger = logging.getLogger("print_redirect")
    print_logger.setLevel(logging.INFO)
    # Ensure it propagates to root so the structured handler formats it
    print_logger.propagate = True
    try:
        print_logger.addFilter(context_filter)
    except Exception:
        pass


request_id_ctx_var: ContextVar[str] = ContextVar("request_id", default="-")


class StartupError(Exception):
    pass


class PodTerminatedError(Exception):
    def __init__(
        self,
        pod_name: str = "unknown",
        reason: str = "Unknown",
        status_code: int = 503,
        events: List[dict] = None,
    ):
        """
        events: List of dicts with keys:
          - timestamp: datetime
          - reason: str
          - message: str

        sample event:
        {
            'timestamp': datetime.datetime(2025, 7, 13, 16, 45, 46, tzinfo=tzutc()),
            'reason': 'Evicted',
            'message': 'The node was low on resource: memory. Threshold quantity: 100Mi, available: 3404Ki.'
        }
        """
        self.pod_name = pod_name
        self.reason = reason
        self.status_code = status_code
        self.events = events or []
        super().__init__(str(self))

    def __getstate__(self):
        """Serialize the exception state for transmission over HTTP."""
        # Convert datetime objects to ISO format strings for JSON serialization
        serialized_events = []
        for event in self.events:
            serialized_event = event.copy()
            if "timestamp" in serialized_event:
                timestamp = serialized_event["timestamp"]
                # Convert datetime to string if needed
                if hasattr(timestamp, "isoformat"):
                    serialized_event["timestamp"] = timestamp.isoformat()
            serialized_events.append(serialized_event)

        return {
            "pod_name": self.pod_name,
            "reason": self.reason,
            "status_code": self.status_code,
            "events": serialized_events,
        }

    def __setstate__(self, state):
        """Reconstruct the exception from serialized state."""
        self.pod_name = state["pod_name"]
        self.reason = state["reason"]
        self.status_code = state["status_code"]
        self.events = state["events"]

    @classmethod
    def from_dict(cls, state):
        """Reconstruct the exception from a dictionary state."""
        return cls(
            pod_name=state.get("pod_name", "unknown"),
            reason=state.get("reason", "Unknown"),
            status_code=state.get("status_code", 503),
            events=state.get("events", []),
        )

    @property
    def evicted(self) -> bool:
        """True if pod was evicted (ex: node pressure, preemption)."""
        return self.reason == "Evicted" or any(
            "Evicted" in event["reason"] for event in self.events
        )

    @property
    def oom_killed(self) -> bool:
        """True if pod was evicted due to OOM."""
        return self.reason == "OOMKilled" or any(
            "OOMKilled" in event["reason"] for event in self.events
        )

    def __str__(self):
        events_str = "\n".join(
            f"{e['timestamp']} {e['reason']}: {e['message']}" for e in self.events
        )
        base_exc = (
            f"\nPod Name: {self.pod_name}\n"
            f"Reason: {self.reason}\n"
            f"Status Code: {self.status_code}\n"
        )
        if self.events:
            base_exc += f"Recent Events:\n{events_str}"
        return base_exc


class WorkerMembershipChanged(Exception):
    """Raised when worker pods are added or removed during distributed execution."""

    def __init__(
        self,
        added_ips: set = None,
        removed_ips: set = None,
        previous_ips: set = None,
        current_ips: set = None,
        message: str = None,
    ):
        # Support both explicit construction and reconstruction from message
        if message and not (added_ips or removed_ips):
            import ast

            # Reconstruct from message
            import re

            self.added_ips = set()
            self.removed_ips = set()
            self.previous_ips = set()
            self.current_ips = set()

            if "removed during execution:" in message:
                match = re.search(r"removed during execution: ({.*?})", message)
                if match:
                    self.removed_ips = ast.literal_eval(match.group(1))
            elif "added during execution:" in message:
                match = re.search(r"added during execution: ({.*?})", message)
                if match:
                    self.added_ips = ast.literal_eval(match.group(1))
        else:
            # Normal construction
            self.added_ips = added_ips or set()
            self.removed_ips = removed_ips or set()
            self.previous_ips = previous_ips or set()
            self.current_ips = current_ips or set()

            if removed_ips:
                message = f"Critical: {len(removed_ips)} worker(s) removed during execution: {removed_ips}"
            elif added_ips:
                message = f"Warning: {len(added_ips)} worker(s) added during execution: {added_ips}"
            else:
                message = "Worker membership changed"

        super().__init__(message)

    @property
    def is_critical(self) -> bool:
        """Returns True if workers were removed (critical for training)."""
        return bool(self.removed_ips)

    def __getstate__(self):
        """Serialize the exception state."""
        return {
            "message": str(self),
            "added_ips": list(self.added_ips),
            "removed_ips": list(self.removed_ips),
            "previous_ips": list(self.previous_ips),
            "current_ips": list(self.current_ips),
        }

    @classmethod
    def from_dict(cls, data):
        """Reconstruct from serialized state."""
        return cls(
            added_ips=set(data.get("added_ips", [])),
            removed_ips=set(data.get("removed_ips", [])),
            previous_ips=set(data.get("previous_ips", [])),
            current_ips=set(data.get("current_ips", [])),
        )


class StreamType(str, enum.Enum):
    CLI = "cli"
    HTTP_CLIENT = "http_client"


def clean_and_validate_k8s_name(name: str, allow_full_length: bool = True) -> str:
    """Clean and validate a name for K8s compatibility.

    Args:
        name: The name to clean and validate
        allow_full_length: If True, allows and intelligently trims full pod names to 63 chars,
                          preserving k8s-generated portions.
                          If False, limits to 40 chars to leave room for k8s suffixes.
    """
    max_k8s_name_length = 63  # max length allowed by k8s
    max_base_name_length = (
        40  # max module name length to account for added k8s suffixes
    )
    # Regex to comply with k8s service name requirements
    cleaned_name = re.sub(r"[^a-z0-9-]|^[-]|[-]$", "", name.lower())
    if not cleaned_name:
        raise ValueError("Name must contain at least one alphanumeric character.")

    max_length = max_k8s_name_length if allow_full_length else max_base_name_length

    if len(cleaned_name) > max_length:
        if not allow_full_length:
            # For a user provided module name, raise an exception
            error_msg = (
                f"Name length {len(cleaned_name)} exceeds {max_length} characters. "
                "Must leave room for Kubernetes-added suffixes."
            )
            raise ValueError(error_msg)

        match = re.search(r"(-\d+)?-deployment-[a-z0-9]+-[a-z0-9]+", cleaned_name)
        if match:
            k8s_part = match.group(0)
            k8s_start_idx = match.start()

            prefix = cleaned_name[:k8s_start_idx]
            suffix = cleaned_name[k8s_start_idx + len(k8s_part) :]

            total_excess = len(cleaned_name) - max_length

            # If we need to trim, handle each part
            if total_excess > 0:
                # Handle prefix trimming
                if prefix:
                    segments = prefix.split("-")
                    while (
                        len("-".join(segments)) + len(k8s_part) + len(suffix)
                        > max_length
                    ):
                        if len(segments) > 1:
                            segments.pop()
                        else:
                            segments[0] = segments[0][:-1]
                    prefix = "-".join(segments)

                # Handle suffix trimming if still needed
                remaining_length = max_length - (len(prefix) + len(k8s_part))
                if remaining_length > 0:
                    suffix_segments = suffix.split("-")
                    clean_segments = []
                    current_length = 0
                    for seg in suffix_segments:
                        # Only add segment if it's at least 2 chars so the name doesn't look cut off
                        if (
                            len(seg) >= 2
                            and current_length + len(seg) + 1 <= remaining_length
                        ):
                            clean_segments.append(seg)
                            current_length += len(seg) + 1
                    suffix = "-".join(clean_segments)
                else:
                    suffix = ""

            cleaned_name = (
                (prefix + "-" if prefix else "")
                + k8s_part
                + ("-" + suffix if suffix else "")
            )

    return cleaned_name


def is_running_in_kubernetes():
    """
    Determines if the current Python process is running inside a Kubernetes pod.

    Returns:
        bool: True if running in Kubernetes, False otherwise
    """
    # Method 1: Check for Kubernetes service environment variables
    if os.environ.get("KUBERNETES_SERVICE_HOST") is not None:
        return True

    # Method 2: Check for the existence of the Kubernetes service account token file
    if os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount/token"):
        return True

    return False


def _get_rendered_template(
    template_file: str, template_dir: str, **template_vars
) -> str:
    """Helper function to set up and render a template."""
    template_loader = jinja2.FileSystemLoader(searchpath=template_dir)
    template_env = jinja2.Environment(
        loader=template_loader,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
        enable_async=False,
        autoescape=False,
    )
    template = template_env.get_template(template_file)
    return template.render(**template_vars)


def load_template(template_file: str, template_dir: str, **template_vars) -> dict:
    """Load and render a single YAML document template."""
    rendered = _get_rendered_template(template_file, template_dir, **template_vars)
    return yaml.safe_load(rendered)


def load_multi_yaml_template(
    template_file: str, template_dir: str, **template_vars
) -> dict:
    """Load and render a multi-document YAML template."""
    rendered = _get_rendered_template(template_file, template_dir, **template_vars)
    return {"items": list(yaml.safe_load_all(rendered))}


def generate_unique_request_id(endpoint: str, timestamp: str) -> str:
    """Generates a unique request id, based on the method/function endpoint and the call timestamp"""
    raw = f"{endpoint}_{timestamp}"
    unique_id = hashlib.sha256(raw.encode()).hexdigest()[:10]
    return unique_id


def print_log_stream_client(message, last_timestamp, print_pod_name: bool = False):
    formatter = ServerLogsFormatter()
    if message.get("streams"):
        for stream in message["streams"]:
            pod_name = f'({stream.get("stream").get("pod")}) ' if print_pod_name else ""
            for value in stream["values"]:
                # Skip if we've already seen this timestamp
                if last_timestamp is not None and value[0] <= last_timestamp:
                    continue
                last_timestamp = value[0]

                log_line = json.loads(value[1])
                log_name = log_line.get("name")
                if log_name == "print_redirect":
                    message = log_line.get("message")
                    print(
                        f"{pod_name}{formatter.start_color}{message}{formatter.reset_color}"
                    )
                elif log_name != "uvicorn.access":
                    formatted_log = f"{pod_name}{log_line.get('asctime')} | {log_line.get('levelname')} | {log_line.get('message')}"
                    print(
                        f"{formatter.start_color}{formatted_log}{formatter.reset_color}"
                    )
    return last_timestamp


def print_log_stream_cli(message, last_timestamp, print_pod_name: bool = False):
    if message.get("streams"):
        for stream in message["streams"]:
            pod_name = f'({stream.get("stream").get("pod")}) ' if print_pod_name else ""
            for value in stream["values"]:
                # Skip if we've already seen this timestamp
                if last_timestamp is not None and value[0] <= last_timestamp:
                    continue
                last_timestamp = value[0]
                log_line = value[1]
                try:
                    log_line = json.loads(log_line)
                    log_name = log_line.get("name")
                    if log_name == "print_redirect":
                        continue
                        # the print output will be printed in line 250. We need the "print_redirect"
                        # log type only for log streaming in the http client, so we could filter out
                        # the print outputs for a specific request ID. For the CLI --follow option, we
                        # print all logs, so at the moment we don't need to filter by request_id.
                    elif log_name != "uvicorn.access":
                        formatted_log = f"({pod_name}{log_line.get('asctime')} | {log_line.get('levelname')} | {log_line.get('message')}".strip()
                        print(formatted_log)
                except json.JSONDecodeError:
                    print(log_line.strip())

    return last_timestamp


async def stream_logs_websocket_helper(
    uri,
    stop_event,
    stream_type: StreamType = StreamType.HTTP_CLIENT,
    print_pod_name: bool = False,
):
    """Stream logs using Loki's websocket tail endpoint"""
    websocket = None
    try:
        # Track the last timestamp we've seen to avoid duplicates
        last_timestamp = None
        # Track when we should stop
        stop_time = None

        # Add timeout to prevent hanging connections
        websocket = await websockets.connect(
            uri,
            close_timeout=10,  # Max time to wait for close handshake
            ping_interval=20,  # Send ping every 20 seconds
            ping_timeout=10,  # Wait 10 seconds for pong
        )
        try:
            while True:
                # If stop event is set, start counting down
                if stop_event.is_set() and stop_time is None:
                    stop_time = time.time() + 2  # 2 seconds grace period

                # If we're past the grace period, exit
                if stop_time is not None and time.time() > stop_time:
                    break

                try:
                    # Use shorter timeout during grace period
                    timeout = 0.1 if stop_time is not None else 1.0
                    message = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                    try:
                        message = json.loads(message)
                    except json.JSONDecodeError:
                        message = message

                    if stream_type == StreamType.HTTP_CLIENT:
                        last_timestamp = print_log_stream_client(
                            message, last_timestamp, print_pod_name
                        )
                    elif stream_type == StreamType.CLI:
                        last_timestamp = print_log_stream_cli(
                            message, last_timestamp, print_pod_name
                        )
                except asyncio.TimeoutError:
                    # Timeout is expected, just continue the loop
                    continue
                except websockets.exceptions.ConnectionClosed as e:
                    logger.debug(f"WebSocket connection closed: {str(e)}")
                    break
        finally:
            if websocket:
                try:
                    # Use wait_for to prevent hanging on close
                    await asyncio.wait_for(websocket.close(), timeout=1.0)
                except (asyncio.TimeoutError, Exception):
                    pass
    except Exception as e:
        logger.error(f"Error in websocket stream: {e}")
    finally:
        # Ensure websocket is closed even if we didn't enter the try block
        if websocket:
            try:
                # Use wait_for to prevent hanging on close
                await asyncio.wait_for(websocket.close(), timeout=1.0)
            except (asyncio.TimeoutError, Exception):
                pass


def clear_debugging_sessions():
    """Clear any existing debugging sessions when a module is redeployed or pod is terminated."""
    try:
        import web_pdb

        if web_pdb.WebPdb.active_instance is not None:
            logger.info("Clearing existing debugging session")
            try:
                web_pdb.WebPdb.active_instance.remove_trace()
            except Exception as e:
                logger.warning(f"Error removing trace: {e}")
            web_pdb.WebPdb.active_instance = None

    except ImportError:
        # web_pdb not installed, nothing to clean up
        pass
    except Exception as e:
        logger.warning(f"Error clearing debugging session: {e}")


# Register cleanup function to run at exit
atexit.register(clear_debugging_sessions)


def deep_breakpoint(debug_port: int = DEFAULT_DEBUG_PORT):
    """
    Similar to Python's built-in `breakpoint()`, but can be used deep inside distributed code. For SPMD-style
    distributed code like PyTorch, be sure to only call this from one process (e.g. the rank 0 process) to avoid
    blocking all processes in the distributed group.
    """
    # Check if madbg is installed, if not, install it
    try:
        import web_pdb
    except ImportError:
        install_cmd = "uv pip install --system web-pdb"
        import subprocess

        print("Pdb debugger not found, installing it...")
        # Run the install command and propagate logs
        subprocess.run(install_cmd, shell=True, check=True, text=True)
        print("Pdb installed successfully.")

    print(
        "Distributed breakpoint activated. To attach a debugger, run the following command:"
    )
    print(
        f"kt debug {os.environ['POD_NAME']} --port {debug_port} --namespace {os.environ['POD_NAMESPACE']}"
    )

    import web_pdb

    pdb = web_pdb.WebPdb.active_instance
    try:
        if pdb is None:
            pdb = web_pdb.WebPdb(host="", port=debug_port, patch_stdstreams=False)
        else:
            # If the debugger is still attached reset trace to a new location
            pdb.remove_trace()

        # Set the frame to the caller's frame
        pdb.set_trace(sys._getframe(1))  # pylint: disable=protected-access
    except Exception as e:
        # Only clean up if there was an error setting up the debugger
        if pdb:
            pdb.remove_trace()
            web_pdb.WebPdb.active_instance = None
        raise e


def wait_for_app_start(
    port, health_check: str, process: subprocess.Popen, timeout: int = 60
):
    """
    Wait until the app is ready. If health_check if provided, will send HTTP requests to check, otherwise
    will wait until something is listening on the port.
    """
    host = LOCALHOST
    port = int(port)
    logger.debug(f"Trying to connect to http://{host}:{port}{health_check or ''}")
    start_time = time.time()

    if health_check:
        if not health_check.startswith("/"):
            health_check = f"/{health_check}"
        url = f"http://{LOCALHOST}:{port}{health_check}"
        while time.time() - start_time < timeout:
            if process.poll() is not None and process.poll() != 0:
                raise RuntimeError(f"App exited with code {process.poll()}")
            try:
                response = httpx.get(url)
                if response.status_code == 200:
                    return True
            except httpx.ConnectError:
                pass
            time.sleep(0.5)
        raise TimeoutError(
            f"App did not become healthy on {url} within {timeout} seconds"
        )
    else:
        # Fallback to socket check
        while time.time() - start_time < timeout:
            if process.poll() is not None and process.poll() != 0:
                raise RuntimeError(f"App exited with code {process.poll()}")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                try:
                    sock.connect((host, port))
                    return True
                except (ConnectionRefusedError, socket.timeout):
                    time.sleep(0.5)
        raise TimeoutError(
            f"Failed to detect open port {port} for app {url} within {timeout} seconds"
        )


def _serialize_body(body: dict, serialization: str):
    if body is None:
        return {}

    # We only serialize args and kwargs, other settings like "workers" and "restart_procs" are needed inside
    # the http_server, outside the serialization boundary (e.g. the distributed processes)
    # We break them out here as separate params
    body = body or {}

    for kwarg in MAGIC_CALL_KWARGS:
        if kwarg in body.get("kwargs", {}):
            body[kwarg] = body["kwargs"].pop(kwarg)

    if serialization == "pickle":
        args_data = {"args": body.pop("args"), "kwargs": body.pop("kwargs")}
        pickled_args = pickle.dumps(args_data or {})
        encoded_args = base64.b64encode(pickled_args).decode("utf-8")
        body["data"] = encoded_args
        return body
    return body or {}


def _deserialize_response(response, serialization: str):
    if serialization == "pickle":
        response_data = response.json()
        if isinstance(response_data, list):
            # If this is a response from an spmd call, it's a list of serialized dicts
            unpickled_results = []
            for resp in response_data:
                if "data" in resp:
                    encoded_result = resp["data"]
                    pickled_result = base64.b64decode(encoded_result.encode("utf-8"))
                    resp = pickle.loads(pickled_result)
                unpickled_results.append(resp)
            return unpickled_results
        if "data" in response_data:
            encoded_result = response_data["data"]
            pickled_result = base64.b64decode(encoded_result.encode("utf-8"))
            return pickle.loads(pickled_result)
        return response_data
    return response.json()
