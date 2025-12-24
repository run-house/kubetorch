import asyncio
import hashlib
import json
import os
import threading
import time
import urllib.parse
from collections import defaultdict, deque
from datetime import datetime
from typing import Literal, Union

import httpx
import requests
import websockets

from kubetorch.globals import config, DebugConfig, LoggingConfig, MetricsConfig, service_url
from kubetorch.logger import get_logger

from kubetorch.servers.http.utils import (
    _deserialize_response,
    _serialize_body,
    generate_unique_request_id,
    request_id_ctx_var,
)

from kubetorch.serving.constants import DEFAULT_DEBUG_PORT, DEFAULT_NGINX_PORT
from kubetorch.utils import ColoredFormatter, extract_host_port, ServerLogsFormatter

logger = get_logger(__name__)

# Log level priority for filtering
LOG_LEVEL_PRIORITY = {
    "debug": 0,
    "info": 1,
    "warning": 2,
    "error": 3,
    "critical": 4,
}


class LogDeduplicator:
    """Sliding window deduplication using hash sets.

    Prevents duplicate logs from being printed when Loki returns the same log
    multiple times (e.g., logs with identical nanosecond timestamps).

    Uses a sliding window of hash sets to efficiently track recently seen logs
    while automatically evicting old entries to prevent memory growth.
    """

    def __init__(self, window_intervals: int = 5):
        """Initialize the deduplicator.

        Args:
            window_intervals: Number of intervals to keep in the sliding window.
                With 0.5s grace_poll_timeout, 5 intervals = 2.5s window.
        """
        self.windows: deque[set] = deque(maxlen=window_intervals)
        self.windows.append(set())

    def rotate(self):
        """Rotate to a new interval window. Call this on each timeout."""
        self.windows.appendleft(set())  # Newest at front, oldest auto-evicted by maxlen

    def is_duplicate(self, raw_log: str) -> bool:
        """Check if log was seen recently; if not, record it.

        Args:
            raw_log: The raw log string (before JSON deserialization)

        Returns:
            True if this log was already seen, False if it's new
        """
        # MD5 is fast and sufficient for deduplication (not security)
        h = hashlib.md5(raw_log.encode(), usedforsecurity=False).digest()

        # Check newest windows first (most duplicates hit immediately)
        for window in self.windows:
            if h in window:
                return True

        # Not seen - add to current (newest) window
        self.windows[0].add(h)
        return False


class CustomResponse(httpx.Response):
    def raise_for_status(self):
        """Raises parsed server errors or HTTPError for other status codes"""
        if not 400 <= self.status_code < 600:
            return

        if "application/json" in self.headers.get("Content-Type", ""):
            try:
                error_data = self.json()
                if all(k in error_data for k in ["error_type", "message", "traceback", "pod_name"]):
                    error_type = error_data["error_type"]
                    message = error_data.get("message", "")
                    traceback = error_data["traceback"]
                    pod_name = error_data["pod_name"]
                    error_state = error_data.get("state", {})  # Optional serialized state

                    # Try to use the actual exception class if it exists
                    exc = None
                    error_class = None

                    # Import the exception registry
                    try:
                        from kubetorch import EXCEPTION_REGISTRY
                    except ImportError:
                        EXCEPTION_REGISTRY = {}

                    # First check if it's a Python builtin exception
                    import builtins

                    if hasattr(builtins, error_type):
                        error_class = getattr(builtins, error_type)
                    # Otherwise try to use from the kubetorch registry
                    elif error_type in EXCEPTION_REGISTRY:
                        error_class = EXCEPTION_REGISTRY[error_type]

                    if error_class:
                        try:
                            # First try to reconstruct from state if available
                            if error_state and hasattr(error_class, "from_dict"):
                                exc = error_class.from_dict(error_state)
                            # Otherwise try simple construction with message
                            else:
                                exc = error_class(message)
                        except Exception as e:
                            logger.debug(f"Could not reconstruct {error_type}: {e}, will use dynamic type")
                            # Fall back to dynamic creation
                            pass

                    # If we couldn't create the actual exception, fall back to dynamic type creation
                    if not exc:

                        def create_str_method(remote_traceback):
                            def __str__(self):
                                cleaned_traceback = remote_traceback.encode().decode("unicode_escape")
                                return f"{self.args[0]}\n\n{cleaned_traceback}"

                            return __str__

                        # Create the exception class with the custom __str__
                        error_class = type(
                            error_type,
                            (Exception,),
                            {"__str__": create_str_method(traceback)},
                        )

                        exc = error_class(message)

                    # Always add remote_traceback and pod_name
                    exc.remote_traceback = traceback
                    exc.pod_name = pod_name

                    # Wrap the exception to display remote traceback
                    # Create a new class that inherits from the original exception
                    # and overrides __str__ to include the remote traceback
                    class RemoteException(exc.__class__):
                        def __str__(self):
                            # Get the original message
                            original_msg = super().__str__()
                            # Clean up the traceback
                            cleaned_traceback = self.remote_traceback.encode().decode("unicode_escape")
                            return f"{original_msg}\n\n{cleaned_traceback}"

                    # Create wrapped instance without calling __init__
                    wrapped_exc = RemoteException.__new__(RemoteException)
                    # Copy all attributes from the original exception
                    wrapped_exc.__dict__.update(exc.__dict__)
                    # Set the exception args for proper display
                    wrapped_exc.args = (str(exc),)
                    raise wrapped_exc

            except Exception as e:
                # Catchall for errors during exception handling above
                if isinstance(e, RemoteException):
                    # If we caught a RemoteException, it was packaged properly
                    raise
                import httpx

                raise httpx.HTTPStatusError(
                    f"{self.status_code} {self.text}",
                    request=self.request,
                    response=self,
                )
        else:
            # Don't log 502 errors - they're expected during startup as nginx DNS updates
            if self.status_code != 502:
                logger.debug(f"Non-JSON error body: {self.text[:100]}")
            super().raise_for_status()


class CustomSession(httpx.Client):
    def __init__(self):
        limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
        super().__init__(timeout=None, limits=limits)

    def __del__(self):
        self.close()

    def request(self, *args, **kwargs):
        response = super().request(*args, **kwargs)
        response.__class__ = CustomResponse
        return response


class CustomAsyncClient(httpx.AsyncClient):
    def __init__(self):
        limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
        super().__init__(timeout=None, limits=limits)

    async def request(self, *args, **kwargs):
        response = await super().request(*args, **kwargs)
        response.__class__ = CustomResponse
        return response


class HTTPClient:
    """Client for making HTTP requests to a remote service. Port forwards are shared between client
    instances. Each port forward instance is cleaned up when the last reference is closed."""

    def __init__(self, base_url, compute, service_name):
        self.compute = compute
        self.service_name = service_name
        self.base_url = base_url.rstrip("/")
        self.session = CustomSession()
        self._async_client = None
        self._async_client_loop = None  # Track which event loop the client was created on

    def __del__(self):
        self.close()

    def close(self):
        """Close the async HTTP client to prevent resource leaks."""
        if self._async_client:
            try:
                # Close the async client if it's still open
                if not self._async_client.is_closed:
                    # Use asyncio.run if we're in a sync context, otherwise schedule the close
                    try:
                        import asyncio

                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # If we're in an async context, schedule the close
                            loop.create_task(self._async_client.aclose())
                        else:
                            # If we're in a sync context, run the close
                            asyncio.run(self._async_client.aclose())
                    except RuntimeError:
                        # No event loop available, try to create one
                        try:
                            asyncio.run(self._async_client.aclose())
                        except Exception:
                            pass
            except Exception as e:
                logger.debug(f"Error closing async client: {e}")
            finally:
                self._async_client = None

        # Close the session as well
        if self.session:
            try:
                self.session.close()
            except Exception as e:
                logger.debug(f"Error closing session: {e}")
            finally:
                self.session = None

    @property
    def local_port(self):
        """Local port to open the port forward connection with the proxy service. This should match the client port used
        to set the URL of the service in the Compute class."""
        if self.compute:
            return self.compute.client_port()
        return DEFAULT_NGINX_PORT

    @property
    def async_session(self):
        """Get or create async HTTP client.

        Creates a new client if none exists or if the existing client's event loop is closed.
        This handles cases like Jupyter notebooks or applications that restart their event loop.
        """
        if self._async_client is not None:
            # Check if the client can still be used with the current event loop
            try:
                current_loop = asyncio.get_running_loop()
                # httpx.AsyncClient is bound to the event loop it was created on.
                # If the current loop is different, or the old loop is closed, we need a new client.
                if self._async_client.is_closed:
                    self._async_client = None
                elif self._async_client_loop is not None:
                    if self._async_client_loop is not current_loop or self._async_client_loop.is_closed():
                        self._async_client = None
            except RuntimeError:
                # No running loop - the cached client can't be used
                self._async_client = None

        if self._async_client is None:
            try:
                self._async_client_loop = asyncio.get_running_loop()
            except RuntimeError:
                self._async_client_loop = None
            self._async_client = CustomAsyncClient()
        return self._async_client

    def _prepare_request(
        self,
        endpoint: str,
        stream_logs: bool,
        stream_metrics: Union[bool, MetricsConfig, None],
        headers: dict,
        pdb: Union[bool, int],
        serialization: str,
        logging_config: LoggingConfig,
        debug: Union[bool, DebugConfig, None] = None,
    ):
        metrics_config = MetricsConfig()
        if isinstance(stream_metrics, MetricsConfig):
            metrics_config = stream_metrics
            stream_metrics = True
        elif stream_metrics is None:
            stream_metrics = config.stream_metrics if config.stream_metrics is not None else True

        # Handle debug parameter (new) or pdb parameter (backward compatibility)
        # Add deprecation warning for pdb parameter
        if pdb:
            import warnings

            warnings.warn(
                "The 'pdb' parameter is deprecated and will be removed in a future version. "
                "Please use 'debug' parameter instead.",
                DeprecationWarning,
                stacklevel=3,
            )

        # debug parameter takes precedence over pdb
        if debug is not None:
            if isinstance(debug, DebugConfig):
                debug_port = debug.port
                debug_mode = debug.mode
            elif isinstance(debug, bool):
                if debug:
                    debug_port = DEFAULT_DEBUG_PORT
                    # Get debug mode from environment, default to "pdb"
                    debug_mode = os.getenv("KT_DEBUG_MODE", "pdb").lower()
                else:
                    debug_port = None
                    debug_mode = None
            else:
                raise ValueError(
                    f"debug parameter must be a bool or DebugConfig instance, got {type(debug).__name__}. "
                    "Use debug=True or debug=kt.DebugConfig(port=..., mode=...) instead."
                )
        elif pdb:
            # Backward compatibility with pdb parameter
            debug_port = DEFAULT_DEBUG_PORT if isinstance(pdb, bool) else pdb
            debug_mode = os.getenv("KT_DEBUG_MODE", "pdb").lower()
        else:
            debug_port = None
            debug_mode = None

        if debug_port:
            endpoint += f"?debug_port={debug_port}"
            if debug_mode:
                endpoint += f"&debug_mode={debug_mode}"

        request_id = request_id_ctx_var.get("-")
        if request_id == "-":
            timestamp = str(time.time())
            request_id = generate_unique_request_id(endpoint=endpoint, timestamp=timestamp)

        headers = headers or {}
        headers.update({"X-Request-ID": request_id, "X-Serialization": serialization})

        stop_event = threading.Event()
        log_thread = None
        if stream_logs:
            log_thread = threading.Thread(target=self.stream_logs, args=(request_id, stop_event, logging_config))
            log_thread.daemon = True
            log_thread.start()

        if stream_metrics:
            metrics_thread = threading.Thread(target=self.stream_metrics, args=(stop_event, metrics_config))
            metrics_thread.daemon = True
            metrics_thread.start()
        else:
            metrics_thread = None

        return endpoint, headers, stop_event, log_thread, metrics_thread, request_id

    def _prepare_request_async(
        self,
        endpoint: str,
        stream_logs: bool,
        stream_metrics: Union[bool, MetricsConfig, None],
        headers: dict,
        pdb: Union[bool, int],
        serialization: str,
        logging_config: LoggingConfig,
        debug: Union[bool, DebugConfig, None] = None,
    ):
        """Async version of _prepare_request that uses asyncio.Event and tasks instead of threads"""
        metrics_config = MetricsConfig()
        if isinstance(stream_metrics, MetricsConfig):
            metrics_config = stream_metrics
            stream_metrics = True
        elif stream_metrics is None:
            stream_metrics = config.stream_metrics if config.stream_metrics is not None else True

        # Handle debug parameter (new) or pdb parameter (backward compatibility)
        # Add deprecation warning for pdb parameter
        if pdb:
            import warnings

            warnings.warn(
                "The 'pdb' parameter is deprecated and will be removed in a future version. "
                "Please use 'debug' parameter instead.",
                DeprecationWarning,
                stacklevel=3,
            )

        # debug parameter takes precedence over pdb
        if debug is not None:
            if isinstance(debug, DebugConfig):
                debug_port = debug.port
                debug_mode = debug.mode
            elif isinstance(debug, bool):
                if debug:
                    debug_port = DEFAULT_DEBUG_PORT
                    debug_mode = os.getenv("KT_DEBUG_MODE", "pdb").lower()
                else:
                    debug_port = None
                    debug_mode = None
            else:
                raise ValueError(
                    f"debug parameter must be a bool or DebugConfig instance, got {type(debug).__name__}. "
                    "Use debug=True or debug=kt.DebugConfig(port=..., mode=...) instead."
                )
        elif pdb:
            # Backward compatibility with pdb parameter
            debug_port = DEFAULT_DEBUG_PORT if isinstance(pdb, bool) else pdb
            debug_mode = os.getenv("KT_DEBUG_MODE", "pdb").lower()
        else:
            debug_port = None
            debug_mode = None

        if debug_port:
            endpoint += f"?debug_port={debug_port}"
            if debug_mode:
                endpoint += f"&debug_mode={debug_mode}"

        request_id = request_id_ctx_var.get("-")
        if request_id == "-":
            timestamp = str(time.time())
            request_id = generate_unique_request_id(endpoint=endpoint, timestamp=timestamp)

        headers = headers or {}
        headers.update({"X-Request-ID": request_id, "X-Serialization": serialization})

        stop_event = asyncio.Event()
        log_task = None
        if stream_logs:
            log_task = asyncio.create_task(self.stream_logs_async(request_id, stop_event, logging_config))

        metrics_task = None
        if stream_metrics:
            metrics_task = asyncio.create_task(self.stream_metrics_async(request_id, stop_event, metrics_config))

        return endpoint, headers, stop_event, log_task, metrics_task, request_id

    def _make_request(self, method, endpoint, timeout=None, **kwargs):
        # Allow per-request timeout override
        if timeout is not None:
            kwargs["timeout"] = timeout
        response: httpx.Response = getattr(self.session, method)(endpoint, **kwargs)
        response.raise_for_status()
        return response

    async def _make_request_async(self, method, endpoint, **kwargs):
        """Async version of _make_request."""
        response = await getattr(self.async_session, method)(endpoint, **kwargs)
        response.raise_for_status()
        return response

    # ----------------- Stream Helpers ----------------- #
    def _should_display_log(self, log_name: str, log_level: str, log_config: LoggingConfig) -> bool:
        """Determine if a log should be displayed based on config filters.

        Args:
            log_name: The logger name (e.g., "print_redirect", "uvicorn.access")
            log_level: The log level string (e.g., "INFO", "ERROR")
            log_config: The LoggingConfig with filter settings

        Returns:
            True if the log should be displayed, False if it should be filtered out
        """
        # print_redirect logs are always shown (user print statements)
        if log_name == "print_redirect":
            return True

        # Filter system logs (uvicorn, etc.) unless explicitly included
        if not log_config.include_system_logs and log_name in ("uvicorn.access", "uvicorn.error", "uvicorn"):
            return False

        # Filter by log level
        if log_level:
            log_level_lower = log_level.lower()
            min_level = log_config.level.lower()
            if LOG_LEVEL_PRIORITY.get(log_level_lower, 1) < LOG_LEVEL_PRIORITY.get(min_level, 1):
                return False

        return True

    async def _stream_logs_websocket(
        self,
        request_id,
        stop_event: Union[threading.Event, asyncio.Event],
        port: int,
        log_config: LoggingConfig,
        host: str = "localhost",
    ):
        """Stream logs using Loki's websocket tail endpoint.

        Args:
            request_id: The request ID to filter logs for
            stop_event: Event to signal when to start grace period shutdown
            port: The port to connect to
            host: The host to connect to
            log_config: Configuration for log streaming behavior
        """
        formatter = ServerLogsFormatter()
        deduplicator = LogDeduplicator(window_intervals=5)  # ~2.5s dedup window
        websocket = None

        try:
            # Query using labels set by LogCapture (service, namespace, request_id)
            namespace = self.compute.namespace
            query = f'{{service="{self.service_name}", namespace="{namespace}", request_id="{request_id}"}}'
            encoded_query = urllib.parse.quote_plus(query)
            uri = f"ws://{host}:{port}/loki/{namespace}/api/v1/tail?query={encoded_query}"
            # Track when we should stop
            stop_time = None

            # Add timeout to prevent hanging connections
            logger.debug(f"Streaming logs with tail query {uri}")
            websocket = await websockets.connect(
                uri,
                close_timeout=10,  # Max time to wait for close handshake
                ping_interval=20,  # Send ping every 20 seconds
                ping_timeout=10,  # Wait 10 seconds for pong
            )
            try:
                while True:
                    # If stop event is set, start counting down
                    # Handle both threading.Event and asyncio.Event
                    is_stop_set = stop_event.is_set() if hasattr(stop_event, "is_set") else stop_event.is_set()
                    if is_stop_set and stop_time is None:
                        stop_time = time.time() + log_config.grace_period

                    # If we're past the grace period, exit
                    if stop_time is not None and time.time() > stop_time:
                        break

                    try:
                        # Use shorter timeout during grace period
                        timeout = log_config.grace_poll_timeout if stop_time is not None else log_config.poll_timeout
                        message = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                        data = json.loads(message)

                        if data.get("streams"):
                            for stream in data["streams"]:
                                labels = stream["stream"]
                                # Get pod name from Loki labels
                                pod_name = labels.get("pod", "")

                                for value in stream["values"]:
                                    raw_log = value[1]

                                    # Deduplicate using hash-based sliding window
                                    if deduplicator.is_duplicate(raw_log):
                                        continue

                                    # Parse JSON message from LogCapture
                                    try:
                                        log_line = json.loads(raw_log)
                                        log_name = log_line.get("name", "")
                                        log_message = log_line.get("message", "")
                                        log_level = log_line.get("levelname", "INFO")
                                        log_asctime = log_line.get("asctime", "")
                                    except json.JSONDecodeError:
                                        # Fallback for plain text (shouldn't happen with LogCapture)
                                        log_name = ""
                                        log_message = raw_log
                                        log_level = labels.get("level", "INFO")
                                        log_asctime = ""

                                    # Apply log level and system log filters
                                    if not self._should_display_log(log_name, log_level, log_config):
                                        continue

                                    # Format and print the log
                                    if log_config.include_name and pod_name:
                                        prefix = f"({pod_name}) "
                                    else:
                                        prefix = ""

                                    # Strip trailing whitespace/control chars to prevent line overwrites
                                    log_message = log_message.rstrip() if log_message else ""

                                    # Format differently for print statements vs logger output
                                    if log_name == "print_redirect":
                                        print(
                                            f"{formatter.start_color}{prefix}{log_message}{formatter.reset_color}",
                                            flush=True,
                                        )
                                    else:
                                        formatted_log = f"{prefix}{log_asctime} | {log_level} | {log_message}"
                                        print(
                                            f"{formatter.start_color}{formatted_log}{formatter.reset_color}",
                                            flush=True,
                                        )

                    except asyncio.TimeoutError:
                        # Timeout is expected - rotate deduplication window and continue
                        deduplicator.rotate()
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
            # Ensure websocket is closed even if we didn't enter the context
            if websocket:
                try:
                    # Use wait_for to prevent hanging on close
                    await asyncio.wait_for(websocket.close(), timeout=1.0)
                except (asyncio.TimeoutError, Exception):
                    pass

    async def _stream_events_websocket(
        self,
        stop_event: Union[threading.Event, asyncio.Event],
        port: int,
        log_config: LoggingConfig,
        host: str = "localhost",
        start_time_ns: int = None,
    ):
        """Stream K8s events using Loki's websocket tail endpoint.

        Events are pushed by the controller's event watcher with labels:
        - job="kubetorch-events"
        - name=<resource_name>
        - kind=<Pod|Deployment|etc>
        - event_type=<Normal|Warning>
        - reason=<OOMKilled|Scheduled|etc>
        """
        websocket = None
        shown_messages = set()

        # Use provided start time or current time (only show events from now)
        if start_time_ns is None:
            start_time_ns = int(time.time() * 1e9)

        try:
            namespace = self.compute.namespace
            # Query events for resources matching this service name
            event_query = f'{{job="kubetorch-events", namespace="{namespace}", name=~"{self.service_name}.*"}}'
            encoded_query = urllib.parse.quote_plus(event_query)
            # Include start time to only get events from call start (not old OOMs etc)
            uri = f"ws://{host}:{port}/loki/{namespace}/api/v1/tail?query={encoded_query}&start={start_time_ns}"

            stop_time = None
            websocket = await websockets.connect(
                uri,
                close_timeout=10,
                ping_interval=20,
                ping_timeout=10,
            )
            try:
                while True:
                    is_stop_set = stop_event.is_set() if hasattr(stop_event, "is_set") else stop_event.is_set()
                    if is_stop_set and stop_time is None:
                        stop_time = time.time() + log_config.grace_period

                    if stop_time is not None and time.time() > stop_time:
                        break

                    try:
                        timeout = log_config.grace_poll_timeout if stop_time is not None else log_config.poll_timeout
                        message = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                        data = json.loads(message)

                        if data.get("streams"):
                            for stream in data["streams"]:
                                labels = stream["stream"]
                                event_type = labels.get("event_type", "Normal")
                                reason = labels.get("reason", "")

                                # Skip Normal events unless log level includes info
                                log_level = log_config.level.lower() if log_config.level else "info"
                                if log_level in ["warning", "error"] and event_type == "Normal":
                                    continue

                                for value in stream["values"]:
                                    # Filter by timestamp (skip events before call started)
                                    ts_ns = int(value[0])
                                    if ts_ns < start_time_ns:
                                        continue

                                    try:
                                        event_data = json.loads(value[1])
                                        msg = event_data.get("message", value[1])
                                    except json.JSONDecodeError:
                                        msg = value[1]

                                    # Skip expected probe failures
                                    if reason == "Unhealthy" and (
                                        "HTTP probe failed with statuscode: 503" in msg or "Startup probe failed" in msg
                                    ):
                                        continue

                                    # Skip noisy events
                                    ignore_patterns = ("queue-proxy", "failed to get private k8s service endpoints:")
                                    if any(p in msg.lower() for p in ignore_patterns):
                                        continue

                                    # Deduplicate by message content
                                    if msg in shown_messages:
                                        continue
                                    shown_messages.add(msg)

                                    # Format timestamp from event
                                    try:
                                        event_ts = datetime.fromtimestamp(ts_ns / 1e9).strftime("%Y-%m-%d %H:%M:%S")
                                    except Exception:
                                        event_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                                    # Use yellow for warnings, green for normal
                                    if event_type == "Warning":
                                        color = ColoredFormatter.get_color("yellow")
                                    else:
                                        color = ColoredFormatter.get_color("green")
                                    reset = ColoredFormatter.get_color("reset")

                                    # Format like metrics: ({service} events) timestamp | reason: message
                                    prefix = f"({self.service_name} events)"
                                    print(f"{color}{prefix} {event_ts} | {reason}: {msg}{reset}", flush=True)

                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        break
            finally:
                if websocket:
                    try:
                        await asyncio.wait_for(websocket.close(), timeout=1.0)
                    except (asyncio.TimeoutError, Exception):
                        pass
        except Exception as e:
            logger.debug(f"Event stream error (may be expected if no events): {e}")

    def _run_log_stream(self, request_id, stop_event, host, port, log_config: LoggingConfig = None):
        """Helper to run log and event streaming in an event loop"""
        # Set request_id in this thread's context so any logs printed here
        # (e.g., nested service logs) are captured with the correct request_id
        request_id_ctx_var.set(request_id)

        async def run_streams():
            # Run log and event streams in parallel
            await asyncio.gather(
                self._stream_logs_websocket(request_id, stop_event, host=host, port=port, log_config=log_config),
                self._stream_events_websocket(stop_event, host=host, port=port, log_config=log_config),
                return_exceptions=True,  # Don't fail if one stream errors
            )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_streams())
        finally:
            # Cancel all pending tasks to prevent "Task was destroyed but it is pending!" warnings
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            # Wait for all tasks to be cancelled
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

    # ----------------- Metrics Helpers ----------------- #

    def _get_stream_metrics_queries(self, scope: Literal["pod", "resource"], interval: int):
        # lookback window for each Prometheus query
        # For short intervals (1–60s polling): look back ≤ 2 min
        # For slow polling (≥ 1 min): allow up to 5 min lookback
        effective_window = min(max(30, interval * 3), 120 if interval < 60 else 300)
        metric_queries = {}
        if scope == "pod":
            active_pods = self.compute.pod_names()
            if not active_pods:
                logger.warning("No active pods found for service, skipping metrics collection")
                return

            pod_regex = "|".join(active_pods)
            metric_queries = {
                # CPU: seconds of CPU used per second (i.e. cores used)
                # Note: using irate ensures we always capture at least 2 samples in the window
                # https://prometheus.io/docs/prometheus/latest/querying/functions/#irate
                "CPU": f'sum by (pod) (irate(container_cpu_usage_seconds_total{{container!="",pod=~"{pod_regex}"}}[{effective_window}s]))',
                # Memory: Working set in MiB
                "Mem": f'last_over_time(container_memory_working_set_bytes{{container!="",pod=~"{pod_regex}"}}[{effective_window}s]) / 1024 / 1024',
                # GPU metrics from DCGM
                "GPU_SM": f'avg by (pod) (last_over_time(DCGM_FI_DEV_GPU_UTIL{{pod=~"{pod_regex}"}}[{effective_window}s]))',
                "GPUMiB": f'avg by (pod) (last_over_time(DCGM_FI_DEV_FB_USED{{pod=~"{pod_regex}"}}[{effective_window}s]))',
            }

        elif scope == "resource":
            service_name_regex = f"{self.compute.service_name}.+"
            metric_queries = {
                # CPU: Use rate of CPU seconds - cores utilized
                "CPU": f'avg((irate(container_cpu_usage_seconds_total{{container!="",pod=~"{service_name_regex}"}}[{effective_window}s])))',
                # Memory: Working set in MiB
                "Mem": f'avg(last_over_time(container_memory_working_set_bytes{{container!="",pod=~"{service_name_regex}"}}[{effective_window}s]) / 1024 / 1024)',
                # GPU metrics from DCGM
                "GPU_SM": f'avg(last_over_time(DCGM_FI_DEV_GPU_UTIL{{pod=~"{service_name_regex}"}}[{effective_window}s]))',
                "GPUMiB": f'avg(last_over_time(DCGM_FI_DEV_FB_USED{{pod=~"{service_name_regex}"}}[{effective_window}s]))',
            }

        return metric_queries

    def _collect_metrics_common(
        self,
        stop_event,
        http_getter,
        sleeper,
        metrics_config: MetricsConfig,
        is_async: bool = False,
    ):
        """
        Internal shared implementation for collecting and printing live resource metrics
        (CPU, memory, and GPU) for all active pods in the service.

        This function drives both the synchronous (`_collect_metrics`) and asynchronous
        (`_collect_metrics_async`) metric collectors. It repeatedly queries Prometheus for
        metrics related to the service’s pods until the given `stop_event` is set.

        Args:
            stop_event (threading.event or asyncio.Event): A threading.Event or asyncio.Event used to stop collection.
            http_getter (Callable): Callable that fetches Prometheus data — either sync (`requests.get`)
                         or async (`httpx.AsyncClient.get`).
            sleeper (Callable): Callable that sleeps between metric polls — either time.sleep or asyncio.sleep.
            metrics_config (MetricsConfig): User provided configuration controlling metrics collection behavior.
            is_async (bool): If ``True``, runs in async mode (awaits HTTP + sleep calls).
                             If ``False``, runs in blocking sync mode.

        Behavior:
            - Polls Prometheus every 1–5 seconds for CPU, memory, and GPU metrics.
            - Prints a formatted line per pod to stdout.
            - Automatically adapts between synchronous and asynchronous execution modes.

        Note:
            - This function should not be called directly; use `_collect_metrics` or
              `_collect_metrics_async` instead.
            - Stops automatically when `stop_event.set()` is triggered.
        """

        async def maybe_await(obj):
            if is_async and asyncio.iscoroutine(obj):
                return await obj
            return obj

        async def run():
            interval = int(metrics_config.interval)
            metric_queries = self._get_stream_metrics_queries(scope=metrics_config.scope, interval=interval)
            show_gpu = True
            prom_url = f"{service_url()}/prometheus/api/v1/query"

            start_time = time.time()

            while not stop_event.is_set():
                await maybe_await(sleeper(interval))
                pod_data = defaultdict(dict)
                gpu_values = []

                for name, query in metric_queries.items():
                    try:
                        data = await maybe_await(
                            http_getter(
                                prom_url,
                                params={
                                    "query": query,
                                    "lookback_delta": interval,
                                },
                            )
                        )
                        if data.get("status") != "success":
                            continue
                        for result in data["data"]["result"]:
                            m = result["metric"]
                            ts, val = result["value"]
                            pod = m.get("pod", "unknown")
                            val_f = float(val)
                            pod_data[pod][name] = val_f
                            if name in ("GPU%", "GPUMiB"):
                                gpu_values.append(val_f)
                    except Exception as e:
                        logger.error(f"Error loading metrics: {e}")
                        continue

                if not gpu_values:
                    show_gpu = False

                if pod_data:
                    for pod, vals in sorted(pod_data.items()):
                        mem = vals.get("Mem", 0.0)
                        cpu_cores = vals.get("CPU", 0.0)
                        gpu = vals.get("GPU_SM", 0.0)
                        gpumem = vals.get("GPUMiB", 0.0)
                        now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        service_or_pod_info = (
                            f"({pod} metrics) " if metrics_config.scope == "pod" else f"({self.service_name} metrics) "
                        )
                        line = f"{service_or_pod_info}{now_ts} | " f"CPU: {cpu_cores:.2f} | Memory: {mem:.3f}MiB"
                        if show_gpu:
                            line += f" | GPU SM: {gpu:.2f}% | GPU Memory: {gpumem:.3f}MiB"
                        metrics_color = ColoredFormatter.get_color("blue")
                        reset_color = ColoredFormatter.get_color("reset")
                        print(f"{metrics_color}{line}{reset_color}", flush=True)

                elapsed = time.time() - start_time
                sleep_interval = max(interval, int(min(60, 1 + elapsed / 30)))
                await maybe_await(sleeper(sleep_interval))

        # run sync or async depending on mode
        if is_async:
            return run()
        else:
            asyncio.run(run())

    def _collect_metrics(self, stop_event, http_getter, sleeper, metrics_config):
        """
        Synchronous metrics collector.

        Invokes `_collect_metrics_common` in blocking mode to stream metrics. Designed for use in background threads
        where the event loop is *not* running (e.g. standard Python threads).

        Args:
            stop_event: threading.Event to signal termination of metric collection.
            http_getter: Synchronous callable that fetches Prometheus query results.
            sleeper: Blocking sleep callable.
            metrics_config: User provided configuration controlling metrics collection behavior.

        Notes:
            - Runs until `stop_event` is set.
            - Safe to use in multi-threaded environments.
            - Should not be invoked from within an asyncio event loop.
        """
        self._collect_metrics_common(stop_event, http_getter, sleeper, metrics_config=metrics_config, is_async=False)

    async def _collect_metrics_async(self, stop_event, http_getter, sleeper, metrics_config):
        """
        Asynchronous metrics collector.

        Invokes `_collect_metrics_common` in fully async mode. Designed for use when the caller is already
        inside an active asyncio event loop.

        Args:
            stop_event: asyncio.Event to signal termination of metric collection.
            http_getter: Asynchronous callable that fetches Prometheus query results.
            sleeper: Async sleep callable.
            metrics_config: User provided configuration controlling metrics collection behavior.

        Note:
            - Should only be called from within an asyncio context.
            - Automatically terminates once `stop_event` is set.
            - Prints formatted metrics continuously until stopped.
        """
        await self._collect_metrics_common(
            stop_event, http_getter, sleeper, metrics_config=metrics_config, is_async=True
        )

    # ----------------- Core APIs ----------------- #
    def stream_logs(self, request_id, stop_event, log_config: LoggingConfig):
        """Start websocket log streaming in a separate thread.

        Args:
            request_id: The request ID to filter logs for
            stop_event: Event to signal when to start grace period shutdown
            log_config: Configuration for log streaming behavior
        """
        logger.debug(f"Streaming logs for service {self.service_name} (request_id: {request_id})")

        base_url = service_url()
        base_host, base_port = extract_host_port(base_url)
        self._run_log_stream(request_id, stop_event, base_host, base_port, log_config)

    async def stream_logs_async(self, request_id, stop_event, log_config: LoggingConfig):
        """Async version of stream_logs. Start websocket log streaming as an async task.

        Args:
            request_id: The request ID to filter logs for
            stop_event: Event to signal when to start grace period shutdown
            log_config: Configuration for log streaming behavior
        """
        logger.debug(f"Streaming logs for service {self.service_name} (request_id: {request_id})")

        base_url = service_url()
        base_host, base_port = extract_host_port(base_url)
        # Run log and event streams in parallel (same as sync version in _run_log_stream)
        await asyncio.gather(
            self._stream_logs_websocket(request_id, stop_event, host=base_host, port=base_port, log_config=log_config),
            self._stream_events_websocket(stop_event, host=base_host, port=base_port, log_config=log_config),
            return_exceptions=True,  # Don't fail if one stream errors
        )

    async def stream_metrics_async(self, request_id, stop_event, metrics_config):
        """Async GPU/CPU metrics streaming (uses httpx.AsyncClient)."""
        logger.debug(f"Starting async metrics for {self.service_name} (request_id={request_id})")

        async def async_http_get(url, params):
            try:
                resp = await self.async_session.get(url, params=params, timeout=5.0)
                resp.raise_for_status()
                try:
                    return resp.json()
                except json.JSONDecodeError:
                    logger.debug(f"Non-JSON response from {url}: {resp.text[:100]}")
                    return {}
            except Exception as e:
                logger.debug(f"Async metrics request failed for {url} ({params}): {e}")
                return {}

        async def async_sleep(seconds):
            await asyncio.sleep(seconds)

        await self._collect_metrics_async(stop_event, async_http_get, async_sleep, metrics_config)
        logger.debug(f"Stopped async metrics for {request_id}")

    def stream_metrics(self, stop_event, metrics_config: MetricsConfig = None):
        """Synchronous GPU/CPU metrics streaming (uses requests)."""
        logger.debug(f"Streaming metrics for {self.service_name}")
        logger.debug(f"Using metrics config: {metrics_config}")

        def sync_http_get(url, params):
            try:
                resp = requests.get(url, params=params, timeout=5.0)
                resp.raise_for_status()
                try:
                    return resp.json()
                except json.JSONDecodeError:
                    logger.debug(f"Non-JSON response from {url}: {resp.text[:100]}")
                    return {}
            except Exception as e:
                logger.debug(f"Sync metrics request failed for {url} ({params}): {e}")
                return {}

        def sync_sleep(seconds):
            time.sleep(seconds)

        self._collect_metrics(stop_event, sync_http_get, sync_sleep, metrics_config)

    def call_method(
        self,
        endpoint: str,
        stream_logs: bool,
        logging_config: LoggingConfig,
        stream_metrics: Union[bool, MetricsConfig, None] = None,
        body: dict = None,
        headers: dict = None,
        pdb: Union[bool, int] = None,
        serialization: str = "json",
        debug: Union[bool, DebugConfig, None] = None,
    ):
        (endpoint, headers, stop_event, log_thread, metrics_thread, _,) = self._prepare_request(
            endpoint, stream_logs, stream_metrics, headers, pdb, serialization, logging_config, debug
        )
        try:
            json_data = _serialize_body(body, serialization)
            response = self.post(endpoint=endpoint, json=json_data, headers=headers)
            response.raise_for_status()
            return _deserialize_response(response, serialization)
        finally:
            stop_event.set()
            # Block main thread to allow log streaming to complete if shutdown_grace_period > 0
            if log_thread and logging_config.shutdown_grace_period > 0:
                log_thread.join(timeout=logging_config.shutdown_grace_period)

    async def call_method_async(
        self,
        endpoint: str,
        stream_logs: bool,
        logging_config: LoggingConfig,
        stream_metrics: Union[bool, MetricsConfig, None] = None,
        body: dict = None,
        headers: dict = None,
        pdb: Union[bool, int] = None,
        serialization: str = "json",
        debug: Union[bool, DebugConfig, None] = None,
    ):
        """Async version of call_method."""
        (endpoint, headers, stop_event, log_task, monitoring_task, _,) = self._prepare_request_async(
            endpoint, stream_logs, stream_metrics, headers, pdb, serialization, logging_config, debug
        )
        try:
            json_data = _serialize_body(body, serialization)
            response = await self.post_async(endpoint=endpoint, json=json_data, headers=headers)
            response.raise_for_status()
            result = _deserialize_response(response, serialization)

            if stream_logs and log_task:
                await asyncio.sleep(0.5)

            return result
        finally:
            stop_event.set()
            if log_task:
                # Use shutdown_grace_period if set, otherwise use a short default timeout
                timeout = logging_config.shutdown_grace_period if logging_config.shutdown_grace_period > 0 else 0.5
                try:
                    await asyncio.wait_for(log_task, timeout=timeout)
                except asyncio.TimeoutError:
                    # Always cancel on timeout to prevent "Task was destroyed but it is pending!" warnings
                    log_task.cancel()
                    try:
                        await log_task
                    except asyncio.CancelledError:
                        pass
            # Clean up metrics task
            if monitoring_task:
                try:
                    await asyncio.wait_for(monitoring_task, timeout=0.5)
                except asyncio.TimeoutError:
                    monitoring_task.cancel()
                    try:
                        await monitoring_task
                    except asyncio.CancelledError:
                        pass

    def post(self, endpoint, json=None, headers=None):
        return self._make_request("post", endpoint, json=json, headers=headers)

    def put(self, endpoint, json=None, headers=None):
        return self._make_request("put", endpoint, json=json, headers=headers)

    def delete(self, endpoint, json=None, headers=None):
        return self._make_request("delete", endpoint, json=json, headers=headers)

    def get(self, endpoint, headers=None, timeout=None):
        return self._make_request("get", endpoint, headers=headers, timeout=timeout)

    async def post_async(self, endpoint, json=None, headers=None):
        return await self._make_request_async("post", endpoint, json=json, headers=headers)

    async def put_async(self, endpoint, json=None, headers=None):
        return await self._make_request_async("put", endpoint, json=json, headers=headers)

    async def delete_async(self, endpoint, json=None, headers=None):
        return await self._make_request_async("delete", endpoint, json=json, headers=headers)

    async def get_async(self, endpoint, headers=None):
        return await self._make_request_async("get", endpoint, headers=headers)
