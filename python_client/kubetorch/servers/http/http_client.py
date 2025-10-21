import asyncio
import json
import threading
import time
import urllib.parse
from datetime import datetime, timezone
from typing import Optional, Union

import httpx
import websockets

from kubernetes import client
from kubernetes.client.rest import ApiException

from kubetorch.globals import config, service_url
from kubetorch.logger import get_logger

from kubetorch.servers.http.utils import (
    _deserialize_response,
    _serialize_body,
    generate_unique_request_id,
    PodTerminatedError,
    request_id_ctx_var,
)

from kubetorch.serving.constants import (
    DEFAULT_DEBUG_PORT,
    DEFAULT_NGINX_PORT,
    KT_TERMINATION_REASONS,
)
from kubetorch.utils import extract_host_port, ServerLogsFormatter

logger = get_logger(__name__)


class CustomResponse(httpx.Response):
    def raise_for_status(self):
        """Raises parsed server errors or HTTPError for other status codes"""
        if not 400 <= self.status_code < 600:
            return

        if "application/json" in self.headers.get("Content-Type", ""):
            try:
                error_data = self.json()
                if all(
                    k in error_data
                    for k in ["error_type", "message", "traceback", "pod_name"]
                ):
                    error_type = error_data["error_type"]
                    message = error_data.get("message", "")
                    traceback = error_data["traceback"]
                    pod_name = error_data["pod_name"]
                    error_state = error_data.get(
                        "state", {}
                    )  # Optional serialized state

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
                            logger.debug(
                                f"Could not reconstruct {error_type}: {e}, will use dynamic type"
                            )
                            # Fall back to dynamic creation
                            pass

                    # If we couldn't create the actual exception, fall back to dynamic type creation
                    if not exc:

                        def create_str_method(remote_traceback):
                            def __str__(self):
                                cleaned_traceback = remote_traceback.encode().decode(
                                    "unicode_escape"
                                )
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
                            cleaned_traceback = self.remote_traceback.encode().decode(
                                "unicode_escape"
                            )
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
        self._core_api = None
        self._objects_api = None
        self._tracing_enabled = config.tracing_enabled

        self.compute = compute
        self.service_name = service_name
        self.base_url = base_url.rstrip("/")
        self.session = CustomSession()
        self._async_client = None

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
    def core_api(self):
        if self._core_api is None:
            self._core_api = client.CoreV1Api()
        return self._core_api

    @property
    def objects_api(self):
        if self._objects_api is None:
            self._objects_api = client.CustomObjectsApi()
        return self._objects_api

    @property
    def local_port(self):
        """Local port to open the port forward connection with the proxy service. This should match the client port used
        to set the URL of the service in the Compute class."""
        if self.compute:
            return self.compute.client_port()
        return DEFAULT_NGINX_PORT

    @property
    def async_session(self):
        """Get or create async HTTP client."""
        if self._async_client is None:
            self._async_client = CustomAsyncClient()
        return self._async_client

    # ----------------- Error Handling ----------------- #
    def _handle_response_errors(self, response):
        """If we didn't get json back, it could be that the pod is already dead but Knative returns the 500 because
        the service was mid-termination."""
        status_code = response.status_code
        if status_code >= 500 and self._tracing_enabled:
            request_id = response.request.headers.get("X-Request-ID")
            pod_name, start_time = self._load_pod_metadata_from_tempo(
                request_id=request_id
            )
            if pod_name:
                self._handle_500x_error(pod_name, status_code, start_time)
            else:
                logger.debug(f"No pod name found for request {request_id}")

    def _handle_500x_error(
        self, pod_name: str, status_code: int, start_time: float
    ) -> None:
        """Handle 500x errors by surfacing container status and kubernetes events."""
        termination_reason = None
        try:
            pod = self.core_api.read_namespaced_pod(
                name=pod_name, namespace=self.compute.namespace
            )

            # Check container termination states for better reason
            for container_status in pod.status.container_statuses or []:
                state = container_status.state
                # Note: if pod was killed abruptly kubelet might not have time to set the container state
                if state.terminated:
                    termination_reason = state.terminated.reason
                    error_code = state.terminated.exit_code

                    if (
                        termination_reason not in KT_TERMINATION_REASONS
                        and error_code == 137
                    ):
                        termination_reason = "OOMKilled"
                        logger.warning(
                            "OOM suspected: pod exited with code 137 but no termination reason "
                            "found in Kubernetes events"
                        )

                    logger.debug(
                        f"Pod {pod_name} terminated with reason: {termination_reason}"
                    )
                    break

            if termination_reason is None:
                termination_reason = pod.status.reason

        except ApiException as e:
            termination_reason = e.reason if e.reason == "Not Found" else None

        # we are updating termination_reason only if the pod is indeed was not found / terminated.
        if termination_reason in KT_TERMINATION_REASONS:
            # Convert start_time float to ISO8601 format for comparison
            start_dt = datetime.fromtimestamp(start_time, tz=timezone.utc)

            # Fetch pod events since request started
            events = self._get_pod_events_since(pod_name, start_time=start_dt)

            raise PodTerminatedError(
                pod_name=pod_name,
                reason=termination_reason,
                status_code=status_code,
                events=events,
            )

    def _get_pod_events_since(self, pod_name: str, start_time: datetime) -> list[dict]:
        """Fetch all events for a pod since the given start time."""
        try:
            events = self.core_api.list_namespaced_event(
                namespace=self.compute.namespace,
                field_selector=f"involvedObject.name={pod_name}",
            )
            filtered_events = []
            for event in events.items:
                if event.first_timestamp and event.first_timestamp > start_time:
                    filtered_events.append(
                        {
                            "timestamp": event.first_timestamp,
                            "reason": event.reason,
                            "message": event.message,
                        }
                    )
            return filtered_events
        except Exception as e:
            logger.warning(f"Failed to fetch pod events for {pod_name}: {e}")
            return []

    def _query_tempo_internal(
        self, tempo_url: str, request_id: str, retries=5, delay=2.0
    ) -> Optional[tuple[str, float]]:
        """
        Query Tempo for the trace with the given request_id and return:
        - the pod name (`service.instance.id`)
        - the trace start time in epoch seconds

        Note: retries are used to handle Tempo not being ready yet.
        """
        for attempt in range(retries):
            try:
                search_url = f"{tempo_url}/api/search"
                params = {"tags": f"request_id={request_id}"}
                response = httpx.get(search_url, params=params, timeout=2)
                response.raise_for_status()
                traces = response.json().get("traces", [])
                if traces:
                    trace_info = traces[0]
                    trace_id = trace_info["traceID"]
                    start_time_ns = int(trace_info["startTimeUnixNano"])
                    start_time_sec = start_time_ns / 1_000_000_000

                    # Fetch pod name from full trace detail
                    detail_url = f"{tempo_url}/api/traces/{trace_id}"
                    detail_response = httpx.get(detail_url, timeout=2)
                    detail_response.raise_for_status()
                    trace_data = detail_response.json()

                    for batch in trace_data.get("batches", []):
                        for attr in batch.get("resource", {}).get("attributes", []):
                            if attr["key"] == "service.instance.id":
                                pod_name = attr["value"].get("stringValue")
                                return pod_name, start_time_sec

            except Exception as e:
                logger.debug(f"Attempt {attempt + 1} failed to query Tempo: {e}")

            if attempt < retries - 1:
                logger.debug(f"Retrying loading traces in {delay} seconds...")
                time.sleep(delay)

        logger.warning("Failed to load pod metadata from Tempo")
        return None, None

    def _load_pod_metadata_from_tempo(self, request_id: str):
        """Query Tempo for the trace with the given request_id and return the pod name if available.
        Note there are a few reasons why we may fail to load the data from Tempo:
        (1) Flush failure: pod was killed (OOM, preempted, etc.) before the OTEL exporter flushed to Tempo
        (2) Tempo ingestion errors
        """
        base_url = service_url()
        tempo_url = f"{base_url}/tempo"
        return self._query_tempo_internal(tempo_url, request_id)

    def _prepare_request(
        self,
        endpoint: str,
        stream_logs: Union[bool, None],
        headers: dict,
        pdb: Union[bool, int],
        serialization: str,
    ):
        if stream_logs is None:
            stream_logs = config.stream_logs or False

        if pdb:
            debug_port = DEFAULT_DEBUG_PORT if isinstance(pdb, bool) else pdb
            endpoint += f"?debug_port={debug_port}"

        request_id = request_id_ctx_var.get("-")
        if request_id == "-":
            timestamp = str(time.time())
            request_id = generate_unique_request_id(
                endpoint=endpoint, timestamp=timestamp
            )

        headers = headers or {}
        headers.update({"X-Request-ID": request_id, "X-Serialization": serialization})

        stop_event = threading.Event()
        log_thread = None
        if stream_logs:
            log_thread = threading.Thread(
                target=self.stream_logs, args=(request_id, stop_event)
            )
            log_thread.daemon = True
            log_thread.start()

        return endpoint, headers, stop_event, log_thread, request_id

    def _prepare_request_async(
        self,
        endpoint: str,
        stream_logs: Union[bool, None],
        headers: dict,
        pdb: Union[bool, int],
        serialization: str,
    ):
        """Async version of _prepare_request that uses asyncio.Event and tasks instead of threads"""
        if stream_logs is None:
            stream_logs = config.stream_logs or False

        if pdb:
            debug_port = DEFAULT_DEBUG_PORT if isinstance(pdb, bool) else pdb
            endpoint += f"?debug_port={debug_port}"

        request_id = request_id_ctx_var.get("-")
        if request_id == "-":
            timestamp = str(time.time())
            request_id = generate_unique_request_id(
                endpoint=endpoint, timestamp=timestamp
            )

        headers = headers or {}
        headers.update({"X-Request-ID": request_id, "X-Serialization": serialization})

        stop_event = asyncio.Event()
        log_task = None
        if stream_logs:
            log_task = asyncio.create_task(
                self.stream_logs_async(request_id, stop_event)
            )

        return endpoint, headers, stop_event, log_task, request_id

    def _make_request(self, method, endpoint, **kwargs):
        response: httpx.Response = getattr(self.session, method)(endpoint, **kwargs)
        self._handle_response_errors(response)
        response.raise_for_status()
        return response

    async def _make_request_async(self, method, endpoint, **kwargs):
        """Async version of _make_request."""
        response = await getattr(self.async_session, method)(endpoint, **kwargs)
        self._handle_response_errors(response)
        response.raise_for_status()
        return response

    # ----------------- Stream Helpers ----------------- #
    async def _stream_logs_websocket(
        self,
        request_id,
        stop_event: Union[threading.Event, asyncio.Event],
        port: int,
        host: str = "localhost",
    ):
        """Stream logs using Loki's websocket tail endpoint"""
        formatter = ServerLogsFormatter()
        websocket = None
        try:
            query = (
                f'{{k8s_container_name="kubetorch"}} | json | request_id="{request_id}"'
            )
            encoded_query = urllib.parse.quote_plus(query)
            uri = f"ws://{host}:{port}/loki/api/v1/tail?query={encoded_query}"
            # Track the last timestamp we've seen to avoid duplicates
            last_timestamp = None
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
                    is_stop_set = (
                        stop_event.is_set()
                        if hasattr(stop_event, "is_set")
                        else stop_event.is_set()
                    )
                    if is_stop_set and stop_time is None:
                        stop_time = time.time() + 2  # 2 seconds grace period

                    # If we're past the grace period, exit
                    if stop_time is not None and time.time() > stop_time:
                        break

                    try:
                        # Use shorter timeout during grace period
                        timeout = 0.1 if stop_time is not None else 1.0
                        message = await asyncio.wait_for(
                            websocket.recv(), timeout=timeout
                        )
                        data = json.loads(message)

                        if data.get("streams"):
                            for stream in data["streams"]:
                                labels = stream["stream"]
                                service_name = labels.get("kubetorch_com_service")

                                # Determine if this is a Knative service by checking for Knative-specific labels
                                is_knative = (
                                    labels.get("serving_knative_dev_configuration")
                                    is not None
                                )

                                for value in stream["values"]:
                                    # Skip if we've already seen this timestamp
                                    log_line = json.loads(value[1])
                                    log_name = log_line.get("name")
                                    log_message = log_line.get("message")
                                    current_timestamp = value[0]
                                    if (
                                        last_timestamp is not None
                                        and current_timestamp <= last_timestamp
                                    ):
                                        continue
                                    last_timestamp = value[0]

                                    # Choose the appropriate identifier for the log prefix
                                    if is_knative:
                                        log_prefix = service_name
                                    else:
                                        # For deployments, use the pod name from the structured log
                                        log_prefix = log_line.get("pod", service_name)

                                    if log_name == "print_redirect":
                                        print(
                                            f"{formatter.start_color}({log_prefix}) {log_message}{formatter.reset_color}"
                                        )
                                    elif log_name != "uvicorn.access":
                                        formatted_log = f"({log_prefix}) {log_line.get('asctime')} | {log_line.get('levelname')} | {log_message}"
                                        print(
                                            f"{formatter.start_color}{formatted_log}{formatter.reset_color}"
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
            # Ensure websocket is closed even if we didn't enter the context
            if websocket:
                try:
                    # Use wait_for to prevent hanging on close
                    await asyncio.wait_for(websocket.close(), timeout=1.0)
                except (asyncio.TimeoutError, Exception):
                    pass

    def _run_log_stream(self, request_id, stop_event, host, port):
        """Helper to run log streaming in an event loop"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                self._stream_logs_websocket(
                    request_id, stop_event, host=host, port=port
                )
            )
        finally:
            loop.close()

    # ----------------- Core APIs ----------------- #
    def stream_logs(self, request_id, stop_event):
        """Start websocket log streaming in a separate thread"""
        logger.debug(
            f"Streaming logs for service {self.service_name} (request_id: {request_id})"
        )

        base_url = service_url()
        base_host, base_port = extract_host_port(base_url)
        self._run_log_stream(request_id, stop_event, base_host, base_port)

    async def stream_logs_async(self, request_id, stop_event):
        """Async version of stream_logs. Start websocket log streaming as an async task"""
        logger.debug(
            f"Streaming logs for service {self.service_name} (request_id: {request_id})"
        )

        base_url = service_url()
        base_host, base_port = extract_host_port(base_url)
        await self._stream_logs_websocket(
            request_id, stop_event, host=base_host, port=base_port
        )

    def call_method(
        self,
        endpoint: str,
        stream_logs: Union[bool, None] = None,
        body: dict = None,
        headers: dict = None,
        pdb: Union[bool, int] = None,
        serialization: str = "json",
    ):
        endpoint, headers, stop_event, log_thread, _ = self._prepare_request(
            endpoint, stream_logs, headers, pdb, serialization
        )
        try:
            json_data = _serialize_body(body, serialization)
            response = self.post(endpoint=endpoint, json=json_data, headers=headers)
            response.raise_for_status()
            return _deserialize_response(response, serialization)
        finally:
            stop_event.set()

    async def call_method_async(
        self,
        endpoint: str,
        stream_logs: Union[bool, None] = None,
        body: dict = None,
        headers: dict = None,
        pdb: Union[bool, int] = None,
        serialization: str = "json",
    ):
        """Async version of call_method."""
        endpoint, headers, stop_event, log_task, _ = self._prepare_request_async(
            endpoint, stream_logs, headers, pdb, serialization
        )
        try:
            json_data = _serialize_body(body, serialization)
            response = await self.post_async(
                endpoint=endpoint, json=json_data, headers=headers
            )
            response.raise_for_status()
            result = _deserialize_response(response, serialization)

            if stream_logs and log_task:
                await asyncio.sleep(0.5)

            return result
        finally:
            stop_event.set()
            if log_task:
                try:
                    await asyncio.wait_for(log_task, timeout=0.5)
                except asyncio.TimeoutError:
                    log_task.cancel()
                    try:
                        await log_task
                    except asyncio.CancelledError:
                        pass

    def post(self, endpoint, json=None, headers=None):
        return self._make_request("post", endpoint, json=json, headers=headers)

    def put(self, endpoint, json=None, headers=None):
        return self._make_request("put", endpoint, json=json, headers=headers)

    def delete(self, endpoint, json=None, headers=None):
        return self._make_request("delete", endpoint, json=json, headers=headers)

    def get(self, endpoint, headers=None):
        return self._make_request("get", endpoint, headers=headers)

    async def post_async(self, endpoint, json=None, headers=None):
        return await self._make_request_async(
            "post", endpoint, json=json, headers=headers
        )

    async def put_async(self, endpoint, json=None, headers=None):
        return await self._make_request_async(
            "put", endpoint, json=json, headers=headers
        )

    async def delete_async(self, endpoint, json=None, headers=None):
        return await self._make_request_async(
            "delete", endpoint, json=json, headers=headers
        )

    async def get_async(self, endpoint, headers=None):
        return await self._make_request_async("get", endpoint, headers=headers)
