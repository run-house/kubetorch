import asyncio
import json
import os
import threading
import uuid

import websockets

from kubetorch.serving.http_server import logger
from kubetorch.serving.utils import _build_ws_request, _process_ws_response


class RemoteWorkerPool:
    """Manages async WebSocket calls to remote workers.

    Uses persistent WebSocket connections with request_id-based multiplexing.
    Runs its own background event loop so connections persist across calls
    regardless of the caller's threading context.

    Two calling patterns:
    - Async callers (LoadBalancedSupervisor): call_workers_async() directly
      on the background loop via run_coroutine_threadsafe(), or on their own
      loop if they have one.
    - Sync callers (SPMDSupervisor): call_workers() blocks until complete.

    Use RemoteWorkerPool.get_instance() for a shared singleton, or
    instantiate directly for per-supervisor pools.
    """

    _instance = None
    _instance_lock = threading.Lock()

    @classmethod
    def get_instance(cls, quorum_timeout=3600):
        """Get or create the singleton RemoteWorkerPool instance."""
        with cls._instance_lock:
            if cls._instance is None:
                logger.info("Creating singleton RemoteWorkerPool")
                cls._instance = cls(quorum_timeout=quorum_timeout)
            return cls._instance

    def __init__(self, quorum_timeout=300):
        self.quorum_timeout = quorum_timeout
        # WebSocket connection state (all accessed from _loop only)
        self._ws_connections = {}  # worker_ip -> WebSocket
        self._ws_locks = {}  # worker_ip -> asyncio.Lock
        self._pending_requests = {}  # request_id -> asyncio.Future
        self._listener_tasks = {}  # worker_ip -> asyncio.Task
        # Background event loop — keeps connections alive across calls
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="RemoteWorkerPool")
        self._thread.start()

    def call_workers(self, worker_ips, cls_or_fn_name, method_name, params, request_headers, workers_arg="all"):
        """Sync entry point: submit async work to the background loop and block."""
        future = asyncio.run_coroutine_threadsafe(
            self.call_workers_async(
                worker_ips=worker_ips,
                cls_or_fn_name=cls_or_fn_name,
                method_name=method_name,
                params=params,
                request_headers=request_headers,
                workers_arg=workers_arg,
            ),
            self._loop,
        )
        return future.result()  # blocks calling thread

    async def _get_ws_connection(self, worker_ip):
        """Get or create a persistent WebSocket connection to a worker."""
        # Fast path: connection already open — no lock, no await
        ws = self._ws_connections.get(worker_ip)
        if ws is not None and ws.state.value == 1:  # State.OPEN
            return ws

        # Slow path: need to create/reconnect — lock to avoid duplicate connections
        if worker_ip not in self._ws_locks:
            self._ws_locks[worker_ip] = asyncio.Lock()

        async with self._ws_locks[worker_ip]:
            # Re-check after acquiring lock
            ws = self._ws_connections.get(worker_ip)
            if ws is not None and ws.state.value == 1:
                return ws
            self._ws_connections.pop(worker_ip, None)

            port = os.environ["KT_SERVER_PORT"]
            ws_url = f"ws://{worker_ip}:{port}/ws/callable"
            logger.debug(f"Connecting to WebSocket: {ws_url}")
            ws = await websockets.connect(
                ws_url,
                close_timeout=10,
                ping_interval=20,
                ping_timeout=10,
            )
            self._ws_connections[worker_ip] = ws

            task = asyncio.create_task(self._ws_response_listener(worker_ip, ws))
            self._listener_tasks[worker_ip] = task

            logger.debug(f"WebSocket connection established to {worker_ip}")
            return ws

    async def _ws_response_listener(self, worker_ip, ws):
        """Route incoming WebSocket responses by request_id to waiting futures."""
        try:
            async for message in ws:
                try:
                    response = json.loads(message)
                    req_id = response.get("request_id")
                    if req_id and req_id in self._pending_requests:
                        future = self._pending_requests.pop(req_id)
                        if not future.done():
                            future.set_result(response)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON from {worker_ip}: {e}")
        except websockets.exceptions.ConnectionClosed as e:
            logger.debug(f"WebSocket to {worker_ip} closed: {e}")
        except Exception as e:
            logger.error(f"WebSocket listener error for {worker_ip}: {e}")
        finally:
            self._ws_connections.pop(worker_ip, None)
            self._listener_tasks.pop(worker_ip, None)
            # Fail pending requests for this worker
            for req_id, future in list(self._pending_requests.items()):
                if not future.done():
                    future.set_exception(ConnectionError(f"WebSocket to {worker_ip} closed"))

    async def wait_for_worker_health(self, worker_ip, workers_arg="all", quorum_timeout=None):
        """Wait for a worker to become healthy by establishing WebSocket connection + ping."""
        quorum_timeout = quorum_timeout or self.quorum_timeout
        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < quorum_timeout:
            try:
                ws = await self._get_ws_connection(worker_ip)
                pong = await ws.ping()
                await asyncio.wait_for(pong, timeout=5.0)
                return (worker_ip, True)
            except Exception as e:
                logger.debug(f"Health check failed for {worker_ip}: {e}")
                # Clean up failed connection
                ws = self._ws_connections.pop(worker_ip, None)
                if ws:
                    try:
                        await ws.close()
                    except Exception:
                        pass

                if workers_arg == "ready":
                    return (worker_ip, False)

                await asyncio.sleep(1.0)

        return (worker_ip, False)

    async def _call_single_worker_async(self, worker_ip, cls_or_fn_name, method_name, params, request_headers):
        """Send a request to a single worker over WebSocket and await the response."""
        call_request_id = str(uuid.uuid4())
        serialization = request_headers.get("X-Serialization", "json")

        request = _build_ws_request(
            request_id=call_request_id,
            cls_or_fn_name=cls_or_fn_name,
            method_name=method_name,
            params=params,
            serialization=serialization,
            log_request_id=request_headers.get("X-Request-ID", "-"),
            distributed_subcall=True,
        )

        loop = asyncio.get_running_loop()
        response_future = loop.create_future()
        self._pending_requests[call_request_id] = response_future

        max_retries = 3
        for attempt in range(max_retries):
            try:
                ws = await self._get_ws_connection(worker_ip)
                await ws.send(json.dumps(request))
                logger.debug(f"Sent request {call_request_id} to {worker_ip}")

                response = await response_future
                logger.debug(f"Got response for {call_request_id} from {worker_ip}")
                return _process_ws_response(response, serialization)

            except websockets.exceptions.ConnectionClosed as e:
                self._ws_connections.pop(worker_ip, None)
                if attempt < max_retries - 1:
                    logger.warning(f"WebSocket to {worker_ip} closed, reconnecting (attempt {attempt + 1})...")
                    response_future = loop.create_future()
                    self._pending_requests[call_request_id] = response_future
                    await asyncio.sleep(0.5)
                else:
                    self._pending_requests.pop(call_request_id, None)
                    raise ConnectionError(f"WebSocket to {worker_ip} closed after {max_retries} attempts: {e}")

            except Exception as e:
                self._pending_requests.pop(call_request_id, None)
                logger.error(f"Error calling {worker_ip}: {e}")
                raise

    async def call_workers_async(
        self,
        worker_ips,
        cls_or_fn_name,
        method_name,
        params,
        request_headers,
        workers_arg="all",
    ):
        """Async entry point: call workers directly in the caller's event loop.

        For SPMD calls (workers_arg="all"/"ready"), runs health checks first.
        For load-balanced calls (workers_arg=None), skips health checks.
        """
        # Health checks for SPMD-style calls where we need all workers ready.
        # "all" (default) → health check all workers, fail if any unhealthy
        # "ready" → health check, skip unhealthy workers
        # Anything else (None, list, "any") → skip health checks entirely
        if workers_arg in ("all", "ready"):
            logger.info(f"Waiting for {len(worker_ips)} workers (timeout={self.quorum_timeout}s)")
            health_tasks = [self.wait_for_worker_health(ip, workers_arg) for ip in worker_ips]
            health_results = await asyncio.gather(*health_tasks)

            healthy_workers = []
            unhealthy_workers = []
            for worker_ip, is_healthy in health_results:
                if is_healthy:
                    healthy_workers.append(worker_ip)
                else:
                    unhealthy_workers.append(worker_ip)

            if unhealthy_workers:
                if workers_arg == "ready":
                    logger.info(f"Skipping {len(unhealthy_workers)} workers that didn't respond (ready mode)")
                else:
                    logger.error(
                        f"{len(unhealthy_workers)} workers failed to become ready after "
                        f"{self.quorum_timeout}s: {unhealthy_workers[:5]}..."
                    )
                    raise TimeoutError(
                        f"{len(unhealthy_workers)} of {len(worker_ips)} workers did not become ready "
                        f"within {self.quorum_timeout} seconds. "
                        f"This may indicate the pods are still starting or there's a resource constraint. "
                        f"Consider increasing quorum_timeout in .distribute() call."
                    )

            logger.info(f"All {len(healthy_workers)} workers are ready, making distributed calls")
            call_ips = healthy_workers
        else:
            # Load-balanced / single-worker — skip health checks
            call_ips = worker_ips

        # Make calls to workers
        tasks = []
        for worker_ip in call_ips:
            coro = self._call_single_worker_async(worker_ip, cls_or_fn_name, method_name, params, request_headers)
            tasks.append(asyncio.create_task(coro))

        # Use as_completed for fast failure propagation
        responses = []
        pending_tasks = set(tasks)
        for future in asyncio.as_completed(tasks):
            try:
                result = await future
                if result is not None:
                    responses.append(result)
                pending_tasks.discard(future)
            except Exception as e:
                for task in pending_tasks:
                    if not task.done():
                        task.cancel()
                raise e

        return responses

    async def cleanup_async(self):
        """Close all WebSocket connections and cancel listeners."""
        logger.debug(f"Cleaning up {len(self._ws_connections)} WebSocket connections")
        for worker_ip, ws in list(self._ws_connections.items()):
            try:
                await ws.close()
            except Exception:
                pass
        self._ws_connections.clear()

        for task in self._listener_tasks.values():
            task.cancel()
        self._listener_tasks.clear()

        for req_id, future in list(self._pending_requests.items()):
            if not future.done():
                future.cancel()
        self._pending_requests.clear()
