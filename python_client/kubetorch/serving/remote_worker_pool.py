import json
import multiprocessing
import os
import queue
import signal
import threading
import uuid

from kubetorch.serving.http_server import logger
from kubetorch.serving.utils import _build_ws_request, _process_ws_response, kill_process_tree


class RemoteWorkerPool:
    """Manages async HTTP calls to remote workers in a separate process.

    This class implements a singleton pattern - only one pool exists per pod.
    Use RemoteWorkerPool.get_instance() to get the shared instance.
    """

    _instance = None
    _instance_lock = threading.Lock()

    @classmethod
    def get_instance(cls, quorum_timeout=3600, max_workers=200):
        """Get or create the singleton RemoteWorkerPool instance.

        The pool is created lazily on first call and reused thereafter.
        If the existing pool is dead, a new one is created.

        Args:
            quorum_timeout: Timeout for quorum operations (only used on creation)
            max_workers: Max concurrent workers (only used on creation)

        Returns:
            The singleton RemoteWorkerPool instance, started and ready.
        """
        with cls._instance_lock:
            if cls._instance is None or not cls._instance.is_alive():
                if cls._instance is not None:
                    logger.info("RemoteWorkerPool was dead, creating new instance")
                    cls._instance.stop()

                logger.info(f"Creating singleton RemoteWorkerPool (max_workers={max_workers})")
                cls._instance = cls(quorum_timeout=quorum_timeout)
                cls._instance.start(max_workers=max_workers)
                logger.info("Singleton RemoteWorkerPool ready")

            return cls._instance

    def __init__(self, quorum_timeout=300):
        self.quorum_timeout = quorum_timeout
        self.request_queue = multiprocessing.Queue()
        self.response_queue = multiprocessing.Queue()
        self.process = None
        self._running = False
        # Thread-safe response routing for concurrent calls
        self._response_events = {}
        self._response_lock = threading.Lock()
        self._router_thread = None

    def is_alive(self):
        """Check if the pool's subprocess is running."""
        return self.process is not None and self.process.is_alive()

    def start(self, max_workers=2000):
        """Start the worker process and router thread."""
        if self.process:
            raise RuntimeError("WorkerPool already started")

        self._running = True
        # Pass necessary data as arguments to avoid pickling issues
        self.process = multiprocessing.Process(
            target=self._run_async_worker,
            args=(
                self.request_queue,
                self.response_queue,
                self.quorum_timeout,
                max_workers,
            ),
            daemon=True,
        )
        self.process.start()

        # Start router thread for handling responses
        self._router_thread = threading.Thread(target=self._response_router, daemon=True)
        self._router_thread.start()

        logger.debug("Started RemoteWorkerPool process and router thread")

    def stop(self):
        """Stop the worker process and router thread."""
        self._running = False

        # Stop router thread
        if self._router_thread:
            self.response_queue.put({"request_id": "STOP_ROUTER"})
            self._router_thread.join(timeout=1)

        # Stop worker process
        if self.process:
            self.request_queue.put(("SHUTDOWN", None))
            self.process.join(timeout=5)
            if self.process and self.process.is_alive():
                # Kill the process tree to terminate any child processes
                kill_process_tree(self.process.pid, signal.SIGTERM)
                self.process.join(timeout=1)
                if self.process and self.process.is_alive():
                    kill_process_tree(self.process.pid, signal.SIGKILL)
            self.process = None

        # Clear response events
        with self._response_lock:
            self._response_events.clear()

        logger.debug("Stopped RemoteWorkerPool process and router thread")

    def call_workers(
        self,
        worker_ips,
        cls_or_fn_name,
        method_name,
        params,
        request_headers,
        workers_arg="all",
    ):
        """Call remote workers and return responses."""
        if not self.process or not self.process.is_alive():
            raise RuntimeError("RemoteWorkerPool not running")

        # Generate unique request ID
        request_id = str(uuid.uuid4())

        # Submit request
        request_data = {
            "request_id": request_id,
            "worker_ips": worker_ips,
            "cls_or_fn_name": cls_or_fn_name,
            "method_name": method_name,
            "params": params,
            "request_headers": request_headers,
            "workers_arg": workers_arg,
        }

        # Register event for this request
        event = threading.Event()
        with self._response_lock:
            self._response_events[request_id] = (event, None)

        logger.debug(f"RemoteWorkerPool: Submitting request {request_id} to queue for {len(worker_ips)} workers")
        self.request_queue.put(("CALL", request_data))

        # Wait for response with a timeout to prevent hanging forever
        logger.debug(f"RemoteWorkerPool: Waiting for response for request {request_id} from {len(worker_ips)} workers")
        # Wait indefinitely for the response (no timeout for long-running jobs)
        event.wait()

        # Get and cleanup response
        with self._response_lock:
            _, result = self._response_events.pop(request_id)

        logger.debug(f"RemoteWorkerPool: Got response for request {request_id}, type: {type(result).__name__}")

        if isinstance(result, Exception):
            raise result
        return result

    def _response_router(self):
        """Router thread that distributes responses to waiting threads."""
        logger.debug("RemoteWorkerPool response router thread started")
        while self._running:
            try:
                response = self.response_queue.get(timeout=0.1)
                if response.get("request_id") == "STOP_ROUTER":
                    break

                request_id = response.get("request_id")
                logger.debug(f"Response router received response for request {request_id}")
                with self._response_lock:
                    if request_id in self._response_events:
                        event, _ = self._response_events[request_id]
                        # Store the result (either results list or exception)
                        if "error" in response:
                            logger.debug(f"Response router: Setting error for request {request_id}")
                            self._response_events[request_id] = (
                                event,
                                response["error"],
                            )
                        else:
                            logger.debug(
                                f"Response router: Setting {len(response.get('results', []))} results for request {request_id}"
                            )
                            self._response_events[request_id] = (
                                event,
                                response["results"],
                            )
                        event.set()
                        logger.debug(f"Response router: Event set for request {request_id}")
                    else:
                        logger.warning(
                            f"Response router: No event found for request {request_id}, registered events: {list(self._response_events.keys())}"
                        )
            except queue.Empty:
                continue  # Queue timeout, continue checking
            except Exception as e:
                logger.error(f"Error in response router: {e}")
                continue

    @staticmethod
    def _run_async_worker(request_queue, response_queue, quorum_timeout, max_workers=2000):
        """Worker process that handles async WebSocket calls.

        Architecture:
        - Runs in a separate process with its own event loop
        - Maintains persistent WebSocket connections per worker pod
        - Multiplexes requests over connections using request_id routing
        - Processes requests from queue concurrently (multiple requests in flight)

        Nested functions (share queue/timeout state):
        - get_ws_connection: Get or create WebSocket connection to a worker
        - ws_response_listener: Route incoming responses by request_id
        - wait_for_worker_health: Establish connection + ping check
        - call_single_worker: Send request and await response via WebSocket
        - call_workers_async: Orchestrate health checks + calls for all workers
        """
        import asyncio
        import queue

        import websockets

        async def main():
            # Set up signal handler for graceful shutdown
            import signal as sig

            shutdown_event = asyncio.Event()

            # Save existing handlers to chain to them
            original_sigterm_handler = sig.getsignal(sig.SIGTERM)
            original_sigint_handler = sig.getsignal(sig.SIGINT)

            def signal_handler(signum, frame):
                logger.info(f"Received signal {signum}, initiating graceful shutdown")
                shutdown_event.set()

                # Chain to original handler if it exists
                if signum == sig.SIGTERM:
                    if original_sigterm_handler and original_sigterm_handler not in (
                        sig.SIG_DFL,
                        sig.SIG_IGN,
                    ):
                        original_sigterm_handler(signum, frame)
                elif signum == sig.SIGINT:
                    if original_sigint_handler and original_sigint_handler not in (
                        sig.SIG_DFL,
                        sig.SIG_IGN,
                    ):
                        original_sigint_handler(signum, frame)

            sig.signal(sig.SIGTERM, signal_handler)
            sig.signal(sig.SIGINT, signal_handler)

            # WebSocket connection state
            ws_connections = {}  # worker_ip -> WebSocket connection
            ws_locks = {}  # worker_ip -> asyncio.Lock for connection management
            pending_requests = {}  # request_id -> asyncio.Future for response routing
            listener_tasks = {}  # worker_ip -> listener task

            port = os.environ["KT_SERVER_PORT"]

            async def ws_response_listener(worker_ip, ws):
                """Listen for responses on a WebSocket and route them by request_id."""
                try:
                    async for message in ws:
                        try:
                            response = json.loads(message)
                            req_id = response.get("request_id")
                            if req_id and req_id in pending_requests:
                                future = pending_requests.pop(req_id)
                                if not future.done():
                                    future.set_result(response)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON from {worker_ip}: {e}")
                except websockets.exceptions.ConnectionClosed as e:
                    logger.debug(f"WebSocket to {worker_ip} closed: {e}")
                except Exception as e:
                    logger.error(f"WebSocket listener error for {worker_ip}: {e}")
                finally:
                    # Connection closed, clean up
                    if worker_ip in ws_connections:
                        del ws_connections[worker_ip]
                    if worker_ip in listener_tasks:
                        del listener_tasks[worker_ip]
                    # Cancel any pending requests for this worker
                    for req_id, future in list(pending_requests.items()):
                        if not future.done():
                            future.set_exception(ConnectionError(f"WebSocket to {worker_ip} closed"))

            async def get_ws_connection(worker_ip):
                """Get or create a WebSocket connection to a worker."""
                if worker_ip not in ws_locks:
                    ws_locks[worker_ip] = asyncio.Lock()

                async with ws_locks[worker_ip]:
                    # Check if existing connection is still open
                    if worker_ip in ws_connections:
                        ws = ws_connections[worker_ip]
                        if ws.state.value == 1:  # State.OPEN
                            return ws
                        # Connection closed, remove it
                        del ws_connections[worker_ip]

                    # Create new connection
                    ws_url = f"ws://{worker_ip}:{port}/ws/callable"
                    logger.debug(f"Connecting to WebSocket: {ws_url}")
                    ws = await websockets.connect(
                        ws_url,
                        close_timeout=10,
                        ping_interval=20,
                        ping_timeout=10,
                    )
                    ws_connections[worker_ip] = ws

                    # Start response listener for this connection
                    task = asyncio.create_task(ws_response_listener(worker_ip, ws))
                    listener_tasks[worker_ip] = task

                    logger.debug(f"WebSocket connection established to {worker_ip}")
                    return ws

            async def wait_for_worker_health(worker_ip, workers_arg, quorum_timeout):
                """Wait for a worker to become healthy by establishing WebSocket connection."""
                start_time = asyncio.get_event_loop().time()

                while (asyncio.get_event_loop().time() - start_time) < quorum_timeout:
                    try:
                        ws = await get_ws_connection(worker_ip)
                        # Connection established = healthy
                        # Optionally send ping to verify
                        pong = await ws.ping()
                        await asyncio.wait_for(pong, timeout=5.0)
                        return (worker_ip, True)
                    except Exception as e:
                        logger.debug(f"Health check failed for {worker_ip}: {e}")
                        # Clean up failed connection
                        if worker_ip in ws_connections:
                            try:
                                await ws_connections[worker_ip].close()
                            except Exception:
                                pass
                            del ws_connections[worker_ip]

                        if workers_arg == "ready":
                            return (worker_ip, False)

                        await asyncio.sleep(1.0)

                return (worker_ip, False)

            async def call_single_worker(
                worker_ip,
                cls_or_fn_name,
                method_name,
                params,
                request_headers,
                workers_arg,
            ):
                """Call a single worker over WebSocket."""
                # Generate unique request ID for WebSocket response routing
                call_request_id = str(uuid.uuid4())

                # Build request using shared function
                # params are already serialized by the parent call
                request = _build_ws_request(
                    request_id=call_request_id,
                    cls_or_fn_name=cls_or_fn_name,
                    method_name=method_name,
                    params=params,
                    serialization=request_headers.get("X-Serialization", "json"),
                    log_request_id=request_headers.get("X-Request-ID", "-"),
                    distributed_subcall=True,
                )

                # Create future for response
                response_future = asyncio.get_event_loop().create_future()
                pending_requests[call_request_id] = response_future

                max_retries = 3
                serialization = request_headers.get("X-Serialization", "json")
                for attempt in range(max_retries):
                    try:
                        ws = await get_ws_connection(worker_ip)
                        await ws.send(json.dumps(request))
                        logger.debug(f"Sent WebSocket request {call_request_id} to {worker_ip}")

                        # Wait for response (no timeout for long-running jobs)
                        response = await response_future
                        logger.debug(f"Got WebSocket response for {call_request_id} from {worker_ip}")

                        # Use shared response processing (handles errors and deserialization)
                        return _process_ws_response(response, serialization)

                    except websockets.exceptions.ConnectionClosed as e:
                        # Connection closed, remove from cache and retry
                        if worker_ip in ws_connections:
                            del ws_connections[worker_ip]
                        if attempt < max_retries - 1:
                            logger.warning(f"WebSocket to {worker_ip} closed, reconnecting (attempt {attempt + 1})...")
                            # Re-create future for retry
                            response_future = asyncio.get_event_loop().create_future()
                            pending_requests[call_request_id] = response_future
                            await asyncio.sleep(0.5)
                        else:
                            pending_requests.pop(call_request_id, None)
                            raise ConnectionError(f"WebSocket to {worker_ip} closed after {max_retries} attempts: {e}")

                    except Exception as e:
                        pending_requests.pop(call_request_id, None)
                        logger.error(f"Error calling {worker_ip}: {e}")
                        raise

            async def call_workers_async(data):
                """Make async calls to all workers using WebSockets."""
                worker_ips = data["worker_ips"]
                cls_or_fn_name = data["cls_or_fn_name"]
                method_name = data["method_name"]
                params = data["params"]
                request_headers = data["request_headers"]
                workers_arg = data["workers_arg"]

                # Health checks via WebSocket connection
                logger.info(f"Waiting for {len(worker_ips)} workers (WebSocket mode, timeout={quorum_timeout}s)")
                health_tasks = [wait_for_worker_health(ip, workers_arg, quorum_timeout) for ip in worker_ips]
                health_results = await asyncio.gather(*health_tasks)

                # Process results
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
                            f"{len(unhealthy_workers)} workers failed to become ready after {quorum_timeout}s: {unhealthy_workers[:5]}..."
                        )
                        raise TimeoutError(
                            f"{len(unhealthy_workers)} of {len(worker_ips)} workers did not become ready within {quorum_timeout} seconds. "
                            f"This may indicate the pods are still starting or there's a resource constraint. "
                            f"Consider increasing quorum_timeout in .distribute() call."
                        )

                logger.info(f"All {len(healthy_workers)} workers are ready, making distributed calls (WebSocket)")

                # Make calls to ready workers
                tasks = []
                for worker_ip in healthy_workers:
                    coro = call_single_worker(
                        worker_ip,
                        cls_or_fn_name,
                        method_name,
                        params,
                        request_headers,
                        workers_arg,
                    )
                    task = asyncio.create_task(coro)
                    tasks.append(task)

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
                        # Fast failure - cancel remaining tasks
                        for task in pending_tasks:
                            if not task.done():
                                task.cancel()
                        raise e

                return responses

            async def cleanup_connections():
                """Close all WebSocket connections."""
                logger.debug(f"Cleaning up {len(ws_connections)} WebSocket connections")
                for worker_ip, ws in list(ws_connections.items()):
                    try:
                        await ws.close()
                    except Exception:
                        pass
                ws_connections.clear()

                # Cancel listener tasks
                for task in listener_tasks.values():
                    task.cancel()
                listener_tasks.clear()

                # Cancel pending requests
                for req_id, future in list(pending_requests.items()):
                    if not future.done():
                        future.cancel()
                pending_requests.clear()

            logger.debug("RemoteWorkerPool started with WebSocket mode")

            active_tasks = set()

            async def process_call(request_data):
                """Process a CALL request asynchronously."""
                request_id = request_data["request_id"]
                worker_ips = request_data["worker_ips"]
                logger.debug(
                    f"Async worker: Processing request {request_id} with {len(worker_ips)} workers: {worker_ips}"
                )
                try:
                    results = await call_workers_async(request_data)
                    logger.debug(
                        f"Async worker: Successfully got {len(results)} results for request {request_id}, sending response"
                    )
                    response_queue.put({"request_id": request_id, "results": results})
                except Exception as e:
                    logger.error(f"Error processing request {request_id}: {e}")
                    response_queue.put({"request_id": request_id, "error": e})

            # Main request processing loop
            try:
                while not shutdown_event.is_set():
                    try:
                        # Check queue without blocking event loop
                        cmd, request_data = await asyncio.to_thread(request_queue.get, block=True, timeout=0.1)

                        if cmd == "SHUTDOWN" or shutdown_event.is_set():
                            if active_tasks:
                                logger.debug(f"Cancelling {len(active_tasks)} active tasks for shutdown")
                                for task in active_tasks:
                                    task.cancel()
                            break

                        elif cmd == "CALL":
                            task = asyncio.create_task(process_call(request_data))
                            active_tasks.add(task)

                        # Clean up completed tasks periodically
                        active_tasks = {t for t in active_tasks if not t.done()}

                    except queue.Empty:
                        active_tasks = {t for t in active_tasks if not t.done()}
                    except Exception as e:
                        logger.error(f"Error in async worker loop: {e}")
            finally:
                await cleanup_connections()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(main())
        finally:
            loop.close()
