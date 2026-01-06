import multiprocessing
import os
import queue
import threading
import uuid

from kubetorch.serving.http_server import logger


class RemoteWorkerPool:
    """Manages async HTTP calls to remote workers in a separate process."""

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
                self.process.terminate()
                self.process.join(timeout=1)
                if self.process and self.process.is_alive():
                    self.process.kill()
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
        """Worker process that handles async HTTP calls.

        Architecture:
        - Runs in a separate process with its own event loop
        - Maintains a single shared httpx.AsyncClient for connection pooling
        - Processes requests from queue concurrently (multiple requests in flight)
        - Each request involves parallel async HTTP calls to multiple workers

        Nested functions (share queue/timeout state):
        - wait_for_worker_health: Health check with retries
        - call_single_worker: Make HTTP call to one worker
        - call_workers_async: Orchestrate health checks + calls for all workers
        - process_requests: Main loop pulling from queue and dispatching tasks
        """
        import asyncio
        import queue

        import httpx

        async def wait_for_worker_health(client, worker_ip, workers_arg, quorum_timeout):
            """Wait for a worker to become healthy within timeout."""
            port = os.environ["KT_SERVER_PORT"]
            worker_url = f"http://{worker_ip}:{port}"

            start_time = asyncio.get_event_loop().time()

            # Keep trying until timeout
            while (asyncio.get_event_loop().time() - start_time) < quorum_timeout:
                try:
                    resp = await client.get(f"{worker_url}/health", timeout=10.0)
                    if resp.status_code == 200:
                        return (worker_ip, True)  # Return IP and success
                except (httpx.RequestError, httpx.TimeoutException):
                    pass

                # No waiting for quorum if user just wants to call ready workers
                if workers_arg == "ready":
                    return worker_ip, False

                # Wait before retry (1 second, same as original)
                await asyncio.sleep(1.0)

            # Timeout reached
            return worker_ip, False

        async def call_single_worker(
            client,
            worker_ip,
            cls_or_fn_name,
            method_name,
            params,
            request_headers,
            workers_arg,
        ):
            """Call a single worker (assumes health already checked)."""
            port = os.environ["KT_SERVER_PORT"]
            worker_url = f"http://{worker_ip}:{port}"

            # Make the actual call
            call_url = f"{worker_url}/{cls_or_fn_name}"
            if method_name:
                call_url += f"/{method_name}"
            call_url += "?distributed_subcall=true"

            logger.debug(f"Async worker: Making POST to {call_url}")

            # Retry logic for transient failures
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    resp = await client.post(call_url, json=params, headers=request_headers)
                    result = resp.json()
                    break  # Success, exit retry loop
                except httpx.ReadError as e:
                    # Check if this is due to server shutdown (connection reset)
                    if "Connection reset" in str(e) or "Connection closed" in str(e):
                        logger.warning(f"Worker {worker_ip} appears to be shutting down: {e}")
                        raise  # Don't retry on shutdown
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s
                        logger.warning(
                            f"ReadError calling {worker_ip} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s..."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(
                            f"ReadError calling {worker_ip} after {max_retries} attempts: {e}. Worker may be crashed/overloaded."
                        )
                        raise
                except httpx.TimeoutException as e:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Timeout calling {worker_ip} (attempt {attempt + 1}/{max_retries}): {e}. Retrying..."
                        )
                        await asyncio.sleep(1)
                    else:
                        logger.error(f"Timeout calling {worker_ip} after {max_retries} attempts: {e}")
                        raise
                except Exception as e:
                    logger.error(f"Unexpected error calling {worker_ip}: {e}")
                    raise
            logger.debug(
                f"Async worker: Got response from {worker_ip}, type: {type(result).__name__}, "
                f"length: {len(result) if hasattr(result, '__len__') else 'N/A'}"
            )
            # In tree topology, intermediate nodes return aggregated results from their subtree
            # We should preserve the flat list structure
            return result

        async def call_workers_async(client, data):
            """Make async calls to all workers using shared client."""
            worker_ips = data["worker_ips"]
            cls_or_fn_name = data["cls_or_fn_name"]
            method_name = data["method_name"]
            params = data["params"]
            request_headers = data["request_headers"]
            workers_arg = data["workers_arg"]

            # With tree topology limiting fanout, we don't need to batch health checks
            # Each node only calls its direct children in the tree
            logger.info(f"Waiting for {len(worker_ips)} workers to become ready (timeout={quorum_timeout}s)")
            health_tasks = [wait_for_worker_health(client, ip, workers_arg, quorum_timeout) for ip in worker_ips]
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
                    # For "ready" mode, just skip unhealthy workers
                    logger.info(f"Skipping {len(unhealthy_workers)} workers that didn't respond (ready mode)")
                else:
                    # For normal mode, fail if any worker didn't become ready
                    logger.error(
                        f"{len(unhealthy_workers)} workers failed to become ready after {quorum_timeout}s: {unhealthy_workers[:5]}..."
                    )
                    raise TimeoutError(
                        f"{len(unhealthy_workers)} of {len(worker_ips)} workers did not become ready within {quorum_timeout} seconds. "
                        f"This may indicate the pods are still starting or there's a resource constraint. "
                        f"Consider increasing quorum_timeout in .distribute() call."
                    )

            logger.info(f"All {len(healthy_workers)} workers are ready, making distributed calls")

            # Now make the actual calls to ready workers
            # Create tasks (not just coroutines)
            tasks = []
            for worker_ip in healthy_workers:
                coro = call_single_worker(
                    client,
                    worker_ip,
                    cls_or_fn_name,
                    method_name,
                    params,
                    request_headers,
                    workers_arg,
                )
                # Create actual task from coroutine
                task = asyncio.create_task(coro)
                tasks.append(task)

            # Use as_completed for fast failure propagation
            responses = []
            pending_tasks = set(tasks)

            for future in asyncio.as_completed(tasks):
                try:
                    result = await future
                    if result is not None:  # Skip None results
                        responses.append(result)
                    # Remove completed task from pending set
                    pending_tasks.discard(future)
                except Exception as e:
                    # Fast failure - immediately propagate the exception
                    # Cancel remaining tasks to avoid unnecessary work
                    for task in pending_tasks:
                        if not task.done():
                            task.cancel()
                    raise e

            return responses

        # Create and run event loop with shared AsyncClient
        async def main():
            # Set up signal handler for graceful shutdown
            import signal

            shutdown_event = asyncio.Event()

            # Save existing handlers to chain to them
            original_sigterm_handler = signal.getsignal(signal.SIGTERM)
            original_sigint_handler = signal.getsignal(signal.SIGINT)

            def signal_handler(signum, frame):
                logger.info(f"Received signal {signum}, initiating graceful shutdown")
                shutdown_event.set()

                # Chain to original handler if it exists
                if signum == signal.SIGTERM:
                    if original_sigterm_handler and original_sigterm_handler not in (
                        signal.SIG_DFL,
                        signal.SIG_IGN,
                    ):
                        original_sigterm_handler(signum, frame)
                elif signum == signal.SIGINT:
                    if original_sigint_handler and original_sigint_handler not in (
                        signal.SIG_DFL,
                        signal.SIG_IGN,
                    ):
                        original_sigint_handler(signum, frame)

            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

            # Create a single AsyncClient to be shared across all requests
            # Set limits based on max expected workers (passed from parent)
            # We need exactly max_workers connections since health checks and work calls
            # should reuse the same connection per worker (HTTP keep-alive)
            # Add small buffer for edge cases
            buffer = max(100, int(max_workers * 0.1))  # 10% buffer, min 100
            max_conn = max_workers + buffer

            limits = httpx.Limits(
                max_keepalive_connections=max_conn,  # Keep all connections alive
                max_connections=max_conn,  # One connection per worker + buffer
                keepalive_expiry=300.0,  # Keep connections alive for 5 minutes
            )
            timeout = httpx.Timeout(
                connect=10.0,
                read=None,  # No read timeout for long-running jobs
                write=10.0,
                pool=60.0,  # Time to wait for a connection from pool
            )

            logger.debug(
                f"AsyncClient configured with max_connections={max_conn} "
                f"(workers={max_workers} + buffer={buffer}) with 5min keepalive"
            )

            # Note: http2=True would enable HTTP/2 if server supports it
            async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
                logger.debug("Async worker started with shared httpx.AsyncClient")

                active_tasks = set()

                async def process_call(request_data):
                    """Process a CALL request asynchronously."""
                    request_id = request_data["request_id"]
                    worker_ips = request_data["worker_ips"]
                    logger.debug(
                        f"Async worker: Processing request {request_id} with {len(worker_ips)} workers: {worker_ips}"
                    )
                    try:
                        results = await call_workers_async(client, request_data)
                        logger.debug(
                            f"Async worker: Successfully got {len(results)} results for request {request_id}, sending response"
                        )
                        response_queue.put({"request_id": request_id, "results": results})
                    except Exception as e:
                        logger.error(f"Error processing request {request_id}: {e}")
                        response_queue.put({"request_id": request_id, "error": e})

                # Main request processing loop
                while not shutdown_event.is_set():
                    try:
                        # Check queue without blocking event loop
                        cmd, request_data = await asyncio.to_thread(request_queue.get, block=True, timeout=0.1)

                        if cmd == "SHUTDOWN" or shutdown_event.is_set():
                            # Cancel all active tasks immediately for quick cleanup
                            if active_tasks:
                                logger.debug(f"Cancelling {len(active_tasks)} active tasks for shutdown")
                                for task in active_tasks:
                                    task.cancel()
                            break

                        elif cmd == "CALL":
                            # Create a task to handle this request concurrently
                            task = asyncio.create_task(process_call(request_data))
                            active_tasks.add(task)

                        # Clean up completed tasks periodically
                        active_tasks = {t for t in active_tasks if not t.done()}

                    except queue.Empty:
                        # Clean up completed tasks while waiting
                        active_tasks = {t for t in active_tasks if not t.done()}
                    except Exception as e:
                        logger.error(f"Error in async worker loop: {e}")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(main())
        finally:
            loop.close()
