import copy
import multiprocessing
import os
import queue
import subprocess
import threading
import time
import uuid
from bdb import BdbQuit
from concurrent.futures import as_completed, ThreadPoolExecutor
from typing import Dict, Optional

import httpx
from starlette.responses import JSONResponse

from kubetorch.servers.http.http_server import (
    load_callable,
    logger,
    package_exception,
    patch_sys_path,
    request_id_ctx_var,
    run_callable_internal_sync,
)

from .utils import clear_debugging_sessions, is_running_in_kubernetes

# Try to import Monarch components at module level if available
# This helps avoid threading issues with Monarch's Rust bindings
try:
    from monarch._src.actor.allocator import RemoteAllocator, StaticRemoteAllocInitializer

    MONARCH_AVAILABLE = True
except ImportError:
    MONARCH_AVAILABLE = False
    RemoteAllocator = None
    StaticRemoteAllocInitializer = None
except Exception:
    # Catch any other exceptions during import
    MONARCH_AVAILABLE = False
    RemoteAllocator = None
    StaticRemoteAllocInitializer = None


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


class DistributedProcessPool:
    """Unified pool managing distributed processes with single router thread."""

    def __init__(self, process_class, num_processes, max_threads_per_proc=10, **process_kwargs):
        self.process_class = process_class
        self.num_processes = num_processes
        self.max_threads_per_proc = max_threads_per_proc
        self.process_kwargs = process_kwargs  # Additional kwargs to pass to process constructor

        # Processes and queues
        self.processes = []
        self.request_queues = []  # One request queue per process
        self.response_queue = multiprocessing.Queue()  # Single shared response queue

        # Response routing with single router thread
        self._router_thread = None
        self._response_events = {}  # Maps request_id to (threading.Event, response)
        self._response_lock = threading.Lock()
        self._running = False

    def start(self):
        """Start all processes in the pool and the single router thread."""
        if self.processes:
            raise RuntimeError("Pool already started")

        # Create and start all processes in parallel
        with ThreadPoolExecutor(max_workers=self.num_processes) as executor:
            futures = []
            for i in range(self.num_processes):
                future = executor.submit(self._create_and_start_process, i)
                futures.append(future)

            # Wait for all processes to start
            for future in futures:
                future.result()

        # Start single response router thread for entire pool
        self._running = True
        self._router_thread = threading.Thread(target=self._response_router, daemon=True, name="PoolResponseRouter")
        self._router_thread.start()
        logger.debug(f"Started {self.num_processes} processes with single router thread")

    def _create_and_start_process(self, local_rank):
        """Helper to create and start a single process."""
        request_queue = multiprocessing.Queue()
        self.request_queues.append(request_queue)

        process = self.process_class(
            local_rank=local_rank,
            request_queue=request_queue,
            response_queue=self.response_queue,  # Shared response queue
            max_threads=self.max_threads_per_proc,
            **self.process_kwargs,  # Pass additional framework-specific settings
        )
        process.start()
        self.processes.append(process)

    def stop(self):
        """Stop all processes and the router thread."""
        self._running = False

        # Send shutdown signal to all processes (use put_nowait to avoid blocking)
        for q in self.request_queues:
            try:
                q.put_nowait("SHUTDOWN")
            except Exception:
                pass

        # Stop router thread
        try:
            self.response_queue.put_nowait("STOP_ROUTER")
        except Exception:
            pass

        # Wait briefly for router to stop
        if self._router_thread:
            self._router_thread.join(timeout=0.5)

        # Terminate all processes immediately without waiting
        for process in self.processes:
            if process.is_alive():
                process.terminate()

        # Give processes a brief chance to terminate gracefully (reduced timeout)
        for process in self.processes:
            process.join(timeout=0.5)

        # Force kill any remaining processes
        for process in self.processes:
            if process.is_alive():
                logger.warning(f"Force killing process {process.pid}")
                process.kill()
                process.join(timeout=0.1)  # Brief wait to confirm kill

        # Clear all queues
        self._clear_queues()

        # Reset state
        self.processes.clear()
        self.request_queues.clear()
        with self._response_lock:
            self._response_events.clear()

    def call(
        self,
        idx,
        method_name,
        params,
        deployed_as_of,
        request_id,
        distributed_env_vars,
        debug_port,
        serialization,
    ):
        """Call a specific process by index."""
        if idx >= len(self.processes):
            raise ValueError(f"Process index {idx} out of range (have {len(self.processes)} processes)")

        request_unique_id = str(uuid.uuid4())

        # Register this request for response routing
        event = threading.Event()
        with self._response_lock:
            self._response_events[request_unique_id] = (event, None)

        try:
            # Send request to specific process
            self.request_queues[idx].put(
                {
                    "request_unique_id": request_unique_id,
                    "method_name": method_name,
                    "params": params,
                    "deployed_as_of": deployed_as_of,
                    "request_id": request_id,
                    "distributed_env_vars": distributed_env_vars,
                    "debug_port": debug_port,
                    "serialization": serialization,
                    "process_idx": idx,  # Include process index for debugging
                }
            )

            # Wait for response
            event.wait()

            # Get and return the response
            with self._response_lock:
                _, result = self._response_events.pop(request_unique_id)

            return result

        except Exception:
            # Clean up on error
            with self._response_lock:
                self._response_events.pop(request_unique_id, None)
            raise

    def call_all(
        self,
        method_name,
        params_list,
        deployed_as_of,
        request_id,
        distributed_env_vars_list,
        debug_ports,
        serialization,
    ):
        """Call all processes in parallel and return results."""
        if len(params_list) != self.num_processes:
            raise ValueError(f"Expected {self.num_processes} param sets, got {len(params_list)}")

        with ThreadPoolExecutor(max_workers=self.num_processes) as executor:
            futures = []
            for idx in range(self.num_processes):
                future = executor.submit(
                    self.call,
                    idx=idx,
                    method_name=method_name,
                    params=params_list[idx],
                    deployed_as_of=deployed_as_of,
                    request_id=request_id,
                    distributed_env_vars=distributed_env_vars_list[idx],
                    debug_port=debug_ports[idx] if debug_ports else None,
                    serialization=serialization,
                )
                futures.append(future)

            results = []
            for future in futures:
                results.append(future.result())

        return results

    def _response_router(self):
        """Single router thread handling responses from all processes."""
        while self._running:
            try:
                response = self.response_queue.get(timeout=1)
                if response == "STOP_ROUTER":
                    break

                request_id = response.get("request_unique_id")
                with self._response_lock:
                    if request_id in self._response_events:
                        event, _ = self._response_events[request_id]
                        self._response_events[request_id] = (event, response["result"])
                        event.set()
                    else:
                        logger.warning(f"Received response for unknown request: {request_id}")

            except Exception as e:
                if "Empty" not in str(e.__class__.__name__):
                    logger.debug(f"Response router error: {e}")
                continue

    def _clear_queues(self):
        """Clear all pending items from queues."""
        for q in self.request_queues:
            try:
                while not q.empty():
                    q.get_nowait()
            except Exception:
                pass

        try:
            while not self.response_queue.empty():
                self.response_queue.get_nowait()
        except Exception:
            pass

    def __len__(self):
        return len(self.processes)

    def __getitem__(self, idx):
        return self.processes[idx]


class DistributedSupervisor:
    def __init__(self, quorum_workers=None, quorum_timeout=300, monitor_members=True):
        """
        Base class for distributed supervisors. This class should be subclassed for specific distributed
        environments like PyTorch or Ray.

        Args:
            config: Optional configuration object for the distributed environment.
        """
        # Set after creation by the factory function
        self.quorum_workers = quorum_workers
        self.quorum_timeout = quorum_timeout
        self.monitor_members = monitor_members

        self.config_hash = None

        # DNS monitoring state
        self._dns_monitor_thread = None
        self._dns_monitor_running = False
        self._current_workers = set()
        self._workers_lock = threading.Lock()
        self._membership_changes = queue.Queue()
        self._change_subscribers = []
        self._last_dns_check = 0
        self._dns_check_interval = 5  # seconds

    def pod_ips(self):
        """Get pod IPs from DNS, waiting for quorum if specified.

        Will wait up to quorum_timeout seconds for quorum_workers to appear in DNS.
        If quorum_workers is not specified, returns immediately after first DNS query.
        """
        # Primarily for testing
        if not is_running_in_kubernetes():
            return os.environ["LOCAL_IPS"].split(",")

        # Use DNS-based service discovery instead of Kubernetes API
        # Check if pre-computed DNS name is available (should point to headless service for distributed)
        service_dns = os.environ.get("KT_SERVICE_DNS")

        if not service_dns:
            # Fall back to computing DNS name from service and namespace
            service_name = os.environ.get("KT_SERVICE_NAME")
            namespace = os.environ.get("POD_NAMESPACE")

            if not service_name:
                raise RuntimeError("KT_SERVICE environment variable not found")
            if not namespace:
                raise RuntimeError("POD_NAMESPACE environment variable not found")

            # Kubernetes headless service DNS name for distributed pod discovery
            # Format: <service-name>-headless.<namespace>.svc.cluster.local
            service_dns = f"{service_name}-headless.{namespace}.svc.cluster.local"

        import socket
        import time

        start_time = time.time()
        max_wait = self.quorum_timeout if self.quorum_timeout else 0
        expected_workers = self.quorum_workers

        pod_ips = []
        last_count = 0

        while True:
            try:
                # DNS lookup returns all pod IPs for the headless service
                # getaddrinfo returns list of (family, type, proto, canonname, sockaddr)
                addr_info = socket.getaddrinfo(service_dns, None, socket.AF_INET)

                # Extract unique IP addresses from the results
                pod_ips = sorted(list(set([addr[4][0] for addr in addr_info])))

                if not pod_ips:
                    logger.debug(f"No pod IPs found for service {service_dns}")
                else:
                    logger.debug(f"Found {len(pod_ips)} pod IPs via DNS for {service_dns}: {pod_ips}")

            except socket.gaierror as e:
                logger.debug(f"DNS lookup failed for {service_dns}: {e}")
                pod_ips = []

            # Check if we should wait for more workers
            elapsed = time.time() - start_time

            # If we have the expected count, we're done
            if expected_workers and len(pod_ips) >= expected_workers:
                logger.info(f"Found {len(pod_ips)}/{expected_workers} workers after {elapsed:.1f}s")
                return pod_ips

            # If we don't have expected count or timeout is reached, decide what to do
            if elapsed >= max_wait:
                if expected_workers:
                    logger.warning(f"Only found {len(pod_ips)}/{expected_workers} workers after {elapsed:.1f}s timeout")
                else:
                    logger.info(f"Found {len(pod_ips)} workers after {elapsed:.1f}s")
                return pod_ips

            # Log progress if count changed
            if len(pod_ips) != last_count:
                if expected_workers:
                    logger.info(f"{len(pod_ips)}/{expected_workers} workers found, waiting for quorum...")
                else:
                    logger.debug(f"{len(pod_ips)} workers found, no quorum set")
                last_count = len(pod_ips)

            # Wait before retrying
            time.sleep(2)

    def _get_pod_ips_fast(self):
        """Get pod IPs from DNS without waiting for quorum - for monitoring only."""
        # Primarily for testing
        if not is_running_in_kubernetes():
            return os.environ["LOCAL_IPS"].split(",")

        # Use DNS-based service discovery
        service_dns = os.environ.get("KT_SERVICE_DNS")

        if not service_dns:
            service_name = os.environ.get("KT_SERVICE")
            namespace = os.environ.get("POD_NAMESPACE")

            if not service_name or not namespace:
                return []

            service_dns = f"{service_name}-headless.{namespace}.svc.cluster.local"

        import socket

        try:
            # Single DNS lookup, no retries, no waiting
            addr_info = socket.getaddrinfo(service_dns, None, socket.AF_INET)
            # Extract unique IP addresses
            pod_ips = sorted(list(set([addr[4][0] for addr in addr_info])))
            return pod_ips
        except socket.gaierror:
            # DNS lookup failed, return current known workers
            with self._workers_lock:
                return list(self._current_workers)

    def start_dns_monitoring(self):
        """Start DNS monitoring if not already running.
        Should be called by coordinator nodes only."""
        # Skip if monitoring is disabled (e.g., for Ray)
        if not self.monitor_members:
            logger.debug("DNS monitoring disabled for this supervisor")
            return

        with self._workers_lock:
            if self._dns_monitor_thread and self._dns_monitor_thread.is_alive():
                return  # Already running

            # Initialize with current workers
            self._current_workers = set(self.pod_ips())
            logger.debug(f"Starting DNS monitor with {len(self._current_workers)} workers")

            self._dns_monitor_running = True
            self._dns_monitor_thread = threading.Thread(
                target=self._monitor_worker_membership, daemon=True, name="DNSMonitor"
            )
            self._dns_monitor_thread.start()

    def stop_dns_monitoring(self):
        """Stop DNS monitoring thread."""
        self._dns_monitor_running = False
        if self._dns_monitor_thread:
            self._dns_monitor_thread.join(timeout=2)
            self._dns_monitor_thread = None

    def _monitor_worker_membership(self):
        """Monitor DNS for worker membership changes."""
        check_interval = 3  # Start with 3 second checks (faster initial detection)

        while self._dns_monitor_running:
            try:
                # Note that we start this after the delay, because we're doing a DNS check at
                # the start of call_distributed anyway. This thread is only for the recurring checks
                # as the call runs.
                time.sleep(check_interval)

                # Query DNS for current workers - use a faster version
                current_ips = set(self._get_pod_ips_fast())

                with self._workers_lock:
                    if current_ips != self._current_workers:
                        added = current_ips - self._current_workers
                        removed = self._current_workers - current_ips

                        change = {
                            "timestamp": time.time(),
                            "added": added,
                            "removed": removed,
                            "previous": self._current_workers.copy(),
                            "current": current_ips.copy(),
                        }

                        if removed:
                            logger.error(f"Workers REMOVED from cluster: {removed}")
                        if added:
                            logger.warning(f"Workers ADDED to cluster: {added}")

                        # Queue change and notify subscribers
                        self._membership_changes.put(change)
                        for event in self._change_subscribers:
                            event.set()

                        self._current_workers = current_ips

                time.sleep(check_interval)

            except Exception as e:
                logger.error(f"DNS monitor error: {e}")
                time.sleep(3)

    def subscribe_to_membership_changes(self):
        """Subscribe to worker membership changes.
        Returns an event that will be set when changes occur."""
        event = threading.Event()
        with self._workers_lock:
            self._change_subscribers.append(event)
        return event

    def unsubscribe_from_membership_changes(self, event):
        """Unsubscribe from worker membership changes."""
        with self._workers_lock:
            if event in self._change_subscribers:
                self._change_subscribers.remove(event)

    def check_for_membership_changes(self, force_dns_check=False):
        """Check for membership changes and raise exception if any occurred.

        Args:
            force_dns_check: If True, immediately query DNS to check for changes
                            instead of relying on the monitoring thread
        """
        # Skip if monitoring is disabled (e.g., for Ray)
        if not self.monitor_members:
            return
        # Force an immediate DNS check if requested
        if force_dns_check:
            # Use fast DNS query for immediate check
            current_ips = set(self._get_pod_ips_fast())
            with self._workers_lock:
                if current_ips != self._current_workers:
                    added = current_ips - self._current_workers
                    removed = self._current_workers - current_ips

                    # Import here to avoid circular dependency
                    from kubetorch.servers.http.utils import WorkerMembershipChanged

                    # Update current workers
                    self._current_workers = current_ips

                    # Log the change
                    if removed:
                        logger.error(f"Workers REMOVED from cluster (forced check): {removed}")
                    if added:
                        logger.warning(f"Workers ADDED to cluster (forced check): {added}")

                    raise WorkerMembershipChanged(
                        added_ips=added,
                        removed_ips=removed,
                        previous_ips=self._current_workers.copy(),
                        current_ips=current_ips,
                    )

        # Check queued changes from monitoring thread
        try:
            change = self._membership_changes.get_nowait()

            # Import here to avoid circular dependency
            from kubetorch.servers.http.utils import WorkerMembershipChanged

            raise WorkerMembershipChanged(
                added_ips=change["added"],
                removed_ips=change["removed"],
                previous_ips=change["previous"],
                current_ips=change["current"],
            )
        except queue.Empty:
            pass  # No changes

    def setup(self, deployed_as_of: Optional[str] = None):
        # This method should be overridden by subclasses to set up the distributed environment
        raise NotImplementedError("setup() must be implemented by subclasses")

    def cleanup(self):
        """Base cleanup - stop DNS monitoring. Subclasses should call super().cleanup()"""
        self.stop_dns_monitoring()
        # Subclasses should override and call super().cleanup() to add their own cleanup

    def intercept_call(self):
        # This method should be overridden by subclasses to indicate whether to intercept calls
        raise NotImplementedError("intercept_call() must be implemented by subclasses")

    def call_distributed(
        self,
        request,
        cls_or_fn_name: str,
        method_name: Optional[str] = None,
        params: Optional[Dict] = None,
        distributed_subcall: bool = False,
        debug_port: int = False,
        deployed_as_of: Optional[str] = None,
    ):
        # if intercept_call is True, this method should be overridden by subclasses to handle distributing and/or
        # supervising the distributed execution
        raise NotImplementedError("call_distributed() must be implemented by subclasses")


class DistributedProcess(multiprocessing.Process):
    """Base class for distributed processes that run callables in subprocesses."""

    def __init__(self, local_rank, request_queue, response_queue, max_threads=4, **kwargs):
        super().__init__()
        # We don't need the cache miss / reload here because these processes are destroyed and recreated
        # with each .to call.
        os.environ["LOCAL_RANK"] = str(local_rank)
        self._request_queue = request_queue
        self._response_queue = response_queue
        self._max_threads = max_threads
        self._executor = None
        # Store any additional framework-specific settings
        self._settings = kwargs

    def proc_cleanup(self):
        """Override this method to provide framework-specific cleanup."""
        logger.info("Cleaning up debugging sessions...")
        clear_debugging_sessions()
        logger.info("Debugging sessions cleaned up.")

        # Cleanup thread pool
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None

    @classmethod
    def get_distributed_env_vars(cls, worker_ips, node_rank, local_rank, num_local_procs, **settings):
        """Get framework-specific distributed environment variables.

        Args:
            worker_ips: List of all worker IPs
            node_rank: Rank of this node (0-indexed)
            local_rank: Local rank on this node (0-indexed)
            num_local_procs: Number of processes on this node
            **settings: Additional framework-specific settings (e.g., port)

        Returns:
            Dict of environment variables to set
        """
        # Base implementation - no special env vars needed
        return {
            "WORLD_SIZE": str(len(worker_ips) * num_local_procs),
            "RANK": str(node_rank * num_local_procs + local_rank),
            "LOCAL_RANK": str(local_rank),
            "NODE_RANK": str(node_rank),
            "POD_IPS": ",".join(worker_ips),
        }

    @classmethod
    def get_auto_num_processes(cls):
        """Auto-detect the number of processes to use."""
        return 1

    def handle_request(self, request):
        """Handle a single request in a thread."""
        try:
            request_unique_id = request["request_unique_id"]
            method_name = request["method_name"]
            params = request["params"]
            deployed_as_of = request["deployed_as_of"]
            request_id = request["request_id"]
            distributed_env_vars = request["distributed_env_vars"]
            debug_port = request["debug_port"]
            serialization = request["serialization"]

            # Set the request ID in the context for this thread
            token = request_id_ctx_var.set(request_id)

            # Set the environment variables for this thread (note: os.environ is process-wide, might need thread-local storage)
            # For distributed PyTorch calls, these should already be set at process level
            for key, value in distributed_env_vars.items():
                os.environ[key] = value

            try:
                # Load callable if not already loaded or if deployed_as_of changed
                callable_obj = load_callable(
                    deployed_as_of=deployed_as_of,
                    distributed_subprocess=True,
                    reload_cleanup_fn=self.proc_cleanup,
                )

                result = run_callable_internal_sync(
                    callable_obj=callable_obj,
                    cls_or_fn_name=os.environ["KT_CLS_OR_FN_NAME"],
                    method_name=method_name,
                    params=params,
                    serialization=serialization,
                    debug_port=debug_port,
                )

                # Reset the request ID after the call is complete
                request_id_ctx_var.reset(token)

                # Send response back with the unique ID
                self._response_queue.put({"request_unique_id": request_unique_id, "result": result})

            except Exception as e:
                # Reset the request ID even if there was an error
                request_id_ctx_var.reset(token)

                # Package the exception
                try:
                    packaged_exception = package_exception(e)
                except Exception as f:
                    packaged_exception = f

                self._response_queue.put(
                    {
                        "request_unique_id": request_unique_id,
                        "result": packaged_exception,
                    }
                )

        except Exception as e:
            # Last resort error handling
            logger.error(f"Fatal error handling request: {e}")
            self._response_queue.put(
                {
                    "request_unique_id": request.get("request_unique_id", "unknown"),
                    "result": Exception(f"Fatal error in thread: {e}"),
                }
            )

    def run(self):
        """Main process loop with thread pool for concurrent request handling."""
        # Create thread pool for handling requests
        self._executor = ThreadPoolExecutor(max_workers=self._max_threads)

        try:
            while True:
                try:
                    # Block waiting for next request
                    request = self._request_queue.get(timeout=1)

                    # Special sentinel value to signal shutdown
                    if request == "SHUTDOWN":
                        break

                    # Submit request to thread pool for concurrent handling
                    # Check executor exists in case we're shutting down
                    if self._executor:
                        self._executor.submit(self.handle_request, request)
                    else:
                        logger.warning("Executor is None, skipping request (likely shutting down)")

                except Exception as e:
                    # Timeout is normal, continue loop
                    if "Empty" not in str(e.__class__.__name__):
                        logger.error(f"Error getting request from queue: {e}")
                    continue

        except (KeyboardInterrupt, BdbQuit):
            logger.info("Process interrupted, shutting down...")

        finally:
            # Cleanup
            logger.info("Received shutdown signal, cleaning up distributed environment...")
            self.proc_cleanup()
            logger.info("Exiting gracefully.")


class PyTorchProcess(DistributedProcess):
    """PyTorch-specific distributed process."""

    def proc_cleanup(self):
        import torch.distributed as dist

        try:
            dist.destroy_process_group()
            logger.info("Destroyed PyTorch process group.")
        except Exception:
            logger.info("Failed to destroy PyTorch process group, it may not have been initialized: {e}")
            pass
        # Call parent cleanup for debugging sessions
        super().proc_cleanup()

    @classmethod
    def get_distributed_env_vars(cls, worker_ips, node_rank, local_rank, num_local_procs, **settings):
        """Get PyTorch-specific distributed environment variables."""
        port = settings.get("port") or 12345
        env_vars = super().get_distributed_env_vars(worker_ips, node_rank, local_rank, num_local_procs, **settings)
        env_vars.update(
            {
                "MASTER_ADDR": worker_ips[0],
                "MASTER_PORT": str(port),
            }
        )
        return env_vars

    @classmethod
    def get_auto_num_processes(cls):
        """Auto-detect based on GPU availability for PyTorch."""
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.device_count()
        except ImportError:
            pass
        return 1  # Could use os.cpu_count() for CPU-only training


class RayProcess(DistributedProcess):
    """Ray-specific distributed process."""

    def proc_cleanup(self):
        try:
            import ray

            if ray.is_initialized():
                ray.shutdown()
                logger.info("Ray shutdown completed.")
        except ImportError:
            logger.info("Ray not available for cleanup")
        except Exception as e:
            logger.info(f"Failed to shutdown Ray: {e}")
        # Call parent cleanup for debugging sessions
        super().proc_cleanup()


class SPMDDistributedSupervisor(DistributedSupervisor):
    """Base class for SPMD (Single Program Multiple Data) distributed supervisors.

    This class provides common functionality for frameworks that follow the SPMD pattern
    where the same program runs on multiple processes with different data partitions.
    """

    def __init__(
        self,
        process_class=None,
        num_proc=None,
        port=None,
        restart_procs=True,
        max_threads_per_proc=10,
        quorum_timeout=300,
        quorum_workers=None,
        monitor_members=True,
        tree_fanout=50,
        tree_minimum=100,
        **process_kwargs,
    ):
        super().__init__(
            quorum_workers=quorum_workers,
            quorum_timeout=quorum_timeout,
            monitor_members=monitor_members,
        )
        self.process_class = process_class or DistributedProcess
        self.num_proc = num_proc or "auto"
        self.port = port
        self.restart_procs = restart_procs
        self.max_threads_per_proc = max_threads_per_proc
        self.process_pool = None
        self.remote_worker_pool = None  # Pool for async HTTP calls to remote workers
        self.process_kwargs = process_kwargs  # Additional settings to pass to process class
        self.tree_fanout = tree_fanout
        self.tree_minimum = tree_minimum

    def setup(self, deployed_as_of: Optional[str] = None):
        # Set multiprocessing to spawn if not already
        if multiprocessing.get_start_method() != "spawn":
            multiprocessing.set_start_method("spawn", force=True)

        # Get number of processes
        if self.num_proc == "auto":
            num_proc = self.process_class.get_auto_num_processes()
        else:
            num_proc = self.num_proc

        if self.restart_procs:
            logger.debug("restart_procs is True, restarting distributed processes")
            self.cleanup()

        # If the number of processes has changed, we need to clean up the old ones and recreate them
        if self.process_pool is None or len(self.process_pool) != num_proc:
            if self.process_pool:
                logger.debug(
                    f"Number of processes changed from {len(self.process_pool)} to {num_proc}, restarting processes."
                )
                self.cleanup()

            logger.debug("Setting up distributed environment")
            self.process_pool = DistributedProcessPool(
                process_class=self.process_class,
                num_processes=num_proc,
                max_threads_per_proc=self.max_threads_per_proc,
                **self.process_kwargs,  # Pass any additional settings
            )

            # Start all processes (now handled internally by the pool)
            self.process_pool.start()

            self.remote_worker_pool = RemoteWorkerPool(quorum_timeout=self.quorum_timeout)
            self.remote_worker_pool.start()
            logger.debug("Finished setting up distributed processes")

    def cleanup(self):
        # Cleanup the processes
        logger.debug(f"Cleaning up {self.__class__.__name__} distributed processes")

        # Stop DNS monitoring first
        super().cleanup()

        if self.process_pool:
            self.process_pool.stop()
            self.process_pool = None

        if self.remote_worker_pool:
            self.remote_worker_pool.stop()
            self.remote_worker_pool = None

        logger.debug(f"Finished cleaning up {self.__class__.__name__} distributed processes")

    @staticmethod
    def intercept_call():
        return True

    def get_tree_children(self, sorted_ips: list, my_ip: str, fanout: int = 100):
        """Calculate children nodes in a self-organizing tree based on IP indexing.

        Args:
            sorted_ips: List of all worker IPs sorted deterministically
            my_ip: This node's IP address
            fanout: Maximum number of children per node (default 100)

        Returns:
            List of IP addresses that are children of this node
        """
        try:
            my_index = sorted_ips.index(my_ip)
        except ValueError:
            # If not found in list, this node has no children
            return []

        # Calculate the range of children indices
        # In a tree with fanout F, node at index i has children at indices:
        # [i*F + 1, i*F + 2, ..., i*F + F]
        first_child_idx = my_index * fanout + 1
        last_child_idx = min(first_child_idx + fanout, len(sorted_ips))

        if first_child_idx >= len(sorted_ips):
            # No children (leaf node)
            return []

        children = sorted_ips[first_child_idx:last_child_idx]
        if len(children) > 0:
            logger.debug(
                f"Tree topology: Node {my_ip} (index {my_index}) has {len(children)} children "
                f"(indices {first_child_idx}-{last_child_idx-1})"
            )
        return children

    def call_distributed(
        self,
        request,
        cls_or_fn_name: str,
        method_name: Optional[str] = None,
        params: Optional[Dict] = None,
        distributed_subcall: bool = False,
        debug_port: int = False,
        deployed_as_of: Optional[str] = None,
    ):
        # Get the request ID from the headers
        request_id = request.headers.get("X-Request-ID", "-")
        serialization = request.headers.get("X-Serialization", "json")
        params = params or {}

        # If deployed_as_of is None and we're the coordinator, generate a consistent timestamp
        # to use across all workers to prevent reload inconsistencies
        if not distributed_subcall and deployed_as_of is None:
            from datetime import datetime, timezone

            deployed_as_of = datetime.now(timezone.utc).isoformat()

        # Get all the pods in the service, and use the first one as the master.
        # Set the env vars based on whether this is a master or worker
        logger.debug(f"Configuring distributed environment, distributed_subcall={distributed_subcall}")
        this_pod_ip = os.environ["POD_IP"]
        logger.debug(f"This pod IP: {this_pod_ip}")

        # Start DNS monitoring for coordinator nodes
        change_event = None
        if not distributed_subcall:
            # First wait for quorum before starting monitoring
            worker_ips = self.pod_ips()
            # sort the worker IPs to ensure generally consistent tree ordering
            # (avoids thrashing connections due to reordering)
            worker_ips.sort()
            # For tree topology, coordinator is always root (index 0)
            # Move coordinator to front if not already there
            if this_pod_ip in worker_ips:
                worker_ips.remove(this_pod_ip)
            worker_ips.insert(0, this_pod_ip)  # Move coordinator to root of tree
            logger.debug(f"Acting as COORDINATOR - discovered worker IPs: {worker_ips}")
            logger.debug(f"Pod IPs: {worker_ips}")

            # Check if this call uses flexible worker selection (workers="ready")
            # If so, don't start DNS monitoring as the worker set is expected to be flexible
            workers_arg = params.get("workers") if params else None
            should_monitor = workers_arg not in ["ready", "any"]

            if should_monitor:
                # Now that we have quorum, start DNS monitoring
                # Start monitoring (idempotent - won't start if already running)
                self.start_dns_monitoring()

                # Subscribe to membership changes
                change_event = self.subscribe_to_membership_changes()

                # Check for any pending changes after starting monitor
                self.check_for_membership_changes(force_dns_check=True)
            else:
                logger.debug("Skipping DNS monitoring for workers='ready' call")

            # Update distributed env vars to use the tree-ordered IPs
            distributed_env_vars = {
                "POD_IPS": ",".join(worker_ips),
            }
        else:
            logger.debug(f"Acting as WORKER (distributed_subcall=True) at {this_pod_ip}")
            logger.debug(f"Worker received params keys: {list(params.keys()) if params else 'None'}")
            distributed_env_vars = params.pop("distributed_env_vars", None) if params else None
            logger.debug(f"Using distributed_env_vars: {distributed_env_vars}")
            if not distributed_env_vars:
                logger.error(f"No distributed_env_vars found in params: {params}")
                raise RuntimeError("distributed_env_vars must be provided for distributed subcalls")
            worker_ips = distributed_env_vars["POD_IPS"].split(",")

            # Don't debug for subcalls, we only want to debug one process
            debug_port = None

        # Decide topology based on cluster size
        subcall_ips = []
        num_workers = len(worker_ips)
        tree_mode = num_workers >= self.tree_minimum
        if tree_mode:
            # Use tree topology for large clusters
            if distributed_subcall:
                logger.debug(
                    f"Using tree topology for {num_workers} workers (> {self.tree_minimum} threshold) "
                    f"with fanout {self.tree_fanout}"
                )

            # Calculate direct children in the tree
            subcall_ips = self.get_tree_children(
                sorted_ips=worker_ips,
                my_ip=this_pod_ip,
                fanout=self.tree_fanout,  # Each node can have up to 50 children
            )
            logger.debug(f"Tree node {this_pod_ip} will call {len(subcall_ips)} direct children")
        elif not distributed_subcall:
            # Use worker ip list as is for coordiantor node in flat topology
            # Leave subcall_ips = [] for workers in flat topology
            subcall_ips = copy.deepcopy(worker_ips)
            if this_pod_ip in subcall_ips:
                subcall_ips.remove(this_pod_ip)
                logger.debug(f"Removed self ({this_pod_ip}) from subcall list, will call: {subcall_ips}")
            else:
                # This can happen with headless services where POD_IP might not exactly match DNS results
                # Try to match by partial IP or hostname
                logger.warning(
                    f"This pod IP {this_pod_ip} not found in DNS-discovered IPs {worker_ips}. "
                    f"This may indicate DNS propagation delay or hostname/IP mismatch."
                )
                # Still proceed as coordinator but will call all discovered pods
                logger.debug(f"Will call all discovered workers: {subcall_ips}")

        # Always pass the distributed environment variables to workers
        params["distributed_env_vars"] = distributed_env_vars

        # "workers" is passed through as a regular kwarg, not a special extracted one like pdb or serialization
        # because it only applies to distributed calls, not regular ones.
        call_local_procs = True
        workers_arg = params.get("workers", None)
        if workers_arg:
            logger.debug(f"Filtering workers by argument: {workers_arg}")
            if isinstance(workers_arg, list) and workers_arg:
                # Build a set of IPs to include based on the list items
                target_ips = set()

                for item in workers_arg:
                    if isinstance(item, str) and "." in item:
                        # It's an IP address
                        if item not in worker_ips:
                            raise ValueError(f"Worker IP '{item}' not found in available workers: {worker_ips}")
                        target_ips.add(item)
                    elif isinstance(item, int) or (isinstance(item, str) and item.isdigit()):
                        # It's an index
                        idx = int(item) if isinstance(item, str) else item
                        if idx < 0 or idx >= len(worker_ips):
                            raise ValueError(f"Worker index {idx} out of range. Valid range: 0-{len(worker_ips)-1}")
                        target_ips.add(worker_ips[idx])
                    else:
                        raise ValueError(
                            f"Invalid worker specification: {item}. Must be an IP address, "
                            f"integer index, or numeric string."
                        )

                # Filter subcall_ips to only those in the target set
                subcall_ips = [ip for ip in subcall_ips if ip in target_ips]

                # Check if current pod should participate in local processing
                if this_pod_ip not in target_ips:
                    call_local_procs = False

            elif workers_arg == "any":
                # Only call one worker (this one)
                subcall_ips = []
            elif workers_arg == "ready":
                # Filter below in call_worker to only those that respond to /health
                pass
            elif isinstance(workers_arg, str) and workers_arg:
                # Filter the subcall_ips to only those matching the workers_arg string
                subcall_ips = [ip for ip in subcall_ips if workers_arg in ip]
            logger.debug(f"Subcall IPs after filtering: {subcall_ips}")

        # "restart_procs" is passed through as a regular kwarg, not a special extracted one like pdb or serialization
        # because it only applies to distributed calls, not regular ones.
        if params.get("restart_procs", False):
            logger.info("restart_procs parameter is True, restarting processes")
            self.cleanup()
            self.setup(deployed_as_of)

        try:
            node_rank = worker_ips.index(this_pod_ip)
        except ValueError:
            # This pod IP not found in DNS results - may be external service IP
            # Fall back to using POD_IP directly and assume node_rank based on position
            logger.warning(f"Pod IP {this_pod_ip} not found in DNS results {worker_ips}. Using fallback logic.")
            # For now, assume we're the first worker if not found
            node_rank = 0

        # Call the workers using RemoteWorkerPool for async operations
        def call_worker(worker_ip):
            # Keep this function for backward compatibility but it won't be used
            # when RemoteWorkerPool is available
            with httpx.Client(timeout=None) as client:
                port = os.environ["KT_SERVER_PORT"]
                worker_url = f"http://{worker_ip}:{port}"
                # First check that the worker is alive, replicas don't finish setup at exactly the same moment
                # Use quorum_timeout to control how long to wait for workers
                for i in range(int(self.quorum_timeout)):
                    try:
                        resp = client.get(f"{worker_url}/health")
                        if resp.status_code == 200:
                            break
                    except httpx.RequestError:
                        if workers_arg == "ready":
                            logger.debug(f"Worker {worker_ip} not ready, skipping as per 'ready' workers argument")
                            return None
                        time.sleep(1)
                else:
                    # Timeout reached without successful health check
                    logger.warning(f"Worker {worker_ip} failed to respond after {self.quorum_timeout}s timeout")
                    if workers_arg != "ready":
                        raise TimeoutError(
                            f"Worker {worker_ip} did not become ready within {self.quorum_timeout} seconds. "
                            "This may indicate the pod is still starting or there's a resource constraint. "
                            "Consider increasing quorum_timeout in .distribute() call."
                        )

                call_url = (
                    f"{worker_url}/{cls_or_fn_name}/{method_name}?distributed_subcall=true"
                    if method_name is not None
                    else f"{worker_url}/{cls_or_fn_name}?distributed_subcall=true"
                )

                # Clean headers to avoid potential Content-Length issues
                clean_headers = {}
                if request.headers:
                    for key, value in request.headers.items():
                        # Skip headers that could interfere with httpx's automatic handling
                        if key.lower() not in [
                            "content-length",
                            "transfer-encoding",
                            "connection",
                        ]:
                            clean_headers[key] = value

                try:
                    logger.debug(f"Making distributed call to {worker_url}")
                    resp = client.post(
                        url=call_url,
                        json=params,
                        headers=clean_headers,  # Includes deployed_as_of and request_id
                    )
                    return resp
                except (httpx.RequestError, httpx.HTTPError) as e:
                    logger.error(f"Failed to call worker {worker_url}: {e}")
                    raise

        # Prepare per-process parameters
        num_procs = len(self.process_pool)
        params_list = [params] * num_procs
        distributed_env_vars_list = []
        debug_ports = []

        for idx in range(num_procs):
            # Get framework-specific env vars from the process class
            env_vars = self.process_class.get_distributed_env_vars(
                worker_ips=worker_ips
                if "worker_ips" in locals()
                else distributed_env_vars.get("POD_IPS", "").split(","),
                node_rank=node_rank,
                local_rank=idx,
                num_local_procs=num_procs,
                port=self.port,
            )
            # Add any base env vars
            env_vars.update(distributed_env_vars)
            distributed_env_vars_list.append(env_vars)

            # Only debug one process and if debug_port is set
            debug = debug_port and idx == num_procs - 1
            if debug:
                time.sleep(0.25)
            debug_ports.append(debug_port if debug else None)

        # Execute distributed calls in parallel with local processes
        worker_responses = []
        worker_exception = None
        local_exception = None

        # Start both remote and local calls in parallel
        executor = ThreadPoolExecutor(max_workers=2)

        # Submit remote worker calls if needed
        worker_future = None
        if subcall_ips:
            logger.debug(f"Have {len(subcall_ips)} remote workers to call")
            if not self.remote_worker_pool:
                raise RuntimeError("RemoteWorkerPool not initialized. This is required for distributed execution.")
            logger.debug(f"Using existing RemoteWorkerPool to call {len(subcall_ips)} workers")

            def call_remote_workers():
                nonlocal worker_exception
                try:
                    # Prepare headers for remote workers
                    clean_headers = {}
                    if request.headers:
                        for key, value in request.headers.items():
                            if key.lower() not in [
                                "content-length",
                                "transfer-encoding",
                                "connection",
                            ]:
                                clean_headers[key] = value
                    # Always include deployed_as_of in headers for consistency
                    if deployed_as_of:
                        clean_headers["X-Deployed-As-Of"] = deployed_as_of

                    # Call remote workers asynchronously through the pool
                    logger.debug(f"Calling {len(subcall_ips)} remote workers via RemoteWorkerPool: {subcall_ips}")
                    results = self.remote_worker_pool.call_workers(
                        worker_ips=subcall_ips,
                        cls_or_fn_name=cls_or_fn_name,
                        method_name=method_name,
                        params=params,
                        request_headers=clean_headers,
                        workers_arg=workers_arg,
                    )
                    logger.warning(
                        f"RemoteWorkerPool returned {len(results) if results else 0} results from {len(subcall_ips)} workers"
                    )
                    return results
                except Exception as e:
                    # Check if this is a connection error - might indicate worker removal
                    if any(
                        err_type in str(e)
                        for err_type in [
                            "ReadError",
                            "TimeoutException",
                            "RequestError",
                            "HTTPError",
                            "ConnectionError",
                            "Connection reset",
                            "Connection closed",
                        ]
                    ):
                        logger.debug(f"Connection error detected: {e}, checking for membership changes")
                        # Force DNS check to see if workers were removed
                        self.check_for_membership_changes(force_dns_check=True)
                    worker_exception = e
                    raise

            worker_future = executor.submit(call_remote_workers)
            logger.debug(f"Submitted worker_future for {len(subcall_ips)} remote workers")

        else:
            logger.debug(f"No remote workers to call (subcall_ips is empty or None: {subcall_ips})")

        # Check if we need to initialize RemoteWorkerPool for tree topology workers
        if subcall_ips:
            if not self.remote_worker_pool:
                # Initialize RemoteWorkerPool if not already done (needed for tree topology workers)
                logger.warning(
                    f"INITIALIZING RemoteWorkerPool for tree worker at {this_pod_ip} to call {len(subcall_ips)} children"
                )
                self.remote_worker_pool = RemoteWorkerPool(quorum_timeout=self.quorum_timeout)
                self.remote_worker_pool.start(
                    max_workers=min(len(subcall_ips) + 50, 200)
                )  # Size for expected children plus buffer
                logger.warning(f"RemoteWorkerPool initialized successfully for {this_pod_ip}")
            elif (
                not hasattr(self.remote_worker_pool, "process")
                or not self.remote_worker_pool.process
                or not self.remote_worker_pool.process.is_alive()
            ):
                # Pool exists but not started/alive
                logger.warning(f"RemoteWorkerPool exists but not running for {this_pod_ip}, starting it now")
                self.remote_worker_pool.start(max_workers=min(len(subcall_ips) + 50, 200))

        if subcall_ips and not self.remote_worker_pool:
            # RemoteWorkerPool should always be initialized at this point
            raise RuntimeError(
                f"RemoteWorkerPool not available for worker at {this_pod_ip}. "
                "This is required for distributed execution with subcall_ips."
            )

        # Submit local process calls
        def call_local_processes():
            logger.debug(f"Processing {num_procs} local process responses")
            return self.process_pool.call_all(
                method_name=method_name,
                params_list=params_list,
                deployed_as_of=deployed_as_of,
                request_id=request_id,
                distributed_env_vars_list=distributed_env_vars_list,
                debug_ports=debug_ports,
                serialization=serialization,
            )

        # We may not be calling the locally processes if the user specified workers and didn't include this node
        if call_local_procs:
            local_future = executor.submit(call_local_processes)
        else:
            local_future = None

        # Wait for both to complete with fast failure propagation
        try:
            # Use as_completed to get results as they arrive
            futures = [f for f in [worker_future, local_future] if f is not None]
            local_responses = []

            # If we only have local processes (no remote workers), handle that case
            if not futures:
                logger.error("No futures to wait for - this shouldn't happen")
                raise RuntimeError("No distributed work to execute")

            logger.debug(
                f"Waiting for {len(futures)} futures to complete (worker_future={worker_future is not None}, local_future={local_future is not None})"
            )

            # Process futures with periodic membership checks
            from concurrent.futures import FIRST_COMPLETED, wait

            pending_futures = set(futures)
            while pending_futures:
                # Check for membership changes even while waiting
                if change_event and change_event.is_set():
                    logger.debug("Membership change detected, checking...")
                    try:
                        self.check_for_membership_changes()
                    except Exception as e:
                        # Cancel all pending futures immediately
                        logger.error(f"Membership change detected, cancelling futures: {e}")
                        for f in pending_futures:
                            if not f.done():
                                f.cancel()
                        raise e
                    finally:
                        change_event.clear()  # Reset for next change

                # Wait for next future with a short timeout to allow membership checks
                done, pending_futures = wait(pending_futures, timeout=1.0, return_when=FIRST_COMPLETED)

                for future in done:
                    logger.debug(
                        f"Future completed: is_worker={worker_future and future == worker_future}, is_local={local_future and future == local_future}"
                    )
                    try:
                        if worker_future and future == worker_future:
                            logger.debug("Getting results from remote workers future")
                            results = future.result()
                            logger.debug(
                                f"Remote worker future returned: {type(results).__name__} with {len(results) if hasattr(results, '__len__') else 'N/A'} items"
                            )
                            # Process results - they're already JSON-decoded
                            logger.debug(f"Processing {len(results)} results from RemoteWorkerPool")
                            for i, result in enumerate(results):
                                logger.debug(
                                    f"Result {i}: type={type(result).__name__}, "
                                    f"length={len(result) if hasattr(result, '__len__') else 'N/A'}"
                                )
                                if isinstance(result, dict) and "error_type" in result:
                                    # Fast failure - return error immediately
                                    executor.shutdown(wait=False)
                                    return JSONResponse(status_code=500, content=result)
                                # Results from RemoteWorkerPool are already lists (aggregated from subtree in tree topology)
                                if isinstance(result, list):
                                    worker_responses.extend(result)
                                else:
                                    worker_responses.append(result)
                            logger.debug(f"Got {len(worker_responses)} total responses from remote workers")
                        else:  # local_future
                            logger.debug("Getting results from local processes future")
                            local_responses = future.result()
                            logger.debug(f"Got {len(local_responses)} responses from local processes")
                            # Check for errors in local responses
                            for response in local_responses:
                                if isinstance(response, JSONResponse):
                                    # Fast failure - return error immediately
                                    executor.shutdown(wait=False)
                                    return response
                    except Exception as e:
                        # Fast failure - propagate exception immediately
                        executor.shutdown(wait=False)
                        logger.error(f"Error in distributed execution: {e}")
                        raise

        finally:
            # Unsubscribe from membership changes
            if change_event:
                self.unsubscribe_from_membership_changes(change_event)

            logger.debug("Shutting down executor")
            executor.shutdown(wait=False)

        total = len(local_responses) + len(worker_responses)
        if tree_mode and total > 10:  # Only log for tree mode with significant results
            logger.debug(
                f"TREE RESULT AGGREGATION at {this_pod_ip}: {len(local_responses)} local + {len(worker_responses)} remote = {total} total"
            )
        else:
            logger.debug(
                f"Combining {len(local_responses)} local + {len(worker_responses)} remote = {total} total responses"
            )
        logger.debug(f"Distributed_subcall={distributed_subcall}, tree topology={tree_mode}")
        # Log sample of what we're returning for debugging
        if worker_responses:
            logger.debug(f"Sample worker response type: {type(worker_responses[0]).__name__}")
        responses = local_responses + worker_responses
        for response in responses:
            # If the response is a JSONResponse, we need to check if it contains an exception,
            # and "raise" it if so - essentially just returning it immediately rather than the full result list.
            if isinstance(response, JSONResponse):
                return response
            # This is primarily to handle exceptions while packaging an exception, which will cause the server to hang.
            if isinstance(response, Exception):
                raise response
        logger.debug(f"Returning {len(responses)} responses from execute_call")
        return responses


class JaxProcess(DistributedProcess):
    """JAX-specific distributed process."""

    @classmethod
    def get_distributed_env_vars(cls, worker_ips, node_rank, local_rank, num_local_procs, **settings):
        """Get JAX-specific distributed environment variables.

        JAX uses a coordinator address and process ID for distributed setup.
        """
        port = settings.get("port") or 1234  # JAX default coordinator port
        env_vars = super().get_distributed_env_vars(worker_ips, node_rank, local_rank, num_local_procs, **settings)

        # JAX distributed environment variables
        env_vars.update(
            {
                # Coordinator is the first worker
                "JAX_COORDINATOR_ADDRESS": f"{worker_ips[0]}:{port}",
                # Process ID is global rank
                "JAX_PROCESS_ID": str(node_rank * num_local_procs + local_rank),
                # Total number of processes
                "JAX_NUM_PROCESSES": str(len(worker_ips) * num_local_procs),
                # Local device IDs (for GPU/TPU)
                "JAX_LOCAL_DEVICE_IDS": str(local_rank),
            }
        )
        return env_vars

    @classmethod
    def get_auto_num_processes(cls):
        """Auto-detect based on available accelerators for JAX."""
        try:
            import jax

            # JAX can use TPUs, GPUs, or CPUs
            devices = jax.devices()
            return len(devices)
        except Exception:
            return 1

    # JAX doesn't have a global process group to destroy like PyTorch
    # Cleanup is mostly handled automatically
    # def proc_cleanup(self):


class TensorflowProcess(DistributedProcess):
    """TensorFlow-specific distributed process."""

    def proc_cleanup(self):
        """TensorFlow-specific cleanup."""
        try:
            import tensorflow as tf

            # Clear the default graph and reset the session
            tf.keras.backend.clear_session()
            logger.info("TensorFlow process cleanup completed.")
        except ImportError:
            logger.info("TensorFlow not available for cleanup")
        except Exception as e:
            logger.info(f"Failed during TensorFlow cleanup: {e}")
        # Call parent cleanup for debugging sessions
        super().proc_cleanup()

    @classmethod
    def get_distributed_env_vars(cls, worker_ips, node_rank, local_rank, num_local_procs, **settings):
        """Get TensorFlow-specific distributed environment variables.

        TensorFlow uses TF_CONFIG for distributed training configuration.
        """
        import json

        port = settings.get("port") or 2222  # TensorFlow default port
        env_vars = super().get_distributed_env_vars(worker_ips, node_rank, local_rank, num_local_procs, **settings)

        # Build TF_CONFIG for MultiWorkerMirroredStrategy
        worker_addresses = [f"{ip}:{port}" for ip in worker_ips]

        tf_config = {
            "cluster": {"worker": worker_addresses},
            "task": {"type": "worker", "index": node_rank},
        }

        env_vars.update(
            {
                "TF_CONFIG": json.dumps(tf_config),
                # Additional TF env vars for performance
                "TF_FORCE_GPU_ALLOW_GROWTH": "true",
                "TF_GPU_THREAD_MODE": "gpu_private",
            }
        )
        return env_vars

    @classmethod
    def get_auto_num_processes(cls):
        """Auto-detect based on available GPUs for TensorFlow."""
        try:
            import tensorflow as tf

            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                return len(gpus)
        except Exception:
            pass
        return 1


class MonarchProcess(DistributedProcess):
    """Monarch-specific distributed process for single-controller actor framework.

    Similar to Ray, Monarch uses a single controller (rank 0) that manages distributed
    actors across worker nodes. Each node runs a process_allocator service.
    """

    def __init__(self, local_rank, request_queue, response_queue, max_threads=4, **kwargs):
        super().__init__(local_rank, request_queue, response_queue, max_threads, **kwargs)
        self.allocator = None

        # Monarch imports will be done in run() on the main thread
        self.RemoteAllocator = None
        self.StaticRemoteAllocInitializer = None

    def _create_allocator_for_controller(self):
        """Create a RemoteAllocator for the controller (rank 0)."""

        try:
            # Try to import if not already available
            if self.RemoteAllocator is None or self.StaticRemoteAllocInitializer is None:
                try:
                    from monarch._src.actor.allocator import RemoteAllocator, StaticRemoteAllocInitializer

                    self.RemoteAllocator = RemoteAllocator
                    self.StaticRemoteAllocInitializer = StaticRemoteAllocInitializer
                    logger.debug("Monarch components imported")
                except ImportError as e:
                    logger.error(f"Failed to import Monarch: {e}")
                    logger.error("Make sure torchmonarch is installed: pip install torchmonarch")
                    import traceback

                    logger.error(traceback.format_exc())
                    return None
                except Exception as e:
                    logger.error(f"Unexpected error importing Monarch: {e}")
                    import traceback

                    logger.error(traceback.format_exc())
                    return None

            if self.RemoteAllocator is None or self.StaticRemoteAllocInitializer is None:
                logger.error("Monarch components not available. Cannot create allocator.")
                return None

            # Get worker addresses from POD_IPS
            pod_ips = os.environ.get("POD_IPS", "").split(",")
            if not pod_ips or pod_ips == [""]:
                logger.warning("No POD_IPS found, using localhost")
                pod_ips = ["127.0.0.1"]

            # Use tcp! format for channel addresses (Monarch's format)
            # Format: tcp!{ip}:{port} not tcp://{ip}:{port}
            worker_addresses = [f"tcp!{ip}:26600" for ip in pod_ips]
            logger.info(f"Creating Monarch allocator with {len(worker_addresses)} workers")
            logger.debug(f"Worker addresses type: {type(worker_addresses)}")
            logger.debug(f"First address: {worker_addresses[0] if worker_addresses else 'none'}")
            logger.debug(f"First address type: {type(worker_addresses[0]) if worker_addresses else 'none'}")

            # Simple check - don't add complex waiting logic

            # Create initializer with all workers using pre-imported classes
            # StaticRemoteAllocInitializer takes addresses as positional args
            logger.debug(f"About to create StaticRemoteAllocInitializer with args: {worker_addresses}")
            try:
                initializer = self.StaticRemoteAllocInitializer(*worker_addresses)
            except Exception as e:
                logger.error(f"Failed to create StaticRemoteAllocInitializer: {e}")
                import traceback

                logger.debug(f"Traceback: {traceback.format_exc()}")
                raise

            # Return configured allocator using pre-imported class
            # RemoteAllocator takes world_id and initializer
            # Use stable world_id based on service name
            # This allows coordinator failover and process restarts to work correctly
            service_name = os.environ.get("KT_SERVICE_NAME", "monarch-default")
            world_id = service_name
            try:
                allocator = self.RemoteAllocator(world_id=world_id, initializer=initializer)
                logger.info(f"Created allocator with world_id={world_id}")
                return allocator
            except Exception as e:
                logger.error(f"Failed to create RemoteAllocator: {e}")
                import traceback

                logger.debug(f"Traceback: {traceback.format_exc()}")
                raise

        except ImportError as e:
            logger.error(f"Could not import Monarch for allocator creation: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return None
        except Exception as e:
            logger.error(f"Failed to create Monarch allocator: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return None

    def run_user_function(self, callable_obj, method_name, params):
        """Run user function with Monarch-specific setup."""
        import asyncio
        import inspect

        # Get the rank from environment
        rank = int(os.environ.get("NODE_RANK", "0"))
        logger.debug(f"Running user function on rank {rank}")

        # Get the method to call
        if method_name and hasattr(callable_obj, method_name):
            user_method = getattr(callable_obj, method_name)
        else:
            user_method = callable_obj

        logger.info(f"User method: {user_method}")

        # Prepare arguments
        args = params.get("args", []) if params else []
        kwargs = params.get("kwargs", {}) if params else {}

        # Only create and inject allocator for controller (rank 0)
        # Workers will run the user function with allocator=None
        if rank == 0:
            logger.info("Rank 0 (controller) - will create allocator")
            # Controller (rank 0) - create allocator if needed
            if self.allocator is None:
                logger.debug("Creating allocator...")
                self.allocator = self._create_allocator_for_controller()
                if self.allocator is None:
                    logger.error("Failed to create allocator - returned None!")
                else:
                    logger.debug("Allocator created successfully")

            # Inject allocator if function accepts it
            try:
                sig = inspect.signature(user_method)
                if "allocator" in sig.parameters:
                    logger.debug("Injecting allocator into controller function")
                    kwargs["allocator"] = self.allocator
            except Exception as e:
                logger.warning(f"Could not inspect function signature: {e}")
        else:
            # Workers get None for allocator parameter if the function expects it
            try:
                sig = inspect.signature(user_method)
                if "allocator" in sig.parameters:
                    logger.info(f"Worker {rank}: Setting allocator=None")
                    kwargs["allocator"] = None
            except Exception:
                pass

        # Run the function (we're already on the main thread of this process)
        logger.info(f"Rank {rank}: Running user function with args={args}, kwargs keys={list(kwargs.keys())}")

        try:
            if asyncio.iscoroutinefunction(user_method):
                result = asyncio.run(user_method(*args, **kwargs))
            else:
                result = user_method(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = asyncio.run(result)

            # If the result is an exception dict (from the user's try/except),
            # convert it back to a proper exception and raise it
            if isinstance(result, dict) and "status" in result and result["status"] == "error":
                error_msg = result.get("error", "Unknown error")
                traceback_str = result.get("traceback", "")
                logger.error(f"User function returned error dict: {error_msg}")
                logger.error(f"Traceback from user function: {traceback_str}")
                # Raise a RuntimeError with the original error message
                raise RuntimeError(f"Monarch execution failed: {error_msg}\n{traceback_str}")

            return result
        except Exception as e:
            logger.error(f"Exception in run_user_function: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Re-raise the exception to be handled by the caller
            raise

    def run(self):
        """Override run to handle requests on main thread for Monarch."""
        logger.debug("MonarchProcess starting on main thread")

        # Import Monarch on the main thread of this subprocess
        # This is the right place since run() executes on the main thread
        if self.RemoteAllocator is None or self.StaticRemoteAllocInitializer is None:
            try:
                from monarch._src.actor.allocator import RemoteAllocator, StaticRemoteAllocInitializer

                self.RemoteAllocator = RemoteAllocator
                self.StaticRemoteAllocInitializer = StaticRemoteAllocInitializer
            except Exception as e:
                logger.error(f"Failed to import Monarch in run(): {e}")
                import traceback

                logger.error(traceback.format_exc())

        # Monarch requires main thread execution, so we don't use ThreadPoolExecutor
        try:
            while True:
                try:
                    # Block waiting for next request
                    request = self._request_queue.get(timeout=1)

                    # Special sentinel value to signal shutdown
                    if request == "SHUTDOWN":
                        break

                    # Handle request directly on main thread (not in thread pool)
                    self.handle_request(request)

                except queue.Empty:
                    continue
                except Exception as e:
                    if "Empty" not in str(e.__class__.__name__):
                        logger.error(f"Error getting request from queue: {e}")
                    continue

        except (KeyboardInterrupt, BdbQuit):
            logger.debug("MonarchProcess interrupted")
        finally:
            logger.debug("MonarchProcess shutting down")
            self.proc_cleanup()

    def handle_request(self, request):
        """Handle request using Monarch-specific logic."""
        try:
            # Use parent's handle_request but override the actual function execution
            request_unique_id = request["request_unique_id"]
            method_name = request["method_name"]
            params = request["params"]
            deployed_as_of = request["deployed_as_of"]
            request_id = request["request_id"]
            distributed_env_vars = request["distributed_env_vars"]

            # Set environment variables
            for key, value in distributed_env_vars.items():
                os.environ[key] = value

            # Set request context
            token = request_id_ctx_var.set(request_id)

            try:
                # Load callable
                callable_obj = load_callable(
                    deployed_as_of=deployed_as_of,
                    distributed_subprocess=True,
                    reload_cleanup_fn=self.proc_cleanup,
                )

                # Run with our simplified Monarch logic
                result = self.run_user_function(callable_obj, method_name, params)

                # Send response
                self._response_queue.put({"request_unique_id": request_unique_id, "result": result})

            except Exception as e:
                # Package and send error
                packaged_exception = package_exception(e)
                self._response_queue.put(
                    {
                        "request_unique_id": request_unique_id,
                        "result": packaged_exception,
                    }
                )
            finally:
                request_id_ctx_var.reset(token)

        except Exception as e:
            logger.error(f"Error in Monarch request handling: {e}")
            self._response_queue.put(
                {
                    "request_unique_id": request.get("request_unique_id", "unknown"),
                    "result": Exception(f"Fatal error: {e}"),
                }
            )

    def proc_cleanup(self):
        """Monarch-specific cleanup."""
        try:
            # Stop allocator service
            if self.allocator_proc:
                self.allocator_proc.terminate()
                try:
                    self.allocator_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.allocator_proc.kill()
                self.allocator_proc = None
                logger.info("Stopped process_allocator service")

            # Cleanup any Monarch resources
            # Monarch doesn't have a global shutdown like Ray
            logger.debug("Monarch process cleanup completed")

        except Exception as e:
            logger.error(f"Error during Monarch cleanup: {e}")

        # Call parent cleanup
        super().proc_cleanup()

    @classmethod
    def get_distributed_env_vars(cls, worker_ips, node_rank, local_rank, num_local_procs, **settings):
        """Get Monarch-specific environment variables."""
        env_vars = super().get_distributed_env_vars(worker_ips, node_rank, local_rank, num_local_procs, **settings)

        # Monarch uses these for discovery
        env_vars.update(
            {
                "HYPERACTOR_MESH_BOOTSTRAP_ADDR": "tcp://localhost:26600",
                "HYPERACTOR_MESH_INDEX": str(node_rank),
                # Keep POD_IPS for allocator creation
                "POD_IPS": ",".join(worker_ips),
            }
        )

        return env_vars

    @classmethod
    def get_auto_num_processes(cls):
        """Monarch uses one process per node (like Ray)."""
        return 1


# Similar to Ray, Monarch needs special handling as a single-controller framework
class MonarchDistributed(DistributedSupervisor):
    """Monarch distributed supervisor for single-controller actor framework."""

    def __init__(
        self,
        restart_procs=True,
        max_threads=4,
        quorum_timeout=300,
        quorum_workers=None,
        **kwargs,
    ):
        # Monarch doesn't use DNS monitoring like SPMD frameworks
        super().__init__(
            quorum_workers=quorum_workers,
            quorum_timeout=quorum_timeout,
            monitor_members=False,  # Disable DNS monitoring like Ray
        )
        self.restart_procs = restart_procs
        self.max_threads = max_threads
        self.process_pool = None
        self.remote_worker_pool = None

    def setup(self, deployed_as_of: Optional[str] = None):
        """Setup Monarch distributed environment."""
        # Set multiprocessing to spawn
        if multiprocessing.get_start_method() != "spawn":
            multiprocessing.set_start_method("spawn", force=True)

        # Start process_allocator service (like Ray starts its server)
        self._start_allocator_service()

        if self.restart_procs:
            logger.debug("restart_procs is True, restarting Monarch processes")
            self.cleanup()

        if self.process_pool is None:
            logger.debug("Setting up Monarch distributed environment")

            # Create process pool with MonarchProcess
            logger.info("Creating DistributedProcessPool with MonarchProcess class")
            self.process_pool = DistributedProcessPool(
                process_class=MonarchProcess,
                num_processes=1,  # One process per node for Monarch
                max_threads_per_proc=self.max_threads,
            )

            # Start the process
            logger.info("Starting MonarchProcess pool...")
            self.process_pool.start()
            logger.info(f"Started MonarchProcess pool: {self.process_pool}")

            # Create remote worker pool for coordination
            self.remote_worker_pool = RemoteWorkerPool(quorum_timeout=self.quorum_timeout)
            self.remote_worker_pool.start()

            logger.debug("Finished setting up Monarch distributed processes")

    def _start_allocator_service(self):
        """Start the process_allocator service if available."""
        try:
            # Check if process_allocator is already running
            import subprocess

            check_result = subprocess.run(["pgrep", "-f", "process_allocator"], capture_output=True)
            if check_result.returncode == 0:
                logger.info("process_allocator already running")
                return

            # Try to find process_allocator binary
            import shutil

            allocator_path = shutil.which("process_allocator")

            if not allocator_path:
                # Check common installation paths
                import sys

                possible_paths = [
                    "/opt/conda/bin/process_allocator",
                    os.path.join(sys.prefix, "bin", "process_allocator"),
                ]
                for path in possible_paths:
                    if os.path.exists(path) and os.access(path, os.X_OK):
                        allocator_path = path
                        break

            if not allocator_path:
                logger.warning(
                    "process_allocator binary not found. "
                    "Please ensure torchmonarch is properly installed or "
                    "start process_allocator manually in your Docker image."
                )
                return

            # Start process_allocator with the correct arguments
            # Based on monarch/python/monarch/tools/components/hyperactor.py
            allocator_cmd = [
                allocator_path,
                "--port=26600",
                "--program=monarch_bootstrap",
            ]

            logger.info(f"Starting process_allocator: {' '.join(allocator_cmd)}")

            # Start in background
            self.allocator_proc = subprocess.Popen(
                allocator_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )

            # Give it a moment to start
            import time

            time.sleep(2)

            # Check if it's still running
            if self.allocator_proc.poll() is None:
                logger.info(f"process_allocator started successfully (PID: {self.allocator_proc.pid})")
            else:
                stderr = self.allocator_proc.stderr.read().decode() if self.allocator_proc.stderr else ""
                logger.error(f"process_allocator failed to start: {stderr}")
                self.allocator_proc = None

        except Exception as e:
            logger.warning(f"Could not start process_allocator: {e}")
            # Continue anyway - user may have started it differently

    def cleanup(self):
        """Cleanup Monarch distributed environment."""
        logger.debug("Cleaning up Monarch distributed processes")

        # Stop DNS monitoring (though it's disabled for Monarch)
        super().cleanup()

        # Stop process_allocator if we started it
        if hasattr(self, "allocator_proc") and self.allocator_proc:
            try:
                self.allocator_proc.terminate()
                self.allocator_proc.wait(timeout=5)
                logger.info("Stopped process_allocator")
            except Exception as e:
                logger.debug(f"Error stopping process_allocator: {e}")

        if self.process_pool:
            self.process_pool.stop()
            self.process_pool = None

        if self.remote_worker_pool:
            self.remote_worker_pool.stop()
            self.remote_worker_pool = None

        logger.debug("Finished cleaning up Monarch distributed processes")

    @staticmethod
    def intercept_call():
        """Monarch intercepts calls like Ray."""
        return True

    def call_distributed(
        self,
        request,
        cls_or_fn_name: str,
        method_name: Optional[str] = None,
        params: Optional[Dict] = None,
        distributed_subcall: bool = False,
        debug_port: int = False,
        deployed_as_of: Optional[str] = None,
    ):
        """Monarch distributed call - executes on controller node (rank 0)."""
        logger.info("MonarchDistributed.call_distributed called")

        # Ensure setup has been called
        if self.process_pool is None:
            logger.info("Process pool not initialized, calling setup()")
            self.setup(deployed_as_of=deployed_as_of)

        request_id = request.headers.get("X-Request-ID", "-")
        serialization = request.headers.get("X-Serialization", "json")

        # If deployed_as_of is None, generate a consistent timestamp
        if deployed_as_of is None:
            from datetime import datetime, timezone

            deployed_as_of = datetime.now(timezone.utc).isoformat()

        # Start DNS monitoring for worker discovery
        self.start_dns_monitoring()

        # Check for any pending changes before we start
        self.check_for_membership_changes()

        # Get pod IPs with quorum handling
        pod_ips = self.pod_ips()

        # Handle case where no pods are found
        if not pod_ips:
            logger.error(
                f"No pods found for service {os.environ.get('KT_SERVICE')}. "
                "This may indicate the pods aren't ready yet. Consider increasing quorum_timeout in .distribute() call."
            )
            raise RuntimeError(
                "No pods found for Monarch distributed setup. " "Consider increasing quorum_timeout parameter."
            )

        logger.info(f"Found {len(pod_ips)} pod(s) for Monarch distributed setup: {pod_ips}")

        # Store critical environment variables
        self.distributed_env_vars = {}
        critical_env_vars = [
            "KT_SERVICE",
            "KT_SERVICE_NAME",
            "KT_FILE_PATH",
            "KT_MODULE_NAME",
            "KT_CLS_OR_FN_NAME",
        ]
        for env_var in critical_env_vars:
            if env_var in os.environ:
                self.distributed_env_vars[env_var] = os.environ[env_var]

        # Update distributed env vars with current cluster IPs
        self.distributed_env_vars["POD_IPS"] = ",".join(pod_ips)
        self.distributed_env_vars["WORLD_SIZE"] = str(len(pod_ips))
        self.distributed_env_vars["NODE_RANK"] = "0"  # Controller is always rank 0

        logger.debug("Sending call to Monarch subprocess (controller)")

        # Monarch uses only one process per node, call index 0
        result = self.process_pool.call(
            idx=0,
            method_name=method_name,
            params=params,
            deployed_as_of=deployed_as_of,
            request_id=request_id,
            distributed_env_vars=self.distributed_env_vars,
            debug_port=debug_port,
            serialization=serialization,
        )

        # Handle exceptions from subprocess
        if isinstance(result, JSONResponse):
            return result
        if isinstance(result, Exception):
            raise result

        return result


RAY_START_PROC = None


class RayDistributed(DistributedSupervisor):
    def __init__(
        self,
        restart_procs=True,
        max_threads=4,
        quorum_timeout=300,
        quorum_workers=None,
        monitor_members=False,
    ):
        """Ray distributed supervisor - only runs on head node (single controller).

        Args:
            restart_procs: Whether to restart processes on each call
            max_threads: Maximum threads per process
            quorum_timeout: Timeout in seconds for Ray cluster nodes to become ready (default 300s/5min)
        """
        # Ray manages its own membership, so we don't monitor DNS changes
        super().__init__(
            quorum_timeout=quorum_timeout,
            quorum_workers=quorum_workers,
            monitor_members=monitor_members,
        )
        self.restart_procs = restart_procs
        self.distributed_env_vars = None
        self.process_pool = None  # Using pool even for single process for consistency
        self.remote_worker_pool = None  # Pool for async HTTP calls to remote workers
        self.max_threads = max_threads
        self.quorum_timeout = quorum_timeout

    def setup(self, deployed_as_of: Optional[str] = None):
        # Set multiprocessing to spawn if not already
        if multiprocessing.get_start_method() != "spawn":
            multiprocessing.set_start_method("spawn", force=True)

        # Start the Ray server here, if we allow KubeRay to start it in the pod template
        # it's hard to wait for it start properly and we lose the ability to restart if needed.
        global RAY_START_PROC

        # Check if Ray is actually running, not just if our global variable is None
        # (the global variable gets reset when HTTP server restarts)
        ray_running = self._is_ray_running()

        if not ray_running:
            patch_sys_path()

            kuberay_start_cmd = os.environ.get("KUBERAY_GEN_RAY_START_CMD")
            if kuberay_start_cmd:
                full_cmd = f"ulimit -n 65536; {kuberay_start_cmd}"
                logger.info(f"Starting Ray server with command: {full_cmd}")

                try:
                    # Start Ray as a non-blocking subprocess
                    RAY_START_PROC = subprocess.Popen(
                        full_cmd,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        universal_newlines=True,
                        bufsize=1,
                        env=os.environ.copy(),
                    )

                    # Start a thread to stream Ray logs
                    def stream_ray_logs():
                        try:
                            for line in RAY_START_PROC.stdout:
                                logger.info(f"[Ray] {line.strip()}")
                        except Exception as e:
                            logger.error(f"Error streaming Ray logs: {e}")

                    import threading

                    log_thread = threading.Thread(target=stream_ray_logs, daemon=True)
                    log_thread.start()

                    logger.info(f"Ray server started with PID: {RAY_START_PROC.pid}")

                    # Give Ray a moment to start
                    time.sleep(2)

                except Exception as e:
                    logger.error(f"Failed to start Ray server: {e}")
                    RAY_START_PROC = None
                    raise
            else:
                logger.warning("KUBERAY_GEN_RAY_START_CMD environment variable not found")

        logger.debug("Ray distributed supervisor setup completed (pod discovery will be done lazily)")

        # Only the head node runs the subprocess
        this_pod_ip = os.environ["POD_IP"]
        if not os.environ["POD_NAME"].endswith("-head"):
            logger.info(f"Ray worker node {this_pod_ip}, skipping subprocess setup")
            return

        logger.info(f"Ray head node {this_pod_ip}, setting up subprocess")

        # Set Ray environment variables
        self.distributed_env_vars = {"RAY_HEAD_NODE_IP": this_pod_ip}

        # Include critical environment variables so Ray workers can find and load the callable
        critical_env_vars = [
            "PYTHONPATH",
            "KT_FILE_PATH",
            "KT_MODULE_NAME",
            "KT_CLS_OR_FN_NAME",
        ]
        for env_var in critical_env_vars:
            if env_var in os.environ:
                self.distributed_env_vars[env_var] = os.environ[env_var]

        # Cleanup will remove the process pool if found, so we need to check if it was previously initialized
        previously_initialized = self.remote_worker_pool is not None

        if self.restart_procs:
            logger.debug("restart_procs is True, restarting Ray distributed process")
            self.cleanup()

            if previously_initialized:
                pod_ips = self.pod_ips()
                this_pod_ip = os.environ["POD_IP"]

                # Send reload requests to other pods if needed
                self._reload_image_on_other_pods(pod_ips, this_pod_ip, deployed_as_of)

        if self.process_pool is None:
            logger.debug("Setting up Ray distributed process")
            self.process_pool = DistributedProcessPool(
                process_class=RayProcess,
                num_processes=1,  # Ray only needs one process
                max_threads_per_proc=self.max_threads,
            )
            self.process_pool.start()

            # # Start remote worker pool for async HTTP calls if needed
            # Use a reasonable default max_workers since we don't know cluster size yet
            self.remote_worker_pool = RemoteWorkerPool(quorum_timeout=self.quorum_timeout)
            self.remote_worker_pool.start(max_workers=100)  # Default size

            logger.debug("Finished setting up Ray distributed process and remote worker pool")

    def cleanup(self):
        """Clean up Ray distributed process."""
        logger.debug("Cleaning up Ray distributed process")

        # Stop DNS monitoring first
        super().cleanup()

        if self.process_pool:
            self.process_pool.stop()
            self.process_pool = None

        if self.remote_worker_pool:
            self.remote_worker_pool.stop()
            self.remote_worker_pool = None

        logger.debug("Finished cleaning up Ray distributed process")

    @staticmethod
    def intercept_call():
        return True

    def call_distributed(
        self,
        request,
        cls_or_fn_name: str,
        method_name: Optional[str] = None,
        params: Optional[Dict] = None,
        distributed_subcall: bool = False,
        debug_port: int = False,
        deployed_as_of: Optional[str] = None,
    ):
        """Ray distributed call - only executes on head node."""
        request_id = request.headers.get("X-Request-ID", "-")
        serialization = request.headers.get("X-Serialization", "json")

        # If deployed_as_of is None, generate a consistent timestamp
        # to use across all workers to prevent reload inconsistencies
        if deployed_as_of is None:
            from datetime import datetime, timezone

            deployed_as_of = datetime.now(timezone.utc).isoformat()

        if not os.environ["POD_NAME"].endswith("-head"):
            # This should never happen, because the service only points to the head node, Raise an error if it does.
            raise RuntimeError(
                f"Ray distributed call attempted on non-head node {os.environ['POD_NAME']}. "
                "This should only be called on the head node."
            )

        # Start DNS monitoring for the head node
        self.start_dns_monitoring()

        # Check for any pending changes before we start
        self.check_for_membership_changes()

        # The pod_ips() method now handles waiting for quorum
        pod_ips = self.pod_ips()

        # Handle case where no pods are found
        if not pod_ips:
            logger.error(
                f"No pods found for service {os.environ.get('KT_SERVICE')}. "
                "This may indicate the pods aren't ready yet. Consider increasing quorum_timeout in .distribute() call."
            )
            raise RuntimeError(
                "No pods found for Ray distributed setup. " "Consider increasing quorum_timeout parameter."
            )

        logger.info(f"Found {len(pod_ips)} pod(s) for distributed setup: {pod_ips}")

        # Update distributed env vars with current cluster IPs
        self.distributed_env_vars["POD_IPS"] = ",".join(pod_ips)

        logger.debug("Sending call to Ray subprocess")
        # Ray uses only one process, so always call index 0
        result = self.process_pool.call(
            idx=0,
            method_name=method_name,
            params=params,
            deployed_as_of=deployed_as_of,
            request_id=request_id,
            distributed_env_vars=self.distributed_env_vars,
            debug_port=debug_port,
            serialization=serialization,
        )

        # Handle exceptions from subprocess
        if isinstance(result, JSONResponse):
            return result
        if isinstance(result, Exception):
            raise result

        return result

    def _reload_image_on_other_pods(self, pod_ips, this_pod_ip, deployed_as_of):
        """Send /_reload_image requests to all other pods in parallel, with retries for pods that aren't ready."""
        other_pod_ips = [ip for ip in pod_ips if ip != this_pod_ip]

        if not other_pod_ips:
            logger.debug("No other pods to reload")
            return

        logger.info(f"Sending reload requests to {len(other_pod_ips)} other pods: {other_pod_ips}")

        server_port = os.environ.get("KT_SERVER_PORT", "32300")
        total_timeout = self.quorum_timeout  # Use configurable quorum timeout
        retry_interval = 2  # Wait 2 seconds between retry attempts
        start_time = time.time()

        successful_pods = set()
        remaining_pods = set(other_pod_ips)

        while remaining_pods and (time.time() - start_time) < total_timeout:
            logger.debug(f"Attempting to reload {len(remaining_pods)} remaining pods: {list(remaining_pods)}")

            def reload_pod(pod_ip):
                """Send reload request to a single pod."""
                try:
                    # Use a proper HTTP client session to avoid Content-Length issues
                    with httpx.Client(timeout=None) as client:
                        url = f"http://{pod_ip}:{server_port}/_reload_image"
                        # First try a quick health check to see if pod is ready
                        health_url = f"http://{pod_ip}:{server_port}/health"
                        health_response = client.get(health_url, timeout=5)

                        if health_response.status_code != 200:
                            logger.debug(f"Pod {pod_ip} health check failed, will retry later")
                            return False

                        # Pod is healthy, send reload request (no timeout, installs can be long-running)
                        response = client.post(url, headers={"X-Deployed-As-Of": deployed_as_of})
                        if response.status_code == 200:
                            logger.debug(f"Successfully reloaded image on pod {pod_ip}")
                            return True
                        else:
                            logger.warning(f"Pod {pod_ip} reload returned status {response.status_code}")
                            return False

                except Exception as e:
                    logger.debug(f"Failed to reload image on pod {pod_ip}: {e}")
                    raise

            # Try to reload all remaining pods in parallel
            current_attempt_pods = list(remaining_pods)

            with ThreadPoolExecutor(max_workers=min(len(current_attempt_pods), 10)) as executor:
                # Submit reload tasks for remaining pods
                future_to_pod = {executor.submit(reload_pod, pod_ip): pod_ip for pod_ip in current_attempt_pods}

                # Process completed futures
                for future in as_completed(future_to_pod, timeout=None):
                    pod_ip = future_to_pod[future]
                    try:
                        success = future.result()
                        if success:
                            successful_pods.add(pod_ip)
                            remaining_pods.discard(pod_ip)
                    except Exception as e:
                        logger.debug(f"Reload task for pod {pod_ip} failed: {e}")

            if remaining_pods:
                elapsed = time.time() - start_time
                remaining_time = total_timeout - elapsed
                if remaining_time > retry_interval:
                    logger.info(f"Waiting {retry_interval}s before retrying {len(remaining_pods)} pods...")
                    time.sleep(retry_interval)
                else:
                    logger.warning("Timeout approaching, stopping retry attempts")
                    break

        # Log final results
        if successful_pods:
            logger.info(f"Successfully reloaded {len(successful_pods)} pod images: {list(successful_pods)}")

        if remaining_pods:
            logger.warning(f"Failed to reload {len(remaining_pods)} pod images after timeout: {list(remaining_pods)}")

    def _is_ray_running(self):
        """Check if Ray is actually running by trying to connect to the Ray GCS port."""
        try:
            import socket

            # Ray GCS runs on port 6379 by default
            ray_port = 6379
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)  # 1 second timeout
            result = sock.connect_ex(("127.0.0.1", ray_port))
            sock.close()

            if result == 0:
                logger.debug("Ray GCS port 6379 is accessible, Ray appears to be running")
                return True
            else:
                logger.debug("Ray GCS port 6379 is not accessible, Ray is not running")
                return False

        except Exception as e:
            logger.debug(f"Error checking if Ray is running: {e}")
            return False


def distributed_supervisor_factory(distribution_type, *args, **kwargs):
    """
    Factory function to create a distributed supervisor based on the specified type.

    Args:
        distribution_type (str): The type of distributed supervisor to create.
                                Options include 'ray', 'monarch', 'pytorch', 'jax', 'tensorflow', or None for generic SPMD.
        *args: Positional arguments to pass to the supervisor constructor.
        **kwargs: Keyword arguments to pass to the supervisor constructor.
                 Common kwargs include:
                 - quorum_timeout: Timeout in seconds for workers to become ready (default 30 for SPMD, 300 for Ray/Monarch)

    Returns:
        DistributedSupervisor: An instance of the specified distributed supervisor.
    """
    if distribution_type == "ray":
        # Ray uses its own supervisor, not SPMD
        return RayDistributed(*args, **kwargs)
    elif distribution_type == "monarch":
        # Monarch is similar to Ray - single controller framework
        return MonarchDistributed(*args, **kwargs)

    # All other types use SPMDDistributedSupervisor with different process classes
    if distribution_type == "pytorch":
        return SPMDDistributedSupervisor(process_class=PyTorchProcess, *args, **kwargs)
    elif distribution_type == "jax":
        return SPMDDistributedSupervisor(process_class=JaxProcess, *args, **kwargs)
    elif distribution_type == "tensorflow" or distribution_type == "tf":
        return SPMDDistributedSupervisor(process_class=TensorflowProcess, *args, **kwargs)
    elif distribution_type is None or distribution_type == "spmd":
        # Default to base DistributedProcess - no framework-specific dependencies
        return SPMDDistributedSupervisor(process_class=DistributedProcess, *args, **kwargs)
    else:
        raise ValueError(f"Unsupported distributed type: {distribution_type}")
