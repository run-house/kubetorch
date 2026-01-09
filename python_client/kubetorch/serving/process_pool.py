import multiprocessing
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

from kubetorch.serving.http_server import logger


class ProcessPool:
    """Unified pool managing distributed processes with single router thread."""

    def __init__(self, process_class, num_processes, max_threads_per_proc=10, log_queue=None, **process_kwargs):
        self.process_class = process_class
        self.num_processes = num_processes
        self.max_threads_per_proc = max_threads_per_proc
        self.log_queue = log_queue  # Queue for subprocess log collection (from LogCapture)
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
            log_queue=self.log_queue,  # For subprocess log capture
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
        request_id,
        distributed_env_vars,
        debug_port,
        debug_mode,
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
                    "request_id": request_id,
                    "distributed_env_vars": distributed_env_vars,
                    "debug_port": debug_port,
                    "debug_mode": debug_mode,
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
        request_id,
        distributed_env_vars_list,
        debug_ports,
        debug_mode,
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
                    request_id=request_id,
                    distributed_env_vars=distributed_env_vars_list[idx],
                    debug_port=debug_ports[idx] if debug_ports else None,
                    debug_mode=debug_mode if debug_mode else None,
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
