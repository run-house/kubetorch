import asyncio
import base64
import inspect
import multiprocessing
import os
from bdb import BdbQuit
from concurrent.futures import ThreadPoolExecutor
from typing import List

from kubetorch.globals import ProfilerConfig

from kubetorch.serving.http_server import execute_callable_async, load_callable, logger, package_exception
from kubetorch.serving.log_capture import create_subprocess_log_capture
from kubetorch.serving.profiling import run_with_pyspy_profiler, run_with_torch_profiler
from kubetorch.serving.utils import clear_debugging_sessions, request_id_ctx_var

# Match FastAPI/Starlette default thread pool size for sync operations
DEFAULT_THREADPOOL_SIZE = 40


class ProcessWorker(multiprocessing.Process):
    """Base class for distributed processes that run callables in subprocesses.

    Uses an asyncio event loop to handle requests, matching FastAPI's concurrency model:
    - Async callables run directly on the event loop (true async concurrency)
    - Sync callables run in a thread pool via run_in_executor()

    This allows async callables to benefit from cooperative multitasking while
    sync callables don't block the event loop.
    """

    def __init__(
        self, local_rank, request_queue, response_queue, max_threads=DEFAULT_THREADPOOL_SIZE, log_queue=None, **kwargs
    ):
        super().__init__()
        # We don't need the cache miss / reload here because these processes are destroyed and recreated
        # with each .to call.
        os.environ["LOCAL_RANK"] = str(local_rank)
        self._request_queue = request_queue
        self._response_queue = response_queue
        self._max_threads = max_threads
        self._executor = None
        self._loop = None
        self._log_queue = log_queue  # Queue for sending logs back to main process
        self._log_capture = None
        # Store any additional framework-specific settings
        self._settings = kwargs

    def framework_cleanup(self):
        """Override this method to provide framework-specific cleanup for reloads.

        This is called during hot reloads to clean up framework state (e.g., Ray, PyTorch process groups)
        without shutting down the executor, which is still needed for handling requests.
        """
        pass

    def proc_cleanup(self):
        """Full process cleanup called on shutdown. Cleans up framework state, debugging sessions, and executor."""
        # Call framework cleanup first
        self.framework_cleanup()

        logger.info("Cleaning up debugging sessions...")
        clear_debugging_sessions()
        logger.info("Debugging sessions cleaned up.")

        # Cleanup thread pool (only on full shutdown, not during reload)
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None

    @classmethod
    def get_distributed_env_vars(
        cls,
        worker_ips: List[str],
        node_rank: int,
        local_rank: int,
        num_local_procs: int,
        **settings,
    ):
        """Get framework-specific distributed environment variables.

        Args:
            worker_ips (List[str]): List of all worker IPs.
            node_rank (int): Rank of this node (0-indexed).
            local_rank (int): Local rank on this node (0-indexed).
            num_local_procs (int): Number of processes on this node.
            **settings: Additional framework-specific settings (e.g., port).

        Returns:
            Dict of environment variables to set.
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

    async def execute_with_profiling(self, callable_obj, method_name, params, serialization, profiler):
        """Execute callable with profiling using minimal overhead path.
        This method is called directly from handle_request_async when profiling is enabled.
        """

        # Prepare raw data for subprocess (no unpickling/parsing in parent!)
        if serialization == "pickle":
            # Extract raw pickled bytes directly - DO NOT unpickle
            if isinstance(params, dict) and "data" in params:
                stdin_data = base64.b64decode(params["data"].encode("utf-8"))
            elif isinstance(params, str):
                stdin_data = base64.b64decode(params.encode("utf-8"))
            else:
                raise ValueError(f"Unexpected params format for pickle serialization: {type(params)}")
        else:
            # JSON serialization: pass as JSON bytes (no pickling)
            import json

            stdin_data = json.dumps(
                {
                    "args": params.get("args", []),
                    "kwargs": params.get("kwargs", {}),
                }
            ).encode("utf-8")

        # Get user method (only needed for is_async check)
        user_method = getattr(callable_obj, method_name) if method_name else callable_obj
        is_async = inspect.iscoroutinefunction(user_method)

        # Setup profiler config
        profiler_config = ProfilerConfig(**profiler)
        profiler_type = profiler_config.profiler_type

        # Execute with minimal profiling overhead
        if profiler_type == "pytorch":
            result = await run_with_torch_profiler(
                stdin_data, serialization, method_name, is_async, profiler_config, self._loop
            )
        elif profiler_type == "pyspy":
            result = await run_with_pyspy_profiler(
                stdin_data, serialization, method_name, is_async, profiler_config, self._loop
            )
        else:
            raise ValueError(f"Unsupported profiler type: {profiler_type}")

        return result

    async def handle_request_async(self, request):
        """Handle a single request asynchronously.

        For async callables: awaits directly on the event loop (true async concurrency)
        For sync callables: runs in thread pool via run_in_executor()

        For profiling requests: uses minimal execution path to avoid kubetorch overhead
        """
        try:
            request_unique_id = request["request_unique_id"]
            method_name = request["method_name"]
            params = request["params"]
            request_id = request["request_id"]
            distributed_env_vars = request["distributed_env_vars"]
            serialization = request["serialization"]

            # Extract profiler from request or from params
            profiler = request.get("profiler")
            if not profiler and isinstance(params, dict):
                profiler = params.pop(
                    "profiler", None
                )  # Remove profiler from params to avoid double profiling in execute_with_profiling
                params = params.copy()

            # Set the request ID in the context
            token = request_id_ctx_var.set(request_id)

            # Set the environment variables (process-wide)
            # For distributed PyTorch calls, these should already be set at process level
            for key, value in distributed_env_vars.items():
                os.environ[key] = value

            try:
                # Load callable - with push-based reloads, the subprocess is recreated on reload
                # so we don't need to check timestamps
                callable_obj = load_callable(
                    distributed_subprocess=True,
                    reload_cleanup_fn=self.framework_cleanup,
                )

                # Re-add logging handler if user code's logging config removed it
                # (e.g., structlog's dictConfig() can remove all root handlers)
                if self._log_capture:
                    self._log_capture.ensure_handler()

                # run execute_with_profiling to avoid overhead in profiling. Run the user method in a separate process,
                # to avoid profiling the whole calling stack.
                if profiler:
                    result = await self.execute_with_profiling(
                        callable_obj=callable_obj,
                        method_name=method_name,
                        params=params,
                        serialization=serialization,
                        profiler=profiler,
                    )
                else:
                    # Normal path - execute the callable
                    result = await execute_callable_async(
                        callable_obj=callable_obj,
                        cls_or_fn_name=os.environ["KT_CLS_OR_FN_NAME"],
                        method_name=method_name,
                        params=params,
                        serialization=serialization,
                        executor=self._executor,
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
                    "result": Exception(f"Fatal error in async handler: {e}"),
                }
            )

    async def _poll_queue(self):
        """Poll the request queue and dispatch requests to the event loop.

        Uses a short sleep to yield control to the event loop between polls,
        allowing concurrent async handlers to make progress.
        """
        while True:
            try:
                # Non-blocking check with short timeout
                try:
                    request = self._request_queue.get(timeout=0.01)
                except Exception as e:
                    if "Empty" in str(e.__class__.__name__):
                        # No request available, yield to event loop
                        await asyncio.sleep(0.001)
                        continue
                    raise

                # Special sentinel value to signal shutdown
                if request == "SHUTDOWN":
                    break

                # Schedule request handler as a task on the event loop
                # This allows multiple requests to run concurrently
                asyncio.create_task(self.handle_request_async(request))

            except Exception as e:
                logger.error(f"Error getting request from queue: {e}")
                await asyncio.sleep(0.01)

    def run(self):
        """Main process loop with asyncio event loop for concurrent request handling.

        Matches FastAPI's concurrency model:
        - Async callables run directly on the event loop (true async concurrency)
        - Sync callables run in a thread pool via run_in_executor()
        """
        # Set up subprocess log capture to push logs to main process via queue
        # Uses LogCapture in queue mode - same capture logic, different emit target
        self._log_capture = create_subprocess_log_capture(self._log_queue)

        # Create thread pool for sync callables (matches FastAPI's default of 40 threads)
        self._executor = ThreadPoolExecutor(max_workers=self._max_threads)

        # Create and run the event loop
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._poll_queue())

        except (KeyboardInterrupt, BdbQuit):
            logger.info("Process interrupted, shutting down...")

        finally:
            # Cancel any pending tasks
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()

            # Wait for tasks to be cancelled
            if pending:
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

            self._loop.close()

            # Cleanup
            logger.info("Received shutdown signal, cleaning up distributed environment...")
            self.proc_cleanup()
            if self._log_capture:
                self._log_capture.stop()
            logger.info("Exiting gracefully.")
