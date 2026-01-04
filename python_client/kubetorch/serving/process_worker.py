import multiprocessing
import os
from bdb import BdbQuit
from concurrent.futures import ThreadPoolExecutor

from kubetorch.serving.http_server import execute_callable, load_callable, logger, package_exception
from kubetorch.serving.log_capture import create_subprocess_log_capture
from kubetorch.serving.utils import clear_debugging_sessions, request_id_ctx_var


class ProcessWorker(multiprocessing.Process):
    """Base class for distributed processes that run callables in subprocesses."""

    def __init__(self, local_rank, request_queue, response_queue, max_threads=4, log_queue=None, **kwargs):
        super().__init__()
        # We don't need the cache miss / reload here because these processes are destroyed and recreated
        # with each .to call.
        os.environ["LOCAL_RANK"] = str(local_rank)
        self._request_queue = request_queue
        self._response_queue = response_queue
        self._max_threads = max_threads
        self._executor = None
        self._log_queue = log_queue  # Queue for sending logs back to main process
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
            debug_mode = request["debug_mode"]
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
                    reload_cleanup_fn=self.framework_cleanup,
                )

                result = execute_callable(
                    callable_obj=callable_obj,
                    cls_or_fn_name=os.environ["KT_CLS_OR_FN_NAME"],
                    method_name=method_name,
                    params=params,
                    serialization=serialization,
                    debug_port=debug_port,
                    debug_mode=debug_mode,
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
        # Set up subprocess log capture to push logs to main process via queue
        # Uses LogCapture in queue mode - same capture logic, different emit target
        log_capture = create_subprocess_log_capture(self._log_queue)

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
            if log_capture:
                log_capture.stop()
            logger.info("Exiting gracefully.")
