import os
import queue
import subprocess
import threading
from bdb import BdbQuit
from typing import Dict, Optional

from starlette.responses import JSONResponse

from kubetorch.serving.distributed_supervisor import DistributedSupervisor
from kubetorch.serving.http_server import load_callable, logger, package_exception, request_id_ctx_var
from kubetorch.serving.process_worker import ProcessWorker

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


class MonarchProcess(ProcessWorker):
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
            request_id = request["request_id"]
            distributed_env_vars = request["distributed_env_vars"]

            # Set environment variables
            for key, value in distributed_env_vars.items():
                os.environ[key] = value

            # Set request context
            token = request_id_ctx_var.set(request_id)

            try:
                # Load callable - with push-based reloads, the subprocess is recreated on reload
                callable_obj = load_callable(
                    distributed_subprocess=True,
                    reload_cleanup_fn=self.framework_cleanup,
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

    def framework_cleanup(self):
        """Monarch-specific cleanup for reloads."""
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
    """Monarch distributed supervisor for single-controller actor framework.

    Monarch manages its own actor allocation, so DNS monitoring is disabled.
    This supervisor handles:
    - Starting process_allocator service
    - Setting up subprocess for user code execution
    - Coordinating with Monarch's actor framework
    """

    def __init__(
        self,
        max_threads=4,
        **kwargs,
    ):
        """Initialize Monarch supervisor.

        Args:
            max_threads (int, optional): Maximum threads per process. (Default: 4)
            **kwargs: Arguments passed to DistributedSupervisor.
        """
        # Monarch manages its own membership, disable DNS monitoring
        # Force num_processes=1 since Monarch uses one process per node
        super().__init__(
            process_class=MonarchProcess,
            num_processes=1,
            max_threads_per_proc=max_threads,
            monitor_members=False,  # Monarch manages its own membership
            **kwargs,
        )
        self.allocator_proc = None

    def setup(self):
        """Setup Monarch distributed environment."""
        # Start process_allocator service (like Ray starts its server)
        self._start_allocator_service()

        # Call parent setup to create ProcessPool
        # Note: Monarch doesn't use RemoteWorkerPool - it handles distributed
        # coordination via its own process_allocator service
        super().setup()
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
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                universal_newlines=True,
                bufsize=1,
            )

            # Start a thread to stream allocator logs (similar to Ray)
            def stream_allocator_logs():
                try:
                    for line in self.allocator_proc.stdout:
                        logger.info(f"[Allocator] {line.strip()}")
                except Exception as e:
                    logger.debug(f"Allocator log stream ended: {e}")

            allocator_log_thread = threading.Thread(target=stream_allocator_logs, daemon=True)
            allocator_log_thread.start()

            # Give it a moment to start
            import time

            time.sleep(2)

            # Check if it's still running
            if self.allocator_proc.poll() is None:
                logger.info(f"process_allocator started successfully (PID: {self.allocator_proc.pid})")
            else:
                logger.error("process_allocator failed to start")
                self.allocator_proc = None

        except Exception as e:
            logger.warning(f"Could not start process_allocator: {e}")
            # Continue anyway - user may have started it differently

    def cleanup(self):
        """Cleanup Monarch distributed environment."""
        logger.debug("Cleaning up Monarch distributed processes")

        # Stop process_allocator if we started it
        if self.allocator_proc:
            try:
                self.allocator_proc.terminate()
                self.allocator_proc.wait(timeout=5)
                logger.info("Stopped process_allocator")
            except Exception as e:
                logger.debug(f"Error stopping process_allocator: {e}")
            self.allocator_proc = None

        # Call parent cleanup for ProcessPool
        super().cleanup()
        logger.debug("Finished cleaning up Monarch distributed processes")

    def call(
        self,
        request,
        cls_or_fn_name: str,
        method_name: Optional[str] = None,
        params: Optional[Dict] = None,
        distributed_subcall: bool = False,
    ):
        """Monarch distributed call - executes on controller node (rank 0)."""
        logger.info("MonarchDistributed.call called")

        # Ensure setup has been called
        if self.process_pool is None:
            logger.info("Process pool not initialized, calling setup()")
            self.setup()

        request_id = request.headers.get("X-Request-ID", "-")
        serialization = request.headers.get("X-Serialization", "json")

        debug_mode, debug_port = None, None
        debugger: dict = params.get("debugger", None) if params else None
        if debugger:
            debug_mode = debugger.get("mode")
            debug_port = debugger.get("port")

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
            request_id=request_id,
            distributed_env_vars=self.distributed_env_vars,
            debug_port=debug_port,
            debug_mode=debug_mode,
            serialization=serialization,
        )

        # Handle exceptions from subprocess
        if isinstance(result, JSONResponse):
            return result
        if isinstance(result, Exception):
            raise result

        return result
