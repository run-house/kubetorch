"""Execution supervisor base class with ProcessPool-based subprocess execution.

This module provides the ExecutionSupervisor class which handles local execution
of callable functions/classes in isolated subprocesses. This provides:
- Clean module isolation (user code runs in subprocess)
- Simple reload semantics (terminate and recreate subprocess)
- Consistent execution pattern across all modes

For distributed execution with remote workers, see DistributedSupervisor.
"""

import multiprocessing
from typing import Dict, Optional

from starlette.responses import JSONResponse

from kubetorch.serving.http_server import logger
from kubetorch.serving.log_capture import get_subprocess_queue
from kubetorch.serving.process_pool import ProcessPool
from kubetorch.serving.process_worker import ProcessWorker


class ExecutionSupervisor:
    """Base class for execution supervisors using subprocess isolation.

    This class provides local execution via ProcessPool with one or more subprocesses.
    It handles:
    - Creating and managing a pool of worker subprocesses
    - Routing calls to subprocesses
    - Clean restart semantics for redeployment

    Subclass DistributedSupervisor for distributed execution with remote workers.
    """

    def __init__(
        self,
        process_class: ProcessWorker = None,
        num_processes: int = 1,
        max_threads_per_proc: int = 10,
        restart_procs: bool = True,
        **process_kwargs,
    ):
        """Initialize execution supervisor.

        Args:
            process_class (ProcessWorker, optional): The ProcessWorker subclass to use for subprocesses.
                If None, ProcessWorker will be used. (Default: None)
            num_processes (int, optional): Number of local subprocesses to run. Can also be "auto" to use
                process_class.get_auto_num_processes(). (Default: 1)
            max_threads_per_proc (int, optional): Maximum threads per subprocess. (Default: 10)
            restart_procs (bool, optional): Whether to restart processes on setup. (Default: True)
            **process_kwargs: Additional kwargs passed to process class constructor.
        """
        self.process_class = process_class or ProcessWorker
        self.num_processes = num_processes
        self.max_threads_per_proc = max_threads_per_proc
        self.restart_procs = restart_procs
        self.process_kwargs = process_kwargs

        self.process_pool: Optional[ProcessPool] = None
        self.config_hash: Optional[int] = None  # Used by factory to detect config changes

    def setup(self):
        """Set up execution environment with process pool."""
        # Set multiprocessing to spawn if not already
        if multiprocessing.get_start_method() != "spawn":
            multiprocessing.set_start_method("spawn", force=True)

        # Determine actual number of processes
        if self.num_processes == "auto":
            num_proc = self.process_class.get_auto_num_processes()
        else:
            num_proc = self.num_processes

        # Restart processes if requested
        if self.restart_procs and self.process_pool:
            logger.debug("restart_procs is True, restarting processes")
            self.cleanup()

        # Create new pool if needed or if size changed
        if self.process_pool is None or len(self.process_pool) != num_proc:
            if self.process_pool:
                logger.debug(f"Number of processes changed from {len(self.process_pool)} to {num_proc}")
                self.cleanup()

            logger.debug(f"Setting up process pool with {num_proc} processes")
            self.process_pool = ProcessPool(
                process_class=self.process_class,
                num_processes=num_proc,
                max_threads_per_proc=self.max_threads_per_proc,
                log_queue=get_subprocess_queue(),
                **self.process_kwargs,
            )
            self.process_pool.start()
            logger.debug("Process pool started successfully")

    def cleanup(self):
        """Clean up process pool."""
        if self.process_pool:
            logger.debug("Cleaning up process pool")
            self.process_pool.stop()
            self.process_pool = None
            logger.debug("Process pool stopped")

    def call(
        self,
        request,
        cls_or_fn_name: str,
        method_name: Optional[str] = None,
        params: Optional[Dict] = None,
        distributed_subcall: bool = False,
    ):
        """Execute a call through the subprocess pool.

        For local execution (non-distributed), this routes the call to the first
        subprocess. For distributed execution, subclasses override this method
        to coordinate across multiple processes and remote workers.

        Args:
            request: The HTTP request object.
            cls_or_fn_name: Name of the callable.
            method_name: Method name to call (for class instances).
            params: Parameters for the call.
            distributed_subcall: Whether this is a subcall from another node.

        Returns:
            The result of the callable execution.
        """
        request_id = request.headers.get("X-Request-ID", "-")
        serialization = request.headers.get("X-Serialization", "json")

        debug_mode, debug_port = None, None
        debugger: dict = params.get("debugger", None) if params else None
        if debugger:
            debug_mode = debugger.get("mode")
            debug_port = debugger.get("port")

        # For local execution, route to the first (and typically only) subprocess
        logger.debug(f"Routing call to subprocess: {cls_or_fn_name}.{method_name}")
        result = self.process_pool.call(
            idx=0,
            method_name=method_name,
            params=params,
            request_id=request_id,
            distributed_env_vars={},  # No distributed env vars for local execution
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
