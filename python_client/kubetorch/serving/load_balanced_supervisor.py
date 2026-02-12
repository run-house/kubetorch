"""Load-balanced supervisor for routing calls to least-busy workers.

This module provides the LoadBalancedSupervisor class which extends DistributedSupervisor
with load-balanced routing capabilities. Unlike SPMD modes that call all workers,
load-balanced mode routes each call to a single worker based on availability.

Key features:
- Slot-based routing: Track in-flight calls per worker, route to least-busy
- Async implementation: Uses asyncio for high concurrency with minimal overhead
- Queue support: Optionally queue calls when all workers at capacity
- Reuses RemoteWorkerPool WebSocket infrastructure for low-latency routing
"""

import asyncio
import os
from typing import Dict, Optional

from kubetorch.serving.distributed_supervisor import DistributedSupervisor
from kubetorch.serving.http_server import logger
from kubetorch.serving.process_worker import ProcessWorker


class LoadBalancedSupervisor(DistributedSupervisor):
    """Load-balanced supervisor using async slot-based routing.

    Routes each call to a single worker based on availability.
    Uses RemoteWorkerPool's async WebSocket infrastructure for remote calls.

    Unlike SPMD supervisors that broadcast calls to all workers, this supervisor
    routes each call to the worker with the most available capacity.
    """

    def __init__(
        self,
        concurrency: int = 40,
        queue_size: Optional[int] = None,
        **kwargs,
    ):
        """Initialize load-balanced supervisor.

        Args:
            concurrency (int): Max concurrent calls per worker. Matches FastAPI's
                default of 40 threads for consistent behavior. (Default: 40)
            queue_size (int, optional): Max pending calls when all workers busy.
                None means unlimited queue. (Default: None)
            **kwargs: Arguments passed to DistributedSupervisor (quorum_*, monitor_members, etc.)
        """
        # Map concurrency to max_threads_per_proc for parent class
        kwargs["max_threads_per_proc"] = concurrency
        super().__init__(process_class=ProcessWorker, **kwargs)

        self.concurrency = concurrency
        self.queue_size = queue_size

        # Slot tracking (worker_ip -> in_flight count)
        self._worker_slots: Dict[str, int] = {}
        self._slots_lock: Optional[asyncio.Lock] = None

        # Pending call queue
        self._pending_queue: Optional[asyncio.Queue] = None
        self._queue_processor_task: Optional[asyncio.Task] = None

    def setup(self):
        """Set up load-balanced execution environment."""
        super().setup()

        # Create async primitives (must be done in async context or lazily)
        # These will be initialized on first async call
        self._worker_slots = {}

    def cleanup(self):
        """Clean up load-balanced resources."""
        # Cancel queue processor if running
        if self._queue_processor_task and not self._queue_processor_task.done():
            self._queue_processor_task.cancel()
            self._queue_processor_task = None

        self._worker_slots = {}
        self._slots_lock = None
        self._pending_queue = None

        super().cleanup()

    def _ensure_async_primitives(self):
        """Lazily initialize async primitives (must be called from async context)."""
        if self._slots_lock is None:
            self._slots_lock = asyncio.Lock()
        if self._pending_queue is None:
            maxsize = self.queue_size or 0  # 0 = unlimited
            self._pending_queue = asyncio.Queue(maxsize=maxsize)

    def _initialize_slots(self, worker_ips: list):
        """Initialize slot tracking for new workers."""
        for ip in worker_ips:
            if ip not in self._worker_slots:
                self._worker_slots[ip] = 0

    async def _acquire_slot(self, worker_ips: list) -> Optional[str]:
        """Find worker with available slot, return None if all busy.

        Uses least-loaded routing: selects the worker with the most available slots.

        Args:
            worker_ips: List of worker IP addresses to consider.

        Returns:
            IP address of selected worker, or None if all workers at capacity.
        """
        async with self._slots_lock:
            # Find least-busy worker with available slots
            best_worker = None
            best_available = -1

            for ip in worker_ips:
                in_flight = self._worker_slots.get(ip, 0)
                available = self.concurrency - in_flight

                if available > best_available:
                    best_available = available
                    best_worker = ip

            if best_available > 0:
                self._worker_slots[best_worker] = self._worker_slots.get(best_worker, 0) + 1
                logger.debug(
                    f"Acquired slot on {best_worker} " f"({self._worker_slots[best_worker]}/{self.concurrency} in use)"
                )
                return best_worker

            return None

    async def _release_slot(self, worker_ip: str):
        """Release a slot on a worker."""
        async with self._slots_lock:
            self._worker_slots[worker_ip] = max(0, self._worker_slots.get(worker_ip, 0) - 1)
            logger.debug(
                f"Released slot on {worker_ip} " f"({self._worker_slots[worker_ip]}/{self.concurrency} in use)"
            )

    async def call_async(
        self,
        request,
        cls_or_fn_name: str,
        method_name: Optional[str] = None,
        params: Optional[Dict] = None,
        distributed_subcall: bool = False,
    ):
        """Async call method for load-balanced routing.

        Routes each call to a single worker based on availability.

        Args:
            request: The HTTP request object.
            cls_or_fn_name: Name of the callable.
            method_name: Method name to call (for class instances).
            params: Parameters for the call.
            distributed_subcall: Whether this is a subcall from another node.

        Returns:
            The result of the callable execution.
        """
        self._ensure_async_primitives()
        params = params or {}

        if distributed_subcall:
            # Worker receiving routed call - execute locally
            return await self._execute_local_async(request, cls_or_fn_name, method_name, params)

        # Coordinator - find available workers via fast DNS lookup (no quorum wait).
        # Workers can join/leave dynamically; we route to whatever's available now.
        worker_ips = self._get_pod_ips_fast()
        if not worker_ips:
            # No workers found - fall back to quorum-waiting pod_ips() for initial startup
            worker_ips = self.pod_ips()
        self._initialize_slots(worker_ips)

        this_pod_ip = os.environ.get("POD_IP", "")
        worker_ip = await self._acquire_slot(worker_ips)

        if worker_ip is None:
            # All workers at capacity - queue the call if queue is enabled
            if self.queue_size is not None:
                logger.debug("All workers at capacity, queueing call")
                return await self._queue_and_wait(request, cls_or_fn_name, method_name, params)
            else:
                # No queue - wait briefly and retry
                logger.debug("All workers at capacity, waiting for slot...")
                while worker_ip is None:
                    await asyncio.sleep(0.01)
                    worker_ip = await self._acquire_slot(worker_ips)

        try:
            if worker_ip == this_pod_ip:
                # Route to self - use local process pool
                logger.debug(f"Routing call to local worker (self: {this_pod_ip})")
                return await self._execute_local_async(request, cls_or_fn_name, method_name, params)
            else:
                # Route to remote worker
                logger.debug(f"Routing call to remote worker: {worker_ip}")
                return await self._route_to_worker_async(worker_ip, cls_or_fn_name, method_name, params, request)
        finally:
            await self._release_slot(worker_ip)

    async def _execute_local_async(
        self,
        request,
        cls_or_fn_name: str,
        method_name: Optional[str],
        params: Dict,
    ):
        """Execute call locally via process pool."""
        request_id = request.headers.get("X-Request-ID", "-")
        serialization = request.headers.get("X-Serialization", "json")

        debug_mode, debug_port = None, None
        debugger: dict = params.get("debugger", None) if params else None
        if debugger:
            debug_mode = debugger.get("mode")
            debug_port = debugger.get("port")

        # Run in thread pool to avoid blocking event loop
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.process_pool.call(
                idx=0,
                method_name=method_name,
                params=params,
                request_id=request_id,
                distributed_env_vars={},
                debug_port=debug_port,
                debug_mode=debug_mode,
                serialization=serialization,
            ),
        )
        return result

    async def _route_to_worker_async(
        self,
        worker_ip: str,
        cls_or_fn_name: str,
        method_name: Optional[str],
        params: Dict,
        request,
    ):
        """Route call to remote worker via WebSocket.

        Uses RemoteWorkerPool for WebSocket-based communication.
        """
        # Initialize pool if needed
        if not self.remote_worker_pool:
            from kubetorch.serving.remote_worker_pool import RemoteWorkerPool

            self.remote_worker_pool = RemoteWorkerPool(quorum_timeout=self.quorum_timeout)
            self.remote_worker_pool.start()

        request_headers = {
            "X-Request-ID": request.headers.get("X-Request-ID", "-"),
            "X-Serialization": request.headers.get("X-Serialization", "json"),
        }

        # Mark as distributed subcall for the receiving worker
        params_with_flag = dict(params)

        # Call single worker (not all workers like SPMD)
        # Run in thread pool since RemoteWorkerPool.call_workers is sync
        results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.remote_worker_pool.call_workers(
                worker_ips=[worker_ip],
                cls_or_fn_name=cls_or_fn_name,
                method_name=method_name,
                params=params_with_flag,
                request_headers=request_headers,
            ),
        )
        return results[0]  # Single result, not list

    async def _queue_and_wait(
        self,
        request,
        cls_or_fn_name: str,
        method_name: Optional[str],
        params: Dict,
    ):
        """Queue call and wait for available worker."""
        # Create future for result
        result_future = asyncio.get_event_loop().create_future()

        # Queue the pending call
        pending = (request, cls_or_fn_name, method_name, params, result_future)
        await self._pending_queue.put(pending)

        # Ensure queue processor is running
        if self._queue_processor_task is None or self._queue_processor_task.done():
            self._queue_processor_task = asyncio.create_task(self._process_queue())

        # Wait for result
        return await result_future

    async def _process_queue(self):
        """Background task that dispatches queued calls when workers available."""
        worker_ips = self._get_pod_ips_fast() or self.pod_ips()
        this_pod_ip = os.environ.get("POD_IP", "")

        while True:
            try:
                # Get pending call (with timeout to allow checking for shutdown)
                try:
                    pending = await asyncio.wait_for(self._pending_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Check if there are more items
                    if self._pending_queue.empty():
                        break
                    continue

                request, cls_or_fn_name, method_name, params, result_future = pending

                # Wait for available worker
                worker_ip = None
                while worker_ip is None:
                    worker_ip = await self._acquire_slot(worker_ips)
                    if worker_ip is None:
                        await asyncio.sleep(0.01)

                # Route and complete future
                try:
                    if worker_ip == this_pod_ip:
                        result = await self._execute_local_async(request, cls_or_fn_name, method_name, params)
                    else:
                        result = await self._route_to_worker_async(
                            worker_ip, cls_or_fn_name, method_name, params, request
                        )

                    if not result_future.done():
                        result_future.set_result(result)
                except Exception as e:
                    if not result_future.done():
                        result_future.set_exception(e)
                finally:
                    await self._release_slot(worker_ip)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in queue processor: {e}")
                continue

    def call(
        self,
        request,
        cls_or_fn_name: str,
        method_name: Optional[str] = None,
        params: Optional[Dict] = None,
        distributed_subcall: bool = False,
    ):
        """Synchronous call method - wraps async call.

        For load-balanced mode, we prefer the async path for better concurrency.
        This method is provided for compatibility with the ExecutionSupervisor interface.
        """
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # Already in async context - create task
            # This shouldn't happen in normal usage since http_server handles this
            raise RuntimeError(
                "LoadBalancedSupervisor.call() called from async context. "
                "Use call_async() directly or let http_server handle routing."
            )
        else:
            # No event loop - create one and run
            return asyncio.run(self.call_async(request, cls_or_fn_name, method_name, params, distributed_subcall))
