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
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

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
        self._rr_index: int = 0  # Round-robin start index for even distribution

        # Cached DNS results — avoids blocking socket.getaddrinfo() on every call
        self._cached_worker_ips: List[str] = []
        self._cached_worker_ips_time: float = 0.0
        self._dns_cache_ttl: float = 2.0  # seconds

        # Dedicated executor for ProcessPool.call() — avoids the default executor
        # which is limited to min(32, cpu_count+4) threads (5 on 1-CPU pods).
        # Each thread just blocks on threading.Event (zero CPU cost), so sizing
        # to concurrency is safe.
        self._call_executor: Optional[ThreadPoolExecutor] = None

        # Pending call queue
        self._pending_queue: Optional[asyncio.Queue] = None
        self._queue_processor_task: Optional[asyncio.Task] = None

    def setup(self):
        """Set up load-balanced execution environment."""
        super().setup()
        self._worker_slots = {}

    def cleanup(self):
        """Clean up load-balanced resources."""
        # Cancel queue processor if running
        if self._queue_processor_task and not self._queue_processor_task.done():
            self._queue_processor_task.cancel()
            self._queue_processor_task = None

        # Close in-process async WebSocket connections
        if self.remote_worker_pool:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.remote_worker_pool.cleanup_async())
            except RuntimeError:
                # No running loop — run synchronously
                asyncio.run(self.remote_worker_pool.cleanup_async())

        if self._call_executor:
            self._call_executor.shutdown(wait=False)
            self._call_executor = None

        self._worker_slots = {}
        self._pending_queue = None

        super().cleanup()

    def _ensure_async_primitives(self):
        """Lazily initialize async primitives (must be called from async context)."""
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

        Uses least-loaded routing with round-robin tiebreaking for even distribution.
        No lock needed — asyncio is single-threaded and there are no await points here.

        Args:
            worker_ips: List of worker IP addresses to consider.

        Returns:
            IP address of selected worker, or None if all workers at capacity.
        """
        n = len(worker_ips)
        best_worker = None
        best_available = -1

        for i in range(n):
            ip = worker_ips[(self._rr_index + i) % n]
            in_flight = self._worker_slots.get(ip, 0)
            available = self.concurrency - in_flight

            if available > best_available:
                best_available = available
                best_worker = ip

        if best_available > 0:
            self._worker_slots[best_worker] = self._worker_slots.get(best_worker, 0) + 1
            self._rr_index = (self._rr_index + 1) % n
            logger.debug(
                f"Acquired slot on {best_worker} ({self._worker_slots[best_worker]}/{self.concurrency} in use)"
            )
            return best_worker

        return None

    async def _release_slot(self, worker_ip: str):
        """Release a slot on a worker.

        No lock needed — asyncio is single-threaded and there are no await points here.
        """
        self._worker_slots[worker_ip] = max(0, self._worker_slots.get(worker_ip, 0) - 1)
        logger.debug(f"Released slot on {worker_ip} ({self._worker_slots[worker_ip]}/{self.concurrency} in use)")

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
        """
        self._ensure_async_primitives()
        params = params or {}

        if distributed_subcall:
            # Worker receiving routed call - execute locally
            return await self._execute_local_async(request, cls_or_fn_name, method_name, params)

        # Coordinator - use cached DNS to avoid blocking getaddrinfo() on every call.
        now = time.monotonic()
        if not self._cached_worker_ips or (now - self._cached_worker_ips_time) > self._dns_cache_ttl:
            worker_ips = self._get_pod_ips_fast()
            if not worker_ips:
                worker_ips = self.pod_ips()
            self._cached_worker_ips = worker_ips
            self._cached_worker_ips_time = now
        worker_ips = self._cached_worker_ips
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

        # Use dedicated executor — the default executor is limited to
        # min(32, cpu_count+4) threads which bottlenecks 1-CPU pods at 5 threads.
        # Each thread just blocks on threading.Event (zero CPU), so this is safe.
        if self._call_executor is None:
            self._call_executor = ThreadPoolExecutor(max_workers=self.concurrency)

        result = await asyncio.get_event_loop().run_in_executor(
            self._call_executor,
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
        """Route call to remote worker via async WebSocket.

        Uses RemoteWorkerPool's in-process async path — no subprocess, no pickle.
        """
        if not self.remote_worker_pool:
            from kubetorch.serving.remote_worker_pool import RemoteWorkerPool

            self.remote_worker_pool = RemoteWorkerPool(quorum_timeout=self.quorum_timeout)
            # No .start() needed — async path doesn't use subprocess

        request_headers = {
            "X-Request-ID": request.headers.get("X-Request-ID", "-"),
            "X-Serialization": request.headers.get("X-Serialization", "json"),
        }

        # workers_arg=None skips health checks — load-balanced routes to
        # whatever workers are available, no quorum needed.
        results = await self.remote_worker_pool.call_workers_async(
            worker_ips=[worker_ip],
            cls_or_fn_name=cls_or_fn_name,
            method_name=method_name,
            params=params,
            request_headers=request_headers,
            workers_arg=None,
        )
        return results[0]

    async def _queue_and_wait(
        self,
        request,
        cls_or_fn_name: str,
        method_name: Optional[str],
        params: Dict,
    ):
        """Queue call and wait for available worker."""
        result_future = asyncio.get_event_loop().create_future()

        pending = (request, cls_or_fn_name, method_name, params, result_future)
        await self._pending_queue.put(pending)

        # Ensure queue processor is running
        if self._queue_processor_task is None or self._queue_processor_task.done():
            self._queue_processor_task = asyncio.create_task(self._process_queue())

        return await result_future

    async def _process_queue(self):
        """Background task that dispatches queued calls when workers available."""
        worker_ips = self._get_pod_ips_fast() or self.pod_ips()
        this_pod_ip = os.environ.get("POD_IP", "")

        while True:
            try:
                try:
                    pending = await asyncio.wait_for(self._pending_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    if self._pending_queue.empty():
                        break
                    continue

                request, cls_or_fn_name, method_name, params, result_future = pending

                worker_ip = None
                while worker_ip is None:
                    worker_ip = await self._acquire_slot(worker_ips)
                    if worker_ip is None:
                        await asyncio.sleep(0.01)

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
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            raise RuntimeError(
                "LoadBalancedSupervisor.call() called from async context. "
                "Use call_async() directly or let http_server handle routing."
            )
        else:
            return asyncio.run(self.call_async(request, cls_or_fn_name, method_name, params, distributed_subcall))
