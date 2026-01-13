import copy
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import httpx
from starlette.responses import JSONResponse

from kubetorch.serving.distributed_supervisor import DistributedSupervisor
from kubetorch.serving.global_http_clients import get_sync_client
from kubetorch.serving.http_server import logger
from kubetorch.serving.process_worker import ProcessWorker
from kubetorch.serving.remote_worker_pool import RemoteWorkerPool


class SPMDDistributedSupervisor(DistributedSupervisor):
    """SPMD (Single Program Multiple Data) distributed supervisor.

    This class provides distributed execution for frameworks that follow the SPMD pattern
    where the same program runs on multiple processes with different data partitions.

    Features:
    - Multi-process local execution (configurable num_proc)
    - Remote worker coordination via RemoteWorkerPool
    - Tree topology for large clusters (>100 workers)
    - DNS-based worker discovery with quorum support
    """

    def __init__(
        self,
        process_class=None,
        num_proc=None,
        port=None,
        tree_fanout=50,
        tree_minimum=100,
        **kwargs,
    ):
        """Initialize SPMD supervisor.

        Args:
            process_class (type, optional): ProcessWorker subclass for framework-specific execution. (Default: None)
            num_proc (int, optional): Number of local processes ("auto" to detect from process_class). (Default: None)
            port (int, optional): Port for distributed communication (framework-specific). (Default: None)
            tree_fanout (int, optional): Max children per node in tree topology. (Default: 50)
            tree_minimum (int, optional): Min cluster size to use tree topology. (Default: 100)
            **kwargs: Arguments passed to DistributedSupervisor (quorum_*, monitor_members, etc.)
        """
        # Map num_proc to num_processes for parent class
        super().__init__(
            process_class=process_class or ProcessWorker,
            num_processes=num_proc or "auto",
            **kwargs,
        )
        self.port = port
        self.tree_fanout = tree_fanout
        self.tree_minimum = tree_minimum

    # Use num_proc as alias for num_processes for backward compatibility
    @property
    def num_proc(self):
        return self.num_processes

    @num_proc.setter
    def num_proc(self, value):
        self.num_processes = value

    def get_tree_children(self, sorted_ips: list, my_ip: str, fanout: int = 100):
        """Calculate children nodes in a self-organizing tree based on IP indexing.

        Args:
            sorted_ips (list): List of all worker IPs sorted deterministically.
            my_ip (str): This node's IP address.
            fanout (int): Maximum number of children per node. (Default: 100)

        Returns:
            List of IP addresses that are children of this node.
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

    def call(
        self,
        request,
        cls_or_fn_name: str,
        method_name: Optional[str] = None,
        params: Optional[Dict] = None,
        distributed_subcall: bool = False,
    ):
        # Get the request ID from the headers
        request_id = request.headers.get("X-Request-ID", "-")
        serialization = request.headers.get("X-Serialization", "json")
        params = params or {}

        # Extract debugger config from params
        debug_mode, debug_port = None, None
        debugger: dict = params.get("debugger", None) if params else None
        if debugger:
            debug_mode = debugger.get("mode")
            debug_port = debugger.get("port")

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
                # Now that we have quorum, start DNS monitoring with the discovered IPs
                # Pass the known worker_ips to avoid re-querying DNS (which may return
                # different results during DNS propagation delays)
                self.start_dns_monitoring(initial_workers=set(worker_ips))

                # Subscribe to membership changes
                change_event = self.subscribe_to_membership_changes()
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
            self.setup()

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
            client = get_sync_client()
            port = os.environ["KT_SERVER_PORT"]
            worker_url = f"http://{worker_ip}:{port}"
            # First check that the worker is alive, replicas don't finish setup at exactly the same moment
            # Use quorum_timeout to control how long to wait for workers
            for i in range(int(self.quorum_timeout)):
                try:
                    resp = client.get(f"{worker_url}/health", timeout=5.0)
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
                    headers=clean_headers,
                    timeout=None,  # No timeout for distributed calls
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
                # Create RemoteWorkerPool lazily when first needed
                # This avoids spawning unnecessary processes for single-pod distributed jobs
                logger.info(f"Creating RemoteWorkerPool to call {len(subcall_ips)} remote workers")
                self.remote_worker_pool = RemoteWorkerPool(quorum_timeout=self.quorum_timeout)
                self.remote_worker_pool.start(max_workers=min(len(subcall_ips) + 50, 200))
            logger.debug(f"Using RemoteWorkerPool to call {len(subcall_ips)} workers")

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
                                "x-deployed-as-of",  # No longer needed with push-based reload
                            ]:
                                clean_headers[key] = value

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

        # Verify RemoteWorkerPool is running if we have remote workers
        # (It should have been created lazily above when subcall_ips was first checked)
        if subcall_ips:
            if not self.remote_worker_pool:
                # Should not happen - pool is created lazily above
                raise RuntimeError(
                    f"RemoteWorkerPool not available for worker at {this_pod_ip}. "
                    "This is required for distributed execution with subcall_ips."
                )
            elif (
                not hasattr(self.remote_worker_pool, "process")
                or not self.remote_worker_pool.process
                or not self.remote_worker_pool.process.is_alive()
            ):
                # Pool exists but not started/alive - restart it
                logger.warning(f"RemoteWorkerPool exists but not running for {this_pod_ip}, starting it now")
                self.remote_worker_pool.start(max_workers=min(len(subcall_ips) + 50, 200))

        # Submit local process calls
        def call_local_processes():
            logger.debug(f"Processing {num_procs} local process responses")
            return self.process_pool.call_all(
                method_name=method_name,
                params_list=params_list,
                request_id=request_id,
                distributed_env_vars_list=distributed_env_vars_list,
                debug_ports=debug_ports,
                debug_mode=debug_mode,
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
