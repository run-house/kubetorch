"""Base classes for distributed execution."""

import multiprocessing
import os
import queue
import threading
import time
from bdb import BdbQuit
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

from kubetorch.logger import get_logger
from kubetorch.servers.http.utils import clear_debugging_sessions, is_running_in_kubernetes

logger = get_logger(__name__)


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
        # Lazy import to avoid circular dependency with http_server
        from kubetorch.servers.http.http_server import (
            load_callable,
            package_exception,
            request_id_ctx_var,
            run_callable_internal_sync,
        )

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
