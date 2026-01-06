import os
import queue
import threading
import time
from typing import Dict, Optional

from kubetorch.serving.http_server import logger
from kubetorch.serving.utils import is_running_in_kubernetes


class ExecutionSupervisor:
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
        # Always compute headless service DNS for distributed mode to ensure we get pod IPs
        # (KT_SERVICE_DNS may have been set before distributed config was applied)
        service_name = os.environ.get("KT_SERVICE_NAME")
        namespace = os.environ.get("POD_NAMESPACE")

        if not service_name:
            raise RuntimeError("KT_SERVICE_NAME environment variable not found")
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

            # If no expected count is set, return immediately with whatever we found
            if not expected_workers:
                if pod_ips:
                    logger.debug(f"{len(pod_ips)} workers found, no quorum set")
                    return pod_ips
                # No pods found yet and no quorum to wait for - wait briefly and retry once
                if elapsed >= 5.0:
                    logger.debug(f"No workers found after {elapsed:.1f}s, no quorum set")
                    return pod_ips

            # If timeout is reached, return what we have
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
                last_count = len(pod_ips)

            # Wait before retrying
            time.sleep(2)

    def _get_pod_ips_fast(self):
        """Get pod IPs from DNS without waiting for quorum - for monitoring only."""
        # Primarily for testing
        if not is_running_in_kubernetes():
            return os.environ["LOCAL_IPS"].split(",")

        # Use DNS-based service discovery
        # Always compute headless service DNS for distributed mode to ensure we get pod IPs
        # (KT_SERVICE_DNS may have been set before distributed config was applied)
        service_name = os.environ.get("KT_SERVICE_NAME")
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
                    from kubetorch.serving.utils import WorkerMembershipChanged

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
            from kubetorch.serving.utils import WorkerMembershipChanged

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
        debug_mode: str = None,
        deployed_as_of: Optional[str] = None,
    ):
        # if intercept_call is True, this method should be overridden by subclasses to handle distributing and/or
        # supervising the distributed execution
        raise NotImplementedError("call_distributed() must be implemented by subclasses")
