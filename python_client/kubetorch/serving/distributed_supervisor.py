"""Distributed supervisor base class with DNS monitoring, quorum, and remote worker support.

This module provides the DistributedSupervisor class which extends ExecutionSupervisor
with distributed execution capabilities including:
- DNS-based pod discovery with quorum support
- Worker membership monitoring
- Remote worker pool for cross-pod execution
"""

import os
import queue
import socket
import threading
import time
from typing import Optional, Set

from kubetorch.serving.execution_supervisor import ExecutionSupervisor
from kubetorch.serving.http_server import logger
from kubetorch.serving.remote_worker_pool import RemoteWorkerPool
from kubetorch.serving.utils import is_running_in_kubernetes, WorkerMembershipChanged


class DistributedSupervisor(ExecutionSupervisor):
    """Base class for distributed supervisors with remote worker support.

    Extends ExecutionSupervisor with:
    - DNS-based pod discovery with quorum waiting
    - Worker membership monitoring and change detection
    - Remote worker pool for HTTP calls to other pods

    Subclasses: SPMDDistributedSupervisor, RayDistributed, MonarchDistributed
    """

    def __init__(
        self,
        quorum_workers: Optional[int] = None,
        quorum_timeout: int = 300,
        monitor_members: bool = True,
        **kwargs,
    ):
        """Initialize distributed supervisor.

        Args:
            quorum_workers (int, optional): Number of workers to wait for before proceeding.
                If None, returns immediately with whatever pods are found. (Default: None)
            quorum_timeout (int, optional): Maximum seconds to wait for quorum. (Default: 300)
            monitor_members (bool, optional): Whether to monitor for worker membership changes.
                Set to False for frameworks like Ray that manage their own membership. (Default: True)
            **kwargs: Additional arguments passed to ExecutionSupervisor.
        """
        super().__init__(**kwargs)
        self.quorum_workers = quorum_workers
        self.quorum_timeout = quorum_timeout
        self.monitor_members = monitor_members

        # Remote worker pool for HTTP calls to other pods
        self.remote_worker_pool: Optional[RemoteWorkerPool] = None

        # DNS monitoring state
        self._dns_monitor_thread: Optional[threading.Thread] = None
        self._dns_monitor_running: bool = False
        self._current_workers: Set[str] = set()
        self._workers_lock = threading.Lock()
        self._membership_changes: queue.Queue = queue.Queue()
        self._change_subscribers: list = []
        self._last_dns_check: float = 0
        self._dns_check_interval: int = 5  # seconds

    def setup(self):
        """Set up distributed environment including process pool.

        Note: RemoteWorkerPool is created lazily when needed (i.e., when there are
        actually remote workers to call). This avoids spawning unnecessary processes
        for single-pod distributed jobs.
        """
        super().setup()

    def cleanup(self):
        """Clean up distributed resources."""
        # Stop DNS monitoring first
        self.stop_dns_monitoring()

        # Stop remote worker pool
        if self.remote_worker_pool:
            self.remote_worker_pool.stop()
            self.remote_worker_pool = None
            logger.debug("Stopped remote worker pool")

        # Clean up process pool
        super().cleanup()

    def pod_ips(self) -> list:
        """Get pod IPs from DNS, waiting for quorum if specified.

        Will wait up to quorum_timeout seconds for quorum_workers to appear in DNS.
        If quorum_workers is not specified, returns immediately after first DNS query.

        Returns:
            List of pod IP addresses discovered via DNS.
        """
        # Primarily for testing
        if not is_running_in_kubernetes():
            return os.environ.get("LOCAL_IPS", "").split(",")

        # Use DNS-based service discovery
        service_name = os.environ.get("KT_SERVICE_NAME")
        namespace = os.environ.get("POD_NAMESPACE")

        if not service_name:
            raise RuntimeError("KT_SERVICE_NAME environment variable not found")
        if not namespace:
            raise RuntimeError("POD_NAMESPACE environment variable not found")

        # Kubernetes headless service DNS name for distributed pod discovery
        service_dns = f"{service_name}-headless.{namespace}.svc.cluster.local"

        start_time = time.time()
        max_wait = self.quorum_timeout if self.quorum_timeout else 0
        expected_workers = self.quorum_workers

        pod_ips = []
        last_count = 0

        while True:
            try:
                # DNS lookup returns all pod IPs for the headless service
                addr_info = socket.getaddrinfo(service_dns, None, socket.AF_INET)
                pod_ips = sorted(list(set([addr[4][0] for addr in addr_info])))

                if not pod_ips:
                    logger.debug(f"No pod IPs found for service {service_dns}")
                else:
                    logger.debug(f"Found {len(pod_ips)} pod IPs via DNS for {service_dns}: {pod_ips}")

            except socket.gaierror as e:
                logger.debug(f"DNS lookup failed for {service_dns}: {e}")
                pod_ips = []

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

            time.sleep(2)

    def _get_pod_ips_fast(self) -> list:
        """Get pod IPs from DNS without waiting for quorum - for monitoring only."""
        if not is_running_in_kubernetes():
            return os.environ.get("LOCAL_IPS", "").split(",")

        service_name = os.environ.get("KT_SERVICE_NAME")
        namespace = os.environ.get("POD_NAMESPACE")

        if not service_name or not namespace:
            return []

        service_dns = f"{service_name}-headless.{namespace}.svc.cluster.local"

        try:
            addr_info = socket.getaddrinfo(service_dns, None, socket.AF_INET)
            pod_ips = sorted(list(set([addr[4][0] for addr in addr_info])))
            return pod_ips
        except socket.gaierror:
            with self._workers_lock:
                return list(self._current_workers)

    def start_dns_monitoring(self, initial_workers: Optional[Set[str]] = None):
        """Start DNS monitoring if not already running.

        Should be called by coordinator nodes only.

        Args:
            initial_workers (set, optional): Pre-discovered worker IPs to use as the baseline.
                If provided, avoids re-querying DNS which can return different results
                during DNS propagation delays. If None, queries DNS to discover workers.
        """
        if not self.monitor_members:
            logger.debug("DNS monitoring disabled for this supervisor")
            return

        with self._workers_lock:
            if self._dns_monitor_thread and self._dns_monitor_thread.is_alive():
                return  # Already running

            # Initialize with provided workers or query DNS
            if initial_workers is not None:
                self._current_workers = set(initial_workers)
                logger.debug(f"Starting DNS monitor with {len(self._current_workers)} pre-discovered workers")
            else:
                self._current_workers = set(self.pod_ips())
                logger.debug(f"Starting DNS monitor with {len(self._current_workers)} workers from DNS")

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
        check_interval = 3  # Start with 3 second checks

        while self._dns_monitor_running:
            try:
                time.sleep(check_interval)

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

                        self._membership_changes.put(change)
                        for event in self._change_subscribers:
                            event.set()

                        self._current_workers = current_ips

                time.sleep(check_interval)

            except Exception as e:
                logger.error(f"DNS monitor error: {e}")
                time.sleep(3)

    def subscribe_to_membership_changes(self) -> threading.Event:
        """Subscribe to worker membership changes.

        Returns:
            An event that will be set when changes occur.
        """
        event = threading.Event()
        with self._workers_lock:
            self._change_subscribers.append(event)
        return event

    def unsubscribe_from_membership_changes(self, event: threading.Event):
        """Unsubscribe from worker membership changes."""
        with self._workers_lock:
            if event in self._change_subscribers:
                self._change_subscribers.remove(event)

    def check_for_membership_changes(self, force_dns_check: bool = False):
        """Check for membership changes and raise exception if any occurred.

        Args:
            force_dns_check: If True, immediately query DNS to check for changes
                            instead of relying on the monitoring thread.

        Raises:
            WorkerMembershipChanged: If worker membership has changed.
        """
        if not self.monitor_members:
            return

        if force_dns_check:
            current_ips = set(self._get_pod_ips_fast())
            with self._workers_lock:
                if current_ips != self._current_workers:
                    added = current_ips - self._current_workers
                    removed = self._current_workers - current_ips
                    previous_ips = self._current_workers.copy()

                    # Update current workers
                    self._current_workers = current_ips

                    if removed:
                        logger.error(f"Workers REMOVED from cluster (forced check): {removed}")
                    if added:
                        logger.warning(f"Workers ADDED to cluster (forced check): {added}")

                    raise WorkerMembershipChanged(
                        added_ips=added,
                        removed_ips=removed,
                        previous_ips=previous_ips,
                        current_ips=current_ips,
                    )

        # Check queued changes from monitoring thread
        try:
            change = self._membership_changes.get_nowait()
            raise WorkerMembershipChanged(
                added_ips=change["added"],
                removed_ips=change["removed"],
                previous_ips=change["previous"],
                current_ips=change["current"],
            )
        except queue.Empty:
            pass
