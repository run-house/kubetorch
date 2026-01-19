"""
Metrics Push for OTEL-free metrics collection.

Pushes metrics directly to Prometheus Pushgateway without OTEL daemonset.
"""

import logging
import os
import threading
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


HEARTBEAT_INTERVAL_DIVISOR = 5


class MetricsPusher:
    """Push metrics directly to Prometheus Pushgateway."""

    def __init__(
        self,
        pushgateway_url: str,
        job_name: str,
        push_interval: float = 15.0,
        grouping_key: Optional[dict] = None,
        ttl_seconds: Optional[int] = None,
    ):
        """
        Initialize metrics pusher.

        Args:
            pushgateway_url (str): URL for Prometheus Pushgateway (e.g., http://pushgateway:9091).
            job_name (str): Job name for Prometheus metrics.
            push_interval (float, optional): Seconds between automatic pushes. (Default: 15.0)
            grouping_key (dict, optional): Additional grouping keys for Pushgateway. (Default: None)
            ttl_seconds (int, optional): Time to live in seconds. (Default: None)
        """
        try:
            from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
        except ImportError:
            logger.warning("prometheus_client not installed, MetricsPusher disabled")
            self._enabled = False
            return

        self._enabled = True
        self.pushgateway_url = pushgateway_url
        self.job_name = job_name
        self.push_interval = push_interval
        self.grouping_key = grouping_key or {}

        # TTL tracking
        self.ttl_seconds = ttl_seconds or 0
        self.heartbeat_interval = (self.ttl_seconds // HEARTBEAT_INTERVAL_DIVISOR) if self.ttl_seconds else 0
        self._last_activity = datetime.now()

        self.registry = CollectorRegistry()
        self._stop_event = threading.Event()
        self._push_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None

        # Standard HTTP metrics
        self.request_count = Counter(
            "http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"],
            registry=self.registry,
        )
        self.request_latency = Histogram(
            "http_request_duration_seconds",
            "HTTP request latency",
            ["method", "endpoint"],
            registry=self.registry,
        )

        # TTL activity metric
        self.last_activity_gauge = Gauge(
            "kubetorch_last_activity_timestamp",
            "Last activity timestamp",
            ["service", "kubetorch_version", "namespace", "service_type"],
            registry=self.registry,
        )

        # Active requests gauge
        self.active_requests_gauge = Gauge(
            "http_server_active_requests",
            "Number of currently active requests",
            ["service", "kubetorch_version", "namespace", "service_type"],
            registry=self.registry,
        )

        # Heartbeat counter (for compatibility with existing TTL queries)
        self.heartbeat_counter = Counter(
            "kt_heartbeat_sent",
            "Total heartbeats sent",
            ["service", "kubetorch_version", "namespace", "service_type"],
            registry=self.registry,
        )

        self._started = False
        self._active_requests = 0

        # Store label values for metrics
        self._labels = {
            "service": os.environ.get("KT_SERVICE", "unknown-service"),
            "kubetorch_version": self._get_package_version(),
            "namespace": os.environ.get("POD_NAMESPACE", "default"),
            "service_type": os.environ.get("KT_DEPLOYMENT_MODE", "deployment"),
        }

    @staticmethod
    def _get_package_version() -> str:
        """Get kubetorch package version."""
        try:
            from importlib.metadata import version

            return version("kubetorch")
        except Exception:
            return "0.0.0"

    def start(self):
        """Start the metrics pusher background threads."""
        if not self._enabled or self._started:
            return

        self._push_thread = threading.Thread(target=self._push_loop, daemon=True)
        self._push_thread.start()

        if self.ttl_seconds > 0:
            self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._heartbeat_thread.start()

        self._started = True
        logger.debug(f"MetricsPusher started - pushing to {self.pushgateway_url}")

    def stop(self):
        """Stop the metrics pusher and push final metrics."""
        if not self._enabled or not self._started:
            return

        self._stop_event.set()

        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2.0)
        if self._push_thread:
            self._push_thread.join(timeout=2.0)

        self._push_now()
        self._started = False

    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record an HTTP request metric."""
        if not self._enabled:
            return

        self.request_count.labels(method=method, endpoint=endpoint, status=status).inc()
        self.request_latency.labels(method=method, endpoint=endpoint).observe(duration)
        self.last_activity_gauge.labels(**self._labels).set(time.time())
        self._last_activity = datetime.now()

    def request_started(self):
        """Called when a request starts."""
        if not self._enabled:
            return
        self._active_requests += 1
        self.active_requests_gauge.labels(**self._labels).inc()
        self.last_activity_gauge.labels(**self._labels).set(time.time())
        self._last_activity = datetime.now()
        self._send_heartbeat()

    def request_finished(self):
        """Called when a request finishes."""
        if not self._enabled:
            return
        self._active_requests = max(0, self._active_requests - 1)  # prevent negative values
        self.active_requests_gauge.labels(**self._labels).dec()
        self.last_activity_gauge.labels(**self._labels).set(time.time())
        self._last_activity = datetime.now()
        self._send_heartbeat()

    def _push_loop(self):
        """Background thread: push metrics periodically."""
        while not self._stop_event.is_set():
            time.sleep(self.push_interval)
            self._push_now()

    def _push_now(self):
        """Push current metrics to Pushgateway."""
        if not self._enabled:
            return

        try:
            from prometheus_client import push_to_gateway

            push_to_gateway(
                self.pushgateway_url,
                job=self.job_name,
                registry=self.registry,
                grouping_key=self.grouping_key,
            )
        except Exception as e:
            logger.debug(f"Failed to push metrics to Pushgateway: {e}")

    def _send_heartbeat(self):
        """Record heartbeat activity in metrics"""
        self.heartbeat_counter.labels(**self._labels).inc()
        logger.debug("Heartbeat recorded - counter incremented")

    def _heartbeat_loop(self):
        """Main heartbeat loop - runs in background thread."""
        while not self._stop_event.is_set():
            # Use wait() instead of sleep() so we can exit quickly on stop
            if self._stop_event.wait(timeout=self.heartbeat_interval):
                break  # Stop event was set

            try:
                # Only send heartbeat if there are active requests
                if self._active_requests > 0:
                    self._send_heartbeat()
                else:
                    # Check if we should still send based on recent activity
                    time_since_activity = (datetime.now() - self._last_activity).total_seconds()
                    if time_since_activity < self.heartbeat_interval:
                        # Recent activity, send heartbeat even if no current requests
                        self._send_heartbeat()
                    else:
                        logger.debug("Skipping heartbeat - no active requests or recent activity")
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")


# Global instance for easy access
_metrics_pusher: Optional[MetricsPusher] = None


def get_metrics_pusher() -> Optional[MetricsPusher]:
    """Get the global MetricsPusher instance."""
    return _metrics_pusher


def init_metrics_pusher(
    pushgateway_url: Optional[str] = None,
    job_name: Optional[str] = None,
    ttl_seconds: Optional[int] = None,
) -> Optional[MetricsPusher]:
    """
    Initialize and start global metrics pusher.

    Automatically constructs Pushgateway URL and job name from environment if not provided.
    """
    global _metrics_pusher

    if _metrics_pusher is not None:
        return _metrics_pusher

    # Get Pushgateway URL from environment
    if pushgateway_url is None:
        pushgateway_host = os.environ.get("PROMETHEUS_PUSHGATEWAY_HOST")
        pushgateway_port = os.environ.get("PROMETHEUS_PUSHGATEWAY_PORT", "9091")
        if not pushgateway_host:
            # Default to cluster-local pushgateway in the kubetorch install namespace
            # (Pushgateway is deployed with the Helm chart, not in the pod's namespace)
            install_namespace = os.environ.get("KT_INSTALL_NAMESPACE", "kubetorch")
            pushgateway_host = f"prometheus-pushgateway.{install_namespace}.svc.cluster.local"
        pushgateway_url = f"http://{pushgateway_host}:{pushgateway_port}"

    # Get job name from environment
    if job_name is None:
        job_name = os.environ.get("KT_SERVICE", "kubetorch-service")

    # Grouping key for Pushgateway
    grouping_key = {
        "pod": os.environ.get("POD_NAME", "unknown"),
        "namespace": os.environ.get("POD_NAMESPACE", "default"),
    }

    _metrics_pusher = MetricsPusher(
        pushgateway_url=pushgateway_url,
        job_name=job_name,
        grouping_key=grouping_key,
        ttl_seconds=ttl_seconds,
    )
    _metrics_pusher.start()
    return _metrics_pusher


def stop_metrics_pusher():
    """Stop the global metrics pusher."""
    global _metrics_pusher
    if _metrics_pusher is not None:
        _metrics_pusher.stop()
        _metrics_pusher = None
