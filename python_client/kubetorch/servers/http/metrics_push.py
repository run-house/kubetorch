"""
Metrics Push for OTEL-free metrics collection.

Pushes metrics directly to Prometheus Pushgateway without OTEL daemonset.
"""

import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class MetricsPusher:
    """Push metrics directly to Prometheus Pushgateway."""

    def __init__(
        self,
        pushgateway_url: str,
        job_name: str,
        push_interval: float = 15.0,
        grouping_key: Optional[dict] = None,
    ):
        """
        Initialize metrics pusher.

        Args:
            pushgateway_url: URL for Prometheus Pushgateway (e.g., http://pushgateway:9091)
            job_name: Job name for Prometheus metrics
            push_interval: Seconds between automatic pushes
            grouping_key: Additional grouping keys for Pushgateway
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

        self.registry = CollectorRegistry()
        self._stop_event = threading.Event()
        self._push_thread: Optional[threading.Thread] = None

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
        self.last_activity = Gauge(
            "kubetorch_last_activity_timestamp",
            "Last activity timestamp",
            registry=self.registry,
        )

        # Active requests gauge
        self.active_requests = Gauge(
            "http_server_active_requests",
            "Number of currently active requests",
            registry=self.registry,
        )

        # Heartbeat counter (for compatibility with existing TTL queries)
        self.heartbeat_counter = Counter(
            "kt_heartbeat_sent",
            "Total heartbeats sent",
            registry=self.registry,
        )

        self._started = False

    def start(self):
        """Start the metrics pusher background thread."""
        if not self._enabled or self._started:
            return

        self._push_thread = threading.Thread(target=self._push_loop, daemon=True)
        self._push_thread.start()
        self._started = True
        logger.debug(f"MetricsPusher started - pushing to {self.pushgateway_url}")

    def stop(self):
        """Stop the metrics pusher and push final metrics."""
        if not self._enabled or not self._started:
            return

        self._stop_event.set()
        self._push_now()
        self._started = False

    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record an HTTP request metric."""
        if not self._enabled:
            return

        self.request_count.labels(method=method, endpoint=endpoint, status=status).inc()
        self.request_latency.labels(method=method, endpoint=endpoint).observe(duration)
        self.last_activity.set(time.time())

    def record_activity(self):
        """Record activity for TTL tracking."""
        if not self._enabled:
            return
        self.last_activity.set(time.time())
        self.heartbeat_counter.inc()

    def request_started(self):
        """Called when a request starts."""
        if not self._enabled:
            return
        self.active_requests.inc()
        self.last_activity.set(time.time())

    def request_finished(self):
        """Called when a request finishes."""
        if not self._enabled:
            return
        self.active_requests.dec()
        self.last_activity.set(time.time())

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


# Global instance for easy access
_metrics_pusher: Optional[MetricsPusher] = None


def get_metrics_pusher() -> Optional[MetricsPusher]:
    """Get the global MetricsPusher instance."""
    return _metrics_pusher


def init_metrics_pusher(
    pushgateway_url: Optional[str] = None,
    job_name: Optional[str] = None,
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
            # Default to cluster-local pushgateway
            namespace = os.environ.get("POD_NAMESPACE", "default")
            pushgateway_host = f"prometheus-pushgateway.{namespace}.svc.cluster.local"
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
    )
    _metrics_pusher.start()
    return _metrics_pusher


def stop_metrics_pusher():
    """Stop the global metrics pusher."""
    global _metrics_pusher
    if _metrics_pusher is not None:
        _metrics_pusher.stop()
        _metrics_pusher = None
