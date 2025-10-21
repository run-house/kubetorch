import asyncio
import logging
import logging.config
import os
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import Response

try:
    from utils import ensure_structured_logging, LOG_CONFIG
except ImportError:
    from .utils import ensure_structured_logging, LOG_CONFIG


HEARTBEAT_INTERVAL_DIVISOR = 5

# Set up our structured JSON logging
logging.config.dictConfig(LOG_CONFIG)
ensure_structured_logging()

logger = logging.getLogger(__name__)
# Set log level based on environment variable
log_level = os.getenv("KT_LOG_LEVEL")
if log_level:
    log_level = log_level.upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))


def get_inactivity_ttl_annotation() -> Optional[int]:
    """
    Get the inactivity TTL from pod annotations.
    Returns TTL in seconds, or None if not found.
    """
    try:
        # Try to get from environment variable first (can be injected via downward API)
        ttl_str = os.getenv("KT_INACTIVITY_TTL")
        if ttl_str:
            return parse_ttl_string(ttl_str)
        return None

    except Exception as e:
        logger.error(f"Error getting pod TTL annotation: {e}")

    return None


def parse_ttl_string(ttl_str: str) -> Optional[int]:
    """Parse TTL string to seconds. Supports formats: 300, 5m, 1h, 1h30m, 1d"""
    ttl_str = ttl_str.strip().lower()

    # If it's just a number, assume seconds
    if ttl_str.isdigit():
        return int(ttl_str)

    # Parse duration strings
    total_seconds = 0
    import re

    # Match patterns like 1h, 30m, 45s
    pattern = r"(\d+)([dhms])"
    matches = re.findall(pattern, ttl_str)

    for value, unit in matches:
        value = int(value)
        if unit == "d":
            total_seconds += value * 24 * 3600
        elif unit == "h":
            total_seconds += value * 3600
        elif unit == "m":
            total_seconds += value * 60
        elif unit == "s":
            total_seconds += value

    return total_seconds if total_seconds > 0 else None


class HeartbeatManager:
    def __init__(self, ttl_seconds: int):
        try:
            from prometheus_client import Counter, Gauge
        except ImportError:
            logger.info(
                "Prometheus client not installed, heartbeat metrics not enabled"
            )
            return None

        self.ttl_seconds = ttl_seconds
        self.heartbeat_interval = self.ttl_seconds // HEARTBEAT_INTERVAL_DIVISOR
        self.active_requests = 0
        self.last_activity = datetime.now()
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.service_name = os.getenv("KT_SERVICE", "unknown-service")
        self.kubetorch_version = os.getenv("KUBETORCH_VERSION", "0.0.0")
        self.service_namespace = os.getenv("POD_NAMESPACE", "default")
        self.service_type = os.getenv("KT_DEPLOYMENT_MODE", "deployment")

        self.heartbeat_counter = Counter(
            "kt_heartbeat_sent",
            "Total heartbeats sent",
            ["service_name", "kubetorch_version", "service_namespace", "service_type"],
        )
        self.active_requests_gauge = Gauge(
            "http_server_active_requests",
            "Number of currently active requests",
            ["service_name", "kubetorch_version", "service_namespace", "service_type"],
        )

        logger.info(
            f"Heartbeat Manager initialized: TTL={self.ttl_seconds}s, Interval={self.heartbeat_interval}s"
        )

    @property
    def labels(self):
        return {
            "service_name": self.service_name,
            "kubetorch_version": self.kubetorch_version,
            "service_namespace": self.service_namespace,
            "service_type": self.service_type,
        }

    async def start(self):
        """Start the heartbeat manager"""
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("Heartbeat started - tracking activity metrics")

    async def stop(self):
        """Stop the heartbeat manager"""
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass

    def request_started(self):
        """Called when a request starts"""
        self.active_requests += 1
        self.active_requests_gauge.labels(**self.labels).inc()
        self.last_activity = datetime.now()

    def request_finished(self):
        """Called when a request finishes"""
        self.active_requests = max(0, self.active_requests - 1)
        self.active_requests_gauge.labels(**self.labels).dec()
        self.last_activity = datetime.now()

    async def _send_heartbeat(self):
        """Record heartbeat activity in metrics"""
        self.heartbeat_counter.labels(**self.labels).inc()
        logger.debug("Heartbeat recorded - counter incremented")

    async def _heartbeat_loop(self):
        """Main heartbeat loop"""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)

                # Only send heartbeat if there are active requests
                if self.active_requests > 0:
                    await self._send_heartbeat()
                else:
                    # Check if we should still send based on recent activity
                    time_since_activity = (
                        datetime.now() - self.last_activity
                    ).total_seconds()
                    if time_since_activity < self.heartbeat_interval:
                        # Recent activity, send heartbeat even if no current requests
                        await self._send_heartbeat()
                    else:
                        logger.debug(
                            "Skipping heartbeat - no active requests or recent activity"
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")


def setup_otel_metrics(app: FastAPI):
    """Setup OpenTelemetry metrics with Prometheus export for FastAPI"""
    try:
        from opentelemetry.exporter.prometheus import PrometheusMetricReader
        from opentelemetry.metrics import set_meter_provider
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.resources import Resource
        from prometheus_client import (
            CollectorRegistry,
            CONTENT_TYPE_LATEST,
            generate_latest,
            Info,
        )
    except ImportError as e:
        logger.info(f"OpenTelemetry metrics not enabled: {e}")
        return app, None

    logger.info("Instrumenting FastAPI app for metrics")

    # Get service info from environment
    service_name = os.getenv(
        "KT_SERVICE_NAME", os.getenv("OTEL_SERVICE_NAME", "unknown-service")
    )
    service_version = os.getenv("KUBETORCH_VERSION", "0.0.0")
    namespace = os.getenv("POD_NAMESPACE", "default")
    service_type = os.getenv("KT_DEPLOYMENT_MODE", "deployment")

    # Create resource with service information
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": service_version,
            "service.namespace": namespace,
            "deployment.environment": namespace,
            "service.type": service_type,
        }
    )

    # Setup Prometheus metric reader
    prometheus_reader = PrometheusMetricReader(
        disable_target_info=False,  # Explicitly enable target_info
    )

    # Create meter provider with the resource and prometheus reader
    meter_provider = MeterProvider(
        resource=resource, metric_readers=[prometheus_reader]
    )

    # Set the global meter provider
    set_meter_provider(meter_provider)

    # Add metrics endpoint
    @app.get("/metrics")
    async def get_metrics(request: Request):
        """Expose Prometheus-formatted OpenTelemetry metrics"""
        registry = CollectorRegistry()

        manager = getattr(request.app.state, "heartbeat_manager", None)
        if manager:
            # Add heartbeat configuration info
            heartbeat_info = Info(
                "heartbeat", "Heartbeat configuration info", registry=registry
            )
            heartbeat_info.info(
                {
                    "ttl_seconds": str(manager.ttl_seconds),
                    "interval_seconds": str(manager.heartbeat_interval),
                }
            )

        # Get all metrics from default registry (includes our heartbeat metrics)
        base_metrics = generate_latest().decode("utf-8")
        additional_metrics = (
            generate_latest(registry).decode("utf-8") if manager else ""
        )

        return Response(
            content=base_metrics + additional_metrics, media_type=CONTENT_TYPE_LATEST
        )

    # Add middleware to track active requests
    @app.middleware("http")
    async def track_requests(request: Request, call_next):
        """Middleware to track active requests for heartbeat"""
        manager = getattr(request.app.state, "heartbeat_manager", None)
        if manager and request.url.path not in ["/metrics", "/health"]:
            manager.request_started()
            try:
                response = await call_next(request)
                return response
            finally:
                manager.request_finished()
        else:
            return await call_next(request)

    logger.info(f"OpenTelemetry metrics enabled for service: {service_name}")
    return app, meter_provider
