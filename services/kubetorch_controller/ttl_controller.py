"""
TTL Controller - Inactivity-based resource cleanup

This module monitors Kubetorch services (Knative, Deployments, RayClusters) and
automatically deletes them after a configurable inactivity period.

Architecture:
- Runs as a background task in the kubetorch_controller
- Queries Prometheus for activity metrics (heartbeats, HTTP requests)
- Updates last-active annotations on services
- Deletes services that exceed their inactivity TTL

Configuration (via environment variables):
- PROMETHEUS_URL: URL of Prometheus server for metrics queries
- WATCH_NAMESPACES: Comma-separated list of namespaces to monitor

Usage:
    The TTL controller is started automatically via the background_tasks lifespan
    when TTL_CONTROLLER_ENABLED=true.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from fastapi.concurrency import run_in_threadpool
from kubernetes import client, config

logger = logging.getLogger(__name__)

# Configuration from environment
PROMETHEUS_URL = os.environ.get(
    "PROMETHEUS_URL", "http://kubetorch-metrics.kubetorch.svc.cluster.local:9090"
)
WATCH_NAMESPACES = os.environ.get("WATCH_NAMESPACES", "default")

# Annotations (constants)
INACTIVITY_ANNOTATION = "kubetorch.com/inactivity-ttl"
LAST_ACTIVE_ANNOTATION = "kubetorch.com/last-active-timestamp"
READY_SINCE_ANNOTATION = "kubetorch.com/ready-since"

# Labels (constants)
KT_MODULE_LABEL = "kubetorch.com/module"
KT_TEMPLATE_LABEL = "kubetorch.com/template"


class TTLController:
    """Controller that deletes Knative services and Deployments after specified inactivity period."""

    def __init__(self):
        """Initialize the controller with isolated K8s client."""
        self._api_client = None
        self.core_api = None
        self.custom_api = None
        self.apps_api = None
        self.prom = None
        self._initialized = False

    def _initialize(self):
        """Lazy initialization of K8s and Prometheus clients."""
        if self._initialized:
            return

        from prometheus_api_client import PrometheusConnect

        # Create isolated K8s client to avoid connection pool conflicts
        (
            self._api_client,
            self.core_api,
            self.apps_api,
            self.custom_api,
        ) = self._create_isolated_k8s_client()

        # Initialize Prometheus client
        self.prom = PrometheusConnect(url=PROMETHEUS_URL, disable_ssl=True)
        self._initialized = True
        logger.info(f"TTL controller initialized with Prometheus at {PROMETHEUS_URL}")

    def _create_isolated_k8s_client(self):
        """Create an isolated K8s client for this controller.

        Uses a separate Configuration and ApiClient to avoid connection pool
        conflicts with the main request handlers.

        Returns:
            Tuple of (ApiClient, CoreV1Api, AppsV1Api, CustomObjectsApi)
        """
        try:
            controller_config = client.Configuration()
            try:
                config.load_incluster_config(client_configuration=controller_config)
            except config.ConfigException:
                config.load_kube_config(client_configuration=controller_config)

            api_client = client.ApiClient(configuration=controller_config)
            return (
                api_client,
                client.CoreV1Api(api_client=api_client),
                client.AppsV1Api(api_client=api_client),
                client.CustomObjectsApi(api_client=api_client),
            )
        except Exception as e:
            logger.error(
                f"Failed to create isolated K8s client for TTL controller: {e}"
            )
            raise

    def close(self):
        """Close the K8s API client to release connection pools."""
        if self._api_client:
            try:
                self._api_client.close()
            except Exception:
                pass

    async def run(self, interval_seconds: int = 300):
        """Main loop: run reconciliation every interval.

        Args:
            interval_seconds: Time between reconciliation cycles (default: 300 = 5 minutes)
        """
        logger.info(f"TTL controller starting with {interval_seconds}s interval")

        while True:
            try:
                await self.reconcile()
            except asyncio.CancelledError:
                logger.info("TTL controller cancelled")
                break
            except Exception as e:
                logger.error(f"TTL reconciliation error: {e}")

            await asyncio.sleep(interval_seconds)

    async def reconcile(self):
        """Main reconciliation logic - runs in thread pool for blocking calls."""
        logger.info("Starting TTL reconciliation cycle")

        # Initialize on first run (lazy init)
        if not self._initialized:
            await run_in_threadpool(self._initialize)

        # Check Prometheus connection
        try:
            await run_in_threadpool(self.prom.check_prometheus_connection)
            logger.debug(f"Connected to Prometheus at {PROMETHEUS_URL}")
        except Exception as e:
            logger.error(f"Failed to connect to Prometheus at {PROMETHEUS_URL}: {e}")
            return

        # Get all services with TTL annotation
        services: List[Tuple[Dict, str]] = await run_in_threadpool(
            self._fetch_all_services
        )

        if not services:
            logger.info("No services with TTL annotations found")
            return

        logger.info(f"Found {len(services)} services with TTL annotations")

        # Update all active timestamps
        last_active_mapping = await self._update_all_active_timestamps(services)

        # Check each service for inactivity and delete if needed
        for service, service_type in services:
            service_name = self._get_service_name(service, service_type) or "unknown"

            should_delete = await run_in_threadpool(
                self._should_delete_service, service, service_type, last_active_mapping
            )

            if should_delete:
                logger.info(f"Service {service_name} exceeded inactivity TTL, deleting")
                await self._delete_service(service, service_type)
            else:
                logger.debug(f"Service {service_name} still within inactivity TTL")

        logger.info("Completed TTL reconciliation cycle")

    # =========================================================================
    # Service metadata helpers
    # =========================================================================

    def _get_service_name(self, service, service_type: str) -> Optional[str]:
        """Extract service name from either service type."""
        if service_type in ("knative", "raycluster"):
            return service.get("metadata", {}).get("name")
        elif service_type == "deployment":
            return service.metadata.name if service.metadata else None
        return None

    def _get_service_namespace(self, service, service_type: str) -> Optional[str]:
        """Extract service namespace from either service type."""
        if service_type in ("knative", "raycluster"):
            return service.get("metadata", {}).get("namespace")
        elif service_type == "deployment":
            return service.metadata.namespace if service.metadata else None
        return None

    def _get_service_annotations(self, service, service_type: str) -> Dict:
        """Extract service annotations from either service type."""
        if service_type in ("knative", "raycluster"):
            return service.get("metadata", {}).get("annotations", {})
        elif service_type == "deployment":
            return service.metadata.annotations or {} if service.metadata else {}
        return {}

    def _get_service_creation_timestamp(
        self, service, service_type: str
    ) -> Optional[str]:
        """Extract service creation timestamp from either service type."""
        if service_type in ("knative", "raycluster"):
            return service.get("metadata", {}).get("creationTimestamp")
        elif service_type == "deployment":
            return (
                service.metadata.creation_timestamp.isoformat()
                if service.metadata and service.metadata.creation_timestamp
                else None
            )
        return None

    def _get_service_status(self, service, service_type: str) -> Dict:
        """Extract service status from either service type."""
        if service_type in ("knative", "raycluster"):
            return service.get("status", {})
        elif service_type == "deployment":
            return (
                service.status.__dict__
                if hasattr(service, "status") and service.status
                else {}
            )
        return {}

    # =========================================================================
    # Service fetching
    # =========================================================================

    def _fetch_all_services(self) -> List[Tuple[Dict, str]]:
        """Get all Knative services and Deployments with inactivity TTL annotation."""
        if not WATCH_NAMESPACES:
            logger.error("WATCH_NAMESPACES is not set, skipping reconciliation")
            return []

        namespaces = WATCH_NAMESPACES.split(",")
        logger.info(f"Fetching services in namespaces: {namespaces}")
        ttl_services = []

        for namespace in namespaces:
            namespace = namespace.strip()
            if not namespace:
                continue

            # Fetch Knative services
            try:
                services = self.custom_api.list_namespaced_custom_object(
                    group="serving.knative.dev",
                    version="v1",
                    namespace=namespace,
                    plural="services",
                    label_selector=KT_MODULE_LABEL,
                )

                knative_services = [
                    (service, "knative")
                    for service in services.get("items", [])
                    if INACTIVITY_ANNOTATION
                    in service.get("metadata", {}).get("annotations", {})
                ]
                logger.info(
                    f"Found {len(knative_services)} Knative services with TTL in {namespace}"
                )
                ttl_services.extend(knative_services)

            except Exception as e:
                logger.error(f"Error fetching Knative services in {namespace}: {e}")

            # Fetch Deployments
            try:
                deployments = self.apps_api.list_namespaced_deployment(
                    namespace=namespace,
                    label_selector=f"{KT_MODULE_LABEL},{KT_TEMPLATE_LABEL}=deployment",
                )

                deployment_services = [
                    (deployment, "deployment")
                    for deployment in deployments.items
                    if deployment.metadata.annotations
                    and INACTIVITY_ANNOTATION in deployment.metadata.annotations
                ]
                logger.info(
                    f"Found {len(deployment_services)} Deployments with TTL in {namespace}"
                )
                ttl_services.extend(deployment_services)

            except Exception as e:
                logger.error(f"Error fetching Deployments in {namespace}: {e}")

            # Fetch RayClusters
            try:
                rayclusters = self.custom_api.list_namespaced_custom_object(
                    group="ray.io",
                    version="v1",
                    namespace=namespace,
                    plural="rayclusters",
                    label_selector=f"{KT_TEMPLATE_LABEL}=raycluster",
                )
                raycluster_services = [
                    (raycluster, "raycluster")
                    for raycluster in rayclusters.get("items", [])
                    if INACTIVITY_ANNOTATION
                    in raycluster.get("metadata", {}).get("annotations", {})
                ]
                logger.info(
                    f"Found {len(raycluster_services)} RayClusters with TTL in {namespace}"
                )
                ttl_services.extend(raycluster_services)
            except Exception as e:
                logger.error(f"Error fetching RayClusters in {namespace}: {e}")

        return ttl_services

    # =========================================================================
    # Activity tracking
    # =========================================================================

    def _get_service_last_active(
        self, service_name: str, service_namespace: str, service_type: str
    ) -> Optional[datetime]:
        """Determine when a service was last active based on annotations."""
        try:
            if service_type == "knative":
                service = self.custom_api.get_namespaced_custom_object(
                    group="serving.knative.dev",
                    version="v1",
                    namespace=service_namespace,
                    plural="services",
                    name=service_name,
                )
                annotations = service.get("metadata", {}).get("annotations", {})
            elif service_type == "deployment":
                service = self.apps_api.read_namespaced_deployment(
                    name=service_name,
                    namespace=service_namespace,
                )
                annotations = (
                    service.metadata.annotations or {} if service.metadata else {}
                )
            elif service_type == "raycluster":
                service = self.custom_api.get_namespaced_custom_object(
                    group="ray.io",
                    version="v1",
                    namespace=service_namespace,
                    plural="rayclusters",
                    name=service_name,
                )
                annotations = service.get("metadata", {}).get("annotations", {})
            else:
                return None

            if LAST_ACTIVE_ANNOTATION in annotations:
                timestamp_str = annotations[LAST_ACTIVE_ANNOTATION]
                return datetime.fromisoformat(timestamp_str)

        except Exception as e:
            logger.error(f"Error getting service {service_name}: {e}")

        return None

    def _update_last_active_annotation(
        self,
        service_name: str,
        namespace: str,
        timestamp: datetime,
        service_type: str,
        is_first_ready: bool,
    ):
        """Update the last-active annotation on a service."""
        try:
            # Make timezone-naive timestamps UTC-aware
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)

            timestamp_str = timestamp.isoformat()
            patch_body = {
                "metadata": {
                    "annotations": {
                        LAST_ACTIVE_ANNOTATION: timestamp_str,
                        READY_SINCE_ANNOTATION: timestamp_str
                        if is_first_ready
                        else None,
                    }
                }
            }

            if service_type == "knative":
                self.custom_api.patch_namespaced_custom_object(
                    group="serving.knative.dev",
                    version="v1",
                    namespace=namespace,
                    plural="services",
                    name=service_name,
                    body=patch_body,
                )
            elif service_type == "deployment":
                self.apps_api.patch_namespaced_deployment(
                    name=service_name,
                    namespace=namespace,
                    body=patch_body,
                )
            elif service_type == "raycluster":
                self.custom_api.patch_namespaced_custom_object(
                    group="ray.io",
                    version="v1",
                    namespace=namespace,
                    plural="rayclusters",
                    name=service_name,
                    body=patch_body,
                )
            else:
                logger.error(f"Invalid service type: {service_type}")
                return

            logger.info(
                f"Updated last-active timestamp for {service_name} ({service_type}) to {timestamp_str}"
            )

        except Exception as e:
            logger.error(
                f"Error updating annotation for {service_name} ({service_type}): {e}"
            )

    def _fetch_service_timestamp_from_metrics(
        self, service_name: str, namespace: str, inactivity_ttl: str
    ) -> Optional[datetime]:
        """Get the last active timestamp from metrics."""
        # REQUEST_COUNT = "http_server_duration_milliseconds_count"
        HEARTBEAT_COUNT = "kt_heartbeat_sent_total"
        SERVICE_NAME_LABEL = "service"  # Relabelled from __meta_kubernetes_pod_label_kubetorch_com_service
        NAMESPACE_LABEL = "namespace"  # Relabelled from __meta_kubernetes_namespace

        timestamp = None

        # Get the last heartbeat count change timestamp within the inactivity TTL
        latest_heartbeat_query = f'max(max_over_time((clamp_max((changes({{"{HEARTBEAT_COUNT}", {SERVICE_NAME_LABEL}="{service_name}", {NAMESPACE_LABEL}="{namespace}"}}[1m]) > 0), 1) * time())[{inactivity_ttl}:]))'
        result = self.prom.custom_query(query=latest_heartbeat_query)
        latest_heartbeat = (
            float(result[0]["value"][1]) if result and result[0]["value"][1] else None
        )
        logger.debug(f"Latest heartbeat: {latest_heartbeat}")
        if latest_heartbeat:
            timestamp = latest_heartbeat

        # # Get the timestamp of the last request count change within the inactivity TTL
        # latest_request_query = f'max(max_over_time((clamp_max((changes({{"{REQUEST_COUNT}", pod=~"{service_name}.*", namespace="{namespace}"}}[1m]) > 0), 1) * time())[{inactivity_ttl}:]))'
        # result = self.prom.custom_query(query=latest_request_query)
        # latest_change = (
        #     float(result[0]["value"][1]) if result and result[0]["value"][1] else None
        # )
        # logger.debug(f"Latest request: {latest_change}")
        # if latest_change and (not timestamp or latest_change > timestamp):
        #     timestamp = latest_change

        if timestamp:
            return datetime.fromtimestamp(timestamp, timezone.utc)
        return None

    async def _update_all_active_timestamps(self, services: List[Tuple[Dict, str]]):
        """Updates last-active timestamps for all services that have had traffic."""
        last_active_mapping = {}
        for service, service_type in services:
            service_name = self._get_service_name(service, service_type)
            namespace = self._get_service_namespace(service, service_type)
            if not service_name or not namespace:
                logger.error(
                    f"Skipping service {service} because it has no name or namespace"
                )
                continue

            annotations = self._get_service_annotations(service, service_type)
            last_active = annotations.get(LAST_ACTIVE_ANNOTATION)
            inactivity_ttl = annotations.get(INACTIVITY_ANNOTATION)
            ready_since = annotations.get(READY_SINCE_ANNOTATION)

            timestamp = None
            is_service_pending = False
            if last_active:
                last_active_mapping[service_name] = datetime.fromisoformat(last_active)

            try:
                # Check if service is in pending state (still launching)
                if not ready_since:
                    is_service_pending = await run_in_threadpool(
                        self._is_service_pending, service, service_type
                    )
                    if is_service_pending:
                        logger.info(
                            f"Service {service_name} is still pending/launching, skipping activity check"
                        )
                        timestamp = datetime.now(timezone.utc)

                if not timestamp:
                    timestamp = await run_in_threadpool(
                        self._fetch_service_timestamp_from_metrics,
                        service_name,
                        namespace,
                        inactivity_ttl,
                    )

                if timestamp:
                    if not last_active or timestamp > datetime.fromisoformat(
                        last_active
                    ):
                        await run_in_threadpool(
                            self._update_last_active_annotation,
                            service_name,
                            namespace,
                            timestamp,
                            service_type,
                            is_first_ready=(not ready_since and not is_service_pending),
                        )
                        last_active_mapping[service_name] = timestamp
                    else:
                        logger.debug(
                            f"Skipping update for {service_name} - timestamp not newer"
                        )

            except Exception as e:
                logger.error(f"Error updating timestamp for {service_name}: {e}")

        return last_active_mapping

    # =========================================================================
    # Deletion logic
    # =========================================================================

    async def _delete_service(self, service: dict, service_type: str):
        """Delete a service using the shared teardown function."""
        from helpers.delete_helpers import teardown_services_by_name

        service_name = self._get_service_name(service, service_type)
        namespace = self._get_service_namespace(service, service_type)

        if not service_name or not namespace:
            logger.error(
                f"Skipping service {service} because it has no name or namespace"
            )
            return

        logger.info(f"Tearing down {service_type} {service_name} in {namespace}")

        # Use the shared teardown function for complete cleanup
        result = await teardown_services_by_name(
            namespace=namespace,
            service_names=[service_name],
            force=True,  # TTL deletions should be immediate
        )

        if not result.success:
            for error in result.errors:
                logger.error(f"Teardown error for {service_name}: {error}")
        else:
            logger.info(f"Successfully tore down {service_type} {service_name}")

    def _should_delete_service(
        self, service: Dict, service_type: str, last_active_mapping: Dict[str, datetime]
    ) -> bool:
        """Determine if a service should be deleted based on inactivity period."""
        service_name = self._get_service_name(service, service_type)
        namespace = self._get_service_namespace(service, service_type)
        annotations = self._get_service_annotations(service, service_type)

        if not service_name or not namespace:
            logger.error(
                f"Skipping service {service} because it has no name or namespace"
            )
            return False

        if INACTIVITY_ANNOTATION not in annotations:
            return False

        try:
            # Parse TTL duration (e.g., "24h", "7d")
            ttl_str = annotations[INACTIVITY_ANNOTATION]
            ttl_seconds = self._parse_duration(ttl_str)

            # Get last active time
            last_active = last_active_mapping.get(
                service_name
            ) or self._get_service_last_active(service_name, namespace, service_type)
            if last_active is None:
                # Service never received traffic, use creation time
                creation_time_str = self._get_service_creation_timestamp(
                    service, service_type
                )
                if creation_time_str:
                    if creation_time_str.endswith("Z"):
                        creation_time_str = creation_time_str.replace("Z", "+00:00")
                    last_active = datetime.fromisoformat(creation_time_str)
                else:
                    return False

            # Check if inactive period exceeds TTL
            last_active = last_active.astimezone(timezone.utc)
            now = datetime.now(timezone.utc)
            inactive_seconds = (now - last_active).total_seconds()
            logger.info(
                f"Service {service_name} inactive for {inactive_seconds:.0f}s (TTL: {ttl_seconds}s)"
            )

            return inactive_seconds > ttl_seconds

        except Exception as e:
            logger.error(f"Error determining deletion status for {service_name}: {e}")
            return False

    # =========================================================================
    # Pending state detection
    # =========================================================================

    def _is_service_pending(self, service: Dict, service_type: str) -> bool:
        """Check if a service is in pending state (still launching)."""
        service_name = self._get_service_name(service, service_type)
        namespace = self._get_service_namespace(service, service_type)

        if not service_name or not namespace:
            return False

        try:
            if service_type == "knative":
                status = self._get_service_status(service, service_type)
                conditions = status.get("conditions", [])
                for condition in conditions:
                    if condition.get("type") == "Ready":
                        cond_status = condition.get("status")
                        if cond_status == "Unknown":
                            last_transition = condition.get("lastTransitionTime")
                            if last_transition:
                                try:
                                    transition_time = datetime.fromisoformat(
                                        last_transition.replace("Z", "+00:00")
                                    )
                                    now = datetime.now(timezone.utc)
                                    time_in_unknown = (
                                        now - transition_time
                                    ).total_seconds()

                                    # If Unknown for > 10 minutes, likely stuck
                                    if time_in_unknown > 600:
                                        logger.info(
                                            f"Service {service_name} Unknown for {time_in_unknown:.0f}s - likely stuck"
                                        )
                                        return False
                                    logger.info(
                                        f"Service {service_name} Unknown for {time_in_unknown:.0f}s (launching)"
                                    )
                                    return True
                                except Exception:
                                    return False
                            else:
                                return False
                        elif cond_status == "False":
                            reason = condition.get("reason", "")
                            if any(
                                keyword in reason.lower()
                                for keyword in ["pending", "creating", "launching"]
                            ):
                                logger.info(
                                    f"Service {service_name} is pending: {reason}"
                                )
                                return True

            elif service_type == "deployment":
                if hasattr(service, "status") and service.status:
                    if service.status.conditions:
                        for condition in service.status.conditions:
                            if condition.type == "Progressing":
                                if (
                                    condition.status == "True"
                                    and condition.reason == "NewReplicaSetAvailable"
                                ):
                                    break
                                elif (
                                    condition.status == "True"
                                    and condition.reason
                                    in [
                                        "ReplicaSetUpdated",
                                        "NewReplicaSetCreated",
                                    ]
                                ):
                                    logger.info(
                                        f"Deployment {service_name} progressing: {condition.reason}"
                                    )
                                    return True

                    if service.status.replicas != service.status.ready_replicas or (
                        service.status.unavailable_replicas is not None
                        and service.status.unavailable_replicas > 0
                    ):
                        logger.info(
                            f"Deployment {service_name} has unavailable replicas"
                        )
                        return True

            elif service_type == "raycluster":
                status = self._get_service_status(service, service_type)
                state = status.get("state", "").lower()
                if state == "ready":
                    return False
                else:
                    return True

            # Check if pods are in pending state
            try:
                pods_list = None

                if service_type == "knative":
                    pods_list = self.core_api.list_namespaced_pod(
                        namespace=namespace,
                        label_selector=f"serving.knative.dev/service={service_name}",
                    )
                elif service_type == "deployment":
                    if (
                        hasattr(service, "spec")
                        and service.spec
                        and service.spec.selector
                    ):
                        labels = service.spec.selector.match_labels or {}
                        label_selector = ",".join(
                            [f"{k}={v}" for k, v in labels.items()]
                        )
                        pods_list = self.core_api.list_namespaced_pod(
                            namespace=namespace,
                            label_selector=label_selector,
                        )

                if pods_list:
                    for pod in pods_list.items:
                        if pod.status.phase == "Pending":
                            logger.info(
                                f"Service {service_name} has pending pod: {pod.metadata.name}"
                            )
                            return True

                        for container_status in pod.status.container_statuses or []:
                            if (
                                container_status.state
                                and container_status.state.waiting
                            ):
                                reason = container_status.state.waiting.reason
                                if reason in [
                                    "ContainerCreating",
                                    "PodInitializing",
                                    "Pending",
                                ]:
                                    logger.debug(
                                        f"Service {service_name} container pending: {reason}"
                                    )
                                    return True

            except Exception as e:
                logger.debug(f"Error checking pods for service {service_name}: {e}")

            return False

        except Exception as e:
            logger.debug(f"Error checking pending status for {service_name}: {e}")
            return False

    # =========================================================================
    # Utilities
    # =========================================================================

    def _parse_duration(self, duration_str: str) -> int:
        """Parse duration string like '24h' or '7d' to seconds."""
        unit = duration_str[-1].lower()
        value = int(duration_str[:-1])

        if unit == "s":
            return value
        elif unit == "m":
            return value * 60
        elif unit == "h":
            return value * 3600
        elif unit == "d":
            return value * 86400
        else:
            raise ValueError(f"Unknown time unit: {unit}")
