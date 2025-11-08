import os
import re
import time
from datetime import datetime, timezone
from typing import List, Optional

from kubernetes import client

import kubetorch as kt
import kubetorch.serving.constants as serving_constants
from kubetorch.logger import get_logger
from kubetorch.resources.compute.utils import (
    check_pod_events_for_errors,
    check_pod_status_for_errors,
    check_revision_for_errors,
    ServiceTimeoutError,
)
from kubetorch.servers.http.utils import load_template
from kubetorch.serving.autoscaling import AutoscalingConfig
from kubetorch.serving.base_service_manager import BaseServiceManager
from kubetorch.serving.utils import nested_override, pod_is_running

logger = get_logger(__name__)


class KnativeServiceManager(BaseServiceManager):
    """Service manager for Knative services with autoscaling capabilities."""

    def _create_or_update_knative_service(
        self,
        name: str,
        module_name: str,
        pod_template: dict,
        autoscaling_config: AutoscalingConfig = None,
        gpu_annotations: dict = None,
        inactivity_ttl: str = None,
        custom_labels: dict = None,
        custom_annotations: dict = None,
        custom_template: dict = None,
        scheduler_name: str = None,
        queue_name: str = None,
        dryrun: bool = False,
    ) -> dict:
        """Creates or updates a Knative service based on the provided configuration.

        Returns:
            Dict
        """
        # Clean the module name to remove any invalid characters for labels
        clean_module_name = re.sub(r"[^A-Za-z0-9.-]|^[-.]|[-.]$", "", module_name)

        labels = {
            **self.base_labels,
            serving_constants.KT_MODULE_LABEL: clean_module_name,
            serving_constants.KT_SERVICE_LABEL: name,
            serving_constants.KT_TEMPLATE_LABEL: "ksvc",
        }

        if custom_labels:
            labels.update(custom_labels)

        # Template labels (exclude template label - that's only for the top-level resource)
        template_labels = {
            **self.base_labels,
            serving_constants.KT_MODULE_LABEL: clean_module_name,
            serving_constants.KT_SERVICE_LABEL: name,
        }

        if custom_labels:
            template_labels.update(custom_labels)

        template_annotations = {
            "networking.knative.dev/ingress.class": "kourier.ingress.networking.knative.dev",
        }

        annotations = {
            "prometheus.io/scrape": "true",
            "prometheus.io/port": "8080",
            "prometheus.io/path": serving_constants.PROMETHEUS_HEALTH_ENDPOINT,
            "serving.knative.dev/container-name": "kubetorch",
            "serving.knative.dev/probe-path": "/health",
        }
        if custom_annotations:
            annotations.update(custom_annotations)

        if scheduler_name and queue_name:
            labels["kai.scheduler/queue"] = queue_name  # Useful for queries, etc
            template_labels["kai.scheduler/queue"] = queue_name  # Required for KAI to schedule pods
            # Note: KAI wraps the Knative revision in a podgroup, expecting at least 1 pod to schedule initially
            # Only set min-scale=1 if user hasn't explicitly provided a min_scale value
            if autoscaling_config.min_scale is None:
                template_annotations["autoscaling.knative.dev/min-scale"] = "1"

        # Add autoscaling annotations (config always provided)
        autoscaling_annotations = autoscaling_config.convert_to_annotations()
        template_annotations.update(autoscaling_annotations)

        # Add progress deadline if specified (not an autoscaling annotation)
        if autoscaling_config.progress_deadline is not None:
            template_annotations["serving.knative.dev/progress-deadline"] = autoscaling_config.progress_deadline

        if inactivity_ttl:
            annotations[serving_constants.INACTIVITY_TTL_ANNOTATION] = inactivity_ttl
            logger.info(f"Configuring auto-down after idle timeout ({inactivity_ttl})")

        template_annotations.update(annotations)

        if gpu_annotations:
            template_annotations.update(gpu_annotations)

        deployment_timestamp = datetime.now(timezone.utc).isoformat()
        template_annotations.update({"kubetorch.com/deployment_timestamp": deployment_timestamp})

        # Set containerConcurrency based on autoscaling config
        # When using concurrency-based autoscaling, set containerConcurrency to match
        # the target to ensure the container's limit aligns with autoscaler expectations
        template_vars = {
            "name": name,
            "namespace": self.namespace,
            "annotations": annotations,
            "template_annotations": template_annotations,
            "labels": labels,
            "template_labels": template_labels,
            "pod_template": pod_template,
        }

        if autoscaling_config.concurrency is not None:
            template_vars["container_concurrency"] = autoscaling_config.concurrency

        service = load_template(
            template_file=serving_constants.KNATIVE_SERVICE_TEMPLATE_FILE,
            template_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"),
            **template_vars,
        )

        if custom_template:
            nested_override(service, custom_template)

        try:
            kwargs = {"dry_run": "All"} if dryrun else {}
            created_service: dict = self.objects_api.create_namespaced_custom_object(
                group="serving.knative.dev",
                version="v1",
                namespace=self.namespace,
                plural="services",
                body=service,
                **kwargs,
            )

            logger.info(
                f"Created Knative service {name} in namespace {self.namespace}",
            )
            return created_service

        except client.exceptions.ApiException as e:
            if e.status == 409:
                logger.info(f"Service {name} already exists, updating")
                existing_service = self.get_knative_service(name)
                return existing_service
            else:
                logger.error(
                    f"Failed to create Knative service: {str(e)}",
                )
                raise e

    def get_knative_service(self, service_name: str) -> dict:
        """Retrieve a Knative service by name."""
        try:
            service = self.objects_api.get_namespaced_custom_object(
                group="serving.knative.dev",
                version="v1",
                namespace=self.namespace,
                plural="services",
                name=service_name,
            )
            return service

        except client.exceptions.ApiException as e:
            logger.error(f"Failed to load Knative service '{service_name}': {str(e)}")
            raise

    def get_deployment_timestamp_annotation(self, service_name: str) -> Optional[str]:
        """Get deployment timestamp annotation for Knative services."""
        try:
            service = self.get_knative_service(service_name)
            if service:
                return (
                    service.get("metadata", {}).get("annotations", {}).get("kubetorch.com/deployment_timestamp", None)
                )
        except client.exceptions.ApiException:
            pass
        return None

    def update_deployment_timestamp_annotation(self, service_name: str, new_timestamp: str) -> str:
        """Update deployment timestamp annotation for Knative services."""
        try:
            patch_body = {"metadata": {"annotations": {"kubetorch.com/deployment_timestamp": new_timestamp}}}
            self.objects_api.patch_namespaced_custom_object(
                group="serving.knative.dev",
                version="v1",
                namespace=self.namespace,
                plural="services",
                name=service_name,
                body=patch_body,
            )
            return new_timestamp
        except client.exceptions.ApiException as e:
            logger.error(f"Failed to update deployment timestamp for Knative service '{service_name}': {str(e)}")
            raise

    def get_knative_service_endpoint(self, service_name: str) -> str:
        """Get the endpoint URL for a Knative service."""
        try:
            service = self.get_knative_service(service_name)

            # Get the URL from the service status
            status = service.get("status", {})
            url = status.get("url")
            if url:
                return url

            # Fallback to constructing URL
            return f"http://{service_name}.{self.namespace}.svc.cluster.local"

        except Exception as e:
            logger.warning(f"Could not get Knative service URL for {service_name}: {e}")
            return f"http://{service_name}.{self.namespace}.svc.cluster.local"

    def create_or_update_service(
        self,
        service_name: str,
        module_name: str,
        pod_template: dict,
        autoscaling_config: AutoscalingConfig = None,
        gpu_annotations: dict = None,
        inactivity_ttl: str = None,
        custom_labels: dict = None,
        custom_annotations: dict = None,
        custom_template: dict = None,
        scheduler_name: str = None,
        queue_name: str = None,
        dryrun: bool = False,
        **kwargs,  # Ignore deployment-specific args like replicas
    ):
        """
        Creates a Knative service with autoscaling capabilities.
        """
        logger.info(f"Deploying Kubetorch autoscaling (Knative) service with name: {service_name}")
        try:
            created_service = self._create_or_update_knative_service(
                name=service_name,
                pod_template=pod_template,
                module_name=module_name,
                autoscaling_config=autoscaling_config,
                gpu_annotations=gpu_annotations,
                inactivity_ttl=inactivity_ttl,
                custom_labels=custom_labels,
                custom_annotations=custom_annotations,
                custom_template=custom_template,
                scheduler_name=scheduler_name,
                queue_name=queue_name,
                dryrun=dryrun,
            )
            return created_service
        except Exception as e:
            logger.error(f"Failed to launch new Knative service: {str(e)}")
            raise e

    def get_endpoint(self, service_name: str) -> str:
        """Get the endpoint URL for a Knative service."""
        return self.get_knative_service_endpoint(service_name)

    def get_pods_for_service(self, service_name: str, **kwargs) -> List[client.V1Pod]:
        """Get all pods associated with this Knative service."""
        return self.get_pods_for_service_static(
            service_name=service_name,
            namespace=self.namespace,
            core_api=self.core_api,
        )

    def _status_condition_ready(self, status: dict) -> bool:
        """Check if service status conditions indicate ready state."""
        conditions = status.get("conditions", [])
        for condition in conditions:
            if condition.get("type") == "Ready":
                return condition.get("status") == "True"
        return False

    def check_service_ready(
        self,
        service_name: str,
        launch_timeout: int,
        objects_api: client.CustomObjectsApi = None,
        core_api: client.CoreV1Api = None,
        queue_name: str = None,
        scheduler_name: str = None,
        **kwargs,
    ) -> bool:
        """Checks if the Knative service is ready to start serving requests.

        Core checks:
        - Service status and conditions
        - Revision status and conditions
        - Pod status and conditions
        - Autoscaling conditions (min-scale, etc.)

        Common failure scenarios handled:
        - Image pull failures or delays
        - Container initialization and setup (pip installs, etc.)
        - User-defined image setup steps
        - Node provisioning delays or failures
        - Service health check failures
        - Container terminations
        - Autoscaling not meeting minimum requirements

        Note:
            This method checks all pods associated with the service, not just the first one.
            Service check will fail fast only for truly unrecoverable conditions (like missing images or autoscaling
            not being triggered or enabled).

            Unless there is a clear reason to terminate, will wait for the full specified timeout
            to allow autoscaling and node provisioning to work (where relevant).

        Args:
            service_name: Name of the Knative service
            launch_timeout: Timeout in seconds to wait for readiness
            objects_api: Objects API instance (uses self.objects_api if None)
            core_api: Core API instance (uses self.core_api if None)
            queue_name: Queue name for scheduling checks
            scheduler_name: Scheduler name for scheduling checks
            **kwargs: Additional arguments

        Returns:
            True if service is ready

        Raises:
            ServiceTimeoutError: If service doesn't become ready within timeout
            QueueUnschedulableError: If pods can't be scheduled due to queue issues
            ResourceNotAvailableError: If required resources aren't available
        """
        if objects_api is None:
            objects_api = self.objects_api
        if core_api is None:
            core_api = self.core_api

        sleep_interval = 2
        start_time = time.time()

        # Instead of spamming logs with each iteration, only log once
        displayed_msgs = {
            "service_status": False,
            "waiting_for_pods": None,
            "revision_status": False,
            "service_readiness": False,
            "autoscaling": False,
        }

        logger.info(f"Checking service {service_name} pod readiness (timeout: {launch_timeout} seconds)")
        iteration = 0
        while (time.time() - start_time) < launch_timeout:
            iteration += 1
            try:
                service = objects_api.get_namespaced_custom_object(
                    group="serving.knative.dev",
                    version="v1",
                    namespace=self.namespace,
                    plural="services",
                    name=service_name,
                )
                status = service.get("status")
                if not status:
                    if not displayed_msgs["service_status"]:
                        logger.info(f"Waiting for service {service_name} status")
                        displayed_msgs["service_status"] = True
                    time.sleep(sleep_interval)
                    continue

                for cond in status.get("conditions", []):
                    if cond.get("type") == "Ready" and cond.get("reason") == "NotOwned":
                        raise kt.KnativeServiceConflictError(
                            f"Knative service '{service_name}' cannot become ready: {cond.get('message')}"
                        )

                # Check autoscaling conditions
                if not displayed_msgs["autoscaling"]:
                    logger.info("Checking autoscaling conditions")
                    displayed_msgs["autoscaling"] = True

                # Get the min-scale from annotations
                min_scale = 0
                if service.get("spec", {}).get("template", {}).get("metadata", {}).get("annotations", {}):
                    min_scale_str = service["spec"]["template"]["metadata"]["annotations"].get(
                        "autoscaling.knative.dev/min-scale", "0"
                    )
                    min_scale = int(min_scale_str)

                if min_scale == 0 and self._status_condition_ready(status):
                    # Service is ready and allowed to scale to zero
                    logger.info(f"Service {service_name} is already marked as ready")
                    return True

                if min_scale == 0:
                    # Always need at least one pod
                    min_scale = 1

                # Get current number of Running pods
                pods = self.get_pods_for_service(service_name)
                running_pods = [p for p in pods if pod_is_running(p)]
                running_pods_count = len(running_pods)

                if running_pods_count < min_scale:
                    for pod in pods:
                        # Check for image pull errors in container status
                        check_pod_status_for_errors(pod, queue_name, scheduler_name)

                        # Check pod events separately from the core API
                        check_pod_events_for_errors(pod, self.namespace, core_api)

                    if (
                        displayed_msgs["waiting_for_pods"] is None
                        or displayed_msgs["waiting_for_pods"] != running_pods_count
                    ):
                        logger.info(
                            f"Waiting for minimum scale ({min_scale} pods), currently have {running_pods_count}"
                        )
                        displayed_msgs["waiting_for_pods"] = running_pods_count
                else:
                    if not displayed_msgs["service_readiness"]:
                        logger.info(
                            f"Min {min_scale} pod{'s are' if min_scale > 1 else ' is'} ready, waiting for service to be marked as ready"
                        )
                        displayed_msgs["service_readiness"] = True

                    if self._status_condition_ready(status):
                        logger.info(f"Service {service_name} is now ready")
                        return True

                if not displayed_msgs["revision_status"]:
                    logger.info("Checking service revision status")
                    displayed_msgs["revision_status"] = True

                latest_revision = status.get("latestCreatedRevisionName")
                if latest_revision:
                    check_revision_for_errors(latest_revision, self.namespace, objects_api)

            except client.exceptions.ApiException:
                raise

            if iteration % 10 == 0:
                elapsed = int(time.time() - start_time)
                remaining = max(0, int(launch_timeout - elapsed))
                logger.info(f"Service is not yet marked as ready " f"(elapsed: {elapsed}s, remaining: {remaining}s)")

            time.sleep(sleep_interval)

        raise ServiceTimeoutError(
            f"Service {service_name} did not become ready within {launch_timeout} seconds. "
            "To update the timeout, set the `launch_timeout` parameter in the Compute class, or set the "
            "environment variable `KT_LAUNCH_TIMEOUT`."
        )

    def teardown_service(self, service_name: str, console=None) -> bool:
        """Teardown Knative service and associated resources.

        Args:
            service_name: Name of the Knative service to teardown
            console: Optional Rich console for output

        Returns:
            True if teardown was successful, False otherwise
        """
        from kubetorch.resources.compute.utils import delete_service

        try:
            # Delete the Knative service
            delete_service(
                custom_api=self.objects_api,
                name=service_name,
                namespace=self.namespace,
                console=console,
            )

            return True

        except Exception as e:
            logger.error(f"Failed to teardown Knative service {service_name}: {e}")
            return False
