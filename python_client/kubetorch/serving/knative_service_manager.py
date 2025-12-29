import os
import time
from typing import List

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
from kubetorch.serving.utils import pod_is_running
from kubetorch.utils import http_conflict, http_not_found

logger = get_logger(__name__)


class KnativeServiceManager(BaseServiceManager):
    """Service manager for Knative services with autoscaling capabilities."""

    def __init__(
        self,
        namespace: str,
        template_label: str = "ksvc",
        api_group: str = "serving.knative.dev",
        api_plural: str = "services",
        api_version: str = "v1",
        service_annotations: dict = None,
    ):
        # Set Knative-specific default annotations
        default_service_annotations = {
            "serving.knative.dev/container-name": "kubetorch",
            "serving.knative.dev/probe-path": "/health",
        }
        if service_annotations:
            default_service_annotations.update(service_annotations)

        super().__init__(
            namespace=namespace,
            template_label=template_label,
            api_group=api_group,
            api_plural=api_plural,
            api_version=api_version,
            service_annotations=default_service_annotations,
        )

        # Knative-specific template annotations
        self.template_annotations = {
            "networking.knative.dev/ingress.class": "kourier.ingress.networking.knative.dev",
        }

    def get_pod_template_path(self) -> List[str]:
        """Get the path to the pod template."""
        return ["spec", "template"]

    def get_replicas(self, manifest: dict) -> int:
        """Get the number of replicas. Returns the min-scale annotation value, or 1 if not set."""
        template_annotations = manifest.get("spec", {}).get("template", {}).get("metadata", {}).get("annotations", {})
        min_scale = template_annotations.get("autoscaling.knative.dev/min-scale")
        if min_scale:
            return int(min_scale)
        return 1

    def set_replicas(self, manifest: dict, value: int) -> None:
        """Set the number of replicas. Sets the min-scale annotation to the specified value."""
        template_metadata = manifest.get("spec", {}).setdefault("template", {}).setdefault("metadata", {})
        template_annotations = template_metadata.setdefault("annotations", {})
        template_annotations["autoscaling.knative.dev/min-scale"] = str(value)

    @classmethod
    def _convert_manifest(
        cls,
        deployment_manifest: dict,
        namespace: str,
        autoscaling_config: AutoscalingConfig = None,
        gpu_annotations: dict = None,
    ) -> dict:
        """Convert a deployment manifest to a Knative service manifest."""
        pod_spec = deployment_manifest["spec"]["template"]["spec"]
        deployment_labels = deployment_manifest["metadata"]["labels"]
        deployment_annotations = deployment_manifest["metadata"]["annotations"]

        labels = deployment_labels.copy()
        labels[serving_constants.KT_TEMPLATE_LABEL] = "ksvc"

        # Template labels (exclude template label - that's only for the top-level resource)
        template_labels = labels.copy()
        template_labels.pop(serving_constants.KT_TEMPLATE_LABEL, None)

        template_annotations = {
            "networking.knative.dev/ingress.class": "kourier.ingress.networking.knative.dev",
        }

        # Get base annotations (Knative-specific ones already in deployment_annotations)
        annotations = deployment_annotations.copy()
        default_knative_annotations = {
            "serving.knative.dev/container-name": "kubetorch",
            "serving.knative.dev/probe-path": "/health",
        }
        annotations.update(default_knative_annotations)

        # Note: KAI wraps the Knative revision in a podgroup, expecting at least 1 pod to schedule initially
        # Only set min-scale=1 if user hasn't explicitly provided a min_scale value
        if autoscaling_config:
            if autoscaling_config.min_scale is None:
                template_annotations["autoscaling.knative.dev/min-scale"] = "1"

            autoscaling_annotations = autoscaling_config.convert_to_annotations()
            template_annotations.update(autoscaling_annotations)

            if autoscaling_config.progress_deadline is not None:
                template_annotations["serving.knative.dev/progress-deadline"] = autoscaling_config.progress_deadline

        template_annotations.update(annotations)

        if gpu_annotations:
            template_annotations.update(gpu_annotations)

        # Set containerConcurrency based on autoscaling config
        # When using concurrency-based autoscaling, set containerConcurrency to match
        # the target to ensure the container's limit aligns with autoscaler expectations
        template_vars = {
            "name": "",  # Will be set during launch
            "namespace": namespace,
            "annotations": annotations,
            "template_annotations": template_annotations,
            "labels": labels,
            "template_labels": template_labels,
            "pod_spec": pod_spec,
        }

        if autoscaling_config and autoscaling_config.concurrency is not None:
            template_vars["container_concurrency"] = autoscaling_config.concurrency

        service = load_template(
            template_file=serving_constants.KNATIVE_SERVICE_TEMPLATE_FILE,
            template_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"),
            **template_vars,
        )

        return service

    def _create_or_update_resource(self, manifest: dict, service_name: str, clean_module_name: str, **kwargs) -> dict:
        try:
            created_service: dict = self.controller_client.create_namespaced_custom_object(
                group="serving.knative.dev",
                version="v1",
                namespace=self.namespace,
                plural="services",
                body=manifest,
            )

            if not kwargs.get("dry_run"):
                logger.info(f"Created Knative service {manifest['metadata']['name']} in namespace {self.namespace}")
            return created_service

        except Exception as e:
            if http_conflict(e):
                logger.info(f"Service {manifest['metadata']['name']} already exists, updating")
                existing_service = self.get_resource(manifest["metadata"]["name"])
                return existing_service

            logger.error(f"Failed to create Knative service: {e}")
            raise

    def get_resource(self, service_name: str) -> dict:
        """Retrieve a Knative service by name."""
        try:
            service = self.controller_client.get_namespaced_custom_object(
                group="serving.knative.dev",
                version="v1",
                namespace=self.namespace,
                plural="services",
                name=service_name,
            )
            return service

        except Exception as e:
            if http_not_found(e):
                return {}

            logger.error(f"Failed to load Knative service '{service_name}': {e}")
            raise

    def update_deployment_timestamp_annotation(self, service_name: str, new_timestamp: str) -> str:
        """Update deployment timestamp annotation for Knative services."""
        try:
            patch_body = self._create_timestamp_patch_body(new_timestamp)
            self.controller_client.patch_namespaced_custom_object(
                group="serving.knative.dev",
                version="v1",
                namespace=self.namespace,
                plural="services",
                name=service_name,
                body=patch_body,
            )
            return new_timestamp
        except Exception as e:
            logger.error(f"Failed to update deployment timestamp for Knative service '{service_name}': {str(e)}")
            raise

    def get_knative_service_endpoint(self, service_name: str) -> str:
        """Get the endpoint URL for a Knative service."""
        try:
            service = self.get_resource(service_name)

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

    def get_endpoint(self, service_name: str) -> str:
        """Get the endpoint URL for a Knative service."""
        return self.get_knative_service_endpoint(service_name)

    def get_pods_for_service(self, service_name: str, **kwargs):
        """Get all pods associated with this Knative service."""
        try:
            # First try to get the service to find the latest revision
            service = self.get_resource(service_name)
            status = service.get("status", {})

            # Prefer latestReadyRevisionName, but fall back to latestCreatedRevisionName
            # This is important for detecting image pull errors - when image pull fails,
            # there's no "ready" revision, only a "created" one with failing pods
            revision_name = status.get("latestReadyRevisionName") or status.get("latestCreatedRevisionName")

            if revision_name:
                # Look for pods with the revision label
                label_selector = f"serving.knative.dev/revision={revision_name}"
                return super().get_pods_for_service(service_name, label_selector=label_selector, **kwargs)

        except Exception as e:
            logger.warning(f"Knative pod lookup failed for {service_name}: {e}")

        # Fallback: use normal KT service label lookup
        return super().get_pods_for_service(service_name, **kwargs)

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
            **kwargs: Additional arguments

        Returns:
            True if service is ready

        Raises:
            ServiceTimeoutError: If service doesn't become ready within timeout
            ResourceNotAvailableError: If required resources aren't available
        """
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
                service = self.controller_client.get_namespaced_custom_object(
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
                        check_pod_status_for_errors(pod)

                        # Check pod events separately from the core API
                        check_pod_events_for_errors(pod, self.namespace)

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
                    check_revision_for_errors(latest_revision, self.namespace)

            except Exception as e:
                if not http_not_found(e):
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
