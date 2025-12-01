import os
import time
from typing import List

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
from kubetorch.serving.utils import pod_is_running

logger = get_logger(__name__)


class KnativeServiceManager(BaseServiceManager):
    """Service manager for Knative services with autoscaling capabilities."""

    template_label = "ksvc"

    @staticmethod
    def _convert_manifest(
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
        labels[serving_constants.KT_TEMPLATE_LABEL] = "ksvc"  # Update template label

        # Template labels (exclude template label - that's only for the top-level resource)
        template_labels = labels.copy()
        template_labels.pop(serving_constants.KT_TEMPLATE_LABEL, None)

        template_annotations = {
            "networking.knative.dev/ingress.class": "kourier.ingress.networking.knative.dev",
        }

        # Get base annotations and add Knative-specific ones
        annotations = deployment_annotations.copy()
        annotations.update(
            {
                "serving.knative.dev/container-name": "kubetorch",
                "serving.knative.dev/probe-path": "/health",
            }
        )

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

    def _update_launchtime_manifest(self, manifest: dict, service_name: str, module_name: str) -> dict:
        """Update manifest with service name and deployment timestamp."""
        clean_module_name = self._clean_module_name(module_name)
        deployment_timestamp, deployment_id = self._get_deployment_timestamp_and_id(service_name)

        service = manifest.copy()
        service["metadata"]["name"] = service_name
        service["metadata"]["labels"][serving_constants.KT_SERVICE_LABEL] = service_name
        service["metadata"]["labels"][serving_constants.KT_MODULE_LABEL] = clean_module_name
        service["metadata"]["labels"][serving_constants.KT_APP_LABEL] = service_name
        service["metadata"]["labels"][serving_constants.KT_DEPLOYMENT_ID_LABEL] = deployment_id
        service["spec"]["template"]["metadata"]["labels"][serving_constants.KT_SERVICE_LABEL] = service_name
        service["spec"]["template"]["metadata"]["labels"][serving_constants.KT_MODULE_LABEL] = clean_module_name
        service["spec"]["template"]["metadata"]["labels"][serving_constants.KT_APP_LABEL] = service_name
        service["spec"]["template"]["metadata"]["labels"][serving_constants.KT_DEPLOYMENT_ID_LABEL] = deployment_id

        service["spec"]["template"]["metadata"]["annotations"][
            "kubetorch.com/deployment_timestamp"
        ] = deployment_timestamp

        return service

    def _create_or_update_resource_from_manifest(self, manifest: dict, dryrun: bool = False) -> dict:
        """Create or update resources from a manifest."""
        try:
            kwargs = {"dry_run": "All"} if dryrun else {}
            created_service: dict = self.objects_api.create_namespaced_custom_object(
                group="serving.knative.dev",
                version="v1",
                namespace=self.namespace,
                plural="services",
                body=manifest,
                **kwargs,
            )

            logger.info(
                f"Created Knative service {manifest['metadata']['name']} in namespace {self.namespace}",
            )
            return created_service

        except client.exceptions.ApiException as e:
            if e.status == 409:
                logger.info(f"Service {manifest['metadata']['name']} already exists, updating")
                existing_service = self.get_resource(manifest["metadata"]["name"])
                return existing_service
            else:
                logger.error(
                    f"Failed to create Knative service: {str(e)}",
                )
                raise e

    def get_resource(self, service_name: str) -> dict:
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

    def update_deployment_timestamp_annotation(self, service_name: str, new_timestamp: str) -> str:
        """Update deployment timestamp annotation for Knative services."""
        try:
            patch_body = self._create_timestamp_patch_body(new_timestamp)
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

    def create_or_update_service(
        self,
        service_name: str,
        module_name: str,
        manifest: dict = None,
        dryrun: bool = False,
        **kwargs,  # Ignore other args (labels, annotations, etc. are in manifest)
    ):
        """
        Creates a Knative service with autoscaling capabilities.
        """
        logger.info(f"Deploying Kubetorch autoscaling (Knative) service with name: {service_name}")

        # Update manifest with actual service name and module name
        updated_manifest = self._update_launchtime_manifest(manifest, service_name, module_name)
        created_service = self._create_or_update_resource_from_manifest(updated_manifest, dryrun)
        return created_service, updated_manifest

    def get_endpoint(self, service_name: str) -> str:
        """Get the endpoint URL for a Knative service."""
        return self.get_knative_service_endpoint(service_name)

    def get_pods_for_service(self, service_name: str, **kwargs) -> List[client.V1Pod]:
        """Get all pods associated with this Knative service."""
        try:
            # First try to get the service to find the latest revision
            service = self.get_resource(service_name)
            status = service.get("status", {})
            latest_ready_revision = status.get("latestReadyRevisionName")

            if latest_ready_revision:
                # Look for pods with the revision label
                label_selector = f"serving.knative.dev/revision={latest_ready_revision}"
                pods = self.core_api.list_namespaced_pod(namespace=self.namespace, label_selector=label_selector)
                return pods.items
            else:
                # Fallback to the base class method if no revision is ready yet
                return super().get_pods_for_service(service_name, **kwargs)
        except Exception as e:
            logger.warning(f"Failed to get pods for Knative service {service_name}: {e}")
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
