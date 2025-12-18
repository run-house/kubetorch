import os
import time
from typing import List

import kubetorch.serving.constants as serving_constants

from kubetorch.logger import get_logger
from kubetorch.resources.compute.utils import (
    check_pod_events_for_errors,
    check_pod_status_for_errors,
    check_replicaset_events_for_errors,
    ServiceTimeoutError,
)
from kubetorch.servers.http.utils import load_template
from kubetorch.serving.base_service_manager import BaseServiceManager
from kubetorch.serving.utils import nested_override
from kubetorch.utils import http_conflict, http_not_found

logger = get_logger(__name__)


class DeploymentServiceManager(BaseServiceManager):
    """Service manager for Kubernetes Deployments with distributed computing support."""

    RESOURCE_TYPE = "deployment"

    def __init__(self, *args, **kwargs):
        kwargs["template_label"] = self.RESOURCE_TYPE
        super().__init__(*args, **kwargs)

    def _delete_resource(self, service_name: str, force: bool = False, **kwargs) -> None:
        """Delete a Deployment and its associated K8s Services."""
        grace_period_seconds = 0 if force else None
        propagation_policy = "Foreground" if force else None

        # Delete the Deployment
        self.controller_client.delete_deployment(
            name=service_name,
            namespace=self.namespace,
            grace_period_seconds=grace_period_seconds,
            propagation_policy=propagation_policy,
        )

        # Delete the associated K8s Service
        try:
            self.controller_client.delete_service(
                namespace=self.namespace,
                name=service_name,
            )
        except Exception as e:
            if not http_not_found(e):
                logger.warning(f"Failed to delete service {service_name}: {e}")

        # Delete the headless service if it exists
        try:
            self.controller_client.delete_service(
                namespace=self.namespace,
                name=f"{service_name}-headless",
            )
        except Exception as e:
            if not http_not_found(e):
                logger.warning(f"Failed to delete headless service {service_name}-headless: {e}")

    @classmethod
    def _build_base_manifest(
        cls,
        pod_spec: dict,
        namespace: str,
        replicas: int = 1,
        inactivity_ttl: str = None,
        custom_labels: dict = None,
        custom_annotations: dict = None,
        custom_template: dict = None,
    ) -> dict:
        """Build a base deployment manifest from pod spec and configuration.

        Returns:
            Deployment manifest dictionary
        """
        # Build labels
        labels = cls._get_labels(
            template_label=cls.RESOURCE_TYPE,
            custom_labels=custom_labels,
        )

        # Template labels (exclude kt template label)
        template_labels = labels.copy()
        template_labels.pop(serving_constants.KT_TEMPLATE_LABEL, None)

        annotations = cls._get_annotations(
            service_annotations=None,
            custom_annotations=custom_annotations,
            inactivity_ttl=inactivity_ttl,
        )

        # Create Deployment manifest
        deployment = load_template(
            template_file=serving_constants.DEPLOYMENT_TEMPLATE_FILE,
            template_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"),
            name="",  # Will be set during launch
            namespace=namespace,
            annotations=annotations,
            template_annotations={},  # Will be filled in during launch
            labels=labels,
            template_labels=template_labels,
            pod_spec=pod_spec,
            replicas=replicas,
        )

        if custom_template:
            nested_override(deployment, custom_template)

        return deployment

    def get_pod_template_path(self) -> List[str]:
        """Get the path to the pod template."""
        return ["spec", "template"]

    def _update_launchtime_manifest(
        self, manifest: dict, service_name: str, clean_module_name: str, deployment_timestamp: str, deployment_id: str
    ) -> dict:
        """Update manifest with service name and deployment timestamp."""
        deployment = super()._update_launchtime_manifest(
            manifest, service_name, clean_module_name, deployment_timestamp, deployment_id
        )

        deployment["spec"].setdefault("selector", {}).setdefault("matchLabels", {})
        deployment["spec"]["selector"]["matchLabels"][serving_constants.KT_SERVICE_LABEL] = service_name
        deployment["spec"]["selector"]["matchLabels"][serving_constants.KT_MODULE_LABEL] = clean_module_name

        return deployment

    def _create_or_update_resource(self, manifest: dict, service_name: str, clean_module_name: str, **kwargs) -> dict:
        deployment = manifest.copy()

        pod_spec = self.pod_spec(deployment)
        server_port = pod_spec.get("containers", [{}])[0].get("ports", [{}])[0].get("containerPort", 32300)
        labels = deployment.get("metadata", {}).get("labels", {})
        annotations = deployment.get("metadata", {}).get("annotations", {})

        # Service labels (exclude kt template label)
        service_labels = labels.copy()
        service_labels.pop(serving_constants.KT_TEMPLATE_LABEL, None)

        dryrun = kwargs.get("dry_run")
        dockerfile = kwargs.get("dockerfile")
        module = kwargs.get("module")
        pod_selector = kwargs.get("pod_selector")
        create_headless_service = kwargs.get("create_headless_service", False)

        try:
            # Step 1: Apply the compute manifest (creates the pods)
            apply_response = self.controller_client.apply(
                service_name=service_name,
                namespace=self.namespace,
                resource_type=self.RESOURCE_TYPE,
                resource_manifest=deployment,
            )

            if apply_response.get("status") == "error":
                raise Exception(f"Apply failed: {apply_response.get('message')}")

            logger.info(f"Applied deployment {deployment['metadata']['name']} in namespace {self.namespace}")

            # Step 2: Register pool via /pool (creates K8s Service to route to pods)
            if not dryrun:
                # Use provided pod_selector (BYO manifest) or build from kubetorch labels
                if pod_selector:
                    # BYO manifest: use user's selector
                    selector = pod_selector
                else:
                    # Standard kubetorch deployment: use kubetorch labels
                    selector = {
                        serving_constants.KT_SERVICE_LABEL: service_name,
                        serving_constants.KT_MODULE_LABEL: clean_module_name,
                    }
                specifier = {
                    "type": "label_selector",
                    "selector": selector,
                }

                pool_response = self.controller_client.register_pool(
                    name=service_name,
                    namespace=self.namespace,
                    specifier=specifier,
                    server_port=server_port,
                    labels=service_labels,
                    annotations=annotations,
                    pool_metadata={
                        "username": self.username,
                    },
                    dockerfile=dockerfile,
                    module=module,
                    resource_kind="Deployment",
                    resource_name=service_name,
                    create_headless_service=create_headless_service,
                )
                if pool_response.get("status") != "success":
                    raise Exception(f"Resource registration failed: {pool_response.get('message')}")
                logger.info(f"Registered {service_name} in namespace {self.namespace}")

            # Return the created resource from apply response
            return apply_response.get("resource", deployment)

        except Exception as e:
            if http_conflict(e):
                logger.info(f"Deployment {deployment['metadata']['name']} already exists, updating")
                existing_deployment = self.get_resource(deployment["metadata"]["name"])

                # Update replicas if different
                existing_replicas = existing_deployment.get("spec", {}).get("replicas", 0)
                desired_replicas = deployment["spec"]["replicas"]
                if existing_replicas != desired_replicas:
                    patch_body = {"spec": {"replicas": desired_replicas}}
                    try:
                        self.controller_client.patch_deployment(
                            namespace=self.namespace,
                            name=deployment["metadata"]["name"],
                            body=patch_body,
                        )
                        logger.info(
                            f"Updated Deployment {deployment['metadata']['name']} replicas to {desired_replicas}"
                        )
                    except Exception as patch_error:
                        logger.error(f"Failed to patch Deployment {deployment['metadata']['name']}: {patch_error}")
                        raise patch_error

                return existing_deployment
            else:
                logger.error(f"Failed to create Deployment: {str(e)}")
                raise e

    def get_replicas(self, manifest: dict) -> int:
        """Get the number of replicas."""
        return manifest.get("spec", {}).get("replicas", 1)

    def set_replicas(self, manifest: dict, value: int) -> None:
        """Set the number of replicas."""
        manifest.setdefault("spec", {})["replicas"] = value

    def get_resource(self, service_name: str) -> dict:
        """Retrieve a Deployment by name."""
        try:
            deployment = self.controller_client.get_deployment(
                namespace=self.namespace,
                name=service_name,
            )
            return deployment
        except Exception as e:
            if http_not_found(e):
                return {}
            logger.error(f"Failed to load Deployment '{service_name}': {e}")
            raise

    def update_deployment_timestamp_annotation(self, service_name: str, new_timestamp: str) -> str:
        """Update deployment timestamp annotation for Deployment services."""
        try:
            patch_body = self._create_timestamp_patch_body(new_timestamp)
            self.controller_client.patch_deployment(
                namespace=self.namespace,
                name=service_name,
                body=patch_body,
            )
            return new_timestamp
        except Exception as e:
            logger.error(f"Failed to update deployment timestamp for '{service_name}': {str(e)}")
            raise

    def get_endpoint(self, service_name: str) -> str:
        """Get the endpoint URL for a Deployment service."""
        return f"http://{service_name}.{self.namespace}.svc.cluster.local:80"

    def check_service_ready(
        self,
        service_name: str,
        launch_timeout: int,
        **kwargs,
    ) -> bool:
        """Checks if the Deployment is ready to start serving requests.

        Args:
            service_name: Name of the Deployment service
            launch_timeout: Timeout in seconds to wait for readiness
            **kwargs: Additional arguments (ignored for Deployments)

        Returns:
            True if service is ready

        Raises:
            ServiceTimeoutError: If service doesn't become ready within timeout
        """
        sleep_interval = 2
        start_time = time.time()

        logger.debug(f"Checking Deployment {service_name} pod readiness (timeout: {launch_timeout} seconds)")

        iteration = 0
        while (time.time() - start_time) < launch_timeout:
            iteration += 1
            try:
                # Get Deployment
                deployment = self.get_resource(service_name)
                if not deployment:
                    logger.debug(f"Waiting for Deployment {service_name} to be created")
                    time.sleep(sleep_interval)
                    continue

                # Check if all replicas are ready
                status = deployment.get("status", {})
                spec = deployment.get("spec", {})
                ready_replicas = status.get("readyReplicas", 0)
                desired_replicas = spec.get("replicas", 0)

                if iteration % 3 == 0:
                    logger.debug(f"Deployment {service_name}: {ready_replicas}/{desired_replicas} replicas ready")

                if ready_replicas >= desired_replicas and desired_replicas > 0:
                    logger.info(f"Pods ready ({ready_replicas}/{desired_replicas} replicas)")
                    return True

                # Check for pod-level issues
                pods = self.get_pods_for_service(service_name)
                for pod in pods:
                    # Check for image pull errors in container status
                    check_pod_status_for_errors(pod)

                    # Check pod events separately from the core API
                    check_pod_events_for_errors(pod, self.namespace)

                # If no pods exist, check for ReplicaSet-level errors (like PriorityClass issues)
                if not pods:
                    check_replicaset_events_for_errors(
                        namespace=self.namespace,
                        service_name=service_name,
                    )

            except Exception as e:
                logger.error(f"Error checking Deployment readiness: {e}")
                raise

            if iteration % 10 == 0:
                elapsed = int(time.time() - start_time)
                remaining = max(0, int(launch_timeout - elapsed))
                logger.info(f"Deployment is not yet ready " f"(elapsed: {elapsed}s, remaining: {remaining}s)")

            time.sleep(sleep_interval)

        raise ServiceTimeoutError(f"Deployment {service_name} is not ready after {launch_timeout} seconds")
