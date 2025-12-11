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
from kubetorch.serving.base_service_manager import BaseServiceManager
from kubetorch.utils import http_conflict, http_not_found

logger = get_logger(__name__)


class DeploymentServiceManager(BaseServiceManager):
    """Service manager for Kubernetes Deployments with distributed computing support."""

    RESOURCE_TYPE = "deployment"

    def __init__(self, *args, **kwargs):
        kwargs["template_label"] = self.RESOURCE_TYPE
        super().__init__(*args, **kwargs)

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
        from kubetorch.servers.http.utils import load_template
        from kubetorch.serving.utils import nested_override

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

    def _get_pod_template_path(self) -> List[str]:
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

    def _create_or_update_resource(
        self, manifest: dict, service_name: str, clean_module_name: str, resource_config: dict = None, **kwargs
    ) -> dict:
        """Create or update Deployment resource via controller.

        The controller handles:
        - Creating the Deployment
        - Creating the regular Service for client access
        - Creating headless Service for distributed deployments
        - Storing resource_config for pod heartbeat delivery
        """
        deployment = manifest.copy()

        dryrun = kwargs.get("dry_run")
        if dryrun:
            return deployment

        # Check if this is a distributed deployment
        is_distributed = self.is_distributed(manifest)
        num_replicas = self.get_replicas(manifest)

        try:
            # Deploy via controller - handles Deployment + Services creation
            # resource_config is stored by controller and sent to pods via heartbeat
            result = self.controller_client.deploy(
                service_name=service_name,
                namespace=self.namespace,
                resource_type=self.RESOURCE_TYPE,
                resource_manifest=deployment,
                resource_config=resource_config,
                distributed=is_distributed,
                num_workers=num_replicas if is_distributed else None,
            )

            status = result.get("status", "")
            if status == "success":
                logger.info(f"Deployed {service_name} in namespace {self.namespace}")
            elif status == "error":
                raise Exception(f"Deploy failed: {result.get('message', 'Unknown error')}")

            # Return the deployment manifest (controller may return the created resource)
            return result.get("resource", deployment)

        except Exception as e:
            if http_conflict(e):
                logger.info(f"Deployment {service_name} already exists, updating replicas")
                existing_deployment = self.get_resource(service_name)

                if existing_deployment:
                    # Update replicas if different
                    existing_replicas = existing_deployment.get("spec", {}).get("replicas", 0)
                    desired_replicas = deployment["spec"]["replicas"]
                    if existing_replicas != desired_replicas:
                        patch_body = {"spec": {"replicas": desired_replicas}}
                        try:
                            self.controller_client.patch_deployment(
                                namespace=self.namespace,
                                name=service_name,
                                body=patch_body,
                            )
                            logger.info(f"Updated Deployment {service_name} replicas to {desired_replicas}")
                        except Exception as patch_error:
                            logger.error(f"Failed to patch Deployment {service_name}: {patch_error}")
                            raise patch_error

                    return existing_deployment

                return deployment
            else:
                logger.error(f"Failed to deploy {service_name}: {str(e)}")
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

        logger.info(f"Checking Deployment {service_name} pod readiness (timeout: {launch_timeout} seconds)")

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
                    logger.info(f"Deployment {service_name} pod(s) are now ready with {ready_replicas} replicas")
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

    def _teardown_associated_resources(self, service_name: str, console=None) -> bool:
        """Teardown associated Kubernetes Services for Deployment."""
        success = True

        # Delete regular service
        try:
            self.controller_client.delete_service(name=service_name, namespace=self.namespace)
            if console:
                console.print(f"✓ Deleted service [blue]{service_name}[/blue]")
            else:
                logger.info(f"Deleted service {service_name}")
        except Exception as e:
            if http_not_found(e):
                if console:
                    console.print(f"[yellow]Note:[/yellow] Service {service_name} not found or already deleted")
                else:
                    logger.info(f"Service {service_name} not found or already deleted")
            else:
                if console:
                    console.print(f"[red]Error:[/red] Failed to delete service {service_name}: {e}")
                else:
                    logger.error(f"Failed to delete service {service_name}: {e}")
                success = False

        # Delete headless service if it exists
        headless_service_name = f"{service_name}-headless"
        try:
            self.controller_client.delete_service(name=headless_service_name, namespace=self.namespace)
            if console:
                console.print(f"✓ Deleted service [blue]{headless_service_name}[/blue]")
            else:
                logger.info(f"Deleted service {headless_service_name}")
        except Exception as e:
            if http_not_found(e):
                # Headless service might not exist, which is fine
                pass
            else:
                if console:
                    console.print(f"[red]Error:[/red] Failed to delete service {headless_service_name}: {e}")
                else:
                    logger.error(f"Failed to delete service {headless_service_name}: {e}")
                success = False

        return success
