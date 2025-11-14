import os
import time
from typing import Tuple

from kubernetes import client

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

logger = get_logger(__name__)


class DeploymentServiceManager(BaseServiceManager):
    """Service manager for Kubernetes Deployments with distributed computing support."""

    template_label = "deployment"

    def _create_or_update_resource(
        self,
        name: str,
        module_name: str,
        pod_template: dict,
        replicas: int = 1,
        inactivity_ttl: str = None,
        custom_labels: dict = None,
        custom_annotations: dict = None,
        custom_template: dict = None,
        scheduler_name: str = None,
        queue_name: str = None,
        dryrun: bool = False,
    ) -> Tuple[dict, bool]:
        """Creates or updates a Deployment for distributed deployments.

        Returns:
            Tuple (created_deployment, is_new_deployment)
        """
        clean_module_name = self._clean_module_name(module_name)
        service_name = name  # Use regular service name, not headless

        labels = self._get_labels(clean_module_name, name, custom_labels, scheduler_name, queue_name)

        # Template labels (exclude template label - that's only for the top-level resource)
        template_labels = labels.copy()
        template_labels.pop(serving_constants.KT_TEMPLATE_LABEL, None)

        # Service labels (also exclude template label - supporting resource, not source-of-truth)
        service_labels = labels.copy()
        service_labels.pop(serving_constants.KT_TEMPLATE_LABEL, None)

        annotations = self._get_annotations(custom_annotations, inactivity_ttl)

        template_annotations = {"kubetorch.com/deployment_timestamp": self._get_deployment_timestamp()}

        # Create Deployment
        deployment = load_template(
            template_file=serving_constants.DEPLOYMENT_TEMPLATE_FILE,
            template_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"),
            name=name,
            namespace=self.namespace,
            annotations=annotations,
            template_annotations=template_annotations,
            labels=labels,
            template_labels=template_labels,
            pod_template=pod_template,
            replicas=replicas,
        )

        if custom_template:
            nested_override(deployment, custom_template)

        # Check if this is a distributed deployment
        env_vars = pod_template.get("containers", [{}])[0].get("env", [])
        is_distributed = any(
            env.get("name") == "KT_DISTRIBUTED_CONFIG" and env.get("value") != "null" and env.get("value")
            for env in env_vars
        )

        # Create regular service with session affinity
        service = load_template(
            template_file=serving_constants.DEPLOYMENT_SERVICE_TEMPLATE_FILE,
            template_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"),
            name=service_name,
            namespace=self.namespace,
            annotations=annotations,
            labels=service_labels,
            deployment_name=name,
            module_name=clean_module_name,
            distributed=False,  # Keep regular service for client access
            server_port=pod_template.get("containers", [{}])[0].get("ports", [{}])[0].get("containerPort", 32300),
        )

        # For distributed deployments, also create a headless service for pod discovery
        headless_service = None
        if is_distributed:
            headless_service = load_template(
                template_file=serving_constants.DEPLOYMENT_SERVICE_TEMPLATE_FILE,
                template_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"),
                name=f"{service_name}-headless",  # Use different name for headless
                namespace=self.namespace,
                annotations=annotations,
                labels=service_labels,
                deployment_name=name,
                module_name=clean_module_name,
                distributed=True,  # Make this one headless
                server_port=pod_template.get("containers", [{}])[0].get("ports", [{}])[0].get("containerPort", 32300),
            )

        try:
            kwargs = {"dry_run": "All"} if dryrun else {}

            # Create regular service first
            try:
                self.core_api.create_namespaced_service(
                    namespace=self.namespace,
                    body=service,
                    **kwargs,
                )
                if not dryrun:
                    logger.info(f"Created service {service_name} in namespace {self.namespace}")
            except client.exceptions.ApiException as e:
                if e.status == 409:
                    logger.info(f"Service {service_name} already exists")
                else:
                    raise

            # Create headless service for distributed pod discovery
            if headless_service:
                try:
                    self.core_api.create_namespaced_service(
                        namespace=self.namespace,
                        body=headless_service,
                        **kwargs,
                    )
                    if not dryrun:
                        logger.info(f"Created headless service {service_name}-headless in namespace {self.namespace}")
                except client.exceptions.ApiException as e:
                    if e.status == 409:
                        logger.info(f"Headless service {service_name}-headless already exists")
                    else:
                        raise

            # Create Deployment
            created_deployment = self.apps_v1_api.create_namespaced_deployment(
                namespace=self.namespace,
                body=deployment,
                **kwargs,
            )

            if dryrun:
                return created_deployment, False

            logger.info(f"Created Deployment {name} in namespace {self.namespace}")
            return created_deployment, True

        except client.exceptions.ApiException as e:
            if e.status == 409:
                logger.info(f"Deployment {name} already exists, updating")
                existing_deployment = self.get_resource(name)

                # Update replicas if different
                if existing_deployment.spec.replicas != replicas:
                    patch_body = {"spec": {"replicas": replicas}}
                    try:
                        self.apps_v1_api.patch_namespaced_deployment(
                            name=name,
                            namespace=self.namespace,
                            body=patch_body,
                        )
                        logger.info(f"Updated Deployment {name} replicas to {replicas}")
                    except Exception as e:
                        logger.error(f"Failed to patch Deployment {name}: {e}")
                        raise e

                return existing_deployment, False
            else:
                logger.error(f"Failed to create Deployment: {str(e)}")
                raise e

    def get_resource(self, service_name: str) -> dict:
        """Retrieve a Deployment by name."""
        try:
            deployment = self.apps_v1_api.read_namespaced_deployment(
                name=service_name,
                namespace=self.namespace,
            )
            return deployment
        except client.exceptions.ApiException as e:
            logger.error(f"Failed to load Deployment '{service_name}': {str(e)}")
            raise

    def update_deployment_timestamp_annotation(self, service_name: str, new_timestamp: str) -> str:
        """Update deployment timestamp annotation for Deployment services."""
        try:
            patch_body = self._create_timestamp_patch_body(new_timestamp)
            self.apps_v1_api.patch_namespaced_deployment(
                name=service_name,
                namespace=self.namespace,
                body=patch_body,
            )
            return new_timestamp
        except client.exceptions.ApiException as e:
            logger.error(f"Failed to update deployment timestamp for '{service_name}': {str(e)}")
            raise

    def create_or_update_service(
        self,
        service_name: str,
        module_name: str,
        pod_template: dict,
        replicas: int = 1,
        inactivity_ttl: str = None,
        custom_labels: dict = None,
        custom_annotations: dict = None,
        custom_template: dict = None,
        scheduler_name: str = None,
        queue_name: str = None,
        dryrun: bool = False,
        **kwargs,  # Ignore Knative-specific args like autoscaling_config, inactivity_ttl, etc.
    ):
        """
        Creates a Deployment service.

        Args:
            service_name (str): Name for the pod/service.
            module_name (str): Name of the module.
            pod_template (dict): Template for the pod, including resource requirements.
            replicas (int): Number of replicas for the service
            custom_labels (dict, optional): Custom labels to add to the service.
            custom_annotations (dict, optional): Custom annotations to add to the service.
            custom_template (dict, optional): Custom template to apply to the service.
            dryrun (bool, optional): Whether to run in dryrun mode (Default: `False`).
        """
        logger.info(f"Deploying Kubetorch service with name: {service_name}")
        try:
            created_service, _ = self._create_or_update_resource(
                name=service_name,
                pod_template=pod_template,
                module_name=module_name,
                replicas=replicas,
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
            logger.error(f"Failed to launch new Deployment: {str(e)}")
            raise e

    def get_endpoint(self, service_name: str) -> str:
        """Get the endpoint URL for a Deployment service."""
        return f"http://{service_name}.{self.namespace}.svc.cluster.local:80"

    def check_service_ready(
        self,
        service_name: str,
        launch_timeout: int,
        core_api: client.CoreV1Api = None,
        **kwargs,
    ) -> bool:
        """Checks if the Deployment is ready to start serving requests.

        Args:
            service_name: Name of the Deployment service
            launch_timeout: Timeout in seconds to wait for readiness
            core_api: Core API instance (uses self.core_api if None)
            **kwargs: Additional arguments (ignored for Deployments)

        Returns:
            True if service is ready

        Raises:
            ServiceTimeoutError: If service doesn't become ready within timeout
        """
        if core_api is None:
            core_api = self.core_api

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
                ready_replicas = deployment.status.ready_replicas or 0
                desired_replicas = deployment.spec.replicas or 0

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
                    check_pod_events_for_errors(pod, self.namespace, core_api)

                # If no pods exist, check for ReplicaSet-level errors (like PriorityClass issues)
                if not pods:
                    check_replicaset_events_for_errors(
                        namespace=self.namespace,
                        service_name=service_name,
                        apps_v1_api=self.apps_v1_api,
                        core_api=core_api,
                    )

            except client.exceptions.ApiException as e:
                logger.error(f"Error checking Deployment readiness: {e}")
                raise

            if iteration % 10 == 0:
                elapsed = int(time.time() - start_time)
                remaining = max(0, int(launch_timeout - elapsed))
                logger.info(f"Deployment is not yet ready " f"(elapsed: {elapsed}s, remaining: {remaining}s)")

            time.sleep(sleep_interval)

        raise ServiceTimeoutError(f"Deployment {service_name} is not ready after {launch_timeout} seconds")

    def teardown_service(self, service_name: str, console=None) -> bool:
        """Teardown Deployment and associated resources.

        Args:
            service_name: Name of the Deployment to teardown
            console: Optional Rich console for output

        Returns:
            True if teardown was successful, False otherwise
        """
        from kubetorch.resources.compute.utils import delete_deployment

        try:
            # Delete the Deployment and its associated service
            delete_deployment(
                apps_v1_api=self.apps_v1_api,
                core_api=self.core_api,
                name=service_name,
                namespace=self.namespace,
                console=console,
            )

            return True

        except Exception as e:
            logger.error(f"Failed to teardown Deployment {service_name}: {e}")
            return False
