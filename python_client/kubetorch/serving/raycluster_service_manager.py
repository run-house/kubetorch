import os
import re
import time
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from kubernetes import client

import kubetorch.serving.constants as serving_constants
from kubetorch.logger import get_logger
from kubetorch.servers.http.utils import load_template
from kubetorch.serving.base_service_manager import BaseServiceManager
from kubetorch.serving.utils import nested_override

logger = get_logger(__name__)


class RayClusterServiceManager(BaseServiceManager):
    """Service manager for Ray clusters with distributed Ray workload support."""

    def _create_or_update_raycluster(
        self,
        name: str,
        module_name: str,
        pod_template: dict,
        replicas: int = 1,
        inactivity_ttl: str = None,
        custom_labels: dict = None,
        custom_annotations: dict = None,
        custom_template: dict = None,
        dryrun: bool = False,
    ) -> Tuple[dict, bool]:
        """Creates or updates a RayCluster for Ray distributed workloads.

        Returns:
            Tuple (created_raycluster, is_new_raycluster)
        """
        clean_module_name = re.sub(r"[^A-Za-z0-9.-]|^[-.]|[-.]$", "", module_name)

        labels = {
            **self.base_labels,
            serving_constants.KT_MODULE_LABEL: clean_module_name,
            serving_constants.KT_SERVICE_LABEL: name,
            serving_constants.KT_TEMPLATE_LABEL: "raycluster",  # Mark as source-of-truth
        }
        if custom_labels:
            labels.update(custom_labels)

        # Template labels (exclude template label - that's only for the top-level resource)
        # Add ray-node-type label to distinguish head from worker nodes
        template_labels = {
            **self.base_labels,
            serving_constants.KT_MODULE_LABEL: clean_module_name,
            serving_constants.KT_SERVICE_LABEL: name,
        }
        if custom_labels:
            template_labels.update(custom_labels)

        # Head node specific labels (for service selector)
        head_template_labels = {
            **template_labels,
            "ray.io/node-type": "head",  # KubeRay standard label
        }

        # Worker node specific labels
        worker_template_labels = {
            **template_labels,
            "ray.io/node-type": "worker",  # KubeRay standard label
        }

        annotations = {
            "prometheus.io/scrape": "true",
            "prometheus.io/path": serving_constants.PROMETHEUS_HEALTH_ENDPOINT,
            "prometheus.io/port": "8080",
            "ray.io/overwrite-container-cmd": "true",
        }
        if custom_annotations:
            annotations.update(custom_annotations)

        deployment_timestamp = datetime.now(timezone.utc).isoformat()
        template_annotations = {"kubetorch.com/deployment_timestamp": deployment_timestamp}

        if inactivity_ttl:
            annotations[serving_constants.INACTIVITY_TTL_ANNOTATION] = inactivity_ttl
            logger.info(f"Configuring auto-down after idle timeout ({inactivity_ttl})")

        # Create RayCluster
        worker_replicas = max(0, replicas - 1)  # Head node counts as 1 replica
        raycluster = load_template(
            template_file=serving_constants.RAYCLUSTER_TEMPLATE_FILE,
            template_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"),
            name=name,
            namespace=self.namespace,
            annotations=annotations,
            template_annotations=template_annotations,
            labels=labels,
            head_template_labels=head_template_labels,
            worker_template_labels=worker_template_labels,
            pod_template=pod_template,
            worker_replicas=worker_replicas,
        )

        # Create Kubernetes Service pointing to head node HTTP server (like Deployments)
        service_labels = {
            **self.base_labels,
            serving_constants.KT_MODULE_LABEL: clean_module_name,
            serving_constants.KT_SERVICE_LABEL: name,
        }
        if custom_labels:
            service_labels.update(custom_labels)

        # Ray clusters are always distributed, so we need headless services for pod discovery
        # Create regular service for client access (head node only)
        service = load_template(
            template_file=serving_constants.RAYCLUSTER_SERVICE_TEMPLATE_FILE,
            template_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"),
            name=name,
            namespace=self.namespace,
            annotations=annotations,
            labels=service_labels,
            deployment_name=name,  # Use same parameter name as deployment for compatibility
            module_name=clean_module_name,
            distributed=False,  # Keep regular service for client access
            server_port=pod_template.get("containers", [{}])[0].get("ports", [{}])[0].get("containerPort", 32300),
        )

        # Create headless service for Ray pod discovery (all nodes)
        headless_service_labels = service_labels.copy()
        headless_service = load_template(
            template_file=serving_constants.RAYCLUSTER_SERVICE_TEMPLATE_FILE,
            template_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"),
            name=f"{name}-headless",
            namespace=self.namespace,
            annotations=annotations,
            labels=headless_service_labels,
            deployment_name=name,
            module_name=clean_module_name,
            distributed=True,  # Make headless for pod discovery
            server_port=pod_template.get("containers", [{}])[0].get("ports", [{}])[0].get("containerPort", 32300),
        )

        # For headless service, select all Ray nodes (not just head)
        headless_service["spec"]["selector"].pop("ray.io/node-type", None)

        if custom_template:
            nested_override(raycluster, custom_template)

        try:
            kwargs = {"dry_run": "All"} if dryrun else {}

            # Create Kubernetes Service first (regular service for client access)
            try:
                self.core_api.create_namespaced_service(
                    namespace=self.namespace,
                    body=service,
                    **kwargs,
                )
                if not dryrun:
                    logger.info(f"Created service {name} in namespace {self.namespace}")
            except client.exceptions.ApiException as e:
                if e.status == 409:
                    logger.info(f"Service {name} already exists")
                else:
                    raise

            # Create headless service for Ray pod discovery (all nodes)
            try:
                self.core_api.create_namespaced_service(
                    namespace=self.namespace,
                    body=headless_service,
                    **kwargs,
                )
                if not dryrun:
                    logger.info(f"Created headless service {name}-headless in namespace {self.namespace}")
            except client.exceptions.ApiException as e:
                if e.status == 409:
                    logger.info(f"Headless service {name}-headless already exists")
                else:
                    raise

            # Create RayCluster
            created_raycluster = None
            try:
                created_raycluster = self.objects_api.create_namespaced_custom_object(
                    group="ray.io",
                    version="v1",
                    namespace=self.namespace,
                    plural="rayclusters",
                    body=raycluster,
                    **kwargs,
                )
            except client.exceptions.ApiException as e:
                if e.status == 404:
                    logger.error(
                        "RayCluster Custom Resource Definition (CRD) not found, please install the KubeRay operator"
                    )
                raise e

            if dryrun:
                return created_raycluster, False

            logger.info(f"Created RayCluster {name} in namespace {self.namespace}")
            return created_raycluster, True

        except client.exceptions.ApiException as e:
            if e.status == 409:
                logger.info(f"RayCluster {name} already exists, updating")
                try:
                    # For RayCluster, we can patch the spec
                    patch_body = {"spec": raycluster["spec"]}
                    updated_raycluster = self.objects_api.patch_namespaced_custom_object(
                        group="ray.io",
                        version="v1",
                        namespace=self.namespace,
                        plural="rayclusters",
                        name=name,
                        body=patch_body,
                    )
                    logger.info(f"Updated RayCluster {name}")
                    return updated_raycluster, False
                except Exception as patch_error:
                    logger.error(f"Failed to patch RayCluster {name}: {patch_error}")
                    raise patch_error

            raise e

    def get_raycluster(self, raycluster_name: str) -> dict:
        """Retrieve a RayCluster by name."""
        try:
            raycluster = self.objects_api.get_namespaced_custom_object(
                group="ray.io",
                version="v1",
                namespace=self.namespace,
                plural="rayclusters",
                name=raycluster_name,
            )
            return raycluster
        except client.exceptions.ApiException as e:
            logger.error(f"Failed to load RayCluster '{raycluster_name}': {str(e)}")
            raise

    def get_deployment_timestamp_annotation(self, service_name: str) -> Optional[str]:
        """Get deployment timestamp annotation for RayCluster services."""
        try:
            raycluster = self.get_raycluster(service_name)
            if raycluster:
                return (
                    raycluster.get("metadata", {})
                    .get("annotations", {})
                    .get("kubetorch.com/deployment_timestamp", None)
                )
        except client.exceptions.ApiException:
            pass
        return None

    def update_deployment_timestamp_annotation(self, service_name: str, new_timestamp: str) -> str:
        """Update deployment timestamp annotation for RayCluster services."""
        try:
            patch_body = {"metadata": {"annotations": {"kubetorch.com/deployment_timestamp": new_timestamp}}}
            self.objects_api.patch_namespaced_custom_object(
                group="ray.io",
                version="v1",
                namespace=self.namespace,
                plural="rayclusters",
                name=service_name,
                body=patch_body,
            )
            return new_timestamp
        except client.exceptions.ApiException as e:
            logger.error(f"Failed to update deployment timestamp for RayCluster '{service_name}': {str(e)}")
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
        dryrun: bool = False,
        **kwargs,  # Ignore Knative-specific args like autoscaling_config, inactivity_ttl, etc.
    ):
        """
        Creates a RayCluster service.

        Args:
            service_name (str): Name for the RayCluster.
            module_name (str): Name of the module.
            pod_template (dict): Template for the pod, including resource requirements.
            replicas (int): Number of replicas for the service (head + workers)
            custom_labels (dict, optional): Custom labels to add to the service.
            custom_annotations (dict, optional): Custom annotations to add to the service.
            custom_template (dict, optional): Custom template to apply to the service.
            dryrun (bool, optional): Whether to run in dryrun mode (Default: `False`).
        """
        logger.info(f"Deploying Kubetorch RayCluster service with name: {service_name}")
        try:
            created_service, is_new_service = self._create_or_update_raycluster(
                name=service_name,
                pod_template=pod_template,
                module_name=module_name,
                replicas=replicas,
                inactivity_ttl=inactivity_ttl,
                custom_labels=custom_labels,
                custom_annotations=custom_annotations,
                custom_template=custom_template,
                dryrun=dryrun,
            )
            return created_service
        except Exception as e:
            logger.error(f"Failed to launch new RayCluster: {str(e)}")
            raise e

    def get_pods_for_service(self, service_name: str, **kwargs) -> List[client.V1Pod]:
        """Get all pods associated with this RayCluster service.

        Args:
            service_name (str): Name of the service

        Returns:
            List[V1Pod]: List of running pods associated with the service.
        """
        return self.get_pods_for_service_static(
            service_name=service_name,
            namespace=self.namespace,
            core_api=self.core_api,
        )

    def get_endpoint(self, service_name: str) -> str:
        """Get the endpoint URL for a RayCluster service.

        Returns the HTTP endpoint for the KubeTorch HTTP server running on the head node,
        just like Deployment services.
        """
        return f"http://{service_name}.{self.namespace}.svc.cluster.local:80"

    def check_service_ready(self, service_name: str, launch_timeout: int, **kwargs) -> bool:
        """Checks if the RayCluster is ready to start serving requests.

        Args:
            service_name: Name of the RayCluster service
            launch_timeout: Timeout in seconds to wait for readiness
            **kwargs: Additional arguments (ignored for RayClusters)

        Returns:
            True if service is ready

        Raises:
            TimeoutError: If service doesn't become ready within timeout
            RuntimeError: If RayCluster fails to start
        """
        sleep_interval = 2
        start_time = time.time()

        logger.info(f"Checking RayCluster {service_name} pod readiness (timeout: {launch_timeout} seconds)")

        iteration = 0
        while (time.time() - start_time) < launch_timeout:
            iteration += 1
            try:
                raycluster = self.get_raycluster(service_name)
                status = raycluster.get("status", {})

                # Check RayCluster state
                state = status.get("state", "-")
                if state == "ready":
                    logger.info(f"RayCluster {service_name} is ready")
                    return True
                elif state == "failed":
                    raise RuntimeError(f"RayCluster {service_name} failed to start")

                # Calculate total expected replicas from head + all worker groups
                spec = raycluster.get("spec", {})

                # Head group replicas
                head_group_spec = spec.get("headGroupSpec", {})
                head_replicas = head_group_spec.get("replicas", 1)

                # Worker group replicas (sum across all worker groups)
                worker_groups = spec.get("workerGroupSpecs", [])
                worker_replicas = sum(worker_group.get("replicas", 0) for worker_group in worker_groups)

                total_expected_replicas = head_replicas + worker_replicas

                # Check pods are running
                pods = self.get_pods_for_service(service_name)
                running_pods = [pod for pod in pods if pod.status.phase == "Running"]

                # Count head and worker pods separately for better logging
                head_pods = [pod for pod in running_pods if pod.metadata.labels.get("ray.io/node-type") == "head"]
                worker_pods = [pod for pod in running_pods if pod.metadata.labels.get("ray.io/node-type") == "worker"]

                # Check for specific error conditions
                if head_pods:
                    head_pod = head_pods[0]
                    # Check for Ray installation errors in head pod
                    ray_error = self._check_ray_installation_error(service_name, head_pod.metadata.name)
                    if ray_error:
                        raise RuntimeError(ray_error)

                if len(running_pods) >= total_expected_replicas:
                    logger.info(
                        f"RayCluster {service_name} is ready with {len(running_pods)} pods "
                        f"({len(head_pods)} head, {len(worker_pods)} worker{'' if len(worker_pods) == 1 else 's'})"
                    )
                    return True

                # Log progress every 30 seconds
                if iteration % (30 // sleep_interval) == 0:
                    elapsed = int(time.time() - start_time)
                    remaining = launch_timeout - elapsed
                    logger.info(
                        f"RayCluster is not yet ready (elapsed: {elapsed}s, remaining: {remaining}s). "
                        f"State: {state}, Running pods: {len(running_pods)}/{total_expected_replicas} "
                        f"({len(head_pods)}/{head_replicas} head, {len(worker_pods)}/{worker_replicas} worker{'' if worker_replicas == 1 else 's'})"
                    )

            except RuntimeError as e:
                raise e
            except Exception as e:
                logger.error(f"Error checking RayCluster readiness: {e}")

            time.sleep(sleep_interval)

        # Timeout reached
        raise TimeoutError(f"RayCluster {service_name} did not become ready within {launch_timeout} seconds")

    def teardown_service(self, service_name: str, console=None) -> bool:
        """Teardown RayCluster and associated resources.

        Args:
            service_name: Name of the RayCluster to teardown
            console: Optional Rich console for output

        Returns:
            True if teardown was successful, False otherwise
        """
        success = True

        try:
            # Delete the RayCluster
            self.objects_api.delete_namespaced_custom_object(
                group="ray.io",
                version="v1",
                namespace=self.namespace,
                plural="rayclusters",
                name=service_name,
            )
            if console:
                console.print(f"✓ Deleted RayCluster [blue]{service_name}[/blue]")
            else:
                logger.info(f"Deleted RayCluster {service_name}")

        except client.exceptions.ApiException as e:
            if e.status == 404:
                if console:
                    console.print(f"[yellow]Note:[/yellow] RayCluster {service_name} not found or already deleted")
                else:
                    logger.info(f"RayCluster {service_name} not found or already deleted")
            else:
                if console:
                    console.print(f"[red]Error:[/red] Failed to delete RayCluster {service_name}: {e}")
                else:
                    logger.error(f"Failed to delete RayCluster {service_name}: {e}")
                success = False

        try:
            # Delete the associated Kubernetes service (created alongside RayCluster)
            self.core_api.delete_namespaced_service(name=service_name, namespace=self.namespace)
            if console:
                console.print(f"✓ Deleted service [blue]{service_name}[/blue]")
            else:
                logger.info(f"Deleted service {service_name}")

        except client.exceptions.ApiException as e:
            if e.status == 404:
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

        return success

    def _check_ray_installation_error(self, service_name: str, head_pod_name: str) -> Optional[str]:
        """Check if there's a Ray installation error in the head pod logs.

        Args:
            service_name: Name of the RayCluster service
            head_pod_name: Name of the head pod

        Returns:
            Error message if Ray installation error is found, None otherwise
        """
        try:
            head_logs = self.core_api.read_namespaced_pod_log(
                name=head_pod_name, namespace=self.namespace, tail_lines=100
            )

            # Check for Ray installation errors
            if "ray: not found" in head_logs or "ray: command not found" in head_logs:
                return (
                    f"RayCluster {service_name} failed to start: Ray is not installed in the container. "
                    f"Please use a Ray-enabled image (e.g., rayproject/ray) or ensure Ray is installed in your container setup."
                )

            # Check for Ray startup errors
            if "Failed to start Ray server" in head_logs:
                return (
                    f"RayCluster {service_name} failed to start: Ray server failed to start. "
                    f"Check the head pod logs for more details."
                )

        except client.exceptions.ApiException as e:
            if e.status != 404:  # Pod might not be ready yet
                logger.warning(f"Could not check head pod logs: {e}")

        return None
