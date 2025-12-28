import copy
import os
import time
from typing import List, Optional

import kubetorch.serving.constants as serving_constants
from kubetorch.logger import get_logger
from kubetorch.servers.http.utils import load_template
from kubetorch.serving.base_service_manager import BaseServiceManager
from kubetorch.utils import http_conflict, http_not_found

logger = get_logger(__name__)


class RayClusterServiceManager(BaseServiceManager):
    """Service manager for Ray clusters with distributed Ray workload support."""

    def __init__(
        self,
        namespace: str,
        template_label: str = "raycluster",
        api_group: str = "ray.io",
        api_plural: str = "rayclusters",
        api_version: str = "v1",
        service_annotations: dict = None,
    ):
        # Set Ray-specific default annotations
        default_service_annotations = {
            "ray.io/overwrite-container-cmd": "true",
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

    def get_resource(self, service_name: str) -> dict:
        """Retrieve a RayCluster by name."""
        try:
            raycluster = self.controller_client.get_namespaced_custom_object(
                group="ray.io",
                version="v1",
                namespace=self.namespace,
                plural="rayclusters",
                name=service_name,
            )
            return raycluster
        except Exception as e:
            if http_not_found(e):
                return {}

            logger.error(f"Failed to load RayCluster '{service_name}': {e}")
            raise

    def update_deployment_timestamp_annotation(self, service_name: str, new_timestamp: str) -> str:
        """Update deployment timestamp annotation for RayCluster services."""
        try:
            patch_body = {"metadata": {"annotations": {"kubetorch.com/deployment_timestamp": new_timestamp}}}
            self.controller_client.patch_namespaced_custom_object(
                group="ray.io",
                version="v1",
                namespace=self.namespace,
                plural="rayclusters",
                name=service_name,
                body=patch_body,
            )
            return new_timestamp
        except Exception as e:
            logger.error(f"Failed to update deployment timestamp for RayCluster '{service_name}': {str(e)}")
            raise

    def get_pod_template_path(self) -> List[str]:
        """Get the path to the pod template (head node)."""
        return ["spec", "headGroupSpec", "template"]

    def normalize_created_service(self, created_service) -> dict:
        """Extract service info from RayCluster resource."""
        return {
            "name": created_service.get("metadata", {}).get("name"),
            "namespace": created_service.get("metadata", {}).get("namespace"),
            "template": created_service["spec"]["headGroupSpec"]["template"],
        }

    def get_replicas(self, manifest: dict) -> int:
        """Get the number of replicas from a RayCluster manifest.

        Returns the sum of head replicas and all worker group replicas.
        """
        spec = manifest.get("spec", {})
        head_replicas = spec.get("headGroupSpec", {}).get("replicas", 1)
        worker_groups = spec.get("workerGroupSpecs", [])
        worker_replicas = sum(wg.get("replicas", 0) for wg in worker_groups)
        return head_replicas + worker_replicas

    def set_replicas(self, manifest: dict, value: int) -> None:
        """Set the number of replicas in a RayCluster manifest.

        Sets worker replicas to (value - 1) since head node counts as 1 replica.
        If no worker group exists, creates one based on the head template.
        """
        worker_replicas = max(0, value - 1)  # Head counts as 1
        spec = manifest.setdefault("spec", {})

        if "workerGroupSpecs" in spec and len(spec["workerGroupSpecs"]) > 0:
            spec["workerGroupSpecs"][0]["replicas"] = worker_replicas
        else:
            if "workerGroupSpecs" not in spec:
                spec["workerGroupSpecs"] = []
            if len(spec["workerGroupSpecs"]) == 0:
                # Need to copy head template as base
                head_spec = spec.get("headGroupSpec", {})
                spec["workerGroupSpecs"].append(
                    {
                        "replicas": worker_replicas,
                        "template": head_spec.get("template", {}),
                    }
                )

    @classmethod
    def _convert_manifest(
        cls,
        deployment_manifest: dict,
        namespace: str,
        replicas: int,
    ) -> dict:
        """Convert a deployment manifest to a RayCluster manifest."""
        pod_spec = deployment_manifest["spec"]["template"]["spec"]
        deployment_labels = deployment_manifest["metadata"]["labels"]
        deployment_annotations = deployment_manifest["metadata"]["annotations"]

        labels = deployment_labels.copy()
        labels[serving_constants.KT_TEMPLATE_LABEL] = "raycluster"

        # Template labels (exclude template label - that's only for the top-level resource)
        template_labels = labels.copy()
        template_labels.pop(serving_constants.KT_TEMPLATE_LABEL, None)

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

        # Get base annotations (Ray-specific ones already in deployment_annotations)
        annotations = deployment_annotations.copy()
        default_ray_annotations = {
            "ray.io/overwrite-container-cmd": "true",
        }
        annotations.update(default_ray_annotations)

        # Calculate worker replicas (head node counts as 1 replica)
        worker_replicas = max(0, replicas - 1)

        template_vars = {
            "name": "",  # Will be set during launch
            "namespace": namespace,
            "annotations": annotations,
            "template_annotations": {},  # Will be filled in during launch
            "labels": labels,
            "head_template_labels": head_template_labels,
            "worker_template_labels": worker_template_labels,
            "pod_spec": pod_spec,
            "worker_replicas": worker_replicas,
        }

        raycluster = load_template(
            template_file=serving_constants.RAYCLUSTER_TEMPLATE_FILE,
            template_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"),
            **template_vars,
        )

        return raycluster

    def _apply_template_metadata_updates(
        self,
        manifest: dict,
        template_labels: dict,
        annotations: dict,
        **kwargs,
    ) -> None:
        """Apply RayCluster-specific template metadata updates for head and worker nodes."""
        # Head node specific labels (for service selector)
        head_template_labels = {
            **template_labels,
            "ray.io/node-type": "head",  # KubeRay standard label
        }

        super()._apply_template_metadata_updates(manifest, head_template_labels, annotations, **kwargs)

        # Worker node specific labels
        worker_template_labels = {
            **template_labels,
            "ray.io/node-type": "worker",  # KubeRay standard label
        }

        # Ensure workerGroupSpecs exists and has at least one worker group
        spec = manifest.setdefault("spec", {})
        worker_group_specs = spec.setdefault("workerGroupSpecs", [{}])
        first_worker_group = worker_group_specs[0]
        first_worker_group.setdefault("template", {}).setdefault("metadata", {})
        first_worker_group["template"]["metadata"].setdefault("labels", {}).update(worker_template_labels)
        first_worker_group["template"]["metadata"].setdefault("annotations", {}).update(annotations)

    def _update_launchtime_manifest(
        self, manifest: dict, service_name: str, clean_module_name: str, deployment_timestamp: str, deployment_id: str
    ) -> dict:
        """Update manifest with service name and deployment timestamp."""
        raycluster = super()._update_launchtime_manifest(
            manifest, service_name, clean_module_name, deployment_timestamp, deployment_id
        )

        # Update worker group templates (head is already updated by base class)
        if "spec" in raycluster and "workerGroupSpecs" in raycluster["spec"]:
            for worker_group in raycluster["spec"]["workerGroupSpecs"]:
                if "template" not in worker_group:
                    worker_group["template"] = {}
                metadata = worker_group["template"].setdefault("metadata", {})
                metadata.setdefault("labels", {})[serving_constants.KT_SERVICE_LABEL] = service_name
                metadata["labels"][serving_constants.KT_MODULE_LABEL] = clean_module_name
                metadata.setdefault("annotations", {})["kubetorch.com/deployment_timestamp"] = deployment_timestamp

        return raycluster

    def _preprocess_manifest_for_launch(self, manifest: dict) -> dict:
        """Preprocess RayCluster manifest: sync worker pod specs with head pod spec."""
        # Ensure worker pod specs match head pod spec or sync over any changes
        if "spec" in manifest and "headGroupSpec" in manifest["spec"]:
            head_pod_spec = self.pod_spec(manifest)
            if head_pod_spec and "workerGroupSpecs" in manifest["spec"]:
                # Copy head pod spec to all worker groups
                for worker_group in manifest["spec"]["workerGroupSpecs"]:
                    if "template" not in worker_group:
                        worker_group["template"] = {}
                    worker_group["template"]["spec"] = copy.deepcopy(head_pod_spec)
        return manifest

    def _create_or_update_resource(self, manifest: dict, service_name: str, clean_module_name: str, **kwargs) -> dict:
        raycluster = manifest

        pod_spec = self.pod_spec(raycluster)
        server_port = pod_spec.get("containers", [{}])[0].get("ports", [{}])[0].get("containerPort", 32300)

        labels = raycluster.get("metadata", {}).get("labels", {})
        annotations = raycluster.get("metadata", {}).get("annotations", {})

        # Service labels (exclude kt template label)
        service_labels = labels.copy()
        service_labels.pop(serving_constants.KT_TEMPLATE_LABEL, None)

        try:
            # Create regular service for client access (head node only)
            service = load_template(
                template_file=serving_constants.RAYCLUSTER_SERVICE_TEMPLATE_FILE,
                template_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"),
                name=service_name,
                namespace=self.namespace,
                annotations=annotations,
                labels=service_labels,
                deployment_name=service_name,
                module_name=clean_module_name,
                distributed=False,  # Keep regular service for client access
                server_port=server_port,
            )

            # For regular service, only select head nodes
            service["spec"]["selector"]["ray.io/node-type"] = "head"

            try:
                self.controller_client.create_service(namespace=self.namespace, body=service, params=kwargs)
                if not kwargs.get("dry_run"):
                    logger.info(f"Created service {service_name} in namespace {self.namespace}")
            except Exception as e:
                if http_conflict(e):
                    logger.info(f"Service {service_name} already exists")
                else:
                    raise

            # Create headless service for Ray pod discovery (all nodes)
            headless_service = load_template(
                template_file=serving_constants.RAYCLUSTER_SERVICE_TEMPLATE_FILE,
                template_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"),
                name=f"{service_name}-headless",
                namespace=self.namespace,
                annotations=annotations,
                labels=service_labels,
                deployment_name=service_name,
                module_name=clean_module_name,
                distributed=True,  # Make headless for pod discovery
                server_port=server_port,
            )

            # For headless service, select all Ray nodes (not just head)
            headless_service["spec"]["selector"].pop("ray.io/node-type", None)

            dryrun = kwargs.get("dry_run")
            try:
                self.controller_client.create_service(
                    namespace=self.namespace,
                    body=headless_service,
                )
                if not dryrun:
                    logger.info(f"Created headless service {service_name}-headless in namespace {self.namespace}")
            except Exception as e:
                if http_conflict(e):
                    logger.info(f"Headless service {service_name}-headless already exists")
                else:
                    raise

            # Create RayCluster
            try:
                created_raycluster = self.controller_client.create_namespaced_custom_object(
                    group="ray.io",
                    version="v1",
                    namespace=self.namespace,
                    plural="rayclusters",
                    body=raycluster,
                    params=kwargs,
                )
            except Exception as e:
                if http_not_found(e):
                    logger.error(
                        "RayCluster Custom Resource Definition (CRD) not found, please install the KubeRay operator"
                    )
                raise

            logger.info(f"Created RayCluster {service_name} in namespace {self.namespace}")
            return created_raycluster

        except Exception as e:
            if http_conflict(e):
                logger.info(f"RayCluster {service_name} already exists, updating")
                try:
                    # For RayCluster, we can patch the spec
                    patch_body = {"spec": raycluster["spec"]}
                    updated_raycluster = self.controller_client.patch_namespaced_custom_object(
                        group="ray.io",
                        version="v1",
                        namespace=self.namespace,
                        plural="rayclusters",
                        name=service_name,
                        body=patch_body,
                    )
                    logger.info(f"Updated RayCluster {service_name}")
                    return updated_raycluster
                except Exception as patch_error:
                    logger.error(f"Failed to patch RayCluster {service_name}: {patch_error}")
                    raise patch_error

            raise e

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
                raycluster = self.get_resource(service_name)
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
                running_pods = [pod for pod in pods if pod.get("status", {}).get("phase") == "Running"]

                # Count head and worker pods separately for better logging
                head_pods = [
                    pod
                    for pod in running_pods
                    if pod.get("metadata", {}).get("labels", {}).get("ray.io/node-type") == "head"
                ]
                worker_pods = [
                    pod
                    for pod in running_pods
                    if pod.get("metadata", {}).get("labels", {}).get("ray.io/node-type") == "worker"
                ]

                # Check for specific error conditions
                if head_pods:
                    head_pod = head_pods[0]
                    # Check for Ray installation errors in head pod
                    head_pod_name = head_pod.get("metadata", {}).get("name")
                    ray_error = self._check_ray_installation_error(service_name, head_pod_name)
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
                if not http_not_found(e):
                    logger.error(f"Error checking RayCluster readiness: {e}")

            time.sleep(sleep_interval)

        # Timeout reached
        raise TimeoutError(f"RayCluster {service_name} did not become ready within {launch_timeout} seconds")

    def _teardown_associated_resources(self, service_name: str, console=None) -> bool:
        """Delete associated Kubernetes Services for RayCluster."""
        success = True

        try:
            # Delete the RayCluster
            self.controller_client.delete_namespaced_custom_object(
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

        except Exception as e:
            if http_not_found(e):
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

        # Delete both regular and headless services
        for service_name_to_delete in [service_name, f"{service_name}-headless"]:
            try:
                self.controller_client.delete_service(name=service_name_to_delete, namespace=self.namespace)
                if console:
                    console.print(f"✓ Deleted service [blue]{service_name_to_delete}[/blue]")
                else:
                    logger.info(f"Deleted service {service_name_to_delete}")

            except Exception as e:
                if http_not_found(e):
                    if console:
                        console.print(
                            f"[yellow]Note:[/yellow] Service {service_name_to_delete} not found or already deleted"
                        )
                    else:
                        logger.info(f"Service {service_name_to_delete} not found or already deleted")
                else:
                    if console:
                        console.print(f"[red]Error:[/red] Failed to delete service {service_name_to_delete}: {e}")
                    else:
                        logger.error(f"Failed to delete service {service_name_to_delete}: {e}")
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
            head_logs = self.controller_client.get_pod_logs(
                namespace=self.namespace, name=head_pod_name, tail_lines=100
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

        except Exception as e:
            # Pod might not be ready yet
            if not http_not_found(e):
                logger.warning(f"Could not check head pod logs: {e}")

        return None
