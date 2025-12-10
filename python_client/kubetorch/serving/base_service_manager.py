import hashlib
import re
from abc import abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Type, Union

from kubernetes import client
from kubernetes.client import AppsV1Api, CoreV1Api, CustomObjectsApi

import kubetorch.serving.constants as serving_constants
from kubetorch import globals

from kubetorch.logger import get_logger

logger = get_logger(__name__)


class BaseServiceManager:
    """Base service manager with common functionality for all service types."""

    def __init__(
        self,
        resource_api: Union[AppsV1Api, CustomObjectsApi],
        core_api: CoreV1Api,
        namespace: str,
        template_label: str = "",
        api_group: str = None,
        api_plural: str = None,
        api_version: str = "v1",
        service_annotations: dict = None,
    ):
        self.global_config = globals.config
        self.namespace = namespace or self.global_config.namespace
        self.resource_api = resource_api
        self.core_api = core_api
        self.template_label = template_label or "deployment"
        self.api_group = api_group
        self.api_plural = api_plural
        self.api_version = api_version
        self.service_annotations = service_annotations or {}

    @property
    def username(self):
        return self.global_config.username

    @classmethod
    def _get_labels(
        cls,
        template_label: str,
        custom_labels: dict = None,
    ) -> dict:
        from kubetorch import __version__

        labels = {
            serving_constants.KT_VERSION_LABEL: __version__,
            serving_constants.KT_TEMPLATE_LABEL: template_label,
            serving_constants.KT_USERNAME_LABEL: globals.config.username,
        }

        if custom_labels:
            labels.update(custom_labels)

        return labels

    @classmethod
    def _get_annotations(
        cls,
        service_annotations: dict = None,
        custom_annotations: dict = None,
        inactivity_ttl: str = None,
    ) -> dict:
        annotations = {
            "prometheus.io/scrape": "true",
            "prometheus.io/path": serving_constants.PROMETHEUS_HEALTH_ENDPOINT,
            "prometheus.io/port": "8080",
        }
        if service_annotations:
            annotations.update(service_annotations)

        if custom_annotations:
            annotations.update(custom_annotations)

        if inactivity_ttl:
            annotations[serving_constants.INACTIVITY_TTL_ANNOTATION] = inactivity_ttl
            logger.info(f"Configuring auto-down after idle timeout ({inactivity_ttl})")

        return annotations

    def _apply_kubetorch_updates(
        self,
        manifest: dict,
        inactivity_ttl: str = None,
        custom_labels: dict = None,
        custom_annotations: dict = None,
        custom_template: dict = None,
        **kwargs,  # Allow subclasses to accept additional kwargs (e.g., gpu_annotations)
    ) -> dict:
        """Apply kubetorch labels and annotations to manifest."""
        from kubetorch.serving.utils import nested_override

        # Get base labels and annotations
        labels = self._get_labels(
            template_label=self.template_label,
            custom_labels=custom_labels,
        )
        template_labels = labels.copy()
        template_labels.pop(serving_constants.KT_TEMPLATE_LABEL, None)
        annotations = self._get_annotations(
            service_annotations=self.service_annotations,
            custom_annotations=custom_annotations,
            inactivity_ttl=inactivity_ttl,
        )

        # Update top-level metadata
        manifest["metadata"].setdefault("labels", {}).update(labels)
        manifest["metadata"].setdefault("annotations", {}).update(annotations)

        # Apply service-specific template metadata updates
        self._apply_template_metadata_updates(manifest, template_labels, annotations, **kwargs)

        # Apply custom template overrides
        if custom_template:
            nested_override(manifest, custom_template)

        return manifest

    def _apply_template_metadata_updates(
        self,
        manifest: dict,
        template_labels: dict,
        annotations: dict,
        path: List[str] = None,
        **kwargs,
    ) -> None:
        """Apply template metadata updates. Override in subclasses for service-specific behavior."""
        # Use template_annotations if available (e.g., for Knative), otherwise use annotations
        template_annotations = getattr(self, "template_annotations", None)
        if template_annotations is not None:
            annotations = template_annotations.copy()
            if "gpu_annotations" in kwargs and kwargs["gpu_annotations"]:
                annotations.update(kwargs["gpu_annotations"])

        template_path = path or self._get_pod_template_path()

        # Navigate to template metadata
        current = manifest
        for key in template_path:
            current = current.setdefault(key, {})
        metadata = current.setdefault("metadata", {})
        metadata.setdefault("labels", {}).update(template_labels)
        metadata.setdefault("annotations", {}).update(annotations)

    def _update_launchtime_manifest(
        self, manifest: dict, service_name: str, clean_module_name: str, deployment_timestamp: str, deployment_id: str
    ) -> dict:
        """Update manifest with service name and deployment timestamp."""
        updated_manifest = manifest.copy()

        # Update top-level metadata
        updated_manifest["metadata"]["name"] = service_name
        updated_manifest["metadata"].setdefault("labels", {})
        updated_manifest["metadata"]["labels"][serving_constants.KT_SERVICE_LABEL] = service_name
        updated_manifest["metadata"]["labels"][serving_constants.KT_MODULE_LABEL] = clean_module_name
        updated_manifest["metadata"]["labels"][serving_constants.KT_APP_LABEL] = service_name
        updated_manifest["metadata"]["labels"][serving_constants.KT_DEPLOYMENT_ID_LABEL] = deployment_id

        # Update template metadata
        template_path = self._get_pod_template_path()
        current = updated_manifest
        for key in template_path:
            current = current.setdefault(key, {})
        metadata = current.setdefault("metadata", {})
        metadata.setdefault("labels", {})
        metadata["labels"][serving_constants.KT_SERVICE_LABEL] = service_name
        metadata["labels"][serving_constants.KT_MODULE_LABEL] = clean_module_name
        metadata["labels"][serving_constants.KT_APP_LABEL] = service_name
        metadata["labels"][serving_constants.KT_DEPLOYMENT_ID_LABEL] = deployment_id

        # Update deployment timestamp annotation
        metadata.setdefault("annotations", {})["kubetorch.com/deployment_timestamp"] = deployment_timestamp

        return updated_manifest

    def normalize_created_service(self, created_service) -> dict:
        """Extract service name, namespace, and pod template from created resource.

        Returns normalized dict with structure:
        {
            "name": str,
            "namespace": str,
            "template": dict  # Pod template
        }
        """
        if isinstance(created_service, dict):
            service_name = created_service.get("metadata", {}).get("name")
            namespace = created_service.get("metadata", {}).get("namespace")

            template_path = self._get_pod_template_path()
            current = created_service
            try:
                for key in template_path:
                    current = current[key]
                pod_template = current
            except (KeyError, TypeError):
                raise ValueError("Failed to find pod template in created service.")

            return {
                "name": service_name,
                "namespace": namespace,
                "template": pod_template,
            }
        else:
            # Assume it's a Kubernetes object (V1Deployment, etc.)
            return {
                "name": created_service.metadata.name,
                "namespace": created_service.metadata.namespace,
                "template": created_service.spec.template,
            }

    @staticmethod
    def _get_service_manager_class(kind: str) -> Type["BaseServiceManager"]:
        from kubetorch.serving.service_manager import (
            DeploymentServiceManager,
            KnativeServiceManager,
            RayClusterServiceManager,
            TrainJobServiceManager,
        )

        if kind.lower() == "deployment":
            return DeploymentServiceManager
        elif kind.lower() in ["service", "knative", "ksvc"]:
            return KnativeServiceManager
        elif kind.lower() == "raycluster":
            return RayClusterServiceManager
        elif kind.lower() in [k.lower() for k in TrainJobServiceManager.SUPPORTED_KINDS]:
            return TrainJobServiceManager

    def _get_deployment_timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _generate_deployment_id(self, service_name: str, timestamp: str) -> str:
        """Generate a unique deployment ID from service name + timestamp.

        Returns a string like 'username-myapp-a1b2c3' where the suffix is a
        6-character hash derived from the timestamp.
        """
        hash_input = f"{service_name}-{timestamp}"
        short_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:6]
        return f"{service_name}-{short_hash}"

    def _get_deployment_timestamp_and_id(self, service_name: str) -> Tuple[str, str]:
        """Get both deployment timestamp and deployment ID together.

        Returns:
            Tuple of (timestamp, deployment_id)
        """
        timestamp = self._get_deployment_timestamp()
        deployment_id = self._generate_deployment_id(service_name, timestamp)
        return timestamp, deployment_id

    def _clean_module_name(self, module_name: str) -> str:
        """Clean module name to remove invalid characters for Kubernetes labels."""
        return re.sub(r"[^A-Za-z0-9.-]|^[-.]|[-.]$", "", module_name)

    def get_deployment_timestamp_annotation(self, service_name: str) -> Optional[str]:
        """Get deployment timestamp annotation for any service type."""
        try:
            resource = self.get_resource(service_name)
            if resource:
                # Handle both dict (Knative/RayCluster) and object (Deployment) formats
                if isinstance(resource, dict):  # Knative/RayCluster
                    return (
                        resource.get("metadata", {})
                        .get("annotations", {})
                        .get("kubetorch.com/deployment_timestamp", None)
                    )
                else:  # V1Deployment object
                    return resource.metadata.annotations.get("kubetorch.com/deployment_timestamp", None)
        except client.exceptions.ApiException:
            pass
        return None

    def _create_timestamp_patch_body(self, new_timestamp: str) -> dict:
        """Create the standard patch body for timestamp annotation updates."""
        return {"metadata": {"annotations": {"kubetorch.com/deployment_timestamp": new_timestamp}}}

    def get_resource(self, service_name: str) -> dict:
        """Get a resource by name."""
        try:
            if isinstance(self.resource_api, AppsV1Api):
                # Standard Kubernetes resource (e.g., Deployment)
                return self.resource_api.read_namespaced_deployment(
                    name=service_name,
                    namespace=self.namespace,
                )
            else:
                # Custom resource (e.g., Knative Service, RayCluster)
                return self.resource_api.get_namespaced_custom_object(
                    group=self.api_group,
                    version=self.api_version,
                    namespace=self.namespace,
                    plural=self.api_plural,
                    name=service_name,
                )
        except client.exceptions.ApiException as e:
            logger.error(f"Failed to load {self.template_label} service '{service_name}': {str(e)}")
            raise

    def _get_pod_template_path(self) -> List[str]:
        """Get the path to the pod template in the manifest as a list of keys.

        To get the pod spec, append ["spec"] to this path.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _get_pod_template_path")

    def pod_spec(self, manifest: dict) -> dict:
        """Get the pod spec from a manifest based on the pod template path."""
        template_path = self._get_pod_template_path()
        current = manifest
        for key in template_path:
            current = current.get(key, {})
        return current.get("spec", {})

    def is_distributed(self, manifest: dict) -> bool:
        """Check if this is a distributed job."""
        pod_spec = self.pod_spec(manifest)
        containers = pod_spec.get("containers", [])
        if containers:
            env_vars = containers[0].get("env", [])
            for env_var in env_vars:
                if (
                    env_var.get("name") == "KT_DISTRIBUTED_CONFIG"
                    and env_var.get("value") != "null"
                    and env_var.get("value")
                ):
                    return True

        return False

    @abstractmethod
    def get_replicas(self, manifest: dict) -> int:
        """Get the number of replicas from the manifest."""
        pass

    @abstractmethod
    def set_replicas(self, manifest: dict, value: int) -> None:
        """Set the number of replicas in the manifest."""
        pass

    def update_deployment_timestamp_annotation(self, service_name: str, new_timestamp: str) -> str:
        """Update deployment timestamp annotation for this service type."""
        try:
            patch_body = self._create_timestamp_patch_body(new_timestamp)

            if isinstance(self.resource_api, AppsV1Api):
                self.resource_api.patch_namespaced_deployment(
                    name=service_name,
                    namespace=self.namespace,
                    body=patch_body,
                )
            else:
                self.resource_api.patch_namespaced_custom_object(
                    group=self.api_group,
                    version=self.api_version,
                    namespace=self.namespace,
                    plural=self.api_plural,
                    name=service_name,
                    body=patch_body,
                )
            return new_timestamp
        except client.exceptions.ApiException as e:
            logger.error(
                f"Failed to update deployment timestamp for {self.template_label} service '{service_name}': {str(e)}"
            )
            raise

    def _create_resource(self, manifest: dict, **kwargs) -> dict:
        """Create a resource."""
        if isinstance(self.resource_api, AppsV1Api):
            return self.resource_api.create_namespaced_deployment(
                namespace=self.namespace,
                body=manifest,
                **kwargs,
            )
        else:
            return self.resource_api.create_namespaced_custom_object(
                group=self.api_group,
                version=self.api_version,
                namespace=self.namespace,
                plural=self.api_plural,
                body=manifest,
                **kwargs,
            )

    def _patch_resource(self, service_name: str, patch_body: dict, **kwargs) -> dict:
        """Patch a resource."""
        if isinstance(self.resource_api, AppsV1Api):
            return self.resource_api.patch_namespaced_deployment(
                name=service_name,
                namespace=self.namespace,
                body=patch_body,
                **kwargs,
            )
        else:
            return self.resource_api.patch_namespaced_custom_object(
                group=self.api_group,
                version=self.api_version,
                namespace=self.namespace,
                plural=self.api_plural,
                name=service_name,
                body=patch_body,
                **kwargs,
            )

    def _delete_resource(self, service_name: str, force: bool = False, **kwargs) -> None:
        """Delete a resource."""
        if force:
            kwargs.setdefault("grace_period_seconds", 0)
            kwargs.setdefault("propagation_policy", "Foreground")

        if isinstance(self.resource_api, AppsV1Api):
            self.resource_api.delete_namespaced_deployment(
                name=service_name,
                namespace=self.namespace,
                **kwargs,
            )
        else:
            self.resource_api.delete_namespaced_custom_object(
                group=self.api_group,
                version=self.api_version,
                namespace=self.namespace,
                plural=self.api_plural,
                name=service_name,
                **kwargs,
            )

    def fetch_kubetorch_config(self) -> dict:
        """Fetch the kubetorch configmap from the namespace."""
        try:
            kubetorch_config = self.core_api.read_namespaced_config_map(
                name="kubetorch-config", namespace=globals.config.install_namespace
            )
            return kubetorch_config.data
        except client.exceptions.ApiException as e:
            if e.status != 404:
                logger.error(f"Error fetching kubetorch config: {e}")
            return {}

    @staticmethod
    def discover_services_static(namespace: str, name_filter: str = None) -> List[Dict]:
        """Static method to discover Kubetorch services without ServiceManager instance.

        Uses parallel API calls for faster discovery across service types.

        Args:
            namespace: Kubernetes namespace
            name_filter: Optional name filter for services

        Returns:
            List of service dictionaries with structure:
            {
                'name': str,
                'template_type': str,  # 'ksvc', 'deployment', 'raycluster'
                'resource': dict,
                'namespace': str,
                'creation_timestamp': str  # ISO format
            }
        """
        import concurrent.futures
        import threading

        objects_api = client.CustomObjectsApi()
        apps_v1_api = client.AppsV1Api()

        services = []
        services_lock = threading.Lock()

        def fetch_knative_services():
            """Fetch Knative services in parallel."""
            try:
                label_selector = f"{serving_constants.KT_TEMPLATE_LABEL}=ksvc"
                knative_services = objects_api.list_namespaced_custom_object(
                    group="serving.knative.dev",
                    version="v1",
                    namespace=namespace,
                    plural="services",
                    label_selector=label_selector,
                )["items"]

                local_services = []
                for svc in knative_services:
                    svc_name = svc["metadata"]["name"]
                    if name_filter and name_filter not in svc_name:
                        continue

                    local_services.append(
                        {
                            "name": svc_name,
                            "template_type": "ksvc",
                            "resource": svc,  # Already a dict
                            "namespace": namespace,
                            "creation_timestamp": svc["metadata"]["creationTimestamp"],
                        }
                    )

                with services_lock:
                    services.extend(local_services)

            except client.exceptions.ApiException as e:
                if e.status != 404:  # Ignore if Knative is not installed
                    logger.warning(f"Failed to list Knative services: {e}")

        def fetch_deployments():
            """Fetch Deployments in parallel."""
            try:
                label_selector = f"{serving_constants.KT_TEMPLATE_LABEL}=deployment"
                deployments = apps_v1_api.list_namespaced_deployment(
                    namespace=namespace,
                    label_selector=label_selector,
                )

                local_services = []
                for deployment in deployments.items:
                    deploy_name = deployment.metadata.name
                    if name_filter and name_filter not in deploy_name:
                        continue

                    # Convert V1Deployment object to dictionary for consistency
                    deployment_dict = client.ApiClient().sanitize_for_serialization(deployment)

                    # Add kind and apiVersion (not included in V1Deployment object)
                    deployment_dict["kind"] = "Deployment"
                    deployment_dict["apiVersion"] = "apps/v1"

                    local_services.append(
                        {
                            "name": deploy_name,
                            "template_type": "deployment",
                            "resource": deployment_dict,  # Now consistently a dict
                            "namespace": namespace,
                            "creation_timestamp": deployment.metadata.creation_timestamp.isoformat() + "Z",
                        }
                    )

                with services_lock:
                    services.extend(local_services)

            except client.exceptions.ApiException as e:
                logger.warning(f"Failed to list Deployments: {e}")

        def fetch_rayclusters():
            """Fetch RayClusters in parallel."""
            try:
                label_selector = f"{serving_constants.KT_TEMPLATE_LABEL}=raycluster"
                rayclusters = objects_api.list_namespaced_custom_object(
                    group="ray.io",
                    version="v1",
                    namespace=namespace,
                    plural="rayclusters",
                    label_selector=label_selector,
                )["items"]

                local_services = []
                for cluster in rayclusters:
                    cluster_name = cluster["metadata"]["name"]
                    if name_filter and name_filter not in cluster_name:
                        continue

                    local_services.append(
                        {
                            "name": cluster_name,
                            "template_type": "raycluster",
                            "resource": cluster,  # Already a dict
                            "namespace": namespace,
                            "creation_timestamp": cluster["metadata"]["creationTimestamp"],
                        }
                    )

                with services_lock:
                    services.extend(local_services)

            except client.exceptions.ApiException as e:
                if e.status != 404:
                    logger.warning(f"Failed to list RayClusters: {e}")

        def fetch_custom_resources():
            """Fetch custom training job resources in parallel."""
            from kubetorch.serving.trainjob_service_manager import TrainJobServiceManager

            local_services = []
            for resource_kind in TrainJobServiceManager.SUPPORTED_KINDS:
                config = TrainJobServiceManager._get_config(resource_kind)
                api_group = config["api_group"]
                plural = config["api_plural"]
                version = config["api_version"]
                try:
                    label_selector = f"{serving_constants.KT_TEMPLATE_LABEL}={resource_kind.lower()}"

                    resources = objects_api.list_namespaced_custom_object(
                        group=api_group,
                        version=version,
                        namespace=namespace,
                        plural=plural,
                        label_selector=label_selector,
                    )["items"]

                    for resource in resources:
                        resource_name = resource["metadata"]["name"]
                        if name_filter and name_filter not in resource_name:
                            continue

                        local_services.append(
                            {
                                "name": resource_name,
                                "template_type": resource_kind.lower(),
                                "resource": resource,
                                "namespace": namespace,
                                "creation_timestamp": resource["metadata"]["creationTimestamp"],
                            }
                        )
                except client.exceptions.ApiException as e:
                    if e.status != 404:
                        logger.warning(f"Failed to list {resource_kind}: {e}")

            with services_lock:
                services.extend(local_services)

        # Execute all API calls in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(fetch_knative_services),
                executor.submit(fetch_deployments),
                executor.submit(fetch_rayclusters),
                executor.submit(fetch_custom_resources),
            ]

            # Wait for all to complete
            concurrent.futures.wait(futures)

        return services

    @staticmethod
    def get_pods_for_service_static(
        service_name: str,
        namespace: str,
        core_api=None,
    ) -> List:
        """Static method to get pods for a service across different service types.

        Args:
            service_name: Name of the service
            namespace: Kubernetes namespace
            core_api: Optional CoreV1Api instance (created if None)

        Returns:
            List of pod objects
        """
        if core_api is None:
            core_api = client.CoreV1Api()

        # Build label selector
        label_selector = f"{serving_constants.KT_SERVICE_LABEL}={service_name}"
        try:
            pods = core_api.list_namespaced_pod(namespace=namespace, label_selector=label_selector)
            return pods.items
        except client.exceptions.ApiException as e:
            logger.warning(f"Failed to list pods for service {service_name}: {e}")
            return []

    def discover_all_services(self, namespace: str = None) -> List[Dict]:
        """Discover all Kubetorch services across different resource types.

        Returns a list of service dictionaries with normalized structure:
        {
            'name': str,
            'template_type': str,  # 'ksvc', 'deployment', 'raycluster'
            'resource': object,    # The actual Kubernetes resource object
            'namespace': str
        }
        """
        return self.discover_services_static(namespace=namespace or self.namespace)

    def create_or_update_service(
        self,
        service_name: str,
        module_name: str,
        manifest: dict = None,
        dryrun: bool = False,
        **kwargs,
    ):
        """Create or update service."""
        logger.info(f"Deploying {manifest['kind']} service with name: {service_name}")
        manifest = self._preprocess_manifest_for_launch(manifest)

        # Update manifest with service name and deployment timestamp
        clean_module_name = self._clean_module_name(module_name)
        deployment_timestamp, deployment_id = self._get_deployment_timestamp_and_id(service_name)
        updated_manifest = self._update_launchtime_manifest(
            manifest, service_name, clean_module_name, deployment_timestamp, deployment_id
        )

        # Create or update the resource
        kwargs = {"dry_run": "All"} if dryrun else {}
        created_service = self._create_or_update_resource(updated_manifest, service_name, clean_module_name, **kwargs)
        return created_service, updated_manifest

    def _preprocess_manifest_for_launch(self, manifest: dict) -> dict:
        """Preprocess manifest before launch if needed for the service type."""
        return manifest

    @abstractmethod
    def _create_or_update_resource(self, manifest: dict, service_name: str, clean_module_name: str, **kwargs) -> dict:
        """Create or update resources from a manifest. service_name and clean_module_name are provided to avoid re-extraction."""
        pass

    def get_endpoint(self, service_name: str) -> str:
        raise NotImplementedError("Subclasses must implement get_endpoint")

    def get_pods_for_service(self, service_name: str, **kwargs) -> List[client.V1Pod]:
        """Get all pods associated with this service."""
        label_selector = f"{serving_constants.KT_SERVICE_LABEL}={service_name}"
        try:
            pods = self.core_api.list_namespaced_pod(namespace=self.namespace, label_selector=label_selector)
            return pods.items
        except client.exceptions.ApiException as e:
            logger.warning(f"Failed to list pods for service {service_name}: {e}")
            return []

    def check_service_ready(self, service_name: str, launch_timeout: int, **kwargs) -> bool:
        """Check if service is ready to serve requests.

        This method should be implemented by subclasses to provide service-type-specific
        readiness checking logic.

        Args:
            service_name: Name of the service to check
            launch_timeout: Timeout in seconds to wait for service to be ready
            **kwargs: Additional arguments for readiness checking

        Returns:
            True if service is ready, raises exception if timeout or error
        """
        raise NotImplementedError("Subclasses must implement check_service_ready")

    def teardown_service(self, service_name: str, console=None, force: bool = False) -> bool:
        """Teardown/delete service and associated resources.

        Args:
            service_name: Name of the service to teardown
            console: Optional Rich console for output
            force: Force deletion without graceful shutdown

        Returns:
            True if teardown was successful, False otherwise
        """
        success = True

        # Delete the main resource
        try:
            self._delete_resource(service_name, force=force)
            resource_type = self.template_label.lower() if self.template_label else "resource"
            if console:
                console.print(f"✓ Deleted {resource_type} [blue]{service_name}[/blue]")
            else:
                logger.info(f"Deleted {resource_type} {service_name}")

        except client.exceptions.ApiException as e:
            if e.status == 404:
                resource_type = self.template_label.lower() if self.template_label else "resource"
                if console:
                    console.print(f"[yellow]Note:[/yellow] {resource_type} {service_name} not found or already deleted")
                else:
                    logger.info(f"{resource_type} {service_name} not found or already deleted")
            else:
                resource_type = self.template_label.lower() if self.template_label else "resource"
                if console:
                    console.print(f"[red]Error:[/red] Failed to delete {resource_type} {service_name}: {e}")
                else:
                    logger.error(f"Failed to delete {resource_type} {service_name}: {e}")
                success = False

        # Delete associated resources
        associated_success = self._teardown_associated_resources(service_name, console)
        return success and associated_success

    def _teardown_associated_resources(self, service_name: str, console=None) -> bool:
        """Teardown associated resources (e.g., Kubernetes Services).

        Returns:
            True if all associated resources were deleted successfully, False otherwise
        """
        return True
