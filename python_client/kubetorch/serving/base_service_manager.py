import hashlib
import importlib
import re
from abc import abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import yaml
from jinja2 import Template

from kubernetes import client, utils
from kubernetes.client import AppsV1Api, CoreV1Api, CustomObjectsApi

import kubetorch.serving.constants as serving_constants
from kubetorch import globals

from kubetorch.logger import get_logger

logger = get_logger(__name__)


class BaseServiceManager:
    """Base service manager with common functionality for all service types."""

    def __init__(
        self,
        objects_api: CustomObjectsApi,
        core_api: CoreV1Api,
        apps_v1_api: AppsV1Api,
        namespace: str,
    ):
        self.objects_api = objects_api
        self.core_api = core_api
        self.apps_v1_api = apps_v1_api

        # Load config
        self.global_config = globals.config
        self.namespace = namespace or self.global_config.namespace

    @property
    def username(self):
        return self.global_config.username

    @staticmethod
    def _get_labels(
        template_label: str,
        custom_labels: dict = None,
        scheduler_name: str = None,
        queue_name: str = None,
    ) -> dict:
        from kubetorch import __version__

        labels = {
            serving_constants.KT_VERSION_LABEL: __version__,
            serving_constants.KT_TEMPLATE_LABEL: template_label,
            serving_constants.KT_USERNAME_LABEL: globals.config.username,
        }

        if custom_labels:
            labels.update(custom_labels)

        if scheduler_name and queue_name:
            labels["kai.scheduler/queue"] = queue_name

        return labels

    @staticmethod
    def _get_annotations(custom_annotations: dict = None, inactivity_ttl: str = None) -> dict:
        annotations = {
            "prometheus.io/scrape": "true",
            "prometheus.io/path": serving_constants.PROMETHEUS_HEALTH_ENDPOINT,
            "prometheus.io/port": "8080",
        }
        if custom_annotations:
            annotations.update(custom_annotations)

        if inactivity_ttl:
            annotations[serving_constants.INACTIVITY_TTL_ANNOTATION] = inactivity_ttl
            logger.info(f"Configuring auto-down after idle timeout ({inactivity_ttl})")

        return annotations

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

    def _apply_yaml_template(self, yaml_file, replace_existing=False, **kwargs):
        with importlib.resources.files("kubetorch.serving.templates").joinpath(yaml_file).open("r") as f:
            template = Template(f.read())

        yaml_content = template.render(**kwargs)
        yaml_objects = list(yaml.safe_load_all(yaml_content))
        k8s_client = client.ApiClient()

        for obj in yaml_objects:
            logger.info(f"Applying {obj.get('kind')}/{obj.get('metadata', {}).get('name')}")
            try:
                if replace_existing:
                    # Try to delete existing resource first
                    try:
                        utils.delete_from_dict(k8s_client, obj)
                    except client.exceptions.ApiException as e:
                        if e.status != 404:  # Ignore if resource doesn't exist
                            raise

                utils.create_from_dict(k8s_client, obj)
                logger.info(f"Successfully applied {obj.get('kind')}/{obj.get('metadata', {}).get('name')}")
            except utils.FailToCreateError as e:
                if "already exists" in str(e):
                    logger.info(f"Resource already exists: {obj.get('kind')}/{obj.get('metadata', {}).get('name')}")
                else:
                    raise

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

    @abstractmethod
    def update_deployment_timestamp_annotation(self, service_name: str, new_timestamp: str) -> str:
        """Update deployment timestamp annotation for this service type."""
        pass

    def _create_timestamp_patch_body(self, new_timestamp: str) -> dict:
        """Create the standard patch body for timestamp annotation updates."""
        return {"metadata": {"annotations": {"kubetorch.com/deployment_timestamp": new_timestamp}}}

    @abstractmethod
    def get_resource(self, service_name: str) -> dict:
        """Get a resource by name. Must be implemented by subclasses."""
        pass

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
    def discover_services_static(
        namespace: str, objects_api=None, apps_v1_api=None, name_filter: str = None
    ) -> List[Dict]:
        """Static method to discover Kubetorch services without ServiceManager instance.

        Uses parallel API calls for faster discovery across service types.

        Args:
            namespace: Kubernetes namespace
            objects_api: Optional CustomObjectsApi instance (created if None)
            apps_v1_api: Optional AppsV1Api instance (created if None)
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

        if objects_api is None:
            objects_api = client.CustomObjectsApi()
        if apps_v1_api is None:
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

        # Execute all API calls in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(fetch_knative_services),
                executor.submit(fetch_deployments),
                executor.submit(fetch_rayclusters),
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
        return self.discover_services_static(
            namespace=namespace or self.namespace,
            objects_api=self.objects_api,
            apps_v1_api=self.apps_v1_api,
        )

    # Abstract methods to be implemented by subclasses
    def create_or_update_service(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement create_or_update_service")

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

    def teardown_service(self, service_name: str, console=None) -> bool:
        """Teardown/delete service and associated resources.

        This method should be implemented by subclasses to provide service-type-specific
        teardown logic.

        Args:
            service_name: Name of the service to teardown
            console: Optional Rich console for output

        Returns:
            True if teardown was successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement teardown_service")
