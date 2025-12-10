import hashlib
import importlib
import re
from abc import abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import yaml
from jinja2 import Template

from kubernetes import client, utils

import kubetorch.serving.constants as serving_constants
from kubetorch import globals

from kubetorch.logger import get_logger
from kubetorch.utils import http_not_found

logger = get_logger(__name__)


class BaseServiceManager:
    """Base service manager with common functionality for all service types."""

    def __init__(
        self,
        namespace: str,
    ):
        self.namespace = namespace or globals.config.namespace

    @property
    def controller_client(self):
        """Get the global controller client instance."""
        return globals.controller_client()

    @property
    def username(self):
        return globals.config.username

    @staticmethod
    def _get_labels(
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

    @staticmethod
    def _create_timestamp_patch_body(new_timestamp: str) -> dict:
        """
        Create a patch body updating only the deployment timestamp annotation.
        Used when reloading code without recreating the Deployment.
        """
        return {
            "spec": {"template": {"metadata": {"annotations": {"kubetorch.com/deployment_timestamp": new_timestamp}}}}
        }

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
            try:
                if replace_existing:
                    try:
                        utils.delete_from_dict(k8s_client, obj)
                    except client.exceptions.ApiException as e:
                        if not http_not_found(e):
                            raise
                utils.create_from_dict(k8s_client, obj)
            except utils.FailToCreateError as e:
                if "already exists" not in str(e):
                    raise

    def get_deployment_timestamp_annotation(self, service_name: str) -> Optional[str]:
        """Get deployment timestamp annotation for any service type."""
        try:
            resource = self.get_resource(service_name)
            if resource:
                if isinstance(resource, dict):
                    return resource.get("metadata", {}).get("annotations", {}).get("kubetorch.com/deployment_timestamp")
                else:
                    return None
        except Exception:
            return None

    @abstractmethod
    def update_deployment_timestamp_annotation(self, service_name: str, new_timestamp: str) -> str:
        """Update deployment timestamp annotation for this service type."""
        pass

    @abstractmethod
    def get_resource(self, service_name: str) -> dict:
        """Get a resource by name via controller or CRDs."""
        pass

    def fetch_kubetorch_config(self) -> dict:
        """Fetch kubetorch-config ConfigMap via controller."""
        try:
            kubetorch_config = self.controller_client.get(
                f"/api/v1/namespaces/{globals.config.install_namespace}/configmaps/kubetorch-config"
            )
            return kubetorch_config.get("data", {})
        except Exception as e:
            if not http_not_found(e):
                logger.error(f"Kubeconfig not found: {e}")
            return {}

    @staticmethod
    def discover_services_static(namespace: str, name_filter: str = None) -> List[Dict]:
        """Discover Knative services, RayClusters, and Deployments via controller client."""

        import concurrent.futures
        import threading

        controller_client = globals.controller_client()

        services = []
        services_lock = threading.Lock()

        def fetch_knative_services():
            """Fetch Knative services in parallel."""
            try:
                label_selector = f"{serving_constants.KT_TEMPLATE_LABEL}=ksvc"
                result = controller_client.list_namespaced_custom_object(
                    group="serving.knative.dev",
                    version="v1",
                    namespace=namespace,
                    plural="services",
                    label_selector=label_selector,
                )
                knative_services = result.get("items", [])

                local_services = []
                for svc in knative_services:
                    svc_name = svc["metadata"]["name"]
                    if name_filter and name_filter not in svc_name:
                        continue
                    local_services.append(
                        {
                            "name": svc_name,
                            "template_type": "ksvc",
                            "resource": svc,
                            "namespace": namespace,
                            "creation_timestamp": svc["metadata"]["creationTimestamp"],
                        }
                    )

                with services_lock:
                    services.extend(local_services)

            except Exception as e:
                if not http_not_found(e):
                    logger.warning(f"Failed to list Knative services: {e}")

        def fetch_deployments():
            try:
                label_selector = f"{serving_constants.KT_TEMPLATE_LABEL}=deployment"
                result = controller_client.list_deployments(
                    namespace=namespace,
                    label_selector=label_selector,
                )
                deployments = result.get("items", [])

                local_services = []
                for dep in deployments:
                    name = dep.get("metadata", {}).get("name")
                    if name_filter and name_filter not in name:
                        continue

                    creation_timestamp = dep.get("metadata", {}).get("creationTimestamp", "")

                    local_services.append(
                        {
                            "name": name,
                            "template_type": "deployment",
                            "resource": dep,
                            "namespace": namespace,
                            "creation_timestamp": creation_timestamp,
                        }
                    )

                with services_lock:
                    services.extend(local_services)

            except Exception as e:
                logger.warning(f"Failed to list Deployments: {e}")

        def fetch_rayclusters():
            try:
                label_selector = f"{serving_constants.KT_TEMPLATE_LABEL}=raycluster"
                result = controller_client.list_namespaced_custom_object(
                    group="ray.io",
                    version="v1",
                    namespace=namespace,
                    plural="rayclusters",
                    label_selector=label_selector,
                )
                clusters = result.get("items", [])

                local_services = []
                for rc in clusters:
                    name = rc["metadata"]["name"]
                    if name_filter and name_filter not in name:
                        continue

                    local_services.append(
                        {
                            "name": name,
                            "template_type": "raycluster",
                            "resource": rc,
                            "namespace": namespace,
                            "creation_timestamp": rc["metadata"]["creationTimestamp"],
                        }
                    )

                with services_lock:
                    services.extend(local_services)

            except Exception as e:
                if not http_not_found(e):
                    logger.warning(f"Failed to list RayClusters: {e}")

        # Run all in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            executor.submit(fetch_knative_services)
            executor.submit(fetch_deployments)
            executor.submit(fetch_rayclusters)

        return services

    def discover_all_services(self, namespace: str = None) -> List[Dict]:
        return self.discover_services_static(
            namespace=namespace or self.namespace,
        )

    # ----------------------------------------------------------------------
    # PODS (ControllerClient)
    # ----------------------------------------------------------------------
    def normalize_pod(self, pod):
        """Convert pod to dict if needed."""
        if isinstance(pod, dict):
            return pod
        else:
            return client.ApiClient().sanitize_for_serialization(pod)

    def get_pods_for_service(self, service_name: str, **kwargs) -> List[dict]:
        """Get all pods associated with this service."""
        label_selector = f"{serving_constants.KT_SERVICE_LABEL}={service_name}"

        try:
            raw = self.controller_client.list_pods(self.namespace, label_selector=label_selector)
            items = raw.get("items", [])
            normalized = [self.normalize_pod(pod) for pod in items]
            return normalized

        except Exception as e:
            logger.warning(f"Failed to list pods: {e}")
            return []

    # Abstract API (implemented in subclasses)
    def create_or_update_service(self, *args, **kwargs):
        raise NotImplementedError

    def get_endpoint(self, service_name: str) -> str:
        raise NotImplementedError

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
