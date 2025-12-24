"""Unified service manager for all K8s resource types.

This module provides a single ServiceManager class that handles all resource types
(Deployment, Knative, RayCluster, training jobs, etc.) through configuration.
Resource-specific behavior is driven by its relevant config.
"""
import copy
import hashlib
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import kubetorch.serving.constants as serving_constants
from kubetorch import globals
from kubetorch.logger import get_logger
from kubetorch.resources.compute.endpoint import Endpoint
from kubetorch.serving.utils import get_resource_config, SUPPORTED_TRAINING_JOBS
from kubetorch.utils import http_conflict, http_not_found

logger = get_logger(__name__)


class ServiceManager:
    """Unified service manager for all K8s resource types.

    Uses configuration-driven behavior instead of inheritance.
    Resource-specific logic is handled via `RESOURCE_CONFIGS`.
    """

    def __init__(
        self,
        resource_type: str,
        namespace: str,
        service_annotations: dict = None,
    ):
        """Initialize the service manager.

        Args:
            resource_type: Type of resource (deployment, knative, raycluster, pytorchjob, etc.)
            namespace: Kubernetes namespace
            service_annotations: Optional annotations to apply to services
        """
        self.resource_type = resource_type.lower()
        self.namespace = namespace or globals.config.namespace
        self.service_annotations = service_annotations or {}
        self.config = get_resource_config(self.resource_type)

        # API config from resource config
        self.api_group = self.config.get("api_group")
        self.api_version = self.config.get("api_version", "v1")
        self.api_plural = self.config.get("api_plural")
        self.template_label = self.config.get("template_label", self.resource_type)

    @property
    def controller_client(self):
        """Get the global controller client instance."""
        return globals.controller_client()

    # =========================================================================
    # Manifest Navigation (config-driven)
    # =========================================================================

    def get_pod_template_path(self) -> List[str]:
        """Get the path to the pod template in the manifest."""
        path = self.config.get("pod_template_path")
        if path is None:
            raise ValueError(f"Resource type {self.resource_type} has no pod template path (selector-only?)")
        return path

    def pod_spec(self, manifest: dict) -> dict:
        """Get the pod spec from a manifest."""
        template_path = self.get_pod_template_path()
        current = manifest
        for key in template_path:
            current = current.get(key, {})
        return current.get("spec", {})

    # =========================================================================
    # Replicas (config-driven with type-specific logic)
    # =========================================================================

    def get_replicas(self, manifest: dict) -> int:
        """Get the number of replicas from the manifest."""
        if self.resource_type == "knative":
            return self._get_knative_replicas(manifest)
        elif self.resource_type == "raycluster":
            return self._get_raycluster_replicas(manifest)
        elif self.resource_type in SUPPORTED_TRAINING_JOBS:
            return self._get_trainjob_replicas(manifest)
        else:
            # Standard deployment path
            return manifest.get("spec", {}).get("replicas", 1)

    def set_replicas(self, manifest: dict, value: int, distributed_config: dict = None) -> None:
        """Set the number of replicas in the manifest."""
        if self.resource_type == "knative":
            self._set_knative_replicas(manifest, value)
        elif self.resource_type == "raycluster":
            self._set_raycluster_replicas(manifest, value)
        elif self.resource_type in SUPPORTED_TRAINING_JOBS:
            self._set_trainjob_replicas(manifest, value, distributed_config)
        else:
            # Standard deployment path
            manifest.setdefault("spec", {})["replicas"] = value

    def _get_knative_replicas(self, manifest: dict) -> int:
        """Get min-scale from Knative annotations."""
        annotations = manifest.get("spec", {}).get("template", {}).get("metadata", {}).get("annotations", {})
        min_scale = annotations.get("autoscaling.knative.dev/min-scale", "0")
        return int(min_scale) if min_scale else 0

    def _set_knative_replicas(self, manifest: dict, value: int) -> None:
        """Set min-scale in Knative annotations."""
        spec = manifest.setdefault("spec", {})
        template = spec.setdefault("template", {})
        metadata = template.setdefault("metadata", {})
        annotations = metadata.setdefault("annotations", {})
        annotations["autoscaling.knative.dev/min-scale"] = str(value)

    def _get_raycluster_replicas(self, manifest: dict) -> int:
        """Get total replicas (head + workers) from RayCluster."""
        spec = manifest.get("spec", {})
        head_replicas = spec.get("headGroupSpec", {}).get("replicas", 1)
        worker_groups = spec.get("workerGroupSpecs", [])
        worker_replicas = sum(wg.get("replicas", 0) for wg in worker_groups)
        return head_replicas + worker_replicas

    def _set_raycluster_replicas(self, manifest: dict, value: int) -> None:
        """Set worker replicas in RayCluster (head is always 1)."""
        worker_replicas = max(0, value - 1)
        spec = manifest.setdefault("spec", {})

        if "workerGroupSpecs" in spec and len(spec["workerGroupSpecs"]) > 0:
            spec["workerGroupSpecs"][0]["replicas"] = worker_replicas
        else:
            if "workerGroupSpecs" not in spec:
                spec["workerGroupSpecs"] = []
            if len(spec["workerGroupSpecs"]) == 0:
                head_spec = spec.get("headGroupSpec", {})
                spec["workerGroupSpecs"].append(
                    {
                        "replicas": worker_replicas,
                        "template": head_spec.get("template", {}),
                    }
                )

    def _get_trainjob_replicas(self, manifest: dict) -> int:
        """Get total replicas from training job."""
        specs_key = self.config.get("replica_specs_key")
        if not specs_key:
            return 1
        spec = manifest.get("spec", {})
        replica_specs = spec.get(specs_key, {})
        return sum(rs.get("replicas", 0) for rs in replica_specs.values() if isinstance(rs, dict))

    def _set_trainjob_replicas(self, manifest: dict, value: int, distributed_config: dict = None) -> None:
        """Set replicas in training job (primary=1, rest are workers)."""
        specs_key = self.config.get("replica_specs_key")
        primary_replica = self.config.get("primary_replica")
        if not specs_key or not primary_replica:
            return

        worker_replicas = max(0, value - 1)
        spec = manifest.setdefault("spec", {})
        replica_specs = spec.setdefault(specs_key, {})

        # Primary always has 1 replica
        primary_spec = replica_specs.setdefault(primary_replica, {})
        primary_spec["replicas"] = 1

        # Workers get the rest
        worker_spec = replica_specs.setdefault("Worker", {})
        worker_spec["replicas"] = worker_replicas

        if "template" not in worker_spec:
            primary_template = primary_spec.get("template", {})
            worker_spec["template"] = copy.deepcopy(primary_template)

        # Update distributed config if provided
        if distributed_config is not None:
            import json

            env_value = json.dumps(distributed_config)
            for replica_name in [primary_replica, "Worker"]:
                rs = replica_specs.get(replica_name, {})
                pod_spec = rs.get("template", {}).get("spec", {})
                containers = pod_spec.get("containers", [])
                for container in containers:
                    env_list = container.setdefault("env", [])
                    updated = False
                    for env_var in env_list:
                        if env_var.get("name") == "KT_DISTRIBUTED_CONFIG":
                            env_var["value"] = env_value
                            updated = True
                            break
                    if not updated:
                        env_list.append({"name": "KT_DISTRIBUTED_CONFIG", "value": env_value})

    # =========================================================================
    # Labels and Annotations
    # =========================================================================

    def _get_labels(self, custom_labels: dict = None) -> dict:
        """Get standard kubetorch labels."""
        from kubetorch import __version__

        labels = {
            serving_constants.KT_VERSION_LABEL: __version__,
            serving_constants.KT_TEMPLATE_LABEL: self.template_label,
            serving_constants.KT_USERNAME_LABEL: globals.config.username,
        }
        if custom_labels:
            labels.update(custom_labels)
        return labels

    def _get_annotations(
        self,
        custom_annotations: dict = None,
        inactivity_ttl: str = None,
    ) -> dict:
        """Get standard kubetorch annotations."""
        annotations = {
            "prometheus.io/scrape": "true",
            "prometheus.io/path": serving_constants.PROMETHEUS_HEALTH_ENDPOINT,
            "prometheus.io/port": "8080",
        }
        if self.service_annotations:
            annotations.update(self.service_annotations)
        if custom_annotations:
            annotations.update(custom_annotations)
        if inactivity_ttl:
            annotations[serving_constants.INACTIVITY_TTL_ANNOTATION] = inactivity_ttl
        return annotations

    # =========================================================================
    # Manifest Updates
    # =========================================================================

    def _apply_kubetorch_metadata_to_manifest(
        self,
        manifest: dict,
        inactivity_ttl: str = None,
        custom_labels: dict = None,
        custom_annotations: dict = None,
        custom_template: dict = None,
        **kwargs,
    ) -> dict:
        """Apply kubetorch labels and annotations to manifest."""
        from kubetorch.serving.utils import nested_override

        labels = self._get_labels(custom_labels)
        template_labels = labels.copy()
        template_labels.pop(serving_constants.KT_TEMPLATE_LABEL, None)
        annotations = self._get_annotations(custom_annotations, inactivity_ttl)

        # Update top-level metadata
        manifest["metadata"].setdefault("labels", {}).update(labels)
        manifest["metadata"].setdefault("annotations", {}).update(annotations)

        # Apply template metadata updates
        if self.config.get("pod_template_path"):
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
    ) -> None:
        """Apply template metadata updates."""
        template_path = path or self.get_pod_template_path()

        # Navigate to template metadata, handling both dict keys and list indices
        current = manifest
        for key in template_path:
            if isinstance(key, int):
                if key < len(current):
                    current = current[key]
                else:
                    return  # nothing to update
            else:
                current = current.setdefault(key, {})
        metadata = current.setdefault("metadata", {})
        metadata.setdefault("labels", {}).update(template_labels)
        metadata.setdefault("annotations", {}).update(annotations)

        # For RayCluster and training jobs, also update worker templates
        worker_path = self.config.get("worker_template_path")
        if worker_path and path is None:  # Don't recurse
            self._apply_template_metadata_updates(manifest, template_labels, annotations, path=worker_path)

    def _update_launchtime_manifest(
        self,
        manifest: dict,
        service_name: str,
        clean_module_name: str,
        deployment_timestamp: str,
        deployment_id: str,
    ) -> dict:
        """Update manifest with service name and deployment timestamp."""
        updated = copy.deepcopy(manifest)

        # Update top-level metadata
        updated["metadata"]["name"] = service_name
        updated["metadata"].setdefault("labels", {})
        updated["metadata"]["labels"][serving_constants.KT_SERVICE_LABEL] = service_name
        updated["metadata"]["labels"][serving_constants.KT_MODULE_LABEL] = clean_module_name
        updated["metadata"]["labels"][serving_constants.KT_APP_LABEL] = service_name
        updated["metadata"]["labels"][serving_constants.KT_DEPLOYMENT_ID_LABEL] = deployment_id

        # For Deployments, update selector.matchLabels to match template labels
        if self.resource_type == "deployment":
            updated["spec"].setdefault("selector", {}).setdefault("matchLabels", {})
            updated["spec"]["selector"]["matchLabels"][serving_constants.KT_SERVICE_LABEL] = service_name
            updated["spec"]["selector"]["matchLabels"][serving_constants.KT_MODULE_LABEL] = clean_module_name

        # Update template metadata
        if self.config.get("pod_template_path"):
            template_path = self.get_pod_template_path()
            current = updated
            for key in template_path:
                if isinstance(key, int):
                    if key < len(current):
                        current = current[key]
                    else:
                        current = None
                        break
                else:
                    current = current.setdefault(key, {})
            if current is None:
                return updated
            metadata = current.setdefault("metadata", {})
            metadata.setdefault("labels", {})[serving_constants.KT_SERVICE_LABEL] = service_name
            metadata["labels"][serving_constants.KT_MODULE_LABEL] = clean_module_name
            metadata["labels"][serving_constants.KT_APP_LABEL] = service_name
            metadata["labels"][serving_constants.KT_DEPLOYMENT_ID_LABEL] = deployment_id
            metadata.setdefault("annotations", {})["kubetorch.com/deployment_timestamp"] = deployment_timestamp

            # Also update worker template for distributed resources
            worker_path = self.config.get("worker_template_path")
            if worker_path:
                current = updated
                for key in worker_path:
                    if isinstance(key, int):
                        if key < len(current):
                            current = current[key]
                        else:
                            break
                    else:
                        current = current.get(key, {})
                if current:
                    metadata = current.setdefault("metadata", {})
                    metadata.setdefault("labels", {})[serving_constants.KT_SERVICE_LABEL] = service_name
                    metadata["labels"][serving_constants.KT_MODULE_LABEL] = clean_module_name
                    metadata.setdefault("annotations", {})["kubetorch.com/deployment_timestamp"] = deployment_timestamp

        return updated

    # =========================================================================
    # Service Config / Endpoint Resolution
    # =========================================================================

    def _resolve_service_config(
        self, endpoint: Optional[Endpoint], service_name: str, pool_selector: dict
    ) -> Optional[dict]:
        """Resolve service config from endpoint or use resource-specific default."""
        if endpoint:
            return endpoint.to_service_config()

        default_routing = self.config.get("default_routing")
        if default_routing is None:
            # Use pool selector for service routing
            return None
        elif default_routing == "knative_url":
            # Knative provides its own URL
            url = f"http://{service_name}.{self.namespace}.svc.cluster.local"
            return {"type": "url", "url": url}
        elif isinstance(default_routing, dict):
            # Merge pool selector with default routing labels
            service_selector = {**pool_selector, **default_routing}
            return {"type": "selector", "selector": service_selector}
        else:
            return None

    # =========================================================================
    # Core Operations
    # =========================================================================

    def create_or_update_service(
        self,
        service_name: str,
        module_name: str,
        manifest: dict = None,
        deployment_timestamp: str = None,
        dryrun: bool = False,
        dockerfile: str = None,
        module: dict = None,
        create_headless_service: bool = False,
        endpoint: Optional[Endpoint] = None,
        pod_selector: Optional[Dict[str, str]] = None,
    ) -> Tuple[dict, dict]:
        """Create or update a Kubernetes service and register it with the controller.

        Deploys the provided manifest to the cluster and registers the resource with the
        kubetorch controller for pod tracking and routing.

        Args:
            service_name (str): Name of the service to create or update.
            module_name (str): Name of the module being deployed.
            manifest (dict, optional): Kubernetes manifest dict defining the workload.
            deployment_timestamp (str, optional): ISO timestamp for the deployment.
                If not provided, current time is used.
            dryrun (bool, optional): If True, skip actual deployment and resource registration.
                Defaults to False.
            dockerfile (str, optional): Dockerfile content for image building.
            module (dict, optional): Module metadata to store with the resource registration.
            create_headless_service (bool, optional): If `True`, create a headless K8s Service
                for distributed pod discovery. Defaults to `False`.
            endpoint (Endpoint, optional): Custom endpoint configuration for routing.
                Use for custom URLs or subset routing.
            pod_selector (Dict[str, str], optional): Custom label selector for resource
                registration. If provided, the controller watches pods matching this
                selector instead of the default kubetorch labels. Used with ``from_manifest()``
                when providing a custom selector with an existing K8s manifest.

        Returns:
            Tuple[dict, dict]: A tuple of (created_service, updated_manifest) where
                created_service is the K8s resource returned by the controller and
                updated_manifest is the manifest with applied labels and timestamps.
        """
        logger.info(f"Deploying {manifest.get('kind', self.resource_type)} service with name: {service_name}")

        # Preprocess manifest
        manifest = self._preprocess_manifest_for_launch(manifest)

        # Update manifest with service name and deployment timestamp
        clean_module_name = self._clean_module_name(module_name)
        timestamp, deployment_id = self._get_deployment_timestamp_and_id(service_name, deployment_timestamp)
        updated_manifest = self._update_launchtime_manifest(
            manifest, service_name, clean_module_name, timestamp, deployment_id
        )

        # Create or update the resource with the controller
        created_service = self._apply_and_register_pool(
            manifest=updated_manifest,
            service_name=service_name,
            clean_module_name=clean_module_name,
            dry_run=dryrun,
            dockerfile=dockerfile,
            module=module,
            create_headless_service=create_headless_service,
            endpoint=endpoint,
            pod_selector=pod_selector,
        )

        return created_service, updated_manifest

    def _preprocess_manifest_for_launch(self, manifest: dict) -> dict:
        """Preprocess manifest before launch."""
        # Sync worker pod specs with primary for distributed resources
        if self.resource_type == "raycluster":
            return self._preprocess_raycluster_manifest(manifest)
        elif self.resource_type in SUPPORTED_TRAINING_JOBS:
            return self._preprocess_trainjob_manifest(manifest)
        elif self.resource_type == "deployment":
            return self._preprocess_deployment_manifest(manifest)
        return manifest

    def _preprocess_deployment_manifest(self, manifest: dict) -> dict:
        """Preprocess deployment manifest - add Recreate strategy for multi-replica deployments."""
        replicas = manifest.get("spec", {}).get("replicas", 1)
        if replicas > 1:
            # Use Recreate strategy for multi-replica deployments to avoid
            # mixed old/new pods during updates, which causes SIGTERM errors
            # when traffic is routed to terminating pods
            manifest.setdefault("spec", {})["strategy"] = {"type": "Recreate"}
        return manifest

    def _preprocess_raycluster_manifest(self, manifest: dict) -> dict:
        """Sync worker pod specs with head pod spec for RayCluster."""
        if "spec" in manifest and "headGroupSpec" in manifest["spec"]:
            head_pod_spec = self.pod_spec(manifest)
            if head_pod_spec and "workerGroupSpecs" in manifest["spec"]:
                for worker_group in manifest["spec"]["workerGroupSpecs"]:
                    if "template" not in worker_group:
                        worker_group["template"] = {}
                    worker_group["template"]["spec"] = copy.deepcopy(head_pod_spec)
        return manifest

    def _preprocess_trainjob_manifest(self, manifest: dict) -> dict:
        """Sync worker pod spec with primary for training jobs."""
        specs_key = self.config.get("replica_specs_key")
        primary_replica = self.config.get("primary_replica")
        if not specs_key or not primary_replica:
            return manifest

        # Set job mode for MXJob
        if manifest.get("kind") == "MXJob":
            manifest.setdefault("spec", {}).setdefault("jobMode", "Train")

        # Sync worker spec if distributed
        if self._is_distributed(manifest):
            spec = manifest.get("spec", {})
            replica_specs = spec.get(specs_key, {})
            primary_pod_spec = self.pod_spec(manifest)

            if primary_pod_spec:
                worker_spec = replica_specs.setdefault("Worker", {}).setdefault("template", {})
                worker_spec["spec"] = copy.deepcopy(primary_pod_spec)

        return manifest

    def _is_distributed(self, manifest: dict) -> bool:
        """Check if this is a distributed job."""
        # Check KT_DISTRIBUTED_CONFIG env var
        pod_spec = self.pod_spec(manifest)
        containers = pod_spec.get("containers", [])
        if containers:
            env_vars = containers[0].get("env", [])
            for env_var in env_vars:
                if (
                    env_var.get("name") == "KT_DISTRIBUTED_CONFIG"
                    and env_var.get("value")
                    and env_var.get("value") != "null"
                ):
                    return True

        # Check replicas for training jobs
        if self.resource_type in SUPPORTED_TRAINING_JOBS:
            return self.get_replicas(manifest) > 1

        return False

    def _load_pool_metadata(self):
        return {"username": globals.config.username}

    def _apply_and_register_pool(
        self,
        manifest: dict,
        service_name: str,
        clean_module_name: str,
        dry_run: bool = False,
        dockerfile: str = None,
        module: dict = None,
        create_headless_service: bool = False,
        endpoint: Optional[Endpoint] = None,
        pod_selector: Optional[Dict[str, str]] = None,
    ) -> dict:
        """Create or update resource via controller. First applies the manifest to create the relevant pod(s), then
        registers the resource with the kubetorch controller."""
        pod_spec = self.pod_spec(manifest)
        server_port = pod_spec.get("containers", [{}])[0].get("ports", [{}])[0].get("containerPort", 32300)

        labels = manifest.get("metadata", {}).get("labels", {})
        annotations = manifest.get("metadata", {}).get("annotations", {})

        try:
            # Step 1: Apply the manifest via controller to create k8s resources
            manifest_replicas = manifest.get("spec", {}).get("replicas", "NOT_SET")
            logger.debug(f"Applying manifest for {service_name}: replicas={manifest_replicas}")
            apply_response = self.controller_client.apply(
                service_name=service_name,
                namespace=self.namespace,
                resource_type=self.resource_type,
                resource_manifest=manifest,
            )

            if apply_response.get("status") == "error":
                raise Exception(f"Apply failed: {apply_response.get('message')}")

            logger.info(
                f"Applied {manifest.get('kind', self.resource_type)} {service_name} in namespace {self.namespace}"
            )

            # Step 2: Register resource with the controller
            if not dry_run:
                # Use custom pod_selector if provided, otherwise use KT labels for pods created by kubetorch
                if pod_selector:
                    pool_selector_for_specifier = pod_selector
                else:
                    pool_selector_for_specifier = {
                        serving_constants.KT_SERVICE_LABEL: service_name,
                        serving_constants.KT_MODULE_LABEL: clean_module_name,
                    }
                specifier = {
                    "type": "label_selector",
                    "selector": pool_selector_for_specifier,
                }

                # Resolve service config (uses same selector as pool specifier for routing)
                service_config = self._resolve_service_config(endpoint, service_name, pool_selector_for_specifier)

                # Get resource kind for pool registration
                resource_kind = self.config.get("resource_kind") or manifest.get("kind")
                pool_metadata = self._load_pool_metadata()

                pool_response = self.controller_client.register_pool(
                    name=service_name,
                    namespace=self.namespace,
                    specifier=specifier,
                    service=service_config,
                    server_port=server_port,
                    labels=labels,
                    annotations=annotations,
                    pool_metadata=pool_metadata,
                    dockerfile=dockerfile,
                    module=module,
                    resource_kind=resource_kind,
                    resource_name=service_name,
                    create_headless_service=create_headless_service,
                )

                if pool_response.get("status") not in ("success", "warning", "partial"):
                    raise Exception(f"Resource registration failed: {pool_response.get('message')}")

                logger.info(f"Registered {service_name} to kubetorch controller in namespace {self.namespace}")

            return apply_response.get("resource", manifest)

        except Exception as e:
            if http_conflict(e):
                logger.info(f"{manifest.get('kind', self.resource_type)} {service_name} already exists, updating")
                existing = self.get_resource(service_name)
                return existing
            raise

    def check_service_ready(self, service_name: str, launch_timeout: int = 300, **kwargs) -> bool:
        """Check if service is ready via client-side polling.

        Uses short server-side timeouts with client-side retry loop to avoid
        proxy timeout issues (e.g., nginx 60s gateway timeout).

        Args:
            service_name: Name of the service
            launch_timeout: Total timeout in seconds

        Returns:
            True if ready, raises exception on timeout or error
        """
        import time

        start_time = time.time()
        poll_interval = 5  # Client-side wait between checks
        server_timeout = 30  # Each server call waits max 30 seconds

        while True:
            elapsed = time.time() - start_time
            remaining = launch_timeout - elapsed

            if remaining <= 0:
                raise TimeoutError(f"Service {service_name} not ready after {launch_timeout}s")

            # Use shorter of remaining time or server_timeout
            this_timeout = min(server_timeout, max(5, remaining))

            try:
                response = self.controller_client.get(
                    f"/controller/check-ready/{self.namespace}/{service_name}",
                    params={
                        "resource_type": self.resource_type,
                        "timeout": int(this_timeout),
                        "poll_interval": 2,
                    },
                    timeout=this_timeout + 10,  # HTTP timeout slightly longer
                )

                if response.get("ready"):
                    return True

                # Check for non-timeout errors (pod failures, cluster failures, etc.)
                details = response.get("details", {})
                if details.get("error_type"):
                    error_msg = response.get("message", "Service not ready")
                    raise RuntimeError(error_msg)

            except TimeoutError:
                # HTTP timeout - continue polling
                pass

            # Not ready yet - wait before next poll
            time.sleep(poll_interval)

    # =========================================================================
    # Resource Operations
    # =========================================================================

    def get_resource(self, service_name: str) -> dict:
        """Get a resource by name."""
        if self.resource_type == "deployment":
            return self.controller_client.get_deployment(
                namespace=self.namespace,
                name=service_name,
            )
        else:
            return self.controller_client.get_namespaced_custom_object(
                group=self.api_group,
                version=self.api_version,
                namespace=self.namespace,
                plural=self.api_plural,
                name=service_name,
            )

    def get_deployment_timestamp_annotation(self, service_name: str) -> Optional[str]:
        """Get deployment timestamp annotation from a resource."""
        try:
            resource = self.get_resource(service_name)
            return resource.get("metadata", {}).get("annotations", {}).get("kubetorch.com/deployment_timestamp")
        except Exception:
            return None

    def get_endpoint(self, service_name: str) -> str:
        """Get the endpoint URL for a service."""
        if self.resource_type == "knative":
            # Knative uses its own routing
            return f"http://{service_name}.{self.namespace}.svc.cluster.local"
        else:
            # All other types use K8s Service
            return f"http://{service_name}.{self.namespace}.svc.cluster.local:80"

    def get_pods_for_service(self, service_name: str, label_selector: str = None) -> List[dict]:
        """Get all pods associated with this service."""
        label_selector = label_selector or f"{serving_constants.KT_SERVICE_LABEL}={service_name}"
        raw = self.controller_client.list_pods(self.namespace, label_selector=label_selector)
        return raw.get("items", [])

    # =========================================================================
    # Teardown
    # =========================================================================

    def teardown_service(self, service_name: str, console=None, force: bool = False) -> bool:
        """Teardown/delete service and associated resources."""
        success = True

        # Delete the K8s resource (skip for selector-only pools which have no K8s resource)
        if self.resource_type != "selector":
            try:
                self._delete_resource(service_name, force=force)
                if console:
                    console.print(f"✓ Deleted {self.resource_type} [blue]{service_name}[/blue]")
                else:
                    logger.info(f"Deleted {self.resource_type} {service_name}")
            except Exception as e:
                if http_not_found(e):
                    if console:
                        console.print(
                            f"[yellow]Note:[/yellow] {self.resource_type} {service_name} not found or already deleted"
                        )
                    else:
                        logger.info(f"{self.resource_type} {service_name} not found or already deleted")
                else:
                    if console:
                        console.print(f"[red]Error:[/red] Failed to delete {self.resource_type} {service_name}: {e}")
                    else:
                        logger.error(f"Failed to delete {self.resource_type} {service_name}: {e}")
                    success = False

        # Delete from controller
        try:
            self.controller_client.delete_pool(namespace=self.namespace, name=service_name)
        except Exception as e:
            if not http_not_found(e):
                if console:
                    console.print(f"[red]Error:[/red] Failed to delete {service_name} from controller: {e}")
                else:
                    logger.error(f"Failed to delete {service_name} from controller: {e}")
                success = False

        # Delete associated services for RayCluster
        if self.resource_type == "raycluster":
            for svc_suffix in ["", "-headless"]:
                try:
                    self.controller_client.delete_service(
                        namespace=self.namespace,
                        name=f"{service_name}{svc_suffix}",
                    )
                except Exception as e:
                    if not http_not_found(e):
                        logger.warning(f"Failed to delete service {service_name}{svc_suffix}: {e}")

        return success

    def _delete_resource(self, service_name: str, force: bool = False) -> None:
        """Delete a K8s resource."""
        kwargs = {}
        if force:
            kwargs["grace_period_seconds"] = 0
            kwargs["propagation_policy"] = "Foreground"

        if self.resource_type == "deployment":
            self.controller_client.delete_deployment(
                namespace=self.namespace,
                name=service_name,
                **kwargs,
            )
        else:
            self.controller_client.delete_namespaced_custom_object(
                group=self.api_group,
                version=self.api_version,
                namespace=self.namespace,
                plural=self.api_plural,
                name=service_name,
                **kwargs,
            )

    # =========================================================================
    # Utilities
    # =========================================================================

    def load_service_info(self, created_service: dict) -> dict:
        """Extract service name, namespace, and pod template from created resource."""
        service_name = created_service.get("metadata", {}).get("name")
        namespace = created_service.get("metadata", {}).get("namespace")

        template_path = self.get_pod_template_path()
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

    def _get_deployment_timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _generate_deployment_id(self, service_name: str, timestamp: str) -> str:
        """Generate a unique deployment ID from service name + timestamp."""
        hash_input = f"{service_name}-{timestamp}"
        short_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:6]
        return f"{service_name}-{short_hash}"

    def _get_deployment_timestamp_and_id(self, service_name: str, deployment_timestamp: str = None) -> Tuple[str, str]:
        """Get both deployment timestamp and deployment ID."""
        timestamp = deployment_timestamp or self._get_deployment_timestamp()
        deployment_id = self._generate_deployment_id(service_name, timestamp)
        return timestamp, deployment_id

    def _clean_module_name(self, module_name: str) -> str:
        """Clean module name to remove invalid characters for Kubernetes labels."""
        return re.sub(r"[^A-Za-z0-9.-]|^[-.]|[-.]$", "", module_name)

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

    # =========================================================================
    # Service Discovery
    # =========================================================================

    @staticmethod
    def discover_services(namespace: str, name_filter: str = None) -> List[Dict]:
        """Discover all Kubetorch services (Knative, Deployments, RayClusters, training jobs, selector pools).

        Args:
            namespace: Kubernetes namespace to search
            name_filter: Optional filter to match service names

        Returns:
            List of service dictionaries with normalized structure:
            {
                'name': str,
                'template_type': str,  # 'ksvc', 'deployment', 'raycluster', 'pytorchjob', 'selector'
                'resource': dict,      # The Kubernetes resource object
                'namespace': str,
                'creation_timestamp': str
            }
        """
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
            """Fetch Deployments in parallel."""
            try:
                label_selector = f"{serving_constants.KT_TEMPLATE_LABEL}=deployment"
                result = controller_client.list_deployments(
                    namespace=namespace,
                    label_selector=label_selector,
                )
                deployments = result.get("items", [])

                local_services = []
                for deployment in deployments:
                    deploy_name = deployment.get("metadata", {}).get("name")
                    if name_filter and name_filter not in deploy_name:
                        continue

                    creation_timestamp = deployment.get("metadata", {}).get("creationTimestamp", "")

                    local_services.append(
                        {
                            "name": deploy_name,
                            "template_type": "deployment",
                            "resource": deployment,
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
                for cluster in clusters:
                    cluster_name = cluster["metadata"]["name"]
                    if name_filter and name_filter not in cluster_name:
                        continue

                    local_services.append(
                        {
                            "name": cluster_name,
                            "template_type": "raycluster",
                            "resource": cluster,
                            "namespace": namespace,
                            "creation_timestamp": cluster["metadata"]["creationTimestamp"],
                        }
                    )

                with services_lock:
                    services.extend(local_services)

            except Exception as e:
                if not http_not_found(e):
                    logger.warning(f"Failed to list RayClusters: {e}")

        def fetch_custom_resources():
            """Fetch custom training job resources in parallel."""
            local_services = []
            for resource_kind in SUPPORTED_TRAINING_JOBS:
                config = get_resource_config(resource_kind)
                api_group = config["api_group"]
                plural = config["api_plural"]
                version = config["api_version"]
                try:
                    label_selector = f"{serving_constants.KT_TEMPLATE_LABEL}={resource_kind}"

                    result = controller_client.list_namespaced_custom_object(
                        group=api_group,
                        version=version,
                        namespace=namespace,
                        plural=plural,
                        label_selector=label_selector,
                    )
                    resources = result.get("items", [])

                    for resource in resources:
                        resource_name = resource["metadata"]["name"]
                        if name_filter and name_filter not in resource_name:
                            continue

                        local_services.append(
                            {
                                "name": resource_name,
                                "template_type": resource_kind,
                                "resource": resource,
                                "namespace": namespace,
                                "creation_timestamp": resource["metadata"]["creationTimestamp"],
                            }
                        )
                except Exception as e:
                    if not http_not_found(e):
                        logger.warning(f"Failed to list {resource_kind}: {e}")

            with services_lock:
                services.extend(local_services)

        def fetch_selector_pools():
            """Fetch selector-based pools from controller database."""
            try:
                resp = controller_client.list_pools(namespace=namespace)
                pools = resp.get("pools", [])

                local_services = []
                for pool in pools:
                    # Only include selector-based pools (others are already discovered via K8s resources)
                    specifier = pool.get("specifier") or {}
                    if specifier.get("type") != "label_selector":
                        continue

                    # Skip pools that have a KT-managed backing K8s resource - these are already
                    # discovered by the K8s resource fetchers (fetch_deployments, fetch_rayclusters, etc.)
                    # We detect KT-managed resources by checking for the KT_TEMPLATE_LABEL.
                    # Selector-only pools (user-deployed resources) don't have this label.
                    pool_labels = pool.get("labels") or {}
                    if pool.get("resource_kind") and serving_constants.KT_TEMPLATE_LABEL in pool_labels:
                        continue

                    pool_name = pool.get("name")
                    if name_filter and name_filter not in pool_name:
                        continue

                    # Get pods using selector from specifier (if available)
                    pods_for_pool = []
                    selector = specifier.get("selector")
                    if selector:
                        # Build label selector string from dict
                        label_selector = ",".join(f"{k}={v}" for k, v in selector.items())
                        try:
                            pods_result = controller_client.list_pods(
                                namespace=namespace, label_selector=label_selector
                            )
                            pods_for_pool = pods_result.get("items", [])
                        except Exception as e:
                            logger.warning(f"Failed to list pods for pool {pool_name}: {e}")

                    # Create a synthetic resource dict for display compatibility
                    pool_metadata = pool.get("pool_metadata") or {}
                    labels = (pool.get("labels") or {}).copy()
                    username = pool_metadata.get("username", "")
                    if username:
                        labels[serving_constants.KT_USERNAME_LABEL] = username

                    num_pods = len(pods_for_pool)
                    synthetic_resource = {
                        "metadata": {
                            "name": pool_name,
                            "namespace": namespace,
                            "creationTimestamp": pool.get("created_at", ""),
                            "labels": labels,
                            "annotations": pool.get("annotations") or {},
                        },
                        "spec": {
                            "replicas": num_pods,  # For status calculation
                        },
                        "status": {
                            "readyReplicas": num_pods,  # Assume pods are ready
                            "replicas": num_pods,
                        },
                        # Extra fields for selector-based pools
                        "_pods": pods_for_pool,  # Actual pod objects from K8s
                        "_pool_metadata": pool_metadata,
                        "_selector": selector,
                    }

                    local_services.append(
                        {
                            "name": pool_name,
                            "template_type": "selector",
                            "resource": synthetic_resource,
                            "namespace": namespace,
                            "creation_timestamp": pool.get("created_at", ""),
                        }
                    )

                with services_lock:
                    services.extend(local_services)

            except Exception as e:
                logger.warning(f"Failed to list selector pools: {e}")

        # Execute all API calls in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(fetch_knative_services),
                executor.submit(fetch_deployments),
                executor.submit(fetch_rayclusters),
                executor.submit(fetch_custom_resources),
                executor.submit(fetch_selector_pools),
            ]

            # Wait for all to complete
            concurrent.futures.wait(futures)

        return services
