"""Unified service manager for all K8s resource types.

This module provides a single ServiceManager class that handles all resource types
(Deployment, Knative, RayCluster, training jobs, etc.) through configuration.
Resource-specific behavior is driven by its relevant config.
"""
import copy
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import kubetorch.provisioning.constants as provisioning_constants
from kubetorch import globals
from kubetorch.logger import get_logger
from kubetorch.provisioning.utils import get_resource_config, SUPPORTED_TRAINING_JOBS
from kubetorch.resources.compute.endpoint import Endpoint
from kubetorch.resources.compute.utils import ServiceTimeoutError
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
            resource_type (str): Type of resource (deployment, knative, raycluster, pytorchjob, etc.).
            namespace (str): Kubernetes namespace.
            service_annotations (dict): Optional annotations to apply to services. (Default: None)
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

    def get_pod_template_path(self, override: List[str] = None) -> Optional[List[str]]:
        """Get the path to the pod template in the manifest.

        For BYO manifests with unknown resource types, returns None if no override is provided.
        The caller should handle None gracefully (e.g., skip pod spec extraction).
        """
        if override:
            return override
        return self.config.get("pod_template_path")

    def pod_spec(self, manifest: dict, pod_template_path: List[str] = None) -> dict:
        """Get the pod spec from a manifest.

        Returns empty dict if pod template path is not configured and no override is provided.
        This allows BYO manifests to work without requiring pod spec extraction.
        """
        from kubetorch.resources.compute.utils import navigate_path

        template_path = self.get_pod_template_path(override=pod_template_path)
        if template_path is None:
            # Unknown resource type with no override - return empty pod spec
            # Server port will default to 32300, which is fine for most cases
            return {}
        pod_template = navigate_path(manifest, template_path)
        if pod_template is None:
            return {}
        return pod_template.get("spec", {})

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
            manifest.setdefault("spec", {})["replicas"] = value or 1

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

        # Distributed config flows via WebSocket metadata from the controller,
        # not via env vars in the manifest

    # =========================================================================
    # Labels and Annotations
    # =========================================================================

    def _get_labels(self, custom_labels: dict = None) -> dict:
        """Get standard kubetorch labels."""
        from kubetorch import __version__

        labels = {
            provisioning_constants.KT_VERSION_LABEL: __version__,
            provisioning_constants.KT_TEMPLATE_LABEL: self.template_label,
            provisioning_constants.KT_USERNAME_LABEL: globals.config.username,
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
        annotations = {}
        if self.service_annotations:
            annotations.update(self.service_annotations)
        if custom_annotations:
            annotations.update(custom_annotations)
        if inactivity_ttl:
            annotations[provisioning_constants.INACTIVITY_TTL_ANNOTATION] = inactivity_ttl
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
        pod_template_path: List[str] = None,
        **kwargs,
    ) -> dict:
        """Apply kubetorch labels and annotations to manifest.

        Args:
            pod_template_path: Override path to pod template for BYO manifests
                with non-standard pod template locations.
        """
        from kubetorch.provisioning.utils import nested_override

        labels = self._get_labels(custom_labels)
        template_labels = labels.copy()
        template_labels.pop(provisioning_constants.KT_TEMPLATE_LABEL, None)
        annotations = self._get_annotations(custom_annotations, inactivity_ttl)

        # Update top-level metadata
        manifest["metadata"].setdefault("labels", {}).update(labels)
        manifest["metadata"].setdefault("annotations", {}).update(annotations)

        # Apply template metadata updates (use override if provided)
        effective_path = pod_template_path or self.config.get("pod_template_path")
        if effective_path:
            self._apply_template_metadata_updates(manifest, template_labels, annotations, path=effective_path, **kwargs)

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
            if isinstance(current, list):
                # Handle list index (can be int or string like "0")
                try:
                    idx = int(key)
                    if idx < len(current):
                        current = current[idx]
                    else:
                        return  # nothing to update
                except (ValueError, IndexError):
                    return
            elif isinstance(current, dict):
                current = current.setdefault(key, {})
            else:
                return
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
        pod_template_path: List[str] = None,
    ) -> dict:
        """Update manifest with service name and deployment timestamp.

        Args:
            pod_template_path: Override path to pod template for BYO manifests
                with non-standard pod template locations.
        """
        from kubetorch.resources.compute.utils import navigate_path

        updated = copy.deepcopy(manifest)

        # Update top-level metadata
        updated["metadata"]["name"] = service_name
        updated["metadata"].setdefault("labels", {})
        updated["metadata"]["labels"][provisioning_constants.KT_SERVICE_LABEL] = service_name
        updated["metadata"]["labels"][provisioning_constants.KT_MODULE_LABEL] = clean_module_name
        updated["metadata"]["labels"][provisioning_constants.KT_APP_LABEL] = service_name

        # For Deployments, update selector.matchLabels - only use service label for selection
        # (module label is informational, not used for pod selection)
        if self.resource_type == "deployment":
            updated["spec"].setdefault("selector", {}).setdefault("matchLabels", {})
            updated["spec"]["selector"]["matchLabels"][provisioning_constants.KT_SERVICE_LABEL] = service_name

        # Update template metadata (use override if provided)
        effective_path = pod_template_path or self.config.get("pod_template_path")
        if effective_path:
            pod_template = navigate_path(updated, effective_path, create_missing=True)
            if pod_template is not None:
                metadata = pod_template.setdefault("metadata", {})
                metadata.setdefault("labels", {})[provisioning_constants.KT_SERVICE_LABEL] = service_name
                metadata["labels"][provisioning_constants.KT_MODULE_LABEL] = clean_module_name
                metadata["labels"][provisioning_constants.KT_APP_LABEL] = service_name

            # Also update worker template for distributed resources (known types only)
            worker_path = self.config.get("worker_template_path")
            if worker_path:
                worker_template = navigate_path(updated, worker_path)
                if worker_template is not None:
                    metadata = worker_template.setdefault("metadata", {})
                    metadata.setdefault("labels", {})[provisioning_constants.KT_SERVICE_LABEL] = service_name
                    metadata["labels"][provisioning_constants.KT_MODULE_LABEL] = clean_module_name

        return updated

    # =========================================================================
    # Service Config / Endpoint Resolution
    # =========================================================================

    def _resolve_service_config(
        self, endpoint: Optional[Endpoint], service_name: str, workload_selector: dict
    ) -> Optional[dict]:
        """Resolve service config from endpoint or use resource-specific default."""
        if endpoint:
            return endpoint.to_service_config()

        default_routing = self.config.get("default_routing")
        if default_routing is None:
            # Use workload selector for service routing
            return None
        elif default_routing == "knative_url":
            # Knative provides its own URL
            url = f"http://{service_name}.{self.namespace}.svc.cluster.local"
            return {"type": "url", "url": url}
        elif isinstance(default_routing, dict):
            # Merge workload selector with default routing labels
            service_selector = {**workload_selector, **default_routing}
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
        deployment_mode: str = None,
        distributed_config: dict = None,
        runtime_config: dict = None,
        pod_template_path: List[str] = None,
        launch_id: str = None,
    ) -> Tuple[dict, dict, str]:
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
            deployment_mode (str, optional): Deployment mode (e.g., "deployment", "knative").
            distributed_config (dict, optional): Distributed configuration.
            runtime_config (dict, optional): Runtime configuration that flows via WebSocket.
                Includes log_streaming_enabled, metrics_enabled, inactivity_ttl, log_level,
                allowed_serialization. These can change between deploys without pod recreation.
            pod_template_path (List[str], optional): Custom path to pod template for BYO manifests
                with non-standard pod template locations.

        Returns:
            Tuple[dict, dict, str]: A tuple of (`created_service`, `updated_manifest`, `service_url`) where
                `created_service` is the K8s resource returned by the controller, `updated_manifest` is the manifest
                with applied labels and timestamps, and `service_url` is the url registered with the controller.
        """
        logger.info(f"Deploying {manifest.get('kind', self.resource_type)} service with name: {service_name}")

        # Note: Module metadata and runtime env vars (KT_SERVICE_NAME, KT_SERVICE_DNS, etc.)
        # are now sent via controller WebSocket instead of being baked into the manifest.
        # See http_server.py ControllerWebSocket for details.

        # Preprocess manifest (syncs worker pod specs from primary, including env vars)
        manifest = self._preprocess_manifest_for_launch(manifest)

        # Update manifest with service name and deployment metadata
        clean_module_name = self._clean_module_name(module_name)
        timestamp = deployment_timestamp or self._get_deployment_timestamp()
        updated_manifest = self._update_launchtime_manifest(
            manifest, service_name, clean_module_name, timestamp, pod_template_path=pod_template_path
        )

        # Create or update the resource with the controller
        result = self._apply_and_register_workload(
            manifest=updated_manifest,
            service_name=service_name,
            dry_run=dryrun,
            dockerfile=dockerfile,
            module=module,
            create_headless_service=create_headless_service,
            endpoint=endpoint,
            pod_selector=pod_selector,
            deployment_mode=deployment_mode,
            distributed_config=distributed_config,
            runtime_config=runtime_config,
            pod_template_path=pod_template_path,
            launch_id=launch_id,
        )
        created_service = result["resource"]
        service_url = result.get("service_url")

        return created_service, updated_manifest, service_url

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
        replicas = manifest.get("spec", {}).get("replicas") or 1
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
        """Check if this is a distributed job.

        Distributed config flows via WebSocket, so we check replicas for training jobs.
        """
        if self.resource_type in SUPPORTED_TRAINING_JOBS:
            return self.get_replicas(manifest) > 1

        return False

    def _load_workload_metadata(
        self,
        deployment_mode: str = None,
        distributed_config: dict = None,
        runtime_config: dict = None,
        launch_id: str = None,
        dockerfile: str = None,
    ) -> dict:
        """Build workload metadata dict for controller registration.

        Args:
            deployment_mode: Deployment mode (e.g., "deployment", "knative").
            distributed_config: Distributed configuration for SPMD.
            runtime_config: Runtime configuration that flows via WebSocket to pods.
                Includes log_streaming_enabled, metrics_enabled, inactivity_ttl,
                log_level, allowed_serialization.
            launch_id: Unique ID for this launch/deployment (changes on each .to() call).
            dockerfile: Dockerfile contents for image build.
        """
        metadata = {"username": globals.config.username}
        if deployment_mode:
            metadata["deployment_mode"] = deployment_mode
        if distributed_config:
            metadata["distributed_config"] = distributed_config
        if runtime_config:
            metadata["runtime_config"] = runtime_config
        if launch_id:
            metadata["launch_id"] = launch_id
        if dockerfile:
            metadata["dockerfile"] = dockerfile
        return metadata

    def _apply_and_register_workload(
        self,
        manifest: dict,
        service_name: str,
        dry_run: bool = False,
        dockerfile: str = None,
        module: dict = None,
        create_headless_service: bool = False,
        endpoint: Optional[Endpoint] = None,
        pod_selector: Optional[Dict[str, str]] = None,
        deployment_mode: str = None,
        distributed_config: dict = None,
        runtime_config: dict = None,
        pod_template_path: List[str] = None,
        launch_id: str = None,
    ) -> dict:
        """Create or update resource via controller using the deploy endpoint. Applies the manifest and registers the workload."""
        pod_spec = self.pod_spec(manifest, pod_template_path=pod_template_path)
        server_port = pod_spec.get("containers", [{}])[0].get("ports", [{}])[0].get("containerPort", 32300)

        labels = manifest.get("metadata", {}).get("labels", {})
        annotations = manifest.get("metadata", {}).get("annotations", {})

        # Use custom pod_selector if provided, otherwise use KT service label for pods
        if pod_selector:
            workload_selector_for_specifier = pod_selector
        else:
            workload_selector_for_specifier = {
                provisioning_constants.KT_SERVICE_LABEL: service_name,
            }
        specifier = {
            "type": "label_selector",
            "selector": workload_selector_for_specifier,
        }

        service_config = self._resolve_service_config(endpoint, service_name, workload_selector_for_specifier)
        workload_metadata = self._load_workload_metadata(
            deployment_mode=deployment_mode,
            distributed_config=distributed_config,
            runtime_config=runtime_config,
            launch_id=launch_id,
            dockerfile=dockerfile,
        )

        # Build auto_termination config for CRD spec.autoTermination
        # This is separate from runtime_config (which flows to pods via WebSocket)
        auto_termination = None
        if runtime_config and runtime_config.get("inactivity_ttl"):
            auto_termination = {"inactivityTtl": runtime_config["inactivity_ttl"]}

        try:
            manifest_replicas = manifest.get("spec", {}).get("replicas", "NOT_SET")
            logger.debug(f"Deploying {service_name}: replicas={manifest_replicas}")

            deploy_response = self.controller_client.deploy(
                service_name=service_name,
                namespace=self.namespace,
                resource_type=self.resource_type,
                resource_manifest=manifest,
                specifier=specifier,
                service=service_config,
                server_port=server_port,
                labels=labels,
                annotations=annotations,
                workload_metadata=workload_metadata,
                module=module,
                create_headless_service=create_headless_service,
                auto_termination=auto_termination,
            )

            # Check apply result
            if deploy_response.get("apply_status") == "error":
                raise Exception(f"Apply failed: {deploy_response.get('apply_message')}")

            logger.info(
                f"Applied {manifest.get('kind', self.resource_type)} {service_name} in namespace {self.namespace}"
            )

            # Check workload registration result
            if not dry_run:
                workload_status = deploy_response.get("workload_status")
                if workload_status not in ("success", "warning", "partial"):
                    raise Exception(f"Resource registration failed: {deploy_response.get('workload_message')}")

                logger.info(f"Registered {service_name} to kubetorch controller in namespace {self.namespace}")

            return {
                "resource": deploy_response.get("resource", manifest),
                "service_url": deploy_response.get("service_url"),
            }

        except Exception as e:
            if http_conflict(e):
                logger.info(f"{manifest.get('kind', self.resource_type)} {service_name} already exists, updating")
                # Return the manifest we tried to apply - resource already exists with similar config
                return {"resource": manifest, "service_url": None}
            raise

    def check_service_ready(
        self, service_name: str, launch_timeout: int = 300, launch_id: str = None, **kwargs
    ) -> bool:
        """Check if service is ready via client-side polling.

        Uses short server-side timeouts with client-side retry loop to avoid
        proxy timeout issues (e.g., nginx 60s gateway timeout).

        The controller reads from the KubetorchWorkload CRD status to determine readiness,
        including launch_id verification for re-deployments.

        Args:
            service_name (str): Name of the service.
            launch_timeout (int): Total timeout in seconds. (Default: 300)
            launch_id (str, optional): Launch ID to verify. If provided, waits until the
                workload's readyLaunchId matches, ensuring the correct deployment is ready.

        Returns:
            True if ready. Raises exception on timeout or error.
        """
        import time

        start_time = time.time()
        poll_interval = 5  # Client-side wait between checks
        server_timeout = 30  # Each server call waits max 30 seconds

        while True:
            elapsed = time.time() - start_time
            remaining = launch_timeout - elapsed

            if remaining <= 0:
                raise ServiceTimeoutError(f"Service {service_name} not ready after {launch_timeout}s")

            # Use shorter of remaining time or server_timeout
            this_timeout = min(server_timeout, max(5, remaining))

            try:
                params = {
                    "resource_type": self.resource_type,
                    "timeout": int(this_timeout),
                    "poll_interval": 2,
                }
                if launch_id:
                    params["launch_id"] = launch_id

                response = self.controller_client.get(
                    f"/controller/check-ready/{self.namespace}/{service_name}",
                    params=params,
                    timeout=this_timeout + 10,  # HTTP timeout slightly longer
                )

                if response.get("ready"):
                    # If launch_id was provided and controller supports it, verify it matches.
                    # If ready_launch_id is absent/None, the controller hasn't implemented
                    # launch_id verification yet -- accept ready as-is (backwards compat).
                    ready_launch_id = response.get("ready_launch_id")
                    if launch_id and ready_launch_id and ready_launch_id != launch_id:
                        # Controller is ready but for a different launch_id, keep waiting
                        logger.debug(
                            f"Service ready but launch_id mismatch: expected {launch_id}, got {ready_launch_id}"
                        )
                    else:
                        return True

                # Check for errors from CRD status
                error = response.get("error")
                if error:
                    self._raise_error_from_crd_status(error)

                # Legacy error handling (for backwards compatibility during migration)
                details = response.get("details", {})
                error_type = details.get("error_type")
                if error_type:
                    error_msg = response.get("message", "Service not ready")

                    # Check for image pull errors based on message content
                    error_msg_lower = error_msg.lower()
                    if (
                        "image pull" in error_msg_lower
                        or "imagepullbackoff" in error_msg_lower
                        or "errimagepull" in error_msg_lower
                    ):
                        from kubetorch import ImagePullError

                        raise ImagePullError(f"Container image failed to pull: {error_msg}")

                    if error_type in ("replicaset_error", "pod_error", "resource_error", "revision_error"):
                        from kubetorch import ResourceNotAvailableError

                        raise ResourceNotAvailableError(error_msg)
                    raise RuntimeError(error_msg)

            except TimeoutError:
                # HTTP timeout - continue polling
                pass

            # Not ready yet - wait before next poll
            time.sleep(poll_interval)

    def _raise_error_from_crd_status(self, error: dict):
        """Convert CRD error status to Python exception.

        Args:
            error: Error dict from CRD status with keys: type, message, reason, podName, containerName
        """
        from kubetorch import ImagePullError, PodTerminatedError, ResourceNotAvailableError, ServiceHealthError

        error_type = error.get("type", "Unknown")
        message = error.get("message", "Unknown error")
        pod_name = error.get("podName")

        if error_type == "ImagePullError":
            raise ImagePullError(f"Container image failed to pull: {message}")
        elif error_type == "OOMKilled":
            raise PodTerminatedError(pod_name=pod_name, reason="OOMKilled")
        elif error_type in ("CrashLoopBackOff", "StartupError", "FailedMount"):
            raise ServiceHealthError(message)
        elif error_type in ("ResourceNotAvailable", "ContainerError"):
            raise ResourceNotAvailableError(message)
        else:
            raise RuntimeError(f"Workload error ({error_type}): {message}")

    # =========================================================================
    # Resource Operations
    # =========================================================================

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
        label_selector = label_selector or f"{provisioning_constants.KT_SERVICE_LABEL}={service_name}"
        raw = self.controller_client.list_pods(self.namespace, label_selector=label_selector)
        return raw.get("items", [])

    # =========================================================================
    # Utilities
    # =========================================================================

    def load_service_info(self, created_service: dict, pod_template_path: List[str] = None) -> dict:
        """Extract service name, namespace, and pod template from created resource.

        Args:
            created_service: The created K8s resource dict
            pod_template_path: Optional custom path to pod template (for BYO manifests with non-standard paths)
        """
        from kubetorch.resources.compute.utils import navigate_path

        service_name = created_service.get("metadata", {}).get("name")
        namespace = created_service.get("metadata", {}).get("namespace")

        template_path = pod_template_path or self.get_pod_template_path()
        if template_path is None:
            # BYO manifest with unknown resource type - no pod template path configured
            pod_template = {}
        else:
            pod_template = navigate_path(created_service, template_path)
            if pod_template is None:
                raise ValueError("Failed to find pod template in created service.")

        return {
            "name": service_name,
            "namespace": namespace,
            "template": pod_template,
        }

    def _get_deployment_timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

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
        """Discover all Kubetorch workloads.

        Args:
            namespace (str): Kubernetes namespace to search.
            name_filter (str, optional): Filter to match service names. (Default: None)

        Returns:
            List of service dictionaries with normalized structure:
            {
                'name': str,
                'template_type': str,  # lowercase kind (e.g., 'deployment', 'jobset', 'pytorchjob')
                'resource': dict,      # Synthetic resource dict for display
                'namespace': str,
                'creation_timestamp': str
            }
        """
        controller_client = globals.controller_client()

        response = controller_client.discover_resources(
            namespace=namespace,
            name_filter=name_filter,
            include_pods=True,  # Fetch pods in bulk
        )

        services = []

        for workload in response.get("workloads", []):
            workload_name = workload.get("name")
            resource_kind = workload.get("kind") or workload.get("resource_kind")

            # template_type is just the lowercased kind, or "selector" if unknown
            template_type = (resource_kind or "selector").lower()

            # Get pods from the workload (already fetched in bulk by controller)
            pods_for_workload = workload.get("pods", [])

            # Get specifier and selector
            specifier = workload.get("specifier")
            if not isinstance(specifier, dict):
                specifier = {}
            selector = specifier.get("selector", {})

            # Build a synthetic resource dict for display compatibility
            workload_metadata = workload.get("workload_metadata") or {}
            labels = (workload.get("labels") or {}).copy()
            username = workload_metadata.get("username", "")
            if username:
                labels[provisioning_constants.KT_USERNAME_LABEL] = username

            num_pods = len(pods_for_workload)
            synthetic_resource = {
                "metadata": {
                    "name": workload_name,
                    "namespace": namespace,
                    "creationTimestamp": workload.get("created_at", ""),
                    "labels": labels,
                    "annotations": workload.get("annotations") or {},
                },
                "spec": {
                    "replicas": num_pods,
                },
                "status": {
                    "readyReplicas": num_pods,
                    "replicas": num_pods,
                },
                "_pods": pods_for_workload,
                "_workload_metadata": workload_metadata,
                "_selector": selector,
            }

            services.append(
                {
                    "name": workload_name,
                    "template_type": template_type,
                    "resource": synthetic_resource,
                    "namespace": namespace,
                    "creation_timestamp": workload.get("created_at", ""),
                }
            )

        return services
