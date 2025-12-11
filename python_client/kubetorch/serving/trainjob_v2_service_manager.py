import copy
import os
import time
from typing import List

from kubernetes import client

import kubetorch.serving.constants as serving_constants
from kubetorch.logger import get_logger
from kubetorch.servers.http.utils import load_template
from kubetorch.serving.base_service_manager import BaseServiceManager

logger = get_logger(__name__)


class TrainJobV2ServiceManager(BaseServiceManager):
    """Service manager for Kubeflow Training Operator v2 TrainJob resources.

    TrainJob (v2) uses a simplified structure compared to PyTorchJob/TFJob:
    - API: trainer.kubeflow.org/v1alpha1
    - Uses spec.trainer with numNodes instead of replicaSpecs
    - Uses runtimeRef to specify the training runtime (e.g., torch-distributed)
    - Uses resourcesPerNode (not resources) for resource requests/limits

    Example TrainJob manifest:
        apiVersion: trainer.kubeflow.org/v1alpha1
        kind: TrainJob
        metadata:
          name: pytorch-simple
        spec:
          runtimeRef:
            name: torch-distributed
          trainer:
            numNodes: 2
            image: docker.io/kubeflowkatib/pytorch-mnist:v1beta1-45c5727
            command:
              - python3
              - /opt/pytorch-mnist/mnist.py
            resourcesPerNode:
              requests:
                nvidia.com/gpu: "1"
    """

    SUPPORTED_KINDS = ["TrainJob"]

    @staticmethod
    def _get_config(kind: str) -> dict:
        """Compute configuration for TrainJob."""
        # Handle case-insensitive matching
        kind_lower = kind.lower()
        if kind_lower not in [k.lower() for k in TrainJobV2ServiceManager.SUPPORTED_KINDS]:
            raise ValueError(
                f"Unsupported training job type: {kind}. "
                f"Supported types: {', '.join(TrainJobV2ServiceManager.SUPPORTED_KINDS)}"
            )

        return {
            "container_name": "node",  # TrainJob v2 runtime uses "node" as container name
            "api_group": "trainer.kubeflow.org",
            "api_plural": "trainjobs",
            "api_version": "v1alpha1",
        }

    def __init__(
        self,
        namespace: str,
        kind: str = "TrainJob",
        api_group: str = "trainer.kubeflow.org",
        api_version: str = "v1alpha1",
        service_annotations: dict = None,
    ):
        config = self._get_config(kind)

        super().__init__(
            namespace=namespace,
            template_label="trainjob",
            api_group=api_group,
            api_plural=config["api_plural"],
            api_version=api_version,
            service_annotations=service_annotations,
        )

    def _get_pod_template_path(self, manifest: dict = None) -> List[str]:
        """Get the path to the pod template in TrainJob.

        Returns path to spec.template if it exists (user manifest with full pod template),
        otherwise returns path to spec.trainer (simplified format).
        """
        if manifest:
            template_spec = manifest.get("spec", {}).get("template", {}).get("spec", {})
            if template_spec.get("containers"):
                return ["spec", "template"]
        return ["spec", "trainer"]

    def pod_spec(self, manifest: dict) -> dict:
        """Get the pod spec from the manifest.

        TrainJob v2 supports two structures:
        1. Resources directly in spec.trainer (image, command, env, resources)
        2. Full pod template in spec.template.spec.containers (user-provided)

        We check for spec.template first (user manifest with full pod template),
        then fall back to spec.trainer (simplified TrainJob v2 format).
        """
        spec = manifest.get("spec", {})

        # Check for full pod template first (user-provided manifest structure)
        template_spec = spec.get("template", {}).get("spec", {})
        if template_spec.get("containers"):
            return template_spec

        # Fall back to trainer section (simplified TrainJob v2 format)
        trainer = spec.get("trainer", {})
        if not trainer:
            return {}

        # Add required fields if missing, but keep reference to trainer dict
        trainer.setdefault("name", "node")
        trainer.setdefault("env", [])

        return {"containers": [trainer]}

    def normalize_created_service(self, created_service) -> dict:
        """Extract service info from TrainJob resource.

        TrainJob v2 doesn't have a traditional pod template, so we construct
        a template-like structure from the trainer spec for compatibility.
        """
        trainer = created_service.get("spec", {}).get("trainer", {})
        pod_spec = self.pod_spec(created_service)
        # Ensure container name is "node" for TrainJob v2
        if pod_spec.get("containers"):
            pod_spec["containers"][0]["name"] = "node"
        # Construct a template structure that matches what _launch expects
        template = {
            "metadata": {
                "labels": trainer.get("podLabels", {}),
                "annotations": trainer.get("podAnnotations", {}),
            },
            "spec": pod_spec,
        }
        return {
            "name": created_service.get("metadata", {}).get("name"),
            "namespace": created_service.get("metadata", {}).get("namespace"),
            "template": template,
        }

    def is_distributed(self, manifest: dict) -> bool:
        """Check if this is a distributed job (numNodes > 1)."""
        return self.get_replicas(manifest) > 1

    def get_replicas(self, manifest: dict) -> int:
        """Get total replicas (numNodes) from manifest."""
        trainer = manifest.get("spec", {}).get("trainer", {})
        return trainer.get("numNodes", 1)

    def set_replicas(self, manifest: dict, value: int, distributed_config: dict = None) -> None:
        """Set numNodes in manifest."""
        spec = manifest.setdefault("spec", {})
        trainer = spec.setdefault("trainer", {})
        trainer["numNodes"] = value

        if distributed_config is not None:
            import json

            env = trainer.setdefault("env", [])
            env_updated = False
            for env_var in env:
                if env_var.get("name") == "KT_DISTRIBUTED_CONFIG":
                    env_var["value"] = json.dumps(distributed_config)
                    env_updated = True
                    break

            if not env_updated:
                env.append({"name": "KT_DISTRIBUTED_CONFIG", "value": json.dumps(distributed_config)})

    def _apply_template_metadata_updates(
        self,
        manifest: dict,
        template_labels: dict,
        annotations: dict,
        **kwargs,
    ) -> None:
        """Apply updates to template metadata using trainer.podLabels/podAnnotations."""
        trainer = manifest.setdefault("spec", {}).setdefault("trainer", {})
        trainer.setdefault("podLabels", {}).update(template_labels)
        trainer.setdefault("podAnnotations", {}).update(annotations)

    def _update_launchtime_manifest(
        self, manifest: dict, service_name: str, clean_module_name: str, deployment_timestamp: str, deployment_id: str
    ) -> dict:
        """Update manifest with service name and deployment timestamp."""
        updated = super()._update_launchtime_manifest(
            manifest, service_name, clean_module_name, deployment_timestamp, deployment_id
        )

        trainer = updated.setdefault("spec", {}).setdefault("trainer", {})
        pod_labels = trainer.setdefault("podLabels", {})
        pod_labels[serving_constants.KT_SERVICE_LABEL] = service_name
        pod_labels[serving_constants.KT_MODULE_LABEL] = clean_module_name

        pod_annotations = trainer.setdefault("podAnnotations", {})
        pod_annotations["kubetorch.com/deployment_timestamp"] = deployment_timestamp

        return updated

    def _preprocess_manifest_for_launch(self, manifest: dict) -> dict:
        """Preprocess TrainJob v2 manifest before launch.

        TrainJob v2 API (trainer.kubeflow.org/v1alpha1) only accepts specific fields
        in spec.trainer. This method:
        1. Copies supported fields from spec.template.spec.containers to spec.trainer
           (kubetorch may set values in the template container via pod_spec)
        2. Converts 'resources' to 'resourcesPerNode' (required by TrainJob v2 API)
        3. Removes fields not supported by the Trainer struct

        The Trainer struct accepts: image, command, args, env, numNodes,
        resourcesPerNode, numProcPerNode
        """
        spec = manifest.get("spec", {})

        # Copy supported fields from template container to trainer
        # kubetorch's pod_spec may point to spec.template.spec for modifications,
        # but TrainJob v2 only reads from spec.trainer
        template_containers = spec.get("template", {}).get("spec", {}).get("containers", [])
        if template_containers:
            container = template_containers[0]
            trainer = spec.setdefault("trainer", {})

            # Fields supported by TrainJob v2 Trainer struct
            # Template container values take precedence since kubetorch modifies
            # the template container (via pod_spec), not spec.trainer directly
            supported_fields = ["image", "command", "args", "env", "resources"]
            for field in supported_fields:
                if field in container:
                    if field == "env":
                        # Merge env vars - template container values take precedence
                        trainer_env = {e.get("name"): e for e in trainer.get("env", [])}
                        container_env = {e.get("name"): e for e in container[field]}
                        # Container values override trainer values
                        trainer_env.update(container_env)
                        trainer["env"] = list(trainer_env.values())
                    elif field == "resources":
                        # Merge resources - container values take precedence
                        trainer.setdefault(field, {})
                        for res_type in ["requests", "limits"]:
                            if res_type in container[field]:
                                trainer[field].setdefault(res_type, {}).update(container[field][res_type])
                    else:
                        # For image, command, args - template container value wins
                        trainer[field] = copy.deepcopy(container[field])

        trainer = spec.get("trainer", {})
        if not trainer:
            return manifest

        # Convert 'resources' to 'resourcesPerNode' for TrainJob v2 API
        # kubetorch sets resources via _set_container_resource which uses the
        # standard Kubernetes 'resources' field, but TrainJob v2 expects 'resourcesPerNode'
        if "resources" in trainer and "resourcesPerNode" not in trainer:
            trainer["resourcesPerNode"] = trainer.pop("resources")

        # Remove fields not supported by TrainJob v2 Trainer struct
        # These fields are valid in a container spec but not in trainer.kubeflow.org Trainer
        unsupported_fields = [
            "name",  # Added by pod_spec() for compatibility, not needed
            "ports",
            "volumeMounts",  # Use podTemplateOverrides instead
            "securityContext",
            "livenessProbe",
            "readinessProbe",
            "startupProbe",
            "resources",  # Already converted to resourcesPerNode above
            "workingDir",
            "imagePullPolicy",
        ]
        for field in unsupported_fields:
            trainer.pop(field, None)

        return manifest

    def _create_or_update_resource(self, manifest: dict, service_name: str, clean_module_name: str, **kwargs) -> dict:
        """Create or update TrainJob resource via CustomObjectsApi."""
        server_port = serving_constants.DEFAULT_KT_SERVER_PORT

        labels = manifest.get("metadata", {}).get("labels", {})
        annotations = manifest.get("metadata", {}).get("annotations", {})
        service_labels = labels.copy()
        service_labels.pop(serving_constants.KT_TEMPLATE_LABEL, None)

        try:
            is_distributed = self.is_distributed(manifest)

            # Create ClusterIP service for kubetorch client routing (to rank 0)
            # Use a different name than service_name because JobSet creates a headless service
            # with name=service_name for pod DNS resolution (required for PET_MASTER_ADDR to resolve)
            kt_service_name = f"{service_name}-kt"
            service = load_template(
                template_file=serving_constants.DEPLOYMENT_SERVICE_TEMPLATE_FILE,
                template_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"),
                name=kt_service_name,
                namespace=self.namespace,
                annotations=annotations,
                labels=service_labels,
                deployment_name=service_name,
                module_name=clean_module_name,
                distributed=False,
                server_port=server_port,
            )
            service["spec"]["selector"] = {"jobset.sigs.k8s.io/jobset-name": service_name}
            if is_distributed:
                service["spec"]["selector"]["batch.kubernetes.io/job-completion-index"] = "0"

            try:
                params = {"dryRun": "All"} if kwargs.get("dry_run") else None
                self.controller_client.create_service(namespace=self.namespace, body=service, params=params)
                if not kwargs.get("dry_run"):
                    logger.info(f"Created service {kt_service_name} in namespace {self.namespace}")
            except Exception as e:
                if hasattr(e, "response") and e.response.status_code == 409:
                    logger.info(f"Service {kt_service_name} already exists")
                elif "409" in str(e) or "AlreadyExists" in str(e):
                    logger.info(f"Service {kt_service_name} already exists")
                else:
                    raise

            # Note: JobSet automatically creates a headless service named {service_name}
            # for pod DNS resolution. We don't create one ourselves.

            created_resource = None
            try:
                created_resource = self._create_resource(manifest, **kwargs)
            except client.exceptions.ApiException as e:
                if e.status == 404:
                    logger.error(
                        "TrainJob Custom Resource Definition (CRD) not found. "
                        "Please install the Kubeflow Training Operator v2."
                    )
                raise e

            logger.info(f"Created TrainJob {service_name} in namespace {self.namespace}")
            return created_resource
        except client.exceptions.ApiException as e:
            if e.status == 409:
                logger.info(f"TrainJob {service_name} already exists, updating")
                patch_body = {"spec": manifest["spec"]}
                updated_resource = self._patch_resource(service_name, patch_body, **kwargs)
                logger.info(f"Updated TrainJob {service_name}")
                return updated_resource
            raise e

    def check_service_ready(self, service_name: str, launch_timeout: int = None, **kwargs) -> bool:
        """Check resource readiness by validating job conditions and pod status."""
        sleep_interval = 2
        start_time = time.time()
        launch_timeout = launch_timeout or 600  # Default 10 minutes

        logger.info(f"Checking TrainJob {service_name} readiness (timeout: {launch_timeout} seconds)")

        iteration = 0
        while (time.time() - start_time) < launch_timeout:
            iteration += 1
            try:
                resource = self.get_resource(service_name)
                status = resource.get("status", {})
                conditions = status.get("conditions", [])
                trainer = resource.get("spec", {}).get("trainer", {})
                expected_nodes = trainer.get("numNodes", 1)

                for condition in conditions:
                    if condition.get("type") == "Failed" and condition.get("status") == "True":
                        raise RuntimeError(
                            f"TrainJob {service_name} failed: {condition.get('message', 'Unknown error')}"
                        )

                is_job_running = any(
                    c.get("type") in ("Succeeded", "Running", "Created") and c.get("status") == "True"
                    for c in conditions
                )

                if is_job_running or not conditions:
                    pods = self.get_pods_for_service(service_name)
                    running_pods = [pod for pod in pods if pod.status.phase == "Running"]

                    if len(running_pods) >= expected_nodes and expected_nodes > 0:
                        logger.info(
                            f"TrainJob {service_name} is ready with {len(running_pods)}/{expected_nodes} nodes running"
                        )
                        return True

                if iteration % (30 // sleep_interval) == 0:
                    elapsed = int(time.time() - start_time)
                    remaining = launch_timeout - elapsed
                    pods = self.get_pods_for_service(service_name)
                    running_pods = [pod for pod in pods if pod.status.phase == "Running"]

                    logger.info(
                        f"TrainJob is not yet ready (elapsed: {elapsed}s, remaining: {remaining}s). "
                        f"Running nodes: {len(running_pods)}/{expected_nodes}"
                    )

            except RuntimeError:
                raise
            except Exception as e:
                logger.error(f"Error checking TrainJob readiness: {e}")

            time.sleep(sleep_interval)

        raise TimeoutError(f"TrainJob {service_name} did not become ready within {launch_timeout} seconds")

    def get_endpoint(self, service_name: str) -> str:
        """Get endpoint for TrainJob service."""
        # Use the -kt suffix service we created for kubetorch client routing
        return f"http://{service_name}-kt.{self.namespace}.svc.cluster.local:80"

    def get_routing_service_name(self, service_name: str) -> str:
        """Get the service name used for NGINX proxy routing.

        TrainJobV2 uses {service_name}-kt to avoid conflicting with JobSet's
        headless service which uses {service_name} for pod DNS resolution.
        """
        return f"{service_name}-kt"

    def get_container_name(self) -> str:
        """TrainJob v2 runtime uses 'node' as the container name."""
        return "node"

    def supports_distributed_config(self) -> bool:
        """Training jobs support distributed config."""
        return True

    def get_service_dns(self, service_name: str, namespace: str, is_distributed: bool) -> str:
        """Get the DNS name for service discovery.

        TrainJob v2 uses JobSet which creates a headless service with the same name
        as the TrainJob (no -headless suffix).
        """
        return f"{service_name}.{namespace}.svc.cluster.local"

    def set_pod_spec(self, manifest: dict, pod_spec: dict) -> None:
        """Set the pod spec in the manifest.

        For TrainJob v2, container fields go to spec.trainer and pod-level fields
        go to spec.podTemplateOverrides.
        """
        containers = pod_spec.get("containers", [])
        if containers:
            container = containers[0]
            trainer = manifest.setdefault("spec", {}).setdefault("trainer", {})
            # Update trainer with container fields
            trainer.update(container)

        # Handle pod-level fields via podTemplateOverrides
        pod_level_fields = {}
        for field in ["nodeSelector", "tolerations", "volumes", "serviceAccountName", "affinity"]:
            if field in pod_spec:
                pod_level_fields[field] = pod_spec[field]

        if pod_level_fields:
            spec = manifest.setdefault("spec", {})
            overrides = spec.setdefault("podTemplateOverrides", [])

            # Find or create override for "node" job (the trainer pods)
            node_override = None
            for override in overrides:
                if any(t.get("name") == "node" for t in override.get("targetJobs", [])):
                    node_override = override
                    break

            if node_override is None:
                node_override = {"targetJobs": [{"name": "node"}], "spec": {}}
                overrides.append(node_override)

            override_spec = node_override.setdefault("spec", {})
            override_spec.update(pod_level_fields)

    def set_env_vars_in_manifest(self, manifest: dict, env_vars: dict) -> None:
        """Set environment variables in spec.trainer.env for TrainJob v2."""
        trainer = manifest.setdefault("spec", {}).setdefault("trainer", {})
        env = trainer.setdefault("env", [])
        for name, value in env_vars.items():
            updated = False
            for env_var in env:
                if env_var.get("name") == name:
                    env_var["value"] = value
                    updated = True
                    break
            if not updated:
                env.append({"name": name, "value": value})

    def get_pods_for_service(self, service_name: str, **kwargs) -> List[dict]:
        """Get all pods associated with this TrainJob.

        TrainJob v2 uses JobSet labels, not kubetorch labels for pods.
        """
        # TrainJob v2 pods are labeled with jobset.sigs.k8s.io/jobset-name
        label_selector = f"jobset.sigs.k8s.io/jobset-name={service_name}"
        try:
            result = self.controller_client.list_pods(namespace=self.namespace, label_selector=label_selector)
            return result.get("items", [])
        except Exception as e:
            logger.warning(f"Failed to list pods for TrainJob {service_name}: {e}")
            return []

    def _teardown_associated_resources(self, service_name: str, console=None) -> bool:
        """Delete associated Kubernetes Services."""
        success = True

        # Delete the -kt ClusterIP service we created for kubetorch client routing
        # Note: JobSet creates/deletes its own headless service named {service_name}
        kt_service_name = f"{service_name}-kt"
        try:
            self.controller_client.delete_service(namespace=self.namespace, name=kt_service_name, ignore_not_found=True)
            if console:
                console.print(f"✓ Deleted service [blue]{kt_service_name}[/blue]")
            else:
                logger.info(f"Deleted service {kt_service_name}")
        except Exception as e:
            if console:
                console.print(f"[red]Error:[/red] Failed to delete service {kt_service_name}: {e}")
            else:
                logger.error(f"Failed to delete service {kt_service_name}: {e}")
            success = False

        return success
