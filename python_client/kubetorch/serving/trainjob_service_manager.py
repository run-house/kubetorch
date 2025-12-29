import copy
import os
import time
from typing import List

import kubetorch.serving.constants as serving_constants
from kubetorch.logger import get_logger
from kubetorch.servers.http.utils import load_template
from kubetorch.serving.base_service_manager import BaseServiceManager
from kubetorch.utils import http_conflict, http_not_found

logger = get_logger(__name__)


class TrainJobServiceManager(BaseServiceManager):
    """Service manager for Kubernetes training job resources.

    Supports training job types: PyTorchJob, TFJob, MXJob, XGBoostJob.
    Configuration is automatically computed from the kind.
    """

    SUPPORTED_KINDS = ["PyTorchJob", "TFJob", "MXJob", "XGBoostJob"]

    @staticmethod
    def _get_config(kind: str) -> dict:
        """Compute configuration for a training job based on its kind."""
        if kind not in TrainJobServiceManager.SUPPORTED_KINDS:
            raise ValueError(
                f"Unsupported training job type: {kind}. "
                f"Supported types: {', '.join(TrainJobServiceManager.SUPPORTED_KINDS)}"
            )

        primary_replica = {
            "PyTorchJob": "Master",
            "TFJob": "Chief",
            "MXJob": "Scheduler",
            "XGBoostJob": "Master",
        }

        container_name = {
            "PyTorchJob": "pytorch",
            "TFJob": "tensorflow",
            "MXJob": "mxnet",
            "XGBoostJob": "xgboost",
        }

        replica_specs_key_map = {
            "PyTorchJob": "pytorchReplicaSpecs",
            "TFJob": "tfReplicaSpecs",
            "MXJob": "mxReplicaSpecs",
            "XGBoostJob": "xgbReplicaSpecs",
        }

        config = {
            "replica_specs_key": replica_specs_key_map[kind],
            "primary_replica": primary_replica[kind],
            "worker_replica": "Worker",
            "container_name": container_name[kind],
            "api_group": "kubeflow.org",
            "api_plural": kind.lower() + "s",
            "api_version": "v1",
        }
        return config

    def __init__(
        self,
        namespace: str,
        kind: str,
        api_group: str = "kubeflow.org",
        api_version: str = "v1",
        service_annotations: dict = None,
    ):
        # Normalize kind's capitalization
        for supported_kind in TrainJobServiceManager.SUPPORTED_KINDS:
            if supported_kind.lower() == kind.lower():
                kind = supported_kind
                break

        config = self._get_config(kind)

        super().__init__(
            namespace=namespace,
            template_label=kind.lower(),
            api_group=api_group,
            api_plural=kind.lower() + "s",
            api_version=api_version,
            service_annotations=service_annotations,
        )

        self.replica_specs_key = config["replica_specs_key"]
        self.primary_replica = config["primary_replica"]
        self.worker_replica = config["worker_replica"]

    def get_pod_template_path(self) -> List[str]:
        """Get the path to the primary replica pod template."""
        return ["spec", self.replica_specs_key, self.primary_replica, "template"]

    def _get_replica_counts(self, service_name: str, spec: dict) -> dict:
        """Get current and expected replica counts for master and worker pods.

        Returns:
            Dict with keys: running_pods, expected_replicas, master_pods, worker_pods,
                           expected_master, expected_worker
        """
        pods = self.get_pods_for_service(service_name)
        running_pods = [pod for pod in pods if pod.get("status", {}).get("phase") == "Running"]

        # Count by replica type using Kubeflow standard labels
        master_pods = [
            pod
            for pod in running_pods
            if pod.get("metadata", {}).get("labels", {}).get("training.kubeflow.org/replica-type")
            == self.primary_replica.lower()
        ]
        worker_pods = [
            pod
            for pod in running_pods
            if pod.get("metadata", {}).get("labels", {}).get("training.kubeflow.org/replica-type")
            == self.worker_replica.lower()
        ]

        # Get expected counts from spec
        replica_specs = spec.get(self.replica_specs_key, {})
        expected_master = replica_specs.get(self.primary_replica, {}).get("replicas", 0)
        expected_worker = replica_specs.get(self.worker_replica, {}).get("replicas", 0)
        expected_replicas = sum(
            replica_spec.get("replicas", 0) for replica_spec in replica_specs.values() if isinstance(replica_spec, dict)
        )

        return {
            "running_pods": running_pods,
            "expected_replicas": expected_replicas,
            "master_pods": master_pods,
            "worker_pods": worker_pods,
            "expected_master": expected_master,
            "expected_worker": expected_worker,
        }

    def normalize_created_service(self, created_service) -> dict:
        """Extract service info from training job resource."""
        return {
            "name": created_service.get("metadata", {}).get("name"),
            "namespace": created_service.get("metadata", {}).get("namespace"),
            "template": created_service["spec"][self.replica_specs_key][self.primary_replica]["template"],
        }

    def is_distributed(self, manifest: dict) -> bool:
        """Check if this is a distributed job.

        For training jobs, check both KT_DISTRIBUTED_CONFIG env var and worker replicas count.
        """
        return super().is_distributed(manifest) or self.get_replicas(manifest) > 1

    def get_replicas(self, manifest: dict) -> int:
        """Get total replicas from manifest by summing across all replica types."""
        spec = manifest.get("spec", {})
        replica_specs = spec.get(self.replica_specs_key, {})
        return sum(replica_spec.get("replicas", 0) for replica_spec in replica_specs.values())

    def set_replicas(self, manifest: dict, value: int, distributed_config: dict = None) -> None:
        """Set replicas in manifest."""
        spec = manifest.setdefault("spec", {})
        worker_replicas = max(0, value - 1)  # Primary counts as 1
        replica_specs = spec.setdefault(self.replica_specs_key, {})

        primary_spec = replica_specs.setdefault(self.primary_replica, {})
        primary_spec["replicas"] = 1

        worker_spec = replica_specs.setdefault(self.worker_replica, {})
        worker_spec["replicas"] = worker_replicas

        if "template" not in worker_spec:
            primary_template = primary_spec.get("template", {})
            worker_spec["template"] = copy.deepcopy(primary_template)

        # Update distributed config if provided
        if distributed_config is not None:
            import json

            env_vars_to_set = {"KT_DISTRIBUTED_CONFIG": json.dumps(distributed_config)}

            # Set in both primary and worker replica containers
            for replica_name in [self.primary_replica, self.worker_replica]:
                replica_spec = replica_specs.get(replica_name, {})
                pod_spec = replica_spec.get("template", {}).get("spec", {})
                containers = pod_spec.get("containers", [])

                for container in containers:
                    if "env" not in container:
                        container["env"] = []
                    env_updated = False
                    for env_var in container["env"]:
                        if env_var.get("name") == "KT_DISTRIBUTED_CONFIG":
                            env_var["value"] = env_vars_to_set["KT_DISTRIBUTED_CONFIG"]
                            env_updated = True
                            break

                    if not env_updated:
                        container["env"].append(
                            {"name": "KT_DISTRIBUTED_CONFIG", "value": env_vars_to_set["KT_DISTRIBUTED_CONFIG"]}
                        )

    def _apply_template_metadata_updates(
        self,
        manifest: dict,
        template_labels: dict,
        annotations: dict,
        **kwargs,
    ) -> None:
        """Apply updates to template metadata for primary and worker replica templates."""
        super()._apply_template_metadata_updates(manifest, template_labels, annotations, **kwargs)
        if self.is_distributed(manifest):
            worker_path = ["spec", self.replica_specs_key, self.worker_replica, "template"]
            super()._apply_template_metadata_updates(manifest, template_labels, annotations, path=worker_path, **kwargs)

    def _update_launchtime_manifest(
        self, manifest: dict, service_name: str, clean_module_name: str, deployment_timestamp: str, deployment_id: str
    ) -> dict:
        """Update manifest with service name and deployment timestamp."""
        updated = super()._update_launchtime_manifest(
            manifest, service_name, clean_module_name, deployment_timestamp, deployment_id
        )

        spec = updated.get("spec", {})
        replica_specs = spec.get(self.replica_specs_key, {})
        worker_spec = replica_specs.get(self.worker_replica, {})
        worker_replicas = worker_spec.get("replicas", 0)

        if worker_replicas > 0 and "template" in worker_spec:
            metadata = worker_spec["template"].setdefault("metadata", {})
            worker_labels = metadata.setdefault("labels", {})

            # Ensure worker pods have the same labels as primary pods for consistency
            primary_spec = replica_specs.get(self.primary_replica, {})
            primary_metadata = primary_spec.get("template", {}).get("metadata", {})
            primary_labels = primary_metadata.get("labels", {})

            for key, value in primary_labels.items():
                if key not in worker_labels:
                    worker_labels[key] = value

            worker_labels[serving_constants.KT_SERVICE_LABEL] = service_name
            worker_labels[serving_constants.KT_MODULE_LABEL] = clean_module_name
            metadata.setdefault("annotations", {})["kubetorch.com/deployment_timestamp"] = deployment_timestamp

        return updated

    def _preprocess_manifest_for_launch(self, manifest: dict) -> dict:
        """Sync worker pod spec with primary pod spec if this is a distributed job."""
        spec = manifest.get("spec", {})

        if manifest.get("kind") == "MXJob" and "jobMode" not in spec:
            spec["jobMode"] = "Train"

        # Only sync worker spec if this is a distributed job
        if self.is_distributed(manifest):
            replica_specs = spec.get(self.replica_specs_key, {})
            primary_pod_spec = self.pod_spec(manifest)

            if primary_pod_spec:
                worker_spec = replica_specs.setdefault(self.worker_replica, {}).setdefault("template", {})
                worker_spec["spec"] = copy.deepcopy(primary_pod_spec)

        return manifest

    def _create_or_update_resource(self, manifest: dict, service_name: str, clean_module_name: str, **kwargs) -> dict:
        """Create or update custom resource via CustomObjectsApi."""
        pod_spec = self.pod_spec(manifest)
        server_port = pod_spec.get("containers", [{}])[0].get("ports", [{}])[0].get("containerPort", 32300)

        labels = manifest.get("metadata", {}).get("labels", {})
        annotations = manifest.get("metadata", {}).get("annotations", {})
        service_labels = labels.copy()
        service_labels.pop(serving_constants.KT_TEMPLATE_LABEL, None)

        try:
            service = load_template(
                template_file=serving_constants.DEPLOYMENT_SERVICE_TEMPLATE_FILE,
                template_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"),
                name=service_name,
                namespace=self.namespace,
                annotations=annotations,
                labels=service_labels,
                deployment_name=service_name,
                module_name=clean_module_name,
                distributed=False,  # Regular service for client access
                server_port=server_port,
            )

            service["spec"]["selector"][serving_constants.KT_SERVICE_LABEL] = service_name

            is_distributed = self.is_distributed(manifest)
            if is_distributed:
                service["spec"]["selector"]["training.kubeflow.org/replica-type"] = self.primary_replica.lower()

            try:
                self.controller_client.create_service(
                    namespace=self.namespace,
                    body=service,
                    params=kwargs,
                )
                if not kwargs.get("dry_run"):
                    logger.info(f"Created service {service_name} in namespace {self.namespace}")
            except Exception as e:
                if http_conflict(e):
                    logger.info(f"Service {service_name} already exists")
                else:
                    raise

            if is_distributed:
                # Create headless service for distributed pod discovery
                headless_service = load_template(
                    template_file=serving_constants.DEPLOYMENT_SERVICE_TEMPLATE_FILE,
                    template_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"),
                    name=f"{service_name}-headless",
                    namespace=self.namespace,
                    annotations=annotations,
                    labels=service_labels,
                    deployment_name=service_name,
                    module_name=clean_module_name,
                    distributed=True,  # Headless service for pod discovery
                    server_port=server_port,
                )

                headless_service["spec"]["selector"].pop("training.kubeflow.org/replica-type", None)

                try:
                    self.controller_client.create_service(
                        namespace=self.namespace,
                        body=headless_service,
                        params=kwargs,
                    )
                    if not kwargs.get("dry_run"):
                        logger.info(f"Created headless service {service_name}-headless in namespace {self.namespace}")
                except Exception as e:
                    if http_conflict(e):
                        logger.info(f"Headless service {service_name}-headless already exists")
                    else:
                        raise

            # Create the training job resource (always, not just for distributed)
            try:
                created_resource = self._create_resource(manifest, **kwargs)
            except Exception as e:
                if http_not_found(e):
                    logger.error(
                        "Kubeflow Custom Resource Definition (CRD) not found, please install the Kubeflow Training Operator"
                    )
                raise e

            logger.info(f"Created {manifest.get('kind', 'resource')} {service_name} in namespace {self.namespace}")
            return created_resource
        except Exception as e:
            if http_conflict(e):
                logger.info(f"{manifest.get('kind', 'resource')} {service_name} already exists, updating")
                patch_body = {"spec": manifest["spec"]}
                updated_resource = self._patch_resource(service_name, patch_body, **kwargs)
                logger.info(f"Updated {manifest.get('kind', 'resource')} {service_name}")
                return updated_resource
            raise e

    def check_service_ready(self, service_name: str, launch_timeout: int, **kwargs) -> bool:
        """Check resource readiness by validating job conditions and pod status."""
        sleep_interval = 2
        start_time = time.time()
        resource_kind = self.template_label

        logger.info(f"Checking {resource_kind} {service_name} readiness (timeout: {launch_timeout} seconds)")

        iteration = 0
        while (time.time() - start_time) < launch_timeout:
            iteration += 1
            try:
                resource = self.get_resource(service_name)
                status = resource.get("status", {})
                conditions = status.get("conditions", [])
                spec = resource.get("spec", {})

                for condition in conditions:
                    if condition.get("type") == "Failed" and condition.get("status") == "True":
                        raise RuntimeError(
                            f"{resource_kind} {service_name} failed: {condition.get('message', 'Unknown error')}"
                        )

                # Check if ready
                is_job_running = any(
                    c.get("type") in ("Succeeded", "Running") and c.get("status") == "True" for c in conditions
                )

                if is_job_running or not conditions:
                    counts = self._get_replica_counts(service_name, spec)

                    if len(counts["running_pods"]) >= counts["expected_replicas"] and counts["expected_replicas"] > 0:
                        logger.info(
                            f"{resource_kind} {service_name} is ready with {len(counts['running_pods'])} pods "
                            f"({len(counts['master_pods'])} master, {len(counts['worker_pods'])} worker"
                            f"{'' if len(counts['worker_pods']) == 1 else 's'})"
                        )
                        return True

                # Log progress every 30 seconds
                if iteration % (30 // sleep_interval) == 0:
                    elapsed = int(time.time() - start_time)
                    remaining = launch_timeout - elapsed
                    counts = self._get_replica_counts(service_name, spec)

                    logger.info(
                        f"{resource_kind} is not yet ready (elapsed: {elapsed}s, remaining: {remaining}s). "
                        f"Running pods: {len(counts['running_pods'])}/{counts['expected_replicas']} "
                        f"({len(counts['master_pods'])}/{counts['expected_master']} master, "
                        f"{len(counts['worker_pods'])}/{counts['expected_worker']} worker"
                        f"{'' if counts['expected_worker'] == 1 else 's'})"
                    )

            except RuntimeError:
                raise
            except Exception as e:
                if not http_not_found(e):
                    logger.error(f"Error checking {resource_kind} readiness: {e}")

            time.sleep(sleep_interval)

        raise TimeoutError(f"{resource_kind} {service_name} did not become ready within {launch_timeout} seconds")

    def get_resource(self, service_name: str) -> dict:
        """Get a training job resource by name."""
        return self.controller_client.get_namespaced_custom_object(
            group=self.api_group,
            version=self.api_version,
            namespace=self.namespace,
            plural=self.api_plural,
            name=service_name,
        )

    def update_deployment_timestamp_annotation(self, service_name: str, new_timestamp: str) -> str:
        """Update deployment timestamp annotation for this training job."""
        patch_body = {"metadata": {"annotations": {"kubetorch.com/deployment_timestamp": new_timestamp}}}
        self.controller_client.patch_namespaced_custom_object(
            group=self.api_group,
            version=self.api_version,
            namespace=self.namespace,
            plural=self.api_plural,
            name=service_name,
            body=patch_body,
        )
        return new_timestamp

    def get_endpoint(self, service_name: str) -> str:
        """Get endpoint for primary replica service."""
        return f"http://{service_name}.{self.namespace}.svc.cluster.local:80"

    def _teardown_associated_resources(self, service_name: str, console=None) -> bool:
        """Delete associated Kubernetes Services."""
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
                # Headless service might not exist (non-distributed job), which is fine
                pass
            else:
                if console:
                    console.print(f"[red]Error:[/red] Failed to delete service {headless_service_name}: {e}")
                else:
                    logger.error(f"Failed to delete service {headless_service_name}: {e}")
                success = False

        return success
