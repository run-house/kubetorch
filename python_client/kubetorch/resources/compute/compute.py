import copy
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

import kubetorch.constants as constants
import kubetorch.serving.constants as serving_constants

from kubetorch import data_store, globals
from kubetorch.globals import LoggingConfig

from kubetorch.logger import get_logger
from kubetorch.resources.callables.utils import find_locally_installed_version
from kubetorch.resources.compute.utils import _get_sync_package_paths, _run_bash
from kubetorch.resources.images.image import Image, ImageSetupStepType
from kubetorch.resources.secrets.kubernetes_secrets_client import KubernetesSecretsClient
from kubetorch.resources.volumes.volume import Volume
from kubetorch.servers.http.utils import is_running_in_kubernetes, load_template
from kubetorch.serving.autoscaling import AutoscalingConfig
from kubetorch.serving.utils import pod_is_running

from kubetorch.utils import extract_host_port, load_head_node_pod

logger = get_logger(__name__)


class Compute:
    def __init__(
        self,
        cpus: Union[str, int] = None,
        memory: str = None,
        disk_size: str = None,
        gpus: Union[str, int] = None,
        gpu_type: str = None,
        priority_class_name: str = None,
        gpu_memory: str = None,
        namespace: str = None,
        image: "Image" = None,
        labels: Dict = None,
        annotations: Dict = None,
        volumes: List[Union[str, Volume]] = None,
        node_selector: Dict = None,
        service_template: Dict = None,
        tolerations: List[Dict] = None,
        env_vars: Dict = None,
        secrets: List[Union[str, "Secret"]] = None,
        freeze: bool = False,
        kubeconfig_path: str = None,
        service_account_name: str = None,
        image_pull_policy: str = None,
        inactivity_ttl: str = None,
        gpu_anti_affinity: bool = None,
        launch_timeout: int = None,
        working_dir: str = None,
        shared_memory_limit: str = None,
        allowed_serialization: Optional[List[str]] = None,
        replicas: int = None,
        logging_config: LoggingConfig = None,
        queue_name: str = None,
        _skip_template_init: bool = False,
    ):
        """Initialize the compute requirements for a Kubetorch service.

        Args:
            cpus (str, int, optional): CPU resource request. Can be specified in cores ("1.0") or millicores ("1000m").
            memory (str, optional): Memory resource request. Can use binary (Ki, Mi, Gi) or decimal (K, M, G) units.
            disk_size (str, optional): Ephemeral storage request. Uses same format as memory.
            gpus (str or int, optional): Number of GPUs to request. Fractional GPUs not currently supported.
            gpu_type (str, optional): GPU type to request. Corresponds to the "nvidia.com/gpu.product" label on the
                node (if GPU feature discovery is enabled), or a full string like "nvidia.com/gpu.product: L4" can be
                passed, which will be used to set a `nodeSelector` on the service. More info below.
            gpu_memory (str, optional): GPU memory request (e.g., "4Gi"). Will still request whole GPU but limit
                memory usage.
            priority_class_name (str, optional): Name of the Kubernetes priority class to use for the service. If
                not specified, the default priority class will be used.
            namespace (str, optional): Kubernetes namespace. Defaults to global config default, or "default".
            image (Image, optional): Kubetorch image configuration. See :class:`Image` for more details.
            labels (Dict, optional): Kubernetes labels to apply to the service.
            annotations (Dict, optional): Kubernetes annotations to apply to the service.
            volumes (List[Union[str or Volume]], optional): Volumes to attach to the service. Can be specified as a
                list of volume names (strings) or Volume objects. If using strings, they must be the names of existing
                PersistentVolumeClaims (PVCs) in the specified namespace.
            node_selector (Dict, optional): Kubernetes node selector to constrain pods to specific nodes. Should be a
                dictionary of key-value pairs, e.g. `{"node.kubernetes.io/instance-type": "g4dn.xlarge"}`.
            service_template (Dict, optional): Nested dictionary of service template arguments to apply to the service. E.g.
                ``{"spec": {"template": {"spec": {"nodeSelector": {"node.kubernetes.io/instance-type": "g4dn.xlarge"}}}}}``
            tolerations (List[Dict], optional): Kubernetes tolerations to apply to the service. Each toleration should
                be a dictionary with keys like "key", "operator", "value", and "effect". More info
                `here <https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/>`__.
            env_vars (Dict, optional): Environment variables to set in containers.
            secrets (List[Union[str, Secret]], optional): Secrets to mount or expose.
            freeze (bool, optional): Whether to freeze the compute configuration (e.g. for production).
            kubeconfig_path (str, optional): Path to local kubeconfig file used for cluster authentication.
            service_account_name (str, optional): Kubernetes service account to use.
            image_pull_policy (str, optional): Container image pull policy.
                More info `here <https://kubernetes.io/docs/concepts/containers/images/#image-pull-policy>`__.
            inactivity_ttl (str, optional): Time-to-live after inactivity. Once hit, the service will be destroyed.
                Values below 1m may cause premature deletion.
            gpu_anti_affinity (bool, optional): Whether to prevent scheduling the service on a GPU, should no GPUs be requested.
                Can also control globally by setting the `KT_GPU_ANTI_AFFINITY` environment variable. (Default: ``False``)
            launch_timeout (int, optional): Determines how long to wait for the service to ready before giving up.
                If not specified, will wait {serving_constants.KT_LAUNCH_TIMEOUT} seconds.
                Note: you can also control this timeout globally by setting the `KT_LAUNCH_TIMEOUT` environment variable.
            replicas (int, optional): Number of replicas to create for deployment-based services. Can also be set via
                the `.distribute(workers=N)` method for distributed training. If not specified, defaults to 1 for new
                manifests. (Default: None)
            working_dir (str, optional): Working directory to use inside the remote images. Must be an absolute path (e.g. `/kt`)
            shared_memory_limit (str, optional):  Maximum size of the shared memory filesystem (/dev/shm) available to
                each pod created by the service. Value should be a Kubernetes quantity string, for example: "512Mi",
                "2Gi", "1G", "1024Mi", "100M". If not provided, /dev/shm will default to the pod's memory limit (if set)
                or up to half the node's RAM.
            logging_config (LoggingConfig, optional): Configuration for logging behavior on this service. Controls
                log level, streaming options, and grace periods. See :class:`LoggingConfig` for details.
            queue_name (str, optional): Kueue LocalQueue name for GPU scheduling. When set, adds the
                ``kueue.x-k8s.io/queue-name`` label to the pod template metadata. For training jobs
                (PyTorchJob, TFJob, etc.), also sets ``spec.runPolicy.suspend: true`` so Kueue can
                manage admission. Requires Kueue to be installed in the cluster.

        Note:

            CPUs:
                - Decimal core count: "0.5", "1.0", "2.0"
                - Millicores: "500m", "1000m", "2000m"

            Memory:
                - Bytes: "1000000"
                - Binary units: "1Ki", "1Mi", "1Gi", "1Ti"
                - Decimal units: "1K", "1M", "1G", "1T"

            GPU Specifications:
                1. ``gpus`` for whole GPUs: "1", "2"
                2. ``gpu_memory``: "$Gi", "16Gi"

            Disk Size:
                - Same format as memory

        Note:
            - Memory/disk values are case sensitive (Mi != mi)
            - When using ``gpu_memory``, a whole GPU is still requested but memory is limited

        Examples:

        .. code-block:: python

            import kubetorch as kt

            # Basic CPU/Memory request
            compute = kt.Compute(cpus="0.5", memory="2Gi")

            # GPU request with memory limit
            compute = kt.Compute(gpu_memory="4Gi", cpus="1.0")

            # Multiple whole GPUs
            compute = kt.Compute(gpus="2", memory="16Gi")
        """
        self.default_config = {}

        self._endpoint = None
        self._service_manager = None
        self._autoscaling_config = None
        self._kubeconfig_path = kubeconfig_path

        self._image = image
        self._secrets = secrets
        self._secrets_client = None
        self._volumes = volumes
        self._logging_config = logging_config or LoggingConfig()
        self._manifest = None

        # Skip template initialization if loading from existing service
        if _skip_template_init:
            return

        template_vars = {
            "cpus": cpus,
            "memory": memory,
            "disk_size": disk_size,
            "gpus": gpus,
            "priority_class_name": priority_class_name,
            "gpu_type": gpu_type,
            "gpu_memory": gpu_memory,
            "namespace": namespace,
            "image": image,
            "volumes": volumes,
            "node_selector": node_selector,
            "tolerations": tolerations,
            "env_vars": env_vars,
            "secrets": secrets,
            "freeze": freeze,
            "service_account_name": service_account_name,
            "image_pull_policy": image_pull_policy,
            "inactivity_ttl": inactivity_ttl,
            "gpu_anti_affinity": gpu_anti_affinity,
            "launch_timeout": launch_timeout,
            "working_dir": working_dir,
            "shared_memory_limit": shared_memory_limit,
            "allowed_serialization": allowed_serialization,
        }

        # Build pod spec with defaults
        pod_spec, template_vars = self._build_kubetorch_pod_spec(config=template_vars)

        # Prepare manifest annotations
        manifest_annotations = annotations.copy() if annotations else {}
        gpu_annotations = {"gpu-memory": gpu_memory} if gpu_memory else {}
        manifest_annotations.update(gpu_annotations)
        if self._kubeconfig_path is not None:
            manifest_annotations[serving_constants.KUBECONFIG_PATH_ANNOTATION] = self._kubeconfig_path

        # Build initial manifest based on deployment type
        from kubetorch.serving.service_manager import DeploymentServiceManager

        # Prepare labels, including Kueue queue label if specified
        manifest_labels = labels.copy() if labels else {}
        if queue_name:
            manifest_labels[serving_constants.KUEUE_QUEUE_NAME_LABEL] = queue_name

        self._manifest = DeploymentServiceManager._build_base_manifest(
            pod_spec=pod_spec,
            namespace=template_vars["namespace"],
            replicas=replicas if replicas else 1,
            inactivity_ttl=template_vars["inactivity_ttl"],
            custom_labels=manifest_labels,
            custom_annotations=manifest_annotations,
            custom_template=service_template or {},
        )

    @classmethod
    def from_template(cls, service_info: dict):
        """Create a Compute object from a deployed Kubernetes resource."""
        if "resource" not in service_info:
            raise ValueError("service_info missing required key: resource")

        compute = cls(_skip_template_init=True)
        compute._manifest = service_info["resource"]
        return compute

    @classmethod
    def from_manifest(cls, manifest: Union[Dict, str]):
        """Create a Compute instance from a user-provided Kubernetes manifest.

        The user manifest is used as the baseline, and kubetorch-specific default
        configurations (env vars, labels, annotations, setup script) are merged in
        if missing from the user manifest.

        Args:
            manifest: Kubernetes manifest dict or path to YAML file

        Returns:
            Compute instance

        Examples:

        .. code-block:: python

            import kubetorch as kt

            compute = kt.Compute.from_manifest(user_manifest)

            # Override properties after creation
            compute.cpus = "2"
            compute.image = kt.images.Debian().pip_install(["numpy"])
        """
        # Load manifest from file if provided as a string
        if isinstance(manifest, str):
            with open(manifest, "r") as f:
                manifest = yaml.safe_load(f)

        # Validate manifest
        if "kind" not in manifest or "apiVersion" not in manifest:
            raise ValueError("Manifest must have a 'kind' and 'apiVersion' field")

        # Create instance with minimal init
        compute = cls(_skip_template_init=True)
        compute._manifest = copy.deepcopy(manifest)
        compute._manifest.setdefault("metadata", {})
        compute._manifest.setdefault("spec", {})

        # Extract kubeconfig_path from manifest annotations if present
        user_annotations = compute._manifest["metadata"].get("annotations", {})
        compute._kubeconfig_path = user_annotations.get(serving_constants.KUBECONFIG_PATH_ANNOTATION)

        # Merge kubetorch defaults into user manifest
        compute._build_and_merge_kubetorch_defaults()

        return compute

    def _build_and_merge_kubetorch_defaults(self):
        """Build minimal kubetorch manifest with defaults, then merge user manifest."""
        # Build kubetorch pod spec with empty defaults
        pod_spec, template_vars = self._build_kubetorch_pod_spec(config={})

        # Merge kubetorch pod spec into user pod spec
        # This keeps user values where they exist, adds kubetorch defaults where missing
        self._merge_pod_specs(pod_spec)

        manifest_annotations = {}
        if self._kubeconfig_path is not None:
            manifest_annotations[serving_constants.KUBECONFIG_PATH_ANNOTATION] = self._kubeconfig_path

        # Apply kubetorch-specific metadata to user manifest
        self.service_manager._apply_kubetorch_updates(
            manifest=self._manifest,
            custom_labels={},
            custom_annotations=manifest_annotations,
            inactivity_ttl=template_vars["inactivity_ttl"],
        )

    def _build_kubetorch_pod_spec(self, config: dict):
        """
        Build the kubetorch pod spec with defaults from config.

        Args:
            config: Configuration dict with kwargs (None values mean use defaults)

        Returns:
            tuple: (pod_spec dict, template_vars dict)
        """
        namespace = config.get("namespace") or globals.config.namespace
        server_port = serving_constants.DEFAULT_KT_SERVER_PORT
        service_account_name = config.get("service_account_name") or serving_constants.DEFAULT_SERVICE_ACCOUNT_NAME
        log_streaming_enabled = (
            globals.config.cluster_config.get("log_streaming_enabled", True) if globals.config.cluster_config else True
        )
        metrics_enabled = (
            globals.config.cluster_config.get("metrics_enabled", True) if globals.config.cluster_config else True
        )

        server_image = self._get_server_image(config.get("image") or self._image)

        gpus = None if config.get("gpus") == 0 else config.get("gpus")
        requested_resources = self._get_requested_resources(
            config.get("cpus"),
            config.get("memory"),
            config.get("disk_size"),
            gpus,
        )
        secret_env_vars, secret_volumes = self._extract_secrets(config.get("secrets") or self._secrets, namespace)
        processed_volumes = self._process_volumes(config.get("volumes") or self._volumes)
        volume_mounts, volume_specs = self._volumes_for_pod_template(processed_volumes)

        env_vars = config.get("env_vars") or {}

        # Set KT_LOG_LEVEL from logging_config, falling back to env var
        if not env_vars.get("KT_LOG_LEVEL"):
            if self._logging_config.level:
                env_vars["KT_LOG_LEVEL"] = self._logging_config.level.upper()
            elif os.getenv("KT_LOG_LEVEL"):
                env_vars["KT_LOG_LEVEL"] = os.getenv("KT_LOG_LEVEL")
        if os.getenv("KT_DEBUG_MODE") and not env_vars.get("KT_DEBUG_MODE"):
            # If KT_DEBUG_MODE is set, add it to env vars so the debug mode is set on the server
            env_vars["KT_DEBUG_MODE"] = os.getenv("KT_DEBUG_MODE")

        allowed_serialization = config.get("allowed_serialization")
        if not allowed_serialization:
            allowed_serialization_env_var = os.getenv("KT_ALLOWED_SERIALIZATION", None)
            if allowed_serialization_env_var:
                allowed_serialization = allowed_serialization_env_var.split(",")
        node_selector = self._get_node_selector(config.get("node_selector") or {}, config.get("gpu_type"))

        all_tolerations = self._get_tolerations(gpus, config.get("tolerations") or [])

        template_vars = {
            "server_image": server_image,
            "server_port": server_port,
            "env_vars": env_vars,
            "resources": requested_resources,
            "node_selector": node_selector,
            "secret_env_vars": secret_env_vars or [],
            "secret_volumes": secret_volumes or [],
            "volume_mounts": volume_mounts,
            "volume_specs": volume_specs,
            "service_account_name": service_account_name,
            "config_env_vars": self._get_config_env_vars(allowed_serialization),
            "image_pull_policy": config.get("image_pull_policy"),
            "namespace": namespace,
            "freeze": config.get("freeze", False),
            "gpu_anti_affinity": config.get("gpu_anti_affinity"),
            "working_dir": config.get("working_dir"),
            "tolerations": all_tolerations,
            "shm_size_limit": config.get("shared_memory_limit"),
            "priority_class_name": config.get("priority_class_name"),
            "launch_timeout": self._get_launch_timeout(config.get("launch_timeout")),
            "inactivity_ttl": config.get("inactivity_ttl"),
            "log_streaming_enabled": log_streaming_enabled,
            "metrics_enabled": metrics_enabled,
            "setup_script": "",
        }

        pod_spec = load_template(
            template_file=serving_constants.POD_TEMPLATE_FILE,
            template_dir=os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "serving",
                "templates",
            ),
            **template_vars,
        )
        return pod_spec, template_vars

    # ----------------- Helper Methods ----------------- #

    def _get_manifest_metadata(self):
        """Get metadata from the primary resource in the manifest."""
        # Return metadata from the manifest itself
        return self._manifest.get("metadata", {})

    def _merge_pod_specs(self, kt_pod_spec: dict):
        """Merge kubetorch pod spec into user pod spec.

        User values take precedence over kubetorch defaults. For lists and dicts,
        kubetorch values are added where user doesn't have them, but existing user
        values are preserved.
        """
        from kubetorch.serving.trainjob_service_manager import TrainJobServiceManager
        from kubetorch.serving.utils import nested_merge

        if self.kind in TrainJobServiceManager.SUPPORTED_KINDS:
            config = TrainJobServiceManager._get_config(self.kind)
            container_name = config["container_name"]
        else:
            container_name = "kubetorch"
        if not self.pod_spec or not self.pod_spec.get("containers"):
            self.pod_spec = kt_pod_spec
            return

        # Merge container
        user_container = self._container()
        kt_container = kt_pod_spec.get("containers", [None])[0]

        if kt_container:
            # Special handling for resources: only merge if kubetorch has actual values
            kt_resources = kt_container.get("resources")
            if kt_resources and isinstance(kt_resources, dict):
                requests = kt_resources.get("requests")
                limits = kt_resources.get("limits")
                # Filter out None values and check if any remain
                filtered_requests = (
                    {k: v for k, v in requests.items() if v is not None}
                    if (requests and isinstance(requests, dict))
                    else {}
                )
                filtered_limits = (
                    {k: v for k, v in limits.items() if v is not None} if (limits and isinstance(limits, dict)) else {}
                )

                if not (filtered_requests or filtered_limits):
                    # No actual resource values, remove from kt_container before merging
                    kt_container = copy.deepcopy(kt_container)
                    kt_container.pop("resources", None)

            # Use general merge function - user values take precedence
            nested_merge(user_container, kt_container)

        # Set main container name to kubetorch container name
        user_container["name"] = container_name

        # Merge pod-level fields
        # Remove containers from kt_pod_spec since we've already merged it
        kt_pod_spec_for_merge = copy.deepcopy(kt_pod_spec)
        kt_pod_spec_for_merge.pop("containers", None)

        # Use general merge for all pod-level fields
        nested_merge(self.pod_spec, kt_pod_spec_for_merge)

        # Store the merged pod spec for use in worker replicas
        merged = self.pod_spec

        # For RayCluster, copy the merged head pod spec to the worker group
        if self.kind == "RayCluster":
            spec = self._manifest.get("spec", {})
            worker_group_specs = spec.get("workerGroupSpecs", [])
            if worker_group_specs:
                worker_group = worker_group_specs[0]
                worker_template = worker_group.setdefault("template", {})
                worker_template["spec"] = copy.deepcopy(merged)

        # For training jobs, ensure worker replicas
        if self.kind in ["PyTorchJob", "TFJob", "MXJob", "XGBoostJob"]:
            service_manager = self.service_manager
            spec = self._manifest.get("spec", {})
            replica_specs = spec.get(service_manager.replica_specs_key, {})
            worker_spec = replica_specs.get(service_manager.worker_replica, {})
            if worker_spec:
                worker_template = worker_spec.setdefault("template", {})
                worker_pod_spec = worker_template.setdefault("spec", {})

                # Merge all pod-level fields from merged spec minus containers
                merged_without_containers = {k: v for k, v in merged.items() if k != "containers"}
                nested_merge(worker_pod_spec, merged_without_containers)

    # ----------------- Properties ----------------- #
    @property
    def kubeconfig_path(self):
        if self._kubeconfig_path is None:
            self._kubeconfig_path = os.getenv("KUBECONFIG") or constants.DEFAULT_KUBECONFIG_PATH
        return str(Path(self._kubeconfig_path).expanduser())

    @kubeconfig_path.setter
    def kubeconfig_path(self, value: str):
        """Set kubeconfig_path and update the manifest annotations."""
        self._kubeconfig_path = value

        self._manifest.setdefault("metadata", {}).setdefault("annotations", {})

        if value is not None:
            self._manifest["metadata"]["annotations"][serving_constants.KUBECONFIG_PATH_ANNOTATION] = value
        elif serving_constants.KUBECONFIG_PATH_ANNOTATION in self._manifest["metadata"].get("annotations", {}):
            del self._manifest["metadata"]["annotations"][serving_constants.KUBECONFIG_PATH_ANNOTATION]

    @property
    def manifest(self):
        """Get the current resource manifest."""
        return self._manifest

    @property
    def logging_config(self) -> LoggingConfig:
        """Get the logging configuration for this compute."""
        return self._logging_config

    @logging_config.setter
    def logging_config(self, value: LoggingConfig):
        """Set the logging configuration for this compute."""
        self._logging_config = value

        if self._manifest is not None:
            try:
                container = self._container()
                if value and value.level:
                    env_vars = {"KT_LOG_LEVEL": value.level.upper()}
                    self._set_env_vars_in_container(container, env_vars)
            except (ValueError, AttributeError):
                pass

    @property
    def pod_spec(self):
        """Get the pod spec from the manifest."""
        template_path = self.service_manager.get_pod_template_path()
        path = template_path + ["spec"]

        if path:
            current = self._manifest
            for key in path:
                if current is None:
                    return None
                current = current.get(key)
            return current
        return None

    @pod_spec.setter
    def pod_spec(self, value: dict):
        """Set the pod spec in the manifest."""
        template_path = self.service_manager.get_pod_template_path()
        path = template_path + ["spec"]

        if path:
            current = self._manifest
            for key in path[:-1]:
                current = current.setdefault(key, {})
            current[path[-1]] = value

    @property
    def service_manager(self):
        if self._service_manager is None:
            from kubetorch.serving.deployment_service_manager import DeploymentServiceManager
            from kubetorch.serving.knative_service_manager import KnativeServiceManager
            from kubetorch.serving.raycluster_service_manager import RayClusterServiceManager
            from kubetorch.serving.trainjob_service_manager import TrainJobServiceManager

            service_manager_mapping = {
                "deployment": DeploymentServiceManager,
                "knative": KnativeServiceManager,
                "raycluster": RayClusterServiceManager,
            }
            kwargs = {
                "namespace": self.namespace,
            }
            if self.deployment_mode not in service_manager_mapping:
                kwargs["kind"] = self.kind
                self._service_manager = TrainJobServiceManager(**kwargs)
            else:
                self._service_manager = service_manager_mapping[self.deployment_mode](**kwargs)
        return self._service_manager

    @property
    def secrets_client(self):
        if not self.secrets:
            # Skip creating secrets client if no secrets are provided
            return None

        if self._secrets_client is None:
            self._secrets_client = KubernetesSecretsClient(
                namespace=self.namespace, kubeconfig_path=self.kubeconfig_path
            )
        return self._secrets_client

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, value: "Image"):
        self._image = value
        if self._image.image_id:
            self.server_image = self._image.image_id

    @property
    def endpoint(self):
        if self._endpoint is None and self.service_name:
            self._endpoint = self.service_manager.get_endpoint(self.service_name)
        return self._endpoint

    @endpoint.setter
    def endpoint(self, endpoint: str):
        self._endpoint = endpoint

    def _container(self):
        """Get the container from the pod spec."""
        if "containers" not in self.pod_spec:
            raise ValueError("pod_spec missing 'containers' field.")

        from kubetorch.serving.trainjob_service_manager import TrainJobServiceManager

        expected_name = (
            self.kind.lower().replace("job", "") if self.kind in TrainJobServiceManager.SUPPORTED_KINDS else "kubetorch"
        )
        containers = self.pod_spec.get("containers")
        for container in containers:
            if container.get("name") == expected_name:
                return container
        # If no properly named container found, return the first container
        return containers[0]

    def _container_env(self):
        container = self._container()
        if "env" not in container:
            return []
        return container["env"]

    def _set_env_vars_in_container(self, container: dict, env_vars: dict) -> None:
        """Set or update environment variables in a container spec.

        Updates existing env vars if they exist, otherwise appends new ones.
        Modifies the container dict in place.
        """
        container.setdefault("env", [])

        # Track which env vars we've updated
        updated_names = set()

        # Update existing env vars
        for env_var in container["env"]:
            env_name = env_var.get("name")
            if env_name in env_vars:
                env_var["value"] = env_vars[env_name]
                updated_names.add(env_name)

        # Append any env vars that weren't found
        for name, value in env_vars.items():
            if name not in updated_names:
                container["env"].append({"name": name, "value": value})

    def _set_container_resource(self, resource_name: str, value: str, limits: bool = False):
        container = self._container()

        # Ensure resources dict exists
        container.setdefault("resources", {})

        # Ensure requests dict exists
        if not container["resources"].get("requests"):
            container["resources"]["requests"] = {}

        container["resources"]["requests"][resource_name] = value

        if limits:
            container["resources"].setdefault("limits", {})
            container["resources"]["limits"][resource_name] = value

    def _get_container_resource(self, resource_name: str) -> Optional[str]:
        resources = self._container().get("resources", {})
        requests = resources.get("requests") or {}
        return requests.get(resource_name)

    # -------------- Properties From Template -------------- #
    @property
    def server_image(self):
        return self._container().get("image")

    @server_image.setter
    def server_image(self, value: str):
        """Set the server image in the pod spec."""
        self._container()["image"] = value

    @property
    def server_port(self):
        container = self._container()
        ports = container.get("ports", [])
        if ports and len(ports) > 0:
            return ports[0].get("containerPort")
        # Return default if ports not found
        import kubetorch.serving.constants as serving_constants

        return serving_constants.DEFAULT_KT_SERVER_PORT

    @server_port.setter
    def server_port(self, value: int):
        """Set the server port in the pod spec."""
        container = self._container()
        container.setdefault("ports", [])
        if len(container["ports"]) == 0:
            container["ports"].append({})
        container["ports"][0]["containerPort"] = value

    @property
    def env_vars(self):
        # extract user-defined environment variables from rendered pod spec
        kt_env_vars = [
            "POD_NAME",
            "POD_NAMESPACE",
            "POD_IP",
            "POD_UUID",
            "MODULE_NAME",
            "KUBETORCH_VERSION",
            "UV_LINK_MODE",
            "OTEL_SERVICE_NAME",
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "OTEL_EXPORTER_OTLP_PROTOCOL",
            "OTEL_TRACES_EXPORTER",
            "OTEL_PROPAGATORS",
            "KT_SERVER_PORT",
            "KT_FREEZE",
            "KT_INACTIVITY_TTL",
            "KT_ALLOWED_SERIALIZATION",
            "KT_FILE_PATH",
            "KT_MODULE_NAME",
            "KT_CLS_OR_FN_NAME",
            "KT_CALLABLE_TYPE",
            "KT_LAUNCH_ID",
            "KT_SERVICE_NAME",
            "KT_SERVICE_DNS",
        ]
        user_env_vars = {}
        for env_var in self._container_env():
            # skip if it was set by kubetorch
            if env_var["name"] not in kt_env_vars and "value" in env_var:
                user_env_vars[env_var["name"]] = env_var["value"]
        return user_env_vars

    @property
    def resources(self):
        return self._container().get("resources")

    @property
    def cpus(self):
        return self._get_container_resource("cpu")

    @cpus.setter
    def cpus(self, value: str):
        """
        Args:
            value: CPU value (e.g., "2", "1000m", "0.5")
        """
        self._set_container_resource("cpu", str(value))

    @property
    def memory(self):
        return self._get_container_resource("memory")

    @memory.setter
    def memory(self, value: str):
        """
        Args:
            value: Memory value (e.g., "4Gi", "2048Mi")
        """
        self._set_container_resource("memory", str(value))

    @property
    def disk_size(self):
        return self._get_container_resource("ephemeral-storage")

    @disk_size.setter
    def disk_size(self, value: str):
        """
        Args:
            value: Disk size (e.g., "10Gi", "5000Mi")
        """
        self._set_container_resource("ephemeral-storage", value)

    @property
    def gpus(self):
        return self._get_container_resource("nvidia.com/gpu")

    @gpus.setter
    def gpus(self, value: Union[str, int]):
        """
        Args:
            value: Number of GPUs (e.g., 1, "2")
        """
        self._set_container_resource("nvidia.com/gpu", str(value), limits=True)

    @property
    def gpu_type(self):
        node_selector = self.pod_spec.get("nodeSelector") or {}
        return node_selector.get("nvidia.com/gpu.product")

    @gpu_type.setter
    def gpu_type(self, value: str):
        """
        Args:
            value: GPU product name (e.g., "L4", "V100", "A100", "T4")
        """
        self.pod_spec.setdefault("nodeSelector", {})
        self.pod_spec["nodeSelector"]["nvidia.com/gpu.product"] = value

    @property
    def gpu_memory(self):
        annotations = {}
        manifest_metadata = self._manifest.get("metadata", {})
        if manifest_metadata:
            annotations = manifest_metadata.get("annotations", {})
        return annotations.get("gpu-memory")

    @gpu_memory.setter
    def gpu_memory(self, value: str):
        """
        Args:
            value: GPU memory in MiB (e.g., "4096", "8192", "16384")
        """
        self._manifest.setdefault("metadata", {}).setdefault("annotations", {})

        if value is not None:
            self._manifest["metadata"]["annotations"]["gpu-memory"] = value
        elif "gpu-memory" in self._manifest["metadata"].get("annotations", {}):
            del self._manifest["metadata"]["annotations"]["gpu-memory"]

    @property
    def volumes(self):
        if not self._volumes:
            volumes = []
            if "volumes" in self.pod_spec:
                for volume in self.pod_spec["volumes"]:
                    # Skip the shared memory volume
                    if volume["name"] == "dshm":
                        continue
                    # Skip secret volumes
                    if "secret" in volume:
                        continue
                    # Only include PVC volumes
                    if "persistentVolumeClaim" in volume:
                        volumes.append(volume["name"])
            self._volumes = volumes
        return self._volumes

    @volumes.setter
    def volumes(self, value: List[Union[str, Volume]]):
        """
        Set volumes and update the manifest pod spec.

        Args:
            value: List of volumes (strings or Volume objects) to add to existing volumes
        """
        processed_volumes = self._process_volumes(value)
        new_volume_names = [vol.name for vol in (processed_volumes or [])]
        self._volumes = list(dict.fromkeys((self._volumes or []) + new_volume_names))

        volume_mounts, volume_specs = self._volumes_for_pod_template(processed_volumes)

        container = self._container()
        volume_mounts_list = container.setdefault("volumeMounts", [])
        existing_mount_names = {
            vm["name"] for vm in volume_mounts_list if vm["name"] != "dshm" and not vm["name"].startswith("secrets-")
        }

        for mount in volume_mounts or []:
            if mount["name"] not in existing_mount_names:
                volume_mounts_list.append(mount)
                existing_mount_names.add(mount["name"])

        volumes_list = self.pod_spec.setdefault("volumes", [])
        existing_vol_names = {
            vol.get("name") for vol in volumes_list if vol.get("name") != "dshm" and "secret" not in vol
        }

        for vol_spec in volume_specs or []:
            vol_name = vol_spec.get("name")
            if vol_name and vol_name not in existing_vol_names:
                volumes_list.append(vol_spec)
                existing_vol_names.add(vol_name)

    @property
    def shared_memory_limit(self):
        if "volumes" not in self.pod_spec:
            return None

        for volume in self.pod_spec["volumes"]:
            if volume.get("name") == "dshm" and "emptyDir" in volume:
                empty_dir = volume["emptyDir"] or {}
                return empty_dir.get("sizeLimit")

        return None

    @shared_memory_limit.setter
    def shared_memory_limit(self, value: str):
        """
        Args:
            value: Size limit (e.g., "512Mi", "1Gi", "2G")
        """
        self.pod_spec.setdefault("volumes", [])

        # Find existing dshm volume and update it
        for volume in self.pod_spec["volumes"]:
            if volume.get("name") == "dshm" and "emptyDir" in volume:
                volume["emptyDir"]["sizeLimit"] = value
                return

        # Add new dshm volume if not found
        self.pod_spec["volumes"].append({"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": value}})

    # Alias for backward compatibility (deprecated)
    @property
    def shm_size_limit(self):
        """Deprecated: Use shared_memory_limit instead."""
        return self.shared_memory_limit

    @shm_size_limit.setter
    def shm_size_limit(self, value: str):
        """Deprecated: Use shared_memory_limit instead."""
        self.shared_memory_limit = value

    @property
    def node_selector(self):
        return self.pod_spec.get("nodeSelector")

    @node_selector.setter
    def node_selector(self, value: dict):
        """
        Args:
            value: Label key-value pairs (e.g., {"node-type": "gpu"})
        """
        self.pod_spec["nodeSelector"] = value

    @property
    def secret_env_vars(self):
        secret_env_vars = []
        container = self._container()
        if "env" in container:
            for env_var in container["env"]:
                if "valueFrom" in env_var and "secretKeyRef" in env_var["valueFrom"]:
                    secret_ref = env_var["valueFrom"]["secretKeyRef"]
                    # Find existing secret or create new entry
                    secret_name = secret_ref["name"]

                    # Check if we already have this secret
                    existing_secret = None
                    for secret in secret_env_vars:
                        if secret.get("secret_name") == secret_name:
                            existing_secret = secret
                            break

                    if existing_secret:
                        existing_secret.setdefault("env_vars", [])
                        if env_var["name"] not in existing_secret["env_vars"]:
                            existing_secret["env_vars"].append(env_var["name"])
                    else:
                        secret_env_vars.append({"secret_name": secret_name, "env_vars": [env_var["name"]]})
        return secret_env_vars

    @property
    def secret_volumes(self):
        secret_volumes = []
        if "volumes" in self.pod_spec:
            for volume in self.pod_spec["volumes"]:
                if "secret" in volume:
                    secret_name = volume["secret"]["secretName"]
                    # Find corresponding volume mount
                    mount_path = None
                    container = self._container()
                    if "volumeMounts" in container:
                        for mount in container["volumeMounts"]:
                            if mount["name"] == volume["name"]:
                                mount_path = mount["mountPath"]
                                break

                    secret_volumes.append(
                        {
                            "name": volume["name"],
                            "secret_name": secret_name,
                            "path": mount_path or f"/secrets/{volume['name']}",
                        }
                    )
        return secret_volumes

    @property
    def volume_mounts(self):
        volume_mounts = []
        container = self._container()
        if "volumeMounts" in container:
            for mount in container["volumeMounts"]:
                # Skip the default dshm mount
                if mount["name"] != "dshm":
                    volume_mounts.append({"name": mount["name"], "mountPath": mount["mountPath"]})
        return volume_mounts

    @property
    def service_account_name(self):
        return self.pod_spec.get("serviceAccountName")

    @service_account_name.setter
    def service_account_name(self, value: str):
        """Set service account name in the pod spec."""
        self.pod_spec["serviceAccountName"] = value

    @property
    def config_env_vars(self):
        from kubetorch.config import ENV_MAPPINGS

        config_env_vars = {}
        container = self._container()
        if "env" in container:
            for env_var in container["env"]:
                # Filter for config-related env vars (those that start with KT_ or are known config vars)
                if env_var["name"] in ENV_MAPPINGS.keys():
                    if "value" in env_var and env_var["value"]:
                        config_env_vars[env_var["name"]] = env_var["value"]
        return config_env_vars

    @property
    def image_pull_policy(self):
        return self._container().get("imagePullPolicy")

    @image_pull_policy.setter
    def image_pull_policy(self, value: str):
        """Set image pull policy in the pod spec."""
        self._container()["imagePullPolicy"] = value

    @property
    def kind(self):
        """Get the manifest kind, defaulting to 'Deployment' if not set."""
        if self._manifest is None:
            return "Deployment"
        return self._manifest.get("kind", "Deployment")

    @property
    def namespace(self):
        return self._manifest.get("metadata", {}).get("namespace") if self._manifest else globals.config.namespace

    @namespace.setter
    def namespace(self, value: str):
        self._manifest.setdefault("metadata", {})
        self._manifest["metadata"]["namespace"] = value

    @property
    def python_path(self):
        if self.image and self.image.python_path:
            return self.image.python_path

        container = self._container()
        if "env" in container:
            for env_var in container["env"]:
                if env_var["name"] == "KT_PYTHON_PATH" and "value" in env_var:
                    return env_var["value"]
        return None

    @property
    def freeze(self):
        container = self._container()
        if "env" in container:
            for env_var in container["env"]:
                if env_var["name"] == "KT_FREEZE" and "value" in env_var:
                    return env_var["value"].lower() == "true"
        return False

    @freeze.setter
    def freeze(self, value: bool):
        if value == self.freeze:
            return

        container = self._container()

        self._set_env_vars_in_container(container, {"KT_FREEZE": str(value).lower()})

        # Update securityContext from pod template based on freeze value
        if value:  # Remove SYS_PTRACE capability when freeze is enabled
            security_context = container.get("securityContext")
            if not security_context:
                return

            capabilities_dict = security_context.get("capabilities")
            if not capabilities_dict:
                return

            capabilities = capabilities_dict.get("add")
            if not capabilities or "SYS_PTRACE" not in capabilities:
                return

            capabilities.remove("SYS_PTRACE")
            if not capabilities:  # empty after removal
                del capabilities_dict["add"]
                if not capabilities_dict:
                    del security_context["capabilities"]
        else:  # When freeze is False, ensure SYS_PTRACE capability is present
            capabilities = (
                container.setdefault("securityContext", {}).setdefault("capabilities", {}).setdefault("add", [])
            )
            if "SYS_PTRACE" not in capabilities:
                capabilities.append("SYS_PTRACE")

    @property
    def allowed_serialization(self):
        """Get allowed_serialization from the container's KT_ALLOWED_SERIALIZATION env var."""
        container = self._container()
        if "env" in container:
            for env_var in container["env"]:
                if env_var["name"] == "KT_ALLOWED_SERIALIZATION" and "value" in env_var:
                    value = env_var["value"]
                    if value:
                        return value.split(",")
        return None

    @allowed_serialization.setter
    def allowed_serialization(self, value: Optional[List[str]]):
        """Set allowed_serialization and update the manifest pod spec."""
        container = self._container()

        if value:
            env_value = ",".join(value)
            self._set_env_vars_in_container(container, {"KT_ALLOWED_SERIALIZATION": env_value})
        else:
            if "env" in container:
                container["env"] = [
                    env_var for env_var in container["env"] if env_var.get("name") != "KT_ALLOWED_SERIALIZATION"
                ]

    @property
    def secrets(self):
        if not self._secrets:
            secrets = []

            # Extract secrets from environment variables
            container = self._container()
            if "env" in container:
                for env_var in container["env"]:
                    if "valueFrom" in env_var and "secretKeyRef" in env_var["valueFrom"]:
                        secret_ref = env_var["valueFrom"]["secretKeyRef"]
                        if secret_ref["name"] not in secrets:
                            secrets.append(secret_ref["name"])

            # Extract secrets from volumes
            if "volumes" in self.pod_spec:
                for volume in self.pod_spec["volumes"]:
                    if "secret" in volume:
                        secret_name = volume["secret"]["secretName"]
                        if secret_name not in secrets:
                            secrets.append(secret_name)

            self._secrets = secrets

        return self._secrets

    @secrets.setter
    def secrets(self, value: List[Union[str, "Secret"]]):
        """
        Set secrets and update the manifest pod spec.

        Args:
            value: List of secrets (strings or Secret objects) to add to existing secrets
        """
        # Combine secrets avoiding duplicates
        seen_names = set()
        combined_secrets = []
        for secret in (self._secrets or []) + (value or []):
            name = secret.name if hasattr(secret, "name") else secret
            if name not in seen_names:
                combined_secrets.append(secret)
                seen_names.add(name)
        self._secrets = combined_secrets

        secret_env_vars, secret_volumes = self._extract_secrets(value, self.namespace)

        container = self._container()
        env_list = container.setdefault("env", [])
        existing_env_names = {
            env_var["name"] for env_var in env_list if env_var.get("valueFrom", {}).get("secretKeyRef")
        }

        for secret_dict in secret_env_vars or []:
            for key in secret_dict["env_vars"]:
                if key not in existing_env_names:
                    env_list.append(
                        {"name": key, "valueFrom": {"secretKeyRef": {"name": secret_dict["secret_name"], "key": key}}}
                    )
                    existing_env_names.add(key)

        volume_mounts_list = container.setdefault("volumeMounts", [])
        existing_mount_names = {vm["name"] for vm in volume_mounts_list if vm["name"].startswith("secrets-")}

        for secret_dict in secret_volumes or []:
            if secret_dict["name"] not in existing_mount_names:
                volume_mounts_list.append(
                    {"name": secret_dict["name"], "mountPath": secret_dict["path"], "readOnly": True}
                )
                existing_mount_names.add(secret_dict["name"])

        volumes_list = self.pod_spec.setdefault("volumes", [])
        existing_vol_names = {vol.get("name") for vol in volumes_list if "secret" in vol}

        for secret_dict in secret_volumes or []:
            if secret_dict["name"] not in existing_vol_names:
                volumes_list.append({"name": secret_dict["name"], "secret": {"secretName": secret_dict["secret_name"]}})
                existing_vol_names.add(secret_dict["name"])

    @property
    def gpu_annotations(self):
        return {"gpu-memory": self.gpu_memory} if self.gpu_memory else {}

    @property
    def gpu_anti_affinity(self):
        affinity = self.pod_spec.get("affinity", {})
        node_affinity = affinity.get("nodeAffinity", {})
        required = node_affinity.get("requiredDuringSchedulingIgnoredDuringExecution", {})
        node_selector_terms = required.get("nodeSelectorTerms", [])

        for term in node_selector_terms:
            match_expressions = term.get("matchExpressions", [])
            for expr in match_expressions:
                if expr.get("key") == "nvidia.com/gpu" and expr.get("operator") == "DoesNotExist":
                    return True
        return False

    @gpu_anti_affinity.setter
    def gpu_anti_affinity(self, value: bool):
        if value:
            self.pod_spec.setdefault("affinity", {}).setdefault("nodeAffinity", {}).setdefault(
                "requiredDuringSchedulingIgnoredDuringExecution", {}
            )
            self.pod_spec["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"][
                "nodeSelectorTerms"
            ] = [{"matchExpressions": [{"key": "nvidia.com/gpu", "operator": "DoesNotExist"}]}]
        else:
            # Remove the affinity rule when value is False
            affinity = self.pod_spec.get("affinity")
            if not affinity:
                return

            node_affinity = affinity.get("nodeAffinity")
            if not node_affinity:
                return

            required = node_affinity.get("requiredDuringSchedulingIgnoredDuringExecution")
            if not required:
                return

            node_selector_terms = required.get("nodeSelectorTerms")
            if not node_selector_terms:
                return

            # Filter out GPU DoesNotExist expressions from each term
            updated_terms = []
            for term in node_selector_terms:
                match_expressions = term.get("matchExpressions", [])
                if match_expressions:
                    filtered = [
                        expr
                        for expr in match_expressions
                        if not (expr.get("key") == "nvidia.com/gpu" and expr.get("operator") == "DoesNotExist")
                    ]
                    if filtered:  # Keep term only if it has remaining expressions
                        updated_terms.append({**term, "matchExpressions": filtered})
                else:
                    updated_terms.append(term)  # Keep terms without matchExpressions

            # Update or remove nodeSelectorTerms
            if updated_terms:
                required["nodeSelectorTerms"] = updated_terms
            else:
                # Clean up empty structures
                del required["nodeSelectorTerms"]
                if not required:
                    del node_affinity["requiredDuringSchedulingIgnoredDuringExecution"]
                    if not node_affinity:
                        del affinity["nodeAffinity"]
                        if not affinity:
                            del self.pod_spec["affinity"]

    @property
    def concurrency(self):
        return self.pod_spec.get("containerConcurrency")

    @concurrency.setter
    def concurrency(self, value: int):
        self.pod_spec["containerConcurrency"] = value

    @property
    def working_dir(self):
        return self._container().get("workingDir")

    @working_dir.setter
    def working_dir(self, value: str):
        """Set working directory in the pod spec."""
        self._container()["workingDir"] = value

    @property
    def priority_class_name(self):
        return self.pod_spec.get("priorityClassName")

    @priority_class_name.setter
    def priority_class_name(self, value: str):
        """Set priority class name in the pod spec."""
        self.pod_spec["priorityClassName"] = value

    @property
    def metrics_enabled(self):
        container = self._container()
        if "env" in container:
            for env_var in container["env"]:
                if env_var["name"] == "KT_METRICS_ENABLED" and "value" in env_var:
                    return env_var["value"].lower() == "true"
        return True  # Default to True

    @property
    def launch_timeout(self):
        container = self._container()
        if "startupProbe" in container:
            startup_probe = container["startupProbe"]
            if "failureThreshold" in startup_probe:
                # Convert back from failure threshold (launch_timeout // 5)
                return startup_probe["failureThreshold"] * 5
        return None

    @launch_timeout.setter
    def launch_timeout(self, value: int):
        container = self._container()
        container.setdefault("startupProbe", {})
        # Convert timeout to failure threshold (launch_timeout // 5)
        container["startupProbe"]["failureThreshold"] = value // 5

    @property
    def inactivity_ttl(self):
        container = self._container()
        if "env" in container:
            for env_var in container["env"]:
                if env_var["name"] == "KT_INACTIVITY_TTL" and "value" in env_var:
                    return env_var["value"] if not env_var["value"] == "None" else None

        # Try to extract from manifest
        metadata = self._get_manifest_metadata()
        annotations = metadata.get("annotations", {})
        if "kubetorch.com/inactivity-ttl" in annotations:
            return annotations["kubetorch.com/inactivity-ttl"]

        return None

    @inactivity_ttl.setter
    def inactivity_ttl(self, value: str):
        if value and (not isinstance(value, str) or not re.match(r"^\d+[smhd]$", value)):
            raise ValueError("Inactivity TTL must be a string, e.g. '5m', '1h', '1d'")
        if value and not self.metrics_enabled:
            logger.warning(
                "Inactivity TTL requires metrics collection to be enabled. " "Please update your Kubetorch Helm chart."
            )

        container = self._container()
        container.setdefault("env", [])

        # Find existing KT_INACTIVITY_TTL env var and update it
        for env_var in container["env"]:
            if env_var["name"] == "KT_INACTIVITY_TTL":
                env_var["value"] = value if value is not None else "None"
                return

        # Add new env var if not found
        container["env"].append(
            {
                "name": "KT_INACTIVITY_TTL",
                "value": value if value is not None else "None",
            }
        )

    @property
    def name(self):
        container = self._container()
        if "env" in container:
            for env_var in container["env"]:
                if env_var["name"] == "KT_SERVICE_NAME" and "value" in env_var:
                    return env_var["value"] if not env_var["value"] == "None" else None
        return None

    @property
    def raycluster(self):
        container = self._container()
        if "ports" in container:
            for port in container["ports"]:
                if port.get("name") == "ray-gcs":
                    return True
        return False

    @property
    def autoscaling_config(self):
        """Get autoscaling config from manifest or stored value."""
        if self._autoscaling_config is not None:
            return self._autoscaling_config

        from kubetorch.serving.autoscaling import AutoscalingConfig

        try:
            return AutoscalingConfig.from_manifest(self._manifest)
        except Exception:
            return {}

    @property
    def distributed_config(self):
        # First try to get from pod spec
        template_config = {}
        container = self._container()
        if "env" in container:
            for env_var in container["env"]:
                if env_var["name"] == "KT_DISTRIBUTED_CONFIG" and "value" in env_var and env_var["value"]:
                    import json

                    try:
                        template_config = json.loads(env_var["value"])
                    except (json.JSONDecodeError, TypeError):
                        template_config = env_var["value"]
                    break

        return template_config

    @distributed_config.setter
    def distributed_config(self, config: dict):
        """Set distributed config in all containers."""
        import json

        from kubetorch.serving.trainjob_service_manager import TrainJobServiceManager

        # Populate defaults if config is missing values
        workers = config.get("workers")
        config["distribution_type"] = config.get("distribution_type", "spmd")
        config["quorum_timeout"] = config.get("quorum_timeout", self.launch_timeout)
        config["quorum_workers"] = config.get("quorum_workers", workers or self.replicas)
        self.replicas = workers or config["quorum_workers"]

        # Serialize the config to JSON, ensuring it's always a string
        # Check for non-serializable values and raise an error with details
        non_serializable_keys = []
        for key, value in config.items():
            try:
                json.dumps(value)
            except (TypeError, ValueError) as e:
                non_serializable_keys.append(f"'{key}': {type(value).__name__} - {str(e)}")

        if non_serializable_keys:
            raise ValueError(
                f"Distributed config contains non-serializable values: {', '.join(non_serializable_keys)}. "
                f"All values must be JSON serializable (strings, numbers, booleans, lists, dicts)."
            )

        distribution_type = config.get("distribution_type", "spmd")
        service_dns = None
        if distribution_type == "ray":
            service_dns = "ray-head-svc"
        elif distribution_type == "pytorch":
            service_dns = "rank0"

        # Prepare env vars to set
        env_vars_to_set = {"KT_DISTRIBUTED_CONFIG": json.dumps(config)}
        if service_dns:
            env_vars_to_set["KT_SERVICE_DNS"] = service_dns

        # For training jobs (PyTorchJob, TFJob, etc.), set in both Master and Worker replica containers
        if self.kind in TrainJobServiceManager.SUPPORTED_KINDS:
            service_manager = self.service_manager
            spec = self._manifest.get("spec", {})
            replica_specs = spec.get(service_manager.replica_specs_key, {})

            for replica_name in [service_manager.primary_replica, service_manager.worker_replica]:
                replica_spec = replica_specs.get(replica_name, {})
                pod_spec = replica_spec.get("template", {}).get("spec", {})
                containers = pod_spec.get("containers", [])

                for container in containers:
                    self._set_env_vars_in_container(container, env_vars_to_set)
        # For RayCluster, set in both head and worker group containers
        elif self.kind == "RayCluster":
            spec = self._manifest.get("spec", {})

            # Set in head group
            head_spec = spec.get("headGroupSpec", {})
            head_pod_spec = head_spec.get("template", {}).get("spec", {})
            head_containers = head_pod_spec.get("containers", [])
            for container in head_containers:
                self._set_env_vars_in_container(container, env_vars_to_set)

            # Set in worker groups
            worker_group_specs = spec.get("workerGroupSpecs", [])
            for worker_group in worker_group_specs:
                worker_pod_spec = worker_group.get("template", {}).get("spec", {})
                worker_containers = worker_pod_spec.get("containers", [])
                for container in worker_containers:
                    self._set_env_vars_in_container(container, env_vars_to_set)
        else:
            # Regular manifest - set config in main container
            container = self._container()
            self._set_env_vars_in_container(container, env_vars_to_set)

    @property
    def deployment_mode(self):
        if self.kind == "Service":
            return "knative"
        return self.kind.lower()

    # ----------------- Service Level Properties ----------------- #

    @property
    def service_name(self):
        return self._manifest.get("metadata", {}).get("name")

    @service_name.setter
    def service_name(self, value: str):
        current_name = self._manifest.get("metadata", {}).get("name")
        if current_name and not current_name == value:
            raise ValueError("Service name cannot be changed after it has been set")
        self._manifest.setdefault("metadata", {})
        self._manifest["metadata"]["name"] = value

    # ----------------- GPU Properties ----------------- #

    @property
    def tolerations(self):
        return self.pod_spec.get("tolerations", []) if self.pod_spec else []

    @property
    def replicas(self):
        return self.service_manager.get_replicas(self._manifest)

    @replicas.setter
    def replicas(self, value: int):
        # Set replicas in manifest and applies to relevant containers
        from kubetorch.serving.trainjob_service_manager import TrainJobServiceManager

        if isinstance(self.service_manager, TrainJobServiceManager):
            # For kubeflow training service managers, also update distributed config
            distributed_config = self.distributed_config

            if not distributed_config:
                distributed_config = {
                    "distribution_type": "spmd",
                    "quorum_timeout": self.launch_timeout,
                }
            distributed_config["quorum_workers"] = value
            self.service_manager.set_replicas(self._manifest, value, distributed_config=distributed_config)
        else:
            self.service_manager.set_replicas(self._manifest, value)

    @property
    def annotations(self):
        metadata = self._get_manifest_metadata()
        return metadata.get("annotations", {})

    @property
    def labels(self):
        metadata = self._get_manifest_metadata()
        return metadata.get("labels", {})

    @property
    def queue_name(self) -> Optional[str]:
        pod_template = self.pod_template
        if pod_template:
            queue = pod_template.get("metadata", {}).get("labels", {}).get(serving_constants.KUEUE_QUEUE_NAME_LABEL)
            if queue:
                return queue
        # Fall back to top-level manifest metadata (for from_manifest cases)
        return self._manifest.get("metadata", {}).get("labels", {}).get(serving_constants.KUEUE_QUEUE_NAME_LABEL)

    @queue_name.setter
    def queue_name(self, value: Optional[str]):
        """Set the Kueue queue name.

        Adds the kueue.x-k8s.io/queue-name label to both top-level and pod template metadata.
        For training jobs (PyTorchJob, TFJob, etc.), also sets spec.runPolicy.suspend = True
        so that Kueue can manage workload admission. When clearing queue_name, also clears
        the suspend flag.
        """
        queue_label = {serving_constants.KUEUE_QUEUE_NAME_LABEL: value} if value else {}

        # Add to top-level metadata
        if value:
            self.add_labels(queue_label)
        else:
            # Remove label if value is None
            metadata = self._get_manifest_metadata()
            metadata.get("labels", {}).pop(serving_constants.KUEUE_QUEUE_NAME_LABEL, None)

        # Add to pod template metadata
        self.add_pod_template_labels(
            queue_label if value else {}, remove_keys=[serving_constants.KUEUE_QUEUE_NAME_LABEL] if not value else []
        )

        # For training jobs, manage runPolicy.suspend for Kueue admission control
        kind = self._manifest.get("kind", "")
        if kind in ["PyTorchJob", "TFJob", "MXJob", "XGBoostJob"]:
            if value:
                # Set runPolicy.suspend = True for training jobs when using Kueue
                self._manifest.setdefault("spec", {}).setdefault("runPolicy", {})["suspend"] = True
            else:
                # Clear runPolicy.suspend = False for training jobs when using Kueue
                run_policy = self._manifest.get("spec", {}).get("runPolicy", {})
                if "suspend" in run_policy:
                    run_policy["suspend"] = False

    @property
    def pod_template(self):
        """Get the pod template from the manifest (includes metadata and spec)."""
        template_path = self.service_manager.get_pod_template_path()
        if template_path:
            current = self._manifest
            for key in template_path:
                current = current.get(key, {})
            return current
        return {}

    def add_labels(self, labels: Dict):
        """Add or update labels in the manifest metadata.

        Args:
            labels (Dict): Dictionary of labels to add or update.
        """
        if not labels:
            return

        metadata = self._get_manifest_metadata()
        metadata.setdefault("labels", {})
        metadata["labels"].update(labels)

    def add_pod_template_labels(self, labels: Dict, remove_keys: List[str] = None):
        """Add or update labels in the pod template metadata.

        This is useful for labels that need to be on the pod itself, such as
        Kueue queue labels (kueue.x-k8s.io/queue-name).

        Args:
            labels (Dict): Dictionary of labels to add or update.
            remove_keys (List[str], optional): List of label keys to remove.
        """
        pod_template = self.pod_template
        if not pod_template:
            return

        metadata = pod_template.setdefault("metadata", {})
        template_labels = metadata.setdefault("labels", {})

        if labels:
            template_labels.update(labels)

        if remove_keys:
            for key in remove_keys:
                template_labels.pop(key, None)

    def add_annotations(self, annotations: Dict):
        """Add or update annotations in the manifest metadata.

        Args:
            annotations (Dict): Dictionary of annotations to add or update.
        """
        if not annotations:
            return

        metadata = self._get_manifest_metadata()
        metadata.setdefault("annotations", {})
        metadata["annotations"].update(annotations)

    def add_tolerations(self, tolerations: List[Dict]):
        """Add or update tolerations in the pod spec.

        Args:
            tolerations (List[Dict]): List of toleration dictionaries to add or update. Each toleration should have keys like
                "key", "operator", "value", and "effect".
        """
        if not tolerations:
            return

        if not self.pod_spec:
            raise ValueError("pod_spec is not available. Cannot add tolerations.")

        existing_tolerations = self.pod_spec.get("tolerations", [])

        # Create a dictionary keyed by "key" for easy lookup and override
        toleration_dict = {}
        tolerations_without_key = []

        for tol in existing_tolerations:
            key = tol.get("key")
            if key:
                toleration_dict[key] = tol
            else:
                tolerations_without_key.append(tol)

        # Update or add new tolerations
        for tol in tolerations:
            key = tol.get("key")
            if key:
                toleration_dict[key] = tol
            else:
                # If no key, append it (though this is unusual)
                tolerations_without_key.append(tol)

        # Combine all tolerations: those with keys (merged) + those without keys (appended)
        self.pod_spec["tolerations"] = list(toleration_dict.values()) + tolerations_without_key

    def add_env_vars(self, env_vars: Dict):
        """Add or update environment variables in the container spec.

        Args:
            env_vars (Dict): Dictionary of environment variables to add or update. Existing env vars with the same key will be overridden.
        """
        if not env_vars:
            return

        container = self._container()
        self._set_env_vars_in_container(container, env_vars)

    # ----------------- Init Template Setup Helpers ----------------- #
    def _get_server_image(self, image):
        """Return base server image"""
        image = self.image.image_id if self.image and self.image.image_id else None

        if not image or image == serving_constants.KUBETORCH_IMAGE_TRAPDOOR:
            return serving_constants.SERVER_IMAGE_MINIMAL

        return image

    def _get_requested_resources(self, cpus, memory, disk_size, gpus):
        """Return requested resources."""
        requests = {}
        limits = {}

        # Add CPU if specified
        if cpus:
            requests["cpu"] = str(cpus)

        # Add Memory if specified
        if memory:
            requests["memory"] = memory

        # Add Storage if specified
        if disk_size:
            requests["ephemeral-storage"] = disk_size

        # Add GPU if specified
        if gpus:
            requests["nvidia.com/gpu"] = str(gpus)
            limits["nvidia.com/gpu"] = str(gpus)

        # Only include non-empty dicts
        resources = {}
        if requests:
            resources["requests"] = requests
        if limits:
            resources["limits"] = limits

        return resources

    def _get_launch_timeout(self, launch_timeout):
        if launch_timeout:
            return int(launch_timeout)
        default_launch_timeout = (
            self.default_config["launch_timeout"]
            if "launch_timeout" in self.default_config
            else serving_constants.KT_LAUNCH_TIMEOUT
        )
        return int(os.getenv("KT_LAUNCH_TIMEOUT", default_launch_timeout))

    def _get_config_env_vars(self, allowed_serialization):
        config_env_vars = globals.config._get_config_env_vars()
        if allowed_serialization:
            config_env_vars["KT_ALLOWED_SERIALIZATION"] = ",".join(allowed_serialization)
        return config_env_vars

    @property
    def image_install_cmd(self):
        return self.image.install_cmd if self.image and self.image.install_cmd else None

    def client_port(self) -> int:
        base_url = globals.service_url()
        _, port = extract_host_port(base_url)
        return port

    # ----------------- GPU Init Template Setup Helpers ----------------- #

    def _get_tolerations(self, gpus, tolerations):
        user_tolerations = tolerations if tolerations else []

        # add required GPU tolerations for GPU workloads
        if gpus:
            required_gpu_tolerations = [
                {
                    "key": "nvidia.com/gpu",
                    "operator": "Exists",
                    "effect": "NoSchedule",
                },
                {
                    "key": "dedicated",
                    "operator": "Equal",
                    "value": "gpu",
                    "effect": "NoSchedule",
                },
            ]

            all_tolerations = user_tolerations.copy()
            for req_tol in required_gpu_tolerations:
                if not any(
                    t["key"] == req_tol["key"]
                    and t.get("operator") == req_tol.get("operator")
                    and t.get("effect") == req_tol["effect"]
                    and (req_tol.get("value") is None or t.get("value") == req_tol.get("value"))
                    for t in all_tolerations
                ):
                    all_tolerations.append(req_tol)
            return all_tolerations

        return user_tolerations if user_tolerations else None

    # ----------------- Generic Helpers ----------------- #
    def _load_kubetorch_global_config(self):
        global_config = {}
        kubetorch_config = self.service_manager.fetch_kubetorch_config()
        if kubetorch_config:
            defaults_yaml = kubetorch_config.get("COMPUTE_DEFAULTS", "")
            if defaults_yaml:
                try:
                    validated_config = {}
                    config_dict = yaml.safe_load(defaults_yaml)
                    for key, value in config_dict.items():
                        # Check for values as dictionaries with keys 'key' and 'value'
                        if (
                            isinstance(value, list)
                            and len(value) > 0
                            and isinstance(value[0], dict)
                            and "key" in value[0]
                            and "value" in value[0]
                        ):
                            validated_config[key] = {item["key"]: item["value"] for item in value}
                        elif value is not None:
                            validated_config[key] = value
                    global_config = validated_config
                except yaml.YAMLError as e:
                    logger.error(f"Failed to load kubetorch global config: {str(e)}")

        if global_config:
            for key in ["inactivity_ttl"]:
                # Set values from global config where the value is not already set
                if key in global_config and self.__getattribute__(key) is None:
                    self.__setattr__(key, global_config[key])
            for key in ["labels", "annotations", "env_vars"]:
                # Merge global config with existing config for dictionary values
                if key in global_config and isinstance(global_config[key], dict):
                    self.__setattr__(key, {**global_config[key], **self.__getattribute__(key)})
            if "image_id" in global_config:
                if self.image is None:
                    self.image = Image(image_id=global_config["image_id"])
                elif self.image.image_id is None:
                    self.image.image_id = global_config["image_id"]

        return global_config

    # ----------------- Launching a new service (Knative or StatefulSet) ----------------- #
    def _launch(
        self,
        service_name: str,
        install_url: str,
        pointer_env_vars: Dict,
        metadata_env_vars: Dict,
        startup_rsync_command: str = None,
        launch_id: str = None,
        deployment_timestamp: str = None,
        dryrun: bool = False,
    ):
        """Creates a new service on the compute for the provided service. If the service already exists,
        it will update the service with the latest copy of the code."""
        # Finalize pod spec with launch time env vars
        self._update_launch_env_vars(service_name, pointer_env_vars, metadata_env_vars, launch_id)
        self._upload_secrets_list()

        setup_script = self._get_setup_script(install_url, startup_rsync_command)

        container = self._container()
        if "args" in container and len(container["args"]) > 0:
            container["args"][0] = setup_script
        else:
            container["args"] = [setup_script]

        # Create service using the appropriate service manager
        (created_service, updated_manifest,) = self.service_manager.create_or_update_service(
            service_name=service_name,
            module_name=pointer_env_vars["KT_MODULE_NAME"],
            manifest=self._manifest,
            deployment_timestamp=deployment_timestamp,
            dryrun=dryrun,
        )
        self._manifest = updated_manifest

        service_info = self.service_manager.normalize_created_service(created_service)
        service_name = service_info["name"]
        service_template = {
            "metadata": {
                "name": service_info["name"],
                "namespace": service_info["namespace"],
            },
            "spec": {"template": service_info["template"]},
        }

        logger.debug(f"Successfully deployed {self.deployment_mode} service {service_name}")

        return service_template

    async def _launch_async(
        self,
        service_name: str,
        install_url: str,
        pointer_env_vars: Dict,
        metadata_env_vars: Dict,
        startup_rsync_command: str = None,
        launch_id: str = None,
        deployment_timestamp: str = None,
        dryrun: bool = False,
    ):
        """Async version of _launch. Creates a new service on the compute for the provided service.
        If the service already exists, it will update the service with the latest copy of the code."""

        import asyncio

        loop = asyncio.get_event_loop()

        service_template = await loop.run_in_executor(
            None,
            self._launch,
            service_name,
            install_url,
            pointer_env_vars,
            metadata_env_vars,
            startup_rsync_command,
            launch_id,
            deployment_timestamp,
            dryrun,
        )

        return service_template

    def _update_launch_env_vars(self, service_name, pointer_env_vars, metadata_env_vars, launch_id):
        kt_env_vars = {
            **pointer_env_vars,
            **metadata_env_vars,
            "KT_LAUNCH_ID": launch_id,
            "KT_SERVICE_NAME": service_name,
            "KT_SERVICE_DNS": (
                f"{service_name}-headless.{self.namespace}.svc.cluster.local"
                if self.distributed_config
                else f"{service_name}.{self.namespace}.svc.cluster.local"
            ),
            "KT_DEPLOYMENT_MODE": self.deployment_mode,
        }
        if "OTEL_SERVICE_NAME" not in self.config_env_vars.keys():
            kt_env_vars["OTEL_SERVICE_NAME"] = service_name

        # Ensure cluster config env vars are set
        if globals.config.cluster_config:
            if globals.config.cluster_config.get("log_streaming_enabled", True):
                kt_env_vars["KT_LOG_STREAMING_ENABLED"] = True
            if globals.config.cluster_config.get("metrics_enabled", True):
                kt_env_vars["KT_METRICS_ENABLED"] = True

        # Ensure all environment variable values are strings for Kubernetes compatibility
        kt_env_vars = self._serialize_env_vars(kt_env_vars)

        updated_env_vars = set()
        for env_var in self._container_env():
            if env_var["name"] in kt_env_vars:
                env_var["value"] = kt_env_vars[env_var["name"]]
                updated_env_vars.add(env_var["name"])
        for key, val in kt_env_vars.items():
            if key not in updated_env_vars:
                self._container_env().append({"name": key, "value": val})

    def _serialize_env_vars(self, env_vars: Dict) -> Dict:
        import json

        serialized_vars = {}
        for key, value in env_vars.items():
            if value is None:
                serialized_vars[key] = "null"
            elif isinstance(value, (dict, list)):
                try:
                    serialized_vars[key] = json.dumps(value)
                except (TypeError, ValueError):
                    serialized_vars[key] = str(value)
            elif isinstance(value, (bool, int, float)):
                serialized_vars[key] = str(value)
            else:
                serialized_vars[key] = value
        return serialized_vars

    def _extract_secrets(self, secrets, namespace):
        if is_running_in_kubernetes():
            return [], []

        secret_env_vars = []
        secret_volumes = []
        if secrets:
            secrets_client = KubernetesSecretsClient(namespace=namespace, kubeconfig_path=self.kubeconfig_path)
            secret_objects = secrets_client.convert_to_secret_objects(secrets=secrets)
            (
                secret_env_vars,
                secret_volumes,
            ) = secrets_client.extract_envs_and_volumes_from_secrets(secret_objects)

        return secret_env_vars, secret_volumes

    def _get_setup_script(self, install_url, startup_rsync_command):
        # Load the setup script template
        from kubetorch.servers.http.utils import _get_rendered_template

        setup_script = _get_rendered_template(
            serving_constants.KT_SETUP_TEMPLATE_FILE,
            template_dir=os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "serving",
                "templates",
            ),
            python_path=self.python_path,
            freeze=self.freeze,
            install_url=install_url or globals.config.install_url,
            install_cmd=self.image_install_cmd,
            server_image=self.server_image,
            rsync_kt_local_cmd=startup_rsync_command,
            server_port=self.server_port,
        )
        return setup_script

    def _upload_secrets_list(self):
        """Upload secrets to Kubernetes. Called during launch time, not during init."""
        if is_running_in_kubernetes():
            return

        if self.secrets:
            logger.debug("Uploading secrets to Kubernetes")
            self.secrets_client.upload_secrets_list(secrets=self.secrets)

    def _get_node_selector(self, node_selector, gpu_type):
        if gpu_type:
            if ":" in gpu_type:
                # Parse "key: value" format
                key, value = gpu_type.split(":", 1)
                node_selector[key.strip()] = value.strip()
            else:
                # Default to nvidia.com/gpu.product
                node_selector["nvidia.com/gpu.product"] = gpu_type
        return node_selector

    def pod_names(self):
        """Returns a list of pod names."""
        pods = self.pods()
        return [
            pod.get("metadata", {}).get("name")
            for pod in pods
            if pod_is_running(pod) and pod.get("metadata", {}).get("name")
        ]

    def pods(self):
        return self.service_manager.get_pods_for_service(self.service_name)

    # ------------------------------- Volumes ------------------------------ #
    def _process_volumes(self, volumes) -> Optional[List[Volume]]:
        """Process volumes input into standardized format"""
        if volumes is None:
            volumes = globals.config.volumes

        if volumes is None:
            return None

        if isinstance(volumes, list):
            processed_volumes = []
            for vol in volumes:
                if isinstance(vol, str):
                    # list of volume names (assume they exist)
                    volume = Volume.from_name(vol)
                    processed_volumes.append(volume)

                elif isinstance(vol, Volume):
                    # list of Volume objects (create them if they don't already exist)
                    # Default the volume namespace to compute namespace if not provided
                    if vol.namespace is None:
                        vol.namespace = self.namespace
                    vol.create()
                    processed_volumes.append(vol)

                else:
                    raise ValueError(f"Volume list items must be strings or Volume objects, got {type(vol)}")

            return processed_volumes

        else:
            raise ValueError(f"Volumes must be a list, got {type(volumes)}")

    def _volumes_for_pod_template(self, volumes):
        """Convert processed volumes to template format"""
        volume_mounts = []
        volume_specs = []

        if volumes:
            for volume in volumes:
                # Add volume mount
                volume_mounts.append({"name": volume.name, "mountPath": volume.mount_path})

                # Add volume spec
                volume_specs.append(volume.pod_template_spec())

        return volume_mounts, volume_specs

    # ----------------- Functions using K8s implementation ----------------- #
    def _wait_for_endpoint(self):
        retries = 20
        for i in range(retries):
            endpoint = self.endpoint
            if endpoint:
                return endpoint
            else:
                logger.info(f"Endpoint not available (attempt {i + 1}/{retries})")
            time.sleep(2)

        logger.error(f"Endpoint not available for {self.service_name}")
        return None

    def _status_condition_ready(self, status) -> bool:
        """
        Checks if the Knative Service status conditions include a 'Ready' condition with status 'True'.
        This indicates that the service is ready to receive traffic.

        Notes:
            - This does not check pod status or readiness, only the Knative Service's own readiness condition.
            - A service can be 'Ready' even if no pods are currently running (e.g., after scaling to zero).
        """
        for condition in status.get("conditions", []):
            if condition.get("type") == "Ready" and condition.get("status") == "True":
                logger.debug(f"Service {self.service_name} is ready")
                return True
        return False

    def _check_service_ready(self):
        """Checks if the service is ready to start serving requests.

        Delegates to the appropriate service manager's check_service_ready method.
        """
        return self.service_manager.check_service_ready(
            service_name=self.service_name, launch_timeout=self.launch_timeout
        )

    async def _check_service_ready_async(self):
        """Async version of _check_service_ready. Checks if the service is ready to start serving requests.

        Delegates to the appropriate service manager's check_service_ready method.
        """
        import asyncio

        loop = asyncio.get_event_loop()

        return await loop.run_in_executor(
            None,
            self._check_service_ready,
        )

    def is_up(self):
        """Whether the pods are running."""
        try:
            pods = self.pods()
            if not pods:
                return False
            for pod in pods:
                phase = pod.get("status", {}).get("phase")
                pod_name = pod.get("metadata", {}).get("name", "unknown")
                if phase != "Running":
                    logger.info(f"Pod {pod_name} is not running. Status: {phase}")
                    return False
        except Exception:
            return False
        return True

    def _rsync_svc_url(self):
        """Get the rsync pod URL for startup commands."""
        client = data_store.RsyncClient(self.namespace, self.service_name)
        return client.get_rsync_pod_url()

    def ssh(self, pod_name: str = None):
        if pod_name is None:
            pods = self.pods()
            running_pods = [pod for pod in pods if pod_is_running(pod)]

            if not running_pods:
                raise RuntimeError(f"No running pods found for service {self.service_name}")

            pod_name = load_head_node_pod(running_pods, deployment_mode=self.deployment_mode)

        ssh_cmd = f"kubectl exec -it {pod_name} -n {self.namespace} -- /bin/bash"
        subprocess.run(shlex.split(ssh_cmd), check=True)

    def get_env_vars(self, keys: Union[List[str], str] = None):
        keys = [keys] if isinstance(keys, str) else keys
        env_vars = {}
        for env_var in self._container_env():
            if not keys or (env_var["name"] in keys and "value" in env_var):
                env_vars[env_var["name"]] = env_var["value"]
        return env_vars

    # ----------------- Image Related Functionality ----------------- #

    def pip_install(
        self,
        reqs: Union[List[str], str],
        node: Optional[str] = None,
        override_remote_version: bool = False,
    ):
        """Pip install reqs onto compute pod(s)."""
        reqs = [reqs] if isinstance(reqs, str) else reqs
        python_path = self.image.python_path if self.image else "python3"
        pip_install_cmd = f"{python_path} -m pip install"
        try:
            result = self.run_bash("cat .kt/kt_pip_install_cmd 2>/dev/null || echo ''", node=node)
            if result and result[0][0] == 0 and result[0][1].strip():
                pip_install_cmd = result[0][1].strip()
        except Exception:
            pass

        for req in reqs:
            base = self.working_dir or "."
            remote_editable = self.run_bash(f"[ -d {base}/{req} ]", node=node)[0][0] == 0
            if remote_editable:
                req = f"{base}/{req}"
            else:
                local_version = find_locally_installed_version(req)
                if local_version is not None:
                    if not override_remote_version:
                        installed_remotely = (
                            self.run_bash(
                                f"{python_path} -c \"import importlib.util; exit(0) if importlib.util.find_spec('{req}') else exit(1)\"",
                                node=node,
                            )[0][0]
                            == 0
                        )
                        if installed_remotely:
                            logger.info(f"{req} already installed. Skipping.")
                            continue
                    else:
                        req = f"{req}=={local_version}"

            logger.info(f"Pip installing {req} with: {pip_install_cmd} {req}")
            self.run_bash(f"{pip_install_cmd} {req}", node=node)

    def sync_package(
        self,
        package: str,
        node: Optional[str] = None,
    ):
        """Sync package (locally installed, or path to package) to compute pod(s)."""
        full_path, dest_dir = _get_sync_package_paths(package)
        logger.info(f"Syncing over package at {full_path} to {dest_dir}")
        # Use RsyncClient directly - files go to rsync pod, then sync to service pods at startup
        client = data_store.RsyncClient(self.namespace, self.service_name)
        client.upload(source=full_path, dest=dest_dir)

    def run_bash(
        self,
        commands,
        node: Union[str, List[str]] = None,
        container: Optional[str] = None,
    ):
        """Run bash commands on the pod(s)."""
        pod_names = self.pod_names() if node in ["all", None] else [node] if isinstance(node, str) else node

        return _run_bash(
            commands=commands,
            pod_names=pod_names,
            namespace=self.namespace,
            container=container,
        )

    def _image_setup_and_instructions(self, rsync: bool = True):
        """
        Return image instructions in Dockerfile format, and optionally rsync over content to the rsync pod.
        """
        instructions = ""

        if not self.image:
            return instructions

        logger.debug("Writing out image instructions.")

        if self.image.image_id:
            instructions += f"FROM {self.server_image}\n"
        if self.image.python_path:
            instructions += f"ENV KT_PYTHON_PATH {self.image.python_path}\n"

        # image_id is ignored, used directly in server_image
        for step in self.image.setup_steps:
            if step.step_type == ImageSetupStepType.CMD_RUN:
                commands = step.kwargs.get("command")
                commands = [commands] if isinstance(commands, str) else commands
                for i in range(len(commands)):
                    if i != 0:
                        instructions += "\n"
                    instructions += f"RUN {commands[i]}"
            elif step.step_type == ImageSetupStepType.PIP_INSTALL:
                reqs = step.kwargs.get("reqs")
                reqs = [reqs] if isinstance(reqs, str) else reqs
                for i in range(len(reqs)):
                    if i != 0:
                        instructions += "\n"
                    if self.image_install_cmd:
                        install_cmd = self.image_install_cmd
                    else:
                        install_cmd = "$KT_PIP_INSTALL_CMD"

                    # Pass through the requirement string directly without quoting
                    # This allows users to pass any pip arguments they want
                    # e.g., "--pre torchmonarch==0.1.0rc7" or "numpy>=1.20"
                    instructions += f"RUN {install_cmd} {reqs[i]}"

                    if step.kwargs.get("force"):
                        instructions += " # force"
            elif step.step_type == ImageSetupStepType.SYNC_PACKAGE:
                # using package name instead of paths, since the folder path in the rsync pod will just be the package name
                full_path, dest_dir = _get_sync_package_paths(step.kwargs.get("package"))
                if rsync:
                    # Use RsyncClient directly - files go to rsync pod, then sync to service pods at startup
                    client = data_store.RsyncClient(self.namespace, self.service_name)
                    client.upload(source=full_path, dest=dest_dir)
                instructions += f"COPY {full_path} {dest_dir}"
            elif step.step_type == ImageSetupStepType.RSYNC:
                source_path = step.kwargs.get("source")
                dest_dir = step.kwargs.get("dest")
                contents = step.kwargs.get("contents")
                filter_options = step.kwargs.get("filter_options")
                force = step.kwargs.get("force")

                if rsync:
                    # Use RsyncClient directly - files go to rsync pod, then sync to service pods at startup
                    client = data_store.RsyncClient(self.namespace, self.service_name)
                    client.upload(
                        source=source_path,
                        dest=dest_dir,
                        contents=contents,
                        filter_options=filter_options,
                        force=force,
                    )
                # Generate COPY instruction with explicit destination
                if dest_dir:
                    instructions += f"COPY {source_path} {dest_dir}"
                else:
                    # No dest specified - use basename of source as destination
                    dest_name = Path(source_path).name
                    instructions += f"COPY {source_path} {dest_name}"
            elif step.step_type == ImageSetupStepType.SET_ENV_VARS:
                for key, val in step.kwargs.get("env_vars").items():
                    # single env var per line in the dockerfile
                    instructions += f"ENV {key} {val}\n"
            if step.kwargs.get("force") and step.step_type != ImageSetupStepType.PIP_INSTALL:
                instructions += " # force"
            instructions += "\n"

        return instructions

    # ----------------- Copying over for now... TBD ----------------- #
    def __getstate__(self):
        """Remove local stateful values before pickle serialization."""
        state = self.__dict__.copy()
        # Remove local stateful values that shouldn't be serialized
        state["_endpoint"] = None
        state["_service_manager"] = None
        state["_secrets_client"] = None
        return state

    def __setstate__(self, state):
        """Restore state after pickle deserialization."""
        self.__dict__.update(state)
        # Reset local stateful values to None to ensure clean initialization
        self._endpoint = None
        self._service_manager = None
        self._secrets_client = None

    # ------------ Distributed / Autoscaling Helpers -------- #
    def distribute(
        self,
        distribution_type: str = None,
        workers: int = None,
        quorum_timeout: int = None,
        quorum_workers: int = None,
        monitor_members: bool = None,
        **kwargs,
    ):
        """Configure the distributed worker compute needed by each service replica.

        Args:
            distribution_type (str): The type of distributed supervisor to create.
                Options: ``spmd`` (default, if empty), ``"pytorch"``, ``"ray"``, ``"monarch"``, ``"jax"``, or ``"tensorflow"``.
            workers (int): Int representing the number of workers to create, with identical compute resources to
                the service compute. Or List of ``<int, Compute>`` pairs specifying the number of workers and the compute
                resources for each worker StatefulSet.
            quorum_timeout (int, optional): Timeout in seconds for workers to become ready and join the cluster.
                Defaults to `launch_timeout` if not provided, for both SPMD frameworks and for Ray.
                Increase this if workers need more time to start (e.g., during node autoscaling or loading down data
                during initialization).
            **kwargs: Additional framework-specific parameters (e.g., num_proc, port).

        Note:
            List of ``<int, Compute>`` pairs is not yet supported for workers.

        Examples:

        .. code-block:: python

            import kubetorch as kt

            remote_fn = kt.fn(simple_summer, service_name).to(
                kt.Compute(
                    cpus="2",
                    memory="4Gi",
                    image=kt.Image(image_id="rayproject/ray"),
                    launch_timeout=300,
                ).distribute("ray", workers=2)
            )

            gpus = kt.Compute(
                gpus=1,
                image=kt.Image(image_id="nvcr.io/nvidia/pytorch:23.10-py3"),
                launch_timeout=600,
                inactivity_ttl="4h",
            ).distribute("pytorch", workers=4)
        """
        # Check for conflicting configuration
        if self.autoscaling_config:
            raise ValueError(
                "Cannot use both .distribute() and .autoscale() on the same compute instance. "
                "Use .distribute() for fixed replicas with distributed training, or .autoscale() for auto-scaling services."
            )

        # Configure distributed settings
        # Note: We default to simple SPMD distribution ("spmd") if nothing specified and compute.workers > 1

        # User can override quorum if they want to set a lower threshold
        # For non-Ray cases, default to workers if not provided
        quorum_workers = quorum_workers or workers
        distributed_config = {
            "distribution_type": distribution_type or "spmd",
            "quorum_timeout": quorum_timeout or self.launch_timeout,
            "quorum_workers": quorum_workers,
        }
        if monitor_members is not None:
            # Note: Ray manages its own membership, so it's disabled by default in the supervisor
            # It's enabled by default for SPMD.
            distributed_config["monitor_members"] = monitor_members
        distributed_config.update(kwargs)

        if workers:
            if not isinstance(workers, int):
                raise ValueError("Workers must be an integer. List of <integer, Compute> pairs is not yet supported")
            self.replicas = workers

        if distributed_config:
            self.distributed_config = distributed_config

            distribution_type = distributed_config.get("distribution_type")
            if distribution_type == "ray":
                # Convert to RayCluster manifest
                from kubetorch.serving.service_manager import RayClusterServiceManager

                self._manifest = RayClusterServiceManager._convert_manifest(
                    deployment_manifest=self._manifest,
                    namespace=self.namespace,
                    replicas=self.replicas,
                )

            # Invalidate cached service manager so it gets recreated with the right type
            self._service_manager = None

        return self

    def autoscale(self, **kwargs):
        """Configure the service with the provided autoscaling parameters.

        You can pass any of the following keyword arguments:

        Args:
            target (int): The concurrency/RPS/CPU/memory target per pod.
            window (str): Time window for scaling decisions, e.g. "60s".
            metric (str): Metric to scale on: "concurrency", "rps", "cpu", "memory" or custom.
                Note: "cpu" and "memory" require autoscaler_class="hpa.autoscaling.knative.dev".
            target_utilization (int): Utilization % to trigger scaling (1-100).
            min_scale (int): Minimum number of replicas. 0 allows scale to zero.
            max_scale (int): Maximum number of replicas.
            initial_scale (int): Initial number of pods.
            concurrency (int): Maximum concurrent requests per pod (containerConcurrency).
                If not set, pods accept unlimited concurrent requests.
            scale_to_zero_pod_retention_period (str): Time to keep last pod before scaling
                to zero, e.g. "30s", "1m5s".
            scale_down_delay (str): Delay before scaling down, e.g. "15m". Only for KPA.
            autoscaler_class (str): Autoscaler implementation:
                - "kpa.autoscaling.knative.dev" (default, supports concurrency/rps)
                - "hpa.autoscaling.knative.dev" (supports cpu/memory/custom metrics)
            progress_deadline (str): Time to wait for deployment to be ready, e.g. "10m".
                Must be longer than startup probe timeout.
            **extra_annotations: Additional Knative autoscaling annotations.

        Note:
            The service will be deployed as a Knative service.

            Timing-related defaults are applied if not explicitly set (for ML workloads):
            - scale_down_delay="1m" (avoid rapid scaling cycles)
            - scale_to_zero_pod_retention_period="10m" (keep last pod longer before scale to zero)
            - progress_deadline="10m" or greater (ensures enough time for initialization, automatically adjusted based on launch_timeout)

        Examples:

        .. code-block:: python

            import kubetorch as kt

            remote_fn = kt.fn(my_fn_obj).to(
                kt.Compute(
                    cpus=".1",
                ).autoscale(min_replicas=1)
            )

            remote_fn = kt.fn(summer).to(
                compute=kt.Compute(
                    cpus=".01",
                ).autoscale(min_scale=3, scale_to_zero_grace_period=50),
            )
        """
        # Check for conflicting configuration
        if self.distributed_config:
            raise ValueError(
                "Cannot use both .distribute() and .autoscale() on the same compute instance. "
                "Use .distribute() for fixed replicas with distributed training, or .autoscale() for auto-scaling services."
            )

        # Apply timing-related defaults for ML workloads to account for initialization overhead
        # (heavy dependencies, model loading, etc. affect both CPU and GPU workloads)
        if "scale_down_delay" not in kwargs:
            kwargs["scale_down_delay"] = "1m"
            logger.debug("Setting scale_down_delay=1m to avoid thrashing")

        if "scale_to_zero_pod_retention_period" not in kwargs:
            kwargs["scale_to_zero_pod_retention_period"] = "10m"
            logger.debug("Setting scale_to_zero_pod_retention_period=10m to avoid thrashing")

        if "progress_deadline" not in kwargs:
            # Ensure progress_deadline is at least as long as launch_timeout
            default_deadline = "10m"  # 600 seconds
            if self.launch_timeout:
                # Convert launch_timeout (seconds) to a duration string
                # Add some buffer (20% or at least 60 seconds)
                timeout_with_buffer = max(self.launch_timeout + 60, int(self.launch_timeout * 1.2))
                if timeout_with_buffer > 600:  # If larger than default
                    default_deadline = f"{timeout_with_buffer}s"
            kwargs["progress_deadline"] = default_deadline
            logger.debug(f"Setting progress_deadline={default_deadline} to allow time for initialization")

        autoscaling_config = AutoscalingConfig(**kwargs)
        if autoscaling_config:
            self._autoscaling_config = autoscaling_config

            # Convert manifest to Knative service manifest
            from kubetorch.serving.service_manager import KnativeServiceManager

            self._manifest = KnativeServiceManager._convert_manifest(
                deployment_manifest=self._manifest,
                namespace=self.namespace,
                autoscaling_config=autoscaling_config,
                gpu_annotations=self.gpu_annotations,
            )

            # Invalidate cached service manager so it gets recreated with KnativeServiceManager
            self._service_manager = None

        return self
