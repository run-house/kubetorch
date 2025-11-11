import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

import yaml

from kubernetes import client, config
from kubernetes.client import V1ResourceRequirements

import kubetorch.constants as constants
import kubetorch.serving.constants as serving_constants

from kubetorch import globals

from kubetorch.logger import get_logger
from kubetorch.resources.callables.utils import find_locally_installed_version
from kubetorch.resources.compute.utils import (
    _get_rsync_exclude_options,
    _get_sync_package_paths,
    _run_bash,
    find_available_port,
    RsyncError,
)
from kubetorch.resources.compute.websocket import WebSocketRsyncTunnel
from kubetorch.resources.images.image import Image, ImageSetupStepType
from kubetorch.resources.secrets.kubernetes_secrets_client import KubernetesSecretsClient
from kubetorch.resources.volumes.volume import Volume
from kubetorch.servers.http.utils import is_running_in_kubernetes, load_template
from kubetorch.serving.autoscaling import AutoscalingConfig
from kubetorch.serving.service_manager import DeploymentServiceManager, KnativeServiceManager, RayClusterServiceManager
from kubetorch.serving.utils import GPUConfig, pod_is_running, RequestedPodResources

from kubetorch.utils import extract_host_port, http_to_ws, load_kubeconfig

logger = get_logger(__name__)


class Compute:
    def __init__(
        self,
        cpus: Union[str, int] = None,
        memory: str = None,
        disk_size: str = None,
        gpus: Union[str, int] = None,
        queue: str = None,
        priority_class_name: str = None,
        gpu_type: str = None,
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
        replicas: int = 1,
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
            queue (str, optional): Name of the Kubernetes queue that will be responsible for placing the service's
                pods onto nodes. Controls how cluster resources are allocated and prioritized for this service.
                Pods will be scheduled according to the quota, priority, and limits configured for the queue.
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
                the `.distribute(workers=N)` method for distributed training. (Default: 1)
            working_dir (str, optional): Working directory to use inside the remote images. Must be an absolute path (e.g. `/kt`)
            shared_memory_limit (str, optional):  Maximum size of the shared memory filesystem (/dev/shm) available to
                each pod created by the service. Value should be a Kubernetes quantity string, for example: "512Mi",
                "2Gi", "1G", "1024Mi", "100M". If not provided, /dev/shm will default to the pod's memory limit (if set)
                or up to half the node's RAM.

        Note:
            **Resource Specification Formats:**

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
        self._namespace = namespace or globals.config.namespace

        self._objects_api = None
        self._core_api = None
        self._apps_v1_api = None
        self._node_v1_api = None

        self._image = image
        self._service_name = None
        self._secrets = secrets
        self._secrets_client = None
        self._volumes = volumes
        self._queue = queue

        # service template args to store
        self.replicas = replicas
        self.labels = labels or {}
        self.annotations = annotations or {}
        self.service_template = service_template or {}
        self._gpu_annotations = {}  # Will be populated during init or from_template

        # Skip template initialization if loading from existing service
        if _skip_template_init:
            return

        # determine pod template vars
        server_port = serving_constants.DEFAULT_KT_SERVER_PORT
        service_account_name = service_account_name or serving_constants.DEFAULT_SERVICE_ACCOUNT_NAME
        otel_enabled = (
            globals.config.cluster_config.get("otel_enabled", False) if globals.config.cluster_config else False
        )
        server_image = self._get_server_image(image, otel_enabled, inactivity_ttl)
        gpus = None if gpus in (0, None) else gpus
        gpu_config = self._load_gpu_config(gpus, gpu_memory, gpu_type)
        self._gpu_annotations = self._get_gpu_annotations(gpu_config)
        requested_resources = self._get_requested_resources(cpus, memory, disk_size, gpu_config)
        secret_env_vars, secret_volumes = self._extract_secrets(secrets)
        volume_mounts, volume_specs = self._volumes_for_pod_template(volumes)
        scheduler_name = self._get_scheduler_name(queue)
        node_selector = self._get_node_selector(node_selector.copy() if node_selector else {}, gpu_type)
        all_tolerations = self._get_tolerations(gpus, tolerations)

        env_vars = env_vars or {}
        if os.getenv("KT_LOG_LEVEL") and not env_vars.get("KT_LOG_LEVEL"):
            # If KT_LOG_LEVEL is set, add it to env vars so the log level is set on the server
            env_vars["KT_LOG_LEVEL"] = os.getenv("KT_LOG_LEVEL")

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
            "config_env_vars": self._get_config_env_vars(allowed_serialization or ["json"]),
            "image_pull_policy": image_pull_policy,
            "namespace": self._namespace,
            "freeze": freeze,
            "gpu_anti_affinity": gpu_anti_affinity,
            "working_dir": working_dir,
            "tolerations": all_tolerations,
            "shm_size_limit": shared_memory_limit,
            "priority_class_name": priority_class_name,
            "launch_timeout": self._get_launch_timeout(launch_timeout),
            "queue_name": self.queue_name(),
            "scheduler_name": scheduler_name,
            "inactivity_ttl": inactivity_ttl,
            "otel_enabled": otel_enabled,
            # launch time arguments
            "raycluster": False,
            "setup_script": "",
        }

        self.pod_template = load_template(
            template_file=serving_constants.POD_TEMPLATE_FILE,
            template_dir=os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "serving",
                "templates",
            ),
            **template_vars,
        )

    @classmethod
    def from_template(cls, service_info: dict):
        """Create a Compute object from a deployed Kubernetes resource."""
        if "resource" not in service_info:
            raise ValueError("service_info missing required key: resource")

        resource = service_info["resource"]
        kind = resource.get("kind", "Unknown")

        if kind == "RayCluster":
            template_path = resource["spec"]["headGroupSpec"]["template"]
        elif kind in ["Deployment", "Service"]:  # Deployment or Knative Service
            template_path = resource["spec"]["template"]
        else:
            raise ValueError(
                f"Unsupported resource kind: '{kind}'. "
                f"Supported kinds are: Deployment, Service (Knative), RayCluster"
            )

        template_metadata = template_path["metadata"]
        pod_spec = template_path["spec"]

        annotations = template_metadata.get("annotations", {})

        compute = cls(_skip_template_init=True)
        compute.pod_template = pod_spec

        # Set properties from manifest
        compute._namespace = service_info["namespace"]
        compute.replicas = resource["spec"].get("replicas")
        compute.labels = template_metadata.get("labels", {})
        compute.annotations = annotations
        compute._autoscaling_config = annotations.get("autoscaling.knative.dev/config", {})
        compute._queue = template_metadata.get("labels", {}).get("kai.scheduler/queue")
        compute._kubeconfig_path = annotations.get(serving_constants.KUBECONFIG_PATH_ANNOTATION)

        # Extract GPU annotations directly from template annotations
        gpu_annotation_keys = ["gpu-memory", "gpu-fraction"]
        compute._gpu_annotations = {k: v for k, v in annotations.items() if k in gpu_annotation_keys}

        return compute

    # ----------------- Properties ----------------- #
    @property
    def objects_api(self):
        if self._objects_api is None:
            self._objects_api = client.CustomObjectsApi()
        return self._objects_api

    @property
    def core_api(self):
        if self._core_api is None:
            load_kubeconfig()
            self._core_api = client.CoreV1Api()
        return self._core_api

    @property
    def apps_v1_api(self):
        if self._apps_v1_api is None:
            self._apps_v1_api = client.AppsV1Api()
        return self._apps_v1_api

    @property
    def node_v1_api(self):
        if self._node_v1_api is None:
            self._node_v1_api = client.NodeV1Api()
        return self._node_v1_api

    @property
    def kubeconfig_path(self):
        if self._kubeconfig_path is None:
            self._kubeconfig_path = os.getenv("KUBECONFIG") or constants.DEFAULT_KUBECONFIG_PATH
        return str(Path(self._kubeconfig_path).expanduser())

    @property
    def service_manager(self):
        if self._service_manager is None:
            self._load_kube_config()
            # Select appropriate service manager based on configuration
            if self.deployment_mode == "knative":
                # Use KnativeServiceManager for autoscaling services
                self._service_manager = KnativeServiceManager(
                    objects_api=self.objects_api,
                    core_api=self.core_api,
                    apps_v1_api=self.apps_v1_api,
                    namespace=self.namespace,
                )
            elif self.deployment_mode == "raycluster":
                # Use RayClusterServiceManager for Ray distributed workloads
                self._service_manager = RayClusterServiceManager(
                    objects_api=self.objects_api,
                    core_api=self.core_api,
                    apps_v1_api=self.apps_v1_api,
                    namespace=self.namespace,
                )
            else:
                # Use DeploymentServiceManager for regular deployments
                self._service_manager = DeploymentServiceManager(
                    objects_api=self.objects_api,
                    core_api=self.core_api,
                    apps_v1_api=self.apps_v1_api,
                    namespace=self.namespace,
                )
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

    @property
    def endpoint(self):
        if self._endpoint is None and self.service_name:
            self._endpoint = self.service_manager.get_endpoint(self.service_name)
        return self._endpoint

    @endpoint.setter
    def endpoint(self, endpoint: str):
        self._endpoint = endpoint

    def _container(self):
        """Get the container from the pod template."""
        if "containers" not in self.pod_template:
            raise ValueError("pod_template missing 'containers' field.")
        return self.pod_template["containers"][0]

    def _container_env(self):
        container = self._container()
        if "env" not in container:
            return []
        return container["env"]

    def _set_container_resource(self, resource_name: str, value: str):
        container = self._container()

        # Ensure resources dict exists
        if "resources" not in container:
            container["resources"] = {}

        # Ensure requests dict exists
        if "requests" not in container["resources"]:
            container["resources"]["requests"] = {}

        # Ensure limits dict exists
        if "limits" not in container["resources"]:
            container["resources"]["limits"] = {}

        # Set both requests and limits to the same value
        container["resources"]["requests"][resource_name] = value
        container["resources"]["limits"][resource_name] = value

    def _get_container_resource(self, resource_name: str) -> Optional[str]:
        resources = self._container().get("resources", {})
        requests = resources.get("requests", {})
        return requests.get(resource_name)

    # -------------- Properties From Template -------------- #
    @property
    def server_image(self):
        return self._container().get("image")

    @server_image.setter
    def server_image(self, value: str):
        """Set the server image in the pod template."""
        self._container()["image"] = value

    @property
    def server_port(self):
        return self._container()["ports"][0].get("containerPort")

    @server_port.setter
    def server_port(self, value: int):
        """Set the server port in the pod template."""
        self._container()["ports"][0]["containerPort"] = value

    @property
    def env_vars(self):
        # extract user-defined environment variables from rendered pod template
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
        self._set_container_resource("cpu", value)

    @property
    def memory(self):
        return self._get_container_resource("memory")

    @memory.setter
    def memory(self, value: str):
        """
        Args:
            value: Memory value (e.g., "4Gi", "2048Mi")
        """
        self._set_container_resource("memory", value)

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
        self._set_container_resource("nvidia.com/gpu", str(value))

    @property
    def gpu_type(self):
        node_selector = self.pod_template.get("nodeSelector")
        if node_selector and "nvidia.com/gpu.product" in node_selector:
            return node_selector["nvidia.com/gpu.product"]
        return None

    @gpu_type.setter
    def gpu_type(self, value: str):
        """
        Args:
            value: GPU product name (e.g., "L4", "V100", "A100", "T4")
        """
        if "nodeSelector" not in self.pod_template:
            self.pod_template["nodeSelector"] = {}
        self.pod_template["nodeSelector"]["nvidia.com/gpu.product"] = value

    @property
    def gpu_memory(self):
        annotations = self.pod_template.get("annotations", {})
        if "gpu-memory" in annotations:
            return annotations["gpu-memory"]
        return None

    @gpu_memory.setter
    def gpu_memory(self, value: str):
        """
        Args:
            value: GPU memory in MiB (e.g., "4096", "8192", "16384")
        """
        if "annotations" not in self.pod_template:
            self.pod_template["annotations"] = {}
        self.pod_template["annotations"]["gpu-memory"] = value

    @property
    def volumes(self):
        if not self._volumes:
            volumes = []
            if "volumes" in self.pod_template:
                for volume in self.pod_template["volumes"]:
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

    @property
    def shared_memory_limit(self):
        if "volumes" not in self.pod_template:
            return None

        for volume in self.pod_template["volumes"]:
            if volume.get("name") == "dshm" and "emptyDir" in volume:
                empty_dir = volume["emptyDir"]
                return empty_dir.get("sizeLimit")

        return None

    @shared_memory_limit.setter
    def shared_memory_limit(self, value: str):
        """
        Args:
            value: Size limit (e.g., "512Mi", "1Gi", "2G")
        """
        if "volumes" not in self.pod_template:
            self.pod_template["volumes"] = []

        # Find existing dshm volume and update it
        for volume in self.pod_template["volumes"]:
            if volume.get("name") == "dshm" and "emptyDir" in volume:
                volume["emptyDir"]["sizeLimit"] = value
                return

        # Add new dshm volume if not found
        self.pod_template["volumes"].append({"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": value}})

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
        return self.pod_template.get("nodeSelector")

    @node_selector.setter
    def node_selector(self, value: dict):
        """
        Args:
            value: Label key-value pairs (e.g., {"node-type": "gpu"})
        """
        self.pod_template["nodeSelector"] = value

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
                        if "env_vars" not in existing_secret:
                            existing_secret["env_vars"] = []
                        if env_var["name"] not in existing_secret["env_vars"]:
                            existing_secret["env_vars"].append(env_var["name"])
                    else:
                        secret_env_vars.append({"secret_name": secret_name, "env_vars": [env_var["name"]]})
        return secret_env_vars

    @property
    def secret_volumes(self):
        secret_volumes = []
        if "volumes" in self.pod_template:
            for volume in self.pod_template["volumes"]:
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
        return self.pod_template.get("serviceAccountName")

    @service_account_name.setter
    def service_account_name(self, value: str):
        """Set service account name in the pod template."""
        self.pod_template["serviceAccountName"] = value

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
        """Set image pull policy in the pod template."""
        self._container()["imagePullPolicy"] = value

    @property
    def namespace(self):
        return self._namespace

    @namespace.setter
    def namespace(self, value: str):
        self._namespace = value

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
            if "volumes" in self.pod_template:
                for volume in self.pod_template["volumes"]:
                    if "secret" in volume:
                        secret_name = volume["secret"]["secretName"]
                        if secret_name not in secrets:
                            secrets.append(secret_name)

            self._secrets = secrets

        return self._secrets

    @property
    def gpu_anti_affinity(self):
        if "affinity" in self.pod_template and "nodeAffinity" in self.pod_template["affinity"]:
            node_affinity = self.pod_template["affinity"]["nodeAffinity"]
            if "requiredDuringSchedulingIgnoredDuringExecution" in node_affinity:
                required = node_affinity["requiredDuringSchedulingIgnoredDuringExecution"]
                if "nodeSelectorTerms" in required:
                    for term in required["nodeSelectorTerms"]:
                        if "matchExpressions" in term:
                            for expr in term["matchExpressions"]:
                                if expr.get("key") == "nvidia.com/gpu" and expr.get("operator") == "DoesNotExist":
                                    return True
        return False

    @property
    def concurrency(self):
        return self.pod_template.get("containerConcurrency")

    @concurrency.setter
    def concurrency(self, value: int):
        self.pod_template["containerConcurrency"] = value

    @property
    def working_dir(self):
        return self._container().get("workingDir")

    @working_dir.setter
    def working_dir(self, value: str):
        """Set working directory in the pod template."""
        self._container()["workingDir"] = value

    @property
    def priority_class_name(self):
        return self.pod_template.get("priorityClassName")

    @priority_class_name.setter
    def priority_class_name(self, value: str):
        """Set priority class name in the pod template."""
        self.pod_template["priorityClassName"] = value

    @property
    def otel_enabled(self):
        container = self._container()
        if "env" in container:
            for env_var in container["env"]:
                if env_var["name"] == "KT_OTEL_ENABLED" and "value" in env_var:
                    return env_var["value"].lower() == "true"
        return False

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
        if "startupProbe" not in container:
            container["startupProbe"] = {}
        # Convert timeout to failure threshold (launch_timeout // 5)
        container["startupProbe"]["failureThreshold"] = value // 5

    def queue_name(self):
        if self.queue is not None:
            return self.queue

        default_queue = globals.config.queue
        if default_queue:
            return default_queue

    @property
    def queue(self):
        return self._queue

    @queue.setter
    def queue(self, value: str):
        self._queue = value

    @property
    def scheduler_name(self):
        return self._get_scheduler_name(self.queue_name())

    @property
    def inactivity_ttl(self):
        container = self._container()
        if "env" in container:
            for env_var in container["env"]:
                if env_var["name"] == "KT_INACTIVITY_TTL" and "value" in env_var:
                    return env_var["value"] if not env_var["value"] == "None" else None
        return None

    @inactivity_ttl.setter
    def inactivity_ttl(self, value: str):
        if value and (not isinstance(value, str) or not re.match(r"^\d+[smhd]$", value)):
            raise ValueError("Inactivity TTL must be a string, e.g. '5m', '1h', '1d'")
        if value and not self.otel_enabled:
            logger.warning(
                "Inactivity TTL is only supported when OTEL is enabled, please update your Kubetorch Helm chart and restart the nginx proxy"
            )

        container = self._container()
        if "env" not in container:
            container["env"] = []

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
        return self._autoscaling_config

    @property
    def distributed_config(self):
        # First try to get from pod template
        template_config = None
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

        # Return template config if available, otherwise return stored config
        return template_config

    @distributed_config.setter
    def distributed_config(self, config: dict):
        # Update pod template with distributed config
        container = self._container()
        if "env" not in container:
            container["env"] = []

        # Update or add KT_SERVICE_DNS, KT_DISTRIBUTED_CONFIG env vars
        import json

        service_dns = None
        if config and config.get("distribution_type") == "ray":
            service_dns = "ray-head-svc"
        elif config and config.get("distribution_type") == "pytorch":
            service_dns = "rank0"

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

        service_dns_found, distributed_config_found = False, False
        for env_var in self._container_env():
            if env_var["name"] == "KT_SERVICE_DNS" and service_dns:
                env_var["value"] = service_dns
                service_dns_found = True
            elif env_var["name"] == "KT_DISTRIBUTED_CONFIG":
                env_var["value"] = json.dumps(config)
                distributed_config_found = True

        # Add any missing env vars
        if service_dns and not service_dns_found:
            container["env"].append({"name": "KT_SERVICE_DNS", "value": service_dns})
        if not distributed_config_found:
            container["env"].append({"name": "KT_DISTRIBUTED_CONFIG", "value": json.dumps(config)})

    @property
    def deployment_mode(self):
        # Determine deployment mode based on distributed config and autoscaling.
        # For distributed workloads, always use the appropriate deployment mode
        if self.distributed_config:
            distribution_type = self.distributed_config.get("distribution_type")
            if distribution_type == "pytorch":
                return "deployment"
            elif distribution_type == "ray":
                return "raycluster"

        # Use Knative for autoscaling services
        if self.autoscaling_config:
            return "knative"

        # Default to deployment mode for simple workloads
        return "deployment"

    # ----------------- Service Level Properties ----------------- #

    @property
    def service_name(self):
        # Get service name from pod template if available, otherwise return stored service name
        if not self._service_name:
            for env_var in self._container_env():
                if env_var["name"] == "KT_SERVICE_NAME" and "value" in env_var:
                    self._service_name = env_var["value"] if not env_var["value"] == "None" else None
                    break
        return self._service_name

    @service_name.setter
    def service_name(self, value: str):
        """Set the service name."""
        if self._service_name and not self._service_name == value:
            raise ValueError("Service name cannot be changed after it has been set")
        self._service_name = value

    # ----------------- GPU Properties ----------------- #

    @property
    def tolerations(self):
        return self.pod_template.get("tolerations", [])

    @property
    def gpu_annotations(self):
        # GPU annotations for KAI scheduler
        return self._gpu_annotations

    # ----------------- Init Template Setup Helpers ----------------- #
    def _get_server_image(self, image, otel_enabled, inactivity_ttl):
        """Return base server image"""
        image = self.image.image_id if self.image and self.image.image_id else None

        if not image or image == serving_constants.KUBETORCH_IMAGE_TRAPDOOR:
            # No custom image or Trapdoor → pick OTEL or default
            if self._server_should_enable_otel(otel_enabled, inactivity_ttl):
                return serving_constants.SERVER_IMAGE_WITH_OTEL
            return serving_constants.SERVER_IMAGE_MINIMAL

        return image

    def _get_requested_resources(self, cpus, memory, disk_size, gpu_config):
        """Return requested resources."""
        requests = {}
        limits = {}

        # Add CPU if specified
        if cpus:
            requests["cpu"] = RequestedPodResources.cpu_for_resource_request(cpus)
            limits["cpu"] = requests["cpu"]

        # Add Memory if specified
        if memory:
            requests["memory"] = RequestedPodResources.memory_for_resource_request(memory)
            limits["memory"] = requests["memory"]

        # Add Storage if specified
        if disk_size:
            requests["ephemeral-storage"] = disk_size
            limits["ephemeral-storage"] = disk_size

        # Add GPU if specified
        gpu_config: dict = gpu_config
        gpu_count = gpu_config.get("count", 1)
        if gpu_config:
            if gpu_config.get("sharing_type") == "memory":
                # TODO: not currently supported
                # For memory-sharing GPUs, we don't need to request any additional resources - the KAI scheduler
                # will handle it thru annotations
                return V1ResourceRequirements()
            elif gpu_config.get("sharing_type") == "fraction":
                # For fractional GPUs, we still need to request the base GPU resource
                requests["nvidia.com/gpu"] = "1"
                limits["nvidia.com/gpu"] = "1"
            elif not gpu_config.get("sharing_type"):
                # Whole GPUs
                requests["nvidia.com/gpu"] = str(gpu_count)
                limits["nvidia.com/gpu"] = str(gpu_count)

        # Only include non-empty dicts
        resources = {}
        if requests:
            resources["requests"] = requests
        if limits:
            resources["limits"] = limits

        return V1ResourceRequirements(**resources).to_dict()

    def _get_launch_timeout(self, launch_timeout):
        if launch_timeout:
            return int(launch_timeout)
        default_launch_timeout = (
            self.default_config["launch_timeout"]
            if "launch_timeout" in self.default_config
            else serving_constants.KT_LAUNCH_TIMEOUT
        )
        return int(os.getenv("KT_LAUNCH_TIMEOUT", default_launch_timeout))

    def _get_scheduler_name(self, queue_name):
        return serving_constants.KAI_SCHEDULER_NAME if queue_name else None

    def _get_config_env_vars(self, allowed_serialization):
        config_env_vars = globals.config._get_config_env_vars()
        if allowed_serialization:
            config_env_vars["KT_ALLOWED_SERIALIZATION"] = ",".join(allowed_serialization)

        return config_env_vars

    def _server_should_enable_otel(self, otel_enabled, inactivity_ttl):
        return otel_enabled and inactivity_ttl

    def _should_install_otel_dependencies(self, server_image, otel_enabled, inactivity_ttl):
        return (
            self._server_should_enable_otel(otel_enabled, inactivity_ttl)
            and server_image != serving_constants.SERVER_IMAGE_WITH_OTEL
        )

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

    def _get_gpu_annotations(self, gpu_config):
        # https://blog.devops.dev/struggling-with-gpu-waste-on-kubernetes-how-kai-schedulers-sharing-unlocks-efficiency-1029e9bd334b
        if gpu_config is None:
            return {}

        if gpu_config.get("sharing_type") == "memory":
            return {
                "gpu-memory": str(gpu_config["gpu_memory"]),
            }
        elif gpu_config.get("sharing_type") == "fraction":
            return {
                "gpu-fraction": str(gpu_config["gpu_fraction"]),
            }
        else:
            return {}

    def _load_gpu_config(self, gpus, gpu_memory, gpu_type) -> dict:
        if all(x is None for x in [gpus, gpu_memory, gpu_type]):
            return {}

        if gpus is not None:
            if isinstance(gpus, (int, float)):
                if gpus <= 0:
                    raise ValueError("GPU count must be greater than 0")
                if gpus < 1:
                    raise ValueError("Fractional GPUs are not currently supported. Please use whole GPUs.")
            if not str(gpus).isdigit():
                raise ValueError("Unexpected format for GPUs, expecting a numeric count")

        gpu_config = {
            "count": int(gpus) if gpus else 1,
            "sharing_type": None,
            "gpu_memory": None,
            "gpu_type": None,
        }

        # Handle memory specification
        if gpu_memory is not None:
            if not isinstance(gpu_memory, str):
                raise ValueError("GPU memory must be a string with suffix Mi, Gi, or Ti")

            units = {"mi": 1, "gi": 1024, "ti": 1024 * 1024}
            val = gpu_memory.lower()

            for suffix, factor in units.items():
                if val.endswith(suffix):
                    try:
                        num = float(val[: -len(suffix)])
                        mi_value = int(num * factor)
                        gpu_config["sharing_type"] = "memory"
                        gpu_config["gpu_memory"] = str(mi_value)
                        break  # Successfully parsed, exit the loop
                    except ValueError:
                        raise ValueError("Invalid numeric value in GPU memory spec")
            else:
                # Only raise error if no suffix matched
                raise ValueError("GPU memory must end with Mi, Gi, or Ti")

        if gpu_type is not None:
            gpu_config["gpu_type"] = gpu_type

        return GPUConfig(**gpu_config).to_dict()

    # ----------------- Generic Helpers ----------------- #
    def _load_kube_config(self):
        try:
            config.load_incluster_config()
        except config.config_exception.ConfigException:
            # Fall back to a local kubeconfig file
            if not Path(self.kubeconfig_path).exists():
                raise FileNotFoundError(f"Kubeconfig file not found: {self.kubeconfig_path}")
            config.load_kube_config(config_file=self.kubeconfig_path)

        # Reset the cached API clients so they'll be reinitialized with the loaded config
        self._objects_api = None
        self._core_api = None
        self._apps_v1_api = None

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
        startup_rsync_command: Optional[str],
        launch_id: Optional[str],
        dryrun: bool = False,
    ):
        """Creates a new service on the compute for the provided service. If the service already exists,
        it will update the service with the latest copy of the code."""
        # Finalize pod template with launch time env vars
        self._update_launch_env_vars(service_name, pointer_env_vars, metadata_env_vars, launch_id)
        self._upload_secrets_list()

        setup_script = self._get_setup_script(install_url, startup_rsync_command)
        self._container()["args"][0] = setup_script

        # Handle service template creation
        # Use the replicas property for deployment scaling
        replicas = self.replicas

        # Prepare annotations for service creation, including kubeconfig path if provided
        if self._kubeconfig_path is not None:
            self.annotations[serving_constants.KUBECONFIG_PATH_ANNOTATION] = self._kubeconfig_path

        # Create service using the appropriate service manager
        # KnativeServiceManager will handle autoscaling config, inactivity_ttl, etc.
        # ServiceManager will handle replicas for deployments and rayclusters
        created_service = self.service_manager.create_or_update_service(
            service_name=service_name,
            module_name=pointer_env_vars["KT_MODULE_NAME"],
            pod_template=self.pod_template,
            replicas=replicas,
            autoscaling_config=self.autoscaling_config,
            gpu_annotations=self.gpu_annotations,
            inactivity_ttl=self.inactivity_ttl,
            custom_labels=self.labels,
            custom_annotations=self.annotations,
            custom_template=self.service_template,
            deployment_mode=self.deployment_mode,
            dryrun=dryrun,
            scheduler_name=self.scheduler_name,
            queue_name=self.queue_name(),
        )

        # Handle service creation result based on resource type
        if isinstance(created_service, dict):
            # For custom resources (RayCluster, Knative), created_service is a dictionary
            service_name = created_service.get("metadata", {}).get("name")
            kind = created_service.get("kind", "")

            if kind == "RayCluster":
                # RayCluster has headGroupSpec instead of template
                service_template = {
                    "metadata": {
                        "name": service_name,
                        "namespace": created_service.get("metadata", {}).get("namespace"),
                    },
                    "spec": {"template": created_service["spec"]["headGroupSpec"]["template"]},
                }
            else:
                # For Knative services and other dict-based resources
                service_template = created_service["spec"]["template"]
        else:
            # For Deployments, created_service is a V1Deployment object
            service_name = created_service.metadata.name
            # Return dict format for compatibility with tests and reload logic
            service_template = {
                "metadata": {
                    "name": created_service.metadata.name,
                    "namespace": created_service.metadata.namespace,
                },
                "spec": {"template": created_service.spec.template},
            }

        logger.debug(f"Successfully deployed {self.deployment_mode} service {service_name}")

        return service_template

    async def _launch_async(
        self,
        service_name: str,
        install_url: str,
        pointer_env_vars: Dict,
        metadata_env_vars: Dict,
        startup_rsync_command: Optional[str],
        launch_id: Optional[str],
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
            if globals.config.cluster_config.get("otel_enabled"):
                kt_env_vars["KT_OTEL_ENABLED"] = True

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

    def _extract_secrets(self, secrets):
        if is_running_in_kubernetes():
            return [], []

        secret_env_vars = []
        secret_volumes = []
        if secrets:
            secrets_client = KubernetesSecretsClient(namespace=self.namespace, kubeconfig_path=self.kubeconfig_path)
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
            install_otel=self._should_install_otel_dependencies(
                self.server_image, self.otel_enabled, self.inactivity_ttl
            ),
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
        return [pod.metadata.name for pod in pods if pod_is_running(pod)]

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
                    volume = Volume.from_name(vol, create_if_missing=False, core_v1=self.core_api)
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
            service_name=self.service_name,
            launch_timeout=self.launch_timeout,
            objects_api=self.objects_api,
            core_api=self.core_api,
            queue_name=self.queue_name(),
            scheduler_name=self.scheduler_name,
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
                if pod.status.phase != "Running":
                    logger.info(f"Pod {pod.metadata.name} is not running. Status: {pod.status.phase}")
                    return False
        except client.exceptions.ApiException:
            return False
        return True

    def _base_rsync_url(self, local_port: int):
        return f"rsync://localhost:{local_port}/data/{self.namespace}/{self.service_name}"

    def _rsync_svc_url(self):
        return f"rsync://kubetorch-rsync.{self.namespace}.svc.cluster.local:{serving_constants.REMOTE_RSYNC_PORT}/data/{self.namespace}/{self.service_name}/"

    def ssh(self, pod_name: str = None):
        pod_name = pod_name or self.pod_names()[0]
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
        python_path = self.image.python_path or "python3"
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
        self.rsync(source=full_path, dest=dest_dir)

    def run_bash(
        self,
        commands,
        node: Union[str, List[str]] = None,
        container: Optional[str] = None,
    ):
        """Run bash commands on the pod(s)."""
        self._load_kube_config()

        pod_names = self.pod_names() if node in ["all", None] else [node] if isinstance(node, str) else node

        return _run_bash(
            commands=commands,
            core_api=self.core_api,
            pod_names=pod_names,
            namespace=self.namespace,
            container=container,
        )

    def _create_rsync_target_dir(self):
        """Create the subdirectory for this particular service in the rsync pod."""
        subdir = f"/data/{self.namespace}/{self.service_name}"

        label_selector = f"app={serving_constants.RSYNC_SERVICE_NAME}"
        pod_name = (
            self.core_api.list_namespaced_pod(namespace=self.namespace, label_selector=label_selector)
            .items[0]
            .metadata.name
        )
        subdir_cmd = f"kubectl exec {pod_name} -n {self.namespace} -- mkdir -p {subdir}"
        logger.info(f"Creating directory on rsync pod with cmd: {subdir_cmd}")
        subprocess.run(subdir_cmd, shell=True, check=True)

    def _run_rsync_command(self, rsync_cmd, create_target_dir: bool = True):
        backup_rsync_cmd = rsync_cmd
        if "--mkpath" not in rsync_cmd and create_target_dir:
            # Warning: --mkpath requires rsync 3.2.0+
            # Note: --mkpath allows the rsync daemon to create all intermediate directories that may not exist
            # https://download.samba.org/pub/rsync/rsync.1#opt--mkpath
            rsync_cmd = rsync_cmd.replace("rsync ", "rsync --mkpath ", 1)
            logger.debug(f"Rsync command: {rsync_cmd}")

            resp = subprocess.run(
                rsync_cmd,
                shell=True,
                capture_output=True,
                text=True,
            )
            if resp.returncode != 0:
                if (
                    create_target_dir
                    and ("rsync: --mkpath" in resp.stderr or "rsync: unrecognized option" in resp.stderr)
                    and not is_running_in_kubernetes()
                ):
                    logger.warning(
                        "Rsync failed: --mkpath is not supported, falling back to creating target dir. "
                        "Please upgrade rsync to 3.2.0+ to improve performance."
                    )
                    self._create_rsync_target_dir()
                    return self._run_rsync_command(backup_rsync_cmd, create_target_dir=False)

                raise RsyncError(rsync_cmd, resp.returncode, resp.stdout, resp.stderr)
        else:
            import fcntl
            import pty
            import select

            logger.debug(f"Rsync command: {rsync_cmd}")

            leader, follower = pty.openpty()
            proc = subprocess.Popen(
                shlex.split(rsync_cmd),
                stdout=follower,
                stderr=follower,
                text=True,
                close_fds=True,
            )
            os.close(follower)

            # Set to non-blocking mode
            flags = fcntl.fcntl(leader, fcntl.F_GETFL)
            fcntl.fcntl(leader, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            buffer = b""
            transfer_completed = False
            error_patterns = [
                r"rsync\(\d+\): error:",
                r"rsync error:",
                r"@ERROR:",
            ]
            error_regexes = [re.compile(pattern, re.IGNORECASE) for pattern in error_patterns]

            try:
                with os.fdopen(leader, "rb", buffering=0) as stdout:
                    while True:
                        rlist, _, _ = select.select([stdout], [], [], 0.1)  # 0.1 sec timeout for responsiveness
                        if stdout in rlist:
                            try:
                                chunk = os.read(stdout.fileno(), 1024)
                            except BlockingIOError:
                                continue  # no data available, try again

                            if not chunk:  # EOF
                                break

                            buffer += chunk
                            while b"\n" in buffer:
                                line, buffer = buffer.split(b"\n", 1)
                                decoded_line = line.decode(errors="replace").strip()
                                logger.debug(f"{decoded_line}")

                                for error_regex in error_regexes:
                                    if error_regex.search(decoded_line):
                                        raise RsyncError(rsync_cmd, 1, decoded_line, decoded_line)

                                if "total size is" in decoded_line and "speedup is" in decoded_line:
                                    transfer_completed = True

                            if transfer_completed:
                                break

                        exit_code = proc.poll()
                        if exit_code is not None:
                            if exit_code != 0:
                                raise RsyncError(
                                    rsync_cmd,
                                    exit_code,
                                    output=decoded_line,
                                    stderr=decoded_line,
                                )
                            break

                    proc.terminate()
            except Exception as e:
                proc.terminate()
                raise e

            if not transfer_completed:
                logger.error("Rsync process ended without completion message")
                proc.terminate()
                raise subprocess.CalledProcessError(
                    1,
                    rsync_cmd,
                    output="",
                    stderr="Rsync completed without success indication",
                )

            logger.info("Rsync operation completed successfully")

    async def _run_rsync_command_async(self, rsync_cmd: str, create_target_dir: bool = True):
        """Async version of _run_rsync_command using asyncio.subprocess."""
        import asyncio

        if "--mkpath" not in rsync_cmd and create_target_dir:
            # Warning: --mkpath requires rsync 3.2.0+
            # Note: --mkpath allows the rsync daemon to create all intermediate directories that may not exist
            # https://download.samba.org/pub/rsync/rsync.1#opt--mkpath
            rsync_cmd = rsync_cmd.replace("rsync ", "rsync --mkpath ", 1)
            logger.debug(f"Rsync command: {rsync_cmd}")

            # Use asyncio.create_subprocess_shell for shell commands
            proc = await asyncio.create_subprocess_shell(
                rsync_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout_bytes, stderr_bytes = await proc.communicate()
            stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
            stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""

            if proc.returncode != 0:
                if proc.returncode is None:
                    proc.terminate()
                if "rsync: --mkpath" in stderr or "rsync: unrecognized option" in stderr:
                    error_msg = (
                        "Rsync failed: --mkpath is not supported, please upgrade your rsync version to 3.2.0+ to "
                        "improve performance (e.g. `brew install rsync`)"
                    )
                    raise RsyncError(rsync_cmd, proc.returncode, stdout, error_msg)

                raise RsyncError(rsync_cmd, proc.returncode, stdout, stderr)

    def _get_rsync_cmd(
        self,
        source: Union[str, List[str]],
        dest: str,
        rsync_local_port: int,
        contents: bool = False,
        filter_options: str = None,
        force: bool = False,
    ):
        if dest:
            # Handle tilde prefix - treat as relative to home/working directory
            if dest.startswith("~/"):
                # Strip ~/ prefix to make it relative
                dest = dest[2:]

            # Handle absolute vs relative paths
            if dest.startswith("/"):
                # For absolute paths, store under special __absolute__ subdirectory in the rsync pod
                # to preserve the path structure
                dest_for_rsync = f"__absolute__{dest}"
            else:
                # Relative paths work as before
                dest_for_rsync = dest
            remote_dest = f"{self._base_rsync_url(rsync_local_port)}/{dest_for_rsync}"
        else:
            remote_dest = self._base_rsync_url(rsync_local_port)

        source = [source] if isinstance(source, str) else source

        for src in source:
            if not Path(src).expanduser().exists():
                raise ValueError(f"Could not locate path to sync up: {src}")

        exclude_options = _get_rsync_exclude_options()

        expanded_sources = []
        for s in source:
            path = Path(s).expanduser().absolute()
            if not path.exists():
                raise ValueError(f"Could not locate path to sync up: {s}")

            path_str = str(path)
            if contents and path.is_dir() and not str(s).endswith("/"):
                path_str += "/"
            expanded_sources.append(path_str)

        source_str = " ".join(expanded_sources)

        rsync_cmd = f"rsync -avL {exclude_options}"

        if filter_options:
            rsync_cmd += f" {filter_options}"

        if force:
            rsync_cmd += " --ignore-times"

        rsync_cmd += f" {source_str} {remote_dest}"
        return rsync_cmd

    def _get_rsync_in_cluster_cmd(
        self,
        source: Union[str, List[str]],
        dest: str,
        contents: bool = False,
        filter_options: str = None,
        force: bool = False,
    ):
        """Generate rsync command for in-cluster execution."""
        # Handle tilde prefix in dest - treat as relative to home/working directory
        if dest and dest.startswith("~/"):
            dest = dest[2:]  # Strip ~/ prefix to make it relative

        source = [source] if isinstance(source, str) else source
        if self.working_dir:
            source = [src.replace(self.working_dir, "") for src in source]

        if contents:
            if self.working_dir:
                source = [s if s.endswith("/") or not Path(self.working_dir, s).is_dir() else s + "/" for s in source]
            else:
                source = [s if s.endswith("/") or not Path(s).is_dir() else s + "/" for s in source]

        source_str = " ".join(source)

        exclude_options = _get_rsync_exclude_options()

        base_remote = self._rsync_svc_url()

        if dest is None:
            # no dest specified -> use base
            remote = base_remote
        elif dest.startswith("rsync://"):
            # if full rsync:// URL -> use as-is
            remote = dest
        else:
            # if relative subdir specified -> append to base
            remote = base_remote + dest.lstrip("/")

        # rsync wants the remote last; ensure it ends with '/' so we copy *into* the dir
        if not remote.endswith("/"):
            remote += "/"

        rsync_command = f"rsync -av {exclude_options}"
        if filter_options:
            rsync_command += f" {filter_options}"
        if force:
            rsync_command += " --ignore-times"

        rsync_command += f" {source_str} {remote}"
        return rsync_command

    def _rsync(
        self,
        source: Union[str, List[str]],
        dest: str,
        rsync_local_port: int,
        contents: bool = False,
        filter_options: str = None,
        force: bool = False,
    ):
        rsync_cmd = self._get_rsync_cmd(source, dest, rsync_local_port, contents, filter_options, force)
        self._run_rsync_command(rsync_cmd)

    async def _rsync_async(
        self,
        source: Union[str, List[str]],
        dest: str,
        rsync_local_port: int,
        contents: bool = False,
        filter_options: str = None,
        force: bool = False,
    ):
        """Async version of _rsync_helper."""
        rsync_cmd = self._get_rsync_cmd(source, dest, rsync_local_port, contents, filter_options, force)
        await self._run_rsync_command_async(rsync_cmd)

    def _get_websocket_info(self, local_port: int):
        rsync_local_port = local_port or serving_constants.LOCAL_NGINX_PORT
        base_url = globals.service_url()

        # api_url = globals.config.api_url

        # # Determine if we need port forwarding to reach nginx proxy
        # should_port_forward = api_url is None

        # if should_port_forward:
        #     base_url = globals.service_url()
        # else:
        #     # Direct access to nginx proxy via ingress
        #     base_url = api_url  # e.g. "https://your.ingress.domain"

        ws_url = f"{http_to_ws(base_url)}/rsync/{self.namespace}/"
        parsed_url = urlparse(base_url)

        # choose a local ephemeral port for the tunnel
        start_from = (parsed_url.port or rsync_local_port) + 1
        websocket_port = find_available_port(start_from, max_tries=10)
        return websocket_port, ws_url

    def rsync(
        self,
        source: Union[str, List[str]],
        dest: str = None,
        local_port: int = None,
        contents: bool = False,
        filter_options: str = None,
        force: bool = False,
    ):
        """Rsync from local to the rsync pod."""
        # Note: use the nginx port by default since the rsync pod sits behind the nginx proxy
        websocket_port, ws_url = self._get_websocket_info(local_port)

        logger.debug(f"Opening WebSocket tunnel on port {websocket_port} to {ws_url}")
        with WebSocketRsyncTunnel(websocket_port, ws_url) as tunnel:
            self._rsync(source, dest, tunnel.local_port, contents, filter_options, force)

    async def rsync_async(
        self,
        source: Union[str, List[str]],
        dest: str = None,
        local_port: int = None,
        contents: bool = False,
        filter_options: str = None,
        force: bool = False,
    ):
        """Async version of rsync. Rsync from local to the rsync pod."""
        websocket_port, ws_url = self._get_websocket_info(local_port)

        logger.debug(f"Opening WebSocket tunnel on port {websocket_port} to {ws_url}")
        with WebSocketRsyncTunnel(websocket_port, ws_url) as tunnel:
            await self._rsync_async(source, dest, tunnel.local_port, contents, filter_options, force)

    def rsync_in_cluster(
        self,
        source: Union[str, List[str]],
        dest: str = None,
        contents: bool = False,
        filter_options: str = None,
        force: bool = False,
    ):
        """Rsync from inside the cluster to the rsync pod."""
        rsync_command = self._get_rsync_in_cluster_cmd(source, dest, contents, filter_options, force)
        self._run_rsync_command(rsync_command)

    async def rsync_in_cluster_async(
        self,
        source: Union[str, List[str]],
        dest: str = None,
        contents: bool = False,
        filter_options: str = None,
        force: bool = False,
    ):
        """Async version of rsync_in_cluster. Rsync from inside the cluster to the rsync pod."""
        rsync_command = self._get_rsync_in_cluster_cmd(source, dest, contents, filter_options, force)
        await self._run_rsync_command_async(rsync_command)

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
                    self.rsync(full_path, dest=dest_dir)
                instructions += f"COPY {full_path} {dest_dir}"
            elif step.step_type == ImageSetupStepType.RSYNC:
                source_path = step.kwargs.get("source")
                dest_dir = step.kwargs.get("dest")
                contents = step.kwargs.get("contents")
                filter_options = step.kwargs.get("filter_options")
                force = step.kwargs.get("force")

                if rsync:
                    if is_running_in_kubernetes():
                        self.rsync_in_cluster(
                            source_path,
                            dest=dest_dir,
                            contents=contents,
                            filter_options=filter_options,
                            force=force,
                        )
                    else:
                        self.rsync(
                            source_path,
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
        state["_objects_api"] = None
        state["_core_api"] = None
        state["_apps_v1_api"] = None
        state["_node_v1_api"] = None
        state["_secrets_client"] = None
        return state

    def __setstate__(self, state):
        """Restore state after pickle deserialization."""
        self.__dict__.update(state)
        # Reset local stateful values to None to ensure clean initialization
        self._endpoint = None
        self._service_manager = None
        self._objects_api = None
        self._core_api = None
        self._apps_v1_api = None
        self._node_v1_api = None
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
            # Set replicas property instead of storing in distributed_config
            self.replicas = workers

        if distributed_config:
            self.distributed_config = distributed_config
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
            # Invalidate cached service manager so it gets recreated with KnativeServiceManager
            self._service_manager = None

        return self
