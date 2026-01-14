import copy
import os
import socket
import time
import warnings

from pathlib import Path
from typing import Dict, Optional

from kubetorch import globals
from kubetorch.logger import get_logger
from kubetorch.provisioning.autoscaling import AutoscalingConfig
from kubetorch.provisioning.constants import PROMETHEUS_SERVICE_NAME
from kubetorch.serving.global_http_clients import get_sync_client
from kubetorch.serving.utils import is_running_in_kubernetes
from kubetorch.utils import http_not_found

logger = get_logger(__name__)


class KubernetesCredentialsError(Exception):
    pass


def has_k8s_credentials():
    """
    Fast check for K8s credentials - works both in-cluster and external.
    No network calls, no imports needed.
    """
    # Check 1: In-cluster service account
    if (
        Path("/var/run/secrets/kubernetes.io/serviceaccount/token").exists()
        and Path("/var/run/secrets/kubernetes.io/serviceaccount/ca.crt").exists()
    ):
        return True

    # Check 2: Kubeconfig file
    kubeconfig_path = os.environ.get("KUBECONFIG", os.path.expanduser("~/.kube/config"))
    return Path(kubeconfig_path).exists()


def check_kubetorch_versions(response):
    from kubetorch import __version__ as python_client_version, VersionMismatchError

    try:
        data = response.json()
    except ValueError:
        # older nginx proxy versions won't return a JSON
        return

    helm_installed_version = data.get("version")
    if not helm_installed_version:
        logger.debug("No 'version' found in health check response")
        return

    if python_client_version != helm_installed_version:
        msg = (
            f"client={python_client_version}, cluster={helm_installed_version}. "
            "To suppress this error, set the environment variable "
            "`KUBETORCH_IGNORE_VERSION_MISMATCH=1`."
        )
        if not os.getenv("KUBETORCH_IGNORE_VERSION_MISMATCH"):
            raise VersionMismatchError(msg)

        warnings.warn(f"Kubetorch version mismatch: {msg}")


def extract_config_from_nginx_health_check(response):
    """Extract the config from the nginx health check response."""
    try:
        data = response.json()
    except ValueError:
        return
    config = data.get("config", {})
    return config


def wait_for_port_forward(
    process,
    local_port,
    timeout=30,
    health_endpoint: str = None,
    validate_kubetorch_versions: bool = True,
):
    from kubetorch import VersionMismatchError

    start_time = time.time()
    while time.time() - start_time < timeout:
        if process.poll() is not None:
            stderr = process.stderr.read().decode()
            raise Exception(stderr)

        try:
            # Check if socket is open
            with socket.create_connection(("localhost", local_port), timeout=1):
                if not health_endpoint:
                    # If we are not checking HTTP (ex: rsync)
                    return True
        except OSError:
            time.sleep(0.2)
            continue

        if health_endpoint:
            url = f"http://localhost:{local_port}" + health_endpoint
            try:
                # Check if HTTP endpoint is ready
                resp = get_sync_client().get(url, timeout=2)
                if resp.status_code == 200:
                    if validate_kubetorch_versions:
                        check_kubetorch_versions(resp)
                    # Extract config to set outside of function scope
                    config = extract_config_from_nginx_health_check(resp)
                    return config
            except VersionMismatchError as e:
                raise e
            except Exception as e:
                logger.debug(f"Waiting for HTTP endpoint to be ready: {e}")

        time.sleep(0.2)

    raise TimeoutError("Timeout waiting for port forward to be ready")


def pod_is_running(pod: dict) -> bool:
    """Check if pod is running. Pod must be a dict from ControllerClient."""
    status = pod.get("status", {})
    phase = status.get("phase")
    metadata = pod.get("metadata", {})
    deletion_timestamp = metadata.get("deletionTimestamp")
    return phase == "Running" and deletion_timestamp is None


def check_loki_enabled(namespace: str = "default") -> bool:
    """Check if Loki is enabled by checking the data store service in the target namespace.

    Loki is now embedded in the data store, so we check for port 3100 on the data store service.
    """
    controller = globals.controller_client()

    try:
        # Check if data store service exists with Loki port (3100)
        service = controller.get_service(namespace=namespace, name="kubetorch-data-store")
        ports = service.get("spec", {}).get("ports", [])
        for port in ports:
            if port.get("port") == 3100 or port.get("name") == "loki":
                logger.debug(f"Loki enabled in data store service in namespace {namespace}")
                return True
        logger.debug(f"Data store service found but Loki port not exposed in namespace {namespace}")
        return False

    except Exception as e:
        # controller wraps K8s 404 properly
        if http_not_found(e):
            logger.debug(f"Data store service not found in namespace {namespace}")
            return False

        # fallback check: inside cluster, try resolving service DNS
        if is_running_in_kubernetes():
            loki_url = f"http://kubetorch-data-store.{namespace}.svc.cluster.local:3100/loki/api/v1/labels"
            try:
                resp = get_sync_client().get(loki_url, timeout=2)
                if resp.status_code == 200:
                    return True
            except Exception:
                pass

        return False


def check_prometheus_enabled() -> bool:
    """
    Check if Prometheus is enabled using the centralized controller.
    """
    controller = globals.controller_client()
    kt_namespace = globals.config.install_namespace

    try:
        controller.get_service(namespace=kt_namespace, name=PROMETHEUS_SERVICE_NAME)
        logger.debug(f"Prometheus service found in namespace {kt_namespace}")
        return True

    except Exception as e:
        if http_not_found(e):
            logger.debug(f"Prometheus service not found in namespace {kt_namespace}")
            return False

        # DNS fallback for in-cluster execution
        if is_running_in_kubernetes():
            prom_url = f"http://{PROMETHEUS_SERVICE_NAME}.{kt_namespace}.svc.cluster.local/api/v1/labels"
            try:
                resp = get_sync_client().get(prom_url, timeout=2)
                if resp.status_code == 200:
                    return True
            except Exception:
                pass

        return False


def nested_override(original_dict, override_dict):
    for key, value in override_dict.items():
        if key in original_dict:
            if isinstance(original_dict[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                nested_override(original_dict[key], value)
            else:
                original_dict[key] = value  # Custom wins
        else:
            original_dict[key] = value


def nested_merge(user_dict, kt_dict):
    """
    Merge kubetorch dict into user dict, preserving user values.

    Strategy:
    - For simple values (str, int, bool, etc.): user value takes precedence
    - For dicts: recursively merge, user values take precedence
    - For lists:
      - If list items are dicts with a "name" key, merge by name (user items win)
      - If list items are dicts with a "key" key (e.g., tolerations), merge by key
      - Otherwise, append kt items that don't already exist in user list

    Args:
        user_dict (dict): The user's dict (values take precedence).
        kt_dict (dict): The kubetorch dict (values are added where user doesn't have them).

    Returns:
        The merged dict (modifies user_dict in place).
    """
    for key, kt_value in kt_dict.items():
        if key not in user_dict:
            user_dict[key] = copy.deepcopy(kt_value)
        else:
            user_value = user_dict[key]

            # Both have the key - merge based on type
            if isinstance(user_value, dict) and isinstance(kt_value, dict):  # Recursively merge dicts
                nested_merge(user_value, kt_value)
            elif isinstance(user_value, list) and isinstance(kt_value, list):  # Merge lists
                # Check if we're dealing with lists of dicts by checking first item of either list
                first_user_item = user_value[0] if user_value else None
                first_kt_item = kt_value[0] if kt_value else None

                if (first_user_item and isinstance(first_user_item, dict)) or (
                    first_kt_item and isinstance(first_kt_item, dict)
                ):
                    merge_key = None
                    sample_item = first_user_item or first_kt_item
                    if sample_item:
                        if "name" in sample_item:
                            merge_key = "name"
                        elif "key" in sample_item:
                            merge_key = "key"

                    if merge_key:
                        # Merge by the identified key
                        user_dict_by_key = {item.get(merge_key): item for item in user_value if item.get(merge_key)}
                        # Add kt items that don't conflict with user items
                        for kt_item in kt_value:
                            kt_key_value = kt_item.get(merge_key)
                            if kt_key_value and kt_key_value not in user_dict_by_key:
                                # Add kt item if user doesn't have one with this key
                                user_dict_by_key[kt_key_value] = copy.deepcopy(kt_item)
                        # Preserve items without the merge key from both lists
                        user_items_without_key = [item for item in user_value if not item.get(merge_key)]
                        kt_items_without_key = [copy.deepcopy(item) for item in kt_value if not item.get(merge_key)]
                        # Rebuild list: items with key (merged) + user items without key + kt items without key
                        user_dict[key] = list(user_dict_by_key.values()) + user_items_without_key + kt_items_without_key
                    else:
                        # List of dicts without a known merge key - append items that don't exist
                        user_set = set(str(item) for item in user_value)
                        for kt_item in kt_value:
                            if str(kt_item) not in user_set:
                                user_value.append(copy.deepcopy(kt_item))
                else:
                    # Simple list - append items that don't exist
                    user_set = set(str(item) for item in user_value)
                    for kt_item in kt_value:
                        if str(kt_item) not in user_set:
                            user_value.append(copy.deepcopy(kt_item))


# =============================================================================
# Resource Configuration
# =============================================================================

RESOURCE_CONFIGS: Dict[str, dict] = {
    "deployment": {
        "pod_template_path": ["spec", "template"],
        "api_group": "apps",
        "api_version": "v1",
        "api_plural": "deployments",
        "default_routing": None,  # Route to all pods
        "template_label": "deployment",
    },
    "knative": {
        "pod_template_path": ["spec", "template"],
        "api_group": "serving.knative.dev",
        "api_version": "v1",
        "api_plural": "services",
        "default_routing": "knative_url",  # Knative provides its own URL
        "template_label": "ksvc",
        "resource_kind": "KnativeService",
    },
    "raycluster": {
        "pod_template_path": ["spec", "headGroupSpec", "template"],
        "worker_template_path": ["spec", "workerGroupSpecs", 0, "template"],
        "api_group": "ray.io",
        "api_version": "v1",
        "api_plural": "rayclusters",
        "default_routing": {"ray.io/node-type": "head"},  # Route to head only
        "template_label": "raycluster",
    },
    "pytorchjob": {
        "pod_template_path": ["spec", "pytorchReplicaSpecs", "Master", "template"],
        "worker_template_path": ["spec", "pytorchReplicaSpecs", "Worker", "template"],
        "replica_specs_key": "pytorchReplicaSpecs",
        "primary_replica": "Master",
        "container_name": "pytorch",
        "api_group": "kubeflow.org",
        "api_version": "v1",
        "api_plural": "pytorchjobs",
        "default_routing": {"training.kubeflow.org/replica-type": "master"},
        "template_label": "pytorchjob",
    },
    "tfjob": {
        "pod_template_path": ["spec", "tfReplicaSpecs", "Chief", "template"],
        "worker_template_path": ["spec", "tfReplicaSpecs", "Worker", "template"],
        "replica_specs_key": "tfReplicaSpecs",
        "primary_replica": "Chief",
        "container_name": "tensorflow",
        "api_group": "kubeflow.org",
        "api_version": "v1",
        "api_plural": "tfjobs",
        "default_routing": {"training.kubeflow.org/replica-type": "chief"},
        "template_label": "tfjob",
    },
    "mxjob": {
        "pod_template_path": ["spec", "mxReplicaSpecs", "Scheduler", "template"],
        "worker_template_path": ["spec", "mxReplicaSpecs", "Worker", "template"],
        "replica_specs_key": "mxReplicaSpecs",
        "primary_replica": "Scheduler",
        "container_name": "mxnet",
        "api_group": "kubeflow.org",
        "api_version": "v1",
        "api_plural": "mxjobs",
        "default_routing": {"training.kubeflow.org/replica-type": "scheduler"},
        "template_label": "mxjob",
    },
    "xgboostjob": {
        "pod_template_path": ["spec", "xgbReplicaSpecs", "Master", "template"],
        "worker_template_path": ["spec", "xgbReplicaSpecs", "Worker", "template"],
        "replica_specs_key": "xgbReplicaSpecs",
        "primary_replica": "Master",
        "container_name": "xgboost",
        "api_group": "kubeflow.org",
        "api_version": "v1",
        "api_plural": "xgboostjobs",
        "default_routing": {"training.kubeflow.org/replica-type": "master"},
        "template_label": "xgboostjob",
    },
    "selector": {
        # Selector-only mode - no K8s resource, just pod discovery
        "pod_template_path": None,
        "default_routing": None,
        "template_label": "selector",
    },
}


def get_resource_config(resource_type: str) -> dict:
    """Get configuration for a resource type."""
    resource_type = resource_type.lower()
    if resource_type not in RESOURCE_CONFIGS:
        raise ValueError(
            f"Unknown resource type: {resource_type}. " f"Supported types: {', '.join(RESOURCE_CONFIGS.keys())}"
        )
    return RESOURCE_CONFIGS[resource_type]


# =============================================================================
# Training Job Constants
# =============================================================================

# Supported training job types (derived from RESOURCE_CONFIGS)
SUPPORTED_TRAINING_JOBS = ["pytorchjob", "tfjob", "mxjob", "xgboostjob"]


# =============================================================================
# Manifest Building Functions
# =============================================================================


def build_deployment_manifest(
    pod_spec: dict,
    namespace: str,
    replicas: int = 1,
    inactivity_ttl: str = None,
    custom_labels: dict = None,
    custom_annotations: dict = None,
    custom_template: dict = None,
) -> dict:
    """Build a base deployment manifest from pod spec and configuration."""
    import os

    from kubetorch import __version__
    from kubetorch.provisioning import constants as provisioning_constants
    from kubetorch.serving.utils import load_template

    # Build labels
    labels = {
        provisioning_constants.KT_VERSION_LABEL: __version__,
        provisioning_constants.KT_TEMPLATE_LABEL: "deployment",
        provisioning_constants.KT_USERNAME_LABEL: globals.config.username,
    }
    if custom_labels:
        labels.update(custom_labels)

    # Template labels (exclude kt template label)
    template_labels = labels.copy()
    template_labels.pop(provisioning_constants.KT_TEMPLATE_LABEL, None)

    # Build annotations
    annotations = {}
    if custom_annotations:
        annotations.update(custom_annotations)
    if inactivity_ttl:
        annotations[provisioning_constants.INACTIVITY_TTL_ANNOTATION] = inactivity_ttl

    # Create Deployment manifest
    deployment = load_template(
        template_file=provisioning_constants.DEPLOYMENT_TEMPLATE_FILE,
        template_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"),
        name="",  # Will be set during launch
        namespace=namespace,
        annotations=annotations,
        template_annotations={},  # Will be filled in during launch
        labels=labels,
        template_labels=template_labels,
        pod_spec=pod_spec,
        replicas=replicas,
    )

    if custom_template:
        nested_override(deployment, custom_template)

    return deployment


def build_knative_manifest(
    pod_spec: dict,
    namespace: str,
    autoscaling_config: Optional[AutoscalingConfig] = None,
    gpu_annotations: dict = None,
    inactivity_ttl: str = None,
    custom_labels: dict = None,
    custom_annotations: dict = None,
) -> dict:
    """Build a Knative Service manifest from pod spec and configuration."""
    import os

    from kubetorch import __version__
    from kubetorch.provisioning import constants as provisioning_constants
    from kubetorch.serving.utils import load_template

    # Build labels
    labels = {
        provisioning_constants.KT_VERSION_LABEL: __version__,
        provisioning_constants.KT_TEMPLATE_LABEL: "ksvc",
        provisioning_constants.KT_USERNAME_LABEL: globals.config.username,
    }
    if custom_labels:
        labels.update(custom_labels)

    # Template labels (exclude kt template label)
    template_labels = labels.copy()
    template_labels.pop(provisioning_constants.KT_TEMPLATE_LABEL, None)

    # Build annotations
    annotations = {}
    if custom_annotations:
        annotations.update(custom_annotations)
    if inactivity_ttl:
        annotations[provisioning_constants.INACTIVITY_TTL_ANNOTATION] = inactivity_ttl

    # Build template annotations for autoscaling
    template_annotations = {}
    if gpu_annotations:
        template_annotations.update(gpu_annotations)

    if autoscaling_config:
        # AutoscalingConfig is a dataclass - access attributes directly
        if autoscaling_config.min_scale is not None:
            template_annotations["autoscaling.knative.dev/min-scale"] = str(autoscaling_config.min_scale)
        if autoscaling_config.max_scale is not None:
            template_annotations["autoscaling.knative.dev/max-scale"] = str(autoscaling_config.max_scale)
        if autoscaling_config.target is not None:
            template_annotations["autoscaling.knative.dev/target"] = str(autoscaling_config.target)

    knative_service = load_template(
        template_file=provisioning_constants.KNATIVE_SERVICE_TEMPLATE_FILE,
        template_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"),
        name="",
        namespace=namespace,
        annotations=annotations,
        labels=labels,
        template_labels=template_labels,
        template_annotations=template_annotations,
        pod_spec=pod_spec,
    )

    return knative_service


def build_raycluster_manifest(
    pod_spec: dict,
    namespace: str,
    replicas: int = 1,
    inactivity_ttl: str = None,
    custom_labels: dict = None,
    custom_annotations: dict = None,
) -> dict:
    """Build a RayCluster manifest from pod spec and configuration."""
    import os

    from kubetorch import __version__
    from kubetorch.provisioning import constants as provisioning_constants
    from kubetorch.serving.utils import load_template

    # Build labels
    labels = {
        provisioning_constants.KT_VERSION_LABEL: __version__,
        provisioning_constants.KT_TEMPLATE_LABEL: "raycluster",
        provisioning_constants.KT_USERNAME_LABEL: globals.config.username,
    }
    if custom_labels:
        labels.update(custom_labels)

    # Template labels for head and worker pods (exclude kt template label)
    head_template_labels = labels.copy()
    head_template_labels.pop(provisioning_constants.KT_TEMPLATE_LABEL, None)
    worker_template_labels = head_template_labels.copy()

    # Build annotations
    annotations = {}
    if custom_annotations:
        annotations.update(custom_annotations)
    if inactivity_ttl:
        annotations[provisioning_constants.INACTIVITY_TTL_ANNOTATION] = inactivity_ttl

    # Template annotations for pod specs
    template_annotations = annotations.copy()

    # Calculate worker replicas (total - 1 for head)
    worker_replicas = max(0, replicas - 1)

    raycluster = load_template(
        template_file=provisioning_constants.RAYCLUSTER_TEMPLATE_FILE,
        template_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"),
        name="",
        namespace=namespace,
        annotations=annotations,
        labels=labels,
        template_annotations=template_annotations,
        head_template_labels=head_template_labels,
        worker_template_labels=worker_template_labels,
        pod_spec=pod_spec,
        worker_replicas=worker_replicas,
    )

    return raycluster
