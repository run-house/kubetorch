import os
import socket
import time
import warnings

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

import httpx
from kubernetes.client import ApiException, CoreV1Api, V1Pod
from kubernetes.utils import parse_quantity

from kubetorch import globals
from kubetorch.logger import get_logger
from kubetorch.servers.http.utils import is_running_in_kubernetes
from kubetorch.serving.constants import (
    KUBETORCH_MONITORING_NAMESPACE,
    LOKI_GATEWAY_SERVICE_NAME,
    PROMETHEUS_SERVICE_NAME,
    PROMETHEUS_URL,
)
from kubetorch.utils import load_kubeconfig

logger = get_logger(__name__)


@dataclass
class GPUConfig:
    count: Optional[int] = None
    memory: Optional[str] = None
    sharing_type: Optional[Literal["memory", "fraction"]] = None
    gpu_memory: Optional[str] = None
    gpu_fraction: Optional[str] = None
    gpu_type: Optional[str] = None

    def __post_init__(self):
        self.validate()

    def validate(self) -> bool:
        if self.count and not isinstance(self.count, int):
            raise ValueError("GPU count must an int")

        if self.sharing_type == "memory":
            if not self.gpu_memory:
                raise ValueError(
                    "GPU memory must be specified when using memory sharing"
                )
        elif self.sharing_type == "fraction":
            if not self.gpu_fraction:
                raise ValueError(
                    "GPU fraction must be specified when using fraction sharing"
                )
            try:
                fraction = float(self.gpu_fraction)
                if not 0 < fraction <= 1:
                    raise ValueError("GPU fraction must be between 0 and 1")
            except ValueError:
                raise ValueError("GPU fraction must be a valid float between 0 and 1")

        return True

    def to_dict(self) -> dict:
        base_dict = {
            "sharing_type": self.sharing_type,
            "count": self.count,
        }

        if self.memory is not None:
            base_dict["memory"] = self.memory

        if self.sharing_type == "memory" and self.gpu_memory:
            base_dict["gpu_memory"] = self.gpu_memory
        if self.sharing_type == "fraction" and self.gpu_fraction:
            # Convert to millicores format
            fraction = float(self.gpu_fraction)
            base_dict["gpu_fraction"] = f"{int(fraction * 1000)}m"
        if self.gpu_type is not None:
            base_dict["gpu_type"] = self.gpu_type

        return base_dict


class RequestedPodResources:
    """Resources requested in a Kubetorch cluster/compute object. Note these are the values we receive
    from launcher the cluster via a Sky dryrun."""

    # Default overhead percentages to account for filesystem overhead, OS files, logs, container runtime, etc.
    MEMORY_OVERHEAD = 0.20
    CPU_OVERHEAD = 0.10
    DISK_OVERHEAD = 0.15
    GPU_OVERHEAD = 0.0

    MIN_MEMORY_GB = 0.1  # 100Mi minimum
    MIN_CPU_CORES = 0.1  # 100m minimum

    CPU_STEPS = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192]
    MEMORY_STEPS = [0.5, 1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768]

    def __init__(
        self,
        memory: Optional[Union[str, float]] = None,
        cpus: Optional[Union[int, float]] = None,
        disk_size: Optional[int] = None,
        num_gpus: Optional[Union[int, dict]] = None,
    ):

        self.memory = (
            max(float(memory), self.MIN_MEMORY_GB) if memory is not None else None
        )
        self.cpus = (
            max(self.normalize_cpu_value(cpus), self.MIN_CPU_CORES)
            if cpus is not None
            else None
        )
        self.disk_size = disk_size
        self.num_gpus = num_gpus

    def __str__(self):
        # Example: RequestedPodResources(memory=16.0, cpus=4.0, disk=NoneGB, gpus={'A10G': 1})"
        disk_str = f"{self.disk_size}GB" if self.disk_size is not None else "None"
        memory = f"{self.memory}GB" if self.memory is not None else "None"

        return (
            f"RequestedPodResources(memory={memory}, cpus={self.cpus}, disk_size={disk_str}, "
            f"num_gpus={self.num_gpus})"
        )

    def __repr__(self):
        return (
            f"RequestedPodResources(memory={self.memory}, cpus={self.cpus}, "
            f"disk_size={self.disk_size}, num_gpus={self.num_gpus})"
        )

    @classmethod
    def cpu_for_resource_request(cls, cpu_val: int = None):
        if cpu_val is None:
            return None

        # Ensure minimum CPU value
        cpu_val = max(float(cpu_val), cls.MIN_CPU_CORES)

        # Convert to millicores (ex: '4.0' -> 4000m)
        return f"{int(float(cpu_val) * 1000)}m"

    @classmethod
    def memory_for_resource_request(cls, memory_val: Union[str, float, int] = None):
        if memory_val is None:
            return None

        # If it's a number, treat as GB
        if isinstance(memory_val, (int, float)):
            gb_val = max(float(memory_val), cls.MIN_MEMORY_GB)
            memory_val = f"{gb_val}Gi"

        # Validate the string - if invalid will throw a ValueError
        parse_quantity(str(memory_val))

        return str(memory_val)

    @classmethod
    def normalize_cpu_value(
        cls, cpu_value: Optional[Union[int, str, float]]
    ) -> Optional[float]:
        """Convert CPU value to float, handling string values with '+' allowed by Sky and Kubetorch."""
        if cpu_value is None:
            return None

        if isinstance(cpu_value, str):
            # Strip the '+' if present and convert to float
            return float(cpu_value.rstrip("+"))

        return float(cpu_value)


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
            raise Exception(f"Port forward failed: {stderr}")

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
                resp = httpx.get(url, timeout=2)
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


def pod_is_running(pod: V1Pod):
    return pod.status.phase == "Running" and pod.metadata.deletion_timestamp is None


def check_loki_enabled(core_api: CoreV1Api = None) -> bool:
    """Check if loki is enabled"""
    if core_api is None:
        load_kubeconfig()
        core_api = CoreV1Api()

    kt_namespace = globals.config.install_namespace

    try:
        # Check if loki-gateway service exists in the namespace
        core_api.read_namespaced_service(
            name=LOKI_GATEWAY_SERVICE_NAME, namespace=kt_namespace
        )
        logger.debug(f"Loki gateway service found in namespace {kt_namespace}")
    except ApiException as e:
        if e.status == 404:
            logger.debug(f"Loki gateway service not found in namespace {kt_namespace}")
            return False

        # Additional permission-proof check: try to ping the internal Loki gateway URL
        # Needed if running in kubernetes without full kubeconfig permissions
        if is_running_in_kubernetes():
            try:
                loki_url = f"http://loki-gateway.{kt_namespace}.svc.cluster.local/loki/api/v1/labels"
                response = httpx.get(loki_url, timeout=2)
                if response.status_code == 200:
                    logger.debug("Loki gateway is reachable")
                else:
                    logger.debug(f"Loki gateway returned status {response.status_code}")
                    return False
            except Exception as e:
                logger.debug(f"Loki gateway is not reachable: {e}")
                return False

    return True


def check_prometheus_enabled(
    prometheus_url: str, namespace: str, core_api: CoreV1Api = None
) -> bool:
    """Check if Prometheus is enabled and reachable."""
    if prometheus_url and prometheus_url != PROMETHEUS_URL:
        return True

    if core_api is None:
        load_kubeconfig()
        core_api = CoreV1Api()

    is_in_kubernetes = is_running_in_kubernetes()

    # Check namespace exists
    try:
        core_api.read_namespace(name=namespace)
    except ApiException as e:
        if e.status == 404:
            logger.debug(f"Prometheus namespace not found: {namespace}")
            return False

    # Check Prometheus service exists
    try:
        core_api.read_namespaced_service(
            name=PROMETHEUS_SERVICE_NAME, namespace=namespace
        )
        logger.debug(f"Prometheus service found in namespace {namespace}")
    except ApiException as e:
        if e.status == 404:
            logger.debug(f"Prometheus service not found: {PROMETHEUS_SERVICE_NAME}")
            return False

        # If running inside the cluster, try hitting the service directly
        if is_in_kubernetes:
            try:
                response = httpx.get(PROMETHEUS_URL, timeout=2)
                if response.status_code == 200:
                    logger.debug("Prometheus is reachable and healthy")
                else:
                    logger.debug(f"Prometheus returned status {response.status_code}")
                    return False
            except Exception as e:
                logger.debug(f"Prometheus is not reachable: {e}")
                return False

    return True


def check_tempo_enabled(core_api: CoreV1Api = None) -> bool:
    if core_api is None:
        load_kubeconfig()
        core_api = CoreV1Api()

    try:
        otel = core_api.read_namespaced_service(
            name="kubetorch-otel-opentelemetry-collector",
            namespace=KUBETORCH_MONITORING_NAMESPACE,
        )
        tempo = core_api.read_namespaced_service(
            name="kubetorch-otel-tempo-distributor",
            namespace=KUBETORCH_MONITORING_NAMESPACE,
        )
        return otel is not None and tempo is not None

    except ApiException as e:
        if e.status == 404:
            return False
        raise


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
