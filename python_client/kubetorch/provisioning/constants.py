import os

from kubetorch._version import __version__ as KUBETORCH_VERSION

# K8s Configuration
KUBECTL_PORT = 6443
KT_LAUNCH_TIMEOUT = 900  # 15 minutes

# Ports
DEFAULT_NGINX_PORT = 8080
LOCAL_NGINX_PORT = 38080
DEFAULT_KT_SERVER_PORT = 32300  # Standard port of Knative services
DEFAULT_K8S_SERVICE_PORT = 80  # K8s Service port (external), maps to DEFAULT_KT_SERVER_PORT
DEFAULT_DEBUG_PORT = 5678

# Namespaces
KUBETORCH_NAMESPACE = "kubetorch"
DEFAULT_NAMESPACE = "default"

# Images
DEFAULT_IMAGE_NAMESPACE = os.getenv("KUBETORCH_IMAGE_NAMESPACE", "ghcr.io/run-house")


def _versioned_image(image_name: str) -> str:
    return f"{DEFAULT_IMAGE_NAMESPACE}/{image_name}:{KUBETORCH_VERSION}"


SERVER_IMAGE_MINIMAL = _versioned_image("server")
SERVER_IMAGE_WITH_OTEL = _versioned_image("server-otel")

UBUNTU_IMAGE_MINIMAL = _versioned_image("ubuntu")
UBUNTU_IMAGE_WITH_OTEL = _versioned_image("ubuntu-otel")

DEFAULT_PROXY_IMAGE = "ghcr.io/run-house/proxy:v2"
KUBETORCH_IMAGE_TRAPDOOR = "kubetorch"

# Service Accounts
DEFAULT_SERVICE_ACCOUNT_NAME = "kubetorch-service-account"

# Annotations
INACTIVITY_TTL_ANNOTATION = "kubetorch.com/inactivity-ttl"
KUBECONFIG_PATH_ANNOTATION = "kubetorch.com/kubeconfig-path"
ALLOWED_SERIALIZATION_ANNOTATION = "kubetorch.com/allowed-serialization"

# Labels
KT_SERVICE_LABEL = "kubetorch.com/service"
KT_VERSION_LABEL = "kubetorch.com/version"
KT_MODULE_LABEL = "kubetorch.com/module"
KT_USER_IDENTIFIER_LABEL = "kubetorch.com/user-identifier"
KT_USERNAME_LABEL = "kubetorch.com/username"
KT_POD_TYPE_LABEL = "kubetorch.com/pod-type"
KT_TEMPLATE_LABEL = "kubetorch.com/template"
KT_SECRET_NAME_LABEL = "kubetorch.com/secret-name"
KT_APP_LABEL = "app"  # Stable app identifier for easy querying
KUEUE_QUEUE_NAME_LABEL = "kueue.x-k8s.io/queue-name"  # Kueue queue label for GPU scheduling

# Auto-termination labels (placed on KubetorchWorkload CRD for efficient querying)
KT_INACTIVITY_TTL_LABEL = "kubetorch.com/inactivity-ttl"

# Templates
KNATIVE_SERVICE_TEMPLATE_FILE = "knative_service_template.yaml"
POD_TEMPLATE_FILE = "pod_template.yaml"
KT_SETUP_TEMPLATE_FILE = "kt_setup_template.sh.j2"
DEPLOYMENT_TEMPLATE_FILE = "deployment_template.yaml"
RAYCLUSTER_TEMPLATE_FILE = "raycluster_template.yaml"

# Loki
LOKI_GATEWAY_SERVICE_NAME = "loki-gateway"

# Prometheus
PROMETHEUS_SERVICE_NAME = "kubetorch-metrics"

# Grafana
GRAFANA_HEALTH_ENDPOINT = "/api/health"

# HTTP Client
KT_TERMINATION_REASONS = ["OOMKilled", "Evicted", "Not Found"]

# Controller
KUBETORCH_CONTROLLER = "kubetorch-controller"
DEFAULT_NGINX_HEALTH_ENDPOINT = "/health"

# Data Store (formerly Rsync)
RSYNC_LOCAL_PORT = 3873
REMOTE_RSYNC_PORT = 873
DATA_STORE_SERVICE_NAME = "kubetorch-data-store"
DATA_STORE_METADATA_PORT = 8081

# Runhouse
KUBETORCH_UI_SERVICE_NAME = "kubetorch-ui-service"
import os

from kubetorch._version import __version__ as KUBETORCH_VERSION
