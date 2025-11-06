# K8s Configuration
KUBECTL_PORT = 6443
KT_LAUNCH_TIMEOUT = 900  # 15 minutes

# Ports
DEFAULT_NGINX_PORT = 8080
LOCAL_NGINX_PORT = 38080
DEFAULT_KT_SERVER_PORT = 32300  # Standard port of Knative services
DEFAULT_DEBUG_PORT = 5678

# Namespaces
KUBETORCH_NAMESPACE = "kubetorch"
RUNHOUSE_NAMESPACE = "runhouse"
DEFAULT_NAMESPACE = "default"

# Images
SERVER_IMAGE_MINIMAL = "ghcr.io/run-house/server:v3"
SERVER_IMAGE_WITH_OTEL = "ghcr.io/run-house/server-otel:v3"

UBUNTU_IMAGE_MINIMAL = "ghcr.io/run-house/ubuntu:v1"
UBUNTU_IMAGE_WITH_OTEL = "ghcr.io/run-house/ubuntu:v1"

DEFAULT_PROXY_IMAGE = "ghcr.io/run-house/proxy:v2"
KUBETORCH_IMAGE_TRAPDOOR = "kubetorch"

# Service Accounts
DEFAULT_SERVICE_ACCOUNT_NAME = "kubetorch-service-account"

# Annotations
INACTIVITY_TTL_ANNOTATION = "kubetorch.com/inactivity-ttl"
KUBECONFIG_PATH_ANNOTATION = "kubetorch.com/kubeconfig-path"

# Labels
KT_SERVICE_LABEL = "kubetorch.com/service"
KT_VERSION_LABEL = "kubetorch.com/version"
KT_MODULE_LABEL = "kubetorch.com/module"
KT_USER_IDENTIFIER_LABEL = "kubetorch.com/user-identifier"
KT_USERNAME_LABEL = "kubetorch.com/username"
KT_POD_TYPE_LABEL = "kubetorch.com/pod-type"
KT_TEMPLATE_LABEL = "kubetorch.com/template"
KT_SECRET_NAME_LABEL = "kubetorch.com/secret-name"

# Templates
TTL_CONTROLLER_CONFIGMAP_NAME = "kubetorch-ttl-controller-config"
KNATIVE_SERVICE_TEMPLATE_FILE = "knative_service_template.yaml"
POD_TEMPLATE_FILE = "pod_template.yaml"
KT_SETUP_TEMPLATE_FILE = "kt_setup_template.sh.j2"
DEPLOYMENT_TEMPLATE_FILE = "deployment_template.yaml"
DEPLOYMENT_SERVICE_TEMPLATE_FILE = "service_template.yaml"
RAYCLUSTER_TEMPLATE_FILE = "raycluster_template.yaml"
RAYCLUSTER_SERVICE_TEMPLATE_FILE = "raycluster_service_template.yaml"

# Loki
LOKI_GATEWAY_SERVICE_NAME = "loki-gateway"

# Prometheus
PROMETHEUS_SERVICE_NAME = "kubetorch-metrics"

# Grafana
GRAFANA_HEALTH_ENDPOINT = "/api/health"
PROMETHEUS_HEALTH_ENDPOINT = "/metrics"

# KAI
KAI_SCHEDULER_NAME = "kai-scheduler"
KAI_SCHEDULER_LABEL = "kai.scheduler/queue"

# HTTP Client
KT_TERMINATION_REASONS = ["OOMKilled", "Evicted", "Not Found"]

# NGINX
NGINX_GATEWAY_PROXY = "kubetorch-proxy"
DEFAULT_NGINX_HEALTH_ENDPOINT = "/health"

# Rsync
RSYNC_LOCAL_PORT = 3873
REMOTE_RSYNC_PORT = 873
RSYNC_SERVICE_NAME = "kubetorch-rsync"
