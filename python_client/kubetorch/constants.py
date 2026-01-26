LOCALHOST: str = "127.0.0.1"
DEFAULT_KUBECONFIG_PATH = "~/.kube/config"
MAX_PORT_TRIES = 10

# CLI constants
DOUBLE_SPACE_UNICODE = "\u00A0\u00A0"
BULLET_UNICODE = "\u2022"

MAX_USERNAME_LENGTH = 16

CPU_RATE = 0.01
GPU_RATE = 0.05

KT_MOUNT_FOLDER = "ktfs"
DEFAULT_VOLUME_ACCESS_MODE = "ReadWriteMany"

DASHBOARD_PORT = 3001
GRAFANA_PORT = 3000
DEFAULT_TAIL_LENGTH = 100  # log tail length

# Profiling constants
PYSPY_SAMPLE_RATE_HZ = 100
SUPPORTED_PROFILERS = ["pytorch", "pyspy"]
SUPPORTED_PYSPY_OUTPUTS = ["flamegraph", "raw", "speedscope", "chrometrace"]
SUPPORTED_PYTORCH_OUTPUTS = ["chrometrace", "table", "memory_timeline", "stacks"]
SUPPORTED_PYTORCH_TABLE_SORT_KEYS = [
    "cpu_time",
    "cuda_time",
    "cpu_time_total",
    "cuda_time_total",
    "cpu_memory_usage",
    "cuda_memory_usage",
    "self_cpu_memory_usage",
    "self_cuda_memory_usage",
    "count",
]

# HTTPX
CONTROLLER_CONNECT_TIMEOUT = 30.0  # Time to establish TCP connection
CONTROLLER_READ_TIMEOUT = 30.0  # Time to receive response body for quick operations (list, get)
CONTROLLER_WRITE_TIMEOUT = 30.0  # Time to send request body
CONTROLLER_POOL_TIMEOUT = 10.0  # Time to acquire connection from pool
