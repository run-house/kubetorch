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

# HTTPX
CONTROLLER_CONNECT_TIMEOUT = 30.0  # Time to establish TCP connection
CONTROLLER_READ_TIMEOUT = 30.0  # Time to receive response body for quick operations (list, get)
CONTROLLER_WRITE_TIMEOUT = 30.0  # Time to send request body
CONTROLLER_POOL_TIMEOUT = 10.0  # Time to acquire connection from pool
