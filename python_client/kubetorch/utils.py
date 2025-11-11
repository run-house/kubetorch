import enum
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from urllib.parse import urlparse

from kubernetes import client, config

from kubetorch.constants import DEFAULT_KUBECONFIG_PATH, MAX_USERNAME_LENGTH
from kubetorch.globals import config as kt_config
from kubetorch.logger import get_logger
from kubetorch.resources.callables.utils import get_local_install_path

logger = get_logger(__name__)


def extract_host_port(url: str):
    """Extract host and port when needed separately from a URL."""
    p = urlparse(url)
    return p.hostname, (p.port or (443 if p.scheme == "https" else 80))


def http_to_ws(url: str) -> str:
    """Convert HTTP/HTTPS URLs to WebSocket URLs, or return as-is if already WS."""
    if url.startswith("https://"):
        return "wss://" + url[len("https://") :]
    if url.startswith("http://"):
        return "ws://" + url[len("http://") :]
    if url.startswith(("ws://", "wss://")):
        return url  # already WebSocket URL
    # Default to ws:// for bare hostnames
    return "ws://" + url


def validate_username(username):
    if username is None:  # will be used in case we run kt config user username
        return username
    # Kubernetes requires service names to follow DNS-1035 label standards
    original_username = username  # if an exception is raised because the username is invalid, we want to print the original provided name
    username = username.lower().replace("_", "-").replace("/", "-")
    # Make sure the first character is a letter
    if not re.match(r"^[a-z]", username):
        # Strip out all the characters before the first letter with a regex
        username = re.sub(r"^[^a-z]*", "", username)
    username = username[:MAX_USERNAME_LENGTH]
    # Make sure username doesn't end or start with a hyphen
    if username.startswith("-") or username.endswith("-"):
        username = username.strip("-")
    reserved = ["kt", "kubetorch", "knative"]
    if username in reserved:
        raise ValueError(f"{original_username} is one of the reserved names: {', '.join(reserved)}")
    if not re.match(r"^[a-z]([-a-z0-9]*[a-z0-9])?$", username):
        raise ValueError(f"{original_username} must be a valid k8s name")
    return username


def load_kubeconfig():
    try:
        config.load_incluster_config()
    except config.config_exception.ConfigException:
        kubeconfig_path = os.getenv("KUBECONFIG") or DEFAULT_KUBECONFIG_PATH
        abs_path = Path(kubeconfig_path).expanduser()
        if not abs_path.exists():
            raise FileNotFoundError(f"Kubeconfig file not found in path: {abs_path}")
        config.load_kube_config(str(abs_path))


def current_git_branch():
    try:
        # For CI env
        branch = (
            os.environ.get("GITHUB_HEAD_REF")  # PRs: source branch name
            or os.environ.get("GITHUB_REF_NAME")  # Pushes: branch name
            or os.environ.get("CI_COMMIT_REF_NAME")  # GitLab
            or os.environ.get("CIRCLE_BRANCH")  # CircleCI
        )
        if not branch:
            branch = (
                subprocess.check_output(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    stderr=subprocess.DEVNULL,
                )
                .decode("utf-8")
                .strip()
            )
        return branch
    except Exception as e:
        logger.debug(f"Failed to load current git branch: {e}")
        return None


def iso_timestamp_to_nanoseconds(timestamp):
    if timestamp is None:
        dt = datetime.now(timezone.utc)
    elif isinstance(timestamp, datetime):
        dt = timestamp
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    elif isinstance(timestamp, str):
        dt = datetime.fromisoformat(timestamp)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    else:
        raise ValueError(f"Unsupported timestamp type: {type(timestamp)}")
    return int(dt.timestamp() * 1e9)


def get_kt_install_url(freeze: bool = False):
    # Returns:
    #   str: kubetorch install url
    #   bool: whether to install in editable mode
    if kt_config.install_url or freeze:
        return kt_config.install_url, False
    local_kt_path = get_local_install_path("kubetorch")
    if local_kt_path and (Path(local_kt_path) / "pyproject.toml").exists():
        return local_kt_path, True
    elif local_kt_path and local_kt_path.endswith(".whl"):
        return local_kt_path, False
    else:
        import kubetorch as kt

        version = kt.__version__
        return version, False


class LogVerbosity(str, enum.Enum):
    DEBUG = "debug"
    INFO = "info"
    CRITICAL = "critical"


####################################################################################################
# Logging redirection
####################################################################################################
class StreamTee(object):
    def __init__(self, instream, outstreams):
        self.instream = instream
        self.outstreams = outstreams

    def write(self, message):
        self.instream.write(message)
        for stream in self.outstreams:
            if message:
                stream.write(message)
                # We flush here to ensure that the logs are written to the file immediately
                # see https://github.com/run-house/runhouse/pull/724
                stream.flush()

    def writelines(self, lines):
        self.instream.writelines(lines)
        for stream in self.outstreams:
            stream.writelines(lines)
            stream.flush()

    def flush(self):
        self.instream.flush()
        for stream in self.outstreams:
            stream.flush()

    def __getattr__(self, item):
        # Needed in case someone calls a method on instream, such as Ray calling sys.stdout.istty()
        return getattr(self.instream, item)


class capture_stdout:
    """Context manager for capturing stdout to a file, list, or stream, while still printing to stdout."""

    def __init__(self, output=None):
        self.output = output
        self._stream = None

    def __enter__(self):
        if self.output is None:
            self.output = StringIO()

        if isinstance(self.output, str):
            self._stream = open(self.output, "w")
        else:
            self._stream = self.output
        sys.stdout = StreamTee(sys.stdout, [self])
        sys.stderr = StreamTee(sys.stderr, [self])
        return self

    def write(self, message):
        self._stream.write(message)

    def flush(self):
        self._stream.flush()

    @property
    def stream(self):
        if isinstance(self.output, str):
            return open(self.output, "r")
        return self._stream

    def list(self):
        if isinstance(self.output, str):
            return self.stream.readlines()
        return (self.stream.getvalue() or "").splitlines()

    def __str__(self):
        return self.stream.getvalue()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(sys.stdout, "instream"):
            sys.stdout = sys.stdout.instream
        if hasattr(sys.stderr, "instream"):
            sys.stderr = sys.stderr.instream
        self._stream.close()
        return False


####################################################################################################
# Logging formatting
####################################################################################################
class ColoredFormatter:
    COLORS = {
        "black": "\u001b[30m",
        "red": "\u001b[31m",
        "green": "\u001b[32m",
        "yellow": "\u001b[33m",
        "blue": "\u001b[34m",
        "magenta": "\u001b[35m",
        "cyan": "\u001b[36m",
        "white": "\u001b[37m",
        "bold": "\u001b[1m",
        "italic": "\u001b[3m",
        "reset": "\u001b[0m",
    }

    @classmethod
    def get_color(cls, color: str):
        return cls.COLORS.get(color, "")


class ServerLogsFormatter:
    def __init__(self, name: str = None):
        self.name = name
        self.start_color = ColoredFormatter.get_color("cyan")
        self.reset_color = ColoredFormatter.get_color("reset")


def initialize_k8s_clients():
    """Initialize Kubernetes API clients."""
    load_kubeconfig()
    return (
        client.CoreV1Api(),
        client.CustomObjectsApi(),
        client.AppsV1Api(),
    )


def string_to_dict(value):
    try:
        result = json.loads(value or "{}")
        return result if isinstance(result, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}
