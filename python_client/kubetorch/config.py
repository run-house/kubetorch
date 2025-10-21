import os
from functools import cached_property
from pathlib import Path

import yaml

from kubetorch.logger import get_logger


logger = get_logger(__name__)


ENV_MAPPINGS = {
    "username": "KT_USERNAME",
    "license_key": "KT_LICENSE_KEY",
    "namespace": "KT_NAMESPACE",
    "install_namespace": "KT_INSTALL_NAMESPACE",
    "install_url": "KT_INSTALL_URL",
    "stream_logs": "KT_STREAM_LOGS",
    "log_verbosity": "KT_LOG_VERBOSITY",
    "queue": "KT_QUEUE",
    "tracing_enabled": "KT_TRACING_ENABLED",
    "volumes": "KT_VOLUMES",
    "api_url": "KT_API_URL",
    "cluster_config": "KT_CLUSTER_CONFIG",
}

DEFAULT_INSTALL_NAMESPACE = "kubetorch"


class KubetorchConfig:
    CONFIG_FILE = Path("~/.kt/config.yaml")

    def __init__(self):
        self._api_url = None
        self._cluster_config = None
        self._install_namespace = None
        self._install_url = None
        self._license_key = None
        self._log_verbosity = None
        self._namespace = None
        self._queue = None
        self._stream_logs = None
        self._tracing_enabled = None
        self._username = None
        self._volumes = None

    @cached_property
    def file_cache(self):
        return self._load_from_file()

    @cached_property
    def current_context(self):
        try:
            from kubetorch.servers.http.utils import is_running_in_kubernetes

            if is_running_in_kubernetes():
                try:
                    with open(
                        "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
                    ) as f:
                        return f.read().strip()
                except FileNotFoundError:
                    return "default"

            else:
                from kubernetes import config

                from kubetorch.utils import load_kubeconfig

                load_kubeconfig()
                _, active_context = config.list_kube_config_contexts()
                return active_context.get("context", {}).get("namespace", "default")

        except Exception:
            return "default"

    @property
    def username(self):
        """Username to use for Kubetorch deployments.

        Used for authentication and resource naming. Will be validated to ensure Kubernetes compatibility.
        """
        if not self._username:
            if self._get_env_var("username"):
                self._username = self._get_env_var("username")
            else:
                self._username = self.file_cache.get("username")
        return self._username

    @username.setter
    def username(self, value):
        """Set kubetorch username for current process."""
        from kubetorch.utils import validate_username

        validated = validate_username(value)
        if validated != value:
            logger.info(
                f"Username was validated and changed to {validated} to be Kubernetes-compatible."
            )
        self._username = validated

    @property
    def license_key(self):
        """License key for authentication and billing.

        Required for usage reporting and cluster authentication.
        Can be found in the `basic install guide <https://www.run.house/kubetorch/installation>`_.
        """
        if not self._license_key:
            if self._get_env_var("license_key"):
                self._license_key = self._get_env_var("license_key")
            else:
                self._license_key = self.file_cache.get("license_key")
        return self._license_key

    @license_key.setter
    def license_key(self, value: str):
        """Set kubetorch license key for current process."""
        self._license_key = value

    @property
    def queue(self):
        """Default queue name for scheduling services.

        Controls how cluster resources are allocated and prioritized for services.
        See `scheduling and queues <https://www.run.house/kubetorch/advanced-installation#scheduling-and-queues>`_ for more info.
        """
        if not self._queue:
            if self._get_env_var("queue"):
                self._queue = self._get_env_var("queue")
            else:
                self._queue = self.file_cache.get("queue")
        return self._queue

    @queue.setter
    def queue(self, value: str):
        self._queue = value

    @property
    def volumes(self):
        if not self._volumes:
            if self._get_env_var("volumes"):
                self._volumes = self._get_env_var("volumes")
            else:
                self._volumes = self.file_cache.get("volumes")
        return self._volumes

    @volumes.setter
    def volumes(self, values):
        if values is None or values == "None":
            self._volumes = None
        elif isinstance(values, str):
            # Handle comma-separated string
            self._volumes = [v.strip() for v in values.split(",") if v.strip()]
        elif isinstance(values, list):
            self._volumes = values
        else:
            raise ValueError(
                "volumes must be a list of strings or comma-separated string"
            )

    @property
    def api_url(self):
        if not self._api_url:
            if self._get_env_var("api_url"):
                self._api_url = self._get_env_var("api_url")
            else:
                self._api_url = self.file_cache.get("api_url")
        return self._api_url

    @api_url.setter
    def api_url(self, value: str):
        self._api_url = value

    @property
    def namespace(self):
        """Default Kubernetes namespace for Kubetorch deployments.

        All services will be deployed to this namespace unless overridden in the
        Compute resource constructor. If `install_namespace` is set, it will override this namespace.

        Priority:
            1. Explicit override
            2. Environment variable
            3. File cache
            4. In-cluster namespace or kubeconfig current context
        """
        if (
            self.install_namespace
            and self.install_namespace != DEFAULT_INSTALL_NAMESPACE
        ):
            self._namespace = self.install_namespace
        elif self._namespace is None:
            ns = self._get_env_var("namespace") or self.file_cache.get("namespace")
            self._namespace = ns or self.current_context
        return self._namespace

    @namespace.setter
    def namespace(self, value):
        """Set namespace for current process."""
        self._namespace = value

    @property
    def install_namespace(self):
        """Namespace for Kubetorch installation. Used for Kubetorch Cloud clients.

        Priority:
            1. Explicit override
            2. Environment variable
            3. File cache
            4. Default install namespace
        """
        if self._install_namespace is None:
            ns = self._get_env_var("install_namespace") or self.file_cache.get(
                "install_namespace"
            )
            self._install_namespace = ns or DEFAULT_INSTALL_NAMESPACE
        return self._install_namespace

    @install_namespace.setter
    def install_namespace(self, value):
        """Set installnamespace for current process."""
        self._install_namespace = value

    @property
    def install_url(self):
        """URL of the Kubetorch version to install.

        Used when installing Kubetorch in a Docker image or remote environment.
        Can be found in the `basic install guide <https://www.run.house/kubetorch/installation>`_.
        """
        if self._install_url is None:
            if self._get_env_var("install_url"):
                self._install_url = self._get_env_var("install_url")
            else:
                self._install_url = self.file_cache.get("install_url")
        return self._install_url

    @install_url.setter
    def install_url(self, value):
        """Set default kubetorch install url in current process."""
        self._install_url = value

    @property
    def log_verbosity(self):
        """Verbosity of logs streamed from a remote deployment.
        Log levels include ``debug``, ``info``, and ``critical``. Default is ``info``.

        Note:
            Only relevant when ``stream_logs`` is set to ``true``.
        """
        from kubetorch.utils import LogVerbosity

        default_verbosity = LogVerbosity.INFO.value

        if self._log_verbosity is None:
            verbosity_env_var = self._get_env_var("log_verbosity")
            if verbosity_env_var:
                try:
                    verbosity_env_var = LogVerbosity(verbosity_env_var).value
                except ValueError:
                    verbosity_env_var = default_verbosity

                self._log_verbosity = verbosity_env_var
            else:
                self._log_verbosity = self.file_cache.get(
                    "log_verbosity", default_verbosity
                )

        return self._log_verbosity

    @log_verbosity.setter
    def log_verbosity(self, value):
        """Set log verbosity."""
        from kubetorch.utils import LogVerbosity

        try:
            # In case we are unsetting log_verbosity, None is a valid value
            verbosity = LogVerbosity(value).value if value else None
        except ValueError:
            raise ValueError(
                "Invalid log verbosity value. Must be one of: 'debug', 'info', 'critical'."
            )

        self._log_verbosity = verbosity

    @property
    def stream_logs(self):
        """Whether to stream logs for Kubetorch services.

        When enabled, logs from remote services are streamed back to your local environment
        in real-time. Verbosity of the streamed logs can be controlled with ``log_verbosity``. Default is ``True``.

        Note:
            Requires `log streaming <https://www.run.house/kubetorch/advanced-installation#log-streaming>`_
            to be configured in your cluster.
        """
        if self._stream_logs is None:
            if self._get_env_var("stream_logs"):
                self._stream_logs = self._get_env_var("stream_logs").lower() == "true"
            else:
                self._stream_logs = self.file_cache.get(
                    "stream_logs", True
                )  # Default to True
        return self._stream_logs

    @stream_logs.setter
    def stream_logs(self, value):
        """Set log streaming for current process."""
        from kubetorch.serving.utils import check_loki_enabled

        bool_value = value

        if not isinstance(value, bool):
            if value is None:
                pass  # case we are unsetting stream_logs, so None is a valid value
            elif isinstance(value, str) and value.lower() in ["true", "false"]:
                bool_value = value.lower() == "true"
            else:
                raise ValueError("stream_logs must be a boolean value")
        if bool_value:
            # Check if the cluster has loki enabled
            if not check_loki_enabled():
                raise ValueError("Loki is not enabled in the cluster")
        self._stream_logs = bool_value

    @property
    def tracing_enabled(self):
        """Whether to enable distributed tracing for services.

        When enabled, provides detailed trace information for debugging and monitoring Kubetorch deployments.
        Default is ``False``.

        Note:
            Requires telemetry stack to be configured. See
            `traces <https://www.run.house/kubetorch/advanced-installation#traces>`_ for more info.
        """
        if self._tracing_enabled is None:
            if self._get_env_var("tracing_enabled"):
                self._tracing_enabled = (
                    self._get_env_var("tracing_enabled").lower() == "true"
                )
            else:
                self._tracing_enabled = self.file_cache.get(
                    "tracing_enabled", False
                )  # Default to False - flip to true successfully only if configured
        return self._tracing_enabled

    @tracing_enabled.setter
    def tracing_enabled(self, value):
        """Set tracing collection."""
        from kubetorch.serving.utils import check_tempo_enabled

        bool_value = value

        if not isinstance(value, bool):
            if value is None:
                pass  # case we are unsetting tracing_enabled, so None is a valid value
            elif isinstance(value, str) and value.lower() in ["true", "false"]:
                bool_value = value.lower() == "true"
            else:
                raise ValueError("tracing_enabled must be a boolean value")
        if bool_value:
            # Check if the cluster has tempo enabled
            if not check_tempo_enabled():
                raise ValueError(
                    "Open telemetry collector and Tempo distributor not found on the cluster. "
                    "See https://www.run.house/kubetorch/advanced-installation for more info."
                )
        self._tracing_enabled = bool_value

    @property
    def cluster_config(self):
        """Cluster Config.
        Default is ``{}``.
        """
        from kubetorch.utils import string_to_dict

        config = self._cluster_config
        if self._cluster_config is None:
            config = string_to_dict(self._get_env_var("cluster_config") or "")
            if not config:
                config = string_to_dict(self.file_cache.get("cluster_config", "{}"))
            self._cluster_config = config
        return config

    @cluster_config.setter
    def cluster_config(self, value):
        """Set Cluster Config."""
        from kubetorch.utils import string_to_dict

        new_value = value
        if not isinstance(new_value, dict):
            if isinstance(new_value, str):
                new_value = string_to_dict(new_value)
            else:
                new_value = {}  # Default to empty dict
        self._cluster_config = new_value

    def __iter__(self):
        for key in ENV_MAPPINGS:
            value = getattr(self, key)
            if value == "None":
                value = None
            yield key, value

    def set(self, key, value):
        if key not in ENV_MAPPINGS:
            raise ValueError(f"Unknown config key: {key}")
        setattr(self, key, value)
        # if key is 'username' and value is None (unsetting username), we'll get the cached username,and not the
        # new value
        new_value = value if value is None else getattr(self, key)
        return new_value

    def get(self, key):
        if key not in ENV_MAPPINGS:
            raise ValueError(f"Unknown config key: {key}")
        return getattr(self, key)

    def write(self, user_values: dict = None):
        """Write out config to local ``~/.kt/config.yaml``, to be used globally."""
        # Ensure directory exists
        self.CONFIG_FILE.expanduser().parent.mkdir(parents=True, exist_ok=True)
        values = {
            k: str(v) if isinstance(v, dict) else v
            for k, v in dict(self).items()
            if v is not None
        }
        if user_values:
            values.update(user_values)

        # Write to file
        with self.CONFIG_FILE.expanduser().open("w") as stream:
            yaml.safe_dump(values, stream)

    def _get_env_var(self, key):
        return os.getenv(ENV_MAPPINGS[key])

    def _get_config_env_vars(self):
        """Get config values as environment variables with proper KT_ prefixes."""
        env_vars = {}
        for key, value in dict(self).items():
            if value is not None and key in ENV_MAPPINGS:
                env_vars[ENV_MAPPINGS[key]] = value
        return env_vars

    def _load_from_file(self):
        if self.CONFIG_FILE.expanduser().exists():
            with open(self.CONFIG_FILE.expanduser(), "r") as stream:
                return yaml.safe_load(stream) or {}
        return {}
