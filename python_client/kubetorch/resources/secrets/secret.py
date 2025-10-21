import os
from typing import Dict, List, Optional, Tuple

from kubetorch.globals import config

from kubetorch.resources.secrets.utils import read_files_as_secrets_dict


class Secret:
    _DEFAULT_PATH = None
    _DEFAULT_FILENAMES = None
    _DEFAULT_ENV_VARS = {}
    _MAP_FILENAMES_TO_ENV_VARS = {}
    _PROVIDER = None

    def __init__(
        self,
        name: Optional[str] = None,
        provider: Optional[str] = None,
        values: Dict = None,
        path: str = None,
        env_vars: Dict = None,
        override: bool = False,
        **kwargs,
    ):
        """
        Secret class. Built-in provider classes contain default path and/or environment variable mappings,
        based on it's expected usage.

        Note:
            Currently supported built-in providers:
            anthropic, aws, azure, gcp, github, huggingface, lambda, langchain, openai, pinecone, ssh, wandb.

        Args:
            name (str, optional): Name to assign the Kubetorch secret.
            provider (str, optional): Provider corresponding to the secret (e.g. "aws", "gcp").
            values (Dict, optional): Dictionary mapping secret keys to the corresponding secret values.
            path (str, optional): Path where the secret values are held.
            env_vars (Dict, optional): Dictionary mapping secret keys to the corresponding environment variable key.
            override (bool, optional): If True, override the secret's values in Kubernetes if a secret with the same
                name already exists.
        """
        name_prefix = (
            f"{config.username}-" if config.username else ""
        )  # we need the username as prefix in case diffrent users will create the same provider secret
        self._name = (
            name or f"{name_prefix}{provider}" or f"{name_prefix}{self._PROVIDER}"
        )
        self._name = self._name.replace(
            "_", "-"
        )  # cleanup so the name will match k8 standards.
        self._namespace = kwargs.get("namespace", None) or config.namespace
        self._values = values

        self.provider = provider or self._PROVIDER
        self.path = path
        if path:
            filenames = kwargs.get(
                "filenames", None
            )  # we might get filenames as kwarg if we load the secret from name or form config
            updated_path, filenames = self._split_path_if_needed(
                path=path, filenames=filenames
            )
            self.path = updated_path
            self.filenames = filenames
        self.env_vars = env_vars
        self._override = override

        if not any([values, path, env_vars]):
            if self._values_from_path():
                pass
            elif self._values_from_env(self._DEFAULT_ENV_VARS):
                self.env_vars = self._DEFAULT_ENV_VARS
            else:
                raise ValueError(
                    "Secrets values not provided and could not be extracted from default file "
                    f"({self._DEFAULT_PATH}) or env vars ({self._DEFAULT_ENV_VARS.values()}) locations."
                )

    @property
    def name(self):
        """Name of the secret."""
        return self._name

    @property
    def override(self):
        """Should we override secret's values in Kubernetes if a secret with the same name already exists"""
        return self._override

    @property
    def values(self):
        """Secret values."""
        if self._values:
            return self._values
        if self.path:
            return self._values_from_path(self.path)
        if self.env_vars:
            return self._values_from_env(self.env_vars)
        return {}

    def _values_from_env(self, env_vars: Dict = None):
        env_vars = env_vars or self.env_vars
        if not env_vars:
            return {}
        return {key: os.environ[key] for key in env_vars}

    def _values_from_path(self, path: str = None):
        path = path or self.path or self._DEFAULT_PATH
        if not path:
            return {}

        # Double-check that the path is a directory
        path, filenames = self._split_path_if_needed(path)

        values = read_files_as_secrets_dict(path=path, filenames=filenames)
        if values:
            # Only set if the values were successfully found
            if self._MAP_FILENAMES_TO_ENV_VARS:
                env_vars = []
                for filename, env_var in self._MAP_FILENAMES_TO_ENV_VARS.items():
                    if filename in values:
                        values[env_var] = values[filename].strip()
                        del values[filename]
                        env_vars.append(env_var)
                if env_vars:
                    self.env_vars = env_vars
                    self._values = values
                    return values

            self.path = path
            self.filenames = filenames
        return values

    def _split_path_if_needed(
        self, path: str, filenames: list = None
    ) -> Tuple[str, List[str]]:
        """Split path into path and filesnames if a single file is specified as a full path"""
        updated_path = path
        is_default_path = updated_path == self._DEFAULT_PATH
        updated_filenames = getattr(self, "filenames", None) or filenames
        if not updated_filenames:
            if not is_default_path or not self._DEFAULT_FILENAMES:
                # Reform single-file path a directory and filenames list
                updated_filenames = [os.path.basename(path)]
                updated_path = os.path.dirname(path)
            else:
                updated_filenames = self._DEFAULT_FILENAMES
        return updated_path, updated_filenames

    @classmethod
    def from_config(cls, config: dict):
        override_value = config.get("override", "False").lower()
        bool_override_value = override_value == "true"
        config["override"] = bool_override_value
        if "provider" in config:
            from .provider_secrets.providers import _get_provider_class

            provider_class = _get_provider_class(config["provider"])
            return provider_class.from_config(config)
        return cls(**config)

    @classmethod
    def from_name(cls, name, namespace: str = config.namespace):

        from kubetorch.resources.secrets.kubernetes_secrets_client import (
            KubernetesSecretsClient,
        )

        secrets_client = KubernetesSecretsClient(namespace=namespace)
        secret = secrets_client.load_secret(name=name)
        return secret

    @classmethod
    def builtin_providers(cls, as_str: bool = False) -> List:
        """Return list of all Kubetorch providers (as class objects) supported out of the box.

        Args:
            as_str (bool, optional): Whether to return the providers as a string or as a class.
                (Default: ``False``)
        """
        from .provider_secrets.providers import _str_to_provider_class

        if as_str:
            return list(_str_to_provider_class.keys())
        return list(_str_to_provider_class.values())

    @classmethod
    def from_provider(
        cls, provider: str, name: str = None, path: str = None, override: bool = False
    ):
        """Return kubetorch provider secret object

        Args:
            provider (str): Provider's name
            name (str, Optional): Secret name
            path (str, optional): Path where the secret values are held.
            override (Bool, optional): If True, override the secret's values in Kubernetes if a secret with the same name already exists.
        """
        from .provider_secrets.providers import _get_provider_class

        secret_class = _get_provider_class(provider)
        if not secret_class:
            raise ValueError(
                f"{provider} is not a supported provider: {Secret.builtin_providers(as_str=True)}"
            )
        return secret_class(name=name, provider=provider, path=path, override=override)

    @classmethod
    def from_path(cls, path: str, name: str = None, override: bool = False):
        """Return kubetorch provider secret object

        Args:
            path (str): Local path to the secret values file
            name (str, Optional): Secret name
            override (Bool, optional): If True, override the secret's values in Kubernetes if a secret with the same name already exists.
        """
        from .provider_secrets.providers import _get_provider_class

        secret_class = _get_provider_class(path) or Secret
        if not secret_class._PROVIDER and not name:
            raise ValueError("secret name must be provided.")

        return secret_class(name=name, path=path, override=override)

    @classmethod
    def from_env(cls, env_vars: dict, name: str = None, override: bool = False):
        """Return kubetorch provider secret object

        Args:
            env_vars (dict): Dictionary mapping secret keys to the corresponding
            environment variable key.
            name (str, Optional): Secret name
            override (Bool, optional): If True, override the secret's values in Kubernetes if a secret with the same name already exists.
        """
        from .provider_secrets.providers import _get_provider_class

        secret_class = _get_provider_class(env_vars) or Secret
        return secret_class(name=name, env_vars=env_vars, override=override)
