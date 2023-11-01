import copy
import json
import logging
import os
from pathlib import Path

from typing import Dict, List, Optional, Union

import requests

from runhouse.globals import configs, rns_client
from runhouse.resources.blobs.file import File, file
from runhouse.resources.envs.env import Env
from runhouse.resources.hardware import _get_cluster_from, Cluster
from runhouse.resources.resource import Resource
from runhouse.resources.secrets.utils import _check_file_for_mismatches, load_config
from runhouse.rns.utils.api import load_resp_content, read_resp_data
from runhouse.rns.utils.names import _generate_default_name


logger = logging.getLogger(__name__)


class Secret(Resource):
    RESOURCE_TYPE = "secret"

    USER_ENDPOINT = "user/secret"
    GROUP_ENDPOINT = "group/secret"

    DEFAULT_DIR = "~/.rh/secrets"

    def __init__(
        self,
        name: Optional[str],
        values: Dict = {},
        path: str = None,
        env_vars: Dict = None,
        format: str = "json",  # TODO: support other types
        dryrun: bool = False,
        **kwargs,
    ):
        """
        Runhouse Secret object.

        .. note::
            To create a Secret, please use one of the factory methods.
        """
        super().__init__(name=name, dryrun=dryrun)
        self._values = values
        self.path = path
        self.env_vars = env_vars
        self.format = format

    @classmethod
    def builtin_providers(cls, as_str: bool = False) -> list:
        """Return list of all Runhouse providers (as class objects) supported out of the box."""
        from runhouse.resources.secrets.provider_secrets.providers import (
            _str_to_provider_class,
        )

        if as_str:
            return list(_str_to_provider_class.keys())
        return list(_str_to_provider_class.values())

    @classmethod
    def save_secrets(cls, secrets: List[str or "Secret"]):
        from runhouse.resources.secrets.provider_secrets.providers import (
            _str_to_provider_class,
        )
        from runhouse.resources.secrets.secret_factory import provider_secret

        for secret in secrets:
            if isinstance(secret, str):
                if secret in _str_to_provider_class.keys():
                    secret = provider_secret(provider=secret)
                else:
                    secret = cls.from_name(secret)
            secret.save()

    @classmethod
    def delete_from_vault(
        cls, secrets: List[str or "Secret"], headers: str = rns_client.request_headers
    ):
        from runhouse.resources.secrets.provider_secrets.providers import (
            _str_to_provider_class,
        )
        from runhouse.resources.secrets.secret_factory import provider_secret

        for secret in secrets:
            if isinstance(secret, str):
                if secret in _str_to_provider_class.keys():
                    secret = provider_secret(provider=secret)
                else:
                    secret = cls.from_name(secret)
            secret.delete(file=False, headers=headers)

    @staticmethod
    def from_config(config: dict, dryrun: bool = False):
        """Create a Secret object from a config dictionary."""
        resource_subtype = config.get("resource_subtype")
        if "provider" in config:
            from runhouse.resources.secrets.provider_secrets.providers import (
                _get_provider_class,
            )

            provider_class = _get_provider_class(config["provider"])
            return provider_class(**config, dryrun=dryrun)
        if resource_subtype == "ProviderSecret":
            from .provider_secrets.provider_secret import ProviderSecret

            return ProviderSecret(**config, dryrun=dryrun)

        return Secret(**config, dryrun=dryrun)

    @classmethod
    def from_name(cls, name, dryrun=False):
        """Load existing Secret via its name."""
        config = load_config(name, cls.USER_ENDPOINT)
        config["name"] = name
        return cls.from_config(config=config, dryrun=dryrun)

    @property
    def config_for_rns(self):
        config = super().config_for_rns
        config.update(
            {
                "values": self._values,
                "path": self.path,
                "env_vars": self.env_vars,
            }
        )
        return config

    @property
    def values(self):
        """
        Extract secret key-value pairs from the Secret object.
        The order of operations for retrieving the values:

        - Values if they were provided upon object instantiation
        - Extracted from the Secret path, if exists locally
        - Extracted from environment variables
        """
        if self._values:
            return self._values
        if self.path:
            values = self._from_path(self.path)
            if values:
                return values
        try:
            return self._from_env()
        except KeyError:
            return {}

    def _add_to_rh_config(self):
        if self.path:
            path_config = {self.name: self.path}
            configs.set_nested(key="secrets", value=path_config)

    def _from_path(self, path: Optional[str] = None):
        path = path or self.path or f"{self.DEFAULT_DIR}/{self.name}.{self.format}"
        if isinstance(path, File):
            if path.exists_in_system():
                try:
                    values = json.loads(path.fetch(mode="r", deserialize=False))
                except json.decoder.JSONDecodeError as e:
                    logger.error(
                        f"Error loading config from {path.path} on {path.system.name}: {e}"
                    )
                    return {}
                return values
        elif path and os.path.exists(os.path.expanduser(path)):
            with open(os.path.expanduser(path), "r") as f:
                try:
                    values = json.load(f)
                except json.decoder.JSONDecodeError as e:
                    logger.error(f"Error loading config from {path}: {e}")
                    return {}
            return values
        return {}

    def _from_env(self, keys: List = None, env_vars: Dict = None):
        values = {}
        keys = keys or (self.env_vars.keys() if self.env_vars else {})
        env_vars = env_vars or self.env_vars or {key: key for key in keys}

        if not keys:
            return {}

        for key in keys:
            values[key] = os.environ[env_vars[key]]
        return values

    # TODO: refactor this code to reuse rns_client save_config code instead of rewriting
    def save(self, values: bool = True, headers: str = rns_client.request_headers):
        """Save the secret config, into Vault if the user is logged in,
        or to local if not or if the resource is a local resource.

        Args:
            values (bool, optional): Whether to save the secret values into the config.
                By default, will save secret values only if the Secret was explicitly constructed
                with secret values passed in. If set to True, will extract secret values
                (from path, env, etc) and save them in the config. If set to False, will
                not save any secret values, even if constructed with secret values passed.
        """
        config = self.config_for_rns
        config["name"] = self.rns_address
        if values and not config["values"]:
            config["values"] = self.values
        elif values is False:
            config["values"] = {}

        if self.rns_address.startswith("/"):
            logger.info(f"Saving config for {self.name} to Vault")
            payload = rns_client.resource_request_payload(config)
            resp = requests.put(
                f"{rns_client.api_server_url}/{self.USER_ENDPOINT}/{self.name}",
                data=json.dumps(payload),
                headers=headers,
            )
            if resp.status_code != 200:
                raise Exception(
                    f"Failed to upload secrets in Vault: {load_resp_content(resp)}"
                )
        else:
            config_path = os.path.expanduser(f"~/.rh/secrets/{self.name}.json")
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)
            logger.info(f"Saving config for {self.rns_address} to: {config_path}")

        self._add_to_rh_config()
        return self

    def write(
        self,
        path: Union[str, Path] = None,
        keys: List[str] = None,
        overwrite: bool = False,
    ):
        """Write the secret values to local filepath.

        Args:
            path (Path or str, optional): Path to write down the secret to. If not provided, defaults
                to the secret path variable (if exists), or to a default location in the Runhouse directory.
            keys (List[str], optional): List of keys corresponding to the secret values to write down.
                If left empty, all secret values will be written down.

        Returns:
            Secret object consisting of the given keys at the path.

        Example:
            >>> secret.write()  # writes down secret values to secret.path

            >>> # writes down api_key key-value pair to "new/secrets/file"
            >>> secret.write(path="new/secrets/file", keys="api_key")
        """
        path = path or self.path
        if not path:
            raise Exception(
                f"Secret {self.name} was not constructed with a path. "
                "Please pass in a path to this function to save/write down"
                "the secret locally."
            )

        values = {key: self.values[key] for key in keys} if keys else self.values
        new_secret = copy.deepcopy(self)
        new_secret._values = values
        new_secret.path = path

        if not isinstance(path, File):
            path = os.path.expanduser(path)

        if _check_file_for_mismatches(path, self._from_path(path), values, overwrite):
            return self

        if isinstance(path, File):
            data = json.dumps(values)
            path.write(data, serialize=False, mode="w")
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(values, f)
            new_secret._add_to_rh_config()

        return new_secret

    def set_env_vars(
        self,
        keys: List[str] = None,
        env_vars: Dict = None,
        overwrite: bool = False,
    ):
        """
        Set the environment variables in the python env.

        Args:
            keys (List[str], optional): The keys corresponding to the secret values to write down.
            env_vars (Dict, optional): The mapping of secret key to the corresponding environment
                variable name.
            overwrite (bool, optional): Whether to overwrite existing env vars with the same key in the
                path or Python environment. Note that this is irrecoverable. (Default: False)

        Returns:
            Secret with the given keys and path, if provided.

        Example:
            >>> secret.set_env_vars(path="secret.env")
            >>> secret.set_env_vars(keys="api_key", env_vars={"api_key": "MY_API_KEY"})
        """
        values = self.values
        keys = keys or values.keys()
        values = {key: values[key] for key in keys}
        env_vars = env_vars or self.env_vars or {key: key for key in keys}

        existing_keys = dict(os.environ).keys()
        for key in env_vars.keys():
            if env_vars[key] not in existing_keys or overwrite:
                os.environ[env_vars[key]] = values[key]

    def delete_file(
        self,
        path: Union[str, Path, File] = None,
    ):
        """Delete the secret file.

        Args:
            path (str, optional): Path to delete the secret file from. If none is provided,
                deletes the path corresponding to the secret class.

        Example:
            >>> secret.delete_file()
        """
        path = path or self.path

        if isinstance(path, File):
            path.rm()
        elif path and os.path.exists(os.path.expanduser(path)):
            os.remove(os.path.expanduser(path))

    def unset_env_vars(
        self,
        keys: List[str] = None,
        env_vars: Dict = None,
    ):
        """
        Delete the env vars to the path if provided, and from the Python os environment if
        ``python_env`` is set to True.

        Args:
            keys (List[str], optional): The keys corresponding to the secret values to remove. Must be a subset
                of secret values keys. If none is provided, will delete the entire set of values keys corresponding
                to the secret.
            env_vars (Dict, optional): The mapping of secret key to the corresponding environment
                variable name.

        Returns:
            Resulting secret with the given keys removed from it, or None if all keys are removed.

        Example:
            >>> secret.delete_env_vars(path="secret.env")
            >>> secret.delete_env_vars(keys="api_key", env_vars={"api_key": "MY_API_KEY"})
        """
        values = self.values
        del_keys = keys or values.keys()
        values = {
            key: values[key] for key in del_keys
        }  # could be used to check not deleting conflicting stuff
        env_vars = env_vars or self.env_vars or {key: key for key in del_keys}

        if keys:
            new_secret = copy.copy(self)

        existing_keys = dict(os.environ).keys()
        for key in env_vars.keys():
            if key not in existing_keys:
                logger.warning(
                    "Key {key} already does not exist in os.environ. Skipping deleting."
                )
            elif os.environ[key] != values[key]:
                logger.warning(
                    f"Secret value corresponding to {key} env var does not match the value in os.environ. "
                    "Skipping deleting."
                )
            else:
                del os.environ[key]
                if new_secret.env_vars and key in new_secret.env_vars:
                    del new_secret.env_vars[key]
                if new_secret._values and key in new_secret._values:
                    del new_secret._values[key]

        if keys:
            return new_secret
        return None

    def _delete_local_config(self):
        config_path = os.path.expanduser(f"~/.rh/secrets/{self.name}.json")
        if os.path.exists(config_path):
            os.remove(config_path)

    def _delete_vault_config(self, headers: str = rns_client.request_headers):
        resp = requests.delete(
            f"{rns_client.api_server_url}/{self.USER_ENDPOINT}/{self.name}",
            headers=headers,
        )
        if resp.status_code != 200:
            logger.error(
                f"Failed to delete secret {self.name} from Vault: {load_resp_content(resp)}"
            )

    def delete(self, file: bool = False, headers: str = rns_client.request_headers):
        """Delete the secret config from Vault/local. Optionally also delete secret file.

        Args:
            file (bool): Whether to also delete the file containing secret values. (Default: False)
        """
        if self.rns_address.startswith("/"):
            self._delete_vault_config(headers)
        else:
            self._delete_local_config()
        if file:
            self.delete_file()
        configs.delete_provider(self.name)

    # Q: is the way we send .to(cluster) secure?
    def to(
        self,
        system: Union[str, Cluster],
        path: Union[str, Path] = None,
        env: Union[str, Env] = None,
        values: bool = True,
        name: Optional[str] = None,
    ):
        """Return a copy of the secret on a system.

        Args:
            system (str or Cluster): Cluster to send the secret to
            path (str or Path, optional): Path on cluster to write down the secret values to.
                If not provided, secret values are not written down.
            env (str or Env, optional): Env to send the secret to. This will save down the secrets
                as env vars in the env.
            values (bool, optional): Whether to save down the values in the resource config. (Default: True)
            name (str, ooptional): Name to assign the resource on the cluster.

        Example:
            >>> secret.to(my_cluster, path=secret.path)
        """
        system = _get_cluster_from(system)
        if system.on_this_cluster():
            if name and not self.name == name:
                self.rename(name)
            if path:
                self.write(path=path)
                new_secret = copy.deepcopy(self)
                new_secret.path = path
                return new_secret
            return self

        new_secret = copy.deepcopy(self)
        new_secret.name = name or self.name or _generate_default_name(prefix="secret")

        # TODO: should we write it down by default if the local secret has it written down?
        new_secret.path = path
        if values and not new_secret._values:
            new_secret._values = new_secret.values

        key = system.put_resource(new_secret)

        if path:
            remote_file = self._file_to(key, system, path)
            new_secret.path = remote_file
        if env:
            env_vars = {v: self.values[self.env_vars[k]] for k, v in self.env_vars}
            system.call(env, "_set_env_vars", env_vars)

        return new_secret

    def _file_to(
        self,
        key: str,
        system: Union[str, Cluster],
        path: Union[str, Path] = None,
    ):
        if self._values:
            system.call(key, "write", path=path)
            remote_file = file(path=path, system=system)
        else:
            remote_file = file(path=self.path).to(system, path=path)
        return remote_file

    def is_local(self):
        """Whether the secret config is stored locally (as opposed to Vault)."""
        path = os.path.expanduser(f"~/.rh/secrets/{self.name}.json")
        if os.path.exists(os.path.expanduser(path)):
            return True
        return False

    def in_vault(self, headers=rns_client.request_headers):
        """Whether the secret is stored in Vault"""
        resp = requests.get(
            f"{rns_client.api_server_url}/{self.USER_ENDPOINT}/{self.name}",
            headers=headers,
        )
        if resp.status_code != 200:
            return False
        if read_resp_data(resp)[self.name]:
            return True
        return False

    def is_present(self):
        pass

    def share(self):
        pass
