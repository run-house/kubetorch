import copy
import json
import os
from typing import Any, Dict, Optional, Union

from runhouse.globals import configs, rns_client
from runhouse.resources.blobs import file
from runhouse.resources.blobs.file import File
from runhouse.resources.envs.env import Env
from runhouse.resources.hardware.cluster import Cluster
from runhouse.resources.hardware.utils import _get_cluster_from
from runhouse.resources.secrets.secret import Secret
from runhouse.resources.secrets.utils import _check_file_for_mismatches


class ProviderSecret(Secret):
    _PROVIDER = None
    _DEFAULT_CREDENTIALS_PATH = None
    _DEFAULT_ENV_VARS = {}

    def __init__(
        self,
        name: Optional[str] = None,
        provider: Optional[str] = None,
        values: Dict = None,
        path: str = None,
        env_vars: Dict = None,
        dryrun: bool = False,
        **kwargs,
    ):
        """
        Provider Secret class.

        .. note::
            To create a ProviderSecret, please use the factory method :func:`provider_secret`.
        """
        super().__init__(name=name, values=values, dryrun=dryrun)
        self.provider = provider or self._PROVIDER
        self.path = path
        self.env_vars = env_vars

        if not any([values, path, env_vars]):
            if self._from_path(self._DEFAULT_CREDENTIALS_PATH):
                self.path = self._DEFAULT_CREDENTIALS_PATH
            elif self._from_env(self._DEFAULT_ENV_VARS):
                self.env_vars = self._DEFAULT_ENV_VARS
            else:
                raise ValueError(
                    "Secrets values not provided and could not be extracted from default file "
                    f"({self._DEFAULT_CREDENTIALS_PATH}) or env vars ({self._DEFAULT_ENV_VARS.values()}) locations."
                )

    @property
    def values(self):
        if self._values:
            return self._values
        elif self.path:
            return self._from_path(self.path)
        elif self.env_vars:
            return self._from_env(self.env_vars)
        return {}

    @property
    def config_for_rns(self):
        config = super().config_for_rns
        config.update({"provider": self.provider})
        if self.path:
            config.update({"path": self.path})
        if self.env_vars:
            config.update({"env_vars": self.env_vars})
        return config

    @staticmethod
    def from_config(config: dict, dryrun: bool = False):
        """Create a ProviderSecret object from a config dictionary."""
        return ProviderSecret(**config, dryrun=dryrun)

    def save(self, values: bool = True, headers: str = rns_client.request_headers):
        if not self.name:
            self.name = self.provider
        super().save(values=values, headers=headers)

    def delete(self, headers: str = rns_client.request_headers, contents: bool = False):
        """Delete the secret config from Vault/local. Optionally also delete contents of secret file or env vars."""
        if self.path and contents:
            if isinstance(self.path, File):
                if self.path.exists_in_system():
                    self.path.rm()
            else:
                if os.path.exists(os.path.expanduser(self.path)):
                    os.remove(os.path.expanduser(self.path))
        elif self.env_vars and contents:
            for (_, env_var) in self.env_vars.keys():
                if env_var in os.environ:
                    del os.environ[env_var]
        super().delete(headers=headers)

    def write(
        self,
        file: bool = False,
        env: bool = False,
        path: Union[str, File] = None,
        env_vars: Dict = None,
        overwrite: bool = False,
    ):
        if not self.values:
            raise ValueError("Could not determine values to write down.")
        if (file or path) and (env or env_vars):
            raise ValueError("Can only save to one of file or env at a given time.")
        if not any([file, env, path, env_vars]):
            file = True  # default write to file

        if file or path:
            path = path or self.path or self._DEFAULT_CREDENTIALS_PATH
            return self._write_to_file(path, values=self.values, overwrite=overwrite)
        elif env or env_vars:
            env_vars = env_vars or self.env_vars or self._DEFAULT_ENV_VARS
            return self._write_to_env(env_vars, values=self.values, overwrite=overwrite)

    def to(
        self,
        system: Union[str, Cluster],
        path: Union[str, File] = None,
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
        path = path or self.path

        if system.on_this_cluster():
            if not env and not path == self.path:
                if name and not self.name == name:
                    self.rename(name)
                return self
            self.write(path=path, env=env)
            new_secret = copy.deepcopy(self)
            new_secret._values = None
            new_secret.path = path
            new_secret.name = name or self.name
            return new_secret

        new_secret = copy.deepcopy(self)
        new_secret.name = name or self.name or self.provider
        if values and not new_secret._values:
            new_secret._values = self.values

        key = system.put_resource(new_secret)
        if path:
            new_secret.path = self._file_to(key, system, path, self.values)
        else:
            new_secret.path = file(path=self.path, system=system)

        if env or self.env_vars:
            env_vars = {self.env_vars[key]: self.values[key] for key in self.values}
            system.call(env, "_write_env_vars", env_vars)
        return new_secret

    def _file_to(
        self,
        key: str,
        system: Union[str, Cluster],
        path: Union[str, File] = None,
        values: Any = None,
    ):
        if self.path:
            if isinstance(path, File):
                path = path.path
            remote_file = file(path=self.path).to(system, path=path)
        else:
            system.call(key, "_write_to_file", path=path, values=values)
            remote_file = file(path=path, system=system)
        return remote_file

    def _write_to_file(self, path: Union[str, File], values: Any, overwrite: bool):
        new_secret = copy.deepcopy(self)
        if not _check_file_for_mismatches(
            path, self._from_path(path), values, overwrite
        ):
            if isinstance(path, File):
                path.write(data=values, mode="w")
            else:
                full_path = os.path.expanduser(path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w") as f:
                    json.dump(values, f, indent=4)
                self._add_to_rh_config(full_path)

        new_secret._values = None
        new_secret.path = path
        return new_secret

    def _write_to_env(self, env_vars: Dict, values: Any, overwrite: bool):
        existing_keys = dict(os.environ).keys()
        for key in env_vars.keys():
            if env_vars[key] not in existing_keys or overwrite:
                os.environ[env_vars[key]] = values[key]

        new_secret = copy.deepcopy(self)
        new_secret._values = None
        new_secret.env_vars = env_vars
        return new_secret

    def _from_env(self, env_vars: Dict = None):
        env_vars = env_vars or self.env_vars
        if not env_vars:
            return {}

        values = {}
        keys = self.env_vars.keys() if self.env_vars else {}
        env_vars = env_vars or self.env_vars or {key: key for key in keys}

        if not keys:
            return {}

        for key in keys:
            values[key] = os.environ[env_vars[key]]
        return values

    def _from_path(self, path: Union[str, File] = None):
        path = path or self.path
        if not path:
            return ""

        if isinstance(path, File):
            contents = path.fetch(mode="r")
            try:
                return json.loads(contents)
            except json.decoder.JSONDecodeError:
                return contents
        else:
            path = os.path.expanduser(path)
            if os.path.exists(path):
                with open(path) as f:
                    try:
                        contents = json.load(f)
                    except json.decoder.JSONDecodeError:
                        contents = f.read()
                    return contents
        return {}

    def _add_to_rh_config(self, path):
        if not self.name:
            self.name = self.provider
        configs.set_nested(key="secrets", value={self.name: path})
