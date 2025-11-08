import base64
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

from kubernetes import client, config
from kubernetes.client import V1SecretList
from kubernetes.client.rest import ApiException

import kubetorch
from kubetorch import globals
from kubetorch.constants import DEFAULT_KUBECONFIG_PATH
from kubetorch.logger import get_logger
from kubetorch.resources.secrets import Secret
from kubetorch.resources.secrets.utils import get_k8s_identity_name
from kubetorch.servers.http.utils import is_running_in_kubernetes
from kubetorch.serving.constants import KT_USER_IDENTIFIER_LABEL, KT_USERNAME_LABEL

logger = get_logger(__name__)


class KubernetesSecretsClient:
    def __init__(self, namespace: str = None, kubeconfig_path: str = None):
        self._kubeconfig_path = kubeconfig_path
        self.namespace = namespace or globals.config.namespace

        # Load config
        self.kt_config = globals.config

        # Derive User ID from config context
        self.user_id = get_k8s_identity_name()

        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config(config_file=self.kubeconfig_path)
        self.api_client = client.CoreV1Api()

    @property
    def kubeconfig_path(self):
        if not self._kubeconfig_path:
            self._kubeconfig_path = os.getenv("KUBECONFIG") or DEFAULT_KUBECONFIG_PATH
        return str(Path(self._kubeconfig_path).expanduser())

    # -------------------------------------
    # SECRETS APIS
    # -------------------------------------
    def load_secret(self, name: str) -> Optional[Secret]:
        secret_dict = self._read_secret(name)
        if not secret_dict:
            return None
        secret = Secret.from_config(secret_dict)
        return secret

    def delete_secret(self, name: str, console: "Console" = None) -> bool:
        """Delete secret with provided name for current user."""
        name = name if self._read_secret(name=name) else self._format_secret_name(name)
        return self._delete_secret(name=name, console=console)

    def delete_all_secrets(self, username: Optional[str] = None) -> bool:
        """Delete all secrets for current user."""
        return self._delete_all_secrets_for_user(username=username)

    def convert_to_secret_objects(self, secrets: List[Union[str, Secret]]) -> List[Secret]:
        """
        Converts a list of strings and Secrets into Secret objects without uploading.
        """
        from kubetorch.resources.secrets.secret_factory import secret as secret_factory

        secret_objects = []
        for secret_or_string in secrets:
            # Create a provider secret if only the name is provided
            secret = (
                secret_factory(provider=secret_or_string) if isinstance(secret_or_string, str) else secret_or_string
            )
            secret_objects.append(secret)

        return secret_objects

    def upload_secrets_list(self, secrets: List[Union[str, Secret]]) -> List[Secret]:
        """Uploads secrets to Kubernetes Secrets to be used in knative yaml."""
        if is_running_in_kubernetes():
            return []

        # Convert to Secret objects first
        secret_objects = self.convert_to_secret_objects(secrets)

        synced_secrets = []
        for secret in secret_objects:
            success = self.create_or_update_secret(secret=secret)
            if success:
                synced_secrets.append(secret)

        return synced_secrets

    def extract_envs_and_volumes_from_secrets(
        self,
        secrets: List[Secret] = None,
    ) -> Tuple[list, list]:
        if not secrets:
            return [], []

        env_vars = []
        volumes = []
        for secret in secrets:
            secret_name = self._format_secret_name(secret.name)
            if secret.env_vars:
                env_vars.append(
                    {
                        "env_vars": secret.env_vars,
                        "secret_name": secret_name,
                    }
                )
            if secret.path:
                path = secret.path.replace("~", "/root")
                if not secret.filenames:
                    # Reformat path to only include directory
                    path = os.path.dirname(path)
                volumes.append(
                    {
                        "name": f"secrets-{secret.name}",
                        "secret_name": secret_name,
                        "path": path,
                    }
                )
        return env_vars, volumes

    def _format_secret_name(self, name: str) -> str:
        """Appends user ID to name to ensure uniqueness."""
        user = self.kt_config.username or "global"
        if user in name:
            return name
        return f"kt.secret.{user}.{name}"

    def _read_secret(self, name: str) -> Optional[dict]:
        secret_name = name
        try:
            secret = self.api_client.read_namespaced_secret(name, self.namespace)
        except ApiException as e:
            if e.status == 404:  # secret does not exist, try to load with formatted k8 name
                secret_name = self._format_secret_name(name)
                try:
                    secret = self.api_client.read_namespaced_secret(secret_name, self.namespace)
                except ApiException as e:
                    if e.status == 404:
                        logger.info(
                            f"Secret {secret_name} not found in namespace {self.namespace}",
                        )
                        return None
                    else:
                        logger.error(
                            f"Failed to read secret {name} from Kubernetes: {str(e)}",
                        )
                        return None
            else:
                logger.error(
                    f"Failed to read secret {name} from Kubernetes: {str(e)}",
                )
                return None
        except Exception as e:
            logger.error(
                f"Unexpected error occurred while reading secret {secret_name} from Kubernetes: {str(e)}",
            )
            return None

        override = (
            secret.metadata.annotations.get("kubetorch.com/override", "False") if secret.metadata.annotations else None
        )
        path = (
            secret.metadata.annotations.get("kubetorch.com/secret-path", None) if secret.metadata.annotations else None
        )
        filenames = (
            secret.metadata.annotations.get("kubetorch.com/secret-filenames", None)
            if secret.metadata.annotations
            else None
        )
        filenames = json.loads(filenames) if filenames else filenames

        secret_config = {
            "name": secret_name,
            "namespace": self.namespace,
            "override": override,
        }

        if path:
            secret_config["path"] = path
            secret_config["filenames"] = filenames
            return secret_config

        decoded_values = {k: base64.b64decode(v).decode("utf-8") for k, v in secret.data.items()}
        secret_config["values"] = decoded_values

        mount_type = secret.metadata.labels.get("kubetorch.com/mount-type", None)
        if mount_type == "env":
            secret_config["env_vars"] = list(decoded_values.keys())

        return secret_config

    def _build_secret_body(self, secret: Secret):
        secret_name = self._format_secret_name(secret.name)
        provider = secret.provider
        mount_type = "volume" if secret.path else "env"
        encoded_data = {k: base64.b64encode(v.encode()).decode() for k, v in secret.values.items()}
        labels = {
            "kubetorch.com/mount-type": mount_type,
            "kubetorch.com/secret-name": secret.name,
        }
        if self.user_id:
            labels[KT_USER_IDENTIFIER_LABEL] = self.user_id
        if self.kt_config.username:
            labels[KT_USERNAME_LABEL] = self.kt_config.username
        if provider:
            labels["kubetorch.com/provider"] = provider
        annotations = {"kubetorch.com/override": str(secret.override)}
        if secret.path:
            annotations["kubetorch.com/secret-path"] = secret.path
            annotations["kubetorch.com/secret-filenames"] = json.dumps(secret.filenames)

        metadata = client.V1ObjectMeta(
            name=secret_name,
            namespace=self.namespace,
            labels=labels,
            annotations=annotations,
        )
        secret_body = client.V1Secret(metadata=metadata, data=encoded_data, type="Opaque")  # default secret type

        return secret_body

    def create_secret(self, secret: Secret, console: "Console" = None):
        secret_name = self._format_secret_name(secret.name)
        secret_body = self._build_secret_body(secret=secret)

        try:
            self.api_client.create_namespaced_secret(self.namespace, secret_body)
            if console:
                console.print("[bold green]✔ Secret created successfully[/bold green]")
                console.print(f"  Name: [cyan]{secret.name}[/cyan]")
                console.print(f"  Namespace: [cyan]{self.namespace}[/cyan]")
            else:
                logger.info(f"Created new Kubernetes secret {secret.name}")
            return True

        except ApiException as e:
            if console:
                if e.status == 409:
                    msg = f"[yellow]Secret '{secret.name}' already exists in namespace {self.namespace}, skipping creation[/yellow]"
                else:
                    msg = f"[red]Failed to create Kubernetes secret {secret_name}: {str(e)}[/red]"
                console.print(msg)
                return False
            else:
                raise e

        except Exception as e:
            if console:
                msg = f"Unexpected error occurred while creating new Kubernetes secret {secret.name}: {str(e)}"
                console.print(f"[red]{msg}[/red]")
                return False
            else:
                raise e

    def _get_existing_secret(self, secret: Secret):
        try:
            return self.api_client.read_namespaced_secret(secret.name, self.namespace)
        except ApiException as e:
            if e.status == 404:  # try to load secret with kubernetes formatted name
                formatted_name = self._format_secret_name(secret.name)
                try:
                    return self.api_client.read_namespaced_secret(formatted_name, self.namespace)
                except ApiException:
                    return None

    def update_secret(self, secret: Secret, console: "Console" = None):
        existing_secret = self._get_existing_secret(secret)
        if existing_secret is None:
            if console:
                console.print(f"[red]Failed to update secret {secret.name}: secret does not exist[/red]")
                return False
            else:
                raise kubetorch.SecretNotFound(secret_name=secret.name, namespace=self.namespace)

        if not secret.override:
            decoded_values = {k: base64.b64decode(v).decode("utf-8") for k, v in existing_secret.data.items()}
            if not decoded_values == secret.values:
                msg = f"Secret {secret.name} exists with different values and `secret.override` not set to True."
                if console:
                    console.print(f"[yellow]{msg}[/yellow]")
                    return False
                else:
                    raise ValueError(msg)
            else:
                msg = f"Secret {secret.name} already exists with the same values."
                console.print(f"[bold green]{msg}[/bold green]") if console else logger.info(msg)
                return True

        secret_name = self._format_secret_name(secret.name)
        secret_body = self._build_secret_body(secret=secret)

        try:
            self.api_client.replace_namespaced_secret(secret_name, self.namespace, secret_body)
            if console:
                console.print("[bold green]✔ Secret updated successfully[/bold green]")
                console.print(f"  Name: [cyan]{secret.name}[/cyan]")
                console.print(f"  Namespace: [cyan]{self.namespace}[/cyan]")
            else:
                logger.info(f"Updated existing Kubernetes secret {secret_name}")
            return True

        except Exception as e:
            if console:
                console.print(f"[red]Failed to update secret {secret.name}: {str(e)}[/red]")
                return False
            raise e

    def create_or_update_secret(self, secret: Secret, console: "Console" = None):
        try:
            return self.update_secret(secret, console)
        except kubetorch.SecretNotFound:
            # if secret not found, try to create the secret.
            return self.create_secret(secret, console)

    def _delete_secret(self, name: str, console: "Console" = None):
        name = self._format_secret_name(name)
        try:
            self.api_client.delete_namespaced_secret(name, self.namespace)
            if console:
                console.print(f"✓ Deleted secret [blue]{name}[/blue]")
            else:
                logger.info(f"Deleted Kubernetes secret: {name}")
            return True

        except ApiException as e:
            msg = f"Failed to delete Kubernetes secret: {name}: {str(e)}"
            console.print(msg) if console else logger.error(msg)
            return False

        except Exception as e:
            msg = f"Unexpected error occurred while deleting Kubernetes secret {name}: {str(e)}"
            console.print(msg) if console else logger.error(msg)
            return False

    def _delete_all_secrets_for_user(self, username: Optional[str] = None):
        username = username or self.kt_config.username
        label_selector = f"{KT_USERNAME_LABEL}={username}"
        try:
            secrets: V1SecretList = self.api_client.list_namespaced_secret(
                namespace=self.namespace, label_selector=label_selector
            )
            deleted_all = True

            for secret in secrets.items:
                secret_name = secret.metadata.name
                try:
                    self.api_client.delete_namespaced_secret(secret_name, self.namespace)
                    logger.info(
                        f"Deleted Kubernetes secret {secret_name} for user {username}",
                    )

                except ApiException as e:
                    logger.warning(
                        f"Failed to delete specific Kubernetes secret {secret_name} for user {username}: {str(e)}",
                    )
                    deleted_all = False

            return deleted_all

        except ApiException as e:
            logger.error(
                f"Failed to list Kubernetes secrets: {str(e)}",
            )
            return False

        except Exception as e:
            logger.error(
                f"Unexpected error occurred while listing Kubernetes secrets: {str(e)}",
            )
