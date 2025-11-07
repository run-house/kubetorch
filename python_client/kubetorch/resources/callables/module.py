import asyncio
import json
import tempfile
import threading
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Union

import websockets

from kubetorch.globals import config, service_url, service_url_async
from kubetorch.logger import get_logger
from kubetorch.resources.callables.utils import get_names_for_reload_fallbacks, locate_working_dir

from kubetorch.resources.compute.utils import (
    delete_cached_service_data,
    delete_configmaps,
    load_configmaps,
    VersionMismatchError,
)
from kubetorch.servers.http.http_client import HTTPClient
from kubetorch.servers.http.utils import (
    clean_and_validate_k8s_name,
    generate_unique_request_id,
    is_running_in_kubernetes,
)
from kubetorch.serving.utils import has_k8s_credentials, KubernetesCredentialsError
from kubetorch.utils import (
    extract_host_port,
    get_kt_install_url,
    iso_timestamp_to_nanoseconds,
    LogVerbosity,
    ServerLogsFormatter,
)

logger = get_logger(__name__)


class Module:
    MODULE_TYPE = None

    def __init__(
        self,
        name: str,
        pointers: tuple,
    ):
        self._compute = None
        self._deployment_timestamp = None
        self._service_config = None
        self._http_client = None
        self._get_if_exists = True
        self._reload_prefixes = None
        self._serialization = "json"  # Default serialization format
        self._async = False
        self._remote_pointers = None
        self._service_name = None

        self.pointers = pointers
        self.name = clean_and_validate_k8s_name(name, allow_full_length=False) if name else None

    @property
    def module_name(self):
        """Name of the function or class."""
        return self.pointers[2]

    @property
    def reload_prefixes(self):
        return self._reload_prefixes or []

    @reload_prefixes.setter
    def reload_prefixes(self, value: Union[str, List[str]]):
        """Set the reload_prefixes property."""
        if isinstance(value, (list)):
            self._reload_prefixes = value
        elif isinstance(value, str):
            self._reload_prefixes = [value]
        else:
            raise ValueError("`reload_prefixes` must be a string or a list.")

    @property
    def namespace(self):
        """Namespace where the service is deployed."""
        if self.compute is not None:
            return self.compute.namespace
        return config.namespace

    @property
    def service_name(self):
        """Name of the knative service, formatted according to k8s regex rules."""
        if self._service_name:
            return self._service_name

        service_name = self.name

        if config.username and not self.reload_prefixes and not service_name.startswith(config.username + "-"):
            service_name = f"{config.username}-{service_name}"

        self._service_name = clean_and_validate_k8s_name(service_name, allow_full_length=True)
        return self._service_name

    @service_name.setter
    def service_name(self, value: str):
        self._service_name = clean_and_validate_k8s_name(value, allow_full_length=True)

    @property
    def compute(self):
        """Compute object corresponding to the module."""
        return self._compute

    @compute.setter
    def compute(self, compute: "Compute"):
        self._compute = compute

    @property
    def deployment_timestamp(self):
        if not self._deployment_timestamp:
            self._deployment_timestamp = self.compute.service_manager.get_deployment_timestamp_annotation(
                self.service_name
            )
        return self._deployment_timestamp

    @deployment_timestamp.setter
    def deployment_timestamp(self, value: str):
        self._deployment_timestamp = value

    @property
    def remote_pointers(self):
        if self._remote_pointers:
            return self._remote_pointers

        source_dir, _ = locate_working_dir(self.pointers[0])
        relative_module_path = Path(self.pointers[0]).expanduser().relative_to(source_dir)
        source_dir_name = Path(source_dir).name
        if self.compute.working_dir is not None:
            container_module_path = str(Path(self.compute.working_dir) / source_dir_name / relative_module_path)
        else:
            # Leave it as relative path
            container_module_path = str(Path(source_dir_name) / relative_module_path)
        self._remote_pointers = (
            container_module_path,
            self.pointers[1],
            self.pointers[2],
        )
        return self._remote_pointers

    @property
    def service_config(self) -> dict:
        """Knative service configuration loaded from Kubernetes API."""
        return self._service_config

    @service_config.setter
    def service_config(self, value: dict):
        self._service_config = value

    @property
    def base_endpoint(self):
        """Endpoint for the module."""
        if is_running_in_kubernetes():
            if not self._compute.endpoint:
                return self._compute._wait_for_endpoint()
            return self._compute.endpoint
        # URL format when using the NGINX proxy
        return f"http://localhost:{self._compute.client_port()}/{self.namespace}/{self.service_name}"

    @property
    def request_headers(self):
        if self.compute.freeze:
            return {}

        if self.deployment_timestamp:
            return {"X-Deployed-As-Of": self.deployment_timestamp}

        return {}

    @property
    def serialization(self):
        """Default serialization format for this module."""
        return self._serialization

    @serialization.setter
    def serialization(self, value: str):
        """Set the default serialization format for this module."""
        if value not in ["json", "pickle"]:
            raise ValueError("Serialization must be 'json' or 'pickle'")
        self._serialization = value

    @property
    def async_(self):
        """Whether to run the function or class methods in async mode."""
        return self._async

    @async_.setter
    def async_(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("`async_` must be a boolean")
        self._async = value

    @classmethod
    def from_name(
        cls,
        name: str,
        namespace: str = None,
        reload_prefixes: Union[str, List[str]] = [],
    ):
        """Reload an existing callable by its service name."""
        from kubernetes import client
        from kubernetes.config import ConfigException, load_incluster_config, load_kube_config

        import kubetorch as kt

        try:
            load_incluster_config()
        except ConfigException:
            load_kube_config()
        objects_api = client.CustomObjectsApi()
        apps_v1_api = client.AppsV1Api()
        core_v1_api = client.CoreV1Api()

        namespace = namespace or config.namespace
        if isinstance(reload_prefixes, str):
            reload_prefixes = [reload_prefixes]
        potential_names = get_names_for_reload_fallbacks(name=name, prefixes=reload_prefixes)

        # Use unified service discovery from BaseServiceManager
        from kubetorch.serving.service_manager import BaseServiceManager

        all_services = BaseServiceManager.discover_services_static(
            namespace=namespace, objects_api=objects_api, apps_v1_api=apps_v1_api
        )

        # Create name-to-service lookup for efficient searching
        service_dict = {svc["name"]: svc for svc in all_services}

        # Try to find the first matching service across all service types
        for candidate in potential_names:

            service_info = service_dict.get(candidate)
            if service_info is None:
                continue

            compute = kt.Compute.from_template(service_info)

            pods = core_v1_api.list_namespaced_pod(
                namespace=namespace,
                label_selector=f"kubetorch.com/service={name}",
            )
            volumes = []

            # TODO: handle case where service is scaled to 0?
            if pods.items:
                # Use runtime Pod spec
                pod = pods.items[0]
                for v in pod.spec.volumes or []:
                    if v.persistent_volume_claim:
                        existing_volume = kt.Volume.from_name(name=v.name)
                        volumes.append(existing_volume)

            module_args = compute.get_env_vars(
                [
                    "KT_FILE_PATH",
                    "KT_MODULE_NAME",
                    "KT_CLS_OR_FN_NAME",
                    "KT_CALLABLE_TYPE",
                    "KT_INIT_ARGS",
                ]
            )
            pointers = (
                module_args["KT_FILE_PATH"],
                module_args["KT_MODULE_NAME"],
                module_args["KT_CLS_OR_FN_NAME"],
            )

            if module_args.get("KT_CALLABLE_TYPE") == "cls":
                init_args = json.loads(module_args.get("KT_INIT_ARGS") or "{}")
                reloaded_module = kt.Cls(name=candidate, pointers=pointers, init_args=init_args)
            elif module_args.get("KT_CALLABLE_TYPE") == "fn":
                reloaded_module = kt.Fn(name=candidate, pointers=pointers)
            else:
                raise ValueError(f"Unknown module type: {module_args.get('KT_CALLABLE_TYPE')}")

            reloaded_module.service_name = candidate
            reloaded_module.compute = compute
            return reloaded_module

        raise ValueError(
            f"Service '{name}' not found in namespace '{namespace}' with reload_prefixes={reload_prefixes}"
        )

    def _client(self, *args, **kwargs):
        """Return the client through which to interact with the remote Module.
        If compute is not yet set, attempt to reload it.
        """
        if self._http_client is not None:
            return self._http_client

        if self.compute is None or self.service_config is None:
            namespace = self.namespace
            # When rebuilding the http client on reload, need to know whether to look for a prefix
            reload_prefixes = self.reload_prefixes
            logger.debug(
                f"Attempting to reload service '{self.service_name}' in namespace '{namespace}' with "
                f"reload_prefixes={reload_prefixes}"
            )
            reloaded_module = Module.from_name(
                name=self.service_name,
                namespace=namespace,
                reload_prefixes=reload_prefixes,
            )

            # Update settable attributes with reloaded module values
            self.compute = self.compute or reloaded_module.compute
            self.service_config = reloaded_module.service_config
            self.pointers = reloaded_module.pointers
            self.name = reloaded_module.name
            self.service_name = reloaded_module.service_name

        self._http_client = HTTPClient(
            base_url=self.endpoint(*args, **kwargs),
            compute=self.compute,
            service_name=self.service_name,
        )

        return self._http_client

    def endpoint(self, method_name: str = None):
        if not hasattr(self, "init_args"):
            return f"{self.base_endpoint}/{self.module_name}"
        else:
            return f"{self.base_endpoint}/{self.module_name}/{method_name}"

    def deploy(self):
        """
        Helper method to deploy modules specified by the @compute decorator. Used by `kt deploy` CLI command.
        Deploys the module to the specified compute.
        """
        if self.compute is None:
            raise ValueError("Compute must be set before deploying the module.")
        return self.to(self.compute, init_args=getattr(self, "init_args", None))

    async def deploy_async(self):
        """
        Async helper method to deploy modules specified by the @compute decorator. Used by `kt deploy` CLI command
        when multiple modules are present. Deploys the module to the specified compute asynchronously.
        """
        if self.compute is None:
            raise ValueError("Compute must be set before deploying the module.")
        return await self.to_async(self.compute, init_args=getattr(self, "init_args", None))

    def to(
        self,
        compute: "Compute",
        init_args: Dict = None,
        stream_logs: Union[bool, None] = None,
        verbosity: Union[LogVerbosity, str] = None,
        get_if_exists: bool = False,
        reload_prefixes: Union[str, List[str]] = [],
        dryrun: bool = False,
    ):
        """
        Send the function or class to the specified compute.

        Args:
            compute (Compute): The compute to send the function or class to.
            init_args (Dict, optional): Initialization arguments, which may be relevant for a class.
            stream_logs (bool, optional): Whether to stream logs during service launch. If None, uses the global
                config value.
            verbosity (Union[verbosity, str], optional): Verbosity of the logs streamed back to the client.
                If not specified, will stream select service logs. Can also be controlled globally via the config
                value `log_verbosity`. Supported values: "debug", "info", "critical".
            get_if_exists (Union[bool, List[str]], optional): Controls how service lookup is performed to determine
                whether to send the service to the compute.

                - If False (default): Do not attempt to reload the service.
                - If True: Attempt to find an existing service using a standard fallback order
                  (e.g., username, git branch, then prod). If found, re-use that existing service.
            reload_prefixes (Union[str, List[str]], optional): A list of prefixes to use when reloading the function
                (e.g., ["qa", "prod", "git-branch-name"]). If not provided, will use the current username,
                git branch, and prod.
            dryrun (bool, optional): Whether to setup and return the object as a dryrun (``True``),
                or to actually launch the compute and service (``False``).
        Returns:
            Module: The module instance.

        Example:

        .. code-block:: python

            import kubetorch as kt

            remote_cls = kt.cls(SlowNumpyArray, name=name).to(
                kt.Compute(cpus=".1"),
                init_args={"size": 10},
                stream_logs=True
            )
        """
        if not has_k8s_credentials():
            raise KubernetesCredentialsError(
                "Kubernetes credentials not found. Please ensure you are running in a Kubernetes cluster or have a valid kubeconfig file."
            )

        if get_if_exists:
            try:
                existing_service = self._get_existing_service(reload_prefixes)
                if existing_service:
                    logger.debug(f"Reusing existing service: {existing_service.service_name}")
                    return existing_service
            except Exception as e:
                logger.info(
                    f"Service {self.service_name} not found in namespace {self.compute.namespace} "
                    f"with reload_prefixes={reload_prefixes}: {str(e)}"
                )

        self.compute = compute
        self.compute.service_name = self.service_name

        if hasattr(self, "init_args"):
            self.init_args = init_args

        # We need the deployment timestamp at the start of the update so we know that artifacts deployed **after**
        # this time are part of the current deployment. We actually set it at the end to ensure that the deployment is
        # successful.
        logger.debug(f"Deploying module: {self.service_name}")
        deployment_timestamp = datetime.now(timezone.utc).isoformat()
        install_url, use_editable = get_kt_install_url(self.compute.freeze)

        if not dryrun and not self.compute.freeze:
            self._rsync_repo_and_image_patches(install_url, use_editable, init_args)

        self._launch_service(
            install_url,
            use_editable,
            init_args,
            deployment_timestamp,
            stream_logs,
            verbosity,
            dryrun,
        )

        return self

    async def to_async(
        self,
        compute: "Compute",
        init_args: Dict = None,
        stream_logs: Union[bool, None] = None,
        verbosity: Union[LogVerbosity, str] = None,
        get_if_exists: bool = False,
        reload_prefixes: Union[str, List[str]] = [],
        dryrun: bool = False,
    ):
        """
        Async version of the `.to` method. Send the function or class to the specified compute asynchronously.

        Args:
            compute (Compute): The compute to send the function or class to.
            init_args (Dict, optional): Initialization arguments, which may be relevant for a class.
            stream_logs (bool, optional): Whether to stream logs during service launch. If None, uses the global
                config value.
            verbosity (Union[verbosity, str], optional): Verbosity of the logs streamed back to the client.
                If not specified, will stream select service logs. Can also be controlled globally via the config
                value `log_verbosity`. Supported values: "debug", "info", "critical".
            get_if_exists (Union[bool, List[str]], optional): Controls how service lookup is performed to determine
                whether to send the service to the compute.

                - If False (default): Do not attempt to reload the service.
                - If True: Attempt to find an existing service using a standard fallback order
                  (e.g., username, git branch, then prod). If found, re-use that existing service.
            reload_prefixes (Union[str, List[str]], optional): A list of prefixes to use when reloading the function
                (e.g., ["qa", "prod", "git-branch-name"]). If not provided, will use the current username,
                git branch, and prod.
            dryrun (bool, optional): Whether to setup and return the object as a dryrun (``True``),
                or to actually launch the compute and service (``False``).
        Returns:
            Module: The module instance.

        Example:

        .. code-block:: python

            import kubetorch as kt

            remote_cls = await kt.cls(SlowNumpyArray, name=name).to_async(
                kt.Compute(cpus=".1"),
                init_args={"size": 10},
                stream_logs=True
            )
        """
        if get_if_exists:
            try:
                existing_service = await self._get_existing_service_async(reload_prefixes)
                if existing_service:
                    logger.debug(f"Reusing existing service: {existing_service.service_name}")
                    return existing_service
            except Exception as e:
                logger.info(
                    f"Service {self.compute.service_name} not found in namespace {self.compute.namespace} "
                    f"with reload_prefixes={reload_prefixes}: {str(e)}"
                )

        self.compute = compute
        self.compute.service_name = self.service_name

        if hasattr(self, "init_args"):
            self.init_args = init_args

        logger.debug(f"Deploying module: {self.service_name}")
        deployment_timestamp = datetime.now(timezone.utc).isoformat()
        install_url, use_editable = get_kt_install_url(self.compute.freeze)

        if not dryrun and not self.compute.freeze:
            await self._rsync_repo_and_image_patches_async(install_url, use_editable, init_args)

        await self._launch_service_async(
            install_url,
            use_editable,
            init_args,
            deployment_timestamp,
            stream_logs,
            verbosity,
            dryrun,
        )

        return self

    def _get_existing_service(self, reload_prefixes):
        try:
            existing_service = Module.from_name(
                self.service_name,
                namespace=self.namespace,
                reload_prefixes=reload_prefixes,
            )
            if existing_service:
                if self.compute:
                    # Replace the compute object, if the user has already constructed it locally
                    existing_service.compute = self.compute
                logger.info(
                    f"Existing service '{self.service_name}' found in namespace '{self.namespace}', not "
                    f"redeploying."
                )
                return existing_service
        except Exception as e:
            raise ValueError(
                f"Failed to reload service {self.service_name} in namespace {self.namespace} "
                f"and reload_prefixes={reload_prefixes}: {str(e)}"
            )

    async def _get_existing_service_async(self, reload_prefixes):
        try:
            existing_service = Module.from_name(
                self.service_name,
                namespace=self.namespace,
                reload_prefixes=reload_prefixes,
            )
            if existing_service:
                if self.compute:
                    # Replace the compute object, if the user has already constructed it locally
                    existing_service.compute = self.compute
                logger.info(
                    f"Existing service '{self.service_name}' found in namespace '{self.namespace}', not "
                    f"redeploying."
                )
                return existing_service
        except Exception as e:
            raise ValueError(
                f"Failed to reload service {self.service_name} in namespace {self.namespace} "
                f"and reload_prefixes={reload_prefixes}: {str(e)}"
            )

    def _get_rsync_dirs_and_dockerfile(self, install_url, use_editable, init_args):
        source_dir, has_kt_dir = locate_working_dir(self.pointers[0])
        rsync_dirs = [str(source_dir)]
        if not has_kt_dir:
            # Use the source file (.py) instead of directory
            source_file = Path(f"{self.pointers[0]}/{self.pointers[1]}.py")
            rsync_dirs = [str(source_file)]
            logger.info(f"Package root not found; syncing file {source_file}")
        else:
            logger.info(f"Package root identified at {source_dir}; syncing directory")

        if install_url.endswith(".whl") or (use_editable and install_url != str(source_dir)):
            rsync_dirs.append(install_url)

        pointer_env_vars = self._get_pointer_env_vars(self.remote_pointers)
        metadata_env_vars = self._get_metadata_env_vars(init_args)
        service_dockerfile = self._get_service_dockerfile({**pointer_env_vars, **metadata_env_vars})
        return rsync_dirs, service_dockerfile

    def _rsync_repo_and_image_patches(self, install_url, use_editable, init_args):
        logger.debug("Rsyncing data to the rsync pod")
        rsync_dirs, service_dockerfile = self._get_rsync_dirs_and_dockerfile(install_url, use_editable, init_args)
        self._construct_and_rsync_files(rsync_dirs, service_dockerfile)
        logger.debug(f"Rsync completed for service {self.service_name}")

    async def _rsync_repo_and_image_patches_async(self, install_url, use_editable, init_args):
        logger.debug("Rsyncing data to the rsync pod")
        rsync_dirs, service_dockerfile = self._get_rsync_dirs_and_dockerfile(install_url, use_editable, init_args)
        await self._construct_and_rsync_files_async(rsync_dirs, service_dockerfile)
        logger.debug(f"Rsync completed for service {self.service_name}")

    def _launch_service(
        self,
        install_url,
        use_editable,
        init_args,
        deployment_timestamp,
        stream_logs,
        verbosity,
        dryrun,
    ):
        # Start log streaming if enabled
        stop_event = threading.Event()
        log_thread = None
        if stream_logs is None:
            stream_logs = config.stream_logs or False

        launch_request_id = "-"
        if stream_logs and not dryrun:
            if verbosity is None:
                verbosity = config.log_verbosity

            # Create a unique request ID for this launch sequence
            launch_request_id = f"launch_{generate_unique_request_id('launch', deployment_timestamp)}"

            # Start log streaming in a separate thread
            log_thread = threading.Thread(
                target=self._stream_launch_logs,
                args=(
                    launch_request_id,
                    stop_event,
                    verbosity,
                    deployment_timestamp,
                ),
            )
            log_thread.daemon = True
            log_thread.start()

        try:
            startup_rsync_command = self._startup_rsync_command(use_editable, install_url, dryrun)

            # Launch the compute in the form of a service with the requested resources
            service_config = self.compute._launch(
                service_name=self.compute.service_name,
                install_url=install_url if not use_editable else None,
                pointer_env_vars=self._get_pointer_env_vars(self.remote_pointers),
                metadata_env_vars=self._get_metadata_env_vars(init_args),
                startup_rsync_command=startup_rsync_command,
                launch_id=launch_request_id,
                dryrun=dryrun,
            )
            self.service_config = service_config

            if not self.compute.freeze and not dryrun:
                self.deployment_timestamp = self.compute.service_manager.update_deployment_timestamp_annotation(
                    service_name=self.service_name,
                    new_timestamp=deployment_timestamp,
                )
            if not dryrun:
                self.compute._check_service_ready()
                # Additional health check to ensure HTTP server is ready
                self._wait_for_http_health()
        finally:
            # Stop log streaming
            if log_thread:
                stop_event.set()

    async def _launch_service_async(
        self,
        install_url,
        use_editable,
        init_args,
        deployment_timestamp,
        stream_logs,
        verbosity,
        dryrun,
    ):
        # Start log streaming if enabled
        stop_event = asyncio.Event()
        log_task = None
        if stream_logs is None:
            stream_logs = config.stream_logs or False

        launch_request_id = "-"
        if stream_logs and not dryrun:
            if verbosity is None:
                verbosity = config.log_verbosity

            # Create a unique request ID for this launch sequence
            launch_request_id = f"launch_{generate_unique_request_id('launch', deployment_timestamp)}"

            # Start log streaming as an async task
            log_task = asyncio.create_task(
                self._stream_launch_logs_async(
                    launch_request_id,
                    stop_event,
                    verbosity,
                    deployment_timestamp,
                )
            )

        try:
            startup_rsync_command = self._startup_rsync_command(use_editable, install_url, dryrun)

            # Launch the compute in the form of a service with the requested resources
            # Use the async version of _launch
            service_config = await self.compute._launch_async(
                service_name=self.compute.service_name,
                install_url=install_url if not use_editable else None,
                pointer_env_vars=self._get_pointer_env_vars(self.remote_pointers),
                metadata_env_vars=self._get_metadata_env_vars(init_args),
                startup_rsync_command=startup_rsync_command,
                launch_id=launch_request_id,
                dryrun=dryrun,
            )
            self.service_config = service_config

            if not self.compute.freeze and not dryrun:
                self.deployment_timestamp = self.compute.service_manager.update_deployment_timestamp_annotation(
                    service_name=self.service_name,
                    new_timestamp=deployment_timestamp,
                )
            if not dryrun:
                await self.compute._check_service_ready_async()
                await self._wait_for_http_health_async()
        finally:
            # Stop log streaming
            if log_task:
                stop_event.set()
                try:
                    await asyncio.wait_for(log_task, timeout=2.0)
                except asyncio.TimeoutError:
                    log_task.cancel()
                    try:
                        await log_task
                    except asyncio.CancelledError:
                        pass

    def _get_service_dockerfile(self, metadata_env_vars):
        image_instructions = self.compute._image_setup_and_instructions()

        if image_instructions:
            image_instructions += "\n"
        for key, val in metadata_env_vars.items():
            if isinstance(val, Dict):
                val = json.dumps(val)
            image_instructions += f"ENV {key} {val}\n"

        logger.debug(f"Generated Dockerfile for service {self.service_name}:\n{image_instructions}")
        return image_instructions

    def _construct_and_rsync_files(self, rsync_dirs, service_dockerfile):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_file = Path(tmpdir) / ".kt" / "image.dockerfile"
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            temp_file.write_text(service_dockerfile)

            source_dir = str(Path(tmpdir) / ".kt")
            rsync_dirs.append(source_dir)

            logger.debug(f"Rsyncing directories: {rsync_dirs}")
            if is_running_in_kubernetes():
                self.compute.rsync_in_cluster(rsync_dirs)
            else:
                self.compute.rsync(rsync_dirs)

    async def _construct_and_rsync_files_async(self, rsync_dirs, service_dockerfile):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_file = Path(tmpdir) / ".kt" / "image.dockerfile"
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            temp_file.write_text(service_dockerfile)

            source_dir = str(Path(tmpdir) / ".kt")
            rsync_dirs.append(source_dir)

            logger.debug(f"Rsyncing directories: {rsync_dirs}")
            if is_running_in_kubernetes():
                await self.compute.rsync_in_cluster_async(rsync_dirs)
            else:
                await self.compute.rsync_async(rsync_dirs)

    def _startup_rsync_command(self, use_editable, install_url, dryrun):
        if dryrun:
            return None

        if use_editable or (install_url and install_url.endswith(".whl")):
            # rsync from the rsync pod's file system directly
            startup_cmd = self.compute._rsync_svc_url()
            cmd = f"rsync -av {startup_cmd} ."
            return cmd

        return None

    def teardown(self):
        """Delete the service and all associated resources."""
        logger.info(f"Deleting service: {self.service_name}")

        # Use the compute's service manager - it already knows the correct type!
        teardown_success = self.compute.service_manager.teardown_service(
            service_name=self.service_name,
        )

        if not teardown_success:
            logger.error(f"Failed to teardown service {self.service_name}")
            return

        configmaps = load_configmaps(
            core_api=self.compute.core_api,
            service_name=self.service_name,
            namespace=self.compute.namespace,
        )
        if configmaps:
            logger.info(f"Deleting {len(configmaps)} configmap{'' if len(configmaps) == 1 else 's'}")
            delete_configmaps(
                core_api=self.compute.core_api,
                configmaps=configmaps,
                namespace=self.compute.namespace,
            )

        logger.info("Deleting service data from cache in rsync pod")
        delete_cached_service_data(
            core_api=self.compute.core_api,
            service_name=self.service_name,
            namespace=self.compute.namespace,
        )

    def _get_pointer_env_vars(self, remote_pointers):
        (container_file_path, module_name, cls_or_fn_name) = remote_pointers
        return {
            "KT_FILE_PATH": container_file_path,
            "KT_MODULE_NAME": module_name,
            "KT_CLS_OR_FN_NAME": cls_or_fn_name,
        }

    def _get_metadata_env_vars(
        self,
        init_args: Dict,
    ) -> Dict:
        # TODO: add other callable metadata in addition to pointers (`is_generator`, `is_async`, etc.)
        import json

        distributed_config = self.compute.distributed_config
        return {
            "KT_INIT_ARGS": init_args,
            "KT_CALLABLE_TYPE": self.MODULE_TYPE,
            "KT_DISTRIBUTED_CONFIG": json.dumps(distributed_config) if distributed_config else None,
        }

    def _stream_launch_logs(
        self,
        request_id: str,
        stop_event: threading.Event,
        verbosity: LogVerbosity,
        deployment_timestamp: str,
    ):
        """Stream logs and events during service launch sequence."""
        try:
            # Only use "kubetorch" container to exclude queue-proxy (e.g. Knative sidecars) container logs which
            # are spammy with tons of healthcheck calls
            pod_query = f'{{k8s_container_name="kubetorch"}} | json | request_id="{request_id}"'
            event_query = f'{{service_name="unknown_service"}} | json | k8s_object_name=~"{self.service_name}.*" | k8s_namespace_name="{self.namespace}"'

            encoded_pod_query = urllib.parse.quote_plus(pod_query)
            encoded_event_query = urllib.parse.quote_plus(event_query)
            logger.debug(f"Streaming launch logs and events for service {self.service_name}")

            def start_log_threads(host, port):
                def run_pod_logs():
                    self._run_log_stream(
                        request_id,
                        stop_event,
                        host,
                        port,
                        encoded_pod_query,
                        verbosity,
                        deployment_timestamp,
                        dedup=True,
                    )

                def run_event_logs():
                    self._run_log_stream(
                        request_id,
                        stop_event,
                        host,
                        port,
                        encoded_event_query,
                        verbosity,
                        deployment_timestamp,
                    )

                pod_thread = threading.Thread(target=run_pod_logs, daemon=True)
                event_thread = threading.Thread(target=run_event_logs, daemon=True)

                pod_thread.start()
                event_thread.start()

                # Don't block indefinitely on joins - use short timeouts
                pod_thread.join(timeout=1.0)
                event_thread.join(timeout=1.0)

            base_url = service_url()
            host, port = extract_host_port(base_url)
            logger.debug(f"Streaming launch logs with url={base_url} host={host} and local port {port}")
            start_log_threads(host, port)

        except Exception as e:
            logger.error(f"Failed to stream launch logs: {e}")
            raise e

    async def _stream_launch_logs_async(
        self,
        request_id: str,
        stop_event: asyncio.Event,
        verbosity: LogVerbosity,
        deployment_timestamp: str,
    ):
        """Async version of _stream_launch_logs. Stream logs and events during service launch sequence."""
        try:
            # Only use "kubetorch" container to exclude queue-proxy (e.g. Knative sidecars) container logs which
            # are spammy with tons of healthcheck calls
            pod_query = f'{{k8s_container_name="kubetorch"}} | json | request_id="{request_id}"'
            event_query = f'{{service_name="unknown_service"}} | json | k8s_object_name=~"{self.service_name}.*" | k8s_namespace_name="{self.namespace}"'

            encoded_pod_query = urllib.parse.quote_plus(pod_query)
            encoded_event_query = urllib.parse.quote_plus(event_query)
            logger.debug(f"Streaming launch logs and events for service {self.service_name}")

            base_url = await service_url_async()
            host, port = extract_host_port(base_url)
            logger.debug(f"Streaming launch logs with url={base_url} host={host} and local port {port}")

            # Create async tasks for both log streams
            pod_task = asyncio.create_task(
                self._stream_logs_websocket(
                    request_id,
                    stop_event,
                    host=host,
                    port=port,
                    query=encoded_pod_query,
                    log_verbosity=verbosity,
                    deployment_timestamp=deployment_timestamp,
                    dedup=True,
                )
            )

            event_task = asyncio.create_task(
                self._stream_logs_websocket(
                    request_id,
                    stop_event,
                    host=host,
                    port=port,
                    query=encoded_event_query,
                    log_verbosity=verbosity,
                    deployment_timestamp=deployment_timestamp,
                )
            )

            # Wait for both tasks to complete or be cancelled
            try:
                await asyncio.gather(pod_task, event_task, return_exceptions=True)
            except Exception as e:
                logger.error(f"Error in async log streaming: {e}")

        except Exception as e:
            logger.error(f"Failed to stream launch logs: {e}")
            raise e

    def _run_log_stream(
        self,
        request_id: str,
        stop_event: threading.Event,
        host: str,
        port: int,
        query: str,
        log_verbosity: LogVerbosity,
        deployment_timestamp: str,
        dedup: bool = False,
    ):
        """Helper to run log streaming in an event loop"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                self._stream_logs_websocket(
                    request_id,
                    stop_event,
                    host=host,
                    port=port,
                    query=query,
                    log_verbosity=log_verbosity,
                    deployment_timestamp=deployment_timestamp,
                    dedup=dedup,
                )
            )
        finally:
            loop.close()

    async def _run_log_stream_async(
        self,
        request_id: str,
        stop_event: asyncio.Event,
        host: str,
        port: int,
        query: str,
        log_verbosity: LogVerbosity,
        deployment_timestamp: str,
        dedup: bool = False,
    ):
        """Async helper to run log streaming directly in the current event loop"""
        await self._stream_logs_websocket(
            request_id,
            stop_event,
            host=host,
            port=port,
            query=query,
            log_verbosity=log_verbosity,
            deployment_timestamp=deployment_timestamp,
            dedup=dedup,
        )

    async def _stream_logs_websocket(
        self,
        request_id: str,
        stop_event: Union[threading.Event, asyncio.Event],
        host: str,
        port: int,
        query: str,
        log_verbosity: LogVerbosity,
        deployment_timestamp: str,
        dedup: bool = False,
    ):
        """Stream logs and events using Loki's websocket tail endpoint"""
        try:
            uri = f"ws://{host}:{port}/loki/api/v1/tail?query={query}"

            # Track the last timestamp we've seen to avoid duplicates
            last_timestamp = None

            # Track when we should stop
            stop_time = None

            # Track most recent deployment timestamp to filter out old logs / events
            start_timestamp = iso_timestamp_to_nanoseconds(deployment_timestamp)

            shown_event_messages = set()

            # Track seen log messages for deduplication
            seen_log_messages = set() if dedup else None

            # For formatting the server setup logs
            formatters = {}
            base_formatter = ServerLogsFormatter()
            websocket = None
            try:
                # Add timeout to prevent hanging connections
                websocket = await websockets.connect(
                    uri,
                    close_timeout=10,  # Max time to wait for close handshake
                    ping_interval=20,  # Send ping every 20 seconds
                    ping_timeout=10,  # Wait 10 seconds for pong
                )
                while True:
                    # If stop event is set, start counting down
                    # Handle both threading.Event and asyncio.Event
                    is_stop_set = stop_event.is_set() if hasattr(stop_event, "is_set") else stop_event.is_set()
                    if is_stop_set and stop_time is None:
                        stop_time = time.time() + 2  # 2 second grace period

                    # If we're past the grace period, exit
                    if stop_time is not None and time.time() > stop_time:
                        break

                    try:
                        # Use shorter timeout during grace period
                        timeout = 0.1 if stop_time is not None else 1.0
                        message = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                        data = json.loads(message)

                        if data.get("streams"):
                            for stream in data["streams"]:
                                labels = stream.get("stream", {})
                                is_event = "k8s_event_count" in list(labels.keys())
                                for value in stream["values"]:
                                    ts_ns = int(value[0])
                                    if start_timestamp is not None and ts_ns < start_timestamp:
                                        continue
                                    log_line = value[1]
                                    if is_event:
                                        event_type = labels.get("detected_level", "")
                                        if log_verbosity == LogVerbosity.CRITICAL and event_type == "Normal":
                                            # skip Normal events in MINIMAL
                                            continue

                                        try:
                                            msg = log_line
                                            reason = (labels.get("k8s_event_reason", ""),)

                                            # Note: relevant starting in release 0.1.19 (using OTel instead of Alloy)
                                            if isinstance(reason, tuple):
                                                reason = reason[0]

                                            event_type = labels.get("detected_level", "")

                                            if reason == "Unhealthy" and (
                                                "HTTP probe failed with statuscode: 503" in msg
                                                or "Startup probe failed" in msg
                                            ):
                                                # HTTP probe failures are expected during setup
                                                continue

                                            ignore_patterns = (
                                                "queue-proxy",
                                                "resolving reference: address not set for kind = service",
                                                "failed to get private k8s service endpoints:",
                                            )
                                            # Ignore queue-proxy events and gateway setup events
                                            if any(pattern in msg.lower() for pattern in ignore_patterns):
                                                continue

                                            if msg in shown_event_messages:
                                                # Only show unique event messages
                                                continue

                                            shown_event_messages.add(msg)

                                        except Exception:
                                            # If parsing fails, just print the event as is
                                            pass

                                        if event_type == "Normal":
                                            if log_verbosity in [
                                                LogVerbosity.INFO,
                                                LogVerbosity.DEBUG,
                                            ]:
                                                print(f'[EVENT] reason={reason} "{msg}"')
                                        else:
                                            print(f'[EVENT] type={event_type} reason={reason} "{msg}"')
                                        continue

                                    # Skip if we've already seen this timestamp
                                    if last_timestamp is not None and value[0] <= last_timestamp:
                                        continue
                                    last_timestamp = value[0]
                                    if log_verbosity in [
                                        LogVerbosity.DEBUG,
                                        LogVerbosity.INFO,
                                    ]:
                                        try:
                                            log_dict = json.loads(log_line)
                                        except json.JSONDecodeError:
                                            # setup steps pre server start are not JSON formatted
                                            log_dict = None

                                        if log_dict is not None:
                                            # at this stage we are post setup
                                            pod_name = log_dict.get("pod", request_id)
                                            levelname = log_dict.get("levelname", "INFO")
                                            ts = log_dict.get("asctime")
                                            message = log_dict.get("message", "")

                                            if (
                                                log_verbosity == LogVerbosity.CRITICAL
                                                and levelname not in ["ERROR", "CRITICAL"]
                                            ) or (log_verbosity == LogVerbosity.INFO and levelname == "DEBUG"):
                                                continue

                                            log_line = f"{levelname} | {ts} | {message}"
                                            if pod_name not in formatters:
                                                formatters[pod_name] = ServerLogsFormatter(pod_name)
                                            formatter = formatters[pod_name]
                                        else:
                                            # streaming pre server setup logs, before we have the pod name
                                            formatter = base_formatter

                                        newline = "" if log_dict is None else None
                                        formatted_line = f"{formatter.start_color}{f'({self.service_name}) '}{log_line}{formatter.reset_color}"

                                        # Check for duplicates if dedup is enabled
                                        if seen_log_messages is not None:
                                            if message in seen_log_messages:
                                                continue
                                            seen_log_messages.add(message)

                                        print(formatted_line, end=newline)
                    except asyncio.TimeoutError:
                        # Timeout is expected, just continue the loop
                        continue
                    except websockets.exceptions.ConnectionClosed as e:
                        logger.debug(f"WebSocket connection closed: {str(e)}")
                        break
            finally:
                if websocket:
                    try:
                        # Use wait_for to prevent hanging on close
                        await asyncio.wait_for(websocket.close(), timeout=1.0)
                    except (asyncio.TimeoutError, Exception):
                        pass
        except Exception as e:
            logger.error(f"Error in websocket stream: {e}")
            raise e
        finally:
            # Ensure websocket is closed even if we didn't enter the try block
            if websocket:
                try:
                    # Use wait_for to prevent hanging on close
                    await asyncio.wait_for(websocket.close(), timeout=1.0)
                except (asyncio.TimeoutError, Exception):
                    pass

    def _wait_for_http_health(self, timeout=60, retry_interval=0.1, backoff=2, max_interval=10):
        """Wait for the HTTP server to be ready by checking the /health endpoint.

        Args:
            timeout: Maximum time to wait in seconds
            retry_interval: Time between health check attempts in seconds
        """
        import time

        logger.info(f"Waiting for HTTP server to be ready for service {self.service_name}")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                client = self._client()
                response = client.get(
                    endpoint=f"{self.base_endpoint}/health",
                    headers=self.request_headers,
                )
                if response.status_code == 200:
                    logger.info(f"HTTP server is ready for service {self.service_name}")
                    return
                else:
                    logger.debug(f"Health check returned status {response.status_code}, retrying...")

            except VersionMismatchError as e:
                raise e

            except Exception as e:
                logger.debug(f"Health check failed: {e}, retrying...")

            time.sleep(retry_interval)
            retry_interval *= backoff  # Exponential backoff
            # Cap the retry interval to a maximum value
            retry_interval = min(retry_interval, max_interval)

        # If we get here, we've timed out
        logger.warning(f"HTTP health check timed out after {timeout}s for service {self.service_name}")

    async def _wait_for_http_health_async(self, timeout=60, retry_interval=0.1, backoff=2, max_interval=10):
        """Async version of _wait_for_http_health. Wait for the HTTP server to be ready by checking the /health endpoint.

        Args:
            timeout: Maximum time to wait in seconds
            retry_interval: Time between health check attempts in seconds
        """
        import asyncio

        logger.info(f"Waiting for HTTP server to be ready for service {self.service_name}")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                client = self._client()
                response = client.get(
                    endpoint=f"{self.base_endpoint}/health",
                    headers=self.request_headers,
                )
                if response.status_code == 200:
                    logger.info(f"HTTP server is ready for service {self.service_name}")
                    return
                else:
                    logger.debug(f"Health check returned status {response.status_code}, retrying...")
            except Exception as e:
                logger.debug(f"Health check failed: {e}, retrying...")

            await asyncio.sleep(retry_interval)
            retry_interval *= backoff  # Exponential backoff
            # Cap the retry interval to a maximum value
            retry_interval = min(retry_interval, max_interval)

        # If we get here, we've timed out
        logger.warning(f"HTTP health check timed out after {timeout}s for service {self.service_name}")

    def __getstate__(self):
        """Remove local stateful values before pickle serialization."""
        state = self.__dict__.copy()
        # Remove local stateful values that shouldn't be serialized
        state["_http_client"] = None
        state["_service_config"] = None
        state["_remote_pointers"] = None
        # Pointers need to be converted to not be absolute paths if we're passing
        # the service elsewhere, e.g. into another service
        state["pointers"] = self.remote_pointers
        return state

    def __setstate__(self, state):
        """Restore state after pickle deserialization."""
        self.__dict__.update(state)
        # Reset local stateful values to None to ensure clean initialization
        self._http_client = None
        self._service_config = None
        self._remote_pointers = None

    def __del__(self):
        if hasattr(self, "_http_client") and self._http_client is not None:
            try:
                self._http_client.close()
            except Exception as e:
                logger.debug(f"Error closing HTTPClient in Module deletion: {e}")
            finally:
                self._http_client = None
