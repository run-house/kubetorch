import os
import re
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Dict

from kubetorch.globals import config
from kubetorch.logger import get_logger

from kubetorch.resources.callables.module import Module
from kubetorch.resources.compute.compute import Compute
from kubetorch.resources.compute.utils import ServiceTimeoutError
from kubetorch.servers.http.utils import is_running_in_kubernetes
from kubetorch.utils import get_kt_install_url

logger = get_logger(__name__)


class App(Module):
    MODULE_TYPE = "app"

    def __init__(
        self,
        compute: Compute,
        cli_command: str,
        pointers: tuple,
        name: str = None,
        run_async: bool = False,
    ):
        """
        Initialize an App object for remote execution.

        .. note::

            To create an App, please use the factory method :func:`app` in conjunction with the `kt run` CLI command.

        Args:
            compute (Compute): Compute
            cli_command (str): CLI command to run on the compute.
            pointers (tuple): A tuple containing references needed to locate the app file, of the format
                (current working directory, path of file relative to cwd, None)
            name (str, optional): Name to assign the app. If not provided, will be based on the name of the file in
                which the app was defined.
            run_async (bool, optional): Whether to run the app async. (Default: ``False``)
        """
        super().__init__(name=name, pointers=pointers)
        self.cli_command = cli_command
        self.pointers = pointers
        self.name = name or self.module_name
        self._compute = compute
        self._run_async = run_async
        self._remote_pointers = None

        self._http_client = None

    @property
    def module_name(self):
        return os.path.splitext(self.pointers[1])[0]

    def from_name(self):
        raise ValueError("Reloading app is not supported.")

    def setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self.handle_termination_signal)
        signal.signal(signal.SIGTERM, self.handle_termination_signal)

    def handle_termination_signal(self, signum, frame):
        red = "\u001b[31m"
        reset = "\u001b[0m"

        logger.info(f"{red}Received {signal.Signals(signum).name}. Exiting parent process.{reset}")
        self._print_kt_cmds()
        sys.exit(0)

    def deploy(self):
        """
        Deploy the app to the compute specified by the app arguments.
        """
        self.compute.service_name = self.service_name

        install_url, use_editable = get_kt_install_url(self.compute.freeze)
        if not self.compute.freeze:
            deployment_timestamp = datetime.now(timezone.utc).isoformat()
            self._rsync_repo_and_image_patches(install_url, use_editable, init_args={})
        else:
            deployment_timestamp = None

        self.setup_signal_handlers()

        stream_logs = not self._run_async
        self._launch_service(install_url, use_editable, deployment_timestamp, stream_logs)

    def _get_service_dockerfile(self, metadata_env_vars):
        image_instructions = super()._get_service_dockerfile(metadata_env_vars)

        remote_script = os.path.join(self.remote_pointers[0], self.remote_pointers[1])
        local_script = r"\b" + re.escape(self.remote_pointers[1]) + r"\b"
        remote_cmd = re.sub(local_script, remote_script, self.cli_command)

        image_instructions += f"CMD {remote_cmd}\n"
        return image_instructions

    def _launch_service(
        self,
        install_url,
        use_editable,
        deployment_timestamp,
        stream_logs,
    ):
        trigger_reload = self.compute.is_up()
        if self._run_async:
            thread = threading.Thread(
                target=super()._launch_service,
                args=(
                    install_url,
                    use_editable,
                    {},
                    deployment_timestamp,
                    stream_logs,
                    config.log_verbosity,
                    False,
                ),
            )
            thread.start()

            if trigger_reload:
                self._update_service(stream_logs, deployment_timestamp)
                time.sleep(1)
            else:
                # wait for pods to be ready before exiting out
                start_time = time.time()
                while not self.compute.is_up() and time.time() - start_time < 60:
                    time.sleep(5)

                if not self.compute.is_up():
                    raise ServiceTimeoutError(f"Service {self.service_name} is not up after 60 seconds.")
        else:
            super()._launch_service(
                install_url,
                use_editable,
                init_args={},
                deployment_timestamp=deployment_timestamp,
                stream_logs=stream_logs,
                verbosity=config.log_verbosity,
                dryrun=False,
            )

            if trigger_reload:
                self._update_service(stream_logs, deployment_timestamp)

    def _update_service(self, stream_logs, deployment_timestamp):
        client = self._client()

        if self._run_async:
            thread = threading.Thread(
                target=client.call_method,
                args=(
                    self.endpoint(),
                    stream_logs,
                ),
                kwargs={"headers": {"X-Deployed-As-Of": deployment_timestamp}},
            )
            thread.start()
            time.sleep(1)
            sys.exit()
        else:
            client.call_method(
                self.endpoint(),
                stream_logs=stream_logs,
                headers={"X-Deployed-As-Of": deployment_timestamp},
            )

    def _print_kt_cmds(self):
        logger.info(f"To see logs, run: kt logs {self.service_name}.")
        logger.info(f"To teardown service, run: kt teardown {self.service_name}")

    def endpoint(self):
        return f"{self.base_endpoint}/_reload_image"


def app(
    name: str = None,
    port: int = None,
    health_check: str = None,
    **kwargs: Dict,
):
    """
    Builds and deploys an instance of :class:`App`.

    Args:
        name (str, optional): Name to give the remote app. If not provided, will be based off the name of the file in
            which the app was defined.
        port (int, optional): Server port to expose, if the app starts an HTTP server.
        health_check (str, optional): Health check endpoint, if running a server, to check when server is up and ready.
        **kwargs: Compute kwargs, to define the compute on which to run the app on.

    Examples:

    Define the ``kt.app`` object and compute in your Python file:

    .. code-block:: python

        import kubetorch as kt

        # Define the app at the top of the Python file to deploy
        # train.py
        kt.app(name="my-app", image=kt.Image("docker-latest"), cpus="0.01")

        if __name__ == "__main__":
            ...

    Deploy and run the app remotely using the ``kt run`` CLI command:

    .. code-block:: bash

        kt run python train.py --epochs 5
        kt run fastapi run my_app.py --name fastapi-app
    """
    if not os.getenv("KT_RUN") == "1" or is_running_in_kubernetes():
        return None

    if name and os.getenv("KT_RUN_NAME") and not (name == os.getenv("KT_RUN_NAME")):
        raise ValueError(
            f"Name mismatch between kt.App definition ({name}) and kt run command ({os.getenv('KT_RUN_NAME')})."
        )
    name = name or os.getenv("KT_RUN_NAME")
    cli_command = os.getenv("KT_RUN_CMD")  # set in kt run
    run_async = os.getenv("KT_RUN_ASYNC") == 1

    env_vars = kwargs.get("env_vars", {})
    if port:
        env_vars["KT_APP_PORT"] = port
    if health_check:
        env_vars["KT_APP_HEALTHCHECK"] = health_check
    kwargs["env_vars"] = env_vars
    compute = Compute(**kwargs)

    main_file = os.getenv("KT_RUN_FILE") or os.path.abspath(sys.modules["__main__"].__file__)
    relative_path = os.path.relpath(main_file, os.getcwd())
    pointers = [os.getcwd(), relative_path, None]
    relative_cli_command = re.sub(main_file, relative_path, cli_command)

    kt_app = App(
        compute=compute,
        cli_command=relative_cli_command,
        pointers=pointers,
        name=name,
        run_async=run_async,
    )
    return kt_app
