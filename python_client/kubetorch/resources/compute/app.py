import os
import re
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Dict

from kubetorch.logger import get_logger
from kubetorch.resources.callables.module import Module
from kubetorch.resources.compute.compute import Compute
from kubetorch.resources.compute.utils import ServiceTimeoutError
from kubetorch.serving.utils import is_running_in_kubernetes
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

    def _get_service_dockerfile(self):
        image_instructions = super()._get_service_dockerfile()

        remote_script = os.path.join(self.remote_pointers[0], self.remote_pointers[1])
        local_script = r"\b" + re.escape(self.remote_pointers[1]) + r"\b"
        remote_cmd = re.sub(local_script, remote_script, self.cli_command)

        image_instructions += f"CMD {remote_cmd}\n"
        return image_instructions

    def _wait_for_http_health(self, timeout=120, retry_interval=0.5, backoff=1.5, max_interval=2):
        """Wait for the HTTP server to be ready. For apps, only check /health (server up).

        Apps have a different lifecycle - the app process starts after image setup,
        and we track its status via /app/status in _wait_for_app_exit.
        """
        logger.info(f"Polling {self.service_name} service health endpoint")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                client = self._client()
                response = client.get(
                    endpoint=f"{self.base_endpoint}/health",
                    headers=self.request_headers,
                    timeout=5,
                )
                if response.status_code == 200:
                    logger.info(f"Service {self.service_name} server is up")
                    return
                else:
                    logger.debug(f"Health check returned status {response.status_code}, retrying...")
            except Exception as e:
                if "502" not in str(e) and "503" not in str(e):
                    logger.debug(f"Health check failed: {e}, retrying...")

            time.sleep(retry_interval)
            retry_interval = min(retry_interval * backoff, max_interval)

        raise ServiceTimeoutError(f"Service {self.service_name} server not ready after {timeout}s")

    def _launch_service(
        self,
        install_url,
        use_editable,
        deployment_timestamp,
        stream_logs,
    ):
        if self._run_async:
            thread = threading.Thread(
                target=super()._launch_service,
                args=(
                    install_url,
                    use_editable,
                    {},
                    deployment_timestamp,
                    stream_logs,
                    False,
                ),
            )
            thread.start()

            # Wait for pods to be ready before exiting out
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
                dryrun=False,
            )

            # After service is ready, continue streaming logs until the app process exits
            if stream_logs:
                # Use current time if no deployment_timestamp (frozen apps)
                log_start_timestamp = deployment_timestamp or datetime.now(timezone.utc).isoformat()
                self._wait_for_app_exit(log_start_timestamp)

    def _wait_for_app_exit(self, deployment_timestamp: str, poll_interval: float = 1.0):
        """Poll app status and continue streaming logs until app exits.

        Args:
            deployment_timestamp (str): Timestamp for log filtering
            poll_interval (float, optional): Seconds between status polls
        """
        import asyncio
        import urllib.parse

        from kubetorch.globals import service_url
        from kubetorch.utils import extract_host_port

        logger.debug("Waiting for app process to exit, streaming logs in the meantime...")

        stop_event = threading.Event()
        log_thread = None

        try:
            # Start a log streaming thread similar to _stream_launch_logs
            base_url = service_url()
            host, port = extract_host_port(base_url)

            # Build query for app logs
            pod_query = f'{{service=~"{self.service_name}.*", namespace="{self.namespace}"}}'
            encoded_query = urllib.parse.quote_plus(pod_query)

            def stream_app_logs():
                """Stream logs from the app until stop_event is set."""
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(
                        self._stream_logs_websocket(
                            request_id="app_run",
                            stop_event=stop_event,
                            host=host,
                            port=port,
                            query=encoded_query,
                            deployment_timestamp=deployment_timestamp,
                            namespace=self.namespace,
                            dedup=True,
                        )
                    )
                finally:
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        task.cancel()
                    if pending:
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    loop.close()

            log_thread = threading.Thread(target=stream_app_logs, daemon=True)
            log_thread.start()

            # Poll /app/status until the app exits
            client = self._client()

            while True:
                try:
                    response = client.get(
                        endpoint=f"{self.base_endpoint}/app/status",
                        headers=self.request_headers,
                        timeout=5,
                    )
                    if response.status_code == 200:
                        status = response.json()
                        if status.get("running") is False:
                            exit_code = status.get("exit_code", 0)
                            if exit_code != 0:
                                logger.warning(f"App exited with code {exit_code}")
                            else:
                                logger.info("App exited successfully")
                            break
                        elif status.get("running") is None:
                            # Not an app deployment (shouldn't happen)
                            logger.debug("No app process found, stopping wait")
                            break
                    else:
                        logger.debug(f"App status check returned {response.status_code}")
                except Exception as e:
                    logger.debug(f"Error checking app status: {e}")

                time.sleep(poll_interval)

            # Give logs a grace period to catch up
            time.sleep(self.logging_config.grace_period)

        finally:
            # Stop log streaming
            stop_event.set()
            if log_thread:
                log_thread.join(timeout=2.0)

    def _print_kt_cmds(self):
        logger.info(f"To see logs, run: kt logs {self.service_name}.")
        logger.info(f"To teardown service, run: kt teardown {self.service_name}")


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
