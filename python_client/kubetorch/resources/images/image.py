from enum import Enum
from typing import Any, Dict, List, Union

# Internal class to represent the image construction process
class ImageSetupStepType(Enum):
    """Enum for valid Image setup step types"""

    CMD_RUN = "cmd_run"
    RSYNC = "rsync"
    PIP_INSTALL = "pip_install"
    SYNC_PACKAGE = "sync_package"
    SET_ENV_VARS = "set_env_vars"


class ImageSetupStep:
    def __init__(
        self,
        step_type: ImageSetupStepType,
        **kwargs: Dict[str, Any],
    ):
        """
        A component of the Kubetorch Image, consisting of the step type (e.g. packages, set_env_vars),
        along with arguments to provide to the function corresponding to the step type.

        Args:
            step_type (ImageSetupStepType): Type of setup step used to provide the Image.
            kwargs (Dict[str, Any]): Please refer to the corresponding functions in ``Image`` to determine
                the correct keyword arguments to provide.
        """
        self.step_type = step_type
        self.kwargs = kwargs


class Image:
    def __init__(
        self,
        name: str = None,
        image_id: str = None,
        python_path: str = None,
        install_cmd: str = None,
    ):
        """
        Kubetorch Image object, specifying cluster setup properties and steps.

        Args:
            name (str, optional): Name to assign the Kubetorch image.
            image_id (str, optional): Machine image to use, if any. (Default: ``None``)
            python_path (str, optional): Absolute path to the Python executable to use for remote server and installs.
                (Default: ``None``)
            install_cmd (str, optional): Custom pip/uv install command to use for package installations.
                If not provided, will be inferred based on python_path and available tools (preferring to use uv).
                Examples: "uv pip install", "python -m pip install", "/path/to/.venv/bin/python -m uv pip install"
                (Default: ``None``)

        Note:
            For convenience, Kubetorch provides ready-to-use base images under ``kt.images``.
            These cover common environments like Python, CUDA, and Ray:

              * ``kt.images.Python310()``, ``kt.images.Python311()``, ``kt.images.Python312()``
              * ``kt.images.Debian()``
              * ``kt.images.Ubuntu()``
              * ``kt.images.Ray()`` (defaults to the latest Ray release)

            You can also use flexible factories:

              * ``kt.images.python("3.12")``
              * ``kt.images.pytorch()`` (defaults to "nvcr.io/nvidia/pytorch:23.12-py3")
              * ``kt.images.ray("2.32.0-py311")``

            These base images can be further customized with methods like
            ``.pip_install()``, ``.set_env_vars()``, ``.sync_package()``, etc.

        Example:

            .. code-block:: python

                import kubetorch as kt

                custom_image = (
                    kt.Image(name="base_image")
                    .pip_install(["numpy", "pandas"])
                    .set_env_vars({"OMP_NUM_THREADS": 1})
                )
                debian_image = (
                    kt.images.Debian()
                    .pip_install(["numpy", "pandas"])
                    .set_env_vars({"OMP_NUM_THREADS": 1})
                )
        """

        self.name = name
        self.image_id = image_id
        self.python_path = python_path
        self.install_cmd = install_cmd

        self.setup_steps = []
        self.docker_secret = None

    @staticmethod
    def _setup_step_config(step: ImageSetupStep):
        """Get ImageSetupStep config"""
        config = {
            "step_type": step.step_type.value,
            "kwargs": step.kwargs,
        }
        return config

    @staticmethod
    def _setup_step_from_config(step: Dict):
        """Convert setup step config (dict) to ImageSetupStep object"""
        step_type = step["step_type"]
        kwargs = step["kwargs"]
        return ImageSetupStep(
            step_type=ImageSetupStepType(step_type),
            **kwargs,
        )

    def from_docker(self, image_id: str):
        """Set up and use an existing Docker image.

        Args:
            image_id (str): Docker image in the following format ``"<registry>/<image>:<tag>"``
        """
        if self.image_id:
            raise ValueError(
                "Setting both a machine image and docker image is not yet supported."
            )
        self.image_id = image_id
        return self

    ########################################################
    # Steps to build the image
    ########################################################

    def pip_install(
        self,
        reqs: List[Union["Package", str]],
        force: bool = False,
    ):
        """Pip install the given packages.

        Args:
            reqs (List[Package or str]): List of packages to pip install on cluster and env.
                Each string is passed directly to the pip/uv command, allowing full control
                over pip arguments. Examples:
                - Simple package: ``"numpy"``
                - Version constraint: ``"pandas>=1.2.0"``
                - With pip flags: ``"--pre torch==2.0.0rc1"``
                - Multiple flags: ``"--index-url https://pypi.org/simple torch"``
            force (bool, optional): Whether to force re-install a package, if it already exists on the compute. (Default: ``False``)

        Example:
            .. code-block:: python

                import kubetorch as kt

                image = (
                    kt.images.Debian()
                    .pip_install([
                        "numpy>=1.20",
                        "pandas",
                        "--pre torchmonarch==0.1.0rc7",  # Install pre-release
                        "--index-url https://test.pypi.org/simple/ mypackage"
                    ])
                )
        """

        self.setup_steps.append(
            ImageSetupStep(
                step_type=ImageSetupStepType.PIP_INSTALL,
                reqs=reqs,
                force=force,
            )
        )
        return self

    def set_env_vars(self, env_vars: Dict):
        """Set environment variables with support for variable expansion.

        Environment variables can reference other variables using shell-style expansion:
        - ``$VAR`` or ``${VAR}`` syntax to reference existing variables
        - Variables are expanded when the container starts
        - Variables are expanded in the order they are defined

        Args:
            env_vars (Dict): Dict of environment variables and values to set.
                Values can include references to other environment variables.

        Example:
            .. code-block:: python

                import kubetorch as kt

                image = (
                    kt.images.Debian()
                    .set_env_vars({
                        "BASE_PATH": "/usr/local",
                        "BIN_PATH": "$BASE_PATH/bin",  # Expands to /usr/local/bin
                        "PATH": "$BIN_PATH:$PATH",      # Prepends to existing PATH
                        "LD_LIBRARY_PATH": "/opt/lib:${LD_LIBRARY_PATH}",  # Appends to existing
                        "CUSTOM": "${HOME}/data",       # Uses HOME from container
                    })
                )

        Note:
            - Variables are expanded using Python's ``os.path.expandvars()``
            - Undefined variables remain as literal strings (e.g., ``$UNDEFINED`` stays as ``$UNDEFINED``)
            - To include a literal ``$``, escape it with backslash: ``\\$``
        """
        # TODO - support .env files
        self.setup_steps.append(
            ImageSetupStep(
                step_type=ImageSetupStepType.SET_ENV_VARS,
                env_vars=env_vars,
            )
        )
        return self

    def sync_package(
        self,
        package: str,
        force: bool = False,
    ):
        """Sync local package over and add to path.

        Args:
            package (Package or str): Package to sync. Either the name of a local editably installed package, or
                the path to the folder to sync over.
            force (bool, optional): Whether to re-sync the package over, if already previously synced over. (Default: ``False``)
        """
        self.setup_steps.append(
            ImageSetupStep(
                step_type=ImageSetupStepType.SYNC_PACKAGE,
                package=package,
                force=force,
            )
        )
        return self

    def run_bash(
        self,
        command: str,
        force: bool = False,
    ):
        """Run bash commands during image setup.

        Executes shell commands during container initialization. Commands run in the
        order they are defined and can be used to install software, configure the
        environment, or start background services.

        Args:
            command (str): Shell command(s) to run on the cluster. Supports:
                - Single commands: ``"apt-get update"``
                - Chained commands: ``"apt-get update && apt-get install -y curl"``
                - Background processes: ``"jupyter notebook --no-browser &"``
            force (bool): Whether to rerun the command on the cluster, if previously run in image setup already. (Default: ``False``)

        Example:
            .. code-block:: python

                import kubetorch as kt

                image = (
                    kt.images.Debian()
                    .run_bash("apt-get update && apt-get install -y vim")
                    .run_bash("pip install jupyter")
                    .run_bash("jupyter notebook --no-browser --port=8888 &")  # Runs in background
                )

        Note:
            - Commands ending with ``&`` run in the background and won't block image setup
            - Background processes continue running after setup completes
            - The setup waits 0.5s to catch immediate failures in background processes
            - Use ``&&`` to chain commands that depend on each other
            - Use ``;`` to run commands sequentially regardless of success/failure
        """
        self.setup_steps.append(
            ImageSetupStep(
                step_type=ImageSetupStepType.CMD_RUN,
                command=command,
                force=force,
            )
        )
        return self

    def rsync(
        self,
        source: str,
        dest: str = None,
        contents: bool = False,
        filter_options: str = None,
        force: bool = False,
    ):
        """Sync files or directories from local machine to the remote container.

        This method efficiently transfers files to remote containers using rsync,
        which only copies changed files and supports compression. Files are first
        uploaded to a jump pod and then distributed to worker pods on startup.

        Args:
            source (str): Path to the local file or directory to sync. Supports:
                - Absolute paths: ``/path/to/file``
                - Relative paths: ``./data/file.txt``
                - Home directory paths: ``~/documents/data.csv``
            dest (str, optional): Target path on the remote container. Supports:
                - Absolute paths: ``/data/config.yaml`` (places file at exact location)
                - Relative paths: ``configs/settings.json`` (relative to working directory)
                - Tilde paths: ``~/results/output.txt`` (relative to working directory, ~ is stripped)
                - None: Uses the basename of source in working directory
            contents (bool, optional): For directories only - whether to copy the contents
                or the directory itself.
                If ``True`` the contents of the source directory are copied to the destination,
                and the source directory itself is not created at the destination.
                If ``False`` the source directory along with its contents are copied to the
                destination, creating an additional directory layer at the destination.
                (Default: ``False``)
            filter_options (str, optional): Additional rsync filter options. These are added
                to (not replacing) the default filters. By default, rsync excludes:

                - Files from ``.gitignore`` (if present)
                - Files from ``.ktignore`` (if present)
                - Common Python artifacts: ``*.pyc``, ``__pycache__``
                - Virtual environments: ``.venv``
                - Git metadata: ``.git``

                Your filter_options are appended after these defaults. Examples:

                - Exclude more patterns: ``"--exclude='*.log' --exclude='temp/'"``
                - Include specific files: ``"--include='important.log' --exclude='*.log'"``
                - Override all defaults: Set ``KT_RSYNC_FILTERS`` environment variable

                (Default: ``None``)
            force (bool, optional): When ``True``, forces rsync to transfer all files
                regardless of modification times by using ``--ignore-times`` flag. This ensures
                all files are copied even if timestamps suggest they haven't changed.
                Useful when timestamp-based change detection is unreliable.
                Note: Files are always synced when deploying with ``.to()``, this flag just
                affects how rsync determines which files need updating.
                (Default: ``False``)

        Returns:
            Image: Returns self for method chaining.

        Examples:
            .. code-block:: python

                import kubetorch as kt

                # Basic file sync
                image = (
                    kt.images.Debian()
                    .rsync("./config.yaml", "app/config.yaml")
                )

                # Sync to absolute path
                image = (
                    kt.images.Python312()
                    .rsync("./model_weights.pth", "/models/weights.pth")
                )

                # No destination specified - uses basename
                image = (
                    kt.images.Ubuntu()
                    .rsync("/local/data/dataset.csv")  # Goes to ./dataset.csv
                )

                # Directory sync - copy directory itself
                image = (
                    kt.images.Debian()
                    .rsync("./src", "app")  # Creates app/src/
                )

                # Directory sync - copy contents only
                image = (
                    kt.images.Debian()
                    .rsync("./src", "app", contents=True)  # Contents go directly into app/
                )

                # Multiple rsync operations with filtering
                image = (
                    kt.images.Python312()
                    .rsync("./data", "/data", filter_options="--exclude='*.tmp'")
                    .rsync("./configs", "~/configs")
                    .rsync("./scripts")
                    .pip_install(["numpy", "pandas"])
                )

                # Force re-sync for development
                image = (
                    kt.images.Debian()
                    .rsync("./rapidly_changing_code", "app", force=True)
                )

        Note:
            - Absolute destination paths (starting with ``/``) place files at exact locations
            - Relative paths and ``~/`` paths are relative to the container's working directory
            - The ``contents`` parameter only affects directory sources, not files
            - Default exclusions include ``.gitignore`` patterns, ``__pycache__``, ``.venv``, and ``.git``
            - User-provided ``filter_options`` are added to (not replacing) the default filters
            - To completely override filters, set the ``KT_RSYNC_FILTERS`` environment variable
            - Use ``force=True`` to bypass timestamp checks and transfer all files
        """

        self.setup_steps.append(
            ImageSetupStep(
                step_type=ImageSetupStepType.RSYNC,
                source=source,
                dest=dest,
                contents=contents,
                filter_options=filter_options,
                force=force,
            )
        )
        return self
