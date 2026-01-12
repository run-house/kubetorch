import warnings
from pathlib import Path
from typing import Any, Dict, List, Union


class Image:
    """Kubetorch Image object, specifying cluster setup properties and steps."""

    SUPPORTED_DOCKERFILE_INSTRUCTIONS = {"FROM", "RUN", "ENV", "COPY", "CMD", "ENTRYPOINT"}

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

                # Load from existing Dockerfile
                dockerfile_image = kt.Image.from_dockerfile("./Dockerfile")
        """

        self.name = name
        self.image_id = image_id
        self.python_path = python_path
        self.install_cmd = install_cmd

        # List of dockerfile contents, excluding FROM and KT_PYTHON_PATH
        self._dockerfile_contents: List[str] = []

        # List of copy operations to rsync to datastore at deploy time
        self.copy_operations: List[Dict[str, Any]] = []

        # CMD and ENTRYPOINT from Dockerfile (for kt apply)
        self._cmd: List[str] = []
        self._entrypoint: List[str] = []

    @property
    def contents(self) -> str:
        """Build and return complete dockerfile content as a string."""
        lines = []
        if self.image_id:
            lines.append(f"FROM {self.image_id}")
        if self.python_path:
            lines.append(f"ENV KT_PYTHON_PATH {self.python_path}")
        lines.extend(self._dockerfile_contents)
        return "\n".join(lines) + "\n" if lines else ""

    def from_docker(self, image_id: str):
        """Set up and use an existing Docker image.

        Args:
            image_id (str): Docker image in the following format ``"<registry>/<image>:<tag>"``
        """
        if self.image_id:
            raise ValueError("Setting both a machine image and docker image is not yet supported.")
        self.image_id = image_id
        return self

    @classmethod
    def from_dockerfile(cls, dockerfile_path: str, name: str = None) -> "Image":
        """
        Construct an Image from an existing Dockerfile.

        Supports: FROM, RUN, ENV, COPY, CMD, ENTRYPOINT instructions. Comments (#) are preserved.
        CMD and ENTRYPOINT are used to determine the command to run only for kt apply, and are ignored
        in other use cases.

        Args:
            dockerfile_path (str): Path to the Dockerfile
            name (str, optional): Optional name for the image

        Example:
            .. code-block:: python

                import kubetorch as kt

                # Load from existing Dockerfile
                image = kt.Image.from_dockerfile("./Dockerfile")

                # Extend with additional steps
                image = (
                    kt.Image.from_dockerfile("./base.dockerfile", name="my-image")
                    .pip_install(["extra-package"])
                    .set_env_vars({"MY_VAR": "value"})
                )
        """
        path = Path(dockerfile_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dockerfile not found: {path}")

        with open(path, "r") as f:
            lines = f.readlines()

        image = cls(name=name)
        image._parse_dockerfile(lines)
        return image

    def _parse_dockerfile(self, lines: List[str]):
        """Parse dockerfile lines and populate _dockerfile_contents and copy_operations."""
        current_line = ""
        line_num = 0

        for raw_line in lines:
            line_num += 1
            stripped = raw_line.strip()

            if not stripped:
                if current_line:
                    self._process_dockerfile_instruction(current_line, line_num)
                    current_line = ""
            elif stripped.endswith("\\"):
                current_line += stripped[:-1].strip() + " "
            else:
                current_line += stripped
                self._process_dockerfile_instruction(current_line, line_num)
                current_line = ""

        # Handle final line if no newline at end
        if current_line:
            self._process_dockerfile_instruction(current_line, line_num)

    def _process_dockerfile_instruction(self, line: str, line_num: int):
        """Process a single dockerfile instruction."""
        if line.startswith("#"):
            self._dockerfile_contents.append(line)
            return

        parts = line.split(maxsplit=1)
        instruction = parts[0].upper()

        if instruction not in self.SUPPORTED_DOCKERFILE_INSTRUCTIONS:
            raise ValueError(
                f"Unsupported Dockerfile instruction '{instruction}' at line {line_num}. "
                f"Kubetorch Image supports: {', '.join(sorted(self.SUPPORTED_DOCKERFILE_INSTRUCTIONS))}. "
                f"Remove unsupported instructions or use a pre-built Docker image with from_docker()."
            )
        if len(parts) < 2:
            raise ValueError(f"Invalid {instruction} instruction at line {line_num}: {line}")

        if instruction == "FROM":
            self.image_id = parts[1].strip()
        elif instruction == "ENV":
            env_content = parts[1]
            if env_content.startswith("KT_PYTHON_PATH"):
                next_char = env_content[len("KT_PYTHON_PATH")]
                if next_char == "=":
                    self.python_path = env_content.split("=", 1)[1].strip()
                elif next_char == " ":
                    self.python_path = env_content.split(maxsplit=1)[1].strip()
            else:
                self._dockerfile_contents.append(line)
        elif instruction == "RUN":
            self._dockerfile_contents.append(line)
        elif instruction == "COPY":
            copy_parts = line.split()
            if len(copy_parts) < 3:
                raise ValueError(f"Invalid COPY instruction at line {line_num}: {line}")
            source = copy_parts[1]
            dest = copy_parts[2]

            self._dockerfile_contents.append(line)
            self.copy_operations.append(
                {
                    "source": source,
                    "dest": dest,
                    "contents": False,
                    "filter_options": None,
                    "force": False,
                }
            )
        elif instruction == "CMD":
            self._cmd = self._parse_cmd_entrypoint(parts[1], line_num)
        elif instruction == "ENTRYPOINT":
            self._entrypoint = self._parse_cmd_entrypoint(parts[1], line_num)

    def _parse_cmd_entrypoint(self, value: str, line_num: int) -> List[str]:
        """Parse CMD or ENTRYPOINT value, supporting both exec and shell forms."""
        import json

        value = value.strip()

        # Exec form: starts with [ and ends with ]
        if value.startswith("["):
            try:
                parsed = json.loads(value)
                return " ".join(parsed)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in CMD/ENTRYPOINT at line {line_num}: {e}")

        # String format
        return value

    @property
    def dockerfile_command(self) -> str:
        """Extract and format the command to run from dockerfile CMD and ENTRYPOINT."""
        if self._entrypoint and self._cmd:
            # ENTRYPOINT + CMD: CMD provides arguments to ENTRYPOINT
            return f"{self._entrypoint} {self._cmd}"
        return self._cmd or self._entrypoint

    ########################################################
    # Steps to build the image
    ########################################################

    def pip_install(
        self,
        reqs: List[Union[str, Any]],
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
        install_cmd = self.install_cmd or "$KT_PIP_INSTALL_CMD"
        reqs = [reqs] if isinstance(reqs, str) else reqs

        for req in reqs:
            line = f"RUN {install_cmd} {req}"
            if force:
                line += " # force"
            self._dockerfile_contents.append(line)
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
        for key, val in env_vars.items():
            self._dockerfile_contents.append(f"ENV {key} {val}")
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
        from kubetorch.resources.compute.utils import _get_sync_package_paths

        full_path, dest_dir = _get_sync_package_paths(package)

        line = f"COPY {full_path} {dest_dir}"
        if force:
            line += " # force"
        self._dockerfile_contents.append(line)

        # Add to copy operations list for rsync at deploy time
        self.copy_operations.append(
            {
                "source": str(full_path),
                "dest": dest_dir,
                "contents": False,
                "filter_options": None,
                "force": force,
            }
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
        line = f"RUN {command}"
        if force:
            line += " # force"
        self._dockerfile_contents.append(line)
        return self

    def copy(
        self,
        source: str,
        dest: str = None,
        contents: bool = False,
        filter_options: str = None,
        force: bool = False,
    ):
        """Copy files or directories from local machine to the remote container.

        This method efficiently transfers files to remote containers using rsync,
        which only copies changed files and supports compression. Files are first
        uploaded to a jump pod and then distributed to worker pods on startup.

        Args:
            source (str): Path to the local file or directory to copy.
                Supports absolute (`/path/to/file`), relative (`./data/file.txt`),
                and home directory (`~/documents/data.csv`) paths.
            dest (str, optional): Target path on the remote container.
                Supports absolute (`/data/config.yaml`), relative (`configs/settings.json`),
                and home directory (`~/results/output.txt`) paths. If not provided, uses the basename of the source path.
                (Default: ``None``)
            contents (bool, optional): For directories only - whether to copy the contents
                of the directory itself. If ``True``, the **contents** of the source directory are copied
                into the destination. If ``False`` the source directory itself (along with its contents)
                are copied to the destination, creating an additional directory layer at the destination.
                (Default: ``False``)
            filter_options (str, optional): Additional filter options for the underlying rsync.
                These are added to (not replacing) the default filters, which filter out:
                 `.gitignore`, `.ktignore`, common Python artifacts (`*.pyc`, `__pycache__`),
                 virtual environments (`.venv`), and git metadata (`.git`).

                Your filter_options are appended after these defaults. Examples:

                - Exclude more patterns: ``"--exclude='*.log' --exclude='temp/'"``
                - Include specific files: ``"--include='important.log' --exclude='*.log'"``
                - Override all defaults: Set ``KT_RSYNC_FILTERS`` environment variable

                (Default: ``None``)
            force (bool, optional): When ``True``, forces transfer of all files regardless
                of modification times. This ensures all files are copied even if timestamps
                suggest they haven't changed. Useful when timestamp-based change detection
                is unreliable.
                Note: Files are always synced when deploying with ``.to()``, this flag just
                affects how the underlying rsync determines which files need updating.
                (Default: ``False``)

        Returns:
            Image: Returns self for method chaining.

        Examples:
            .. code-block:: python

                import kubetorch as kt

                # File copy
                image = kt.images.Debian().copy("./config.yaml", "app/config.yaml")
                image = kt.images.Python312().copy("./model_weights.pth", "/models/weights.pth")  # absolute path
                image = kt.images.Ubuntu().copy("/local/data/dataset.csv")  # Goes to ./dataset.csv

                # Directory copy
                image = kt.images.Debian().copy("./src", "app")  # Creates app/src/
                image = kt.images.Debian().copy("./src", "app", contents=True)  # Contents into app/

                # Multiple copy operations with filtering
                image = (
                    kt.images.Python312()
                    .copy("./data", "/data", filter_options="--exclude='*.tmp'")
                    .copy("./configs", "~/configs")
                    .copy("./scripts")
                    .pip_install(["numpy", "pandas"])
                )

                # Force re-copy for development
                image = (
                    kt.images.Debian()
                    .copy("./rapidly_changing_code", "app", force=True)
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
        # Resolve source path and determine destination
        source_path = Path(source).expanduser().resolve()

        dest_for_dockerfile = dest if dest is not None else source_path.name
        line = f"COPY {source_path} {dest_for_dockerfile}"
        if force:
            line += " # force"
        self._dockerfile_contents.append(line)

        # Add to copy operations list for rsync at deploy time
        self.copy_operations.append(
            {
                "source": str(source_path),
                "dest": dest,
                "contents": contents,
                "filter_options": filter_options,
                "force": force,
            }
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
        """Deprecated: Use :meth:`copy` instead.

        This method is deprecated and will be removed in a future release.
        """
        warnings.warn(
            "Image.rsync() is deprecated and will be removed in a future release. " "Use Image.copy() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.copy(
            source=source,
            dest=dest,
            contents=contents,
            filter_options=filter_options,
            force=force,
        )
