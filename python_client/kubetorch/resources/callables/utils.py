import importlib.metadata as metadata
import inspect
import json
import os
from pathlib import Path
from typing import Callable, Optional, Type, Union

from kubetorch.logger import get_logger

logger = get_logger(__name__)

SHELL_COMMANDS = {"ssh", "run_bash", "rsync"}


def extract_pointers(raw_cls_or_fn: Union[Type, Callable]):
    """Get the path to the module, module name, and function name to be able to import it on the server"""
    if not (isinstance(raw_cls_or_fn, Type) or isinstance(raw_cls_or_fn, Callable)):
        raise TypeError(f"Expected Type or Callable but received {type(raw_cls_or_fn)}")

    # (root_path, module_name, cls_or_fn_name)
    return _get_module_import_info(raw_cls_or_fn)


def _get_module_import_info(raw_cls_or_fn: Union[Type, Callable]):
    """
    Given a class or function in Python, get all the information needed to import it in another Python process.
    """

    # Background on all these dunders: https://docs.python.org/3/reference/import.html
    py_module = inspect.getmodule(raw_cls_or_fn)

    # Need to resolve in case just filename is given
    module_path = _extract_module_path(raw_cls_or_fn)

    # TODO better way of detecting if in a notebook or interactive Python env
    if not module_path or module_path.endswith("ipynb"):
        # The only time __file__ wouldn't be present is if the function is defined in an interactive
        # interpreter or a notebook. We can't import on the server in that case, so we need to cloudpickle
        # the fn to send it over. The __call__ function will serialize the function if we return it this way.
        # This is a short-term hack.
        # return None, "notebook", raw_fn.__name__
        root_path = os.getcwd()
        module_name = "notebook"
        cls_or_fn_name = raw_cls_or_fn.__name__
    else:
        root_path = os.path.dirname(module_path)
        module_name = inspect.getmodulename(module_path)
        # TODO __qualname__ doesn't work when fn is aliased funnily, like torch.sum
        cls_or_fn_name = getattr(raw_cls_or_fn, "__qualname__", raw_cls_or_fn.__name__)

        # Adapted from https://github.com/modal-labs/modal-client/blob/main/modal/_function_utils.py#L94
        if getattr(py_module, "__package__", None):
            module_path = os.path.abspath(py_module.__file__)
            package_paths = [
                os.path.abspath(p) for p in __import__(py_module.__package__).__path__
            ]
            base_dirs = [
                base_dir
                for base_dir in package_paths
                if os.path.commonpath((base_dir, module_path)) == base_dir
            ]

            if len(base_dirs) != 1:
                raise Exception("Wasn't able to find the package directory!")
            root_path = os.path.dirname(base_dirs[0])
            module_name = py_module.__spec__.name

    return root_path, module_name, cls_or_fn_name


def _extract_module_path(raw_cls_or_fn: Union[Type, Callable]):
    py_module = inspect.getmodule(raw_cls_or_fn)

    # Need to resolve in case just filename is given
    module_path = (
        str(Path(inspect.getfile(py_module)).resolve())
        if hasattr(py_module, "__file__")
        else None
    )

    return module_path


def locate_working_dir(start_dir=None):
    """
    Locate the working directory of the project.

    Args:
        start_dir (str, optional): The directory to start searching from. Defaults to the current working directory.

    Returns:
        tuple: A tuple containing the working directory and a boolean indicating if a project directory was found.
    """
    if start_dir is None:
        start_dir = os.getcwd()

    # Search first for anything that represents a Python package
    target_files = [
        ".git",
        "setup.py",
        "setup.cfg",
        "pyproject.toml",
        "requirements.txt",
    ]

    dir_with_target = _find_directory_containing_any_file(
        start_dir, target_files, searched_dirs=set()
    )
    found_project_dir = dir_with_target is not None
    return (dir_with_target if found_project_dir else start_dir), found_project_dir


def _find_directory_containing_any_file(dir_path, files, searched_dirs=None):
    if Path(dir_path) == Path.home() or dir_path == Path("/"):
        return None

    if any(Path(dir_path, file).exists() for file in files):
        return str(dir_path)

    searched_dirs.add(dir_path)
    parent_path = Path(dir_path).parent
    if parent_path in searched_dirs:
        return None
    return _find_directory_containing_any_file(
        parent_path, files, searched_dirs=searched_dirs
    )


def get_local_install_path(package_name: str) -> Optional[str]:
    from importlib.metadata import distributions

    for dist in distributions():
        direct_url_json = dist.read_text("direct_url.json")
        if direct_url_json and dist.metadata["Name"].lower() == package_name.lower():
            try:
                url = json.loads(direct_url_json).get("url", None)
                if url:
                    if url.startswith("file://"):
                        return url[len("file://") :]
            except json.JSONDecodeError:
                pass
    return None


def find_locally_installed_version(package_name: str) -> Optional[str]:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def get_names_for_reload_fallbacks(name: str, prefixes: list[str] = []):
    from kubetorch.globals import config
    from kubetorch.servers.http.utils import clean_and_validate_k8s_name
    from kubetorch.utils import current_git_branch, validate_username

    current_prefix = config.username
    fallback_prefixes = []

    if prefixes:
        fallback_prefixes = prefixes
    else:
        # try reloading based on current username or current git branch (in that order)
        branch = current_git_branch()
        if branch:
            # Ensure that we use the truncated branch name that was used to create the service initially
            valid_branch = validate_username(branch)
            # Note: username/prefix takes precedence over branch (in the event they differ)
            fallback_prefixes = [
                v for v in (current_prefix, valid_branch) if v is not None
            ]
        else:
            fallback_prefixes = [current_prefix] if current_prefix else []

    potential_names = [
        clean_and_validate_k8s_name(f"{prefix}-{name}", allow_full_length=True)
        for prefix in fallback_prefixes
    ]
    if not prefixes and name not in potential_names:
        # try loading the bare name (i.e. prod mode) last, but only if we're not looking for specific prefixes
        potential_names.append(name)

    return potential_names
