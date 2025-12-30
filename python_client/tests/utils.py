import importlib
import inspect
import logging
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

from kubetorch.globals import config
from kubetorch.utils import current_git_branch, validate_username

from pydantic import BaseModel

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def get_test_logger(name=None):
    """Use a generic logger for testing that doesn't require a kubetorch dependency."""
    logger = logging.getLogger(name or __name__)

    # Avoid adding handlers if they already exist
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


def get_test_fn_name():
    """Get the name of the test function."""
    # This is a workaround to get the test function name in a way that works with pytest
    # and doesn't require pytest to be imported in this module.
    frame = sys._getframe(1)  # Get the caller's frame
    name = frame.f_code.co_name if frame else "unknown-test-function"
    # K8s resource names must be RFC 1123 compliant (lowercase alphanumeric + hyphens)
    return name.replace("_", "-")


logger = get_test_logger()


########################################################################################
# Functions and classes that will be sent to kt.compute as part of in multiple tests
########################################################################################


def simple_summer(a, b):
    return a + b


def simple_summer_with_logs(a, b):
    for i in range(a + b):
        logger.info(f"This {i}th log from nested service")
    return a + b


async def async_simple_summer(a, b, return_times=False):
    import asyncio
    import time

    if return_times:
        start_time = time.time()
        await asyncio.sleep(2)
        return start_time, time.time()

    await asyncio.sleep(0.1)
    return a + b


class OSInfoResponse(BaseModel):
    name: str
    value: str


class TestModel(BaseModel):
    name: str
    value: int


class SlowNumpyArray:
    def __init__(self, size=5, sleep_time=None):
        import time

        import numpy as np

        if sleep_time is not None:
            time.sleep(sleep_time)

        self.size = size
        self.arr = np.zeros(self.size)
        self._hidden_1 = "hidden"

    def print_and_log(self, i):
        print(f"Hello from the cluster stdout! {i}")
        logger.info(f"Hello from the cluster logs! {i}")
        self.arr[i] = i
        return f"Hello from the cluster! {self.arr}"

    @classmethod
    def local_home(cls, local=True):
        return os.path.expanduser("~")

    @classmethod
    def home(cls):
        return os.path.expanduser("~")

    @classmethod
    def cpu_count(cls):
        return os.cpu_count()

    @classmethod
    def os_info(cls, os_info_requests: List["OSInfoRequest"]):
        import os

        responses = []
        for request in os_info_requests:
            value = getattr(os, request.method, None)()
            responses.append(OSInfoResponse(name=request.method, value=str(value)))
        return responses

    def size_minus_cpus(self):
        return self.size - self.cpu_count()

    @classmethod
    def factory_constructor(cls, size=5):
        return cls(size=size)

    def method_with_breakpoint(self):
        """Method that includes a breakpoint for testing debug functionality."""

        x = 42
        breakpoint()  # This will trigger the debug server
        print(x)
        print("Stop point")
        return f"Breakpoint method executed, x={x}"


class TorchArray:
    def __init__(self, size=5):
        import torch

        self.size = size
        self.arr = torch.zeros(self.size)

    def print_and_log(self, i):
        import torch

        torch.distributed.init_process_group(backend="gloo")
        print(f"Rank {torch.distributed.get_rank()} of {torch.distributed.get_world_size()} initialized")
        print(f"Hello from the cluster stdout! {i}")
        logger.info(f"Hello from the cluster logs! {i}")
        self.arr[i] = i
        torch.distributed.destroy_process_group()
        return f"Hello from the cluster! {self.arr}"

    def raise_exception(self, exception_message):
        raise ValueError(exception_message)


class MemoryHog:
    def __init__(self):
        import numpy as np

        # Allocate 2GB array and override it to be used
        self.data = np.ones((512, 512, 1024), dtype=np.float32)  # ~1GB
        self.data2 = np.ones((512, 512, 1024), dtype=np.float32)  # Another ~1GB
        # Force memory allocation by doing an operation
        self.result = np.sum(self.data + self.data2)

    def print_memory_usage(self):
        import psutil

        print(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024} MB")


class GPUMemoryHog:
    def __init__(self):
        import torch

        # Try to allocate more memory than a typical GPU has
        # Most GPUs have 8-16GB, so trying to allocate 32GB should trigger OOM
        self.data = torch.ones((8192, 8192, 128), device="cuda", dtype=torch.float32)  # ~32GB
        # Force memory allocation
        self.result = self.data.sum().item()
        print(f"Allocated and summed {self.data.numel() * 4 / 1e9:.1f}GB on GPU")

    def print_memory_usage(self):
        import torch

        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.1f}GB")


class CrashingService:
    def __init__(self):
        # This will cause the container to crash immediately
        raise RuntimeError("Service intentionally crashing on startup")

    def some_method(self):
        pass


class ResourceHungryService:
    def __init__(self):
        # Just a placeholder - the service won't actually start
        pass

    def some_method(self):
        pass

    def consume_memory(self):
        a = []
        while True:
            a.append(" " * 10**7)  # ~10MB per iteration

    def consume_disk(self):
        i = 0
        while True:
            with open(f"/tmp/fill_{i}.txt", "w") as f:
                f.write("X" * 1024 * 1024 * 5)  # 5MB
            i += 1

    def sleep_forever(self):
        """Stay alive indefinitely to simulate long pod lifetime"""
        import time

        while True:
            time.sleep(10)


def get_os_info():
    return os.uname()


def summer(a, b, sleep_time=None, return_pod_name=False, use_tqdm=False):
    print(f"Hello from the cluster stdout! {a} {b}")
    logger.info(f"Hello from the cluster logs! {a} {b}")

    # To test pickle serialization, test_serialization_formats will pass a Pydantic model here
    # we reuse it for the return value for simplicity
    pydantic_type = type(a) if hasattr(a, "model_dump") else None
    a = a.value if hasattr(a, "model_dump") else a
    b = b.value if hasattr(b, "model_dump") else b

    if use_tqdm:
        import time

        from tqdm import tqdm

        for i in tqdm(range(a + b)):
            logger.info(f"Tqdm iteration {i}")
            time.sleep(0.1)

    if sleep_time is not None:
        import time

        time.sleep(sleep_time)

    if return_pod_name:
        pod_name = os.environ.get("POD_NAME", None)
        return a + b, pod_name

    if pydantic_type is not None:
        return pydantic_type(value=a + b, name="sum_result")

    return a + b


def torch_summer(a, b):
    import torch

    # FYI: Can be retrieved via `kubectl logs summer-00001-deployment-864d74c46c-767qf -c runhouse`
    print(f"Hello from the cluster stdout! {a} {b}")
    logger.info(f"Hello from the cluster logs! {a} {b}")

    rank = os.environ.get("RANK", None)
    logger.info(f"Rank {rank}: MASTER_ADDR {os.environ['MASTER_ADDR']}")
    logger.info(f"Rank {rank}: MASTER_PORT {os.environ['MASTER_PORT']}")
    logger.info(f"Rank {rank}: WORLD_SIZE {os.environ['WORLD_SIZE']}")
    logger.info(f"Rank {rank}: LOCAL_RANK {os.environ['LOCAL_RANK']}")

    return int(torch.sum(torch.tensor([a, b])))


def slow_iteration(array_len: int = 5):
    import time

    arr = [0.0] * array_len
    final_res = []
    for i in range(array_len):
        start_time = time.time()
        print(f"Hello from the cluster stdout! {i}")
        logger.info(f"Hello from the cluster logs! {i}")
        arr[i] = i
        final_res.append(f"Hello from the cluster! {arr}")
        time.sleep(1)
        end_time = time.time()
        logger.info(f"round: {i}, slept for {end_time - start_time}")
    return final_res


def log_n_messages(msg: str = "Hello from cluster logs!", n: int = 10, sleep_time: int = 1):
    for i in range(n):
        logger.info(f"{msg} {i}")
    time.sleep(sleep_time)
    return f"{msg} was logged {n} times"


def get_cuda_version():
    try:
        output = subprocess.check_output(["nvidia-smi"])
        # Convert bytes to string
        output = output.decode("utf-8")
        # Find CUDA Version in the output
        cuda_version = output.split("CUDA Version: ")[1].split(" ")[0]
        return cuda_version
    except:
        return None


def get_env_var(key):
    import os

    return os.environ.get(key)


def python_version_and_path():
    import os
    import sys

    # To check that pip installed in the right environment
    import pytest  # noqa

    return sys.version, os.environ.get("PATH")


OP_SUM = """
def operation(a, b):
    return a+b
"""

OP_MUL = """
def operation(a, b):
    return a*b
"""

#############
# Helpers
#############


def random_string(length=4):
    return str(uuid.uuid4())[:length]


def get_commit_hash():
    try:
        commit_info = (
            subprocess.check_output(["git", "show", "--format=fuller", "--no-patch"])
            .decode("utf-8")
            .strip()
            .split("\n")
        )
        commit_hash = commit_info[0].split(" ")[1][:8]
        return commit_hash
    except subprocess.CalledProcessError:
        return None  # Handle error if not in a Git repo


def generate_test_hash():
    # generate session unique hash
    hash = get_commit_hash()[:4] + random_string(2) or random_string(6)
    name = f"t-{hash}"
    if os.getenv("CI"):
        # If running in CI, add the git branch as the prefix
        branch = current_git_branch()
        if branch:
            name = validate_username(branch)

    return name


def create_random_name_prefix():
    # Note: hash is not used here because it will be used as a prefix via the username
    return f"{datetime.now().strftime('%H%M%S')}"


def get_tests_namespace():
    from tests.conftest import get_test_hash

    return f"kt-test-{get_test_hash()}"


def teardown_test_resources(test_hash):
    test_namespace = get_tests_namespace()
    default_namespace = config.namespace
    # Try to teardown from both namespaces
    for ns in [default_namespace, test_namespace]:
        result = subprocess.run(f"kt teardown -p {test_hash} -n {ns} -y -f", shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            if "403" in result.stderr or "Forbidden" in result.stderr:
                # Namespace not configured in controller RBAC (may happen in local dev)
                logger.info(f"Skipping teardown for namespace {ns} (no RBAC permissions)")
            else:
                # Unexpected error - log but continue with other namespace
                logger.warning(f"Teardown failed for namespace {ns}: {result.stderr}")


def load_callable_from_test_dir(test_dir):
    """Helper method to load callable from the assets directory"""
    assets_dir = Path(__file__).parent / "assets" / Path(test_dir).name
    py_file = next(p for p in assets_dir.glob("*.py") if p.name != "__init__.py")

    spec = importlib.util.spec_from_file_location("module", py_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    name = Path(test_dir).name
    try:
        callable_obj = getattr(module, name.title())
    except AttributeError:
        callable_obj = getattr(module, name.lower())

    is_class = inspect.isclass(callable_obj)
    return name, callable_obj, is_class


def _update_metadata_env_vars(assets_dir, set=True):
    env_vars = [
        "KT_FILE_PATH",
        "KT_MODULE_NAME",
        "KT_CLS_OR_FN_NAME",
        "KT_INIT_ARGS",
        "KT_DISTRIBUTED_CONFIG",
        "KT_FREEZE",
    ]
    for env_var in env_vars:
        os.environ[env_var] = "None"

    with open(assets_dir / "metadata.dockerfile", "r") as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    for line in lines:
        _, key, val = line.split(" ", 2)
        if set:
            if val != "None":
                os.environ[key] = val
        else:
            os.environ.pop(key, None)


def get_sys_module(module_name, path):
    if module_name in sys.modules:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def write_temp_file_fn(temp_fd, temp_path, fn_contents):
    # temp_fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    with os.fdopen(temp_fd, "w") as temp_file:
        temp_file.write(fn_contents)
        temp_file.flush()
    os.chmod(temp_path, 0o644)


def service_deployer(service_name: str):
    import kubetorch as kt

    deployed = kt.fn(simple_summer, service_name).to(kt.Compute(cpus=".1", image=kt.images.Debian()))

    return deployed(1, 2)


def service_deployer_with_raycluster(service_name: str):
    import kubetorch as kt

    deployed = kt.fn(simple_summer, service_name).to(
        kt.Compute(
            cpus="2",
            memory="4Gi",
            image=kt.images.Ray(),
            launch_timeout=450,
        ).distribute("ray", workers=2)
    )

    return deployed(1, 2)


def service_deployer_with_logs(service_name: str):
    import kubetorch as kt

    for i in range(5):
        logger.info(f"This is the {i}th log from the parent service")

    deployed = kt.fn(simple_summer_with_logs, service_name).to(kt.Compute(cpus=".1", image=kt.images.Debian()))

    return deployed(2, 3)


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)
