import asyncio
import enum
import inspect
import os

import pytest

TEST_SESSION_HASH = None

# ==================== Eager Fixture Initialization ====================
# When --eager is passed, all session-scoped async fixtures are initialized
# in parallel at session start, rather than serially on first use.
#
# Usage:
#   pytest --eager          # Parallel fixture init (faster)
#   pytest                  # Serial fixture init (default, for debugging)
# ====================
KUBETORCH_IMAGE = "ghcr.io/run-house/kubetorch:main"


@pytest.fixture
def mock_response():
    """Fixture that provides a mock response for testing API calls."""
    return {"status": "success", "data": {"id": "test-id", "name": "test-name"}}


class TestLevels(str, enum.Enum):
    UNIT = "unit"
    MINIMAL = "minimal"
    RELEASE = "release"
    GPU = "gpu"


DEFAULT_LEVEL = TestLevels.MINIMAL

TEST_LEVEL_HIERARCHY = {
    TestLevels.UNIT: 0,
    TestLevels.MINIMAL: 1,
    TestLevels.RELEASE: 2,
    TestLevels.GPU: 3,
}


def pytest_addoption(parser):
    parser.addoption(
        "--level",
        action="store",
        default=DEFAULT_LEVEL,
        help="Fixture set to spin up: unit, minimal, release, or maximal",
    )

    parser.addoption("--hash", action="store", default=None, help="Commit hash")
    parser.addoption(
        "--detached",
        action="store_true",
        default=False,
        help="Keep test artifacts (disable automatic cleanup)",
    )
    parser.addoption(
        "--eager",
        action="store_true",
        default=False,
        help="Initialize session-scoped async fixtures in parallel at session start",
    )


@pytest.hookimpl(trylast=True)
def pytest_collection_finish(session):
    """After test collection, initialize session-scoped async fixtures in parallel."""
    if not session.config.getoption("--eager"):
        return

    fm = session._fixturemanager

    # Collect all needed fixture names from all tests
    needed_fixtures = set()
    for item in session.items:
        needed_fixtures.update(item.fixturenames)

    # Find session-scoped async fixtures that are needed and have no parameters
    eager_fixtures = []
    for name in needed_fixtures:
        if name not in fm._arg2fixturedefs:
            continue
        for fixturedef in fm._arg2fixturedefs[name]:
            func = fixturedef.func
            # Unwrap to check if original is async
            unwrapped = func
            while hasattr(unwrapped, "__wrapped__"):
                unwrapped = unwrapped.__wrapped__

            # Skip fixtures that have parameters (dependencies we can't resolve)
            sig = inspect.signature(unwrapped)
            if sig.parameters:
                continue

            if fixturedef.scope == "session" and inspect.iscoroutinefunction(unwrapped):
                eager_fixtures.append((name, fixturedef, unwrapped))

    if not eager_fixtures:
        return

    print(f"\n🚀 Eager init: {[name for name, _, _ in eager_fixtures]}")

    # Run all eager fixtures in parallel
    async def run_eager():
        async def run_one(name, fixturedef, func):
            return name, fixturedef, await func()

        tasks = [run_one(n, fd, f) for n, fd, f in eager_fixtures]
        return await asyncio.gather(*tasks)

    results = asyncio.run(run_eager())

    # Inject results directly into pytest's fixture cache
    for name, fixturedef, result in results:
        fixturedef.cached_result = (result, 0, None)

    print(f"✅ Eager init complete: {len(results)} fixtures ready")


def pytest_collection_modifyitems(config, items):
    request_level = config.getoption("level")
    new_items = []

    for item in items:
        test_level = item.get_closest_marker("level")
        if (
            test_level is not None
            and TEST_LEVEL_HIERARCHY[test_level.args[0]]
            == TEST_LEVEL_HIERARCHY[request_level]  # currently we get tests only with the provided label
        ):
            new_items.append(item)

    items[:] = new_items


@pytest.fixture(scope="session", autouse=True)
def test_hash_and_teardown(request):
    from kubetorch.globals import config

    from tests.utils import generate_test_hash, teardown_test_resources

    global TEST_SESSION_HASH
    TEST_SESSION_HASH = os.environ.get("TEST_HASH") or generate_test_hash()

    teardown_hash = TEST_SESSION_HASH

    # Set unique username for the scope of the test. Leave original username if detached
    if not request.config.getoption("detached") or not config.username:
        config.set("username", teardown_hash)
    else:
        teardown_hash = config.username

    yield teardown_hash

    if not request.config.getoption("detached"):
        print(f"Tearing down test artifacts for session with hash: {teardown_hash}")
        teardown_test_resources(teardown_hash)
    else:
        print(f"Keeping test artifacts for session with hash: {teardown_hash}")


def get_compute(compute_type: str):
    import kubetorch as kt

    if compute_type == "ray":
        return kt.Compute(
            cpus="0.5",
            memory="1Gi",
            labels={"test-label": "test_value"},
            gpu_anti_affinity=True,
            launch_timeout=600,
            image=kt.images.Ray()
            .pip_install(["pytest", "pytest-asyncio", "typer", "rich"])
            .run_bash("uv pip install --system --break-system-packages numpy"),
            allowed_serialization=["json", "pickle"],
        ).distribute("ray", workers=2)

    compute = kt.Compute(
        cpus=".01",
        labels={"test-label": "test_value"},
        gpu_anti_affinity=True,
        launch_timeout=450,
        shared_memory_limit="512Mi",
        allowed_serialization=["json", "pickle"],
        image=kt.images.Debian().pip_install(["pytest", "pytest-asyncio", "typer", "rich"]),
    )

    if compute_type == "knative":
        compute = compute.autoscale(min_replicas=1)

    return compute


@pytest.fixture(scope="session")
async def remote_fn():
    import kubetorch as kt

    from .utils import summer

    compute_type = os.getenv("TEST_COMPUTE_TYPE", "deployment")
    compute = get_compute(compute_type)
    compute.image = compute.image.pip_install(["tqdm"])

    service_name_prefix = os.getenv("SERVICE_NAME_PREFIX", compute_type)
    name = f"{service_name_prefix}-summer"
    fn = await kt.fn(summer, name=name).to_async(compute)
    return fn


@pytest.fixture(scope="session")
async def remote_async_fn():
    import kubetorch as kt

    from .utils import async_simple_summer

    compute_type = os.getenv("TEST_COMPUTE_TYPE", "deployment")
    compute = get_compute(compute_type)

    service_name_prefix = os.getenv("SERVICE_NAME_PREFIX", compute_type)
    name = f"{service_name_prefix}-async-summer"
    fn = await kt.fn(async_simple_summer, name=name).to_async(compute)
    fn.async_ = True
    return fn


@pytest.fixture(scope="session")
async def remote_logs_fn():
    import kubetorch as kt

    from .utils import log_n_messages

    remote_fn = await kt.fn(log_n_messages).to_async(
        kt.Compute(cpus="100m", gpu_anti_affinity=True, image_pull_policy="IfNotPresent")
    )
    return remote_fn


@pytest.fixture(scope="session")
async def remote_logs_fn_autoscaled():
    import kubetorch as kt

    from .utils import log_n_messages

    remote_fn = await kt.fn(log_n_messages, name="log-autoscaled").to_async(
        kt.Compute(
            cpus=".01",
            gpu_anti_affinity=True,
        ).autoscale(min_scale=2)
    )
    return remote_fn


@pytest.fixture(scope="session")
async def remote_cls():
    import kubetorch as kt

    from .utils import SlowNumpyArray

    compute_type = os.getenv("TEST_COMPUTE_TYPE", "deployment")

    common_kwargs = {
        "labels": {"test-label": "test_value"},
        "annotations": {"test-annotation": "test_value"},
        "gpu_anti_affinity": True,
        "allowed_serialization": ["json", "pickle"],
        "env_vars": {"OMP_NUM_THREADS": 1},
        "inactivity_ttl": "5m",
    }
    pip_packages = ["pytest", "pytest-asyncio", "typer", "rich"]
    numpy_install = "uv pip install --system --break-system-packages numpy && uv cache clean"

    if compute_type == "ray":
        compute = kt.Compute(
            **common_kwargs,
            cpus="0.5",
            memory="1Gi",
            launch_timeout=600,
            image=kt.images.Ray().pip_install(pip_packages).run_bash(numpy_install),
        ).distribute("ray", workers=2)
    else:
        compute = kt.Compute(
            **common_kwargs,
            cpus=".1",
            launch_timeout=300,
            tolerations=[
                {
                    "key": "test.toleration.key",
                    "operator": "Equal",
                    "value": "test-value",
                    "effect": "NoSchedule",
                }
            ],
            image=kt.images.Debian().pip_install(pip_packages).run_bash(numpy_install),
        )

        if compute_type == "knative":
            compute = compute.autoscale(min_replicas=1)

    service_name_prefix = os.getenv("SERVICE_NAME_PREFIX", compute_type)
    name = f"{service_name_prefix}-slow-cls"
    remote_cls = await kt.cls(SlowNumpyArray, name=name).to_async(compute=compute, init_args={"size": 10})
    return remote_cls


@pytest.fixture(scope="session")
async def remote_monitoring_fn():
    import kubetorch as kt

    from .utils import slow_iteration

    remote_fn = await kt.fn(slow_iteration).to_async(
        kt.Compute(cpus=".01", gpu_anti_affinity=True, image_pull_policy="Always")
    )
    return remote_fn


@pytest.fixture(scope="session")
async def remote_profiling_pyspy_fn():
    import kubetorch as kt

    from .utils import matrix_dot_np

    compute_type = os.getenv("TEST_COMPUTE_TYPE", "deployment")
    compute = get_compute(compute_type)
    compute.image = compute.image.pip_install(["numpy"])

    remote_fn = await kt.fn(matrix_dot_np).to_async(compute)
    return remote_fn


@pytest.fixture(scope="session")
async def remote_profiling_pyspy_cls():
    import kubetorch as kt

    from .utils import Matrix

    compute_type = os.getenv("TEST_COMPUTE_TYPE", "deployment")
    compute = get_compute(compute_type)
    compute.image = compute.image.pip_install(["numpy"])

    remote_cls = await kt.cls(Matrix).to_async(compute)
    return remote_cls


@pytest.fixture(scope="session")
async def remote_profiling_torch_fn():
    import kubetorch as kt

    from .utils import matrix_dot_torch

    compute = kt.Compute(gpus="1", image=kt.images.pytorch("24.02-py3"))

    remote_fn = await kt.fn(matrix_dot_torch).to_async(compute)
    return remote_fn


@pytest.fixture(scope="session")
async def remote_profiling_torch_cls():
    import kubetorch as kt

    from .utils import Matrix_GPU

    compute = kt.Compute(gpus="1", image=kt.images.pytorch("24.02-py3"))

    remote_cls = await kt.cls(Matrix_GPU).to_async(compute)
    return remote_cls


def get_test_hash():
    return TEST_SESSION_HASH
