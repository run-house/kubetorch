import enum
import os

import pytest

TEST_SESSION_HASH = None


@pytest.fixture
def client():
    """Fixture that provides a KubetorchClient instance for testing."""
    from kubetorch import KubetorchClient

    return KubetorchClient()


@pytest.fixture
def mock_response():
    """Fixture that provides a mock response for testing API calls."""
    return {"status": "success", "data": {"id": "test-id", "name": "test-name"}}


class TestLevels(str, enum.Enum):
    UNIT = "unit"
    MINIMAL = "minimal"
    RELEASE = "release"


DEFAULT_LEVEL = TestLevels.MINIMAL

TEST_LEVEL_HIERARCHY = {
    TestLevels.UNIT: 0,
    TestLevels.MINIMAL: 1,
    TestLevels.RELEASE: 2,
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
            cpus="2",
            memory="3Gi",
            labels={"test-label": "test_value"},
            gpu_anti_affinity=True,
            launch_timeout=450,
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
async def remote_fn(request):
    import kubetorch as kt

    from .utils import summer

    compute_type = os.getenv("TEST_COMPUTE_TYPE", "deployment")
    compute = get_compute(compute_type)

    name = f"{compute_type}-summer"
    fn = await kt.fn(summer, name=name).to_async(compute)
    return fn


@pytest.fixture(scope="session")
async def remote_async_fn(request):
    import kubetorch as kt

    from .utils import async_simple_summer

    compute_type = os.getenv("TEST_COMPUTE_TYPE", "deployment")
    compute = get_compute(compute_type)

    name = f"{compute_type}-async-summer"
    fn = await kt.fn(async_simple_summer, name=name).to_async(compute)
    fn.async_ = True
    return fn


@pytest.fixture(scope="session")
async def remote_logs_fn():
    import kubetorch as kt

    from .utils import log_n_messages

    remote_fn = await kt.fn(log_n_messages).to_async(
        kt.Compute(
            cpus=".01",
            gpu_anti_affinity=True,
        )
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

    compute = kt.Compute(
        cpus=".01",
        image=kt.images.Debian()
        .pip_install(["pytest", "pytest-asyncio", "typer", "rich"])
        .run_bash("uv pip install --system --break-system-packages numpy"),
        env_vars={"OMP_NUM_THREADS": 1},
        annotations={"test-annotation": "test_value"},
        tolerations=[
            {
                "key": "test.toleration.key",
                "operator": "Equal",
                "value": "test-value",
                "effect": "NoSchedule",
            }
        ],
        gpu_anti_affinity=True,
        allowed_serialization=["json", "pickle"],
        launch_timeout=300,
    )

    if compute_type == "knative":
        compute = compute.autoscale(min_replicas=1)

    if compute_type == "ray":
        compute = kt.Compute(
            cpus="2",
            memory="3Gi",
            labels={"test-label": "test_value"},
            gpu_anti_affinity=True,
            launch_timeout=300,
            image=kt.images.Ray()
            .pip_install(["pytest", "pytest-asyncio", "typer", "rich"])
            .run_bash("uv pip install --system --break-system-packages numpy"),
            allowed_serialization=["json", "pickle"],
        ).distribute("ray", workers=2)

    name = f"{compute_type}-slow-cls"
    remote_cls = await kt.cls(SlowNumpyArray, name=name).to_async(compute=compute, init_args={"size": 10})
    return remote_cls


@pytest.fixture(scope="session")
async def remote_monitoring_fn():
    import kubetorch as kt

    from .utils import slow_iteration

    remote_fn = await kt.fn(slow_iteration).to_async(
        kt.Compute(
            cpus=".01",
            gpu_anti_affinity=True,
        )
    )
    return remote_fn


def get_test_hash():
    return TEST_SESSION_HASH
