import asyncio
import os

# Mimic CI for this test suite even locally, to ensure that
# resources are created with the branch name prefix
os.environ["CI"] = "true"

import pytest

from pydantic import BaseModel

from .utils import SlowNumpyArray, summer, TestModel


class OSInfoRequest(BaseModel):
    method: str


@pytest.fixture(autouse=True, scope="session")
def setup_test_env():
    # Keep the launch timeout low for this test suite, unless overridden (ex: for GPU tests)
    os.environ["KT_LAUNCH_TIMEOUT"] = "120"
    # Only set TEST_COMPUTE_TYPE if it's not already set (to allow override from environment)
    if "TEST_COMPUTE_TYPE" not in os.environ:
        os.environ["TEST_COMPUTE_TYPE"] = "deployment"
    yield


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_fn_sync_reload_by_name_only(remote_fn):
    import kubetorch as kt

    # Note: set to `get_if_exists` to `False` in order to stop any fallback and only look for an exact name match
    service_name = remote_fn.service_name
    some_reloaded_fn = kt.fn(summer, name=service_name, get_if_exists=True)
    assert some_reloaded_fn(2, 2) == 4

    another_reloaded_fn = kt.fn(name=service_name, get_if_exists=True)
    assert another_reloaded_fn(2, 3) == 5

    # Uncomment to manually test pdb
    # remote_fn(1, 2, pdb=True)


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_fn_sync_reload_by_name_with_prefixes(remote_fn):
    import kubetorch as kt
    from kubetorch.utils import current_git_branch, validate_username

    # Note: bc this is running in CI, the service name will have a git branch prefix
    current_branch = current_git_branch()
    branch_prefix = validate_username(current_branch)
    # Note: username prefix required for CI tests where TEST_HASH is set in job yaml
    username_prefix = kt.config.username

    another_reloaded_fn = kt.fn(name=remote_fn.name, reload_prefixes=[branch_prefix, username_prefix])
    assert another_reloaded_fn(2, 3) == 5

    # Note: Different way to load function and using a list of prefixes
    compute_type = os.getenv("TEST_COMPUTE_TYPE")
    compute_prefix = f"{username_prefix}-{compute_type}"
    reloaded_fn = kt.fn(summer, reload_prefixes=[branch_prefix, username_prefix, compute_prefix])
    assert reloaded_fn(2, 2) == 4


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_send_fn_to_compute_with_reload(remote_fn):
    import kubetorch as kt

    service_name = remote_fn.service_name
    another_reloaded_fn = kt.fn(summer, name=service_name).to(
        remote_fn.compute,
        get_if_exists=True,
    )
    assert another_reloaded_fn(1, 2) == 3


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_skip_fn_deploy_if_already_exists(remote_fn):
    import kubetorch as kt

    name = remote_fn.name
    reloaded_fn = kt.fn(summer, name=name).to(
        kt.Compute(
            cpus=".01",
            gpu_anti_affinity=True,
            launch_timeout=300,
        ),
        get_if_exists=True,
    )
    assert reloaded_fn.service_name == remote_fn.service_name
    assert reloaded_fn.compute.pods() == remote_fn.compute.pods()


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_failure_to_reload_non_existent_fn(remote_fn):
    import kubetorch as kt

    with pytest.raises(ValueError):
        kt.fn(name="some-random-fn-name")

    with pytest.raises(ValueError):
        kt.fn(name=remote_fn.name, reload_prefixes=["fake-tag"])

    # Note: get_if_exists=False is not allowed without function object
    with pytest.raises(ValueError):
        kt.fn(name=remote_fn.name, get_if_exists=False)


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_serialization_formats(remote_fn, remote_cls):
    """Test that both JSON and pickle serialization work correctly."""
    compute_type = os.getenv("TEST_COMPUTE_TYPE", "deployment")
    if compute_type == "ray":
        pytest.skip("Skipping serialization tests for Ray compute type")

    # Test function serialization
    print("Testing function serialization...")

    # Test pickle serialization
    result_pickle = remote_fn(2, 3, serialization="pickle")
    assert result_pickle == 5

    # Test pickle serialization with Pydantic model
    result_pickle = remote_fn(
        TestModel(name="test_a", value=42),
        TestModel(name="test_b", value=42),
        serialization="pickle",
    )

    assert isinstance(result_pickle, TestModel)
    assert result_pickle.name == "sum_result"
    assert result_pickle.value == 84

    # Test class method serialization
    print("Testing class method serialization...")

    # Test pickle serialization
    cpu_count_pickle = remote_cls.cpu_count(serialization="pickle")
    assert isinstance(cpu_count_pickle, int)

    requests = [
        OSInfoRequest(method="uname"),
        OSInfoRequest(method="cpu_count"),
        OSInfoRequest(method="getpid"),
    ]
    result_pickle = remote_cls.os_info(requests, serialization="pickle")
    assert isinstance(result_pickle, list)
    assert all(isinstance(r, BaseModel) for r in result_pickle)
    assert all(r.name in ["uname", "cpu_count", "getpid"] for r in result_pickle)
    assert result_pickle[1].value == str(cpu_count_pickle)

    # Test module-level serialization setting
    print("Testing module-level serialization setting...")

    try:
        # Set remote_fn to use pickle by default
        original_serialization = remote_fn.serialization
        remote_fn.serialization = "pickle"

        # Now calls should use pickle by default
        result_pickle = remote_fn(TestModel(name="test_a", value=42), TestModel(name="test_b", value=42))
        assert isinstance(result_pickle, TestModel)
        assert result_pickle.name == "sum_result"
        assert result_pickle.value == 84

        # Can still override for individual calls
        result_override = remote_fn(3, 4, serialization="json")
        assert result_override == 7
    except Exception as e:
        raise e
    finally:
        remote_fn.serialization = original_serialization

    try:
        # Test class serialization setting
        original_serialization = remote_cls.serialization
        remote_cls.serialization = "pickle"

        # Test that class methods now use pickle by default
        requests = [OSInfoRequest(method="cpu_count")]
        result_pickle = remote_cls.os_info(requests, serialization="pickle")
        assert isinstance(result_pickle, list)
        assert isinstance(result_pickle[0], BaseModel)
        assert result_pickle[0].name == "cpu_count"
        assert result_pickle[0].value == str(cpu_count_pickle)

        # Can still override for individual calls
        result_override = remote_cls.size_minus_cpus(serialization="json")
        assert isinstance(result_override, int)
    except Exception as e:
        raise e
    finally:
        remote_cls.serialization = original_serialization


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_cls_sync_basic(remote_cls):
    remote_cpu_count = remote_cls.cpu_count()
    assert remote_cls.print_and_log(1) == "Hello from the cluster! [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]"
    assert remote_cls.size_minus_cpus() == 10 - remote_cpu_count


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_cls_sync_reload_by_name_only(remote_cls):
    import kubetorch as kt

    service_name = remote_cls.service_name
    reloaded_cls = kt.cls(SlowNumpyArray, name=service_name, get_if_exists=True)
    assert reloaded_cls.cpu_count() == remote_cls.cpu_count()

    another_reloaded_cls = kt.cls(name=service_name, get_if_exists=True)
    assert another_reloaded_cls.cpu_count() == reloaded_cls.cpu_count()


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_cls_sync_reload_by_name_with_prefixes(remote_cls):
    import kubetorch as kt
    from kubetorch.utils import current_git_branch, validate_username

    # Note: service name will have the git branch prefix
    current_branch = current_git_branch()
    branch_prefix = validate_username(current_branch)
    # Note: username prefix required for CI tests where TEST_HASH is set in job yaml
    username_prefix = kt.config.username

    reloaded_cls = kt.cls(
        SlowNumpyArray,
        name=remote_cls.name,
        reload_prefixes=[branch_prefix, username_prefix],
    )
    assert reloaded_cls.cpu_count() == remote_cls.cpu_count()

    compute_type = os.getenv("TEST_COMPUTE_TYPE")
    compute_prefix = f"{username_prefix}-{compute_type}"
    another_reloaded_cls = kt.cls(
        name=remote_cls.name,
        reload_prefixes=[branch_prefix, username_prefix, compute_prefix],
    )
    assert another_reloaded_cls.cpu_count() == remote_cls.cpu_count()


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_send_cls_to_compute_with_reload(remote_cls):
    import kubetorch as kt

    compute_type = os.getenv("TEST_COMPUTE_TYPE", "deployment")
    if compute_type == "ray":
        pytest.skip("Skipping cls reload tests for Ray compute type")

    service_name = remote_cls.service_name
    reloaded_cls = kt.cls(SlowNumpyArray, name=service_name).to(
        kt.Compute(
            cpus=".01",
            gpu_anti_affinity=True,
            launch_timeout=300,
        ),
        get_if_exists=True,
    )
    assert reloaded_cls.cpu_count() == remote_cls.cpu_count()


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_skip_cls_deploy_if_already_exists(remote_cls):
    import kubetorch as kt

    compute_type = os.getenv("TEST_COMPUTE_TYPE", "deployment")
    if compute_type == "ray":
        pytest.skip("Skipping if already exists tests for Ray compute type")

    name = remote_cls.name
    reloaded_cls = kt.cls(SlowNumpyArray, name=name).to(
        kt.Compute(
            cpus=".01",
            gpu_anti_affinity=True,
            launch_timeout=300,
        ),
        get_if_exists=True,
    )
    assert reloaded_cls.service_name == remote_cls.service_name
    assert reloaded_cls.compute.pods() == remote_cls.compute.pods()


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_fn_async_call(remote_fn):
    async def assert_async_call(fn, *args, **kwargs):
        coroutine = fn(*args, **kwargs)
        assert asyncio.iscoroutine(coroutine)
        result = await coroutine
        assert result == 3

    # use async_ flag in method call
    await assert_async_call(remote_fn, 1, 2, async_=True)

    # use async_ fn property
    try:
        remote_fn.async_ = True
        await assert_async_call(remote_fn, 1, 2)
    finally:
        remote_fn.async_ = False


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_cls_async_call(remote_cls):
    async def assert_async_call(fn, *args, **kwargs):
        coroutine = fn(*args, **kwargs)
        assert asyncio.iscoroutine(coroutine)
        result = await coroutine
        assert isinstance(result, int)

    # use async_ flag in method call
    await assert_async_call(remote_cls.cpu_count, async_=True)

    # use async_ cls property
    try:
        remote_cls.async_ = True
        await assert_async_call(remote_cls.cpu_count)
    finally:
        remote_cls.async_ = False


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_async_fn_call(remote_async_fn):
    # basic async function call
    coroutine = remote_async_fn(1, 2)
    assert asyncio.iscoroutine(coroutine)
    result = await coroutine
    assert result == 3

    # run multiple async function calls and assert that all start times are less than the earliest end time
    async def run_tasks():
        tasks = []
        for i in range(3):
            tasks.append(remote_async_fn(1, 2, return_times=True))
        results = await asyncio.gather(*tasks)
        return results

    results = await run_tasks()

    start_times = [r[0] for r in results]
    end_times = [r[1] for r in results]
    assert max(start_times) < min(end_times)


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_async_to():
    import kubetorch as kt

    from .conftest import get_compute
    from .utils import summer

    compute_type = os.getenv("TEST_COMPUTE_TYPE", "deployment")
    compute = get_compute(compute_type)

    name = f"{compute_type}-summer"
    fn = await kt.fn(summer, name=name).to_async(compute)

    # check running function normally
    assert fn(1, 2) == 3

    # check running function in async mode
    coroutine = fn(1, 2, async_=True)
    assert asyncio.iscoroutine(coroutine)
    result = await coroutine
    assert result == 3
