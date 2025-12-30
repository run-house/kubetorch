import os

import pytest

from .utils import create_random_name_prefix, service_deployer, service_deployer_with_raycluster


@pytest.fixture(autouse=True, scope="session")
def setup_test_env():
    # Keep the launch timeout low for this test suite, unless overridden (ex: for GPU tests)
    os.environ["KT_LAUNCH_TIMEOUT"] = "150"
    os.environ["KT_STREAM"] = "true"
    yield


@pytest.mark.level("minimal")
def test_run_to_on_pod():
    import kubetorch as kt

    prefix = create_random_name_prefix()
    name = f"{prefix}-to-to"
    child_name = f"{prefix}-to-to-child"

    deployer_fn = kt.fn(service_deployer, name).to(
        kt.Compute(
            cpus=".1",
            image=kt.images.Debian(),
            gpu_anti_affinity=True,
        ).autoscale(min_replicas=1)
    )

    # Check that the function returns the correct value
    assert deployer_fn(child_name) == 3


@pytest.mark.level("minimal")
def test_run_to_on_pod_with_raycluster():
    import kubetorch as kt

    prefix = create_random_name_prefix()
    name = f"{prefix}-to-to"
    child_name = f"{prefix}-to-to-child"

    deployer_fn = kt.fn(service_deployer_with_raycluster, name).to(
        kt.Compute(cpus=".1", image=kt.images.Debian(), gpu_anti_affinity=True, launch_timeout=450)
    )

    # Check that the function returns the correct value
    assert deployer_fn(child_name) == 3
