import os

# Mimic CI for this test suite even locally, to ensure that
# resources are created with the branch name prefix
os.environ["CI"] = "true"

import os

import pytest

from .utils import get_test_fn_name, python_version_and_path


@pytest.fixture(autouse=True, scope="session")
def setup_test_env():
    # Keep the launch timeout low for this test suite, unless overridden (ex: for GPU tests)
    os.environ["KT_LAUNCH_TIMEOUT"] = "150"
    os.environ["KT_HTTP_HEALTH_TIMEOUT"] = "120"
    yield


@pytest.mark.level("minimal")
def test_python_path_uv():
    import os

    # if you run the test on a EKS cluster, use the following image_id:
    # #image_id="your-account.dkr.ecr.us-east-1.amazonaws.com/test-python-exe:latest",

    import kubetorch as kt

    python_path = "/kt_app/.venv/bin/python"
    # Use the actual GCP project ID from environment or fallback to placeholder
    gcp_project = os.getenv("GCP_PROJECT_ID", "runhouse-test")
    remote_fn = kt.fn(python_version_and_path, name=get_test_fn_name()).to(
        kt.Compute(
            cpus=".01",
            image=kt.Image(
                image_id=f"us-east1-docker.pkg.dev/{gcp_project}/kubetorch-images/test-python-exe",
                python_path=python_path,
            ).pip_install(["pytest"]),
            gpu_anti_affinity=True,
            launch_timeout=300,
        )
    )

    remote_python_version, remote_path = remote_fn()
    assert "3.9" in remote_python_version
    assert remote_path.startswith(os.path.dirname(python_path))


@pytest.mark.level("minimal")
def test_python_path_conda():
    import os

    # if you run the test on a EKS cluster, use the following image_id:
    # image_id="your-account.dkr.ecr.us-east-1.amazonaws.com/test-conda:latest"
    import kubetorch as kt

    python_path = "/opt/conda/envs/py39/bin/python"
    # Use the actual GCP project ID from environment or fallback to placeholder
    gcp_project = os.getenv("GCP_PROJECT_ID", "runhouse-test")
    remote_fn = kt.fn(python_version_and_path, name=get_test_fn_name()).to(
        kt.Compute(
            cpus=".01",
            image=kt.Image(
                image_id=f"us-east1-docker.pkg.dev/{gcp_project}/kubetorch-images/test-conda:latest",
                python_path=python_path,
            ).pip_install(["pytest"]),
            gpu_anti_affinity=True,
            launch_timeout=300,
        )
    )

    remote_python_version, remote_path = remote_fn()
    assert "3.9" in remote_python_version
    assert remote_path.startswith(os.path.dirname(python_path))
