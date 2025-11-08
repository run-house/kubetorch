import os

# Mimic CI for this test suite even locally, to ensure that
# resources are created with the branch name prefix
os.environ["CI"] = "true"

import subprocess
import time

import pytest

from .utils import get_cuda_version, get_test_fn_name, SlowNumpyArray


@pytest.mark.gpu_test
@pytest.mark.level("minimal")
def test_fn_basic_gpu_support():
    import kubetorch as kt

    gpu = kt.Compute(
        cpus=".1",
        gpus="1",
        image=kt.images.pytorch(),
        env_vars={"OMP_NUM_THREADS": 1},
        launch_timeout=600,
    )

    # Note: in runhouse-k8s cluster we have a node group with an A10G
    remote_fn = kt.fn(get_cuda_version, name=get_test_fn_name()).to(gpu)
    resp = remote_fn()
    assert "12" in resp
    remote_fn.teardown()


@pytest.mark.gpu_test
@pytest.mark.level("minimal")
def test_fn_sync_with_providing_gpu_type():
    import kubetorch as kt

    # Note: in runhouse-k8s cluster we have a node group with nvidia-l4
    remote_fn = kt.fn(get_cuda_version, name=get_test_fn_name()).to(
        kt.Compute(
            cpus=".1",
            gpus="1",
            node_selector={"cloud.google.com/gke-accelerator": "nvidia-l4"},
            image=kt.images.pytorch(),
            env_vars={"OMP_NUM_THREADS": 1},
            launch_timeout=600,
        )
    )
    resp = remote_fn()
    assert "12" in resp
    remote_fn.teardown()


@pytest.mark.gpu_test
@pytest.mark.level("minimal")
def test_fn_sync_on_gpu_with_autoscaling():
    import re

    import kubetorch as kt

    gpu_autoscale = kt.Compute(
        cpus=".1",
        gpus="1",
        image=kt.images.pytorch(),
        env_vars={"OMP_NUM_THREADS": 1},
        launch_timeout=600,
    ).autoscale(min_scale=2, initial_scale=2)

    remote_fn = kt.fn(get_cuda_version, name="fn_sync_gpu_autoscale").to(gpu_autoscale)

    num_requests = 20
    for _ in range(num_requests):
        assert re.fullmatch(r"12\..+", remote_fn())
        time.sleep(0.5)

    pod_names = remote_fn.compute.pod_names()
    assert len(pod_names) >= 2

    # Check logs of each pod to confirm requests were routed to them at least once
    for pod_name in pod_names:
        resp = subprocess.run(["kubectl", "logs", pod_name], capture_output=True, text=True)
        num_requests = resp.stdout.count("POST /get_cuda_version")
        assert num_requests > 0, f"Pod {pod_name} received no requests"

    remote_fn.teardown()


@pytest.mark.gpu_test
@pytest.mark.level("minimal")
def test_fn_sync_with_unsupported_gpu_type():
    import kubetorch as kt

    # Note: No runtime class for A100 configured on the cluster, so this should fail immediately

    with pytest.raises(kt.ResourceNotAvailableError):
        remote_fn = kt.fn(get_cuda_version, name=get_test_fn_name()).to(
            kt.Compute(
                cpus=".1",
                gpus="1",
                node_selector={"cloud.google.com/gke-accelerator": "nvidia-100"},
                image=kt.images.pytorch(),
                env_vars={"OMP_NUM_THREADS": 1},
            )
        )
        resp = remote_fn()
        assert "12" in resp


@pytest.mark.gpu_test
@pytest.mark.level("minimal")
def test_fn_sync_with_invalid_gpu_count():
    import kubetorch as kt

    with pytest.raises(ValueError):
        kt.fn(get_cuda_version).to(
            kt.Compute(
                cpus=".1",
                gpus="A10G:1",
            )
        )

    with pytest.raises(ValueError):
        kt.fn(get_cuda_version).to(
            kt.Compute(
                cpus=".1",
                gpus="0.5",
            )
        )

    with pytest.raises(ValueError):
        kt.fn(get_cuda_version).to(
            kt.Compute(
                cpus=".1",
                gpus="T4",
            )
        )


@pytest.mark.gpu_test
@pytest.mark.level("minimal")
def test_invalid_gpu_type():
    import kubetorch as kt

    with pytest.raises(ValueError):
        remote_cls = kt.cls(SlowNumpyArray, name=get_test_fn_name()).to(
            kt.Compute(
                cpus=".1",
                gpus="nonexistent-gpu-type",
                image=kt.images.pytorch(),
                gpu_anti_affinity=True,
            ),
            init_args={"size": 10},
        )
        remote_cls.print_and_log(1)
