"""
BYO (Bring Your Own) Compute Tests

These tests verify that kubetorch can work with vanilla pods created directly
through the K8s API, without using kubetorch's manifest generation.

The key scenario is:
1. User deploys their own bare Pod via kubectl or K8s API (with kubetorch server image)
2. User creates kt.Compute(selector={"app": "my-pod"}) pointing to their pod
3. kt.fn().to(compute) registers the pool and creates a Service for routing
4. Function calls work correctly

This validates the BYO compute use case where users want full control over their
pod configuration while still using kubetorch's execution model. We specifically
use a bare Pod (not a Deployment) since Deployments are already a supported
resource type in kubetorch's provisioning path.
"""

import os
import time

import kubetorch as kt

import pytest
from kubetorch.provisioning.constants import DEFAULT_KT_SERVER_PORT

from .conftest import KUBETORCH_IMAGE

from .utils import get_hostname, log_n_messages, SlowNumpyArray, summer


@pytest.fixture(autouse=True, scope="module")
def setup_test_env():
    os.environ["KT_LAUNCH_TIMEOUT"] = "180"
    yield


@pytest.fixture(scope="module")
def byo_pod_compute():
    """Create a bare Pod via K8s API and return a Compute pointing to it.

    This fixture creates a single vanilla pod that can be reused across
    all tests in this module, then cleans it up at the end.

    For development/CI, it uploads local kubetorch code to the data store
    and the pod installs it on startup, ensuring tests run against local changes.
    """
    import subprocess
    from pathlib import Path

    from kubernetes import client, config

    # Load k8s config
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()

    core_v1 = client.CoreV1Api()

    pod_name = f"{kt.config.username}-byo-pod"
    namespace = kt.globals.config.namespace
    selector_labels = {"app": pod_name}

    # The service name is used for WebSocket registration AND must match what we use in kt.fn().to()
    # All tests must use this same service_name to properly route to this pod
    service_name = f"{kt.config.username}-byo-compute"

    # Upload local kubetorch code to data store for the pod to install
    kubetorch_src = Path(__file__).parent.parent  # kubetorch/python_client
    data_store_key = f"{service_name}/kubetorch"
    print(f"Uploading local kubetorch from {kubetorch_src} to data store key: {data_store_key}")
    result = subprocess.run(
        [
            "kt",
            "put",
            data_store_key,
            "-s",
            str(kubetorch_src),
            "-f",
            "-n",
            namespace,
            "--exclude",
            "*.pyc",
            "--exclude",
            "__pycache__",
            "--exclude",
            ".venv",
            "--exclude",
            "*.egg-info",
            "--exclude",
            ".git",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to upload kubetorch to data store: {result.stderr}")
    print(f"Uploaded kubetorch to data store: {result.stdout}")

    # Build the startup command:
    # 1. Use Python API to pull local kubetorch from data store (avoids kt get CLI bug) - not
    #    needed when working on stable.
    # 2. pip install to install it
    # 3. kubetorch server start with pool name
    startup_script = f"""
set -e
echo "Fetching local kubetorch from data store..."
python -c "import kubetorch as kt; kt.get('{data_store_key}', dest='/tmp', namespace='{namespace}')"
echo "Installing local kubetorch..."
export KT_LOG_LEVEL=DEBUG
ls /tmp/kubetorch
pip install /tmp/kubetorch/python_client
echo "Starting kubetorch server..."
exec kubetorch server start --pool {service_name}
"""

    # Create a bare Pod (not a Deployment) - this is specifically NOT a resource
    # type that kubetorch provisions, making it a true BYO test
    pod = client.V1Pod(
        api_version="v1",
        kind="Pod",
        metadata=client.V1ObjectMeta(
            name=pod_name,
            namespace=namespace,
            labels=selector_labels,
        ),
        spec=client.V1PodSpec(
            containers=[
                client.V1Container(
                    name="server",
                    image=KUBETORCH_IMAGE,
                    image_pull_policy="Always",
                    # Use shell to run multi-step startup
                    command=["/bin/bash", "-c", startup_script],
                    ports=[
                        client.V1ContainerPort(
                            container_port=DEFAULT_KT_SERVER_PORT,
                            name="http",
                        )
                    ],
                    resources=client.V1ResourceRequirements(requests={"cpu": "100m", "memory": "256Mi"}),
                )
            ],
            # Avoid GPU nodes for tests
            affinity=client.V1Affinity(
                node_affinity=client.V1NodeAffinity(
                    required_during_scheduling_ignored_during_execution=client.V1NodeSelector(
                        node_selector_terms=[
                            client.V1NodeSelectorTerm(
                                match_expressions=[
                                    client.V1NodeSelectorRequirement(
                                        key="nvidia.com/gpu.count",
                                        operator="DoesNotExist",
                                    )
                                ]
                            )
                        ]
                    )
                )
            ),
            restart_policy="Never",  # Bare pods don't restart
        ),
    )

    # Create or replace the pod
    try:
        core_v1.create_namespaced_pod(namespace=namespace, body=pod)
    except client.ApiException as e:
        if e.status == 409:  # Already exists - delete and recreate
            core_v1.delete_namespaced_pod(name=pod_name, namespace=namespace)
            # Wait for deletion
            for _ in range(30):
                try:
                    core_v1.read_namespaced_pod(name=pod_name, namespace=namespace)
                    time.sleep(1)
                except client.ApiException as del_e:
                    if del_e.status == 404:
                        break
            core_v1.create_namespaced_pod(namespace=namespace, body=pod)
        else:
            raise

    # Wait for pod to be ready
    label_selector = ",".join(f"{k}={v}" for k, v in selector_labels.items())
    for _ in range(90):  # 3 minutes timeout
        pods = core_v1.list_namespaced_pod(namespace=namespace, label_selector=label_selector)
        ready_pods = [
            p
            for p in pods.items
            if p.status.phase == "Running" and all(c.ready for c in (p.status.container_statuses or []))
        ]
        if ready_pods:
            break
        time.sleep(2)
    else:
        raise TimeoutError(f"Pod {pod_name} not ready after 180s")

    # We need numpy installed inside the pod
    img = kt.images.ubuntu().pip_install(["numpy"])

    # Create Compute with selector pointing to our pod
    compute = kt.Compute(selector=selector_labels, image=img)

    # Yield both compute and service_name - tests MUST use this service_name
    # in their kt.fn(..., name=service_name) calls for proper WebSocket routing
    yield compute, service_name

    # Cleanup
    try:
        core_v1.delete_namespaced_pod(name=pod_name, namespace=namespace)
    except client.ApiException:
        pass


@pytest.mark.level("minimal")
def test_byo_compute_fn(byo_pod_compute):
    """Test BYO compute with kt.fn - basic function execution."""
    compute, service_name = byo_pod_compute
    remote_fn = kt.fn(summer, name=service_name).to(compute)

    result = remote_fn(5, 10)
    assert result == 15, f"Expected 15, got {result}"

    # Call again to ensure subsequent calls work
    result2 = remote_fn(100, 200)
    assert result2 == 300, f"Expected 300, got {result2}"


@pytest.mark.level("minimal")
def test_byo_compute_cls(byo_pod_compute):
    """Test BYO compute with kt.cls - class instantiation and method calls."""
    compute, service_name = byo_pod_compute
    remote_cls = kt.cls(SlowNumpyArray, name=service_name).to(compute, init_args={"size": 5})

    result = remote_cls.print_and_log(2)
    assert "Hello from the cluster!" in result
    assert "[0. 0. 2. 0. 0.]" in result

    # Test classmethod
    home = remote_cls.home()
    assert home.startswith("/")


@pytest.mark.level("minimal")
def test_byo_compute_logging_function(byo_pod_compute):
    """Test BYO compute with a function that generates logs."""
    compute, service_name = byo_pod_compute
    remote_fn = kt.fn(log_n_messages, name=service_name).to(compute)

    msg = "BYO compute log test"
    n = 5
    result = remote_fn(msg=msg, n=n)
    assert result == f"{msg} was logged {n} times"

    # Call again to ensure consistent behavior
    msg2 = "Second log test"
    result2 = remote_fn(msg=msg2, n=3)
    assert result2 == f"{msg2} was logged 3 times"


@pytest.mark.level("minimal")
def test_byo_compute_hostname(byo_pod_compute):
    """Test BYO compute returns correct hostname."""
    compute, service_name = byo_pod_compute
    remote_fn = kt.fn(get_hostname, name=service_name).to(compute)

    hostname = remote_fn()
    # Hostname should be the pod name
    assert hostname, "Hostname should not be empty"
    assert "byo" in hostname.lower(), f"Expected 'byo' in hostname: {hostname}"


@pytest.mark.level("minimal")
def test_byo_compute_run_bash(byo_pod_compute):
    """Test BYO compute run_bash functionality."""
    compute, service_name = byo_pod_compute
    # Deploy a simple function to get a reference to the compute
    remote_fn = kt.fn(summer, name=service_name).to(compute)

    # Verify function works
    assert remote_fn(1, 2) == 3

    # Run bash command via compute
    result = remote_fn.compute.run_bash("echo 'Hello from BYO pod' && hostname")
    output = "".join(line[1] for line in result if len(line) > 1)
    assert "Hello from BYO pod" in output

    # Run a more complex command
    result2 = remote_fn.compute.run_bash("python3 -c 'import sys; print(sys.version)'")
    output2 = "".join(line[1] for line in result2 if len(line) > 1)
    assert "3." in output2


@pytest.mark.level("minimal")
def test_byo_compute_redeploy_different_callable(byo_pod_compute):
    """Test redeploying different callables to the same BYO pod.

    This is a key BYO use case - the same pod serves different functions
    as the user iterates on their code.
    """
    compute, service_name = byo_pod_compute

    # Deploy summer function
    summer_fn = kt.fn(summer, name=service_name).to(compute)
    assert summer_fn(10, 20) == 30

    # Redeploy with a different function (same pool name)
    hostname_fn = kt.fn(get_hostname, name=service_name).to(compute)
    hostname = hostname_fn()
    assert hostname
    assert "byo" in hostname.lower()

    # Redeploy with log function
    log_fn = kt.fn(log_n_messages, name=service_name).to(compute)
    result = log_fn(msg="redeploy test", n=2)
    assert "was logged 2 times" in result
