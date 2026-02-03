import os
from unittest.mock import MagicMock, patch

import httpx
import kubetorch.provisioning.constants as provisioning_constants
import pytest
from kubernetes import client as k8s_client, config as k8s_config

from .utils import create_random_name_prefix, simple_summer


@pytest.fixture(autouse=True)
def setup_test_env(request):
    """Only set env vars for minimal-level tests that actually deploy to cluster."""
    marker = request.node.get_closest_marker("level")
    if marker and marker.args[0] == "minimal":
        old_gpu_anti_affinity = os.environ.get("KT_GPU_ANTI_AFFINITY")
        old_launch_timeout = os.environ.get("KT_LAUNCH_TIMEOUT")
        os.environ["KT_GPU_ANTI_AFFINITY"] = "True"
        os.environ["KT_LAUNCH_TIMEOUT"] = "150"
        yield
        # Restore original values
        if old_gpu_anti_affinity is None:
            os.environ.pop("KT_GPU_ANTI_AFFINITY", None)
        else:
            os.environ["KT_GPU_ANTI_AFFINITY"] = old_gpu_anti_affinity
        if old_launch_timeout is None:
            os.environ.pop("KT_LAUNCH_TIMEOUT", None)
        else:
            os.environ["KT_LAUNCH_TIMEOUT"] = old_launch_timeout
    else:
        yield


# =============================================================================
# Unit Tests for auto_termination payload
# =============================================================================


@pytest.mark.level("unit")
def test_auto_termination_payload_construction():
    """Test that auto_termination dict is correctly built and passed to controller.deploy()"""
    from kubetorch.provisioning.service_manager import ServiceManager

    # Mock the controller client
    mock_controller = MagicMock()
    mock_controller.deploy.return_value = {
        "apply_status": "success",
        "workload_status": "success",
        "service_url": "http://test-service.default.svc.cluster.local",
        "resource": {"metadata": {"name": "test-service"}},
    }

    # Create a service manager with mocked controller
    with patch("kubetorch.provisioning.service_manager.globals") as mock_globals:
        mock_globals.config.username = "test-user"
        mock_globals.controller_client.return_value = mock_controller

        manager = ServiceManager.__new__(ServiceManager)
        manager.namespace = "default"
        manager.resource_type = "deployment"
        manager.config = {}

        # Call _apply_and_register_workload with inactivity_ttl in runtime_config
        runtime_config = {
            "log_streaming_enabled": True,
            "metrics_enabled": True,
            "inactivity_ttl": "30m",
        }

        # Mock required methods
        manager.pod_spec = MagicMock(return_value={"containers": [{"ports": [{"containerPort": 32300}]}]})
        manager._resolve_service_config = MagicMock(return_value=None)
        manager._load_workload_metadata = MagicMock(return_value={"username": "test-user"})

        manager._apply_and_register_workload(
            manifest={"metadata": {"labels": {}, "annotations": {}}, "spec": {}},
            service_name="test-service",
            runtime_config=runtime_config,
        )

        # Verify auto_termination was passed to deploy()
        deploy_call = mock_controller.deploy.call_args
        assert deploy_call is not None

        # Check that auto_termination kwarg was passed with correct structure
        _, kwargs = deploy_call
        assert "auto_termination" in kwargs
        assert kwargs["auto_termination"] == {"inactivityTtl": "30m"}


@pytest.mark.level("unit")
def test_auto_termination_not_set_when_no_ttl():
    """Test that auto_termination is None when inactivity_ttl is not provided"""
    from kubetorch.provisioning.service_manager import ServiceManager

    # Mock the controller client
    mock_controller = MagicMock()
    mock_controller.deploy.return_value = {
        "apply_status": "success",
        "workload_status": "success",
        "service_url": "http://test-service.default.svc.cluster.local",
        "resource": {"metadata": {"name": "test-service"}},
    }

    # Create a service manager with mocked controller
    with patch("kubetorch.provisioning.service_manager.globals") as mock_globals:
        mock_globals.config.username = "test-user"
        mock_globals.controller_client.return_value = mock_controller

        manager = ServiceManager.__new__(ServiceManager)
        manager.namespace = "default"
        manager.resource_type = "deployment"
        manager.config = {}

        # Call without inactivity_ttl
        runtime_config = {
            "log_streaming_enabled": True,
            "metrics_enabled": True,
        }

        manager.pod_spec = MagicMock(return_value={"containers": [{"ports": [{"containerPort": 32300}]}]})
        manager._resolve_service_config = MagicMock(return_value=None)
        manager._load_workload_metadata = MagicMock(return_value={"username": "test-user"})

        manager._apply_and_register_workload(
            manifest={"metadata": {"labels": {}, "annotations": {}}, "spec": {}},
            service_name="test-service",
            runtime_config=runtime_config,
        )

        # Verify auto_termination was not passed (should be filtered out as None)
        deploy_call = mock_controller.deploy.call_args
        _, kwargs = deploy_call
        # auto_termination should either not be in kwargs or be None
        assert kwargs.get("auto_termination") is None


# =============================================================================
# Integration Tests
#
# These tests verify backward compatibility via annotations on K8s resources.
# Once the controller is updated to write autoTermination to the CRD spec,
# add tests to verify:
#   1. KubetorchWorkload has spec.autoTermination.inactivityTtl set
#   2. KubetorchWorkload has label kubetorch.com/inactivity-ttl with the TTL value
# =============================================================================


@pytest.mark.level("minimal")
def test_autodown_annotation_and_metrics():
    import kubetorch as kt

    name = f"{create_random_name_prefix()}-autodown-annotation"
    namespace = kt.globals.config.namespace
    inactivity_ttl = "10m"

    remote_fn = kt.fn(simple_summer, name=name).to(
        kt.Compute(cpus=".01", inactivity_ttl=inactivity_ttl, namespace=namespace).autoscale(min_scale=1)
    )

    assert remote_fn(1, 2) == 3

    # Use controller client to verify resource exists in discovery
    controller = kt.globals.controller_client()
    workloads = controller.discover_resources(namespace=namespace, name_filter=remote_fn.service_name)
    # For Knative, the K8s kind is "Service" (not "KnativeService")
    knative_services = [r for r in workloads.get("workloads", []) if r.get("kind") == "Service"]
    assert knative_services, f"No Knative service found for {remote_fn.service_name}"

    # Fetch the full Knative service object via k8s API to get metadata
    try:
        k8s_config.load_incluster_config()
    except k8s_config.ConfigException:
        k8s_config.load_kube_config()

    custom_api = k8s_client.CustomObjectsApi()
    knative_service = custom_api.get_namespaced_custom_object(
        group="serving.knative.dev",
        version="v1",
        namespace=namespace,
        plural="services",
        name=remote_fn.service_name,
    )

    # Check that the Knative service has the autodown annotation
    assert knative_service["metadata"]["labels"][provisioning_constants.KT_MODULE_LABEL] is not None
    assert (
        knative_service["metadata"]["annotations"][provisioning_constants.INACTIVITY_TTL_ANNOTATION] == inactivity_ttl
    )

    # Ping the /metrics endpoint and check that the metrics are being pushed
    response = httpx.get(f"{remote_fn.base_endpoint}/metrics")
    assert response.status_code == 200
    assert "kt_heartbeat_sent" in response.text


@pytest.mark.level("minimal")
def test_autodown_deployment():
    import kubetorch as kt

    controller = kt.globals.controller_client()
    name = f"{create_random_name_prefix()}-autodown-deployment"
    namespace = kt.globals.config.namespace
    inactivity_ttl = "10m"

    remote_fn = kt.fn(simple_summer, name=name).to(
        kt.Compute(cpus=".01", inactivity_ttl=inactivity_ttl, namespace=namespace)
    )

    assert remote_fn(1, 2) == 3

    # Check that the service was created
    service = controller.get_service(name=remote_fn.service_name, namespace=namespace)
    assert service

    # Get the env vars on the service pod and check that KT_METRICS_ENABLED is True
    pods_response = controller.list_pods(
        namespace=namespace,
        label_selector=f"kubetorch.com/service={remote_fn.service_name}",
    )
    pods = pods_response.get("items", [])
    assert pods

    pod = pods[0]
    # Try to find kubetorch container, fall back to first container
    container = next((x for x in pod["spec"]["containers"] if x["name"] == "kubetorch"), None)
    if container is None:
        container = pod["spec"]["containers"][0]
    kt_otel_env = next((env for env in container.get("env", []) if env["name"] == "KT_METRICS_ENABLED"), None)
    # KT_METRICS_ENABLED defaults to True if not explicitly set, so we check it's not set to False
    assert kt_otel_env is None or kt_otel_env["value"] == "True"

    # Check that the service has the autodown annotation
    assert service["metadata"]["labels"][provisioning_constants.KT_MODULE_LABEL] is not None
    assert service["metadata"]["annotations"][provisioning_constants.INACTIVITY_TTL_ANNOTATION] == inactivity_ttl

    # Check that the new app and deployment-id labels exist on the pod
    pod_labels = pod["metadata"]["labels"]
    assert pod_labels.get(provisioning_constants.KT_APP_LABEL) == remote_fn.service_name

    # Check that the Deployment also has the new labels
    deployment = controller.get_deployment(name=remote_fn.service_name, namespace=namespace)
    deployment_labels = deployment["metadata"]["labels"]
    assert deployment_labels.get(provisioning_constants.KT_APP_LABEL) == remote_fn.service_name


@pytest.mark.level("minimal")
def test_autodown_raycluster():
    import kubetorch as kt

    controller = kt.globals.controller_client()
    name = f"{create_random_name_prefix()}-autodown-raycluster"
    namespace = kt.globals.config.namespace
    inactivity_ttl = "10m"

    remote_fn = kt.fn(simple_summer, name=name).to(
        kt.Compute(
            cpus="2",
            memory="3Gi",
            launch_timeout=600,
            gpu_anti_affinity=True,
            image=kt.images.Ray(),
            inactivity_ttl=inactivity_ttl,
            namespace=namespace,
        ).distribute("ray", workers=2)
    )

    assert remote_fn(1, 2) == 3

    # Check that the service was created
    service = controller.get_service(name=remote_fn.service_name, namespace=namespace)
    assert service

    # Get the env vars on the service pod and check that KT_METRICS_ENABLED is True
    pods_response = controller.list_pods(
        namespace=namespace,
        label_selector=f"kubetorch.com/service={remote_fn.service_name}",
    )
    pods = pods_response.get("items", [])
    assert pods

    pod = pods[0]
    # Try to find kubetorch container, fall back to first container (Ray pods may have different names)
    container = next((x for x in pod["spec"]["containers"] if x["name"] == "kubetorch"), None)
    if container is None:
        container = pod["spec"]["containers"][0]
    kt_otel_env = next((env for env in container.get("env", []) if env["name"] == "KT_METRICS_ENABLED"), None)
    # KT_METRICS_ENABLED defaults to True if not explicitly set, so we check it's not set to False
    assert kt_otel_env is None or kt_otel_env["value"] == "True"

    # Check that the service has the autodown annotation
    assert service["metadata"]["labels"][provisioning_constants.KT_MODULE_LABEL] is not None
    assert service["metadata"]["annotations"][provisioning_constants.INACTIVITY_TTL_ANNOTATION] == inactivity_ttl

    # Check that the new app and deployment-id labels exist on the pod
    pod_labels = pod["metadata"]["labels"]
    assert pod_labels.get(provisioning_constants.KT_APP_LABEL) == remote_fn.service_name


@pytest.mark.skip("Long running test, skipping for now")
@pytest.mark.level("minimal")
def test_autodown_custom_image():
    import kubetorch as kt

    controller = kt.globals.controller_client()
    name = f"{create_random_name_prefix()}-autodown-custom-img"
    namespace = kt.globals.config.namespace
    inactivity_ttl = "10m"

    remote_fn = kt.fn(simple_summer, name=name).to(
        kt.Compute(
            gpus=1,
            inactivity_ttl=inactivity_ttl,
            namespace=namespace,
            launch_timeout=600,
            image=kt.Image(image_id="nvcr.io/nvidia/pytorch:23.10-py3"),
        )
    )

    assert remote_fn(1, 2) == 3

    # Check that the service was created
    service = controller.get_service(name=remote_fn.service_name, namespace=namespace)
    assert service

    # Get the env vars on the service pod and check that KT_METRICS_ENABLED is True
    pods_response = controller.list_pods(
        namespace=namespace,
        label_selector=f"kubetorch.com/service={remote_fn.service_name}",
    )
    pods = pods_response.get("items", [])
    assert pods

    pod = pods[0]
    # Try to find kubetorch container, fall back to first container
    container = next((x for x in pod["spec"]["containers"] if x["name"] == "kubetorch"), None)
    if container is None:
        container = pod["spec"]["containers"][0]
    kt_otel_env = next((env for env in container.get("env", []) if env["name"] == "KT_METRICS_ENABLED"), None)
    # KT_METRICS_ENABLED defaults to True if not explicitly set, so we check it's not set to False
    assert kt_otel_env is None or kt_otel_env["value"] == "True"

    # Check that the service has the autodown annotation
    assert service["metadata"]["labels"][provisioning_constants.KT_MODULE_LABEL] is not None
    assert service["metadata"]["labels"]["KT_METRICS_ENABLED"] == "True"
    assert service["metadata"]["annotations"][provisioning_constants.INACTIVITY_TTL_ANNOTATION] == inactivity_ttl
