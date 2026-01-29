import os

import httpx
import kubetorch.provisioning.constants as provisioning_constants
import pytest
from kubernetes import client as k8s_client, config as k8s_config

from .utils import create_random_name_prefix, simple_summer


@pytest.fixture(autouse=True, scope="session")
def setup_test_env():
    os.environ["KT_GPU_ANTI_AFFINITY"] = "True"
    # Keep the launch timeout low for this test suite, unless overridden (ex: for GPU tests)
    os.environ["KT_LAUNCH_TIMEOUT"] = "150"
    os.environ["KT_HTTP_HEALTH_TIMEOUT"] = "120"
    yield


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
    knative_services = [
        r
        for r in workloads.get("workloads", [])
        if r.get("api_version") == "serving.knative.dev/v1" and r.get("kind") == "KnativeService"
    ]
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
