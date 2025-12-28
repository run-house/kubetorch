import os

import kubetorch.serving.constants as serving_constants
import pytest

from .utils import create_random_name_prefix, simple_summer


@pytest.fixture(autouse=True, scope="session")
def setup_test_env():
    os.environ["KT_GPU_ANTI_AFFINITY"] = "True"
    # Keep the launch timeout low for this test suite, unless overridden (ex: for GPU tests)
    os.environ["KT_LAUNCH_TIMEOUT"] = "120"
    yield


@pytest.mark.level("minimal")
def test_autodown_annotation():
    import kubetorch as kt

    controller = kt.globals.controller_client()
    name = f"{create_random_name_prefix()}-autodown-annotation"
    namespace = kt.globals.config.namespace
    inactivity_ttl = "10m"

    remote_fn = kt.fn(simple_summer, name=name).to(
        kt.Compute(cpus=".01", inactivity_ttl=inactivity_ttl, namespace=namespace).autoscale(min_scale=1)
    )

    assert remote_fn(1, 2) == 3

    # Check that the service was created
    service = controller.get_service(name=remote_fn.service_name, namespace=namespace)
    assert service

    # Check that the service has the autodown annotation
    assert service["metadata"]["labels"][serving_constants.KT_MODULE_LABEL] is not None
    assert service["metadata"]["annotations"][serving_constants.INACTIVITY_TTL_ANNOTATION] == inactivity_ttl

    # Check that the namespace is in the watch namespaces
    cronjob_configmap = controller.get_config_map(
        name=serving_constants.TTL_CONTROLLER_CONFIGMAP_NAME,
        namespace=serving_constants.KUBETORCH_NAMESPACE,
    )
    assert namespace in cronjob_configmap["data"]["WATCH_NAMESPACES"].split(",")


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
    container = next((x for x in pod["spec"]["containers"] if x["name"] == "kubetorch"), None)
    kt_otel_env = next((env for env in container["env"] if env["name"] == "KT_METRICS_ENABLED"), None)
    assert kt_otel_env["value"] == "True"

    # Check that the service has the autodown annotation
    assert service["metadata"]["labels"][serving_constants.KT_MODULE_LABEL] is not None
    assert service["metadata"]["annotations"][serving_constants.INACTIVITY_TTL_ANNOTATION] == inactivity_ttl

    # Check that the new app and deployment-id labels exist on the pod
    pod_labels = pod["metadata"]["labels"]
    assert pod_labels.get(serving_constants.KT_APP_LABEL) == remote_fn.service_name
    assert pod_labels.get(serving_constants.KT_DEPLOYMENT_ID_LABEL) is not None
    assert pod_labels.get(serving_constants.KT_DEPLOYMENT_ID_LABEL).startswith(remote_fn.service_name)

    # Check that the Deployment also has the new labels
    deployment = controller.get_deployment(name=remote_fn.service_name, namespace=namespace)
    deployment_labels = deployment["metadata"]["labels"]
    assert deployment_labels.get(serving_constants.KT_APP_LABEL) == remote_fn.service_name
    assert deployment_labels.get(serving_constants.KT_DEPLOYMENT_ID_LABEL) is not None
    assert deployment_labels.get(serving_constants.KT_DEPLOYMENT_ID_LABEL).startswith(remote_fn.service_name)

    # Check that the namespace is in the watch namespaces
    cronjob_configmap = controller.get_config_map(
        name=serving_constants.TTL_CONTROLLER_CONFIGMAP_NAME,
        namespace=serving_constants.KUBETORCH_NAMESPACE,
    )
    assert namespace in cronjob_configmap["data"]["WATCH_NAMESPACES"].split(",")


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
    container = next((x for x in pod["spec"]["containers"] if x["name"] == "kubetorch"), None)
    kt_otel_env = next((env for env in container["env"] if env["name"] == "KT_METRICS_ENABLED"), None)
    assert kt_otel_env["value"] == "True"

    # Check that the service has the autodown annotation
    assert service["metadata"]["labels"][serving_constants.KT_MODULE_LABEL] is not None
    assert service["metadata"]["annotations"][serving_constants.INACTIVITY_TTL_ANNOTATION] == inactivity_ttl

    # Check that the new app and deployment-id labels exist on the pod
    pod_labels = pod["metadata"]["labels"]
    assert pod_labels.get(serving_constants.KT_APP_LABEL) == remote_fn.service_name
    assert pod_labels.get(serving_constants.KT_DEPLOYMENT_ID_LABEL) is not None
    assert pod_labels.get(serving_constants.KT_DEPLOYMENT_ID_LABEL).startswith(remote_fn.service_name)


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
    container = next((x for x in pod["spec"]["containers"] if x["name"] == "kubetorch"), None)
    kt_otel_env = next((env for env in container["env"] if env["name"] == "KT_METRICS_ENABLED"), None)
    assert kt_otel_env["value"] == "True"

    # Check that the service has the autodown annotation
    assert service["metadata"]["labels"][serving_constants.KT_MODULE_LABEL] is not None
    assert service["metadata"]["labels"]["KT_METRICS_ENABLED"] == "True"
    assert service["metadata"]["annotations"][serving_constants.INACTIVITY_TTL_ANNOTATION] == inactivity_ttl
