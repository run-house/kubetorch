import copy
import os

import pytest

from .utils import summer


def _get_basic_manifest(kind: str):
    """Generate a minimal manifest for the given kind with test values."""
    base_metadata = {
        "name": "",
        "namespace": "default",
        "labels": {"test-label": "test-app"},
        "annotations": {"test-annotation": "original-value"},
    }

    if kind == "Deployment":
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": base_metadata,
            "spec": {
                "replicas": 2,  # Initial value to be overridden
                "selector": {"matchLabels": {"test-label": "test-app"}},
                "template": {
                    "metadata": {"labels": {"test-label": "test-app"}},
                    "spec": {"containers": []},
                },
            },
        }
    elif kind == "Service":  # Knative
        return {
            "apiVersion": "serving.knative.dev/v1",
            "kind": "Service",
            "metadata": base_metadata,
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "autoscaling.knative.dev/min-scale": "1",
                            "autoscaling.knative.dev/max-scale": "10",
                        },
                        "labels": {"test-label": "test-app"},
                    },
                    "spec": {"containers": []},
                },
            },
        }
    elif kind == "RayCluster":
        return {
            "apiVersion": "ray.io/v1",
            "kind": "RayCluster",
            "metadata": base_metadata,
            "spec": {
                "headGroupSpec": {
                    "serviceType": "ClusterIP",
                    "rayStartParams": {"dashboard-host": "0.0.0.0"},
                    "template": {"spec": {"containers": []}},
                },
                "workerGroupSpecs": [
                    {
                        "replicas": 1,
                        "minReplicas": 1,
                        "maxReplicas": 10,
                        "groupName": "small-group",
                        "rayStartParams": {},
                        "template": {"spec": {"containers": []}},
                    }
                ],
            },
        }
    else:
        raise ValueError(f"Unknown manifest kind: {kind}")


@pytest.fixture(autouse=True, scope="session")
def setup_test_env():
    os.environ["KT_LAUNCH_TIMEOUT"] = "300"
    yield


@pytest.mark.level("minimal")
@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["Deployment", "Service", "RayCluster"])
async def test_byo_manifest_with_overrides(kind):
    """Test BYO manifest with comprehensive kwargs overrides and containers."""
    import kubetorch as kt

    # Get manifest with test values already set (deep copy to avoid modifying the original)
    test_manifest = copy.deepcopy(_get_basic_manifest(kind))

    # Add a container with some initial values
    container = {
        "name": "user-container",
        "image": "user-image:latest",
        "resources": {
            "requests": {
                "cpu": "0.3",
                "memory": "512Mi",
            },
        },
        "env": [
            {"name": "ORIGINAL_ENV", "value": "original_value"},
            {"name": "CONTAINER_ENV", "value": "container_value"},
        ],
    }

    # Add container to the appropriate location based on manifest type
    if test_manifest["kind"] == "Deployment":
        test_manifest["spec"]["template"]["spec"]["containers"] = [container]
    elif test_manifest["kind"] == "Service":  # Knative
        test_manifest["spec"]["template"]["spec"]["containers"] = [container]
    elif test_manifest["kind"] == "RayCluster":
        test_manifest["spec"]["headGroupSpec"]["template"]["spec"]["containers"] = [container]
        test_manifest["spec"]["workerGroupSpecs"][0]["template"]["spec"]["containers"] = [container]

    image_type = "Ray" if kind == "RayCluster" else "Debian"
    image = getattr(kt.images, image_type)()

    # Create compute with comprehensive overrides
    compute = kt.Compute(
        manifest=test_manifest,
        cpus="0.5",
        memory="2Gi",
        replicas=3,
        labels={"custom-label": "custom-value"},
        annotations={"test-annotation": "overridden-value"},
        env_vars={"TEST_ENV": "test_value", "ORIGINAL_ENV": "overridden_value"},
        image=image,
        gpu_anti_affinity=True,
    )
    service_name = f"{kt.config.username}-byo-{kind.lower()}"
    compute.service_name = service_name

    # Verify overrides are applied using compute properties
    assert compute.cpus == "0.5"
    assert compute.memory == "2Gi"
    assert compute.replicas == 3
    assert compute.labels["custom-label"] == "custom-value"
    assert compute.annotations["test-annotation"] == "overridden-value"
    assert compute.env_vars["TEST_ENV"] == "test_value"
    assert compute.server_image == image.image_id
    assert compute.gpu_anti_affinity is True
    assert compute.env_vars["ORIGINAL_ENV"] == "overridden_value"
    assert compute.env_vars["CONTAINER_ENV"] == "container_value"

    # Deploy and test function
    fn = await kt.fn(summer).to_async(compute)
    assert fn.service_name == service_name

    result = fn(5, 10)
    assert result == 15


@pytest.mark.level("minimal")
def test_byo_manifest_extracts_values():
    """Test that values are extracted from manifest when not provided as kwargs."""
    import kubetorch as kt

    # Create manifest with annotations
    manifest = copy.deepcopy(_get_basic_manifest("Deployment"))
    manifest["metadata"]["annotations"]["kubetorch.com/inactivity-ttl"] = "1h"

    compute = kt.Compute(
        manifest=manifest,
        cpus="0.3",
    )

    # Verify extracted values are preserved in annotations
    assert compute.inactivity_ttl == "1h"
