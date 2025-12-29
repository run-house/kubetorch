import copy
import os

import kubetorch.globals
import kubetorch.serving.constants as serving_constants

import pytest
from kubetorch.utils import http_not_found

from .assets.torch_ddp.torch_ddp import torch_ddp
from .utils import summer

QUEUE_LABEL = serving_constants.KUEUE_QUEUE_NAME_LABEL


def _kueue_available() -> bool:
    """Check if Kueue is installed in the cluster."""
    try:
        from kubernetes import client, config

        config.load_kube_config()
        api = client.ApiextensionsV1Api()
        crds = api.list_custom_resource_definition()
        return any("kueue.x-k8s.io" in crd.metadata.name for crd in crds.items)
    except Exception:
        return False


TRAINING_JOB_CONFIG = {
    "PyTorchJob": {
        "replica_specs_key": "pytorchReplicaSpecs",
        "primary_replica": "Master",
        "container_name": "pytorch",
    },
    "TFJob": {"replica_specs_key": "tfReplicaSpecs", "primary_replica": "Chief", "container_name": "tensorflow"},
    "MXJob": {"replica_specs_key": "mxReplicaSpecs", "primary_replica": "Scheduler", "container_name": "mxnet"},
    "XGBoostJob": {
        "replica_specs_key": "xgbReplicaSpecs",
        "primary_replica": "Master",
        "container_name": "xgboost",
    },
}


def _get_basic_manifest(kind: str):
    """Generate a minimal manifest for the given kind with test values."""
    base_metadata = {
        "name": "",
        "namespace": kubetorch.globals.config.namespace,
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
    elif kind in TRAINING_JOB_CONFIG:
        config = TRAINING_JOB_CONFIG[kind]
        primary_replica_spec = {
            "replicas": 1,
            "restartPolicy": "OnFailure",
            "template": {"spec": {"containers": []}},
        }
        worker_replica_spec = {
            "replicas": 2,
            "restartPolicy": "OnFailure",
            "template": {"spec": {"containers": []}},
        }
        spec = {
            config["replica_specs_key"]: {
                config["primary_replica"]: primary_replica_spec,
                "Worker": worker_replica_spec,
            },
        }
        # MXJob requires jobMode field
        if kind == "MXJob":
            spec["jobMode"] = "Train"
        return {
            "apiVersion": "kubeflow.org/v1",
            "kind": kind,
            "metadata": base_metadata,
            "spec": spec,
        }
    else:
        raise ValueError(f"Unknown manifest kind: {kind}")


@pytest.fixture(autouse=True, scope="session")
def setup_test_env():
    os.environ["KT_LAUNCH_TIMEOUT"] = "300"
    yield


@pytest.mark.level("minimal")
@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["Deployment", "Service", "RayCluster", "PyTorchJob", "TFJob", "MXJob", "XGBoostJob"])
async def test_byo_manifest_with_overrides(kind):
    """Test BYO manifest with comprehensive kwargs overrides and containers."""
    import kubetorch as kt

    # Get manifest with test values already set (deep copy to avoid modifying the original)
    test_manifest = copy.deepcopy(_get_basic_manifest(kind))

    # Add a container with some initial values
    container = {
        "name": "kubetorch",
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
    manifest_kind = test_manifest["kind"]

    if manifest_kind == "Deployment":
        test_manifest["spec"]["template"]["spec"]["containers"] = [container]
    elif manifest_kind == "Service":  # Knative
        test_manifest["spec"]["template"]["spec"]["containers"] = [container]
    elif manifest_kind == "RayCluster":
        test_manifest["spec"]["headGroupSpec"]["template"]["spec"]["containers"] = [container]
        test_manifest["spec"]["workerGroupSpecs"][0]["template"]["spec"]["containers"] = [container]
    elif manifest_kind in TRAINING_JOB_CONFIG:
        config = TRAINING_JOB_CONFIG[manifest_kind]
        container["name"] = config["container_name"]
        replica_specs = test_manifest["spec"][config["replica_specs_key"]]
        replica_specs[config["primary_replica"]]["template"]["spec"]["containers"] = [container]
        replica_specs["Worker"]["template"]["spec"]["containers"] = [container]

    image_type = "Ray" if kind == "RayCluster" else "Debian"
    image = getattr(kt.images, image_type)()

    # Create compute with comprehensive overrides
    compute = kt.Compute.from_manifest(test_manifest)
    compute.cpus = "0.5"
    compute.memory = "2Gi"
    compute.replicas = 3
    compute.image = image
    compute.gpu_anti_affinity = True

    service_name = f"{kt.config.username}-byo-{kind.lower()}"
    compute.service_name = service_name

    # Verify overrides are applied using compute properties
    assert compute.cpus == "0.5"
    assert compute.memory == "2Gi"
    assert compute.replicas == 3
    assert compute.server_image == image.image_id
    assert compute.gpu_anti_affinity is True
    assert compute.env_vars["CONTAINER_ENV"] == "container_value"
    assert compute.labels["test-label"] == "test-app"
    assert compute.annotations["test-annotation"] == "original-value"

    # Deploy and test function
    fn = await kt.fn(summer).to_async(compute)
    assert fn.service_name == service_name

    result = fn(5, 10)
    if kind in ["PyTorchJob", "TFJob", "MXJob", "XGBoostJob"]:
        assert isinstance(result, list)
        assert all(r == 15 for r in result)
        assert len(result) == 3
    else:
        assert result == 15


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_byo_manifest_pytorchjob_ddp():
    """Test BYO manifest PyTorchJob running a PyTorch DDP job with Kueue queue integration."""
    if not _kueue_available():
        pytest.skip("Kueue not installed - required for queue-based job admission")

    import kubetorch as kt

    # Create PyTorchJob manifest for distributed training
    pytorch_manifest = copy.deepcopy(_get_basic_manifest("PyTorchJob"))

    # Add container with PyTorch image
    container = {
        "name": "pytorch-container",
        "image": "pytorch/pytorch:latest",
        "resources": {
            "requests": {
                "cpu": "0.5",
                "memory": "1Gi",
            },
        },
    }

    pytorch_manifest["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"]["containers"] = [container]
    pytorch_manifest["spec"]["pytorchReplicaSpecs"]["Worker"]["template"]["spec"]["containers"] = [container]

    image = kt.images.Pytorch2312()

    # Use longer launch_timeout for large PyTorch image pulls (can be 5-10GB+)
    # Distributed execution is automatically detected from worker replicas in the manifest
    compute = kt.Compute.from_manifest(pytorch_manifest)
    compute.cpus = "0.5"
    compute.memory = "2Gi"
    compute.image = image
    compute.launch_timeout = 600  # 10 minutes for large image pulls
    compute.distributed_config = {"quorum_workers": 3}

    # Set Kueue queue for GPU scheduling (if Kueue is installed)
    # This adds the kueue.x-k8s.io/queue-name label and sets runPolicy.suspend = True
    compute.queue_name = "gpu-queue"

    # Verify configuration
    assert compute.cpus == "0.5"
    assert compute.memory == "2Gi"
    assert compute.replicas == 3

    # Verify Kueue queue configuration
    assert compute.queue_name == "gpu-queue"
    assert compute._manifest["metadata"]["labels"].get(QUEUE_LABEL) == "gpu-queue"
    assert compute.pod_template["metadata"]["labels"].get(QUEUE_LABEL) == "gpu-queue"
    # For PyTorchJob, runPolicy.suspend should be True for Kueue admission control
    assert compute._manifest["spec"]["runPolicy"]["suspend"] is True

    # Verify service manager is TrainJobServiceManager
    assert compute.service_manager.__class__.__name__ == "TrainJobServiceManager"
    assert compute.service_manager.template_label == "pytorchjob"
    assert compute.service_manager.primary_replica == "Master"
    assert compute.service_manager.worker_replica == "Worker"

    # Verify distributed config is automatically set
    assert compute.service_manager.is_distributed(compute._manifest)
    assert compute.distributed_config["distribution_type"] == "spmd"
    assert compute.distributed_config["quorum_workers"] == 3

    # Deploy and test DDP function with Kueue queue (requires Kubeflow Training Operator and Kueue)
    try:
        name = "byo-pytorchjob-ddp"
        fn = await kt.fn(torch_ddp, name=name).to_async(compute)

        # Run DDP training with 3 epochs: the function will run on all 3 replicas (1 Master + 2 Workers)
        result = fn(3)

        # Verify the DDP job completed successfully
        assert isinstance(result, list), f"Expected list, got {type(result)}: {result}"
        assert all(r == "Success" for r in result)
        assert len(result) == 3
    except Exception as e:
        # If Kubeflow Training Operator is not available, skip the deployment test
        if http_not_found(e):
            pytest.skip("PyTorchJob CRD not found - Kubeflow Training Operator not installed")
        raise


@pytest.mark.level("unit")
def test_from_manifest_getters_setters():
    """Test Compute object initialization with comprehensive kwargs and verify getters/setters."""
    import kubetorch as kt

    # Create a comprehensive config dict mapping key to (original_value, new_value)
    original_image = kt.images.Debian()
    new_image = kt.images.Ray()
    config = {
        "cpus": ("2.0", "3.0"),
        "memory": ("4Gi", "8Gi"),
        "disk_size": ("10Gi", "20Gi"),
        "gpus": ("1", "2"),
        "gpu_type": ("L4", "A100"),
        "priority_class_name": ("high-priority", "low-priority"),
        "gpu_memory": ("8Gi", "16Gi"),
        "namespace": ("test-namespace", "new-namespace"),
        "image": (original_image, new_image),
        "labels": ({"app": "test", "env": "dev"}, {"app": "test", "env": "prod"}),
        "annotations": ({"description": "test compute"}, {"description": "updated compute"}),
        "node_selector": ({"node-type": "gpu"}, {"node-type": "cpu"}),
        "tolerations": (
            [{"key": "gpu", "operator": "Equal", "value": "true", "effect": "NoSchedule"}],
            [{"key": "cpu", "operator": "Equal", "value": "true", "effect": "NoSchedule"}],
        ),
        "env_vars": (
            {"TEST_ENV": "test_value", "ANOTHER_ENV": "another_value"},
            {"TEST_ENV": "updated_value", "ANOTHER_ENV": "another_value"},
        ),
        "service_account_name": ("test-service-account", "new-service-account"),
        "image_pull_policy": ("Always", "IfNotPresent"),
        "inactivity_ttl": ("30m", "60m"),
        "gpu_anti_affinity": (True, False),
        "launch_timeout": (600, 900),
        "working_dir": ("/kt", "/new-dir"),
        "shared_memory_limit": ("1Gi", "2Gi"),
        "allowed_serialization": (["json", "pickle"], ["json", "pickle", "cloudpickle"]),
        "replicas": (2, 3),
        "freeze": (False, True),
        "kubeconfig_path": (None, "/custom/path/kubeconfig"),
    }

    # Create initial compute using original values (first element)
    init_config = {key: value[0] for key, value in config.items()}
    base_compute = kt.Compute(**init_config)
    compute = kt.Compute.from_manifest(base_compute.manifest)

    # For each key in the config, test getter and setter
    for key, (original_value, new_value) in config.items():
        # Check that initial value matches original value
        initial_value = getattr(compute, key)

        # Handle type conversions and special cases for comparison
        if key == "image":  # check server_image set properly
            initial_value = compute.server_image
            assert initial_value == original_value.image_id
        elif key == "labels" or key == "annotations":
            # Labels and annotations are dicts, check that original values are present
            for k, v in original_value.items():
                assert k in initial_value, f"Key {k} not found in {key}"
                assert initial_value[k] == v, f"Value for {key}[{k}] mismatch: got {initial_value[k]}, expected {v}"
        elif key == "env_vars":  # check that original values are present
            for k, v in original_value.items():
                assert k in initial_value, f"Env var {k} not found"
                assert initial_value[k] == v, f"Env var {k} value mismatch: got {initial_value[k]}, expected {v}"
        elif key == "tolerations":  # check that original toleration is present
            assert len(initial_value) > 0, "Tolerations should be set"
            assert any(tol.get("key") == "gpu" and tol.get("value") == "true" for tol in initial_value)
        elif key == "node_selector":
            assert initial_value["node-type"] == original_value["node-type"]
        elif key == "kubeconfig_path":  # uses a default when None
            continue
        else:
            assert (
                initial_value == original_value
            ), f"Initial value for {key} mismatch: got {initial_value}, expected {original_value}"

        # These properties are set with add_xxxx methods
        skip_properties = {"labels", "annotations", "env_vars", "tolerations"}
        if key in skip_properties:
            continue

        # Set the property to the new value
        setattr(compute, key, new_value)

        # Check that the new value was set properly
        updated_value = getattr(compute, key)
        if key == "image":
            updated_value = getattr(compute, "server_image")
            assert updated_value == new_value.image_id
        elif key == "node_selector":
            assert updated_value["node-type"] == new_value["node-type"]
        else:
            assert (
                updated_value == new_value
            ), f"Updated value for {key} mismatch: got {updated_value}, expected {new_value}"

    # Test add_xxxx methods for read-only properties
    add_methods = ["add_labels", "add_annotations", "add_env_vars", "add_tolerations"]

    for method_name in add_methods:
        key = method_name[4:]  # "add_labels" -> "labels"
        original_value, new_value = config[key]
        add_method = getattr(compute, method_name)
        add_method(new_value)

        updated_value = getattr(compute, key)

        # Verify new values are present

        if key == "tolerations":  # tolerations (list)
            # Check that new tolerations are present
            for new_tol in new_value:
                new_key = new_tol.get("key")
                assert new_key, f"Toleration must have a 'key' field: {new_tol}"
                found_tol = None
                for tol in updated_value:
                    if tol.get("key") == new_key:
                        found_tol = tol
                        break
                assert found_tol is not None, f"New toleration with key '{new_key}' not found after {method_name}"
                # Verify all fields match
                for k, v in new_tol.items():
                    assert (
                        found_tol.get(k) == v
                    ), f"Toleration {new_key}[{k}] mismatch: got {found_tol.get(k)}, expected {v}"

            # Verify original tolerations are still present (if they have different keys)
            for orig_tol in original_value:
                orig_key = orig_tol.get("key")
                if orig_key and not any(nt.get("key") == orig_key for nt in new_value):
                    found_orig = any(tol.get("key") == orig_key for tol in updated_value)
                    assert found_orig, f"Original toleration with key '{orig_key}' should still be present"
        else:  # dict
            for k, v in new_value.items():
                assert k in updated_value, f"New {key} key {k} not found after {method_name}"
                assert updated_value[k] == v, f"{key}[{k}] value mismatch: got {updated_value[k]}, expected {v}"
            # Verify original values are still present (for keys that weren't updated)
            for k, v in original_value.items():
                if k not in new_value:
                    assert k in updated_value, f"Original {key} key {k} should still be present"
                    assert (
                        updated_value[k] == v
                    ), f"Original {key}[{k}] value should be preserved: got {updated_value[k]}, expected {v}"

    # Test distributed_config
    compute.distributed_config = {"distribution_type": "spmd", "quorum_workers": 2}
    dist_config = compute.distributed_config
    assert dist_config["distribution_type"] == "spmd"
    assert dist_config["quorum_workers"] == 2
