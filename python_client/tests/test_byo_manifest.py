import copy
import os
import subprocess

import kubetorch as kt

import kubetorch.globals
import kubetorch.provisioning.constants as provisioning_constants

import pytest

from kubetorch.provisioning import SUPPORTED_TRAINING_JOBS
from kubetorch.provisioning.constants import DEFAULT_KT_SERVER_PORT
from kubetorch.utils import http_not_found

from .assets.torch_ddp.torch_ddp import torch_ddp
from .utils import get_hostname, summer

QUEUE_LABEL = provisioning_constants.KUEUE_QUEUE_NAME_LABEL


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

GPU_ANTI_AFFINITY = {
    "nodeAffinity": {
        "requiredDuringSchedulingIgnoredDuringExecution": {
            "nodeSelectorTerms": [{"matchExpressions": [{"key": "nvidia.com/gpu.count", "operator": "DoesNotExist"}]}]
        }
    }
}


def _make_container(
    name: str = "kubetorch",
    image: str = "user-image:latest",
    cpu: str = "0.3",
    memory: str = "512Mi",
    env: list = None,
) -> dict:
    """Create a container spec with common defaults."""
    container = {
        "name": name,
        "image": image,
        "resources": {"requests": {"cpu": cpu, "memory": memory}},
    }
    if env:
        container["env"] = env
    return container


def _add_container_to_manifest(manifest: dict, container: dict) -> None:
    """Add a container to the appropriate location based on manifest kind."""
    kind = manifest["kind"]

    if kind == "Deployment":
        manifest["spec"]["template"]["spec"]["containers"] = [container]
    elif kind == "Service":  # Knative
        manifest["spec"]["template"]["spec"]["containers"] = [container]
    elif kind == "RayCluster":
        manifest["spec"]["headGroupSpec"]["template"]["spec"]["containers"] = [container]
        manifest["spec"]["workerGroupSpecs"][0]["template"]["spec"]["containers"] = [container]
    elif kind in TRAINING_JOB_CONFIG:
        config = TRAINING_JOB_CONFIG[kind]
        container["name"] = config["container_name"]
        replica_specs = manifest["spec"][config["replica_specs_key"]]
        replica_specs[config["primary_replica"]]["template"]["spec"]["containers"] = [container]
        replica_specs["Worker"]["template"]["spec"]["containers"] = [container]


def _get_basic_manifest(
    kind: str,
    container: dict = None,
    gpu_anti_affinity: bool = False,
):
    """Generate a minimal manifest for the given kind with test values."""
    base_metadata = {
        "name": "",
        "namespace": kubetorch.globals.config.namespace,
        "labels": {"test-label": "test-app"},
        "annotations": {"test-annotation": "original-value"},
    }

    containers = [container] if container else []
    pod_spec = {"containers": containers}
    if gpu_anti_affinity:
        pod_spec["affinity"] = copy.deepcopy(GPU_ANTI_AFFINITY)

    if kind == "Deployment":
        manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": base_metadata,
            "spec": {
                "replicas": 2,  # Initial value to be overridden
                "selector": {"matchLabels": {"test-label": "test-app"}},
                "template": {
                    "metadata": {"labels": {"test-label": "test-app"}},
                    "spec": pod_spec,
                },
            },
        }
    elif kind == "Service":  # Knative
        manifest = {
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
                    "spec": pod_spec,
                },
            },
        }
    elif kind == "RayCluster":
        manifest = {
            "apiVersion": "ray.io/v1",
            "kind": "RayCluster",
            "metadata": base_metadata,
            "spec": {
                "headGroupSpec": {
                    "serviceType": "ClusterIP",
                    "rayStartParams": {"dashboard-host": "0.0.0.0"},
                    "template": {"spec": copy.deepcopy(pod_spec)},
                },
                "workerGroupSpecs": [
                    {
                        "replicas": 1,
                        "minReplicas": 1,
                        "maxReplicas": 10,
                        "groupName": "small-group",
                        "rayStartParams": {},
                        "template": {"spec": copy.deepcopy(pod_spec)},
                    }
                ],
            },
        }
    elif kind in TRAINING_JOB_CONFIG:
        config = TRAINING_JOB_CONFIG[kind]
        replica_pod_spec = copy.deepcopy(pod_spec)
        if container and replica_pod_spec["containers"]:
            replica_pod_spec["containers"][0]["name"] = config["container_name"]

        primary_replica_spec = {
            "replicas": 1,
            "restartPolicy": "OnFailure",
            "template": {"spec": copy.deepcopy(replica_pod_spec)},
        }
        worker_replica_spec = {
            "replicas": 2,
            "restartPolicy": "OnFailure",
            "template": {"spec": copy.deepcopy(replica_pod_spec)},
        }
        spec = {
            config["replica_specs_key"]: {
                config["primary_replica"]: primary_replica_spec,
                "Worker": worker_replica_spec,
            },
        }

        if kind == "MXJob":
            # MXJob requires jobMode field
            spec["jobMode"] = "Train"

        manifest = {
            "apiVersion": "kubeflow.org/v1",
            "kind": kind,
            "metadata": base_metadata,
            "spec": spec,
        }
    else:
        raise ValueError(f"Unknown manifest kind: {kind}")

    return manifest


def get_workload_manifest(workload_name: str, namespace: str):
    from .conftest import KUBETORCH_IMAGE

    """Generate a Deployment manifest for a kubetorch worker workload."""
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": workload_name, "namespace": namespace},
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": {"app": workload_name}},
            "template": {
                "metadata": {"labels": {"app": workload_name}},
                "spec": {
                    "containers": [
                        _make_container(name="worker", image=KUBETORCH_IMAGE, cpu="100m", memory="256Mi")
                        | {"imagePullPolicy": "Always"}
                    ],
                    "affinity": copy.deepcopy(GPU_ANTI_AFFINITY),
                },
            },
        },
    }


@pytest.fixture(autouse=True, scope="session")
def setup_test_env():
    os.environ["KT_LAUNCH_TIMEOUT"] = "300"
    yield


@pytest.mark.level("minimal")
@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["Deployment", "Service", "RayCluster", "PyTorchJob", "TFJob", "MXJob", "XGBoostJob"])
async def test_byo_manifest_with_overrides(kind):
    """Test BYO manifest with comprehensive kwargs overrides and containers."""
    # Create container with env vars for testing
    container = _make_container(
        env=[
            {"name": "ORIGINAL_ENV", "value": "original_value"},
            {"name": "CONTAINER_ENV", "value": "container_value"},
        ]
    )

    # Get manifest with container included
    test_manifest = _get_basic_manifest(kind, container=container)

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
    if kind.lower() in SUPPORTED_TRAINING_JOBS:
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

    # Verify service manager is configured for PyTorchJob
    assert compute.service_manager.__class__.__name__ == "ServiceManager"
    # Verify Kueue queue configuration
    assert compute.queue_name == "gpu-queue"
    assert compute._manifest["metadata"]["labels"].get(QUEUE_LABEL) == "gpu-queue"
    assert compute.pod_template["metadata"]["labels"].get(QUEUE_LABEL) == "gpu-queue"
    # For PyTorchJob, runPolicy.suspend should be True for Kueue admission control
    assert compute._manifest["spec"]["runPolicy"]["suspend"] is True

    # Verify service manager is configured for PyTorchJob
    assert compute.service_manager.template_label == "pytorchjob"
    assert compute.service_manager.config.get("primary_replica") == "Master"
    assert compute.service_manager.config.get("replica_specs_key") == "pytorchReplicaSpecs"

    # Verify distributed config is automatically set
    assert compute.service_manager._is_distributed(compute._manifest)
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
    # Create a comprehensive config dict mapping key to (original_value, new_value)
    original_image = kt.images.Debian()
    new_image = kt.images.Ray()
    # Properties that flow through manifest and can be tested with from_manifest
    # Note: allowed_serialization and freeze flow via WebSocket metadata, not manifest
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
        "replicas": (2, 3),
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


@pytest.mark.level("unit")
def test_from_manifest_custom_pod_template_path():
    """Test BYO manifest with custom pod_template_path for deeply nested pod templates."""
    # Custom CRD manifest where pod template is nested deeper than standard spec.template
    custom_manifest = {
        "apiVersion": "custom.example.io/v1",
        "kind": "MyCustomWorkload",
        "metadata": {
            "name": "test-custom-workload",
            "namespace": kubetorch.globals.config.namespace,
        },
        "spec": {
            "workload": {
                "template": {
                    "metadata": {"labels": {"app": "custom-app"}},
                    "spec": {
                        "containers": [
                            {
                                "name": "main",
                                "image": "my-image:latest",
                                "resources": {"requests": {"cpu": "100m"}},
                            }
                        ],
                    },
                },
            },
        },
    }

    with pytest.raises(ValueError):
        # Without pod_template_path, kubetorch can't find the pod spec for unknown CRDs
        # It won't find the user's "main" container since it's at spec.workload.template
        kt.Compute.from_manifest(
            manifest=copy.deepcopy(custom_manifest),
            selector={"app": "custom-app"},
        )

    # With pod_template_path, Compute correctly finds the pod spec at the nested location
    compute = kt.Compute.from_manifest(
        manifest=copy.deepcopy(custom_manifest),
        selector={"app": "custom-app"},
        pod_template_path="spec.workload.template",
    )

    # Now it finds the user's container
    container_names = [c["name"] for c in compute.pod_spec.get("containers", [])]
    assert "main" in container_names, "With custom path, user's container should be found"
    assert compute.pod_spec["containers"][0]["image"] == "my-image:latest"

    # Verify setters work at the correct location
    compute.cpus = "500m"
    assert (
        compute._manifest["spec"]["workload"]["template"]["spec"]["containers"][0]["resources"]["requests"]["cpu"]
        == "500m"
    )


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_byo_deployment_manifest_with_pod_template_path_override():
    """Test BYO manifest (deployment) with explicit pod_template_path override deploys correctly."""
    from .conftest import KUBETORCH_IMAGE

    pool_name = f"{kt.config.username}-byo-path-override"
    namespace = kt.globals.config.namespace

    # Note: We use "spec.template" here which is already the standard path for Deployments.
    # The purpose of this test is to verify that the pod_template_path override mechanism
    # works correctly end-to-end, not to test a non-standard path. When the override is
    # provided, kubetorch skips auto-detection and uses the path directly.
    deployment_manifest = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": pool_name,
            "namespace": namespace,
        },
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": {"app": pool_name}},
            "template": {
                "metadata": {"labels": {"app": pool_name}},
                "spec": {
                    "containers": [
                        {
                            "name": "kubetorch",
                            "image": KUBETORCH_IMAGE,
                            "imagePullPolicy": "Always",
                            "resources": {"requests": {"cpu": "100m", "memory": "256Mi"}},
                        }
                    ],
                    "affinity": copy.deepcopy(GPU_ANTI_AFFINITY),
                },
            },
        },
    }

    # Create Compute with explicit pod_template_path override
    # This skips kubetorch's default merging and uses manifest as-is
    compute = kt.Compute.from_manifest(
        manifest=deployment_manifest,
        selector={"app": pool_name},
        pod_template_path="spec.template",
    )

    # Verify the override is set and pod_spec is found
    assert compute._pod_template_path_override == ["spec", "template"]
    assert compute.pod_spec is not None
    assert compute.pod_spec["containers"][0]["name"] == "kubetorch"

    # Deploy and verify function works
    remote_fn = kt.fn(summer).to(compute)
    result = remote_fn(5, 10)
    assert result == 15


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_byo_jobset_manifest():
    """Test BYO manifest and path override behavior on a JobSet.

    Uses a JobSet to test the pod_template_path override mechanism. JobSet is ideal because:
    1. It's NOT explicitly supported by kubetorch (not in RESOURCE_CONFIGS)
    2. Its pod template is at spec.replicatedJobs[0].template.spec.template (NOT spec.template)
    3. Deleting the JobSet cleans up all child resources (Jobs, Pods)

    This test verifies that the override correctly finds the pod spec at the nested path, and throws an error
    if no pod template path is configured.
    """
    from .conftest import KUBETORCH_IMAGE

    job_name = f"{kt.config.username}-byo-jobset"
    namespace = kt.globals.config.namespace

    # JobSet manifest - pod template is at spec.replicatedJobs[0].template.spec.template
    # CRD installation: https://jobset.sigs.k8s.io/docs/installation/
    jobset_manifest = {
        "apiVersion": "jobset.x-k8s.io/v1alpha2",
        "kind": "JobSet",
        "metadata": {
            "name": job_name,
            "namespace": namespace,
        },
        "spec": {
            "replicatedJobs": [
                {
                    "name": "worker",
                    "replicas": 1,
                    "template": {
                        "spec": {
                            "parallelism": 1,
                            "completions": 1,
                            "template": {
                                "metadata": {"labels": {"app": job_name}},
                                "spec": {
                                    "restartPolicy": "Never",
                                    "containers": [
                                        {
                                            "name": "kubetorch",
                                            "image": KUBETORCH_IMAGE,
                                            "imagePullPolicy": "Always",
                                            "resources": {"requests": {"cpu": "100m", "memory": "256Mi"}},
                                        }
                                    ],
                                    "affinity": copy.deepcopy(GPU_ANTI_AFFINITY),
                                },
                            },
                        },
                    },
                }
            ],
        },
    }

    # Create Compute without pod_template_path override
    # This should fail because no pod template path is configured for JobSet
    with pytest.raises(ValueError):
        compute = kt.Compute.from_manifest(
            manifest=jobset_manifest,
            selector={"app": job_name},
        )

    try:
        # Create Compute with explicit pod_template_path override
        # Without this, kubetorch would look at spec.template and fail to find the containers
        compute = kt.Compute.from_manifest(
            manifest=jobset_manifest,
            selector={"app": job_name},
            pod_template_path="spec.replicatedJobs.0.template.spec.template",
        )

        # Verify the override is set and pod_spec is found at the correct nested location
        assert compute._pod_template_path_override == ["spec", "replicatedJobs", "0", "template", "spec", "template"]
        assert compute.pod_spec is not None
        assert compute.pod_spec["containers"][0]["name"] == "kubetorch"

        remote_fn = kt.fn(summer).to(compute)
        result = remote_fn(5, 10)
        assert result == 15
    finally:
        subprocess.run(
            ["kubectl", "delete", "jobset", job_name, "-n", namespace, "--ignore-not-found"],
            capture_output=True,
        )


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_byo_manifest_statefulset():
    """Test BYO manifest with a StatefulSet resource type.

    This tests the generic dynamic apply path with a StatefulSet.
    """
    from .conftest import KUBETORCH_IMAGE

    sts_name = f"{kt.config.username}-byo-sts"
    namespace = kt.globals.config.namespace

    # Create a StatefulSet manifest (not one of the built-in supported types)
    statefulset_manifest = {
        "apiVersion": "apps/v1",
        "kind": "StatefulSet",
        "metadata": {
            "name": sts_name,
            "namespace": namespace,
        },
        "spec": {
            "serviceName": sts_name,
            "replicas": 1,
            "selector": {"matchLabels": {"app": sts_name}},
            "template": {
                "metadata": {
                    "labels": {"app": sts_name},
                },
                "spec": {
                    "containers": [
                        {
                            "name": "kubetorch",
                            "image": KUBETORCH_IMAGE,
                            "imagePullPolicy": "Always",
                            "resources": {"requests": {"cpu": "100m", "memory": "256Mi"}},
                        }
                    ],
                    "affinity": copy.deepcopy(GPU_ANTI_AFFINITY),
                },
            },
        },
    }

    # Create Compute from StatefulSet manifest with selector
    # No pod_template_path is needed because StatefulSet has a default built in pod template path.
    compute = kt.Compute.from_manifest(
        manifest=statefulset_manifest,
        selector={"app": sts_name},
    )

    remote_fn = kt.fn(summer).to(compute)

    result = remote_fn(5, 10)
    assert result == 15


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_byo_manifest_with_selector():
    """Test BYO manifest flow: user provides manifest, KT applies it with custom selector.

    Use Case #2: User constructs manifest, applies via KT.

    The user provides their own K8s deployment manifest and a label selector.
    When kt.fn().to(compute) is called:
    1. KT applies the manifest via /apply (creates the deployment)
    2. KT registers the workload via /workload with the user's selector
    3. KT creates a K8s Service that routes to pods matching the selector

    The selector tells KT which pods belong to this compute, allowing proper
    tracking and routing even when the user's manifest uses custom labels.
    """
    workload_name = f"{kt.config.username}-byo-manifest"
    namespace = kt.globals.config.namespace

    # User's raw deployment manifest with custom labels
    # Key requirement: pods have labels matching the selector we'll provide
    byo_manifest = get_workload_manifest(workload_name, namespace)

    # 1. Create Compute from manifest with selector
    # The selector tells kubetorch which pods belong to this compute
    compute = kt.Compute.from_manifest(
        manifest=byo_manifest,
        selector={"app": workload_name},
    )

    # 2. Deploy a function using the standard kt.fn().to() pattern
    # This will:
    # - Apply the manifest (creates/updates the deployment)
    # - Register the workload with the user's selector
    # - Wait for pods to be ready
    # - Deploy the function
    remote_fn = kt.fn(summer).to(compute)

    # 3. Call the function and verify it works
    result = remote_fn(5, 10)
    assert result == 15


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_byo_manifest_with_endpoint_url():
    """Test BYO manifest with user-provided endpoint URL (Mode 2).

    Use Case: User wants KT to create pods that their routing layer will forward traffic to.

    Flow:
    1. User creates their own K8s Service, which is configured to route to pods with selector {app: workload_name}
    2. KT creates pods compute from manifest with labels matching the Service selector, and using user endpoint url
    3. Check that function calls go through user's service URL into KT's pods
    """
    workload_name = f"{kt.config.username}-endpoint-url"
    namespace = kt.globals.config.namespace
    controller = kt.globals.controller_client()

    # User's routing layer - routes traffic to pods with {app: workload_name} label
    user_service_name = f"{workload_name}-user-svc"
    user_port = 82
    user_service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": user_service_name,
            "namespace": namespace,
        },
        "spec": {
            "selector": {"app": workload_name},  # Will route to pods KT creates
            "ports": [{"port": user_port, "targetPort": DEFAULT_KT_SERVER_PORT}],
        },
    }
    controller.create_service(namespace=namespace, body=user_service)

    try:
        # Create Compute from manifest with selector and endpoint
        byo_manifest = get_workload_manifest(workload_name, namespace)
        user_service_url = f"http://{user_service_name}.{namespace}.svc.cluster.local:{user_port}"
        endpoint = kt.Endpoint(url=user_service_url)

        compute = kt.Compute.from_manifest(
            manifest=byo_manifest,
            selector={"app": workload_name},
            endpoint=endpoint,
        )
        assert compute._endpoint_config == endpoint
        assert compute.endpoint == user_service_url

        remote_fn = kt.fn(summer).to(compute)

        # Verify kubetorch did not create its own service (endpoint URL mode should skip service creation)
        kt_service_name = remote_fn.service_name
        with pytest.raises(Exception):
            controller.get_service(name=kt_service_name, namespace=namespace)

        # Traffic flows through user's service here since no KT created service exists
        result = remote_fn(5, 10, sleep_time=5)
        assert result == 15

    finally:
        subprocess.run(["kt", "teardown", user_service_name, "-n", namespace, "-y"])


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_byo_manifest_with_endpoint_selector():
    """Test BYO manifest with custom endpoint selector (Mode 3).

    Use Case: User wants KT to create service but route to specific pod(s).

    This test uses PyTorchJob which naturally creates master + worker pods.
    Use the custom endpoint selector to route calls only to worker pods, overriding
    the default master routing.

    Verify:
    1. Workload selector finds both master and worker pods
    2. Endpoint selector routes calls ONLY to worker pod
    3. Function calls consistently go to worker
    """
    from .conftest import KUBETORCH_IMAGE

    job_name = f"{kt.config.username}-endpoint-sel"
    namespace = kt.globals.config.namespace
    controller = kt.globals.controller_client()

    # Create PyTorchJob manifest with 1 master + 1 worker
    container = _make_container(name="pytorch", image=KUBETORCH_IMAGE, cpu="100m", memory="256Mi")
    container["imagePullPolicy"] = "Always"
    pytorch_manifest = _get_basic_manifest(
        "PyTorchJob",
        container=container,
        gpu_anti_affinity=True,
    )
    pytorch_manifest["spec"]["pytorchReplicaSpecs"]["Worker"]["replicas"] = 1
    pytorch_manifest["metadata"]["labels"] = {}
    pytorch_manifest["metadata"]["annotations"] = {}

    # Workload selector: tracks all pods (master + worker)
    workload_selector = {"training.kubeflow.org/job-name": job_name}

    # Endpoint selector: route only to worker pod
    endpoint_selector = {
        "training.kubeflow.org/job-name": job_name,
        "training.kubeflow.org/replica-type": "worker",
    }
    endpoint = kt.Endpoint(selector=endpoint_selector)
    compute = kt.Compute.from_manifest(
        manifest=pytorch_manifest,
        selector=workload_selector,
        endpoint=endpoint,
    )
    assert compute._endpoint_config == endpoint

    hostname_fn = kt.fn(get_hostname).to(compute)

    # Check that workload selector finds both master and worker pods
    label_selector = ",".join(f"{k}={v}" for k, v in workload_selector.items())
    pods_result = controller.list_pods(namespace=namespace, label_selector=label_selector)
    workload_pods = pods_result.get("items", [])
    assert len(workload_pods) == 2

    pod_types = {w["metadata"]["labels"].get("training.kubeflow.org/replica-type") for w in workload_pods}
    assert pod_types == {"master", "worker"}, f"Expected master and worker pods, got {pod_types}"

    # Check that function calls work. Note: In distributed mode (PyTorchJob with SPMD),
    # ALL replicas participate in execution and return results - the endpoint selector
    # only controls which pod receives the initial request, but doesn't filter results.
    for _ in range(3):
        hostnames = hostname_fn()
        if isinstance(hostnames, list):
            # Distributed jobs return a list of results from all replicas
            # Both master and worker participate in SPMD execution
            assert len(hostnames) == 2, f"Expected 2 hostnames from master+worker, got {len(hostnames)}"
            hostname_types = set()
            for h in hostnames:
                if "worker" in h.lower():
                    hostname_types.add("worker")
                elif "master" in h.lower():
                    hostname_types.add("master")
            assert hostname_types == {
                "master",
                "worker",
            }, f"Expected hostnames from both master and worker in SPMD mode, got: {hostnames}"
        else:
            # Single result - should be from worker (endpoint selector target)
            assert "worker" in hostnames.lower(), (
                f"Expected call to go to worker pod, but got hostname: {hostnames}. "
                f"Endpoint selector should route only to worker."
            )
