"""
Lightweight tests for Kueue integration with Kubetorch.

These tests verify that Kueue queue configuration is correctly applied to
workloads without requiring Kueue to be installed in the cluster. The tests
focus on manifest configuration rather than actual queuing behavior.

For full Kueue integration testing, install Kueue and the resources from:
    charts/kueue/kueue-resources.yaml
"""

import kubetorch as kt
import kubetorch.serving.constants as serving_constants
import pytest

QUEUE_LABEL = serving_constants.KUEUE_QUEUE_NAME_LABEL


@pytest.mark.level("unit")
def test_queue_name_deployment():
    """Test that queue_name adds Kueue labels to Deployment manifests."""
    compute = kt.Compute(
        cpus="1",
        memory="2Gi",
        queue_name="gpu-queue",
    )

    # Verify queue label is set on top-level metadata
    assert compute._manifest["metadata"]["labels"].get(QUEUE_LABEL) == "gpu-queue"

    # Verify queue label is set on pod template metadata
    assert compute.pod_template["metadata"]["labels"].get(QUEUE_LABEL) == "gpu-queue"

    # Verify queue_name property returns the correct value
    assert compute.queue_name == "gpu-queue"


@pytest.mark.level("unit")
def test_queue_name_pytorchjob():
    """Test that queue_name adds Kueue labels and suspend to PyTorchJob manifests."""
    # Create a PyTorchJob manifest
    manifest = {
        "apiVersion": "kubeflow.org/v1",
        "kind": "PyTorchJob",
        "metadata": {"name": "test-job", "namespace": "default"},
        "spec": {
            "pytorchReplicaSpecs": {
                "Master": {
                    "replicas": 1,
                    "template": {"spec": {"containers": []}},
                },
                "Worker": {
                    "replicas": 1,
                    "template": {"spec": {"containers": []}},
                },
            }
        },
    }

    compute = kt.Compute.from_manifest(manifest)
    compute.queue_name = "gpu-queue"

    # Verify queue label is set
    assert compute._manifest["metadata"]["labels"].get(QUEUE_LABEL) == "gpu-queue"
    assert compute.pod_template["metadata"]["labels"].get(QUEUE_LABEL) == "gpu-queue"

    # Verify runPolicy.suspend is True for Kueue admission control
    assert compute._manifest["spec"]["runPolicy"]["suspend"] is True

    # Verify clearing queue_name also clears suspend
    compute.queue_name = None
    assert compute.queue_name is None
    assert compute._manifest["spec"]["runPolicy"]["suspend"] is False


@pytest.mark.level("unit")
def test_queue_name_from_manifest():
    """Test that queue_name is extracted from manifests with pre-existing Kueue labels."""
    manifest = {
        "apiVersion": "kubeflow.org/v1",
        "kind": "PyTorchJob",
        "metadata": {
            "name": "test-job",
            "namespace": "default",
            "labels": {QUEUE_LABEL: "existing-queue"},
        },
        "spec": {
            "pytorchReplicaSpecs": {
                "Master": {
                    "replicas": 1,
                    "template": {"spec": {"containers": []}},
                }
            }
        },
    }

    compute = kt.Compute.from_manifest(manifest)

    # Verify queue_name is extracted from manifest
    assert compute.queue_name == "existing-queue"


@pytest.mark.level("unit")
def test_queue_name_setter_updates_labels():
    """Test that setting queue_name after creation updates all labels correctly."""
    compute = kt.Compute(cpus="1", memory="2Gi")

    # Initially no queue
    assert compute.queue_name is None
    assert QUEUE_LABEL not in compute._manifest["metadata"].get("labels", {})

    # Set queue_name
    compute.queue_name = "new-queue"
    assert compute.queue_name == "new-queue"
    assert compute._manifest["metadata"]["labels"][QUEUE_LABEL] == "new-queue"
    assert compute.pod_template["metadata"]["labels"][QUEUE_LABEL] == "new-queue"

    # Clear queue_name
    compute.queue_name = None
    assert compute.queue_name is None
    assert QUEUE_LABEL not in compute._manifest["metadata"]["labels"]
    assert QUEUE_LABEL not in compute.pod_template["metadata"]["labels"]
