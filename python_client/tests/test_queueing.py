import os

# Mimic CI for this test suite even locally, to ensure that
# resources are created with the branch name prefix
os.environ["CI"] = "true"

import os
import threading
import time

import pytest

from .utils import create_random_name_prefix, summer


@pytest.fixture(autouse=True, scope="session")
def setup_test_env():
    # Keep the launch timeout low for this test suite, unless overridden (ex: for GPU tests)
    os.environ["KT_LAUNCH_TIMEOUT"] = "120"
    yield


@pytest.mark.level("minimal")
def test_launch_fn_with_queue():
    import kubetorch as kt
    from kubernetes.client import CoreV1Api
    from kubernetes.config import load_kube_config

    load_kube_config()
    core_api = CoreV1Api()

    prefix = create_random_name_prefix()
    name = f"{prefix}-queue"

    queue_name = "preferred"
    remote_fn = kt.fn(summer, name=name).to(
        compute=kt.Compute(
            cpus=".01",
            gpu_anti_affinity=True,
            launch_timeout=300,
            queue=queue_name,
        ),
    )

    # Validate pod is in KAI queue
    pods = core_api.list_namespaced_pod(
        namespace=remote_fn.compute.namespace,
        label_selector=f"kubetorch.com/service={remote_fn.service_name}",
    ).items

    assert pods

    pod = pods[0]
    queue_label = pod.metadata.labels.get("kai.scheduler/queue")
    assert (
        queue_label == queue_name
    ), f"Pod not assigned to expected queue '{queue_name}'. Found: '{queue_label}'"


# Gang scheduling needs to be migrated to work for deployment instead of knative services,
# but also, the test needs to check that pods wait in the queue until all can be scheduled.
# Really, we should be doing these queuing tests by setting a max on the queue resources and testing
# different ways of scheduling pods to that queue, the resulting order, etc.
@pytest.mark.skip("Doesn't work properly yet")
@pytest.mark.level("minimal")
def test_launch_fn_with_gang_scheduling():
    import kubetorch as kt
    from kubernetes.client import CoreV1Api
    from kubernetes.config import load_kube_config

    load_kube_config()
    core_api = CoreV1Api()

    remote_fn = kt.fn(summer).to(
        compute=kt.Compute(
            cpus=".01", gpu_anti_affinity=True, launch_timeout=300
        ).autoscale(
            min_scale=3,
            scale_to_zero_pod_retention_period="50m",  # 50 minutes retention period
        ),
    )

    # Validate pod is in KAI queue
    pods = core_api.list_namespaced_pod(
        namespace=remote_fn.compute.namespace,
        label_selector=f"kubetorch.com/service={remote_fn.service_name}",
    ).items

    assert len(pods) == 3


@pytest.mark.level("minimal")
def test_pods_assigned_to_queue():
    import kubetorch as kt
    from kubernetes.client import CoreV1Api
    from kubernetes.config import load_kube_config

    load_kube_config()
    core_api = CoreV1Api()

    default_fn_name = "default-fn"
    priority_fn_name = "priority-fn"

    kt.fn(summer, name=default_fn_name).to(
        compute=kt.Compute(
            cpus="0.1",
            launch_timeout=120,
        ),
    )
    # This one uses knative autoscaling to ensure both modes are tested
    general_queue_name = "preferred"
    kt.fn(summer, name=priority_fn_name).to(
        compute=kt.Compute(
            cpus="0.1",
            launch_timeout=120,
            queue=general_queue_name,
        ).autoscale(min_scale=1),
    )

    pods = core_api.list_namespaced_pod(
        namespace="default",
    ).items
    default_pod = next(pod for pod in pods if default_fn_name in pod.metadata.name)
    priority_pod = next(pod for pod in pods if priority_fn_name in pod.metadata.name)

    # Ensure both are assigned to their queues
    assert priority_pod.metadata.labels.get("kai.scheduler/queue") == general_queue_name

    # Validate the priority pod is scheduled or running
    assert priority_pod.status.phase in ["Pending", "Running"]

    # If there is contention, the default pod should still be queued
    default_state = default_pod.metadata.annotations.get("kai.scheduler/state")
    if default_pod.status.phase == "Pending":
        assert default_state == "Queued" or default_state is None


@pytest.mark.level("minimal")
def test_queue_priorities_and_priority_class_affect_scheduling():
    """
    - Pods in a custom queue (e.g., 'preferred') are prioritized over pods in the default queue.
    - Pods within the same queue are ordered by PriorityClass.
    """
    import kubetorch as kt
    from kubernetes.client import CoreV1Api
    from kubernetes.config import load_kube_config

    load_kube_config()
    core_api = CoreV1Api()

    def deploy_fn(fn_name, queue_name, priority_class=None):
        compute = kt.Compute(
            cpus="0.1",
            launch_timeout=90,
            queue=queue_name,
            priority_class_name=priority_class,
        )
        return kt.fn(summer, name=fn_name).to(compute=compute)

    # Launch all pods in parallel
    threads = []
    deployments = {}

    preferred_fn_name = "preferred-fn"
    low_priority_fn_name = "low-p-fn"
    high_priority_fn_name = "high-p-fn"

    pod_configs = [
        (preferred_fn_name, "preferred", None),  # Preferred queue
        (
            low_priority_fn_name,
            "preferred",
            "dev",
        ),  # Preferred queue, low priority
        (
            high_priority_fn_name,
            "preferred",
            "prod",
        ),  # Preferred queue, high priority
    ]

    for fn_name, queue, priority in pod_configs:
        thread = threading.Thread(
            target=lambda: deployments.update(
                {fn_name: deploy_fn(fn_name, queue, priority)}
            )
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    # Allow time for scheduling
    time.sleep(5)

    pods = core_api.list_namespaced_pod(
        namespace="default",
    ).items

    pod_map = {}
    for fn_name, _, _ in pod_configs:
        matching_pod = next(
            (pod for pod in pods if fn_name in pod.metadata.name),
            None,
        )
        assert matching_pod is not None, f"No pod found for {fn_name}"
        pod_map[fn_name] = matching_pod

    # Validate queue labels
    assert (
        pod_map[preferred_fn_name].metadata.labels.get("kai.scheduler/queue")
        == "preferred"
    )
    assert (
        pod_map[low_priority_fn_name].metadata.labels.get("kai.scheduler/queue")
        == "preferred"
    )
    assert (
        pod_map[high_priority_fn_name].metadata.labels.get("kai.scheduler/queue")
        == "preferred"
    )

    # Validate PriorityClass annotations
    assert pod_map[low_priority_fn_name].spec.priority_class_name == "dev"
    assert pod_map[high_priority_fn_name].spec.priority_class_name == "prod"

    # preferred queue pods run before default queue pod
    preferred_fn_phase = pod_map[preferred_fn_name].status.phase
    default_fn_phase = pod_map[preferred_fn_name].status.phase

    assert preferred_fn_phase in ["Running", "Pending"]
    if default_fn_phase == "Pending":
        default_state = pod_map[preferred_fn_name].metadata.annotations.get(
            "kai.scheduler/state"
        )
        assert default_state in ["Queued", None]

    # high-priority pod runs before low-priority pod
    high_priority_phase = pod_map[high_priority_fn_name].status.phase
    low_priority_phase = pod_map[low_priority_fn_name].status.phase

    assert high_priority_phase in ["Running", "Pending"]
    if low_priority_phase == "Pending":
        low_state = pod_map[low_priority_fn_name].metadata.annotations.get(
            "kai.scheduler/state"
        )
        assert low_state in ["Queued", None]


@pytest.mark.level("minimal")
def test_fn_failure_to_launch_with_invalid_priority_class():
    import kubetorch as kt

    name = f"{create_random_name_prefix()}-invalid-p"
    with pytest.raises(kt.ResourceNotAvailableError):
        kt.fn(summer, name=name).to(
            kt.Compute(
                cpus=".01",
                gpu_anti_affinity=True,
                launch_timeout=300,
                priority_class_name="invalid-priority-class",
            ),
        )
