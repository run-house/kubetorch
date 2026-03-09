import asyncio
import json
import logging
import os
from typing import List, Optional

from core import k8s
from core.constants import (
    DEPLOYMENT_SERVICE_TEMPLATE_FILE,
    KT_MODULE_LABEL,
    KT_TEMPLATE_LABEL,
)
from core.database import get_db, Pool
from core.utils import load_template
from fastapi.concurrency import run_in_threadpool
from kubernetes.client.rest import ApiException
from sqlalchemy import and_

logger = logging.getLogger(__name__)


def get_template_dir():
    """Get the path to the templates directory."""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",  # up from services/
        "templates",
    )


def build_service_for_pool(
    pool_name: str,
    namespace: str,
    server_port: int,
    labels: dict,
    annotations: dict,
    selector: dict,
    headless: bool = False,
):
    """Build a K8s Service manifest for a pool.

    Args:
        pool_name: Name of the pool
        namespace: Kubernetes namespace
        server_port: Port the pods listen on
        labels: Labels to apply to the service
        annotations: Annotations to apply to the service
        selector: Label selector to route traffic to pods
        headless: If True, creates a headless service for pod discovery
    """
    service_labels = (labels or {}).copy()
    service_labels.pop(KT_TEMPLATE_LABEL, None)

    svc_name = f"{pool_name}-headless" if headless else pool_name

    service = load_template(
        template_file=DEPLOYMENT_SERVICE_TEMPLATE_FILE,
        template_dir=get_template_dir(),
        name=svc_name,
        namespace=namespace,
        annotations=annotations or {},
        labels=service_labels,
        deployment_name=pool_name,
        module_name=labels.get(KT_MODULE_LABEL, pool_name) if labels else pool_name,
        distributed=headless,
        server_port=server_port,
    )

    # Override the selector with the pool's selector
    service["spec"]["selector"] = selector.copy()

    return service


async def create_service_if_not_exists(
    service_manifest: dict,
    namespace: str,
    service_name: str,
):
    """Create a Service if it doesn't already exist."""
    try:
        await run_in_threadpool(
            k8s.core_v1.create_namespaced_service,
            namespace=namespace,
            body=service_manifest,
        )
        logger.info(f"Created service '{service_name}' in namespace '{namespace}'")
    except ApiException as se:
        if se.status == 409:
            logger.info(f"Service '{service_name}' already exists, skipping create")
        else:
            logger.error(f"Failed to create service '{service_name}': {se}")
            raise


async def delete_k8s_resource(kind: str, name: str, namespace: str) -> bool:
    """Try to delete a K8s resource of the given kind.

    Returns True if deleted successfully, False if not found.
    Raises exception for other errors.
    """
    try:
        if kind == "Deployment":
            await run_in_threadpool(
                k8s.apps_v1.delete_namespaced_deployment,
                name=name,
                namespace=namespace,
            )
        elif kind == "StatefulSet":
            await run_in_threadpool(
                k8s.apps_v1.delete_namespaced_stateful_set,
                name=name,
                namespace=namespace,
            )
        elif kind == "DaemonSet":
            await run_in_threadpool(
                k8s.apps_v1.delete_namespaced_daemon_set,
                name=name,
                namespace=namespace,
            )
        elif kind == "ReplicaSet":
            await run_in_threadpool(
                k8s.apps_v1.delete_namespaced_replica_set,
                name=name,
                namespace=namespace,
            )
        elif kind in ("PyTorchJob", "TFJob", "MXJob", "XGBoostJob", "MPIJob"):
            # Kubeflow training jobs
            await run_in_threadpool(
                k8s.custom_objects.delete_namespaced_custom_object,
                group="kubeflow.org",
                version="v1",
                namespace=namespace,
                plural=f"{kind.lower()}s",
                name=name,
            )
        elif kind == "RayCluster":
            await run_in_threadpool(
                k8s.custom_objects.delete_namespaced_custom_object,
                group="ray.io",
                version="v1",
                namespace=namespace,
                plural="rayclusters",
                name=name,
            )
        elif kind == "KnativeService":
            await run_in_threadpool(
                k8s.custom_objects.delete_namespaced_custom_object,
                group="serving.knative.dev",
                version="v1",
                namespace=namespace,
                plural="services",
                name=name,
            )
        else:
            logger.warning(f"Unknown resource kind '{kind}', skipping")
            return False

        logger.info(f"Deleted {kind} {name} in namespace {namespace}")
        return True

    except ApiException as e:
        if e.status == 404:
            # Resource not found - not an error, just means it's not this type
            return False
        else:
            logger.warning(f"Failed to delete {kind} {name}: {e}")
            raise


# =============================================================================
# Readiness Check Helpers
# =============================================================================


async def check_deployment_ready(
    namespace: str, name: str, timeout: int, poll_interval: int
):
    """Check if a Deployment is ready.

    Checks:
    - Deployment exists
    - All replicas are ready
    - No pod errors (image pull failures, etc.)
    """
    import time

    from core.models import ReadinessResponse

    start_time = time.time()

    while (time.time() - start_time) < timeout:
        try:
            # Get deployment
            deployment = await run_in_threadpool(
                k8s.apps_v1.read_namespaced_deployment,
                name=name,
                namespace=namespace,
            )

            spec = deployment.spec
            status = deployment.status

            desired_replicas = spec.replicas or 1
            ready_replicas = status.ready_replicas or 0

            if ready_replicas >= desired_replicas and desired_replicas > 0:
                return ReadinessResponse(
                    ready=True,
                    message=f"Deployment ready with {ready_replicas}/{desired_replicas} replicas",
                    resource_type="deployment",
                    details={
                        "desired_replicas": desired_replicas,
                        "ready_replicas": ready_replicas,
                    },
                )

            # Check for pod errors
            error = await check_pods_for_errors(namespace, name)
            if error:
                return ReadinessResponse(
                    ready=False,
                    message=error,
                    resource_type="deployment",
                    details={"error_type": "pod_error"},
                )

            # Check for ReplicaSet creation errors (e.g., missing PriorityClass)
            error = await check_replicaset_events_for_errors(namespace, name)
            if error:
                return ReadinessResponse(
                    ready=False,
                    message=error,
                    resource_type="deployment",
                    details={"error_type": "replicaset_error"},
                )

        except ApiException as e:
            if e.status == 404:
                logger.debug(f"Deployment {name} not found yet, waiting...")
            else:
                raise

        await asyncio.sleep(poll_interval)

    return ReadinessResponse(
        ready=False,
        message=f"Deployment {name} not ready after {timeout} seconds",
        resource_type="deployment",
        details={"timeout": timeout},
    )


async def check_knative_ready(
    namespace: str, name: str, timeout: int, poll_interval: int
):
    """Check if a Knative Service is ready.

    Checks:
    - Service exists and has status
    - Ready condition is True
    - Min-scale pods are running (if configured)
    """
    import time

    from core.models import ReadinessResponse

    start_time = time.time()

    while (time.time() - start_time) < timeout:
        try:
            # Get Knative service
            service = await run_in_threadpool(
                k8s.custom_objects.get_namespaced_custom_object,
                group="serving.knative.dev",
                version="v1",
                namespace=namespace,
                plural="services",
                name=name,
            )

            status = service.get("status", {})
            conditions = status.get("conditions", [])

            # Check Ready condition
            for condition in conditions:
                if condition.get("type") == "Ready":
                    if condition.get("status") == "True":
                        return ReadinessResponse(
                            ready=True,
                            message="Knative service is ready",
                            resource_type="knative",
                            details={"url": status.get("url")},
                        )
                    elif condition.get("reason") == "NotOwned":
                        return ReadinessResponse(
                            ready=False,
                            message=f"Knative service conflict: {condition.get('message')}",
                            resource_type="knative",
                            details={"error_type": "conflict"},
                        )

            # Get the latest revision to check for pod errors
            latest_revision = status.get("latestCreatedRevisionName")
            if latest_revision:
                # Check all pods for this revision (not just running ones)
                pods = await run_in_threadpool(
                    k8s.core_v1.list_namespaced_pod,
                    namespace=namespace,
                    label_selector=f"serving.knative.dev/revision={latest_revision}",
                )

                # Check for pod-level errors first (before checking min-scale)
                for pod in pods.items:
                    error = check_pod_for_errors(pod)
                    if error:
                        return ReadinessResponse(
                            ready=False,
                            message=error,
                            resource_type="knative",
                            details={"error_type": "pod_error"},
                        )

                # Check revision for errors
                error = await check_revision_for_errors(namespace, latest_revision)
                if error:
                    return ReadinessResponse(
                        ready=False,
                        message=error,
                        resource_type="knative",
                        details={"error_type": "revision_error"},
                    )

                # Now check min-scale pods if configured
                annotations = (
                    service.get("spec", {})
                    .get("template", {})
                    .get("metadata", {})
                    .get("annotations", {})
                )
                min_scale_str = annotations.get(
                    "autoscaling.knative.dev/min-scale", "0"
                )
                min_scale = int(min_scale_str) if min_scale_str else 0

                if min_scale > 0:
                    running_pods = [
                        p
                        for p in pods.items
                        if p.status.phase == "Running"
                        and all(c.ready for c in (p.status.container_statuses or []))
                    ]
                    if len(running_pods) < min_scale:
                        # Already checked for errors above, so just continue polling
                        pass

        except ApiException as e:
            if e.status == 404:
                logger.debug(f"Knative service {name} not found yet, waiting...")
            else:
                raise

        await asyncio.sleep(poll_interval)

    return ReadinessResponse(
        ready=False,
        message=f"Knative service {name} not ready after {timeout} seconds",
        resource_type="knative",
        details={"timeout": timeout},
    )


async def check_raycluster_ready(
    namespace: str, name: str, timeout: int, poll_interval: int
):
    """Check if a RayCluster is ready.

    Checks:
    - RayCluster exists
    - Cluster state is "ready"
    - Head and worker pods are running
    """
    import time

    from core.models import ReadinessResponse

    start_time = time.time()

    while (time.time() - start_time) < timeout:
        try:
            # Get RayCluster
            raycluster = await run_in_threadpool(
                k8s.custom_objects.get_namespaced_custom_object,
                group="ray.io",
                version="v1",
                namespace=namespace,
                plural="rayclusters",
                name=name,
            )

            status = raycluster.get("status", {})
            state = status.get("state", "")

            if state == "ready":
                return ReadinessResponse(
                    ready=True,
                    message="RayCluster is ready",
                    resource_type="raycluster",
                    details={"state": state},
                )
            elif state == "failed":
                return ReadinessResponse(
                    ready=False,
                    message="RayCluster failed to start",
                    resource_type="raycluster",
                    details={"state": state, "error_type": "cluster_failed"},
                )

            # Check pod counts
            spec = raycluster.get("spec", {})
            head_replicas = spec.get("headGroupSpec", {}).get("replicas", 1)
            worker_groups = spec.get("workerGroupSpecs", [])
            worker_replicas = sum(wg.get("replicas", 0) for wg in worker_groups)
            expected_pods = head_replicas + worker_replicas

            # Get running pods
            pods = await run_in_threadpool(
                k8s.core_v1.list_namespaced_pod,
                namespace=namespace,
                label_selector=f"ray.io/cluster={name}",
            )
            running_pods = [p for p in pods.items if p.status.phase == "Running"]

            if len(running_pods) >= expected_pods:
                return ReadinessResponse(
                    ready=True,
                    message=f"RayCluster ready with {len(running_pods)} pods",
                    resource_type="raycluster",
                    details={
                        "state": state,
                        "running_pods": len(running_pods),
                        "expected_pods": expected_pods,
                    },
                )

            # Check for Ray installation errors in head pod
            head_pods = [
                p
                for p in pods.items
                if p.metadata.labels.get("ray.io/node-type") == "head"
            ]
            if head_pods:
                error = await check_ray_head_for_errors(
                    namespace, head_pods[0].metadata.name
                )
                if error:
                    return ReadinessResponse(
                        ready=False,
                        message=error,
                        resource_type="raycluster",
                        details={"error_type": "ray_error"},
                    )

        except ApiException as e:
            if e.status == 404:
                logger.debug(f"RayCluster {name} not found yet, waiting...")
            else:
                raise

        await asyncio.sleep(poll_interval)

    return ReadinessResponse(
        ready=False,
        message=f"RayCluster {name} not ready after {timeout} seconds",
        resource_type="raycluster",
        details={"timeout": timeout},
    )


async def check_trainjob_ready(
    namespace: str, name: str, job_type: str, timeout: int, poll_interval: int
):
    """Check if a training job (PyTorchJob, TFJob, etc.) is ready.

    Checks:
    - Job exists
    - Job conditions (Running or Succeeded)
    - Primary and worker pods are running
    """
    import time

    from core.models import ReadinessResponse
    from fastapi import HTTPException

    # Map job type to API plural
    plural_map = {
        "pytorchjob": "pytorchjobs",
        "tfjob": "tfjobs",
        "mxjob": "mxjobs",
        "xgboostjob": "xgboostjobs",
    }
    plural = plural_map.get(job_type.lower())
    if not plural:
        raise HTTPException(status_code=400, detail=f"Unknown job type: {job_type}")

    # Map job type to replica specs key and primary replica name
    config_map = {
        "pytorchjob": {"specs_key": "pytorchReplicaSpecs", "primary": "Master"},
        "tfjob": {"specs_key": "tfReplicaSpecs", "primary": "Chief"},
        "mxjob": {"specs_key": "mxReplicaSpecs", "primary": "Scheduler"},
        "xgboostjob": {"specs_key": "xgbReplicaSpecs", "primary": "Master"},
    }
    config = config_map.get(job_type.lower(), {})

    start_time = time.time()

    while (time.time() - start_time) < timeout:
        try:
            # Get training job
            job = await run_in_threadpool(
                k8s.custom_objects.get_namespaced_custom_object,
                group="kubeflow.org",
                version="v1",
                namespace=namespace,
                plural=plural,
                name=name,
            )

            status = job.get("status", {})
            conditions = status.get("conditions", [])

            # Check for failure
            for condition in conditions:
                if (
                    condition.get("type") == "Failed"
                    and condition.get("status") == "True"
                ):
                    return ReadinessResponse(
                        ready=False,
                        message=f"{job_type} failed: {condition.get('message', 'Unknown error')}",
                        resource_type=job_type.lower(),
                        details={"error_type": "job_failed"},
                    )

            # Check if running or succeeded
            is_running = any(
                c.get("type") in ("Running", "Succeeded") and c.get("status") == "True"
                for c in conditions
            )

            if is_running or not conditions:
                # Check pod counts
                spec = job.get("spec", {})
                replica_specs = spec.get(config.get("specs_key", ""), {})
                expected_replicas = sum(
                    rs.get("replicas", 0)
                    for rs in replica_specs.values()
                    if isinstance(rs, dict)
                )

                # Get running pods
                pods = await run_in_threadpool(
                    k8s.core_v1.list_namespaced_pod,
                    namespace=namespace,
                    label_selector=f"training.kubeflow.org/job-name={name}",
                )
                running_pods = [p for p in pods.items if p.status.phase == "Running"]

                if len(running_pods) >= expected_replicas and expected_replicas > 0:
                    primary_pods = [
                        p
                        for p in running_pods
                        if p.metadata.labels.get("training.kubeflow.org/replica-type")
                        == config.get("primary", "").lower()
                    ]
                    worker_pods = [
                        p
                        for p in running_pods
                        if p.metadata.labels.get("training.kubeflow.org/replica-type")
                        == "worker"
                    ]
                    return ReadinessResponse(
                        ready=True,
                        message=f"{job_type} ready with {len(running_pods)} pods ({len(primary_pods)} primary, {len(worker_pods)} worker)",
                        resource_type=job_type.lower(),
                        details={
                            "running_pods": len(running_pods),
                            "expected_replicas": expected_replicas,
                        },
                    )

        except ApiException as e:
            if e.status == 404:
                logger.debug(f"{job_type} {name} not found yet, waiting...")
            else:
                raise

        await asyncio.sleep(poll_interval)

    return ReadinessResponse(
        ready=False,
        message=f"{job_type} {name} not ready after {timeout} seconds",
        resource_type=job_type.lower(),
        details={"timeout": timeout},
    )


async def check_selector_ready(
    namespace: str, pool_name: str, timeout: int, poll_interval: int
):
    """Check if pods for a selector-only pool are ready.

    Uses the pool's selector from the database to find pods.
    """
    import time

    from core.database import get_db, Pool
    from core.models import ReadinessResponse
    from fastapi import HTTPException

    db = get_db()
    try:
        pool = (
            db.query(Pool)
            .filter(Pool.name == pool_name, Pool.namespace == namespace)
            .first()
        )
        if not pool:
            raise HTTPException(status_code=404, detail=f"Pool {pool_name} not found")

        specifier = json.loads(pool.specifier) if pool.specifier else {}
        selector = specifier.get("selector", {})
        if not selector:
            raise HTTPException(
                status_code=400, detail=f"Pool {pool_name} has no selector"
            )

        label_selector = ",".join(f"{k}={v}" for k, v in selector.items())

        start_time = time.time()

        while (time.time() - start_time) < timeout:
            pods = await run_in_threadpool(
                k8s.core_v1.list_namespaced_pod,
                namespace=namespace,
                label_selector=label_selector,
            )

            running_pods = [
                p
                for p in pods.items
                if p.status.phase == "Running"
                and all(c.ready for c in (p.status.container_statuses or []))
            ]

            if running_pods:
                return ReadinessResponse(
                    ready=True,
                    message=f"Found {len(running_pods)} ready pods",
                    resource_type="selector",
                    details={
                        "running_pods": len(running_pods),
                        "selector": selector,
                    },
                )

            # Check for pod errors
            for pod in pods.items:
                error = check_pod_for_errors(pod)
                if error:
                    return ReadinessResponse(
                        ready=False,
                        message=error,
                        resource_type="selector",
                        details={"error_type": "pod_error"},
                    )

            await asyncio.sleep(poll_interval)

        return ReadinessResponse(
            ready=False,
            message=f"No ready pods found for pool {pool_name} after {timeout} seconds",
            resource_type="selector",
            details={"timeout": timeout, "selector": selector},
        )
    finally:
        db.close()


# =============================================================================
# Error checking helpers
# =============================================================================


async def check_pods_for_errors(namespace: str, service_name: str) -> Optional[str]:
    """Check pods for a service for common errors."""
    try:
        pods = await run_in_threadpool(
            k8s.core_v1.list_namespaced_pod,
            namespace=namespace,
            label_selector=f"kubetorch.com/service={service_name}",
        )
        for pod in pods.items:
            error = check_pod_for_errors(pod)
            if error:
                return error
    except Exception as e:
        logger.warning(f"Error checking pods: {e}")
    return None


def check_pod_for_errors(pod) -> Optional[str]:
    """Check a single pod for common errors.

    Note: We intentionally don't treat "Unschedulable" as an error here.
    Unschedulable is often transient (cluster autoscaler, pods finishing, etc.)
    and should be handled by the client timeout, not as an immediate failure.
    """
    # Container status checks
    container_statuses = pod.status.container_statuses or []

    for cs in container_statuses:
        if cs.state.waiting:
            reason = cs.state.waiting.reason
            message = cs.state.waiting.message or ""

            # Image pull errors
            if reason in ("ImagePullBackOff", "ErrImagePull", "ErrImageNeverPull"):
                return f"Image pull error: {message}"

            # Container creation errors
            if reason == "CreateContainerError":
                return f"Container creation error: {message}"


async def check_revision_for_errors(
    namespace: str, revision_name: str
) -> Optional[str]:
    """Check a Knative revision for errors."""
    try:
        revision = await run_in_threadpool(
            k8s.custom_objects.get_namespaced_custom_object,
            group="serving.knative.dev",
            version="v1",
            namespace=namespace,
            plural="revisions",
            name=revision_name,
        )

        conditions = revision.get("status", {}).get("conditions", [])
        for condition in conditions:
            if condition.get("status") == "False":
                reason = condition.get("reason", "")
                message = condition.get("message", "")

                # Known error reasons
                if reason in ("ContainerMissing", "ContainerCreating"):
                    if "ImagePullBackOff" in message or "ErrImagePull" in message:
                        return f"Image pull error: {message}"

                if reason == "ProgressDeadlineExceeded":
                    return f"Revision timed out: {message}"

    except ApiException as e:
        if e.status != 404:
            logger.warning(f"Error checking revision {revision_name}: {e}")

    return None


async def check_ray_head_for_errors(
    namespace: str, head_pod_name: str
) -> Optional[str]:
    """Check Ray head pod logs for Ray-specific errors."""
    try:
        logs = await run_in_threadpool(
            k8s.core_v1.read_namespaced_pod_log,
            name=head_pod_name,
            namespace=namespace,
            tail_lines=100,
        )

        if "ray: not found" in logs or "ray: command not found" in logs:
            return "Ray is not installed in the container. Use a Ray-enabled image or install Ray in your container setup."

        if "Failed to start Ray server" in logs:
            return "Ray server failed to start. Check the head pod logs for details."

    except ApiException as e:
        if e.status != 404:
            logger.warning(f"Error checking Ray head pod logs: {e}")

    return None


async def check_replicaset_events_for_errors(
    namespace: str, service_name: str
) -> Optional[str]:
    """Check ReplicaSet events for creation errors like missing PriorityClass.

    When a ReplicaSet can't create pods (e.g., missing PriorityClass), there are
    no pods to check. We need to check ReplicaSet events for FailedCreate errors.
    """
    try:
        # Get ReplicaSets associated with this Deployment
        replicasets = await run_in_threadpool(
            k8s.apps_v1.list_namespaced_replica_set,
            namespace=namespace,
            label_selector=f"kubetorch.com/service={service_name}",
        )

        for rs in replicasets.items:
            rs_name = rs.metadata.name

            # Get events for this ReplicaSet
            events = await run_in_threadpool(
                k8s.core_v1.list_namespaced_event,
                namespace=namespace,
                field_selector=f"involvedObject.name={rs_name}",
            )

            for event in events.items:
                reason = event.reason or ""
                etype = event.type or ""
                message = event.message or ""

                if (
                    reason == "FailedCreate"
                    and etype == "Warning"
                    and "forbidden" in message.lower()
                ):
                    return f"ReplicaSet {rs_name} failed to create pods: {message}"

    except Exception as e:
        logger.warning(f"Error checking ReplicaSet events for {service_name}: {e}")

    return None


def _get_pool_for_delete(namespace: str, pool_name: str):
    """Fetch pool from DB for deletion (sync, must be called via run_in_threadpool)."""
    db = get_db()
    try:
        pool = (
            db.query(Pool)
            .filter(Pool.name == pool_name, Pool.namespace == namespace)
            .first()
        )
        if not pool:
            return None

        service_config = (
            json.loads(pool.service_config) if pool.service_config else None
        )
        svc_name = pool.name
        if service_config and "name" in service_config:
            svc_name = service_config["name"]

        return {
            "pool_id": pool.id,
            "svc_name": svc_name,
            "namespace": pool.namespace,
            "resource_kind": pool.resource_kind,
            "labels": pool.labels or {},
        }
    finally:
        db.close()


def _delete_pool_from_db(pool_id: int):
    """Delete pool from DB by ID (sync, must be called via run_in_threadpool)."""
    db = get_db()
    try:
        pool = db.query(Pool).filter(Pool.id == pool_id).first()
        if pool:
            db.delete(pool)
            db.commit()
            return True
        return False
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to delete pool from DB: {e}")
        return False
    finally:
        db.close()


async def delete_pool_helper(namespace: str, pool_name: str):
    """Delete a pool and its associated K8s services.

    Args:
        namespace (str): Kubernetes namespace
        pool_name (str): Name of the pool to delete

    Returns:
        dict: Status message with deleted pool info
    """
    from fastapi import HTTPException

    # Fetch pool info from DB (non-blocking)
    pool_info = await run_in_threadpool(_get_pool_for_delete, namespace, pool_name)
    if not pool_info:
        raise HTTPException(
            status_code=404,
            detail=f"Pool '{pool_name}' not found in namespace '{namespace}'",
        )

    svc_name = pool_info["svc_name"]
    pool_namespace = pool_info["namespace"]

    # Delete associated K8s services (non-blocking)
    for name in [svc_name, f"{svc_name}-headless"]:
        try:
            await run_in_threadpool(
                k8s.core_v1.delete_namespaced_service,
                name=name,
                namespace=pool_namespace,
            )
        except ApiException as e:
            if e.status != 404:
                logger.warning(
                    f"Failed to delete service {name} in namespace {namespace}: {e}"
                )

    # Delete pool from DB (non-blocking)
    deleted = await run_in_threadpool(_delete_pool_from_db, pool_info["pool_id"])
    if not deleted:
        raise HTTPException(status_code=500, detail="Failed to delete pool from DB")

    return {
        "status": "success",
        "message": f"Pool {pool_name} deleted from namespace {namespace}",
        "resource_kind": pool_info["resource_kind"],
        "labels": pool_info["labels"],
    }


def delete_pools_batch(pools_to_delete: List[dict]):
    """
    Delete multiple pools in one DB query.

    Args:
        pools_to_delete (list): List of pool dicts with 'name' and 'namespace' keys

    Returns:
        Number of rows deleted, or False on error
    """
    db = get_db()
    pools_ns_and_names = [
        (pool.get("name"), pool.get("namespace")) for pool in pools_to_delete
    ]

    try:
        if not pools_ns_and_names:
            return True

        # Build a filter condition for all (name, namespace) pairs
        conditions = [
            and_(Pool.name == pool_name, Pool.namespace == pool_namespace)
            for pool_name, pool_namespace in pools_ns_and_names
        ]

        # Delete all matching pools in one query
        result = (
            db.query(Pool)
            .filter(
                # Combine conditions with OR
                *conditions
            )
            .delete(synchronize_session=False)
        )

        db.commit()
        return result  # number of rows deleted
    except Exception:
        db.rollback()
        return False
    finally:
        db.close()
