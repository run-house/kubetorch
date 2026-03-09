import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from auth.middleware import check_name_prefix, check_namespace_access, get_current_user
from auth.models import AuthenticatedUser
from core import k8s
from core.database import get_db, Pool
from core.models import (
    PoolRequest,
    PoolResponse,
    ReadinessResponse,
    ServiceConfigName,
    ServiceConfigSelector,
    ServiceConfigUrl,
)
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from helpers.discover_helpers import discover_resource_from_pods
from helpers.pool_helpers import (
    build_service_for_pool,
    check_deployment_ready,
    check_knative_ready,
    check_raycluster_ready,
    check_selector_ready,
    check_trainjob_ready,
    create_service_if_not_exists,
)
from kubernetes.client.rest import ApiException
from routes.ws_pods import broadcast_reload_via_websocket, pod_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/controller", tags=["pool"])


@router.post("/pool", response_model=PoolResponse)
async def register_pool(
    req: PoolRequest,
    user: Optional[AuthenticatedUser] = Depends(get_current_user),
):
    """Register a compute pool.

    A pool is a logical group of pods that calls can be directed to.
    This endpoint:
    - Saves pool metadata to the database
    - Creates K8s Service(s) for routing to pods
    - Starts pod watcher to track pod IPs via label selector

    Use /apply to create the actual pods/workloads in the cluster.

    Args:
        req (PoolRequest): Pool registration request with namespace, name, and configuration
        user (AuthenticatedUser, optional): Authenticated user from auth middleware
    """
    # Check namespace access
    check_namespace_access(user, req.namespace)

    # Check name prefix restrictions
    check_name_prefix(user, req.name)

    now = datetime.now(timezone.utc)
    db = get_db()

    try:
        # ---- 0. Discover resource type if not provided ---------------------
        resource_kind = req.resource_kind
        resource_name = req.resource_name
        if resource_kind is None:
            # Discover from pods' ownerReferences
            discovered_kind, discovered_name = discover_resource_from_pods(
                req.namespace, req.specifier.selector
            )
            if discovered_kind:
                resource_kind = discovered_kind
                resource_name = discovered_name
                logger.info(
                    f"Discovered resource type for pool '{req.name}': "
                    f"{resource_kind}/{resource_name}"
                )

        # ---- 1. Upsert into SQLite -----------------------------------------
        existing = db.query(Pool).filter(Pool.name == req.name).first()

        specifier_json = req.specifier.model_dump_json()
        service_json = (
            req.service.model_dump_json()
            if req.service and hasattr(req.service, "model_dump_json")
            else json.dumps(req.service)
            if req.service
            else None
        )
        module_json = json.dumps(req.module.model_dump()) if req.module else None
        pool_metadata_json = (
            json.dumps(req.pool_metadata.model_dump()) if req.pool_metadata else None
        )
        labels_json = json.dumps(req.labels) if req.labels else None
        annotations_json = json.dumps(req.annotations) if req.annotations else None

        if existing:
            existing.namespace = req.namespace
            existing.specifier = specifier_json
            existing.service_config = service_json
            existing.dockerfile = req.dockerfile
            existing.module = module_json
            existing.pool_metadata = pool_metadata_json
            existing.server_port = req.server_port
            existing.resource_kind = resource_kind
            existing.resource_name = resource_name
            existing.labels = labels_json
            existing.annotations = annotations_json
            existing.updated_at = now
            existing.last_deployed_at = now
            logger.info(
                f"Updating existing pool in database (name={existing.name}, namespace={existing.namespace}, "
                f"specifier={existing.specifier}, service_config={existing.service_config}, "
                f"dockerfile={existing.dockerfile}, module={existing.module}, server_port={existing.server_port}, "
                f"resource_kind={existing.resource_kind}, resource_name={existing.resource_name},"
                f"annotations={existing.annotations}, labels={existing.labels}, "
                f"create_headless_service={req.create_headless_service})"
            )

        else:
            new_pool = Pool(
                name=req.name,
                namespace=req.namespace,
                specifier=specifier_json,
                service_config=service_json,
                dockerfile=req.dockerfile,
                module=module_json,
                pool_metadata=pool_metadata_json,
                server_port=req.server_port,
                resource_kind=resource_kind,
                resource_name=resource_name,
                labels=labels_json,
                annotations=annotations_json,
                created_at=now,
                updated_at=now,
                last_deployed_at=now,
            )
            db.add(new_pool)
            logger.info(
                f"Registering new pool in database (name={new_pool.name}, namespace={new_pool.namespace}, "
                f"specifier={new_pool.specifier}, service_config={new_pool.service_config}, "
                f"dockerfile={new_pool.dockerfile}, module={new_pool.module}, server_port={new_pool.server_port}, "
                f"resource_kind={new_pool.resource_kind}, resource_name={new_pool.resource_name}, "
                f"annotations={new_pool.annotations}, labels={new_pool.labels}, "
                f"create_headless_service={req.create_headless_service}))"
            )

        db.commit()

        # ---- 2. Ensure service exists ----------------------------------------
        pool_selector = req.specifier.selector.copy()

        # Check if user provided their own service URL - skip service creation
        if isinstance(req.service, ServiceConfigUrl):
            service_url = req.service.url
            logger.info(
                f"Pool '{req.name}' using user-provided service URL: {service_url}"
            )
        else:
            # Determine service selector - may be different from pool selector
            # (e.g., for Ray we track all pods but route to head node only)
            if isinstance(req.service, ServiceConfigSelector):
                service_selector = req.service.selector.copy()
            else:
                service_selector = pool_selector.copy()

            # Determine service name
            svc_name = req.name
            if isinstance(req.service, ServiceConfigName):
                svc_name = req.service.name

            # Ensure service exists (idempotent - safe for both new pools and updates)
            # This handles cases where service was deleted but pool entry remains
            is_knative = resource_kind == "KnativeService"

            if not is_knative:
                # Build and create regular client-facing service
                logger.info(f"Ensuring service exists for pool {svc_name}")
                service_manifest = build_service_for_pool(
                    pool_name=svc_name,
                    namespace=req.namespace,
                    server_port=req.server_port or 32300,
                    labels=req.labels,
                    annotations=req.annotations,
                    selector=service_selector,  # Use service selector for routing
                    headless=False,
                )
                await create_service_if_not_exists(
                    service_manifest, req.namespace, svc_name
                )

                # Create headless service for distributed pod discovery (only if requested)
                if req.create_headless_service:
                    logger.info(f"Ensuring headless service exists for pool {svc_name}")
                    headless_manifest = build_service_for_pool(
                        pool_name=svc_name,
                        namespace=req.namespace,
                        server_port=req.server_port or 32300,
                        labels=req.labels,
                        annotations=req.annotations,
                        selector=pool_selector,  # Use pool selector for pod discovery
                        headless=True,
                    )
                    await create_service_if_not_exists(
                        headless_manifest, req.namespace, f"{svc_name}-headless"
                    )
            else:
                logger.info(
                    f"Pool '{req.name}' is KnativeService, skipping service creation (Knative manages services)"
                )

            service_url = f"{svc_name}.{req.namespace}.svc.cluster.local"

        # ---- 3. Broadcast reload config to pods via WebSocket ----------------
        # Always broadcast if module info is provided - handles both:
        # 1. Updates/reloads of existing pools
        # 2. Selector-only mode where pods already exist
        # If no pods exist yet, broadcast_reload_to_pods returns warning status
        broadcast_result = None
        if req.module:
            logger.info(f"Broadcasting reload to pod(s) for {req.name}")

            # For selector-only mode: find pods that match the selector but may have
            # connected without KT_SERVICE (registered with empty service_name).
            # Query K8s for matching pods and reassign them to this service.
            # Retry a few times since pods may still be connecting via WebSocket.
            if pool_selector:
                try:
                    label_selector = ",".join(
                        f"{k}={v}" for k, v in pool_selector.items()
                    )
                    pods_result = await run_in_threadpool(
                        k8s.core_v1.list_namespaced_pod,
                        namespace=req.namespace,
                        label_selector=label_selector,
                    )
                    k8s_pod_names = [p.metadata.name for p in pods_result.items]

                    if k8s_pod_names:
                        # Retry finding connected pods - they may still be connecting
                        # K8s pod "ready" doesn't mean WebSocket is connected yet
                        found_pods = []
                        max_retries = 10
                        retry_delay = 1.0  # seconds

                        for attempt in range(max_retries):
                            found_pods = await pod_manager.find_pods_by_names(
                                req.namespace, k8s_pod_names
                            )
                            if len(found_pods) >= len(k8s_pod_names):
                                # All pods connected
                                break
                            if found_pods:
                                # Some pods connected - continue waiting for others
                                logger.debug(
                                    f"Found {len(found_pods)}/{len(k8s_pod_names)} pods connected, "
                                    f"waiting for more (attempt {attempt + 1}/{max_retries})"
                                )
                            else:
                                logger.debug(
                                    f"No pods connected yet, waiting "
                                    f"(attempt {attempt + 1}/{max_retries})"
                                )
                            await asyncio.sleep(retry_delay)

                        if found_pods:
                            # Reassign pods that are under wrong service_name
                            pods_to_reassign = [
                                p for p in found_pods if p.service_name != req.name
                            ]
                            if pods_to_reassign:
                                logger.info(
                                    f"Reassigning {len(pods_to_reassign)} pods to service '{req.name}'"
                                )
                                await pod_manager.reassign_pods_to_service(
                                    pods_to_reassign, req.name
                                )
                        else:
                            logger.warning(
                                f"No pods connected after {max_retries} retries for "
                                f"selector {pool_selector}"
                            )
                except Exception as e:
                    logger.warning(
                        f"Failed to reassign pods for selector-only mode: {e}"
                    )

            module_info = req.module.model_dump(exclude_none=True)
            pool_metadata = req.pool_metadata

            # Push to pods connected via WebSocket
            ws_result = await broadcast_reload_via_websocket(
                namespace=req.namespace,
                service_name=req.name,
                module_info=module_info,
                deployed_as_of=now.isoformat(),
                deployment_mode=pool_metadata.deployment_mode
                if pool_metadata
                else None,
                distributed_config=pool_metadata.distributed_config
                if pool_metadata
                else None,
                runtime_config=pool_metadata.runtime_config.model_dump(
                    exclude_none=True
                )
                if pool_metadata and pool_metadata.runtime_config
                else None,
                username=pool_metadata.username if pool_metadata else None,
            )
            logger.info(f"WebSocket broadcast result: {ws_result}")

            broadcast_result = {
                "status": ws_result.get("status", "success"),
                "message": ws_result.get("message", ""),
                "reloaded": ws_result.get("sent", 0),
            }

        # Determine status based on broadcast result
        if broadcast_result:
            if broadcast_result.get("status") == "error":
                status = "error"
                message = f"Pool registered but broadcast failed: {broadcast_result.get('message')}"
            elif broadcast_result.get("status") == "warning":
                status = "warning"
                message = f"Pool registered but no pods found: {broadcast_result.get('message')}"
            elif broadcast_result.get("status") == "partial":
                status = "partial"
                message = f"Pool registered, {broadcast_result.get('message')}"
            else:
                status = "success"
                message = f"Pool registered, {broadcast_result.get('message')}"
        else:
            status = "success"
            message = "Pool registered"

        return PoolResponse(
            name=req.name,
            namespace=req.namespace,
            status=status,
            message=message,
            service_url=service_url,
            specifier=req.specifier.model_dump(),
            service_config=req.service.model_dump()
            if req.service and hasattr(req.service, "model_dump")
            else req.service,
            dockerfile=req.dockerfile,
            module=req.module.model_dump() if req.module else None,
            pool_metadata=req.pool_metadata.model_dump() if req.pool_metadata else None,
            server_port=req.server_port,
            labels=req.labels,
            annotations=req.annotations,
            resource_kind=resource_kind,
            resource_name=resource_name,
            created_at=now.isoformat(),
            updated_at=now.isoformat(),
            last_deployed_at=now.isoformat(),
        )

    except Exception as e:
        db.rollback()
        logger.error(f"Pool registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.get("/pools/{namespace}")
async def list_pools(
    namespace: str,
    user: Optional[AuthenticatedUser] = Depends(get_current_user),
):
    """List all registered pools.

    Args:
        namespace (str): Kubernetes namespace
        user (AuthenticatedUser, optional): Authenticated user from auth middleware
    """
    # Check namespace access
    check_namespace_access(user, namespace)

    db = get_db()
    try:
        pools = db.query(Pool).filter(Pool.namespace == namespace).all()
        return {"pools": [p.to_dict() for p in pools]}
    finally:
        db.close()


@router.get("/pool/{namespace}/{pool_name}", response_model=PoolResponse)
async def get_pool(
    namespace: str,
    pool_name: str,
    user: Optional[AuthenticatedUser] = Depends(get_current_user),
):
    """Get information about a registered pool.

    Args:
        namespace (str): Kubernetes namespace
        pool_name (str): Name of the pool to retrieve
        user (AuthenticatedUser, optional): Authenticated user from auth middleware
    """
    # Check namespace access
    check_namespace_access(user, namespace)

    db = get_db()
    try:
        pool = (
            db.query(Pool)
            .filter(Pool.name == pool_name, Pool.namespace == namespace)
            .first()
        )
        if not pool:
            raise HTTPException(
                status_code=404,
                detail=f"Pool '{pool_name}' not found in namespace '{namespace}'",
            )

        pool_data = pool.to_dict()
        service_config = pool_data.get("service_config")

        # Determine service URL and pod IPs
        pod_ips = None
        if service_config and "url" in service_config:
            service_url = service_config["url"]
        else:
            # label_selector pool
            svc_name = pool.name
            if service_config and "name" in service_config:
                svc_name = service_config["name"]
            service_url = f"{svc_name}.{pool.namespace}.svc.cluster.local"
            # Get pod IPs from connected WebSocket pods
            connected_pods = await pod_manager.get_pods_for_service(
                pool.namespace, pool_name
            )
            pod_ips = [p.pod_ip for p in connected_pods if p.pod_ip] or None

        return PoolResponse(
            **pool_data,
            status="active",
            message="Pool found",
            service_url=service_url,
            pod_ips=pod_ips,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get pool: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.delete("/pool/{namespace}/{pool_name}")
async def delete_pool(
    namespace: str,
    pool_name: str,
    user: Optional[AuthenticatedUser] = Depends(get_current_user),
):
    """Delete a registered pool.

    Args:
        namespace (str): Kubernetes namespace
        pool_name (str): Name of the pool to delete
        user (AuthenticatedUser, optional): Authenticated user from auth middleware
    """
    # Check namespace access
    check_namespace_access(user, namespace)

    db = get_db()
    try:
        pool = (
            db.query(Pool)
            .filter(Pool.name == pool_name, Pool.namespace == namespace)
            .first()
        )
        if not pool:
            raise HTTPException(
                status_code=404,
                detail=f"Pool '{pool_name}' not found in namespace '{namespace}'",
            )

        # Delete associated K8s services
        service_config = (
            json.loads(pool.service_config) if pool.service_config else None
        )
        svc_name = pool.name
        if service_config and "name" in service_config:
            svc_name = service_config["name"]

        # Delete regular and headless services
        for name in [svc_name, f"{svc_name}-headless"]:
            try:
                await run_in_threadpool(
                    k8s.core_v1.delete_namespaced_service,
                    name=name,
                    namespace=pool.namespace,
                )
                logger.info(f"Deleted service {name}")
            except ApiException as e:
                if e.status != 404:
                    logger.warning(
                        f"Failed to delete service {name} in namespace {namespace}: {e}"
                    )

        db.delete(pool)
        db.commit()
        logger.info(f"Deleted pool '{pool_name}'")

        return {
            "status": "success",
            "message": f"Pool {pool_name} deleted from namespace {namespace}",
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to delete pool: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.get("/debug/connections")
async def debug_connections():
    """Debug endpoint to see WebSocket connection state.

    Shows all pools and their connected pods via WebSocket.
    """
    db = get_db()
    try:
        pools = db.query(Pool).all()

        result = {}
        for pool in pools:
            specifier = json.loads(pool.specifier) if pool.specifier else {}
            pool_name = pool.name

            # Get connected pods from WebSocket manager
            connected_pods = await pod_manager.get_pods_for_service(
                pool.namespace, pool_name
            )

            result[pool_name] = {
                "namespace": pool.namespace,
                "selector": specifier.get("selector", {}),
                "connected_pods": [
                    {
                        "name": p.pod_name,
                        "ip": p.pod_ip,
                        "connected_at": p.connected_at.isoformat(),
                    }
                    for p in connected_pods
                ],
                "pod_count": len(connected_pods),
            }

        return result
    finally:
        db.close()


# =============================================================================
# Readiness Check Endpoint
# =============================================================================


@router.get("/check-ready/{namespace}/{name}", response_model=ReadinessResponse)
async def check_ready(
    namespace: str,
    name: str,
    resource_type: str = Query(
        ...,
        description="Resource type: deployment, knative, raycluster, pytorchjob, etc.",
    ),
    timeout: int = Query(300, description="Timeout in seconds"),
    poll_interval: int = Query(2, description="Poll interval in seconds"),
    user: Optional[AuthenticatedUser] = Depends(get_current_user),
):
    """Check if a resource is ready.

    Polls the resource until it's ready or timeout is reached.
    Returns immediately if already ready.

    Args:
        namespace (str): Kubernetes namespace
        name (str): Resource name
        resource_type (str): Type of resource (deployment, knative, raycluster, pytorchjob, tfjob, mxjob, xgboostjob)
        timeout (int): Maximum time to wait in seconds
        poll_interval (int): Time between checks in seconds
        user (AuthenticatedUser, optional): Authenticated user from auth middleware

    Returns:
        ReadinessResponse with ready status and details
    """
    # Check namespace access
    check_namespace_access(user, namespace)
    resource_type = resource_type.lower()

    try:
        if resource_type == "deployment":
            return await check_deployment_ready(namespace, name, timeout, poll_interval)
        elif resource_type == "knative":
            return await check_knative_ready(namespace, name, timeout, poll_interval)
        elif resource_type == "raycluster":
            return await check_raycluster_ready(namespace, name, timeout, poll_interval)
        elif resource_type in ["pytorchjob", "tfjob", "mxjob", "xgboostjob"]:
            return await check_trainjob_ready(
                namespace, name, resource_type, timeout, poll_interval
            )
        else:
            # Selector-based check for selector-only mode and BYO manifests
            # (StatefulSet, DaemonSet, Job, etc.)
            return await check_selector_ready(namespace, name, timeout, poll_interval)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed for {resource_type}/{name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
