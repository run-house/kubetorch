import logging

from core.models import ApplyRequest, DeployRequest, DeployResponse, PoolRequest
from fastapi import APIRouter
from routes.apply import apply_resource
from routes.pool import register_pool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/controller", tags=["deploy"])


@router.post("/deploy", response_model=DeployResponse)
async def deploy_resource(req: DeployRequest):
    """Deploy a K8s resource and register it as a pool.

    This endpoint combines:
    1. /apply - Create K8s resources (Deployment, RayCluster, PyTorchJob, etc.)
    2. /pool - Register the pool for pod tracking and service routing

    Use /pool alone for selector-only mode (existing pods, no manifest to apply).
    """
    # Apply manifest to create K8s resources
    apply_req = ApplyRequest(
        service_name=req.service_name,
        namespace=req.namespace,
        resource_type=req.resource_type,
        resource_manifest=req.resource_manifest,
    )

    logger.info(
        f"Deploy: applying {req.resource_type} manifest for {req.service_name} in {req.namespace}"
    )
    apply_response = await apply_resource(apply_req)

    if apply_response.status == "error":
        logger.error(
            f"Deploy: apply failed for {req.service_name}: {apply_response.message}"
        )
        return DeployResponse(
            service_name=req.service_name,
            namespace=req.namespace,
            resource_type=req.resource_type,
            apply_status="error",
            apply_message=apply_response.message,
            resource=apply_response.resource,
            pool_status="skipped",
            pool_message="Apply failed, pool registration skipped",
        )

    logger.info(f"Deploy: apply succeeded for {req.service_name}, registering pool")

    # Register pool for pod tracking and service creation
    resource_kind_map = {
        "deployment": "Deployment",
        "knative": "KnativeService",
        "raycluster": "RayCluster",
        "pytorchjob": "PyTorchJob",
        "tfjob": "TFJob",
        "mxjob": "MXJob",
        "xgboostjob": "XGBoostJob",
    }
    resource_kind = resource_kind_map.get(req.resource_type.lower(), req.resource_type)

    pool_req = PoolRequest(
        name=req.service_name,
        namespace=req.namespace,
        specifier=req.specifier,
        service=req.service,
        dockerfile=req.dockerfile,
        module=req.module,
        pool_metadata=req.pool_metadata,
        server_port=req.server_port,
        labels=req.labels,
        annotations=req.annotations,
        resource_kind=resource_kind,
        resource_name=req.service_name,
        create_headless_service=req.create_headless_service,
    )

    pool_response = await register_pool(pool_req)

    logger.info(
        f"Deploy: pool registration completed for {req.service_name} with status {pool_response.status}"
    )

    return DeployResponse(
        service_name=req.service_name,
        namespace=req.namespace,
        resource_type=req.resource_type,
        apply_status=apply_response.status,
        apply_message=apply_response.message,
        resource=apply_response.resource,
        pool_status=pool_response.status,
        pool_message=pool_response.message,
        service_url=pool_response.service_url,
        resource_kind=pool_response.resource_kind,
        resource_name=pool_response.resource_name,
        created_at=pool_response.created_at,
        updated_at=pool_response.updated_at,
        last_deployed_at=pool_response.last_deployed_at,
    )
