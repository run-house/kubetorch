import copy
import json
import logging
from typing import Optional

from auth.middleware import check_name_prefix, check_namespace_access, get_current_user
from auth.models import AuthenticatedUser
from core.models import ApplyRequest, ApplyResponse
from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from helpers.apply_helpers import apply_resource_sync

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/controller", tags=["apply"])


@router.post("/apply", response_model=ApplyResponse)
async def apply_resource(
    req: ApplyRequest,
    user: Optional[AuthenticatedUser] = Depends(get_current_user),
):
    """Apply a K8s manifest (operates like `kubectl apply`).

    Works with any Kubernetes resource type - built-in resources (Deployments,
    Services, ConfigMaps) and custom resources (Knative, RayClusters, etc.).
    """
    check_namespace_access(user, req.namespace)
    check_name_prefix(user, req.service_name)

    resource_manifest = copy.deepcopy(req.resource_manifest)

    incoming_replicas = resource_manifest.get("spec", {}).get("replicas", "NOT_SET")
    logger.info(
        f"Apply request for {req.service_name} (type={req.resource_type}): "
        f"incoming_replicas={incoming_replicas}"
    )

    try:
        resource_dict, action = await run_in_threadpool(
            apply_resource_sync,
            resource_manifest,
            req.namespace,
        )

        logger.info(
            f"Applied ({action}) resource {req.service_name} in namespace {req.namespace}"
        )

        return ApplyResponse(
            service_name=req.service_name,
            namespace=req.namespace,
            resource_type=req.resource_type,
            status="success",
            message=f"Resource {action}",
            resource=resource_dict,
        )

    except Exception as e:
        logger.error(f"Apply failed: {e}")

        # Try to extract K8s API error details
        error_message = str(e)
        if hasattr(e, "body") and e.body:
            try:
                body = json.loads(e.body)
                if "message" in body:
                    error_message = body["message"]
            except (json.JSONDecodeError, TypeError):
                pass

        return ApplyResponse(
            service_name=req.service_name,
            namespace=req.namespace,
            resource_type=req.resource_type,
            status="error",
            message=error_message,
        )
