import logging
from typing import Dict, Tuple

from core import k8s
from kubernetes.client.exceptions import ApiException
from kubernetes.dynamic.exceptions import ConflictError

logger = logging.getLogger(__name__)


# TODO: Remove once older SDK versions with manifest bugs are no longer supported
def _remove_null_values(d: dict) -> None:
    """Recursively remove keys with null values from a dict.

    Server-Side Apply (used in /apply) rejects null for object-type fields. Older SDK versions
    may send nodeSelector: null when it should be omitted entirely.
    """
    keys_to_remove = [k for k, v in d.items() if v is None]
    for k in keys_to_remove:
        del d[k]
    for v in d.values():
        if isinstance(v, dict):
            _remove_null_values(v)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    _remove_null_values(item)


# TODO: Remove once older SDK versions with manifest bugs are no longer supported
def _sanitize_manifest(manifest: dict) -> dict:
    """Remove fields that aren't valid in the CRD schema.

    Provides backwards compatibility with older SDK versions.
    """
    kind = manifest.get("kind")

    if kind == "RayCluster":
        spec = manifest.get("spec", {})

        # headGroupSpec.replicas is not valid - head is always 1 replica
        head_group = spec.get("headGroupSpec", {})
        if "replicas" in head_group:
            del head_group["replicas"]

        # headServicePorts is not supported in older Ray operator versions
        if "headServicePorts" in spec:
            del spec["headServicePorts"]

    _remove_null_values(manifest)

    return manifest


def apply_resource_sync(manifest: dict, namespace: str) -> Tuple[Dict, str]:
    """
    Apply a resource using the dynamic client (like kubectl apply).
    Works with any resource type - Deployments, Services, CRDs, etc.
    """
    # BC: Sanitize manifest to support older SDK versions
    manifest = _sanitize_manifest(manifest)

    dyn = k8s.dynamic

    api_version = manifest.get("apiVersion")
    kind = manifest.get("kind")
    name = manifest.get("metadata", {}).get("name")

    # Get the API resource dynamically
    api = dyn.resources.get(api_version=api_version, kind=kind)

    try:
        # Try to create first
        if api.namespaced:
            result = api.create(body=manifest, namespace=namespace)
        else:
            result = api.create(body=manifest)
        return result.to_dict(), "created"
    except ConflictError:
        # Already exists - use strategic merge patch to update
        # This preserves immutable fields (like Deployment selectors) from the existing resource
        logger.info(f"Resource {kind}/{name} already exists, updating via patch...")
        try:
            if api.namespaced:
                result = api.patch(
                    body=manifest,
                    name=name,
                    namespace=namespace,
                    content_type="application/strategic-merge-patch+json",
                )
            else:
                result = api.patch(
                    body=manifest,
                    name=name,
                    content_type="application/strategic-merge-patch+json",
                )
            return result.to_dict(), "updated"
        except ApiException as e:
            # If patch fails (e.g., CRDs that don't support strategic merge),
            # fall back to returning the existing resource
            logger.warning(
                f"Patch failed for {kind}/{name}: {e.reason}, fetching existing resource"
            )
            if api.namespaced:
                result = api.get(name=name, namespace=namespace)
            else:
                result = api.get(name=name)
            return result.to_dict(), "exists"
