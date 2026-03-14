"""
WebSocket endpoint for pod connections.

Pods connect to this endpoint to:
1. Register themselves and request their metadata
2. Receive reload pushes when /pool is called

This replaces the push-to-pods model where the controller connected to each pod.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from core import k8s
from core.database import get_db, Pool

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

# Maximum number of concurrent websocket sends during broadcast.
# Higher values = faster broadcast but more memory/CPU pressure.
# Can be overridden via environment variable.
BROADCAST_CONCURRENCY = int(os.getenv("BROADCAST_CONCURRENCY", "500"))

router = APIRouter(prefix="/controller", tags=["websocket"])


@dataclass
class ConnectedPod:
    """Represents a connected pod."""

    websocket: WebSocket
    pod_name: str
    pod_ip: str
    namespace: str
    service_name: str
    connected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PodConnectionManager:
    """Manages WebSocket connections from pods.

    Pods are grouped by (namespace, service_name) for efficient broadcast.
    """

    def __init__(self):
        # Map of (namespace, service_name) -> dict of pod_name -> ConnectedPod
        self._connections: Dict[tuple, Dict[str, ConnectedPod]] = {}
        self._lock = asyncio.Lock()
        # Track pending reload acks: (namespace, service_name, pod_name) -> Future
        self._pending_acks: Dict[tuple, asyncio.Future] = {}

    async def connect(self, pod: ConnectedPod):
        """Register a new pod connection."""
        key = (pod.namespace, pod.service_name)
        async with self._lock:
            if key not in self._connections:
                self._connections[key] = {}
            self._connections[key][pod.pod_name] = pod
            total = len(self._connections[key])
        logger.info(
            f"Pod connected: {pod.pod_name} ({pod.namespace}/{pod.service_name}), "
            f"total for service: {total}"
        )

    async def disconnect(self, pod: ConnectedPod):
        """Remove a pod connection."""
        key = (pod.namespace, pod.service_name)
        async with self._lock:
            if key in self._connections:
                self._connections[key].pop(pod.pod_name, None)
                if not self._connections[key]:
                    del self._connections[key]
        logger.info(f"Pod disconnected: {pod.pod_name}")

    async def get_pods_for_service(
        self, namespace: str, service_name: str
    ) -> List[ConnectedPod]:
        """Get all connected pods for a service.

        Returns a snapshot of the pod list. The lock is released before returning,
        so callers can safely iterate without blocking other operations.
        """
        key = (namespace, service_name)
        async with self._lock:
            pods_dict = self._connections.get(key, {})
            pods = list(pods_dict.values())
            all_keys = list(self._connections.keys())
        # Log after releasing lock
        logger.info(
            f"Looking up pods for key={key}, found={len(pods)}, all_keys={all_keys}"
        )
        return pods

    async def find_pods_by_names(
        self, namespace: str, pod_names: List[str]
    ) -> List[ConnectedPod]:
        """Find connected pods by their names, regardless of service_name.

        This is used for selector-only mode where pods may have connected
        without knowing their service_name (registered with empty string).

        Args:
            namespace: K8s namespace to search in
            pod_names: List of pod names to find

        Returns:
            List of matching ConnectedPod objects (snapshot, safe to iterate)
        """
        pod_names_set = set(pod_names)
        async with self._lock:
            found_pods = [
                pod
                for key, pods_dict in self._connections.items()
                if key[0] == namespace
                for pod_name, pod in pods_dict.items()
                if pod_name in pod_names_set
            ]
        logger.info(
            f"Found {len(found_pods)} connected pods matching names in {namespace}"
        )
        return found_pods

    async def reassign_pods_to_service(
        self, pods: List[ConnectedPod], new_service_name: str
    ):
        """Reassign pods to a new service_name.

        Used when pods connected with empty service_name and we later
        discover which pool they belong to via selector matching.

        Args:
            pods: List of pods to reassign
            new_service_name: The service name to assign them to
        """
        async with self._lock:
            for pod in pods:
                old_key = (pod.namespace, pod.service_name)
                new_key = (pod.namespace, new_service_name)

                # Remove from old location
                if old_key in self._connections:
                    self._connections[old_key].pop(pod.pod_name, None)
                    if not self._connections[old_key]:
                        del self._connections[old_key]

                # Update pod's service_name
                pod.service_name = new_service_name

                # Add to new location
                if new_key not in self._connections:
                    self._connections[new_key] = {}
                self._connections[new_key][pod.pod_name] = pod

                logger.info(
                    f"Reassigned pod {pod.pod_name} from service '{old_key[1]}' to '{new_service_name}'"
                )

    def resolve_ack(
        self, namespace: str, service_name: str, pod_name: str, result: dict
    ):
        """Resolve a pending ack future when pod sends reload_ack."""
        key = (namespace, service_name, pod_name)
        future = self._pending_acks.pop(key, None)
        if future and not future.done():
            future.set_result(result)
            logger.debug(f"Resolved ack for {pod_name}")

    async def broadcast_to_service(
        self,
        namespace: str,
        service_name: str,
        message: dict,
        wait_for_ack: bool = True,
        ack_timeout: float = 30.0,
    ) -> dict:
        """Broadcast a message to all pods for a service.

        Sends to all pods in parallel using a semaphore to limit concurrency.

        Args:
            namespace: K8s namespace
            service_name: Service/pool name
            message: Message to broadcast
            wait_for_ack: If True and message is a reload, wait for pods to acknowledge
            ack_timeout: Timeout in seconds to wait for acknowledgments

        Returns:
            Dict with status, message, success count, errors
        """
        pods = await self.get_pods_for_service(namespace, service_name)
        if not pods:
            return {
                "status": "warning",
                "message": f"No connected pods for {namespace}/{service_name}",
                "sent": 0,
                "total": 0,
            }

        message_json = json.dumps(message)
        is_reload = message.get("action") == "reload"

        if is_reload:
            callable_name = (message.get("module") or {}).get(
                "cls_or_fn_name", "unknown"
            )
            logger.info(
                f"Broadcasting reload for callable={callable_name} to {len(pods)} pods"
            )

        # Create futures for acks if this is a reload message
        ack_futures = {}
        if is_reload and wait_for_ack:
            for pod in pods:
                key = (namespace, service_name, pod.pod_name)
                future = asyncio.get_running_loop().create_future()
                self._pending_acks[key] = future
                ack_futures[pod.pod_name] = future

        # Send messages to all pods in parallel with concurrency limit
        semaphore = asyncio.Semaphore(BROADCAST_CONCURRENCY)
        errors = []

        async def send_to_pod(pod: ConnectedPod) -> bool:
            """Send message to a single pod. Returns True on success."""
            async with semaphore:
                try:
                    await asyncio.wait_for(
                        pod.websocket.send_text(message_json),
                        timeout=5.0,  # Per-pod send timeout
                    )
                    return True
                except Exception as e:
                    error_msg = f"Failed to send to {pod.pod_name}: {e}"
                    logger.warning(error_msg)
                    errors.append(error_msg)
                    # Clean up pending ack for failed send
                    key = (namespace, service_name, pod.pod_name)
                    self._pending_acks.pop(key, None)
                    if pod.pod_name in ack_futures:
                        ack_futures[pod.pod_name].cancel()
                        del ack_futures[pod.pod_name]
                    return False

        # Send to all pods concurrently
        results = await asyncio.gather(*[send_to_pod(pod) for pod in pods])
        sent = sum(1 for r in results if r)

        # Wait for acknowledgments if this is a reload
        acked = 0
        ack_errors = []
        if is_reload and wait_for_ack and ack_futures:
            try:
                # Wait for all acks with timeout
                done, pending = await asyncio.wait(
                    ack_futures.values(),
                    timeout=ack_timeout,
                    return_when=asyncio.ALL_COMPLETED,
                )

                # Count successful acks
                for pod_name, future in ack_futures.items():
                    if future in done and not future.cancelled():
                        result = future.result()
                        if result.get("status") == "ok":
                            acked += 1
                        else:
                            ack_errors.append(
                                f"{pod_name}: {result.get('message', 'error')}"
                            )
                    elif future in pending:
                        ack_errors.append(f"{pod_name}: timeout")
                        # Clean up pending futures
                        key = (namespace, service_name, pod_name)
                        self._pending_acks.pop(key, None)
                        future.cancel()

            except Exception as e:
                logger.error(f"Error waiting for acks: {e}")
                ack_errors.append(f"wait error: {e}")

        # Determine status
        if is_reload and wait_for_ack:
            if acked == sent and sent > 0:
                status = "success"
                message_text = f"Sent and acknowledged by {acked}/{len(pods)} pods"
            elif acked > 0:
                status = "partial"
                message_text = f"Acknowledged by {acked}/{sent} pods"
            elif sent > 0:
                status = "partial"
                message_text = f"Sent to {sent}/{len(pods)} pods but no acks received"
            else:
                status = "error"
                message_text = "Failed to send to any pods"
        else:
            status = "success" if not errors else ("partial" if sent > 0 else "error")
            message_text = f"Sent to {sent}/{len(pods)} pods"

        all_errors = errors + ack_errors if ack_errors else errors

        return {
            "status": status,
            "message": message_text,
            "sent": sent,
            "acked": acked if is_reload and wait_for_ack else None,
            "total": len(pods),
            "errors": all_errors if all_errors else None,
        }


# Global connection manager
pod_manager = PodConnectionManager()


async def find_pool_for_pod(namespace: str, pod_name: str) -> Optional[str]:
    """Find a pool that matches a pod's labels.

    Used for selector-only mode where pods connect without knowing their service name.
    Queries K8s for the pod's labels, then finds a pool whose selector matches.

    Args:
        namespace: K8s namespace
        pod_name: Name of the pod

    Returns:
        Pool name if found, None otherwise
    """
    db = get_db()
    try:
        # Get pod's labels from K8s
        try:
            pod = await run_in_threadpool(
                k8s.core_v1.read_namespaced_pod,
                name=pod_name,
                namespace=namespace,
            )
            pod_labels = pod.metadata.labels or {}
        except Exception as e:
            logger.warning(f"Failed to get pod {pod_name} labels: {e}")
            return None

        if not pod_labels:
            logger.debug(f"Pod {pod_name} has no labels")
            return None

        # Find pools in this namespace that have a selector
        pools = db.query(Pool).filter(Pool.namespace == namespace).all()

        for pool in pools:
            if not pool.specifier:
                continue

            try:
                specifier = json.loads(pool.specifier)
                selector = specifier.get("selector", {})
                if not selector:
                    continue

                # Check if all selector labels match pod labels
                if all(pod_labels.get(k) == v for k, v in selector.items()):
                    logger.info(
                        f"Found matching pool '{pool.name}' for pod {pod_name} "
                        f"(selector: {selector})"
                    )
                    return pool.name
            except (json.JSONDecodeError, TypeError):
                continue

        logger.debug(f"No pool found matching pod {pod_name} labels: {pod_labels}")
        return None
    except Exception as e:
        logger.error(f"Error finding pool for pod {pod_name}: {e}")
        return None
    finally:
        db.close()


async def try_match_pod_to_pool(connected_pod: ConnectedPod) -> Optional[str]:
    """Try to find and assign a matching pool for a pod with empty service_name.

    If the pod has no service_name set, queries K8s for the pod's labels and finds
    a pool whose selector matches. If found, reassigns the pod to that service.

    Args:
        connected_pod: The connected pod to match

    Returns:
        The matched service name, or the pod's existing service_name if already set
    """
    if connected_pod.service_name:
        return connected_pod.service_name

    matched_pool = await find_pool_for_pod(
        connected_pod.namespace, connected_pod.pod_name
    )
    if not matched_pool:
        return None

    # Reassign pod to the matched service
    old_key = (connected_pod.namespace, connected_pod.service_name)
    connected_pod.service_name = matched_pool

    async with pod_manager._lock:
        if old_key in pod_manager._connections:
            pod_manager._connections[old_key].pop(connected_pod.pod_name, None)
            if not pod_manager._connections[old_key]:
                del pod_manager._connections[old_key]

        new_key = (connected_pod.namespace, connected_pod.service_name)
        if new_key not in pod_manager._connections:
            pod_manager._connections[new_key] = {}
        pod_manager._connections[new_key][connected_pod.pod_name] = connected_pod

    logger.info(f"Reassigned pod {connected_pod.pod_name} to service '{matched_pool}'")
    return matched_pool


def get_pool_metadata(namespace: str, service_name: str) -> Optional[dict]:
    """Look up pool metadata from database.

    Returns:
        Dict with module info, service_name, deployment_mode, runtime_config, etc.
        None if pool not found.
    """
    db = get_db()
    try:
        # Look up pool by name (service_name = pool name)
        pool = (
            db.query(Pool)
            .filter(Pool.name == service_name, Pool.namespace == namespace)
            .first()
        )

        if not pool:
            logger.warning(f"Pool not found: {namespace}/{service_name}")
            return None

        # Parse stored module info
        module_info = json.loads(pool.module) if pool.module else {}
        pointers = module_info.get("pointers") or {}
        pool_metadata = json.loads(pool.pool_metadata) if pool.pool_metadata else {}

        # Determine service DNS based on distributed config
        distributed_config = pool_metadata.get("distributed_config")
        if distributed_config:
            service_dns = f"{pool.name}-headless.{namespace}.svc.cluster.local"
        else:
            service_dns = f"{pool.name}.{namespace}.svc.cluster.local"

        # Extract runtime config for the pod
        runtime_config = pool_metadata.get("runtime_config") or {}

        return {
            "action": "metadata",
            "service_name": pool.name,
            "namespace": pool.namespace,
            "service_dns": service_dns,
            "deployment_mode": pool_metadata.get("deployment_mode"),
            "username": pool_metadata.get("username"),
            "module": {
                "module_name": pointers.get("module_name"),
                "cls_or_fn_name": pointers.get("cls_or_fn_name"),
                "file_path": pointers.get("file_path"),
                "project_root": pointers.get("project_root"),
                "init_args": pointers.get("init_args"),
                "callable_type": module_info.get("type", "fn"),
                "distributed_config": distributed_config,
            },
            # Runtime config flows via WebSocket - can change between deploys
            "runtime_config": runtime_config,
        }
    except Exception as e:
        logger.error(f"Error looking up pool {namespace}/{service_name}: {e}")
        return None
    finally:
        db.close()


@router.websocket("/ws/pods")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for pod connections.

    Protocol:
    1. Pod connects and sends registration:
       {"action": "register", "pod_name": "...", "pod_ip": "...",
        "namespace": "...", "service_name": "...", "request_metadata": true}

    2. Controller responds with metadata (if request_metadata=true):
       {"action": "metadata", "module": {...}, "service_name": "...", ...}

    3. Controller can push reloads at any time:
       {"action": "reload", "module": {...}, ...}

    4. Pod acknowledges reload completion:
       {"action": "reload_ack", "status": "ok"} or
       {"action": "reload_ack", "status": "error", "message": "..."}
       Controller waits for acks before returning from /pool registration.

    5. Pod can send pings to keep connection alive:
       {"action": "ping"}
       Controller responds: {"action": "pong"}
    """
    await websocket.accept()

    connected_pod: Optional[ConnectedPod] = None

    try:
        # Wait for registration message
        data = await websocket.receive_text()
        message = json.loads(data)

        if message.get("action") != "register":
            await websocket.send_text(
                json.dumps(
                    {"action": "error", "message": "First message must be registration"}
                )
            )
            await websocket.close()
            return

        # Create connected pod
        connected_pod = ConnectedPod(
            websocket=websocket,
            pod_name=message.get("pod_name", "unknown"),
            pod_ip=message.get("pod_ip", ""),
            namespace=message.get("namespace", "default"),
            service_name=message.get("service_name", ""),
        )

        logger.info(
            f"Pod registering: pod_name={connected_pod.pod_name}, "
            f"namespace={connected_pod.namespace}, service_name='{connected_pod.service_name}'"
        )

        # Register the connection
        await pod_manager.connect(connected_pod)

        # Send metadata if requested
        if message.get("request_metadata"):
            # If pod has no service_name, try to find matching pool by labels
            service_name_for_lookup = await try_match_pod_to_pool(connected_pod)

            metadata = (
                get_pool_metadata(connected_pod.namespace, service_name_for_lookup)
                if service_name_for_lookup
                else None
            )
            if metadata:
                await websocket.send_text(json.dumps(metadata))
                logger.info(
                    f"Sent metadata to {connected_pod.pod_name}: "
                    f"module={(metadata.get('module') or {}).get('module_name')}"
                )
            else:
                # Pool not registered yet - pod will wait for /pool call
                await websocket.send_text(
                    json.dumps(
                        {
                            "action": "waiting",
                            "message": f"Pool {service_name_for_lookup or '(unknown)'} not registered yet. "
                            "Metadata will be pushed when /pool is called.",
                        }
                    )
                )
                logger.info(
                    f"Pod {connected_pod.pod_name} waiting for pool registration "
                    f"({connected_pod.namespace}/{service_name_for_lookup or '(unknown)'})"
                )

        # Keep connection alive and handle messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                action = message.get("action")

                if action == "ping":
                    await websocket.send_text(json.dumps({"action": "pong"}))
                elif action == "request_metadata":
                    # Pod can request metadata again
                    # If pod has no service_name, try to find matching pool by labels
                    service_name_for_lookup = await try_match_pod_to_pool(connected_pod)

                    metadata = (
                        get_pool_metadata(
                            connected_pod.namespace, service_name_for_lookup
                        )
                        if service_name_for_lookup
                        else None
                    )
                    if metadata:
                        await websocket.send_text(json.dumps(metadata))
                    else:
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "action": "error",
                                    "message": f"Pool {service_name_for_lookup or '(unknown)'} not found",
                                }
                            )
                        )
                elif action == "reload_ack":
                    # Pod acknowledges reload completion
                    pod_manager.resolve_ack(
                        connected_pod.namespace,
                        connected_pod.service_name,
                        connected_pod.pod_name,
                        {
                            "status": message.get("status", "ok"),
                            "message": message.get("message", ""),
                        },
                    )
                    logger.debug(
                        f"Received reload ack from {connected_pod.pod_name}: {message.get('status')}"
                    )
                else:
                    logger.debug(
                        f"Unknown action from {connected_pod.pod_name}: {action}"
                    )

            except WebSocketDisconnect:
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if connected_pod:
            await pod_manager.disconnect(connected_pod)


async def broadcast_reload_via_websocket(
    namespace: str,
    service_name: str,
    module_info: dict,
    deployed_as_of: str = None,
    deployment_mode: str = None,
    distributed_config: dict = None,
    runtime_config: dict = None,
    username: str = None,
) -> dict:
    """Broadcast reload to all connected pods for a service.

    This is called from the /pool endpoint when a pool is registered/updated.

    Args:
        namespace: K8s namespace
        service_name: Service/pool name
        module_info: Module config dict (pointers, type, etc.)
        deployed_as_of: Deployment timestamp (currently unused with push model)
        deployment_mode: Deployment mode (knative, deployment, etc.)
        distributed_config: Distributed configuration (from pool_metadata)
        runtime_config: Runtime configuration (log_streaming_enabled, metrics_enabled, etc.)
        username: Username for KT_USERNAME env var (needed for nested service launches)

    Returns:
        Dict with status, message, sent count, errors
    """
    pointers = module_info.get("pointers") or {}

    message = {
        "action": "reload",
        "service_name": service_name,
        "namespace": namespace,
        "deployment_mode": deployment_mode,
        "username": username,
        "module": {
            "module_name": pointers.get("module_name"),
            "cls_or_fn_name": pointers.get("cls_or_fn_name"),
            "file_path": pointers.get("file_path"),
            "project_root": pointers.get("project_root"),
            "init_args": pointers.get("init_args"),
            "callable_type": module_info.get("type", "fn"),
            "distributed_config": distributed_config,
        },
        # Runtime config flows via WebSocket - can change between deploys
        "runtime_config": runtime_config or {},
    }

    result = await pod_manager.broadcast_to_service(namespace, service_name, message)
    logger.info(
        f"Broadcast reload to {namespace}/{service_name}: "
        f"{result.get('sent', 0)}/{result.get('total', 0)} pods"
    )
    return result
