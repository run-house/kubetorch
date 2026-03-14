"""
Metadata server for kubetorch data sync.

Tracks which pods have published data for each storage key, enabling
peer-to-peer data transfer and load balancing.

Supports multiple data types:
- filesystem: Traditional file/directory data (rsync-based transfer)
- gpu: GPU tensor data (NCCL broadcast-based transfer)
- memory: In-memory data blobs (future)
"""

import asyncio
import hashlib
import json
import logging
import os
import random
import shutil
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from itertools import count
from pathlib import Path
from threading import Lock, Thread
from typing import Dict, List, Optional
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect

from locks import PerKeyRWLock, SimpleLock
from models import (
    BroadcastJoinRequest,
    CompleteRequest,
    DataType,
    GPUBroadcastInfo,
    GPUGetRequest,
    GPUPublishRequest,
    GPUQuorumStatus,
    GPUSourcesRequest,
    SourceRequest,
    StoreRequest,
)
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
MAX_CONCURRENT_PER_SOURCE = int(os.getenv("MAX_CONCURRENT_PER_SOURCE", "30"))
SOURCE_TIMEOUT = 3600  # Remove sources older than 1 hour
CLEANUP_INTERVAL = 300  # Run cleanup every 5 minutes
DATA_ROOT = os.getenv("DATA_ROOT", "/data")
DEFAULT_GPU_QUORUM_TIMEOUT = float(os.getenv("DEFAULT_GPU_QUORUM_TIMEOUT", "5.0"))
NCCL_PORT_START = int(os.getenv("NCCL_PORT_START", "29500"))
# Per-key lock timeout (seconds). Increase if seeing timeout errors under heavy load.
KEY_LOCK_TIMEOUT = float(os.getenv("KEY_LOCK_TIMEOUT", "1.0"))

# In-memory storage with lock for thread safety
metadata: Dict[str, Dict] = defaultdict(
    lambda: {
        "sources": [],
        "store_pod_ip": None,
        "store_concurrent": 0,
        "data_type": DataType.FILESYSTEM,
    }
)

# Per-key read-write lock for metadata operations
# - Different keys can be accessed in parallel
# - Same key: multiple readers OR single writer
# - For iteration operations, use dirty reads with list(metadata.items()) instead
key_lock = PerKeyRWLock(timeout=KEY_LOCK_TIMEOUT)

# GPU broadcast quorum management
# Structure: {key: {broadcast_id: {"participants": [...], "started_at": float, "timeout": float, "world_size": int|None, "master_addr": str, "master_port": int, "status": str}}}
gpu_broadcast_quorums: Dict[str, Dict[str, Dict]] = defaultdict(dict)
gpu_quorum_lock = SimpleLock(timeout=5.0)  # Longer timeout for GPU operations

# Generalized broadcast quorum management (for filesystem and other data types)
# Structure: {broadcast_id: {"keys": [...], "putters": [...], "getters": [...], "started_at": float, "timeout": float, "world_size": int|None, "target_ips": [...], "status": str}}
broadcast_quorums: Dict[str, Dict] = {}
broadcast_quorum_lock = SimpleLock(timeout=5.0)

# GPU broadcast groups - unified quorum for NCCL broadcasts with mixed putters/getters
# Structure: {group_id: {
#   "participants": [{"pod_ip": str, "pod_name": str, "role": "putter"|"getter", "key": str, "rank": int|None, "websocket": WebSocket|None, ...}],
#   "started_at": float, "timeout": float, "world_size": int|None, "target_ips": [...],
#   "status": "waiting"|"ready"|"completed", "master_addr": str|None, "master_port": int|None,
# }}
# GPU broadcast groups - no lock needed since asyncio is single-threaded
# and dict operations are atomic. We only need to be careful about
# operations that span multiple await points.
gpu_broadcast_groups: Dict[str, Dict] = {}

# Default tree fanout for broadcast coordination
DEFAULT_TREE_FANOUT = 50
DEFAULT_FS_TREE_FANOUT = 50  # Higher fanout for filesystem (rsync)

# Filesystem broadcast groups - rolling participation, no quorum waiting
# Structure: {(group_id, key): {
#   "source_ip": str,  # Original putter's IP (from metadata)
#   "source_pod_name": str,
#   "source_path": str,  # Path on source for rsync
#   "rank_counter": itertools.count,  # Atomic counter for rank assignment
#   "participants": {pod_ip: {"pod_name": str, "rank": int, "joined_at": float}},  # Dict keyed by IP
#   "fanout": int,
#   "started_at": float,
# }}
fs_broadcast_groups: Dict[tuple, Dict] = {}
fs_broadcast_lock = Lock()  # Lock only for group creation/deletion, not participant ops


def get_pod_namespace():
    pod_namespace = os.getenv("POD_NAMESPACE", "default")
    return pod_namespace


def get_data_store_service_url(namespace: str = None):
    """Service URL for the data store - used for auto-discovery when data exists in filesystem"""
    pod_namespace = namespace if namespace else get_pod_namespace()
    return f"kubetorch-data-store.{pod_namespace}.svc.cluster.local"


# Helper functions
def key_to_filesystem_path(key: str) -> Optional[Path]:
    """
    Convert a storage key to a filesystem path.

    Keys map directly to filesystem paths: /data/{namespace}/{key}
    """
    pod_namespace = get_pod_namespace()

    if not key:
        return Path(DATA_ROOT) / pod_namespace

    # Strip leading/trailing slashes and use key directly as path
    key = key.strip("/")
    return Path(DATA_ROOT) / pod_namespace / key


def key_exists_in_filesystem(key: str) -> bool:
    """Check if a key exists in the filesystem."""
    fs_path = key_to_filesystem_path(key)
    return fs_path is not None and fs_path.exists()


def find_participant(
    participants: List[Dict], pod_ip: str, key: Optional[str] = None
) -> Optional[Dict]:
    """
    Utility to find a participant in a list by pod_ip, optionally also matching key.
    Returns the matching participant dict, or None if not found.

    Key matching checks both legacy "key" field and new "tensors" list format.
    """
    for p in participants:
        if p["pod_ip"] == pod_ip:
            if key is None:
                return p
            # Check legacy "key" field
            if p.get("key") == key:
                return p
            # Check new "tensors" list format
            for tensor in p.get("tensors", []):
                if tensor.get("key") == key:
                    return p
    return None


def _get_child_name_for_prefix(key: str, prefix: str) -> Optional[str]:
    """Extract the immediate child name for a key under a prefix."""
    # Skip empty keys - they shouldn't be registered
    if not key:
        return None

    if prefix:
        if key == prefix:
            return None
        if key.startswith(prefix + "/"):
            rel_path = key[len(prefix) + 1 :]
            return rel_path.split("/")[0] or None  # Return None if empty
        return None
    else:
        # No prefix - return top-level key
        result = key.split("/")[0] if "/" in key else key
        return result or None  # Return None if empty


def get_or_init_key_data(key: str) -> Dict:
    """Get key data from metadata, initializing from filesystem if needed."""
    if key not in metadata:
        if key_exists_in_filesystem(key):
            metadata[key] = {"sources": [], "store_pod_ip": None, "store_concurrent": 0}
            # Use service URL for auto-discovery - this is stable and works for rsync
            metadata[key]["store_pod_ip"] = get_data_store_service_url()
        else:
            return None
    return metadata[key]


def get_store_src_path(key: str) -> str:
    """Get the rsync source path for store-backed keys.

    Store data is stored at /data/{namespace}/{key}, so the rsync path
    (relative to /data which is the rsync module root) is {namespace}/{key}.
    """
    return f"{get_pod_namespace()}/{key}"


def cleanup_stale_sources():
    """Remove sources that haven't been accessed recently."""
    current_time = time.time()
    keys_to_remove = []

    # Dirty read with snapshot - safe iteration
    for key, key_data in list(metadata.items()):
        # Filter stale sources (in-place modification is atomic enough)
        key_data["sources"] = [
            s
            for s in key_data["sources"]
            if (current_time - s.get("published_at", 0)) < SOURCE_TIMEOUT
        ]
        if not key_data["sources"] and not key_data["store_pod_ip"]:
            keys_to_remove.append(key)

    # Delete empty keys (dict deletion is atomic)
    for key in keys_to_remove:
        metadata.pop(key, None)  # pop with default to avoid KeyError if already deleted

    if keys_to_remove:
        logger.info(f"Cleaned up {len(keys_to_remove)} stale keys")

    # Periodically clean up unused locks
    key_lock.cleanup_unused(set(metadata.keys()))


def start_cleanup_thread():
    """Start background thread for cleaning up stale sources."""

    def cleanup_loop():
        while True:
            try:
                cleanup_stale_sources()
            except Exception as e:
                logger.error(f"Error in cleanup thread: {e}")
            time.sleep(CLEANUP_INTERVAL)

    thread = Thread(target=cleanup_loop, daemon=True)
    thread.start()
    logger.info("Started cleanup thread")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    start_cleanup_thread()
    yield


# Create FastAPI app
app = FastAPI(
    title="Kubetorch Metadata Server",
    description="Tracks pod data availability for peer-to-peer transfer",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


# NOTE: These endpoints must be defined BEFORE the catch-all {key:path} routes
# to avoid FastAPI matching "list" or "stats" as key values


@app.get("/api/v1/keys/list")
async def list_keys(prefix: str = Query("")):
    """List all keys matching a prefix, combining locally-published keys and filesystem contents."""
    prefix = prefix.rstrip("/")
    prefix_depth = len(prefix.split("/")) if prefix else 0

    # Dirty read with snapshot - this is a read-only listing operation
    metadata_snapshot = list(metadata.items())

    # Pass 1: find which immediate children have deeper descendants
    child_names_with_children = set()
    for key, _ in metadata_snapshot:
        child_name = _get_child_name_for_prefix(key, prefix)
        if child_name is None:
            continue
        # If key has more parts than prefix + 1, the immediate child is a directory
        if len(key.split("/")) > prefix_depth + 1:
            child_names_with_children.add(child_name)

    # Pass 2: collect keys with metadata
    local_keys = {}
    for key, key_data in metadata_snapshot:
        child_name = _get_child_name_for_prefix(key, prefix)
        if child_name is None:
            continue

        has_sources = bool(key_data.get("sources"))
        has_children = child_name in child_names_with_children

        # Determine locale - pod name if locally published, "store" otherwise
        locale = "store"
        if has_sources and key_data.get("sources"):
            first_source = key_data["sources"][0]
            pod_name = first_source.get("pod_name")
            if pod_name:
                locale = pod_name

        if child_name not in local_keys:
            local_keys[child_name] = {
                "name": child_name,
                "is_directory": has_children,
                "locale": locale if has_sources else "store",
            }
        elif has_children:
            local_keys[child_name]["is_directory"] = True
            if has_sources and locale != "store":
                local_keys[child_name]["locale"] = locale

    # Get filesystem contents
    filesystem_items = []
    fs_path = key_to_filesystem_path(prefix)
    if fs_path and fs_path.exists() and fs_path.is_dir():
        try:
            for item in fs_path.iterdir():
                if item.name.startswith("."):
                    continue

                if item.name in local_keys:
                    # Item exists in both metadata and filesystem - it's in store
                    local_keys[item.name]["locale"] = "store"
                    if item.is_dir():
                        local_keys[item.name]["is_directory"] = True
                else:
                    filesystem_items.append(
                        {
                            "name": item.name,
                            "is_directory": item.is_dir(),
                            "locale": "store",
                        }
                    )
        except Exception as e:
            logger.warning(f"Failed to list filesystem path {fs_path}: {e}")

    # Combine and sort
    all_items = list(local_keys.values()) + filesystem_items
    all_items.sort(key=lambda x: (not x["is_directory"], x["name"].lower()))

    return {"prefix": prefix, "items": all_items}


@app.get("/api/v1/stats")
async def get_stats():
    """Get statistics about the metadata server."""
    # Dirty read - stats are approximate anyway
    values_snapshot = list(metadata.values())
    total_keys = len(values_snapshot)
    total_sources = sum(len(kd.get("sources", [])) for kd in values_snapshot)
    keys_with_store = sum(1 for kd in values_snapshot if kd.get("store_pod_ip"))

    return {
        "total_keys": total_keys,
        "total_sources": total_sources,
        "keys_with_store_pod": keys_with_store,
    }


# Key-specific routes with path parameters (must come AFTER specific routes)
# NOTE: GPU routes with /gpu/ suffix MUST be defined BEFORE general routes
# because FastAPI matches routes in order, and {key:path} is greedy.


@app.post("/api/v1/keys/{key:path}/gpu/source")
async def get_gpu_source(key: str, request: Optional[GPUSourcesRequest] = None):
    """
    Get source information for GPU tensor(s).
    Supports batch via request.keys or single via path key.
    Always returns {"sources": {key: info, ...}} format.
    """
    # Normalize to list - batch mode uses request.keys, single mode uses path key
    keys = request.keys if request and request.keys else [unquote(key).strip()]
    keys = [k for k in keys if k]  # Filter empty

    if not keys:
        raise HTTPException(status_code=400, detail="Key cannot be empty")

    results = {}
    for key in keys:
        # Per-key write lock (get_or_init_key_data may create entry)
        with key_lock.write(key):
            key_data = get_or_init_key_data(key)

            if key_data is None:
                results[key] = {"found": False}
                continue

            data_type = key_data.get("data_type", DataType.FILESYSTEM)
            if data_type != DataType.GPU:
                results[key] = {"found": False, "reason": f"Not GPU type ({data_type})"}
                continue

            if not key_data["sources"]:
                results[key] = {"found": False, "reason": "No sources"}
                continue

            source = key_data["sources"][0]
            results[key] = {
                "found": True,
                "ip": source["ip"],
                "pod_name": source.get("pod_name"),
                "namespace": source.get("namespace"),
                "gpu_server_port": source.get("gpu_server_port", 29400),
                "nccl_port": source.get("nccl_port", NCCL_PORT_START),
                "is_state_dict": source.get("is_state_dict", False),
                "tensor_keys": source.get("tensor_keys"),
                "tensor_shape": source.get("tensor_shape"),
                "tensor_dtype": source.get("tensor_dtype"),
            }

    return {"sources": results}


@app.get("/api/v1/keys/{key:path}/source")
async def get_source(key: str, external: bool = Query(False)):
    """
    Get an IP address or pod info to rsync data from for the given key.

    Implements load-based distribution:
    - Tracks concurrent requests per source IP
    - If store pod is below max concurrent, return it
    - Otherwise, return a random peer IP that's below max concurrent
    """
    key = unquote(key)

    with key_lock.write(key):
        key_data = get_or_init_key_data(key)

        if key_data is None:
            return {"found": False, "error": "Key not found"}

        available_sources = []
        store_pod_ip = key_data.get("store_pod_ip")

        # Check store pod first (if available and below max concurrent)
        if store_pod_ip:
            store_concurrent = key_data.get("store_concurrent", 0)
            if store_concurrent < MAX_CONCURRENT_PER_SOURCE:
                key_data["store_concurrent"] = store_concurrent + 1
                logger.debug(f"Returning store pod IP {store_pod_ip} for key '{key}'")

                src_path = get_store_src_path(key)

                if external:
                    return {
                        "pod_name": "kubetorch-data-sync",
                        "namespace": get_pod_namespace(),
                        "proxy_through_store": True,
                        "src_path": src_path,
                    }
                return {"ip": store_pod_ip, "src_path": src_path}

        # Check peer sources below max concurrent
        for source in key_data["sources"]:
            if source.get("concurrent_requests", 0) < MAX_CONCURRENT_PER_SOURCE:
                available_sources.append(source)

        if available_sources:
            selected = random.choice(available_sources)
            selected["concurrent_requests"] = selected.get("concurrent_requests", 0) + 1
            selected["request_count"] = selected.get("request_count", 0) + 1

            logger.debug(f"Returning peer IP {selected['ip']} for key '{key}'")

            if external:
                pod_name = selected.get("pod_name")
                namespace = selected.get("namespace", get_pod_namespace())
                src_path = selected.get("src_path")

                if pod_name:
                    result = {
                        "pod_name": pod_name,
                        "namespace": namespace,
                        "proxy_through_store": False,
                    }
                    if src_path:
                        result["src_path"] = src_path
                    return result
                else:
                    return {
                        "pod_name": "kubetorch-data-sync",
                        "namespace": namespace,
                        "proxy_through_store": True,
                        "peer_ip": selected["ip"],
                    }

            result = {"ip": selected["ip"]}
            if selected.get("src_path"):
                result["src_path"] = selected["src_path"]
            return result

        # All sources at max concurrent - return 503
        all_ips = []
        if store_pod_ip:
            all_ips.append(store_pod_ip)
        all_ips.extend([s["ip"] for s in key_data["sources"]])

        raise HTTPException(
            status_code=503,
            detail={"error": "All sources at max concurrent", "ips": all_ips},
        )


@app.post("/api/v1/keys/{key:path}/source/complete")
async def complete_request(key: str, request: CompleteRequest):
    """Notify that a request to a source IP has completed."""
    key = unquote(key)

    with key_lock.write(key):
        key_data = get_or_init_key_data(key)

        if key_data is None:
            return {"found": False, "error": "Key not found"}

        ip = request.ip

        # Decrement store pod concurrent count
        if key_data.get("store_pod_ip") == ip:
            current = key_data.get("store_concurrent", 0)
            key_data["store_concurrent"] = max(0, current - 1)

        # Decrement peer source concurrent count
        for source in key_data["sources"]:
            if source["ip"] == ip:
                current = source.get("concurrent_requests", 0)
                source["concurrent_requests"] = max(0, current - 1)
                break

    return {"success": True}


# NOTE: GPU publish route MUST be defined BEFORE the filesystem publish route
# because FastAPI matches routes in order, and {key:path} is greedy.
# /api/v1/keys/_/gpu/publish must match the GPU route (key="_"), not filesystem (key="_/gpu")
@app.post("/api/v1/keys/{key:path}/gpu/publish")
async def publish_gpu_key(key: str, request: GPUPublishRequest):
    """
    Publish GPU tensor(s). Supports batch via request.keys or single via path.
    """
    # Normalize to list - batch mode uses request.keys, single mode uses path key
    keys = request.keys if request.keys else [unquote(key).strip()]
    keys = [k for k in keys if k]  # Filter empty

    if not keys:
        raise HTTPException(status_code=400, detail="Key cannot be empty")

    namespace = request.namespace or get_pod_namespace()
    published_at = time.time()

    for key in keys:
        with key_lock.write(key):  # Per-key write lock
            if key not in metadata:
                metadata[key] = {
                    "sources": [],
                    "store_pod_ip": None,
                    "store_concurrent": 0,
                    "data_type": DataType.GPU,
                }
            else:
                metadata[key]["data_type"] = DataType.GPU

            existing = [s for s in metadata[key]["sources"] if s["ip"] == request.ip]
            source_data = {
                "ip": request.ip,
                "pod_name": request.pod_name,
                "namespace": namespace,
                "published_at": published_at,
                "tensor_shape": request.tensor_shape,
                "tensor_dtype": request.tensor_dtype,
                "nccl_port": request.nccl_port,
                "gpu_server_port": request.gpu_server_port,
                "is_state_dict": request.is_state_dict,
                "tensor_keys": request.tensor_keys,
                "request_count": 0,
                "concurrent_requests": 0,
            }

            if not existing:
                metadata[key]["sources"].append(source_data)
            else:
                existing[0].update(source_data)

    logger.info(
        f"Published {len(keys)} GPU key(s) from {request.pod_name} ({request.ip})"
    )

    return {"success": True, "count": len(keys), "data_type": DataType.GPU.value}


@app.post("/api/v1/keys/{key:path}/publish")
async def publish_key(key: str, request: SourceRequest):
    """Publish that a pod has data for the given key."""
    key = unquote(key).strip()

    # Reject empty keys
    if not key:
        raise HTTPException(status_code=400, detail="Key cannot be empty")

    # Determine data type
    data_type = DataType.FILESYSTEM
    if request.data_type:
        try:
            data_type = DataType(request.data_type)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid data_type: {request.data_type}"
            )

    with key_lock.write(key):
        if key not in metadata:
            metadata[key] = {
                "sources": [],
                "store_pod_ip": None,
                "store_concurrent": 0,
                "data_type": data_type,
            }
            if data_type == DataType.FILESYSTEM and key_exists_in_filesystem(key):
                metadata[key]["store_pod_ip"] = get_data_store_service_url()
        else:
            # Update data type if provided
            if request.data_type:
                metadata[key]["data_type"] = data_type

        existing = [s for s in metadata[key]["sources"] if s["ip"] == request.ip]

        if not existing:
            source_data = {
                "ip": request.ip,
                "published_at": time.time(),
                "request_count": 0,
                "concurrent_requests": 0,
            }
            if request.pod_name:
                source_data["pod_name"] = request.pod_name
            if request.namespace:
                source_data["namespace"] = request.namespace
            if request.src_path:
                source_data["src_path"] = request.src_path
            # GPU-specific fields
            if request.tensor_shape:
                source_data["tensor_shape"] = request.tensor_shape
            if request.tensor_dtype:
                source_data["tensor_dtype"] = request.tensor_dtype

            metadata[key]["sources"].append(source_data)
            logger.info(
                f"Published key '{key}' from IP '{request.ip}' (type: {data_type.value})"
            )
        else:
            # Update existing entry
            existing[0]["published_at"] = time.time()
            if request.pod_name:
                existing[0]["pod_name"] = request.pod_name
            if request.namespace:
                existing[0]["namespace"] = request.namespace
            if request.src_path:
                existing[0]["src_path"] = request.src_path
            if request.tensor_shape:
                existing[0]["tensor_shape"] = request.tensor_shape
            if request.tensor_dtype:
                existing[0]["tensor_dtype"] = request.tensor_dtype

    return {"success": True, "data_type": data_type.value}


@app.delete("/api/v1/keys/{key:path}/sources/{ip}")
async def remove_source(key: str, ip: str):
    """Remove a source IP from the list of available sources for a key."""
    key = unquote(key)

    with key_lock.write(key):
        key_data = get_or_init_key_data(key)

        if key_data is None:
            return {"found": False, "error": "Key not found"}

        original_count = len(key_data["sources"])
        key_data["sources"] = [s for s in key_data["sources"] if s["ip"] != ip]
        removed = original_count != len(key_data["sources"])

        if removed:
            logger.info(f"Removed unreachable source IP '{ip}' for key '{key}'")

    return {"success": True, "removed": removed}


@app.post("/api/v1/keys/{key:path}/store")
async def register_store_pod(key: str, request: StoreRequest):
    """Register that the store pod itself has data for a key."""
    key = unquote(key)

    with key_lock.write(key):
        if key not in metadata:
            metadata[key] = {
                "sources": [],
                "store_pod_ip": None,
                "store_concurrent": 0,
            }
            if key_exists_in_filesystem(key):
                metadata[key]["store_pod_ip"] = get_data_store_service_url()

        metadata[key]["store_pod_ip"] = request.ip
        logger.info(f"Registered store pod IP '{request.ip}' for key '{key}'")

    return {"success": True}


@app.get("/api/v1/keys/{key:path}")
async def get_key_info(key: str):
    """Get information about a key (for debugging/monitoring)."""
    key = unquote(key)

    with key_lock.write(key):  # Write lock (get_or_init_key_data may create entry)
        key_data = get_or_init_key_data(key)

        if key_data is None:
            return {"found": False, "error": "Key not found"}

        return {
            "key": key,
            "found": True,
            "store_pod_ip": key_data["store_pod_ip"],
            "sources": [
                {
                    "ip": s["ip"],
                    "published_at": s["published_at"],
                    "request_count": s.get("request_count", 0),
                }
                for s in key_data["sources"]
            ],
        }


@app.delete("/api/v1/keys/{key:path}")
async def delete_key(
    key: str, recursive: bool = Query(False), prefix_mode: bool = Query(False)
):
    """Delete a key from both the metadata server and filesystem.

    Args:
        key: The key to delete
        recursive: If True, delete directories recursively (directory semantics - adds /)
        prefix_mode: If True, delete all keys starting with this string (no / added)
    """
    key = unquote(key)

    deleted_from_metadata = False
    deleted_from_filesystem = False
    deleted_metadata_count = 0

    # Delete from metadata
    if prefix_mode:
        # Prefix mode - delete all keys that start with this string (no "/" added)
        keys_to_delete = [k for k in list(metadata.keys()) if k.startswith(key)]
        for k in keys_to_delete:
            metadata.pop(k, None)  # Atomic delete, ignore if already deleted
            deleted_metadata_count += 1
        if deleted_metadata_count > 0:
            deleted_from_metadata = True
            logger.info(
                f"Deleted {deleted_metadata_count} keys from metadata server with string prefix '{key}'"
            )
    elif recursive:
        # Recursive delete - use dirty read with snapshot to find matching keys
        prefix = key + "/" if not key.endswith("/") else key
        keys_to_delete = [
            k for k in list(metadata.keys()) if k == key or k.startswith(prefix)
        ]
        for k in keys_to_delete:
            metadata.pop(k, None)  # Atomic delete, ignore if already deleted
            deleted_metadata_count += 1
        if deleted_metadata_count > 0:
            deleted_from_metadata = True
            logger.info(
                f"Deleted {deleted_metadata_count} keys from metadata server with prefix '{key}'"
            )
    else:
        # Single key delete - use per-key lock
        with key_lock.write(key):
            if key in metadata:
                del metadata[key]
                deleted_from_metadata = True
                deleted_metadata_count = 1
                logger.info(f"Deleted key '{key}' from metadata server")

    # Delete from filesystem
    deleted_fs_count = 0
    if prefix_mode:
        # For prefix_mode, glob the namespace directory for matching entries
        pod_namespace = get_pod_namespace()
        namespace_dir = Path(DATA_ROOT) / pod_namespace
        if namespace_dir.exists():
            try:
                for entry in namespace_dir.iterdir():
                    if entry.name.startswith(key):
                        if entry.is_dir():
                            shutil.rmtree(entry)
                        else:
                            entry.unlink()
                        deleted_fs_count += 1
                        logger.info(f"Deleted filesystem entry '{entry.name}'")
                if deleted_fs_count > 0:
                    deleted_from_filesystem = True
            except Exception as e:
                logger.error(
                    f"Failed to delete filesystem entries with prefix '{key}': {e}"
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to delete from filesystem: {str(e)}",
                )
    else:
        fs_path = key_to_filesystem_path(key)
        if fs_path and fs_path.exists():
            try:
                if fs_path.is_dir():
                    if recursive:
                        shutil.rmtree(fs_path)
                        deleted_from_filesystem = True
                    else:
                        try:
                            fs_path.rmdir()
                            deleted_from_filesystem = True
                        except OSError:
                            raise HTTPException(
                                status_code=400,
                                detail="Directory not empty. Use recursive=true to delete non-empty directories.",
                            )
                else:
                    fs_path.unlink()
                    deleted_from_filesystem = True
                    logger.info(f"Deleted file '{key}' from filesystem")
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to delete '{key}' from filesystem: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to delete from filesystem: {str(e)}",
                )

    return {
        "success": True,
        "deleted_from_metadata": deleted_from_metadata,
        "deleted_from_filesystem": deleted_from_filesystem,
        "deleted_metadata_count": deleted_metadata_count,
        "deleted_fs_count": deleted_fs_count,
    }


@app.post("/api/v1/keys/{key:path}/mkdir")
async def mkdir_key(key: str):
    """Create a directory at the given key path."""
    key = unquote(key)

    fs_path = key_to_filesystem_path(key)
    if fs_path is None:
        raise HTTPException(status_code=400, detail="Invalid key path")

    try:
        fs_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory for key '{key}' at {fs_path}")
        return {"success": True, "path": str(fs_path)}
    except Exception as e:
        logger.error(f"Failed to create directory for key '{key}': {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create directory: {str(e)}"
        )


# ==================== GPU Broadcast Quorum Endpoints ====================


def _cleanup_expired_quorums():
    """Remove expired GPU broadcast quorums."""
    current_time = time.time()
    with gpu_quorum_lock:
        for key in list(gpu_broadcast_quorums.keys()):
            for broadcast_id in list(gpu_broadcast_quorums[key].keys()):
                quorum = gpu_broadcast_quorums[key][broadcast_id]
                # Keep quorums for a bit after timeout for late joiners to get "missed" status
                if current_time - quorum["started_at"] > quorum["timeout"] + 60:
                    del gpu_broadcast_quorums[key][broadcast_id]
                    logger.debug(
                        f"Cleaned up expired GPU quorum {broadcast_id} for key '{key}'"
                    )
            if not gpu_broadcast_quorums[key]:
                del gpu_broadcast_quorums[key]


@app.post("/api/v1/keys/{key:path}/gpu/get")
async def request_gpu_get(key: str, request: GPUGetRequest):
    """
    Request to get a GPU tensor. Implements the rolling quorum window approach.

    Flow:
    1. First requester starts a quorum window with the specified timeout
    2. Subsequent requesters join the same quorum if within the window
    3. When timeout expires, source pod is notified with final world_size
    4. All participants get their rank and connection info
    5. Late arrivers get "missed" status and should retry
    """
    key = unquote(key).strip()

    if not key:
        raise HTTPException(status_code=400, detail="Key cannot be empty")

    # Check if key exists and is GPU type
    with key_lock.write(key):  # Write lock (get_or_init_key_data may create entry)
        key_data = get_or_init_key_data(key)
        if key_data is None:
            raise HTTPException(status_code=404, detail=f"Key '{key}' not found")

        data_type = key_data.get("data_type", DataType.FILESYSTEM)
        if data_type != DataType.GPU:
            raise HTTPException(
                status_code=400,
                detail=f"Key '{key}' is not a GPU tensor (type: {data_type})",
            )

        if not key_data["sources"]:
            raise HTTPException(
                status_code=404, detail=f"No source available for GPU key '{key}'"
            )

        # Get source info (the pod that has the tensor)
        source = key_data["sources"][0]
        tensor_shape = source.get("tensor_shape")
        tensor_dtype = source.get("tensor_dtype")
        master_addr = source["ip"]
        master_port = source.get("nccl_port", NCCL_PORT_START)

    # Now handle quorum logic
    current_time = time.time()
    _cleanup_expired_quorums()

    with gpu_quorum_lock:
        # Find an active quorum for this key, or create a new one
        active_quorum = None
        active_broadcast_id = None

        for broadcast_id, quorum in gpu_broadcast_quorums.get(key, {}).items():
            # Check if quorum is still accepting participants
            elapsed = current_time - quorum["started_at"]
            if elapsed < quorum["timeout"] and quorum["status"] == "waiting":
                active_quorum = quorum
                active_broadcast_id = broadcast_id
                break
            elif quorum["status"] == "ready" and elapsed < quorum["timeout"] + 30:
                # Quorum is formed but broadcast may still be in progress
                # Check if this participant was already part of it
                p = find_participant(quorum["participants"], request.pod_ip)
                if p:
                    # Return their assigned info
                    return GPUBroadcastInfo(
                        broadcast_id=broadcast_id,
                        master_addr=quorum["master_addr"],
                        master_port=quorum["master_port"],
                        rank=p["rank"],
                        world_size=quorum["world_size"],
                        tensor_shape=tensor_shape,
                        tensor_dtype=tensor_dtype,
                        status="ready",
                    )

        if active_quorum is None:
            # Start a new quorum window
            broadcast_id = str(uuid.uuid4())[:8]
            active_quorum = {
                "participants": [],
                "started_at": current_time,
                "timeout": request.quorum_timeout,
                "world_size": None,
                "master_addr": master_addr,
                "master_port": master_port,
                "status": "waiting",
                "source_pod_ip": source["ip"],
                "source_pod_name": source.get("pod_name"),
            }
            gpu_broadcast_quorums[key][broadcast_id] = active_quorum
            active_broadcast_id = broadcast_id
            logger.info(
                f"Started GPU broadcast quorum {broadcast_id} for key '{key}', "
                f"timeout={request.quorum_timeout}s, master={master_addr}:{master_port}"
            )

        # Check if this participant is already in the quorum
        existing_participant = find_participant(
            active_quorum["participants"], request.pod_ip
        )

        if existing_participant is None:
            # Add participant (rank will be assigned when quorum closes)
            participant = {
                "pod_ip": request.pod_ip,
                "pod_name": request.pod_name,
                "namespace": request.namespace or get_pod_namespace(),
                "joined_at": current_time,
                "rank": None,  # Assigned when quorum closes
            }
            active_quorum["participants"].append(participant)
            logger.info(
                f"Added participant {request.pod_name} to GPU quorum {active_broadcast_id} "
                f"(total participants: {len(active_quorum['participants'])})"
            )

    # Return waiting status - client should poll or use websocket for ready notification
    return GPUBroadcastInfo(
        broadcast_id=active_broadcast_id,
        master_addr=master_addr,
        master_port=master_port,
        rank=-1,  # Not assigned yet
        world_size=-1,  # Not known yet
        tensor_shape=tensor_shape,
        tensor_dtype=tensor_dtype,
        status="waiting",
    )


@app.get("/api/v1/keys/{key:path}/gpu/quorum/{broadcast_id}")
async def get_gpu_quorum_status(key: str, broadcast_id: str, pod_ip: str = Query(...)):
    """
    Check the status of a GPU broadcast quorum.

    Clients poll this endpoint to know when the quorum is ready and get their rank.
    """
    key = unquote(key)

    with gpu_quorum_lock:
        if (
            key not in gpu_broadcast_quorums
            or broadcast_id not in gpu_broadcast_quorums[key]
        ):
            raise HTTPException(
                status_code=404, detail=f"Broadcast {broadcast_id} not found"
            )

        quorum = gpu_broadcast_quorums[key][broadcast_id]
        current_time = time.time()
        elapsed = current_time - quorum["started_at"]

        # Check if quorum window has expired and we need to finalize
        if quorum["status"] == "waiting" and elapsed >= quorum["timeout"]:
            # Finalize the quorum - assign ranks
            # Rank 0 is the source pod, ranks 1+ are participants
            world_size = len(quorum["participants"]) + 1  # +1 for source
            quorum["world_size"] = world_size

            for i, participant in enumerate(quorum["participants"]):
                participant["rank"] = i + 1  # Source is rank 0

            quorum["status"] = "ready"
            logger.info(
                f"GPU quorum {broadcast_id} for key '{key}' is ready, "
                f"world_size={world_size}, participants={[p['pod_name'] for p in quorum['participants']]}"
            )

        # Find this participant's rank
        p = find_participant(quorum["participants"], pod_ip)
        participant_rank = p.get("rank", -1) if p else -1

        if participant_rank == -1 and quorum["status"] == "ready":
            # This participant missed the quorum
            return GPUBroadcastInfo(
                broadcast_id=broadcast_id,
                master_addr=quorum["master_addr"],
                master_port=quorum["master_port"],
                rank=-1,
                world_size=quorum["world_size"] or -1,
                tensor_shape=[],
                tensor_dtype="",
                status="missed",
            )

        # Get tensor info from metadata
        tensor_shape = []
        tensor_dtype = ""
        with key_lock.read(key):
            key_data = metadata.get(key)
            if key_data and key_data["sources"]:
                tensor_shape = key_data["sources"][0].get("tensor_shape", [])
                tensor_dtype = key_data["sources"][0].get("tensor_dtype", "")

        return GPUBroadcastInfo(
            broadcast_id=broadcast_id,
            master_addr=quorum["master_addr"],
            master_port=quorum["master_port"],
            rank=participant_rank,
            world_size=quorum["world_size"] or -1,
            tensor_shape=tensor_shape,
            tensor_dtype=tensor_dtype,
            status=quorum["status"],
        )


@app.post("/api/v1/keys/{key:path}/gpu/quorum/{broadcast_id}/complete")
async def complete_gpu_broadcast(key: str, broadcast_id: str, pod_ip: str = Query(...)):
    """
    Mark a participant as having completed the broadcast.

    Called by each participant after successful NCCL broadcast.
    """
    key = unquote(key)

    with gpu_quorum_lock:
        if (
            key not in gpu_broadcast_quorums
            or broadcast_id not in gpu_broadcast_quorums[key]
        ):
            return {"success": False, "error": "Broadcast not found"}

        quorum = gpu_broadcast_quorums[key][broadcast_id]

        p = find_participant(quorum["participants"], pod_ip)
        if p:
            p["completed"] = True
            logger.debug(
                f"Participant {p['pod_name']} completed GPU broadcast {broadcast_id}"
            )

        # Check if all participants completed
        all_completed = all(p.get("completed", False) for p in quorum["participants"])
        if all_completed:
            quorum["status"] = "completed"
            logger.info(
                f"GPU broadcast {broadcast_id} for key '{key}' completed by all participants"
            )

    return {"success": True}


@app.get("/api/v1/gpu/quorums")
async def list_gpu_quorums():
    """List all active GPU broadcast quorums (for debugging/monitoring)."""
    _cleanup_expired_quorums()

    with gpu_quorum_lock:
        result = []
        for key, quorums in gpu_broadcast_quorums.items():
            for broadcast_id, quorum in quorums.items():
                result.append(
                    GPUQuorumStatus(
                        broadcast_id=broadcast_id,
                        key=key,
                        status=quorum["status"],
                        participants=len(quorum["participants"]),
                        world_size=quorum.get("world_size"),
                        master_addr=quorum["master_addr"],
                        master_port=quorum["master_port"],
                        started_at=quorum["started_at"],
                        timeout=quorum["timeout"],
                    )
                )
        return {"quorums": result}


# ==================== Unified GPU Broadcast Group Endpoints ====================


def _get_available_nccl_port_for_group_unlocked() -> int:
    """Get an available NCCL port for a new broadcast group.

    Called from async context - safe since asyncio is single-threaded.
    """
    used_ports = set()
    for group in gpu_broadcast_groups.values():
        if group.get("master_port"):
            used_ports.add(group["master_port"])

    # Find first available port in range
    for port in range(NCCL_PORT_START, NCCL_PORT_START + 100):
        if port not in used_ports:
            return port
    return NCCL_PORT_START  # Fallback


def _check_gpu_group_quorum_satisfied(group: Dict) -> bool:
    """Check if GPU broadcast group quorum conditions are satisfied (OR semantics).

    NOTE: world_size is the number of PARTICIPANTS (unique ip+role pairs), not tensors.
    With SPMD, multiple workers on the same pod are merged into a single participant.
    """
    total_participants = len(group.get("participants", []))

    # OR semantics - any condition being met closes the quorum
    # 1. Timeout elapsed
    if group.get("timeout"):
        if time.time() - group["started_at"] >= group["timeout"]:
            return True

    # 2. World size reached (by participant count)
    if group.get("world_size"):
        if total_participants >= group["world_size"]:
            return True

    # 3. All target IPs joined
    if group.get("target_ips"):
        joined_ips = {p["pod_ip"] for p in group.get("participants", [])}
        if all(ip in joined_ips for ip in group["target_ips"]):
            return True

    return False


async def _finalize_gpu_group(group: Dict) -> None:
    """
    Finalize a GPU broadcast group - assign ranks and build transfer manifest.

    CONSOLIDATION: Participants from the same pod (same pod_ip) are consolidated
    into a single NCCL rank. Each pod has ONE GPU data server that handles all
    NCCL operations for that node.

    Rank assignment:
    - Group participants by pod_ip
    - Each unique pod_ip gets ONE rank (first pod to join = rank 0 = master)
    - All participants on a pod share that pod's rank

    Transfer manifest:
    - For each pod, build consolidated sends/receives lists
    - Local transfers (same pod has putter AND getter for same key) are marked
      for direct copy optimization
    """
    participants = group["participants"]

    # Group participants by pod_ip
    pods_by_ip: Dict[str, List[Dict]] = {}
    pod_order: List[str] = []  # Track join order for rank assignment

    for p in participants:
        pod_ip = p["pod_ip"]
        if pod_ip not in pods_by_ip:
            pods_by_ip[pod_ip] = []
            pod_order.append(pod_ip)
        pods_by_ip[pod_ip].append(p)

    # Assign ONE rank per unique pod (in join order of first participant)
    pod_ranks: Dict[str, int] = {}
    for rank, pod_ip in enumerate(pod_order):
        pod_ranks[pod_ip] = rank
        # Assign the same rank to ALL participants on this pod
        for p in pods_by_ip[pod_ip]:
            p["rank"] = rank

    # World size = number of unique pods
    world_size = len(pod_order)
    group["world_size"] = world_size

    # First pod is the master
    if pod_order:
        group["master_addr"] = pod_order[0]
        group["master_port"] = (
            group.get("master_port") or _get_available_nccl_port_for_group_unlocked()
        )

    group["status"] = "ready"

    # Build a lookup of putters by key (for finding source of each tensor key)
    # With multi-tensor support, each participant has a "tensors" list
    putters_by_key: Dict[str, tuple] = {}  # key -> (participant, tensor_info)
    for p in participants:
        if p["role"] == "putter":
            for tensor in p.get("tensors", []):
                putters_by_key[tensor["key"]] = (p, tensor)

    # Build consolidated transfer manifest for each POD (not each participant)
    # This will be stored in group["pod_manifests"]
    pod_manifests: Dict[str, Dict] = {}

    for pod_ip in pod_order:
        pod_participants = pods_by_ip[pod_ip]
        pod_rank = pod_ranks[pod_ip]

        sends = []  # Tensors this pod needs to broadcast
        receives = []  # Tensors this pod needs to receive
        local_transfers = []  # Same-pod transfers (optimization)

        # Collect putters on this pod
        pod_putters = [p for p in pod_participants if p["role"] == "putter"]
        # Collect getters on this pod
        pod_getters = [p for p in pod_participants if p["role"] == "getter"]

        # Build getter keys on this pod for local transfer detection
        # Maps tensor key -> (getter_participant, tensor_info)
        pod_getter_keys: Dict[str, tuple] = {}
        for g in pod_getters:
            for tensor in g.get("tensors", []):
                pod_getter_keys[tensor["key"]] = (g, tensor)

        # Process all tensors from putters on this pod
        for putter in pod_putters:
            for tensor in putter.get("tensors", []):
                key = tensor["key"]

                # Check for local transfer (getter for same key on same pod)
                if key in pod_getter_keys:
                    getter_participant, getter_tensor = pod_getter_keys[key]
                    local_transfers.append(
                        {
                            "key": key,
                            "src_tensor_key": key,  # GPU server looks up in registry
                            "dest_ipc_handle": getter_tensor.get("dest_ipc_handle"),
                            "shape": tensor.get("shape", []),
                            "dtype": tensor.get("dtype", ""),
                        }
                    )
                    # Remove from pod_getter_keys so we don't also add to receives
                    del pod_getter_keys[key]
                else:
                    # Remote transfer - find getter ranks (unique pods that need this key)
                    getter_ranks = set()
                    for other in participants:
                        if other["role"] == "getter":
                            for other_tensor in other.get("tensors", []):
                                if other_tensor["key"] == key:
                                    getter_ranks.add(other["rank"])

                    if getter_ranks:
                        sends.append(
                            {
                                "key": key,
                                "src_tensor_key": key,  # GPU server looks up in registry
                                "to_ranks": sorted(getter_ranks),
                                "shape": tensor.get("shape", []),
                                "dtype": tensor.get("dtype", ""),
                            }
                        )

        # Remaining getters (not local transfers) need to receive from remote
        for getter in pod_getters:
            for tensor in getter.get("tensors", []):
                key = tensor["key"]
                if key not in [lt["key"] for lt in local_transfers]:
                    putter_info = putters_by_key.get(key)
                    if putter_info:
                        putter_participant, putter_tensor = putter_info
                        receives.append(
                            {
                                "key": key,
                                "from_rank": putter_participant["rank"],
                                "dest_ipc_handle": tensor.get("dest_ipc_handle"),
                                "shape": putter_tensor.get("shape", []),
                                "dtype": putter_tensor.get("dtype", ""),
                            }
                        )

        pod_manifests[pod_ip] = {
            "rank": pod_rank,
            "sends": sends,
            "receives": receives,
            "local_transfers": local_transfers,
            "participants": pod_participants,
        }

    group["pod_manifests"] = pod_manifests

    pod_summary = []
    for pod_ip in pod_order:
        m = pod_manifests[pod_ip]
        n_sends = len(m["sends"])
        n_receives = len(m["receives"])
        n_local = len(m["local_transfers"])
        pod_summary.append(
            f"rank{m['rank']}@{pod_ip}(sends:{n_sends}, recvs:{n_receives}, local:{n_local})"
        )

    logger.info(
        f"Finalized GPU broadcast group {group.get('group_id', 'unknown')}: "
        f"world_size={world_size} ({len(participants)} participants on {len(pod_order)} pods), "
        f"master={group['master_addr']}:{group['master_port']}, pods={pod_summary}"
    )


def _cleanup_expired_gpu_broadcast_groups():
    """Remove expired GPU broadcast groups.

    Groups are removed if:
    - Status is 'completed' and older than 60 seconds
    - Older than 1 hour (max_age)

    Safe to call from async context since asyncio is single-threaded.
    """
    current_time = time.time()
    max_age = 3600
    expired_ids = [
        gid
        for gid, g in gpu_broadcast_groups.items()
        if (g["status"] == "completed" and current_time - g["started_at"] > 60)
        or (current_time - g["started_at"] > max_age)
    ]
    for gid in expired_ids:
        del gpu_broadcast_groups[gid]
        logger.debug(f"Cleaned up expired GPU broadcast group {gid}")


@app.post("/api/v1/gpu/broadcast/{group_id}/complete")
async def complete_gpu_broadcast_group(group_id: str, pod_ip: str = Query(...)):
    """Mark this participant as having completed the GPU broadcast."""
    if group_id not in gpu_broadcast_groups:
        return {"success": False, "error": "Group not found"}

    group = gpu_broadcast_groups[group_id]

    p = find_participant(group["participants"], pod_ip)
    if p:
        p["completed"] = True
        logger.debug(
            f"Participant {p.get('pod_name', pod_ip)} completed GPU broadcast group {group_id}"
        )

    # Check if all completed
    all_completed = all(p.get("completed", False) for p in group["participants"])
    if all_completed:
        group["status"] = "completed"
        logger.info(f"GPU broadcast group {group_id} completed by all participants")

    return {"success": True}


@app.get("/api/v1/gpu/broadcast/groups")
async def list_gpu_broadcast_groups():
    """List all active GPU broadcast groups (for debugging/monitoring)."""
    _cleanup_expired_gpu_broadcast_groups()

    result = []
    for group_id, group in gpu_broadcast_groups.items():
        result.append(
            {
                "group_id": group_id,
                "status": group["status"],
                "participants": len(group["participants"]),
                "world_size": group.get("world_size"),
                "master_addr": group.get("master_addr"),
                "master_port": group.get("master_port"),
                "started_at": group["started_at"],
                "timeout": group.get("timeout"),
                "putters": [
                    p["pod_name"]
                    for p in group["participants"]
                    if p["role"] == "putter"
                ],
                "getters": [
                    p["pod_name"]
                    for p in group["participants"]
                    if p["role"] == "getter"
                ],
            }
        )
    return {"groups": result}


# ==================== WebSocket Broadcast Coordination ====================


def _compute_ancestors(
    participant_ips: List[str], my_rank: int, fanout: int
) -> List[str]:
    """
    Compute the list of ancestor IPs for a given rank in a tree topology.

    Returns list from root to direct parent (ancestors[0] = root, ancestors[-1] = parent).
    """
    if my_rank == 0:
        return []  # Root has no ancestors

    ancestors = []
    current_rank = my_rank

    while current_rank > 0:
        parent_rank = (current_rank - 1) // fanout
        ancestors.insert(0, participant_ips[parent_rank])
        current_rank = parent_rank

    return ancestors


async def _notify_participant(
    participant: Dict,
    group: Dict,
    group_id: str,
    pod_manifest: Optional[Dict] = None,
):
    """
    Send ready notification to a participant via their WebSocket.

    All participants from a pod receive the same manifest. The GPU Data Server
    on each pod is responsible for executing NCCL once (internally coordinating
    among its local participants).

    Args:
        participant: The participant dict
        group: The broadcast group
        group_id: Group identifier
        pod_manifest: Consolidated manifest for this pod (all participants get this)
    """
    ws = participant.get("websocket")
    if ws is None:
        logger.warning(f"No WebSocket for participant {participant.get('pod_name')}")
        return

    my_rank = participant["rank"]

    # Build ancestors list using unique pod IPs (for tree topology)
    pod_manifests = group.get("pod_manifests", {})
    unique_pod_ips = sorted(
        pod_manifests.keys(), key=lambda ip: pod_manifests[ip]["rank"]
    )
    ancestors = _compute_ancestors(unique_pod_ips, my_rank, DEFAULT_TREE_FANOUT)

    message = {
        "event": "ready",
        "group_id": group_id,
        "rank": my_rank,
        "world_size": group["world_size"],
        "master_addr": group["master_addr"],
        "master_port": group["master_port"],
        "ancestors": ancestors,
        "sends": pod_manifest.get("sends", []) if pod_manifest else [],
        "receives": pod_manifest.get("receives", []) if pod_manifest else [],
        "local_transfers": pod_manifest.get("local_transfers", [])
        if pod_manifest
        else [],
    }

    try:
        await ws.send_json(message)
        logger.debug(
            f"Sent ready notification to {participant.get('pod_name')} (rank {my_rank})"
        )
    except Exception as e:
        logger.error(f"Failed to notify participant {participant.get('pod_name')}: {e}")


async def _finalize_and_notify_gpu_group(group: Dict, group_id: str):
    """Finalize group and notify all participants via WebSocket.

    All participants from the same pod receive the same manifest. The GPU Data Server
    on each pod is responsible for coordinating NCCL execution internally.
    """
    await _finalize_gpu_group(group)

    pod_manifests = group.get("pod_manifests", {})

    tasks = []
    for p in group["participants"]:
        pod_ip = p["pod_ip"]
        pod_manifest = pod_manifests.get(pod_ip)
        tasks.append(_notify_participant(p, group, group_id, pod_manifest=pod_manifest))

    await asyncio.gather(*tasks, return_exceptions=True)


@app.websocket("/ws/broadcast/{group_id:path}")
async def websocket_broadcast_join(websocket: WebSocket, group_id: str):
    """
    WebSocket endpoint for joining a broadcast group.

    Protocol:
    1. Client connects and sends join message:
       {"action": "join", "key": str, "role": "putter"|"getter", "pod_ip": str,
        "pod_name": str, "timeout": float|None, "world_size": int|None,
        "tensor_shape": list|None, "tensor_dtype": str|None}

    2. Server responds with queued status:
       {"event": "queued", "group_id": str, "position": int}

    3. When quorum is satisfied, server sends ready notification:
       {"event": "ready", "group_id": str, "rank": int, "world_size": int,
        "master_addr": str, "master_port": int, "ancestors": [str],
        "sends": [...], "receives": [...]}

    4. Client performs NCCL transfer, then sends complete:
       {"action": "complete", "success": bool}

    5. Server acknowledges:
       {"event": "completed"}
    """
    await websocket.accept()
    logger.info(f"WebSocket connection accepted for broadcast group {group_id}")

    my_participant = None

    try:
        # Wait for join message
        logger.info(
            f"Waiting for join message from WebSocket client for group {group_id}"
        )
        try:
            raw_data = await websocket.receive_text()
            logger.info(f"Received raw text: {raw_data[:200]}...")

            join_msg = json.loads(raw_data)
            logger.info(
                f"Parsed join message: action={join_msg.get('action')}, key={join_msg.get('key')}"
            )
        except Exception as recv_err:
            logger.error(f"Error receiving join message: {recv_err}")
            raise

        if join_msg.get("action") != "join":
            await websocket.send_json(
                {"event": "error", "message": "Expected 'join' action"}
            )
            await websocket.close()
            return

        role = join_msg.get("role")
        pod_ip = join_msg.get("pod_ip")
        pod_name = join_msg.get("pod_name")
        timeout = join_msg.get("timeout")
        world_size = join_msg.get("world_size")

        # Parse tensors list from join message
        # Format: {"tensors": [{"key": ..., "shape": ..., "dtype": ..., "dest_ipc_handle": ...}, ...]}
        tensors_list = join_msg.get("tensors")
        if not tensors_list:
            await websocket.send_json(
                {
                    "event": "error",
                    "message": "Missing required field: tensors",
                }
            )
            await websocket.close()
            return

        tensors = []
        for t in tensors_list:
            tensors.append(
                {
                    "key": t.get("key"),
                    "shape": t.get("shape", []),
                    "dtype": t.get("dtype", "torch.float32"),
                    "dest_ipc_handle": t.get("dest_ipc_handle"),
                }
            )

        # Validate required fields
        if not role or not pod_ip:
            await websocket.send_json(
                {
                    "event": "error",
                    "message": "Missing required fields: role, pod_ip",
                }
            )
            await websocket.close()
            return

        # Validate tensors have keys
        if not tensors or not all(t.get("key") for t in tensors):
            await websocket.send_json(
                {
                    "event": "error",
                    "message": "Missing required tensor keys",
                }
            )
            await websocket.close()
            return

        effective_timeout = timeout if timeout is not None else 600.0

        tensor_keys = [t["key"] for t in tensors]
        logger.info(
            f"Processing join for group {group_id}, keys={tensor_keys}, role={role}"
        )

        _cleanup_expired_gpu_broadcast_groups()

        if group_id not in gpu_broadcast_groups:
            gpu_broadcast_groups[group_id] = {
                "group_id": group_id,
                "participants": [],
                "started_at": time.time(),
                "timeout": effective_timeout,
                "world_size": world_size,
                "target_ips": None,
                "status": "waiting",
                "master_addr": None,
                "master_port": _get_available_nccl_port_for_group_unlocked(),
            }
            logger.info(
                f"Created GPU broadcast group {group_id}: timeout={effective_timeout}s, world_size={world_size}"
            )

        group = gpu_broadcast_groups[group_id]

        # Update world_size if provided and not already set
        if world_size is not None and group.get("world_size") is None:
            group["world_size"] = world_size

        # Each kt.put() / kt.get() call is a separate participant
        # (even if multiple calls come from the same pod)
        my_participant = {
            "pod_ip": pod_ip,
            "pod_name": pod_name,
            "role": role,
            "tensors": tensors,  # List of {key, shape, dtype, dest_ipc_handle}
            "joined_at": time.time(),
            "rank": None,
            "sends": [],
            "receives": [],
            "websocket": websocket,
            "completed": False,
        }
        group["participants"].append(my_participant)
        logger.info(
            f"{role.capitalize()} {pod_name} joined GPU broadcast group {group_id} "
            f"with {len(tensors)} tensor(s): {tensor_keys} (total participants: {len(group['participants'])})"
        )

        position = len(group["participants"])
        is_ready = group["status"] == "ready"
        should_finalize = group[
            "status"
        ] == "waiting" and _check_gpu_group_quorum_satisfied(group)

        # Send queued confirmation
        await websocket.send_json(
            {
                "event": "queued",
                "group_id": group_id,
                "position": position,
            }
        )

        # If quorum is satisfied, finalize and notify everyone
        if should_finalize:
            group = gpu_broadcast_groups.get(group_id)
            if group and group["status"] == "waiting":
                await _finalize_and_notify_gpu_group(group, group_id)
        elif is_ready:
            # Group was already ready, send notification to this late joiner
            group = gpu_broadcast_groups.get(group_id)
            if group:
                pod_manifests = group.get("pod_manifests", {})
                pod_manifest = pod_manifests.get(pod_ip)
                await _notify_participant(
                    my_participant, group, group_id, pod_manifest=pod_manifest
                )

        # Keep connection open, handle heartbeats and completion
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)

                if msg.get("action") == "heartbeat":
                    await websocket.send_json({"event": "heartbeat_ack"})

                elif msg.get("action") == "complete":
                    success = msg.get("success", True)
                    if my_participant:
                        my_participant["completed"] = success
                        logger.debug(
                            f"Participant {pod_name} completed broadcast (success={success})"
                        )

                    group = gpu_broadcast_groups.get(group_id)
                    if not group:
                        await websocket.send_json({"event": "completed"})
                        break

                    # Mark the pod as complete (GPU Data Server executes NCCL once per pod)
                    my_pod_ip = my_participant["pod_ip"]
                    if "completed_pods" not in group:
                        group["completed_pods"] = set()
                    group["completed_pods"].add(my_pod_ip)
                    logger.debug(
                        f"Pod {my_pod_ip} completed, total: {len(group['completed_pods'])}"
                    )

                    # Check if all pods have completed
                    pod_manifests = group.get("pod_manifests", {})
                    all_pods = set(pod_manifests.keys())
                    completed_pods = group.get("completed_pods", set())
                    all_pods_done = all_pods <= completed_pods

                    if all_pods_done:
                        group["status"] = "completed"
                        logger.info(
                            f"GPU broadcast group {group_id} completed by all pods"
                        )

                    # Respond to this participant - their pod is done
                    await websocket.send_json({"event": "completed"})
                    break

            except asyncio.TimeoutError:
                # No message received, check if we should timeout the whole group
                group = gpu_broadcast_groups.get(group_id)
                if group and group["status"] == "waiting":
                    if _check_gpu_group_quorum_satisfied(group):
                        await _finalize_and_notify_gpu_group(group, group_id)

                # Send heartbeat to keep connection alive
                try:
                    await websocket.send_json({"event": "heartbeat"})
                except Exception:
                    break  # Connection lost

    except WebSocketDisconnect:
        logger.info(
            f"WebSocket disconnected for {my_participant.get('pod_name') if my_participant else 'unknown'}"
        )
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"event": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        # Clean up websocket reference
        if my_participant:
            my_participant["websocket"] = None


# ==================== Generalized Broadcast Quorum Endpoints ====================


def _generate_broadcast_id(keys: List[str]) -> str:
    """Generate a unique broadcast ID from keys."""
    key_str = ",".join(sorted(keys))
    return hashlib.md5(f"{key_str}-{time.time()}".encode()).hexdigest()[:12]


def _check_broadcast_quorum_satisfied(quorum: Dict) -> bool:
    """Check if broadcast quorum conditions are satisfied (OR semantics)."""
    total_participants = len(quorum.get("putters", [])) + len(quorum.get("getters", []))

    # OR semantics - any condition being met closes the quorum
    # 1. Timeout elapsed
    if quorum.get("timeout"):
        if time.time() - quorum["started_at"] >= quorum["timeout"]:
            return True

    # 2. World size reached
    if quorum.get("world_size"):
        if total_participants >= quorum["world_size"]:
            return True

    # 3. All target IPs joined
    if quorum.get("target_ips"):
        joined_ips = {p["pod_ip"] for p in quorum.get("putters", [])}
        joined_ips.update(p["pod_ip"] for p in quorum.get("getters", []))
        if all(ip in joined_ips for ip in quorum["target_ips"]):
            return True

    return False


def _cleanup_expired_broadcast_quorums():
    """Remove broadcast quorums that are completed or expired."""
    current_time = time.time()
    max_age = 3600

    with broadcast_quorum_lock:
        expired_ids = []
        for broadcast_id, quorum in broadcast_quorums.items():
            age = current_time - quorum["started_at"]
            if quorum["status"] == "completed" and age > 60:
                expired_ids.append(broadcast_id)
            elif age > max_age:
                expired_ids.append(broadcast_id)

        for broadcast_id in expired_ids:
            del broadcast_quorums[broadcast_id]
            logger.debug(f"Cleaned up expired broadcast quorum {broadcast_id}")


@app.post("/api/v1/broadcast/join")
async def join_broadcast(request: BroadcastJoinRequest):
    """
    Join a broadcast quorum for coordinated data transfer.

    Participants can be "putter" (data source) or "getter" (data destination).
    The quorum closes when ANY of the specified conditions is met (OR semantics):
    - timeout: Wait up to N seconds
    - world_size: Wait for N total participants
    - ips: Wait for specific IPs to join
    """
    _cleanup_expired_broadcast_quorums()

    group_id = request.group_id or _generate_broadcast_id(request.keys)

    with broadcast_quorum_lock:
        if group_id not in broadcast_quorums:
            broadcast_quorums[group_id] = {
                "keys": request.keys,
                "putters": [],
                "getters": [],
                "started_at": time.time(),
                "timeout": request.timeout or 30.0,
                "world_size": request.world_size,
                "target_ips": request.ips,
                "status": "waiting",
            }
            logger.info(
                f"Created new broadcast quorum {group_id} for keys: {request.keys}"
            )

        quorum = broadcast_quorums[group_id]

        # Add participant to appropriate list
        participant = {
            "pod_ip": request.pod_ip,
            "pod_name": request.pod_name,
            "namespace": request.namespace,
            "joined_at": time.time(),
            "completed": False,
        }

        if request.role == "putter":
            if not find_participant(quorum["putters"], request.pod_ip):
                quorum["putters"].append(participant)
                logger.debug(f"Putter {request.pod_name} joined broadcast {group_id}")
        else:
            if not find_participant(quorum["getters"], request.pod_ip):
                quorum["getters"].append(participant)
                logger.debug(f"Getter {request.pod_name} joined broadcast {group_id}")

        # Check if quorum is satisfied
        if _check_broadcast_quorum_satisfied(quorum):
            quorum["status"] = "ready"
            logger.info(
                f"Broadcast quorum {group_id} is ready: {len(quorum['putters'])} putters, {len(quorum['getters'])} getters"
            )

    return {
        "broadcast_id": group_id,
        "status": quorum["status"],
        "putters_count": len(quorum["putters"]),
        "getters_count": len(quorum["getters"]),
    }


@app.get("/api/v1/broadcast/{broadcast_id}/status")
async def get_broadcast_status(broadcast_id: str, pod_ip: str = Query(...)):
    """Get current status of a broadcast quorum."""
    _cleanup_expired_broadcast_quorums()

    with broadcast_quorum_lock:
        if broadcast_id not in broadcast_quorums:
            return {
                "status": "not_found",
                "error": f"Broadcast {broadcast_id} not found",
            }

        quorum = broadcast_quorums[broadcast_id]

        # Check if quorum should be finalized
        if quorum["status"] == "waiting" and _check_broadcast_quorum_satisfied(quorum):
            quorum["status"] = "ready"
            logger.info(
                f"Broadcast quorum {broadcast_id} finalized: {len(quorum['putters'])} putters, {len(quorum['getters'])} getters"
            )

        return {
            "broadcast_id": broadcast_id,
            "status": quorum["status"],
            "keys": quorum["keys"],
            "putters": [
                {"pod_ip": p["pod_ip"], "pod_name": p["pod_name"]}
                for p in quorum["putters"]
            ],
            "getters": [
                {"pod_ip": p["pod_ip"], "pod_name": p["pod_name"]}
                for p in quorum["getters"]
            ],
            "started_at": quorum["started_at"],
            "timeout": quorum["timeout"],
        }


@app.post("/api/v1/broadcast/{broadcast_id}/complete")
async def complete_broadcast(broadcast_id: str, pod_ip: str = Query(...)):
    """Mark this participant as having completed the broadcast transfer."""
    with broadcast_quorum_lock:
        if broadcast_id not in broadcast_quorums:
            return {"success": False, "error": "Broadcast not found"}

        quorum = broadcast_quorums[broadcast_id]

        # Mark participant as completed
        p = find_participant(quorum["putters"], pod_ip) or find_participant(
            quorum["getters"], pod_ip
        )
        if p:
            p["completed"] = True
            logger.debug(
                f"Participant {p.get('pod_name', pod_ip)} completed broadcast {broadcast_id}"
            )

        # Check if all participants completed
        all_completed = all(
            p.get("completed", False) for p in quorum["putters"] + quorum["getters"]
        )
        if all_completed:
            quorum["status"] = "completed"
            logger.info(f"Broadcast {broadcast_id} completed by all participants")

    return {"success": True}


# ==================== Filesystem Broadcast Endpoints ====================


def _get_fs_broadcast_parent(group: Dict, rank: int) -> Optional[Dict]:
    """
    Get the parent info for a filesystem broadcast participant.

    In filesystem broadcast, rank 0 is the original source (putter).
    Ranks 1+ are getters arranged in a tree.

    Returns dict with parent_ip, parent_path for rsync, or None if this is rank 0.
    """
    if rank == 0:
        return None  # Source has no parent

    fanout = group.get("fanout", DEFAULT_FS_TREE_FANOUT)
    parent_rank = (rank - 1) // fanout

    if parent_rank == 0:
        # Parent is the original source
        return {
            "parent_ip": group["source_ip"],
            "parent_pod_name": group.get("source_pod_name"),
            "parent_path": group.get("source_path"),
            "parent_rank": 0,
        }
    else:
        # Parent is another getter - participants is a dict keyed by pod_ip
        participants = group.get("participants", {})
        # Find participant with this rank
        for pod_ip, p in participants.items():
            if p.get("rank") == parent_rank:
                return {
                    "parent_ip": pod_ip,
                    "parent_pod_name": p.get("pod_name"),
                    "parent_path": group.get("source_path"),  # Same path
                    "parent_rank": parent_rank,
                }
        # Parent not found - they should exist but may have disconnected
        return None


def _cleanup_expired_fs_broadcast_groups():
    """Remove expired filesystem broadcast groups. Must be called under fs_broadcast_lock."""
    current_time = time.time()
    max_age = 3600

    expired_keys = [
        key
        for key, group in fs_broadcast_groups.items()
        if current_time - group["started_at"] > max_age
    ]
    for key in expired_keys:
        del fs_broadcast_groups[key]
        logger.debug(f"Cleaned up expired filesystem broadcast group {key}")


@app.websocket("/ws/fs-broadcast/{group_id:path}")
async def websocket_fs_broadcast_join(websocket: WebSocket, group_id: str):
    """
    WebSocket endpoint for joining a filesystem broadcast group.

    Key differences from GPU broadcast:
    - Only getters participate (source is discovered from metadata)
    - Rolling participation - returns immediately with parent info
    - Tree topology for efficient propagation
    - Track by (group_id, key) to support multi-key broadcasts

    Protocol:
    1. Client connects and sends join message:
       {"action": "join", "key": str, "pod_ip": str, "pod_name": str,
        "fanout": int|None}

    2. Server looks up source from metadata and responds immediately:
       {"event": "ready", "rank": int, "parent_ip": str|None,
        "parent_pod_name": str|None, "parent_path": str|None,
        "source_ip": str, "source_path": str}

    3. Client closes connection (no completion notification needed)
    """
    await websocket.accept()
    logger.debug(
        f"WebSocket connection accepted for filesystem broadcast group {group_id}"
    )

    try:
        # Wait for join message
        raw_data = await websocket.receive_text()
        join_msg = json.loads(raw_data)

        if join_msg.get("action") != "join":
            await websocket.send_json(
                {"event": "error", "message": "Expected 'join' action"}
            )
            return

        key = join_msg.get("key")
        pod_ip = join_msg.get("pod_ip")
        pod_name = join_msg.get("pod_name")
        fanout = join_msg.get("fanout") or DEFAULT_FS_TREE_FANOUT

        if not all([key, pod_ip]):
            await websocket.send_json(
                {"event": "error", "message": "Missing required fields: key, pod_ip"}
            )
            return

        broadcast_key = (group_id, key)

        # Cleanup expired groups (under lock)
        with fs_broadcast_lock:
            _cleanup_expired_fs_broadcast_groups()

        # Look up source from metadata (no lock needed for dict read)
        source_info = None
        key_data = metadata.get(key)
        if key_data and key_data.get("sources"):
            source = key_data["sources"][0]
            source_info = {
                "ip": source["ip"],
                "pod_name": source.get("pod_name"),
                "src_path": source.get("src_path"),
            }
        elif key_data and key_data.get("store_pod_ip"):
            source_info = {
                "ip": key_data["store_pod_ip"],
                "pod_name": "kubetorch-data-store",
                "src_path": get_store_src_path(key),
            }

        # If not found, try with write lock (may need to init from filesystem)
        if source_info is None:
            try:
                with key_lock.write(key):
                    key_data = get_or_init_key_data(key)
                    if key_data and key_data.get("sources"):
                        source = key_data["sources"][0]
                        source_info = {
                            "ip": source["ip"],
                            "pod_name": source.get("pod_name"),
                            "src_path": source.get("src_path"),
                        }
                    elif key_data and key_data.get("store_pod_ip"):
                        source_info = {
                            "ip": key_data["store_pod_ip"],
                            "pod_name": "kubetorch-data-store",
                            "src_path": get_store_src_path(key),
                        }
            except TimeoutError:
                await websocket.send_json(
                    {"event": "error", "message": "Server overloaded, please retry"}
                )
                return

        if not source_info:
            await websocket.send_json(
                {"event": "error", "message": f"Key '{key}' not found in metadata"}
            )
            return

        # Get or create broadcast group (lock only needed for group creation)
        group = fs_broadcast_groups.get(broadcast_key)
        if group is None:
            with fs_broadcast_lock:
                # Double-check after acquiring lock
                if broadcast_key not in fs_broadcast_groups:
                    fs_broadcast_groups[broadcast_key] = {
                        "source_ip": source_info["ip"],
                        "source_pod_name": source_info.get("pod_name"),
                        "source_path": source_info.get("src_path"),
                        "rank_counter": count(1),  # Atomic counter, starts at 1
                        "participants": {},  # Dict keyed by pod_ip
                        "fanout": fanout,
                        "started_at": time.time(),
                    }
                    logger.info(
                        f"Created filesystem broadcast group {broadcast_key}: "
                        f"source={source_info['ip']}, fanout={fanout}"
                    )
                group = fs_broadcast_groups[broadcast_key]

        # Assign rank and join atomically (no lock needed)
        # First check if already joined
        existing = group["participants"].get(pod_ip)
        if existing:
            rank = existing["rank"]
        else:
            # Get next rank atomically
            rank = next(group["rank_counter"])
            participant = {
                "pod_name": pod_name,
                "rank": rank,
                "joined_at": time.time(),
            }
            # Atomically try to add - if already exists, setdefault returns existing
            actual = group["participants"].setdefault(pod_ip, participant)
            if actual is not participant:
                # Another thread already added this pod, use their rank
                rank = actual["rank"]
            else:
                logger.info(
                    f"Getter {pod_name} joined filesystem broadcast {broadcast_key} "
                    f"as rank {rank} (total getters: {len(group['participants'])})"
                )

        # Compute parent info (no lock needed - just reads)
        parent_info = _get_fs_broadcast_parent(group, rank)

        # Read values for response (no lock needed)
        total_participants = len(group["participants"])
        group_source_ip = group["source_ip"]
        group_source_pod_name = group.get("source_pod_name")
        group_source_path = group.get("source_path")

        # Send ready response immediately (rolling participation)
        ready_msg = {
            "event": "ready",
            "group_id": group_id,
            "key": key,
            "rank": rank,
            "total_participants": total_participants,
            "source_ip": group_source_ip,
            "source_pod_name": group_source_pod_name,
            "source_path": group_source_path,
        }

        if parent_info:
            ready_msg["parent_ip"] = parent_info["parent_ip"]
            ready_msg["parent_pod_name"] = parent_info.get("parent_pod_name")
            ready_msg["parent_path"] = parent_info.get("parent_path")
            ready_msg["parent_rank"] = parent_info["parent_rank"]
        else:
            ready_msg["parent_ip"] = None

        await websocket.send_json(ready_msg)
        # Connection closes after sending ready - no completion needed

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Filesystem broadcast WebSocket error: {e}")
        try:
            await websocket.send_json({"event": "error", "message": str(e)})
        except Exception:
            pass


@app.get("/api/v1/fs/broadcast/groups")
async def list_fs_broadcast_groups():
    """List all active filesystem broadcast groups (for debugging/monitoring)."""
    with fs_broadcast_lock:
        _cleanup_expired_fs_broadcast_groups()

        result = []
        for (group_id, key), group in fs_broadcast_groups.items():
            result.append(
                {
                    "group_id": group_id,
                    "key": key,
                    "source_ip": group["source_ip"],
                    "source_pod_name": group.get("source_pod_name"),
                    "participants": len(group["participants"]),
                    "fanout": group.get("fanout", DEFAULT_FS_TREE_FANOUT),
                    "started_at": group["started_at"],
                }
            )
    return {"groups": result}


class ReportUnreachableRequest(BaseModel):
    group_id: str
    key: str
    reporter_ip: str
    unreachable_ip: str


@app.post("/api/v1/fs/broadcast/report-unreachable")
async def report_unreachable_parent(request: ReportUnreachableRequest):
    """
    Report an unreachable parent in a filesystem broadcast group.

    Removes the unreachable pod from participants and assigns a new parent
    to the reporter. The new parent is randomly selected from pods with
    lower rank than the reporter (to avoid loops).
    """
    import random

    broadcast_key = (request.group_id, request.key)

    with fs_broadcast_lock:
        if broadcast_key not in fs_broadcast_groups:
            return {
                "status": "error",
                "error": f"Broadcast group not found: {request.group_id}/{request.key}",
            }

        group = fs_broadcast_groups[broadcast_key]
        participants = group["participants"]

        # Find the reporter's rank
        reporter_info = participants.get(request.reporter_ip)
        if not reporter_info:
            return {
                "status": "error",
                "error": f"Reporter {request.reporter_ip} not found in group",
            }
        reporter_rank = reporter_info["rank"]

        # Remove the unreachable pod from participants
        if request.unreachable_ip in participants:
            del participants[request.unreachable_ip]
            logger.info(
                f"Removed unreachable pod {request.unreachable_ip} from broadcast group {broadcast_key}"
            )

        # Find a new parent - randomly select from pods with lower rank
        # Pods with lower rank are guaranteed to be ancestors (not descendants)
        candidates = [
            (ip, info)
            for ip, info in participants.items()
            if info["rank"] < reporter_rank
        ]

        if not candidates:
            # No candidates - fall back to source (rank 0)
            return {
                "status": "ok",
                "parent_rank": 0,
                "parent_ip": group["source_ip"],
                "parent_pod_name": group.get("source_pod_name"),
                "source_path": group.get("source_path"),
            }

        # Randomly select a new parent
        new_parent_ip, new_parent_info = random.choice(candidates)

        logger.info(
            f"Assigned new parent for {request.reporter_ip} in {broadcast_key}: "
            f"{new_parent_ip} (rank {new_parent_info['rank']})"
        )

        return {
            "status": "ok",
            "parent_rank": new_parent_info["rank"],
            "parent_ip": new_parent_ip,
            "parent_pod_name": new_parent_info.get("pod_name"),
            "source_path": group.get("source_path"),
        }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("METADATA_SERVER_PORT", "8081"))
    host = os.getenv("METADATA_SERVER_HOST", "0.0.0.0")

    logger.info(f"Starting metadata server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
