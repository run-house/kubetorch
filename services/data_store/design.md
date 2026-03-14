# Data Store Server Design

This directory contains the server-side implementation of the kubetorch data store - the metadata and coordination infrastructure that enables both external file sync and in-cluster P2P data sharing.

## Overview

The data store solves two critical gaps in Kubernetes for machine learning:

1. **Fast deployment**: Sync code and data to your cluster instantly via rsync - no container rebuilds
2. **In-cluster data sharing**: Peer-to-peer data transfer between pods with automatic caching and discovery - the "object store" functionality that Ray users miss

The server runs as a centralized service (`kubetorch-data-sync`) in the cluster, providing:
- **Metadata Server**: Tracks which pods have data for each key, enabling peer-to-peer discovery and load balancing
- **Rsync Daemon**: Stores and serves filesystem data (central store for `locale="store"`)
- **WebSocket Proxy**: Enables rsync access from outside the cluster (external sync)
- **GPU Broadcast Coordination**: Manages NCCL broadcast quorums for GPU tensor transfers
- **Filesystem Broadcast Coordination**: Manages tree-based P2P propagation for scalable file distribution

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        kubetorch-data-sync Pod                               │
│                                                                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────────────┐ │
│  │   Rsync Daemon   │  │  WebSocket Proxy │  │     Metadata Server        │ │
│  │     (port 873)   │  │    (port 8080)   │  │       (port 8081)          │ │
│  │                  │  │                  │  │                            │ │
│  │  /data/{ns}/...  │  │ WS ←→ rsync TCP  │  │ FastAPI app                │ │
│  │  Stores files    │  │ External access  │  │ In-memory metadata store   │ │
│  │                  │  │                  │  │ GPU broadcast coordination │ │
│  └──────────────────┘  └──────────────────┘  └────────────────────────────┘ │
│          ▲                     ▲                         ▲                   │
└──────────┼─────────────────────┼─────────────────────────┼───────────────────┘
           │                     │                         │
   ┌───────┴───────┐     ┌───────┴───────┐        ┌───────┴───────────┐
   │ In-cluster    │     │ External      │        │ All clients       │
   │ rsync clients │     │ rsync clients │        │ (metadata ops)    │
   └───────────────┘     └───────────────┘        └───────────────────┘
```

## Module Responsibilities

### `server.py`
The main metadata server - a FastAPI application tracking data availability.

**Core Data Structures:**

```python
# Key-level metadata
metadata: Dict[str, Dict] = {
    "my-key": {
        "sources": [          # Peer pods with this data
            {"ip": "10.0.0.5", "pod_name": "worker-0", "published_at": ...},
        ],
        "store_pod_ip": "10.0.0.3",  # Store pod (if data synced there)
        "store_concurrent": 2,        # Active requests to store
        "data_type": "filesystem",    # filesystem | gpu | memory
    }
}

# GPU broadcast quorums (legacy single-key approach)
gpu_broadcast_quorums: Dict[str, Dict[str, Dict]]

# GPU broadcast groups (unified multi-key/multi-participant)
gpu_broadcast_groups: Dict[str, Dict]

# Generalized broadcast quorums (filesystem)
broadcast_quorums: Dict[str, Dict]
```

**Key Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/api/v1/keys/{key}/source` | GET | Get peer IP + src_path for download (load balanced) |
| `/api/v1/keys/{key}/source/complete` | POST | Notify transfer complete (decrement counter) |
| `/api/v1/keys/{key}/publish` | POST | Register peer as data source |
| `/api/v1/keys/{key}/store` | POST | Register store pod has data |
| `/api/v1/keys/{key}` | GET | Get key info (debug) |
| `/api/v1/keys/{key}` | DELETE | Delete key (metadata + filesystem). Supports `?recursive=true` for directory semantics or `?prefix_mode=true` for string-prefix matching |
| `/api/v1/keys/{key}/mkdir` | POST | Create directory |
| `/api/v1/keys/list` | GET | List keys with prefix |
| `/api/v1/stats` | GET | Server statistics |
| `/api/v1/keys/{key}/gpu/publish` | POST | Register GPU tensor source |
| `/api/v1/keys/{key}/gpu/source` | GET | Get GPU source info |
| `/api/v1/gpu/broadcast/join` | POST | Join GPU broadcast group (blocking) |
| `/api/v1/gpu/broadcast/groups` | GET | List active GPU groups |
| `/ws/broadcast/{group_id}` | WS | WebSocket for GPU broadcast coordination |
| `/ws/fs-broadcast/{group_id}` | WS | WebSocket for filesystem broadcast (rolling participation) |
| `/api/v1/fs/broadcast/groups` | GET | List active filesystem broadcast groups |
| `/api/v1/broadcast/join` | POST | Join filesystem broadcast quorum (legacy) |
| `/api/v1/services/{name}/cleanup` | DELETE | Clean up resource-scoped keys |

### `websocket_tunnel_server.py`
WebSocket proxy enabling rsync from outside the cluster.

**How it works:**
1. External client connects via WebSocket (port 8080)
2. Server opens TCP connection to local rsync daemon (port 873)
3. Bidirectional data shuttle between WebSocket and TCP

```
External Client                      Data Store Pod
    │                                     │
    │  WebSocket connect ─────────────────►│
    │  (port 8080)                         │
    │                                      ▼
    │                               ┌─────────────┐
    │  Binary rsync data ◄────────►│ TCP:873     │
    │  over WebSocket               │ rsync daemon│
    │                               └─────────────┘
```

### `start.sh`
Entrypoint script that starts all three services:
1. Rsync daemon (`rsync --daemon`)
2. WebSocket proxy (`server_proxy.py`)
3. Metadata server (`uvicorn server:app`)

Handles graceful shutdown on SIGTERM/SIGINT.

### `Dockerfile`
Builds the container image with:
- Python 3.11 slim base
- rsync, net-tools, curl
- FastAPI, uvicorn, websockets
- All server scripts

### `test_server.py`
Integration tests for the metadata server.

## Data Flow Diagrams

### Filesystem Put (client perspective)

```
Client Pod                    Metadata Server               Store Pod
──────────                    ───────────────               ─────────
rsync data to store ──────────────────────────────────────► Receives files
     │                                                       │
     ▼                                                       │
POST /keys/{key}/store ──────► Store in metadata            │
{ip: store_pod_ip}            (register availability) ◄─────┘
```

### Filesystem Get (peer-to-peer)

```
Client Pod                    Metadata Server               Peer Pod
──────────                    ───────────────               ────────
GET /keys/{key}/source ──────► Load balance decision
     │                         - Check store concurrent
     │                         - Check peer concurrent
     │                         - Return least-loaded
     │◄────────────────────── {ip: "10.0.0.5",
     │                         src_path: "default/svc/model"}
     │
     ▼
rsync from peer ─────────────────────────────────────────► Serve data
     │                                                       │
     ▼                                                       │
POST /keys/{key}/source/complete ────────────────────────────┘
{ip: "10.0.0.5"}              Decrement concurrent counter
```

### Source Selection Algorithm

```python
def get_source(key):
    key_data = metadata[key]

    # 1. Check store pod first (if below MAX_CONCURRENT)
    if store_pod_ip and store_concurrent < MAX_CONCURRENT_PER_SOURCE:
        store_concurrent += 1
        # Return IP and src_path for rsync
        return {"ip": store_pod_ip, "src_path": get_store_src_path(key)}

    # 2. Check peer sources below max concurrent
    available_peers = [
        p for p in sources
        if p.concurrent_requests < MAX_CONCURRENT_PER_SOURCE
    ]

    if available_peers:
        selected = random.choice(available_peers)
        selected.concurrent_requests += 1
        # Peer sources have src_path from when they published
        return {"ip": selected.ip, "src_path": selected.src_path}

    # 3. All sources at max - return 503
    raise HTTPException(503, "All sources at max concurrent")
```

### GPU Broadcast Flow (WebSocket-based)

Supports both single tensor and multi-tensor (state_dict) broadcasts:

```
Putter Pod                    Metadata Server              Getter Pod
──────────                    ───────────────              ──────────

ws://server/ws/broadcast/{group_id}
     │                              │                           │
     │  {"action": "join",          │                           │
     │   "role": "putter",          │                           │
     │   "tensors": [               │   (Multi-tensor format)   │
     │     {"key": "layer1.weight", │                           │
     │      "shape": [512,256],     │                           │
     │      "dtype": "float32"},    │                           │
     │     {"key": "layer1.bias",   │                           │
     │      "shape": [512]}],       │                           │
     │   "world_size": 2} ─────────►│                           │
     │                              │                           │
     │◄────────────────── {"event": "queued", "position": 1}    │
     │                              │                           │
     │                              │◄───────────── {"action": "join",
     │                              │                "role": "getter",
     │                              │                "tensors": [...]}
     │                              │
     │                    Quorum satisfied                      │
     │                    _finalize_gpu_group():                │
     │                      - Group by pod_ip                   │
     │                      - Assign ranks (1 per pod)         │
     │                      - First pod = rank 0 = master      │
     │                      - Build transfer manifests         │
     │                        for ALL tensors                  │
     │                              │
     │◄────────────────── {"event": "ready",  ─────────────────►│
     │                     "rank": 0,                           │
     │                     "world_size": 2,                     │
     │                     "master_addr": putter_ip,            │
     │                     "master_port": 29500,                │
     │                     "sends": [                           │
     │                       {key: "layer1.weight", ...},       │
     │                       {key: "layer1.bias", ...}],        │
     │                     "receives": []}                      │
     │                              │                           │
     │                              │        (rank 1, receives=[...])
     │                              │                           │
     ▼                              │                           ▼
Execute NCCL via                    │               Execute NCCL via
local GPU server                    │               local GPU server
(all tensors in same session)       │               (all tensors in same session)
     │                              │                           │
     │  {"action": "complete"} ────►│◄──────── {"action": "complete"}
     │                              │                           │
     │◄──────── {"event": "completed"} ────────────────────────►│
```

**Join message formats (backward compatible):**

Single tensor (legacy):
```json
{"action": "join", "role": "putter", "key": "weights", "tensor_shape": [...], ...}
```

Multi-tensor (new):
```json
{"action": "join", "role": "putter", "tensors": [
  {"key": "layer1.weight", "shape": [...], "dtype": "float32", "dest_ipc_handle": "..."},
  {"key": "layer1.bias", "shape": [...], "dtype": "float32", "dest_ipc_handle": "..."}
], ...}
```

### Key-to-Path Mapping

```python
def key_to_filesystem_path(key: str) -> Path:
    """
    Convert storage key to filesystem path.

    Keys map directly to filesystem paths: /data/{namespace}/{key}

    Examples:
      "my-svc/model"     → /data/{namespace}/my-svc/model
      "checkpoint.pt"    → /data/{namespace}/checkpoint.pt
      "foo/bar/baz"      → /data/{namespace}/foo/bar/baz
    """
    if not key:
        return Path(DATA_ROOT) / namespace

    key = key.strip("/")
    return Path(DATA_ROOT) / namespace / key


def get_store_src_path(key: str) -> str:
    """
    Get the rsync source path for store-backed keys.

    Store data is stored at /data/{namespace}/{key}, so the rsync path
    (relative to /data which is the rsync module root) is {namespace}/{key}.

    This is used by both the /source endpoint and the WebSocket fs-broadcast
    endpoint to ensure consistent path construction.
    """
    return f"{POD_NAMESPACE}/{key}"
```

### GPU Broadcast Group Finalization

```python
async def _finalize_gpu_group(group: Dict):
    """
    Assign ranks and build transfer manifest.

    CONSOLIDATION: Multiple participants from same pod share one NCCL rank.
    Each pod has ONE GPU data server handling all NCCL operations.

    MULTI-TENSOR: Each participant can have multiple tensors. Manifest
    building iterates over all tensors from all participants.
    """
    participants = group["participants"]

    # Group by pod_ip
    pods_by_ip: Dict[str, List[Dict]] = {}
    pod_order: List[str] = []  # Track join order

    for p in participants:
        if p["pod_ip"] not in pods_by_ip:
            pods_by_ip[p["pod_ip"]] = []
            pod_order.append(p["pod_ip"])
        pods_by_ip[p["pod_ip"]].append(p)

    # Assign ONE rank per unique pod
    for rank, pod_ip in enumerate(pod_order):
        for p in pods_by_ip[pod_ip]:
            p["rank"] = rank

    # First pod = master
    group["master_addr"] = pod_order[0]
    group["world_size"] = len(pod_order)

    # Build lookup of all tensor keys -> (participant, tensor_info)
    putters_by_key = {}
    for p in participants:
        if p["role"] == "putter":
            for tensor in p.get("tensors", []):
                putters_by_key[tensor["key"]] = (p, tensor)

    # Build consolidated manifests per pod
    # Iterate over each participant's tensors list
    for pod_ip in pod_order:
        sends, receives, local_transfers = [], [], []

        for putter in pod_putters:
            for tensor in putter.get("tensors", []):
                # Check local transfer vs remote send
                # Build sends/receives for each tensor

        for getter in pod_getters:
            for tensor in getter.get("tensors", []):
                # Build receives for tensors not locally transferred
```

**Participant structure:**
```python
{
    "pod_ip": "10.0.0.1",
    "pod_name": "trainer-0",
    "role": "putter",
    "tensors": [  # List of tensors (new multi-tensor format)
        {"key": "layer1.weight", "shape": [512, 256], "dtype": "float32"},
        {"key": "layer1.bias", "shape": [512], "dtype": "float32"},
    ],
    "rank": 0,
    "websocket": <WebSocket>,
}
```

**Deduplication:** Participants are matched by `(pod_ip, role)` - one participant per pod per role, with all their tensors aggregated.

### Filesystem Broadcast Flow (Rolling Participation)

```
Getter Pod                    Metadata Server
──────────                    ───────────────

ws://server/ws/fs-broadcast/{group_id}
     │
     │  {"action": "join",
     │   "key": "model/weights",
     │   "pod_ip": "10.0.0.5",
     │   "fanout": 50} ─────────►│
     │                           │
     │                           ▼
     │                    Lookup source from metadata
     │                    Get or create broadcast group
     │                    Assign rank (1-indexed, 0 = source)
     │                    Compute parent: (rank-1) // fanout
     │                           │
     │◄────────────────── {"event": "ready",
     │                     "rank": 5,
     │                     "parent_ip": "10.0.0.1",  # Source or earlier getter
     │                     "src_path": "default/svc/model/weights",  # Full rsync path
     │                     "source_ip": "10.0.0.1"}
     │
     ▼
Connection closes
     │
     ▼
rsync from parent_ip ──────────────────────────────► Parent serves via rsync
     │
     ▼
Start local rsync daemon
(becomes potential parent for later joiners)
```

**Key differences from GPU broadcast:**
- Rolling participation: returns immediately with parent info (no quorum wait)
- Only getters participate (source discovered from metadata)
- Tree topology with configurable fanout (default 50 for filesystem)
- No completion notification needed - client closes connection after receiving ready
- Source path (`src_path`) included in ready response for rsync:
  - Store-backed keys use `get_store_src_path(key)` → `{namespace}/{key}`
  - Peer sources use the `src_path` they published with

## Key Design Decisions

### 1. In-Memory Metadata
All metadata is stored in memory for fast access. This is acceptable because:
- Data is recoverable (re-publish from pods)
- Cleanup thread removes stale entries after 1 hour
- Pod restarts are infrequent

### 2. Load-Balanced Source Selection
The `get_source` endpoint implements load balancing:
- Tracks concurrent requests per source (store pod and peers)
- Returns sources below `MAX_CONCURRENT_PER_SOURCE` (default 30)
- Random selection among available sources
- Returns 503 when all sources overloaded

### 3. Peer-to-Peer Architecture
Instead of always going through the store pod:
- Pods can register as sources (`publish` endpoint)
- Getters try peer pods first
- Reduces store pod bottleneck
- Automatic failover (remove unreachable source, retry)

### 4. GPU Broadcast Coordination
NCCL broadcasts require all participants to:
- Know each other's ranks
- Agree on master address/port
- Start simultaneously

The WebSocket-based quorum system:
- Collects participants until condition met (timeout/world_size/ips)
- Assigns ranks deterministically (join order)
- Consolidates same-pod participants to one NCCL rank
- Notifies all when ready

### 5. WebSocket Tunnel for External Access
Rsync uses raw TCP which can't traverse firewalls/load balancers easily. The WebSocket proxy:
- Accepts WebSocket on port 8080
- Tunnels to local rsync daemon on port 873
- Allows external clients (e.g., local development) to rsync

### 6. Quorum Closing Conditions (OR semantics)
GPU broadcast quorums close when ANY condition is met:
- `timeout`: N seconds elapsed
- `world_size`: N participants joined
- `ips`: All specified IPs joined

This "OR" semantic prevents deadlock (waiting forever for a pod that crashed).

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `METADATA_SERVER_PORT` | 8081 | Port for metadata server |
| `METADATA_SERVER_HOST` | 0.0.0.0 | Bind address |
| `MAX_CONCURRENT_PER_SOURCE` | 30 | Max concurrent requests per source |
| `SOURCE_TIMEOUT` | 3600 | Seconds before removing stale sources |
| `DATA_ROOT` | /data | Filesystem root for stored data |
| `POD_NAMESPACE` | default | Kubernetes namespace |
| `POD_IP` | - | This pod's IP (for store registration) |
| `DEFAULT_GPU_QUORUM_TIMEOUT` | 5.0 | Default GPU quorum timeout |
| `NCCL_PORT_START` | 29500 | Starting port for NCCL |

## Container Ports

| Port | Protocol | Service |
|------|----------|---------|
| 873 | TCP | Rsync daemon |
| 8080 | HTTP/WS | WebSocket proxy (external rsync) |
| 8081 | HTTP/WS | Metadata server API |

## Testing

```bash
# Start server locally
python server.py

# Run tests
python test_server.py

# Manual testing
curl http://localhost:8081/health
curl -X POST http://localhost:8081/api/v1/keys/test-key/publish \
  -H "Content-Type: application/json" \
  -d '{"ip": "10.0.0.5"}'
curl http://localhost:8081/api/v1/keys/test-key/source
curl http://localhost:8081/api/v1/stats
```
