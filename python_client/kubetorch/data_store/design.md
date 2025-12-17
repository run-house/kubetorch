# Data Store Client Design

This directory contains the client-side implementation of the kubetorch data store - a key-value store interface for transferring data to and from Kubernetes clusters.

## Overview

The data store provides a unified `put()`/`get()` API for two fundamentally different data types:
- **Filesystem data**: Files/directories transferred via rsync
- **GPU data**: CUDA tensors/state dicts transferred via NCCL broadcast

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Code                                   │
│              kt.put(key, src=...) / kt.get(key, dest=...)       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  data_store_cmds.py                             │
│         Module-level convenience functions (put/get/ls/rm)      │
│         Auto-detects filesystem vs GPU based on parameters      │
└──────────────┬──────────────────────────────────┬───────────────┘
               │                                  │
               ▼                                  ▼
┌──────────────────────────────┐   ┌─────────────────────────────┐
│   data_store_client.py       │   │    gpu_transfer.py          │
│   DataStoreClient class      │   │    GPUTransferManager       │
│   Filesystem data via rsync  │   │    GPU data via NCCL        │
└──────────────┬───────────────┘   └──────────────┬──────────────┘
               │                                  │
               ▼                                  ▼
┌──────────────────────────────┐   ┌─────────────────────────────┐
│     rsync_client.py          │   │    pod_data_server.py       │
│     RsyncClient class        │   │    Per-node data server     │
│     Low-level rsync ops      │   │    GPU + FS broadcast coord │
└──────────────┬───────────────┘   └─────────────────────────────┘
               │
               ▼
┌──────────────────────────────┐
│  websocket_tunnel.py         │
│  WebSocket tunnel for rsync  │
│  (external client access)    │
└──────────────────────────────┘
               │
               ▼
┌──────────────────────────────┐   ┌─────────────────────────────┐
│  metadata_client.py          │   │  types.py / key_utils.py    │
│  MetadataClient              │   │  BroadcastWindow, Locale,   │
│  Communicates with server    │   │  Lifespan, ParsedKey        │
└──────────────────────────────┘   └─────────────────────────────┘
```

## Module Responsibilities

### `__init__.py`
- Package entry point
- Exports public API: `put`, `get`, `ls`, `rm`, `rsync`, `rsync_async`
- Documents supported data types

### `data_store_cmds.py`
Module-level convenience functions that users call directly.

**Key functions:**
- `put(key, src=..., data=...)` - Upload filesystem or GPU data
- `get(key, dest=...)` - Download filesystem or GPU data
- `ls(key)` - List keys/contents
- `rm(key)` - Delete from store

**Auto-detection logic:**
- If `data` parameter is a CUDA tensor → GPU transfer via NCCL
- If `src` parameter is a path → Filesystem transfer via rsync

### `data_store_client.py`
High-level client for filesystem data operations.

**DataStoreClient class:**
- `put()` - Upload files via rsync with two modes:
  - `locale="store"` - Copy to central store pod
  - `locale="local"` - Zero-copy mode, register with metadata server only
- `get()` - Download files with peer-to-peer optimization
- `ls()` / `rm()` / `mkdir()` - Directory operations

**Data flow for put (locale="store"):**
1. Parse key to determine storage path
2. Create RsyncClient with appropriate service name
3. Upload via rsync (direct in-cluster or via WebSocket tunnel)
4. Register key with metadata server

**Data flow for get:**
1. Query metadata server for source info
2. If in-cluster: Try peer-to-peer transfer first, fall back to store
3. If external: Port-forward to peer pod or use store pod

### `rsync_client.py`
Low-level rsync operations.

**RsyncClient class:**
- `upload()` / `download()` - Core rsync operations
- `build_rsync_command()` - Construct rsync CLI commands
- `run_rsync_command()` - Execute with error handling

**Transfer modes:**
- **In-cluster**: Direct rsync to pod service URL
- **External**: WebSocket tunnel to bypass firewall

### `websocket_tunnel.py`
Enables rsync from outside the cluster.

**WebSocketRsyncTunnel class:**
- Opens local TCP socket
- Tunnels traffic over WebSocket to cluster
- Handles connection multiplexing

### `metadata_client.py`
Client for the metadata server API.

**MetadataClient class:**
- `get_source_ip()` - Find peer pod with data
- `publish_key()` - Register local data availability
- `has_store_pod()` - Check if store has key
- `list_keys()` / `delete_key()` - Key management
- `join_broadcast()` / `get_broadcast_status()` - Coordinated transfers
- `cleanup_service_keys()` - Service teardown cleanup

### `gpu_transfer.py`
GPU tensor transfer via NCCL.

**GPUTransferManager class:**
- `publish()` - Register tensor and publish via GPU Data Server
- `retrieve()` - Receive tensor via GPU Data Server

**Transfer flow:**
1. Publish: Call `gpu_client.put_tensor()` which registers IPC handle and publishes to MDS
2. Retrieve: Call `gpu_client.get_tensor()` which queries MDS and performs NCCL transfer

**BroadcastWindow support:**
- Pass broadcast config to `put_tensor()`/`get_tensor()`
- GPU Data Server handles WebSocket coordination with MDS
- One coordinator per pod (the GPU Data Server itself)

### `pod_data_server.py`
Per-node server process for GPU transfers and filesystem broadcast coordination.

**PodDataServer class:**
- Runs as separate process per node
- Holds CUDA IPC handles (not tensors - memory owned by application)
- Performs NCCL broadcasts isolated from application processes
- Tracks registered tensors by PID, cleans up on process exit
- MDS HTTP client for source lookup and key registration
- MDS WebSocket client for broadcast group coordination
- Tracks completed filesystem broadcasts for inter-pod coordination
- Serves local paths to child getters in filesystem broadcast tree

**PodDataServerClient class:**
- Unix socket communication with server (local)
- TCP communication with remote pod data servers (inter-pod)
- High-level API for GPU:
  - `put_tensor()` - Register + MDS publish (+ broadcast coordination if specified)
  - `get_tensor()` - MDS lookup + NCCL receive (+ broadcast coordination if specified)
- High-level API for filesystem broadcasts:
  - `fs_broadcast_complete()` - Notify local server that download finished
  - `fs_broadcast_get_path()` - Request local path from remote parent's server
- Low-level API:
  - `register_tensor()` / `unregister_tensor()` - Just register, no MDS
  - `receive_broadcast()` - Just NCCL, no coordination
  - `execute_broadcast_group()` - Coordinated multi-party transfer

### `types.py`
Core type definitions.

- **Locale**: `"store"` (copy to central pod) or `"local"` (zero-copy)
- **Lifespan**: `"cluster"` (persistent) or `"resource"` (auto-cleanup)
- **BroadcastWindow**: Configuration for coordinated transfers
  - `timeout` - Max wait for participants
  - `world_size` - Expected participant count
  - `ips` - Wait for specific pods
  - `group_id` - Explicit group identifier
  - `fanout` - Tree fanout (default: 2 for GPU, ~50 for filesystem)

### `key_utils.py`
Key parsing utilities.

**ParsedKey dataclass:**
- `service_name` - First path segment (if looks like service name)
- `path` - Remaining path
- `storage_path` - Path for rsync operations
- `full_key` - Normalized key

## Data Flow Diagrams

### Filesystem Put (locale="store")

```
User: kt.put("my-svc/model", src="./weights/")
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ DataStoreClient.put()                                            │
│   1. Parse key → service_name="my-svc", path="model"            │
│   2. Create RsyncClient(service_name="my-svc")                  │
│   3. rsync_client.upload(source="./weights/", dest="model")     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
    In-Cluster                        External
    (direct rsync)                    (WebSocket tunnel)
          │                               │
          ▼                               ▼
┌─────────────────────┐        ┌────────────────────────┐
│ rsync → pod URL     │        │ WebSocketRsyncTunnel   │
│ rsync://svc:873/... │        │ localhost:port → WS    │
└─────────────────────┘        └────────────────────────┘
          │                               │
          └───────────────┬───────────────┘
                          ▼
                   Data Store Pod
                   (receives files)
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ metadata_client.register_store_pod(key, ...)                     │
│   Register key with metadata server for discovery                │
└─────────────────────────────────────────────────────────────────┘
```

### Filesystem Get (peer-to-peer optimization)

```
User: kt.get("my-svc/model", dest="./local/")
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ DataStoreClient.get()                                            │
│   1. Query metadata: get_source_ip() + has_store_pod()          │
│   2. Returns: {ip: "10.0.0.5", src_path: "model"}              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────────────┐
          ▼                                       ▼
    Peer Available                          No Peer / External
    (try P2P first)                         (use store pod)
          │                                       │
          ▼                                       ▼
┌─────────────────────────┐            ┌────────────────────────┐
│ rsync from peer IP      │            │ rsync from store pod   │
│ rsync://10.0.0.5:873/.. │            │ (via port-forward if   │
│                         │            │  external)             │
└─────────────────────────┘            └────────────────────────┘
          │                                       │
          ▼                                       │
    On failure:                                   │
    remove_source() + retry ──────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ metadata_client.complete_request(key, ip)                        │
│   Decrement concurrent request counter                           │
└─────────────────────────────────────────────────────────────────┘
```

### GPU Transfer Flow

```
Putter Pod                              Getter Pod
───────────                             ───────────
kt.put(key="layer1",                    kt.get(key="layer1",
       data=tensor)                            dest=tensor)
     │                                       │
     ▼                                       │
GPUTransferManager.publish()                 │
     │                                       │
     ▼                                       │
GPU Data Server                              │
  register_tensor(key, IPC_handle)           │
     │                                       │
     ▼                                       │
Metadata Server                              │
  publish GPU key                            │
     │                                       ▼
     │                              GPUTransferManager.retrieve()
     │                                       │
     │                                       ▼
     │                              Query metadata for source
     │                                       │
     │                                       ▼
     │                              GPU Data Server
     │                                receive_broadcast()
     │                                       │
     │◄──────────────────────────────────────┘
     │        Request NCCL broadcast
     ▼
NCCL Broadcast
  Master: Putter GPU Server
  Receiver: Getter GPU Server
     │
     ▼
Tensor data transferred via NCCL
```

### GPU Broadcast (BroadcastWindow)

For GPU data, broadcasts use NCCL with quorum-based coordination:

```
Pod A (Putter)              Metadata Server              Pod B (Getter)
─────────────               ───────────────              ─────────────
kt.put(key, data,           ws://server/ws/              kt.get(key, dest,
  broadcast=BW(             gpu-broadcast/{group}          broadcast=BW(...))
    world_size=2))                │                              │
     │                            │                              │
     ▼                            │                              ▼
GPU Data Server ──────────────────┤◄─────────────────── GPU Data Server
connects to WS                    │                     connects to WS
     │                            │                              │
     ▼                            ▼                              │
Send join                   Track participants by pod            │
{action: "join",            (unique by pod_ip)  ◄───────────────Join
 role: "putter", ...}             │                              │
     │                            │                              │
     ▼                            ▼                              │
Wait for quorum ◄────────── Quorum satisfied ──────────► Wait for quorum
     │                      - Assign pod ranks                   │
     │                      - Build pod manifests                │
     │                            │                              │
     ▼                            ▼                              ▼
All participants            Send ready to all            All participants
receive full manifest       participants (no coordinator) receive full manifest
     │                            │                              │
     ▼                            │                              ▼
GPU Data Server             (manifests sent)             GPU Data Server
executes NCCL once                │                      executes NCCL once
(first thread wins)               │                      (first thread wins)
     │                            │                              │
     ▼                            ▼                              ▼
Send "complete" ─────────────► Track by pod ◄──────────Send "complete"
     │                            │                              │
     ▼                            ▼                              ▼
Receive "completed"         All pods done             Receive "completed"
```

### Filesystem Broadcast (Rolling Participation)

For filesystem data, broadcasts use rsync with tree-based propagation and pod data server coordination:

```
Original Putter             Metadata Server              Getters (rolling join)
───────────────             ───────────────              ────────────────────
kt.put(key, src,
  locale="local")
     │
     ▼
Start rsync daemon
     │                                                   Getter 1           Getter 2
     ▼                                                   ────────           ────────
Register with MDS ──────────────────►                   kt.get(key,        kt.get(key,
(publishes source IP                │                     broadcast=BW)      broadcast=BW)
 and src_path)                      │                        │                  │
     │                              │                        ▼                  ▼
     ▼                              │                   Start pod data     Start pod data
Available as source                 │                   server (if needed) server (if needed)
for getters                         │                        │                  │
                                    │                        ▼                  ▼
                                    │                   Connect to WS      Connect to WS
                                    │                   (fs-broadcast)     (fs-broadcast)
                                    │                        │                  │
                                    │                        ▼                  ▼
                                    │                   Send join {key,    Send join {key,
                                    │                    pod_ip, ...}       pod_ip, ...}
                                    ▼                        │                  │
                              Lookup source from             │                  │
                              metadata (ip + src_path)       │                  │
                                    │                        │                  │
                                    ▼                        ▼                  ▼
                              Assign rank 1 ──────────► Ready immediately  Assign rank 2
                              Parent = source             {rank: 1,         Parent = Getter 1
                              (root node)                  parent_ip:            │
                                    │                       source_ip,           ▼
                                    │                       src_path: ...}  Ready {rank: 2,
                                    │                        │               parent_ip:
                                    │                        ▼               getter1_ip}
                                    │                   Rsync from               │
                                    │                   source (root)            │
                                    │                        │                   │
                                    │                        ▼                   ▼
                                    │                   Notify local        Request local path
                                    │                   pod data server     from Getter 1's
                                    │                   (fs_broadcast_      pod data server
                                    │                    complete)          (fs_broadcast_
                                    │                        │               get_path)
                                    │                        │                   │
                                    │                        │                   ▼
                                    │                        │              Pod data server
                                    │                        │              blocks until
                                    │                        │              Getter 1 completes
                                    │                        │                   │
                                    │                        │◄──────────────────┤
                                    │                        │              Returns local path
                                    │                        │                   │
                                    │                        ▼                   ▼
                                    │                   Start rsync        Rsync from
                                    │                   daemon (now a       Getter 1's
                                    │                   potential parent)   local path
                                    │                        │                   │
                                    │                        ▼                   ▼
                                    │                   Send "complete"     Notify local
                                    │◄───────────────────────┘              pod data server
                                    │                                            │
                                    │                                            ▼
                                    │                                       Send "complete"
                                    │◄───────────────────────────────────────────┘
```

**Key differences from GPU broadcast:**
- Only getters participate (source discovered from metadata)
- Rolling participation - returns immediately with parent info
- Tree topology with configurable fanout (~50 for filesystem vs 2 for GPU)
- Each getter becomes a potential parent after completing download
- Pod data server handles inter-pod coordination:
  - Tracks completed broadcasts with local paths
  - Child getters request parent's local path via TCP (not MDS)
  - Blocks until parent completes, then returns absolute path for rsync

## Key Design Decisions

### 1. Unified API for Heterogeneous Data
The same `put()`/`get()` interface handles both filesystem and GPU data. Auto-detection based on parameter types (`src` vs `data`, path vs tensor) keeps user code simple.

### 2. Peer-to-Peer Optimization
For filesystem data, the metadata server tracks which pods have each key. Getters try peer-to-peer transfer first, falling back to the central store only if needed. This reduces load on the store pod.

### 3. GPU Server Isolation
NCCL operations run in a separate per-node process (`GPUDataServer`) rather than in application processes. This:
- Prevents NCCL state corruption from affecting applications
- Allows multiple application processes to share NCCL infrastructure
- Enables automatic cleanup when processes die

### 4. IPC Handle Architecture
GPU tensors are not copied to the GPU server. Instead:
- Application registers tensor's CUDA IPC handle with server
- Server reconstructs tensor view when needed for NCCL
- Memory ownership stays with application
- Automatic cleanup via PID monitoring

### 5. Lifespan Management
Keys can have `lifespan="resource"` for automatic cleanup when a service is torn down. The metadata server tracks service associations and cleans up on service deletion.

### 6. BroadcastWindow Coordination
For coordinated transfers, BroadcastWindow supports two modes:

**GPU Broadcast (NCCL):**
- Quorum-based: wait for all participants before starting
- All participants join via GPU Data Server
- Pod-level rank assignment
- GPU Data Server executes NCCL once per pod
- Binary tree topology (fanout=2)

**Filesystem Broadcast (rsync):**
- Rolling participation: return immediately with parent info
- Tree-based propagation for efficient distribution
- High fanout (~50) for parallel downloads
- Each completed getter becomes a potential parent
- Automatic rsync daemon startup
- Pod data server coordination for parent-child communication:
  - Parent notifies local pod data server when download complete
  - Child requests local path from parent's pod data server (TCP)
  - Pod data server blocks until parent completes, then returns absolute path
  - Uses absolute paths (via `Path.resolve()`) for rsync daemon compatibility
