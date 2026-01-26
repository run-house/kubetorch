# Kubetorch Monarch Integration Design

## Overview

This module enables external Monarch access via Kubetorch's proxy infrastructure. Users can create Monarch HostMeshes, ProcMeshes, and ActorMeshes from outside the Kubernetes cluster, calling actor methods as if they were running locally.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Local Machine (Outside Cluster)                                         │
│                                                                          │
│  from kubetorch.monarch import KubernetesJob                            │
│                                                                          │
│  job = KubernetesJob(compute=kt.Compute(...))                           │
│  state = job.state()                                                    │
│  actors = state.workers.spawn_procs(...).spawn("name", MyActor)         │
│  result = actors.method.call(arg).get()                                 │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Proxy Classes (HostMeshProxy, ProcMeshProxy, ActorMeshProxy)    │   │
│  │  - Local operations: slice, size (no network)                     │   │
│  │  - Remote operations: spawn, call → WebSocket to gateway          │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              │ WebSocket (/ws/callable)                  │
└──────────────────────────────┼───────────────────────────────────────────┘
                               │
┌──────────────────────────────┼───────────────────────────────────────────┐
│  Kubernetes Cluster          │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Pod (runs MonarchGateway via kt.cls + Monarch Worker)           │   │
│  │                                                                   │   │
│  │  1. Monarch Worker (background process):                          │   │
│  │     - Started via run_bash() in image setup                       │   │
│  │     - Runs run_worker_loop_forever(address, ca)                   │   │
│  │     - Listens on tcp://<pod_ip>:26600                             │   │
│  │                                                                   │   │
│  │  2. MonarchGateway (kt.cls):                                      │   │
│  │     - Bootstraps Monarch root client                              │   │
│  │     - Discovers workers via headless service DNS                  │   │
│  │     - Attaches to workers (attach_to_workers)                     │   │
│  │     - Executes spawn_procs, spawn_actors, call_endpoint           │   │
│  │     - Maintains references to meshes/actors/futures               │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              │ TCP (tcp://pod_ip:26600)                  │
│                              ▼                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                      │
│  │  Worker 0   │  │  Worker 1   │  │  Worker N   │                      │
│  │  (Monarch)  │  │  (Monarch)  │  │  (Monarch)  │                      │
│  └─────────────┘  └─────────────┘  └─────────────┘                      │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Components

### Client Side (proxy.py, job.py)

- **KubernetesJob**: Entry point. Deploys MonarchGateway via kt.cls, establishes WebSocket connection, returns JobState with HostMeshProxy.
- **GatewayConnection**: Handles WebSocket communication with the gateway. Supports both WebSocket and HTTP fallback.
- **HostMeshProxy, ProcMeshProxy, ActorMeshProxy**: Local proxy objects mirroring Monarch's API. Local operations (slice, size) execute locally; remote operations (spawn, call) go through the gateway.
- **EndpointProxy**: Provides call(), call_one(), broadcast() for actor endpoints.
- **FutureProxy**: Wraps future IDs, calls gateway to get results when .get() is invoked.

### Server Side (gateway.py)

- **MonarchGateway**: Regular Python class deployed via kt.cls. Maintains state (host_mesh, proc_meshes, actor_meshes, futures). Exposes methods that the client calls over WebSocket/HTTP.

## Communication Flow

1. **Job creation**: `KubernetesJob(compute=...)` stores config
2. **Apply**: Deploys MonarchGateway, establishes WebSocket connection
3. **Initialize**: Gateway discovers worker IPs via headless DNS, calls `attach_to_workers()`
4. **Operations**: Client calls proxy methods → WebSocket message → Gateway executes Monarch API → Response back

## Distribution Mode

Uses `distribution_type="local"` to:
- Avoid SPMD auto-propagation (Monarch handles its own distribution)
- Still create headless service for worker discovery

## Worker Bootstrap

Each pod runs a Monarch worker process (`run_worker_loop_forever`) as a background process:

1. **Default Image** (selected based on GPU usage):
   - **CPU-only**: `kt.images.Debian()` - standard Kubetorch image, fast to pull
   - **GPU**: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` - CUDA support

2. **Image Configuration** (in `_configure_monarch_image`):
   - For CPU-only: installs PyTorch CPU version via `--index-url https://download.pytorch.org/whl/cpu`
   - Installs `torchmonarch-nightly` via pip (provides the `worker` CLI)

3. **Worker Startup** (in `MonarchGateway.__init__`):
   - Checks if port 26600 is already in use (worker already running)
   - If not, starts worker via subprocess running Python inline script
   - Uses `run_worker_loop_forever(address=..., ca="trust_all_connections")` from `monarch.actor`
   - Same approach as SkyPilot's Monarch integration

4. **Gateway Connection**:
   - MonarchGateway discovers all worker IPs via headless DNS
   - Calls `attach_to_workers()` with list of `tcp://<ip>:26600` addresses
   - Creates HostMesh that spans all workers

## WebSocket Endpoint

Added `/ws/callable` endpoint in http_server.py that mirrors the HTTP POST endpoint but over persistent WebSocket. Benefits:
- Lower latency for frequent calls
- Bidirectional communication for future push notifications
- Better suited for interactive/notebook workflows

## General WebSocket Support

Beyond Monarch, WebSocket connection mode is available for any `kt.cls` or `kt.fn`:

```python
# Use WebSocket instead of HTTP for calls
remote = kt.cls(MyClass).to(
    kt.Compute(cpu="1"),
    connection_mode="websocket",  # "http" (default) or "websocket"
)
```

Implementation:
- **WebSocketClient** (http_client.py): Alternative to HTTPClient, maintains persistent WebSocket connection
- **connection_mode** parameter in `Module.to()` and `Module.to_async()`
- **_client()** method returns appropriate client based on `_connection_mode`

The nginx proxy is already configured to support WebSocket upgrades, so no infrastructure changes are needed.
