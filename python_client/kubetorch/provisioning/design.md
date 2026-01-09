# Kubetorch Provisioning Design

This document describes the architecture and lifecycle of kubetorch compute provisioning, covering both **standard provisioning** (kubetorch creates K8s resources) and **BYO (Bring Your Own) compute** (user creates K8s resources, kubetorch manages execution).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Client Machine                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  kt.fn(my_func).to(compute)  or  kt.cls(MyClass).to(compute)        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Kubernetes Cluster                                 │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     Kubetorch Controller                              │  │
│  │  - POST /controller/deploy (apply manifest + register pool)          │  │
│  │  - POST /controller/pool (register pool only, for BYO)               │  │
│  │  - WebSocket /controller/ws/pods (pod connections)                   │  │
│  │  - Tracks pools, pod IPs, broadcasts reloads                         │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│          │                              ▲                                   │
│          │ K8s API                      │ WebSocket                         │
│          ▼                              │                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         Kubetorch Pods                                │  │
│  │  - Run kubetorch HTTP server (FastAPI)                               │  │
│  │  - Connect to controller via WebSocket on startup                    │  │
│  │  - Receive metadata/reload pushes from controller                    │  │
│  │  - Execute user callables (functions/classes)                        │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Components

### Client Side
- **`Compute`** (`resources/compute/compute.py`): Defines resource requirements (CPU, memory, GPU, image)
- **`Module`** (`resources/callables/module.py`): Wraps user's function/class, handles `.to()` deployment
- **`ServiceManager`** (`provisioning/service_manager.py`): Manages K8s resource lifecycle

### Controller Side
- **Pool Registry**: Database tracking registered pools, their selectors, and metadata
- **Pod Watcher**: Tracks pod IPs for routing via label selectors
- **WebSocket Manager**: Maintains connections to pods, broadcasts reloads

### Pod Side
- **HTTP Server** (`serving/http_server.py`): FastAPI app handling function calls
- **ControllerWebSocket**: Maintains connection to controller, receives metadata/reloads
- **Supervisor**: Manages subprocess workers for function execution

---

## Standard Provisioning Path

In standard provisioning, kubetorch creates and manages the K8s resources (Deployment, Knative Service, etc.).

### 1. Initial Setup (`kt.fn(func).to(compute)`)

```
Client                          Controller                      Kubernetes
  │                                │                                │
  │  1. Build manifest from        │                                │
  │     Compute config             │                                │
  │                                │                                │
  │  2. Rsync code to data store   │                                │
  │────────────────────────────────>                                │
  │                                │                                │
  │  3. POST /controller/deploy    │                                │
  │     {manifest, module_info,    │                                │
  │      pool_metadata}            │                                │
  │────────────────────────────────>                                │
  │                                │  4. Apply manifest             │
  │                                │─────────────────────────────────>
  │                                │                                │
  │                                │  5. Register pool in DB        │
  │                                │     Start pod watcher          │
  │                                │                                │
  │                                │  6. Broadcast reload via WS    │
  │                                │     (if pods connected)        │
  │                                │                                │
  │  7. Return success             │                                │
  │<────────────────────────────────                                │
  │                                │                                │
  │  8. Poll for service ready     │                                │
  │────────────────────────────────>                                │
```

**Details:**
1. `Compute._launch()` builds a K8s manifest from the template (deployment, knative, etc.)
2. User code is rsynced to the data store for pods to fetch
3. `ServiceManager.create_or_update_service()` calls controller's `/deploy` endpoint
4. Controller applies the manifest via K8s API
5. Pool is registered in database with selector, module info, and metadata
6. If pods are already connected (e.g., on redeploy), controller broadcasts reload
7. Client receives success response
8. Client polls `/check-ready` until pods are running

### 2. Pod Startup

```
Pod                             Controller
  │                                │
  │  1. Start kubetorch server     │
  │     (KT_SERVICE env var set)   │
  │                                │
  │  2. Connect WebSocket          │
  │────────────────────────────────>
  │     {action: "register",       │
  │      service_name, pod_name,   │
  │      request_metadata: true}   │
  │                                │
  │  3. Lookup pool in DB          │
  │                                │
  │  4. Send metadata              │
  │<────────────────────────────────
  │     {action: "metadata",       │
  │      module, runtime_config}   │
  │                                │
  │  5. Apply metadata             │
  │     (set env vars)             │
  │                                │
  │  6. Run image setup            │
  │     (pip installs, etc.)       │
  │                                │
  │  7. Load callable              │
  │     (import user code)         │
  │                                │
  │  8. Server ready for calls     │
```

**Details:**
- Pod manifest includes `KT_SERVICE` env var identifying the pool
- `ControllerWebSocket` connects to controller on startup
- Controller looks up pool by service name, sends stored metadata
- Pod applies metadata by setting environment variables
- Image setup runs any pip installs defined in the `Image` config
- Callable is loaded and supervisor is initialized

### 3. Making Calls

```
Client                          K8s Service                     Pod
  │                                │                              │
  │  1. POST /func_name            │                              │
  │     {args, kwargs}             │                              │
  │────────────────────────────────>                              │
  │                                │  2. Route to pod             │
  │                                │──────────────────────────────>
  │                                │                              │
  │                                │  3. Validate callable        │
  │                                │     matches config           │
  │                                │                              │
  │                                │  4. Execute in subprocess    │
  │                                │     via Supervisor           │
  │                                │                              │
  │                                │  5. Return result            │
  │                                │<──────────────────────────────
  │  6. Return to client           │                              │
  │<────────────────────────────────                              │
```

**Details:**
- Client makes HTTP POST to K8s Service (routed via controller proxy or direct)
- Pod's `/call/{callable_name}` endpoint validates request matches configured callable
- Supervisor dispatches call to subprocess worker
- Result is serialized and returned

### 4. Redeploying / Reloading

```
Client                          Controller                      Pod
  │                                │                              │
  │  1. kt.fn(new_func).to(compute)│                              │
  │                                │                              │
  │  2. Rsync updated code         │                              │
  │────────────────────────────────>                              │
  │                                │                              │
  │  3. POST /controller/deploy    │                              │
  │     {module: new_func_info}    │                              │
  │────────────────────────────────>                              │
  │                                │  4. Update pool in DB        │
  │                                │                              │
  │                                │  5. Broadcast reload         │
  │                                │──────────────────────────────>
  │                                │     {action: "reload",       │
  │                                │      module: new_func_info}  │
  │                                │                              │
  │                                │  6. Apply new metadata       │
  │                                │     Run image setup          │
  │                                │     Reload callable          │
  │                                │                              │
  │                                │  7. Send ack                 │
  │                                │<──────────────────────────────
  │                                │                              │
  │  8. Return success             │                              │
  │<────────────────────────────────                              │
```

**Details:**
- Same `.to()` call triggers update flow
- Controller broadcasts reload to all connected pods for this service
- Pods receive reload, apply new metadata, and recreate supervisor
- No pod restart required - hot reload of callable

### 5. Teardown

```
Client                          Controller                      Kubernetes
  │                                │                                │
  │  1. kt.teardown(service_name)  │                                │
  │────────────────────────────────>                                │
  │                                │  2. Delete K8s resource        │
  │                                │─────────────────────────────────>
  │                                │                                │
  │                                │  3. Delete pool from DB        │
  │                                │     Stop pod watcher           │
  │                                │                                │
  │  4. Return success             │                                │
  │<────────────────────────────────                                │
```

---

## BYO (Bring Your Own) Compute Path

In BYO mode, users create their own K8s resources and kubetorch just manages execution. This is useful when users need full control over pod configuration.

### 1. Initial Setup

```
User                            Kubernetes
  │                                │
  │  1. Create pod via kubectl     │
  │     or K8s API                 │
  │     (with kubetorch image,     │
  │      --pool argument)          │
  │────────────────────────────────>
  │                                │
```

**Pod startup command:**
```bash
kubetorch server start --pool my-service-name
```

This sets `KT_SERVICE=my-service-name` before starting the HTTP server.

### 2. Pod Connects to Controller

```
Pod                             Controller
  │                                │
  │  1. Connect WebSocket          │
  │────────────────────────────────>
  │     {action: "register",       │
  │      service_name: "my-svc",   │
  │      request_metadata: true}   │
  │                                │
  │  2. Pool not registered yet    │
  │<────────────────────────────────
  │     {action: "waiting",        │
  │      message: "Pool not        │
  │      registered yet..."}       │
  │                                │
  │  3. Server starts in           │
  │     selector-only mode         │
  │     (no callable configured)   │
```

**Details:**
- Pod connects but pool doesn't exist in controller DB yet
- Controller tells pod to wait
- Pod starts HTTP server without a callable loaded

### 3. Client Deploys to BYO Compute

```python
# Client code
compute = kt.Compute(selector={"app": "my-pod"})  # No resource specs
remote_fn = kt.fn(my_func, name="my-service-name").to(compute)
```

```
Client                          Controller                      Pod
  │                                │                              │
  │  1. POST /controller/pool      │                              │
  │     {name: "my-service-name",  │                              │
  │      selector: {"app": "..."}, │                              │
  │      module: func_info}        │                              │
  │────────────────────────────────>                              │
  │                                │  2. Register pool in DB      │
  │                                │     Start pod watcher        │
  │                                │                              │
  │                                │  3. Broadcast reload         │
  │                                │──────────────────────────────>
  │                                │     {action: "reload",       │
  │                                │      module: func_info}      │
  │                                │                              │
  │                                │  4. Apply metadata           │
  │                                │     Run image setup          │
  │                                │     Load callable            │
  │                                │                              │
  │                                │  5. Send ack                 │
  │                                │<──────────────────────────────
  │                                │                              │
  │  6. Return success             │                              │
  │<────────────────────────────────                              │
```

**Key differences from standard path:**
- Client uses `POST /controller/pool` instead of `/deploy` (no manifest)
- `Compute.selector_only` mode - no manifest built or applied
- Controller just registers pool and broadcasts to existing pods
- Pod receives callable info via WebSocket push

### 4. Making Calls

Same as standard path - HTTP requests routed through K8s Service to pods.

### 5. Redeploying Different Callable

```python
# Deploy a different function to the same BYO pod
remote_cls = kt.cls(MyClass, name="my-service-name").to(compute)
```

Same flow as step 3 - controller broadcasts new module info, pod hot-reloads.

### 6. Teardown

```
Client                          Controller
  │                                │
  │  1. kt.teardown(service_name)  │
  │────────────────────────────────>
  │                                │
  │  2. Delete pool from DB        │
  │     (no K8s resource to        │
  │     delete - user manages)     │
  │                                │
  │  3. Return success             │
  │<────────────────────────────────
```

**Note:** BYO pods are not deleted - user manages their lifecycle.

---

## Key Files

| File | Purpose |
|------|---------|
| `provisioning/service_manager.py` | Unified K8s resource management |
| `provisioning/utils.py` | Manifest building, resource configs |
| `provisioning/constants.py` | Labels, annotations, defaults |
| `resources/compute/compute.py` | Compute configuration, `_launch()` |
| `resources/callables/module.py` | `.to()` implementation |
| `serving/http_server.py` | Pod HTTP server, ControllerWebSocket |
| `globals.py` | Controller client, `register_pool()` |

---

## Important Considerations

### Single Worker Requirement
The controller must run with a single uvicorn worker (`WORKERS=1`) because WebSocket connections are stored in-memory. Multiple workers would cause pods to connect to one worker while HTTP requests go to another, breaking the broadcast mechanism.

### WebSocket Connection Lifecycle
- Pods maintain persistent WebSocket connection to controller
- Connection includes automatic reconnection with exponential backoff
- Blocking operations in reload handler use `asyncio.to_thread()` to avoid dropping connection

### Environment Variables Flow
Module metadata flows via WebSocket rather than manifest env vars:
- `KT_MODULE_NAME`: Python module containing the callable
- `KT_CLS_OR_FN_NAME`: Name of the function/class
- `KT_FILE_PATH`: Path to the module file
- `KT_INIT_ARGS`: JSON-encoded init args for classes
- `KT_USERNAME`: User who deployed the service
- `KT_SERVICE`: Service/pool name for routing
