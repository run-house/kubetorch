# HTTP Server Design

## Overview

The HTTP server (`http_server.py`) is the main entry point for the KubeTorch pod runtime. It handles:
- Loading and executing user callables (functions/classes)
- Distributed execution coordination
- Log capture and streaming
- Metrics collection for TTL and monitoring
- Health checks and lifecycle management

## Execution Supervisor Hierarchy

All user code runs in isolated subprocesses via the supervisor system:

```
ExecutionSupervisor (base - ProcessPool only)
│   - Local subprocess execution
│   - No remote workers, no DNS, no quorum
│   - Used for non-distributed mode (default)
│
└── DistributedSupervisor (adds distributed capabilities)
    │   - DNS-based pod discovery with quorum
    │   - Worker membership monitoring
    │   - RemoteWorkerPool for cross-pod execution (created lazily when needed)
    │
    ├── SPMDDistributedSupervisor
    │   - Multi-process local execution (num_proc configurable)
    │   - Tree topology for large clusters (>100 workers)
    │   - Framework-specific process classes (PyTorch, JAX, TensorFlow)
    │
    ├── RayDistributed
    │   - Ray GCS server management
    │   - Head-node only execution pattern
    │   - Disables DNS monitoring (Ray manages membership)
    │
    └── MonarchDistributed
        - process_allocator service management
        - Single controller pattern like Ray
        - Disables DNS monitoring (Monarch manages actors)
```

### Execution Flow

```
HTTP POST /{callable_name}
    ↓
run_callable() in http_server.py
    ↓
load_callable()
    ├─→ Creates ExecutionSupervisor via supervisor_factory()
    ├─→ Default: "local" for non-distributed mode
    └─→ Distributed: "pytorch", "ray", "monarch", etc.
    ↓
SUPERVISOR.call_distributed()
    ├─→ Local: Routes to subprocess via ProcessPool
    └─→ Distributed: Coordinates across local processes + remote workers
```

### Subprocess Isolation Benefits

All user code runs in subprocesses (ProcessPool + ProcessWorker):
- **Module isolation**: User code cannot corrupt main HTTP server
- **Clean reload**: On redeployment, terminate subprocess and recreate
- **Consistent pattern**: Same ProcessPool for local and distributed modes

### ProcessWorker Concurrency Model

ProcessWorker matches FastAPI's concurrency model for user expectations:

```
ProcessWorker subprocess:
  asyncio event loop
    ├─→ Async callables: awaited directly (true async concurrency)
    │     - Many can run concurrently via cooperative multitasking
    │     - No thread overhead, scales to thousands of concurrent calls
    │
    └─→ Sync callables: run via run_in_executor()
          - Offloaded to ThreadPoolExecutor (default: 40 threads)
          - Don't block the event loop
```

Key implementation details:
- `execute_callable_async()`: Used by ProcessWorker, handles async/sync dispatch
- Thread pool size matches FastAPI's default (40 threads) for consistent behavior

### Redeployment (Push-based via WebSocket)

When user code changes (`.to()` called with new callable):
1. Client rsyncs code to data store
2. Client registers pool with controller via `/pool` endpoint
3. Controller broadcasts reload to all connected pods via WebSocket
4. Each pod's `_handle_reload()` receives the message:
   - Applies new metadata (env vars)
   - Runs image setup (rsync, pip installs)
   - Clears callable cache
   - Supervisor's `cleanup()` terminates subprocesses
   - Supervisor's `setup()` creates fresh subprocesses
5. Pod sends acknowledgment back to controller
6. Controller returns success to client after all pods acknowledge

## Log Streaming (`log_capture.py`)

### Architecture

```
Main Process:
  stdout/stderr → LogCapture → Log Store (async batched)
                            ↘→ stdout/stderr (for kubectl logs)

Subprocesses (PDS, Ray, Allocator):
  stdout/stderr → PIPE → Reader Thread → LogCapture.add_log()
                                       ↘→ logger.info() (captured by LogCapture)
```

### Components

1. **LogCapture**: Main class that intercepts all stdout/stderr in the main process
   - Replaces `sys.stdout` and `sys.stderr` with `_StreamInterceptor`
   - Adds `_LogCaptureHandler` to root logger to capture Python logging
   - Batches log entries and pushes to log store asynchronously (100 entries or 1s interval)
   - Forwards all logs to original streams for `kubectl logs` compatibility
   - Provides `subprocess_queue` for multiprocessing.Process subprocesses

2. **Subprocess Log Forwarding**: For Popen-based subprocesses
   - Pod Data Server (PDS): Spawned via Popen, stdout/stderr piped and forwarded via reader threads
   - Ray: stdout/stderr piped and logged via `logger.info()` (captured by LogCapture's handler)
   - Allocator: stdout/stderr piped and logged via `logger.info()` (captured by LogCapture's handler)

### Configuration

- `KT_LOG_STREAMING_ENABLED`: Enable log streaming (default: true)
- `LOG_STORE_HOST` / `LOG_STORE_PORT`: Log store endpoint (auto-detected from `POD_NAMESPACE`)

### Labels

Logs are pushed with structured labels for querying:
- `service`: from KT_SERVICE env var
- `pod`: POD_NAME
- `namespace`: POD_NAMESPACE
- `level`: INFO/ERROR/DEBUG/etc.
- `request_id`: For request-scoped log filtering (from `request_id_ctx_var`)
- `source`: Subprocess source (e.g., "pds" for Pod Data Server)

### Request ID Propagation

The `request_id` label enables filtering logs for a specific remote call. Flow:

1. Client generates `request_id` and includes in `X-Request-ID` header
2. `RequestIDMiddleware` sets `request_id_ctx_var` context variable
3. `_LogCaptureHandler` and `_StreamInterceptor` read from `request_id_ctx_var`
4. Client queries Loki with `request_id` filter to stream logs in real-time

Note: Python logging filters on the root logger don't apply to records that propagate
from child loggers. LogCapture reads `request_id` directly from the context variable
rather than relying on filters.

## Metrics Collection (`metrics_push.py`)

### Architecture

```
Pod HTTP requests → MetricsPusher → Metrics Store (Prometheus Pushgateway)
```

### Components

1. **MetricsPusher**: Push metrics to metrics store
   - `http_requests_total`: Counter for HTTP requests by method/endpoint/status
   - `http_request_duration_seconds`: Histogram for request latency
   - `kubetorch_last_activity_timestamp`: Gauge for TTL tracking
   - `kt_heartbeat_sent`: Counter for activity heartbeats

2. **Middleware**: Tracks request metrics in FastAPI
   - Records request start/finish for active request gauge
   - Records request duration and status

### Configuration

- `KT_METRICS_ENABLED`: Enable metrics collection (default: true)
- Required for TTL (auto-scaling) to work

## Controller WebSocket (`ControllerWebSocket`)

### Overview

Pods maintain a persistent WebSocket connection to the kubetorch controller. This enables:
- Dynamic metadata delivery on pod startup
- Push-based reload when callable changes (no polling required)
- Acknowledgment-based deployment confirmation

### Architecture

```
Pod startup:
  lifespan() starts
    ↓
  ControllerWebSocket.start()
    ↓
  Connect to ws://kubetorch-controller.{install_namespace}/controller/ws/pods
    ↓
  Send registration: {pod_name, pod_ip, namespace, service_name, request_metadata: true}
    ↓
  Controller looks up pool in database
    ├─→ Found: Send metadata {module, runtime_config, service_dns, ...}
    └─→ Not found: Send {action: "waiting"} - pod waits for /pool call
    ↓
  _apply_metadata() sets env vars
    ↓
  run_image_setup() syncs files and runs pip installs
    ↓
  load_callable() creates supervisor

On redeployment (.to() called):
  Client calls /pool → Controller broadcasts reload to all connected pods
    ↓
  _handle_reload() receives reload message
    ↓
  Apply new metadata, run image setup, recreate supervisor
    ↓
  Send acknowledgment: {action: "reload_ack", status: "ok"}
    ↓
  Controller waits for all acks before returning to client
```

### Benefits

1. **Push-based reload**: No polling or timestamp checking required
2. **BYO compute support**: Pods can connect before pool is registered, receive metadata when ready
3. **Acknowledgment-based**: Controller confirms all pods processed reload before returning
4. **Simplified manifests**: Pod templates don't need module-specific env vars

### Pod Identity (without Downward API)

Pod metadata is derived without K8s Downward API env vars:
- `POD_NAME`: from `socket.gethostname()` (K8s sets hostname to pod name)
- `POD_NAMESPACE`: from `/var/run/secrets/kubernetes.io/serviceaccount/namespace`
- `POD_IP`: from `socket.gethostbyname(socket.gethostname())`

### Configuration

- `KT_INSTALL_NAMESPACE`: Namespace where kubetorch controller is installed (for URL construction)
- `KT_SERVICE`: Service name (still set via labels for log streaming)

## Lifecycle

1. **Startup** (in `lifespan` context manager):
   - Initialize LogCapture if `KT_LOG_STREAMING_ENABLED`
   - Initialize MetricsPusher if `KT_METRICS_ENABLED`
   - Start ControllerWebSocket and wait for metadata
   - Register SIGTERM handler for graceful shutdown

2. **Runtime**:
   - Logs are captured and pushed continuously
   - Metrics are pushed every 15 seconds
   - Request middleware tracks active requests
   - ControllerWebSocket maintains connection for reload messages

3. **Shutdown**:
   - Stop ControllerWebSocket
   - Stop LogCapture (flush remaining logs)
   - Stop MetricsPusher (push final metrics)
   - Cleanup distributed supervisor
   - Clear debugging sessions

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `KT_LOG_STREAMING_ENABLED` | `True` | Enable log streaming |
| `KT_METRICS_ENABLED` | `True` | Enable metrics collection |
| `KT_LOG_LEVEL` | `INFO` | Logging level |
| `KT_SERVICE` | - | Service name for log/metrics labels (set by pod template) |
| `KT_INSTALL_NAMESPACE` | `kubetorch` | Namespace for controller WebSocket URL |
| `LOG_STORE_HOST` | auto | Log store hostname |
| `LOG_STORE_PORT` | `3100` | Log store port |

Note: `POD_NAME`, `POD_NAMESPACE`, and `POD_IP` are derived at runtime without Downward API.
See "Pod Identity" section above.
