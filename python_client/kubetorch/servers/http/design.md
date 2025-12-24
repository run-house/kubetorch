# HTTP Server Design

## Overview

The HTTP server (`http_server.py`) is the main entry point for the KubeTorch pod runtime. It handles:
- Loading and executing user callables (functions/classes)
- Distributed execution coordination
- Log capture and streaming
- Metrics collection for TTL and monitoring
- Health checks and lifecycle management

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

## Lifecycle

1. **Startup** (in `lifespan` context manager):
   - Initialize LogCapture if `KT_LOG_STREAMING_ENABLED`
   - Initialize MetricsPusher if `KT_METRICS_ENABLED`
   - Register SIGTERM handler for graceful shutdown

2. **Runtime**:
   - Logs are captured and pushed continuously
   - Metrics are pushed every 15 seconds
   - Request middleware tracks active requests

3. **Shutdown**:
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
| `POD_NAME` | - | Pod name for log labels |
| `POD_NAMESPACE` | `default` | Namespace for service discovery |
| `KT_SERVICE` | - | Service name for log/metrics labels (set by pod template) |
| `LOG_STORE_HOST` | auto | Log store hostname |
| `LOG_STORE_PORT` | `3100` | Log store port |
