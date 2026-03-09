# Log Streaming Migration Plan

## User Requirements (Verbatim)

### Architecture Vision

> "If log streaming is enabled, Kubetorch application pods push their local log files to a queue API via their local daemonset data store pod. This can be done with an API call to the daemonset instructing it to tail a set of files constantly and write them to a particular queue (so the daemonset needs to see the log directories), or just have the application start a separate process which tails the log files and pushes them constantly to a queue put API in the daemonset pod."

> "The daemonset holds a persistent websocket connection to namespace-central metadata store (MDS) service, and makes queue put call to the MDS, which stores a) the queue metadata and 2) the actual stream. I think the queue put call should return the queue address (e.g. Redis service, port, name) to the daemonset to then write to the queue directly."

> "The local client requests the queue from the MDS, which returns metadata for it to query the queue directly, e.g. Redis (similar to how a kt.get returns the address of the rsync service, rather than proxying it directly, which the requester then requests from directly). For calls from outside the cluster, we can reuse the WebSocket tunnel we use for rsync to request from the queue system. The request should be verbose enough to query with prefilter (e.g. only return lines from the stream which contain x pod name or y request id), so we don't need to post-filter in the client."

> "My thought is also that it will be more scalable if all calls to the MDS route through the daemonset pods, which maintain a single websocket connection each to the MDS to route all requests, as opposed to every application pod making connections to the MDS for different use cases or using HTTP calls, which I think will be unscalable for the MDS."

### Migration Order

> "I think what I'd do is start by merging just the Loki binary into the data store image/container (Option A) to start, and then in parallel convert our existing pod data store (PDS) to a daemonset, then add queue support to the MDS, daemonsets, and Python client APIs, and *then* migrate over the log streaming to flow through this flow instead of OTEL, and then remove Loki and OTEL. So ideally this is a somewhat gradual transition."

### Data Store Philosophy

> "I like to think of the data store as a standalone component from Kubetorch, which Kubetorch is built on top of (and we may actually formally separate them soon into separate repos). From that perspective, log streaming functionality doesn't make a ton of sense as a part of a data store. However, I've been considering adding a queue/stream primitive to the data store anyway, that we could use to support service-like Kubetorch modules which need to pull requests off a queue (and move away from our dependency on Knative)."

### Backing Store Considerations

> "I'm more concerned with the memory consumption than the ephemerality. Indeed logs are transient so I don't mind if they're lost if the MDS restarts, but if storing the logs in memory is going to blow up compared to disk-backed (or rotating or spilling over if needed), then I'd rather disk-backed. I'm trying to keep the resource footprint low."

> "I'm not positive Redis is the right solution, it just came to mind because it supports Queues and KV data. But we should discuss the options if there are other tools or a combination of tools which are a better fit."

---

## Migration Phases

Each phase is a **separate PR**. Test and verify before proceeding to the next phase.

---

## Phase 1: Embed Loki in Data Store (PR #1)

### Goal
Move Loki binary from separate `kubetorch-logging` deployment into the data store container. Remove data store from install namespace.

### Current State
- `kubetorch-logging` deployment runs Loki in install namespace
- OTEL daemonset sends logs to `loki-gateway.{install-ns}.svc.cluster.local:3100`
- Data store deployed in each `deployment_namespace` + install namespace
- Clients query `/loki/api/v1/tail?query=...`

### Target State
- Loki embedded in data store container (each namespace has its own Loki)
- OTEL routes logs to namespace-specific data store
- Data store NOT deployed in install namespace (only deployment_namespaces)
- Clients query `/loki/{namespace}/api/v1/tail?query=...`

### Changes

#### 1. Data Store Container (`kubetorch-internal/services/data_store/`)

**Dockerfile** - Add Loki binary:
```dockerfile
FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y rsync net-tools curl unzip && \
    rm -rf /var/lib/apt/lists/*

# Download Loki binary (~50MB)
RUN curl -L https://github.com/grafana/loki/releases/download/v3.5.3/loki-linux-amd64.zip -o /tmp/loki.zip && \
    unzip /tmp/loki.zip -d /usr/local/bin/ && \
    chmod +x /usr/local/bin/loki-linux-amd64 && \
    mv /usr/local/bin/loki-linux-amd64 /usr/local/bin/loki && \
    rm /tmp/loki.zip

RUN pip install --no-cache-dir websockets fastapi uvicorn pydantic

RUN mkdir -p /data /var/log /tmp/loki

COPY websocket_tunnel_server.py /usr/local/bin/server_proxy.py
COPY server.py /app/server.py
COPY start.sh /entrypoint.sh
COPY loki.yaml /etc/loki/loki.yaml
RUN chmod +x /entrypoint.sh

EXPOSE 873 8080 8081 3100

CMD ["/entrypoint.sh"]
```

**loki.yaml** (NEW):
```yaml
auth_enabled: false
server:
  http_listen_port: 3100
common:
  path_prefix: /tmp/loki
  storage:
    filesystem:
      chunks_directory: /tmp/loki/chunks
      rules_directory: /tmp/loki/rules
  replication_factor: 1
  ring:
    kvstore:
      store: inmemory
ingester:
  lifecycler:
    address: 127.0.0.1
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1
    final_sleep: 0s
  chunk_idle_period: 1m
  chunk_retain_period: 30s
schema_config:
  configs:
    - from: "2024-04-01"
      store: tsdb
      object_store: filesystem
      schema: v13
      index:
        prefix: index_
        period: 24h
limits_config:
  retention_period: 24h
  allow_structured_metadata: true
  volume_enabled: true
  max_concurrent_tail_requests: 100
  unordered_writes: true
  max_query_lookback: 0
```

**start.sh** - Add Loki:
```bash
#!/bin/bash
set -e

echo "Starting Loki..."
/usr/local/bin/loki -config.file=/etc/loki/loki.yaml > /var/log/loki.log 2>&1 &
LOKI_PID=$!

echo "Starting rsync daemon..."
rsync --daemon --no-detach --config=/etc/rsyncd.conf &
RSYNC_PID=$!

echo "Starting WebSocket proxy..."
python /usr/local/bin/server_proxy.py > /var/log/websocket_proxy.log 2>&1 &
PROXY_PID=$!

echo "Starting metadata server..."
python -m uvicorn server:app --host 0.0.0.0 --port ${METADATA_SERVER_PORT:-8081} --app-dir /app &
METADATA_PID=$!

cleanup() {
    kill $LOKI_PID $RSYNC_PID $PROXY_PID $METADATA_PID 2>/dev/null || true
    wait 2>/dev/null || true
}

trap cleanup SIGTERM SIGINT
wait -n
cleanup
exit $?
```

#### 2. Helm Charts (`kubetorch/charts/kubetorch/`)

**templates/data-store/namespace-data-store.yaml**:
- Change line 1: Remove install namespace from range
  ```yaml
  # FROM:
  {{- range (.Values.kubetorchConfig.deployment_namespaces | default list | concat (list .Release.Namespace) | uniq) }}
  # TO:
  {{- range (.Values.kubetorchConfig.deployment_namespaces | default list) }}
  ```
- Add Loki port (3100) to Service
- Add port 3100 to Deployment container

**templates/log-streaming/opentelemetry-collector-configmap.yaml**:
- Add routing processor to send logs to correct namespace's data store
- Generate exporters for each deployment namespace

**templates/controller/configmap.yaml**:
- Replace `/loki/` route with namespace-aware `/loki/{namespace}/...`

**DELETE:**
- `templates/log-streaming/logs-deployment.yaml`
- `templates/log-streaming/logs-configmap.yaml`
- `templates/log-streaming/logs-service.yaml`

#### 3. Python Client (`kubetorch/python_client/`)

**cli_utils.py**:
- Update `follow_logs_in_cli()` to include namespace in Loki URL
- `uri = f"{base_url}/loki/{namespace}/api/v1/tail?query=..."`

**resources/callables/module.py**:
- Update `_stream_launch_logs()` similarly

### Files Changed Summary
- `kubetorch-internal/services/data_store/Dockerfile`
- `kubetorch-internal/services/data_store/start.sh`
- `kubetorch-internal/services/data_store/loki.yaml` (NEW)
- `kubetorch/charts/kubetorch/templates/data-store/namespace-data-store.yaml`
- `kubetorch/charts/kubetorch/templates/log-streaming/opentelemetry-collector-configmap.yaml`
- `kubetorch/charts/kubetorch/templates/controller/configmap.yaml`
- `kubetorch/charts/kubetorch/templates/log-streaming/logs-*.yaml` (DELETE)
- `kubetorch/python_client/kubetorch/cli_utils.py`
- `kubetorch/python_client/kubetorch/resources/callables/module.py`

---

## Phase 2: Convert Pod Data Store (PDS) to Daemonset (PR #2)

### Goal
Convert existing Pod Data Store to run as a daemonset instead of per-pod deployment.

### Key Requirements (from user)
- Daemonset has hostPath mount to `/var/log/pods` for log access
- Daemonset maintains single persistent WebSocket connection to namespace MDS
- All calls to MDS from pods on that node route through the daemonset
- This is more scalable than per-pod connections to MDS

### Changes
- Create new daemonset deployment template
- Add hostPath volume mount for log directories
- Implement WebSocket connection to namespace MDS
- Add MDS proxy functionality (forward requests from local pods)

### Files to Create/Modify
- `kubetorch/charts/kubetorch/templates/data-store/daemonset.yaml` (NEW)
- `kubetorch-internal/services/data_store/mds_proxy.py` (NEW)
- Values.yaml for daemonset mode toggle

---

## Phase 3: Add Queue Primitive to MDS (PR #3)

### Goal
Add general-purpose queue/stream primitive to the data store that can be used for:
1. Log streaming (this migration)
2. Request queuing for service-like modules (replace Knative dependency)
3. Other pub/sub use cases

### Key Requirements (from user)
- Queue put call should return queue address (e.g., Redis service, port, name)
- Daemonset writes to queue directly after getting address
- Clients query queue directly (similar to rsync pattern)
- Server-side pre-filtering (by pod_name, request_id) to avoid client post-filtering
- For external access, reuse WebSocket tunnel like rsync

### Backing Store Options (to evaluate)

| Option | Streams/Queues | KV Store | Footprint | Notes |
|--------|---------------|----------|-----------|-------|
| **Redis** | Streams (XADD/XREAD) | Hashes, Strings | ~10MB | Mature, well-known |
| **KeyDB** | Redis-compatible | Redis-compatible | ~10MB | Multi-threaded Redis fork |
| **DragonflyDB** | Redis-compatible | Redis-compatible | ~50MB | Modern, efficient |
| **NATS JetStream** | Native streaming | KV buckets | ~20MB | Lightweight, cloud-native |
| **SQLite + custom** | Custom impl | Native | 0 | No new deps, need streaming impl |

### API Design

```python
# Create/get queue - returns connection info for direct access
POST /api/v1/queues
GET /api/v1/queues/{queue_id}
Response: {
    "queue_id": "logs-{namespace}-{launch_id}",
    "backend": "redis",
    "host": "kubetorch-data-store.{ns}.svc.cluster.local",
    "port": 6379,
    "stream_name": "logs:{launch_id}",
    "filters": ["pod_name", "request_id", "log_level"]
}

# Queue operations
POST /api/v1/queues/{queue_id}/put   # Batch put (returns queue address for direct writes)
GET /api/v1/queues/{queue_id}/tail   # WebSocket tail with server-side filters
```

### Changes
- Add backing store binary to Dockerfile
- Implement queue API endpoints in server.py
- Add queue client to Python SDK
- Optional: Migrate MDS metadata to backing store

---

## Phase 4: Migrate Log Streaming to Queue (PR #4)

### Goal
Replace OTEL+Loki flow with queue-based flow through daemonset.

### Key Requirements (from user)
- App pods push logs to daemonset via:
  - API call to daemonset to tail specific files, OR
  - Separate process in app that pushes to queue API
- Daemonset pushes to namespace MDS queue
- Clients consume from queue with pre-filtering

### Flow
```
App Pod stdout/stderr
    ↓
Container runtime writes to /var/log/pods/{ns}_{pod}_{uid}/{container}/*.log
    ↓
Daemonset tails log files (via API call or file watcher)
    ↓
Daemonset pushes to queue via WebSocket to namespace MDS
    ↓
MDS stores in backing store (Redis/etc)
    ↓
Client requests queue from MDS, gets direct access address
    ↓
Client queries backing store directly with filters
```

### Changes
- Add log tailing capability to daemonset
- Update Python client to use queue API instead of Loki
- Update module.py `_stream_launch_logs()` to use queue
- Update cli_utils.py `follow_logs_in_cli()` to use queue

---

## Phase 5: Remove Loki and OTEL (PR #5)

### Goal
Clean up by removing Loki binary and OTEL daemonset.

### Changes
- Remove Loki binary from data store Dockerfile
- Remove loki.yaml config
- Remove OTEL collector templates entirely
- Update start.sh to remove Loki startup
- Clean up any remaining Loki/OTEL references

---

## Architecture Diagram (Final State)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Node                                        │
│  ┌─────────────┐     ┌─────────────────────────────────────────────┐   │
│  │  App Pod    │     │           Daemonset Data Store Pod          │   │
│  │             │     │  ┌─────────────────┐  ┌──────────────────┐  │   │
│  │ stdout/err  │────▶│  │  Log Collector  │  │  MDS Proxy       │  │   │
│  │ → log files │     │  │  (tail files)   │  │  (WebSocket to   │  │   │
│  └─────────────┘     │  └────────┬────────┘  │   namespace MDS) │  │   │
│        │             │           │           └────────┬─────────┘  │   │
│        │             │           ▼                    │            │   │
│   (MDS requests      │     Queue Put API ─────────────┘            │   │
│    via daemonset)    └─────────────────────────────────────────────┘   │
│        │                                     │                          │
└────────│─────────────────────────────────────│──────────────────────────┘
         │                          WebSocket (single per node)
         │                                     │
         │                                     ▼
         │             ┌─────────────────────────────────────────────────┐
         │             │             Namespace Data Store (MDS)          │
         │             │  ┌──────────────────┐  ┌─────────────────────┐  │
         └────────────▶│  │  Metadata Server │──│  Backing Store      │  │
                       │  │  (FastAPI)       │  │  (Redis/NATS/etc)   │  │
                       │  │  - Queue API     │  │  - Queue storage    │  │
                       │  │  - Data sync API │  │  - MDS metadata     │  │
                       │  └──────────────────┘  └─────────────────────┘  │
                       └─────────────────────────────────────────────────┘
                                               │
                                    Queue address returned
                                               ▼
         ┌─────────────────────────────────────────────────────────────┐
         │                            Client                            │
         │  - Requests queue metadata from MDS (via daemonset proxy)    │
         │  - Connects to backing store directly with filters           │
         │  - For external: WebSocket tunnel to backing store           │
         └─────────────────────────────────────────────────────────────┘
```
