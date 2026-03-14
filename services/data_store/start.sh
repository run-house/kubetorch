#!/bin/bash
# entrypoint.sh

set -e  # Exit on error (but we'll handle background processes)

# Conditionally start Loki for ephemeral log storage
LOKI_PID=""
if [ "${LOKI_ENABLED:-true}" = "true" ]; then
    echo "Starting Loki..."
    /usr/local/bin/loki -config.file=/etc/loki/loki.yaml > /var/log/loki.log 2>&1 &
    LOKI_PID=$!
    echo "Loki started (PID: $LOKI_PID)"
else
    echo "Loki disabled (LOKI_ENABLED=${LOKI_ENABLED})"
fi

# Start rsync daemon in background
echo "Starting rsync daemon..."
rsync --daemon --no-detach --config=/etc/rsyncd.conf &
RSYNC_PID=$!
echo "Rsync daemon started (PID: $RSYNC_PID)"

# Start WebSocket proxy in background
echo "Starting WebSocket proxy..."
python /usr/local/bin/server_proxy.py > /var/log/websocket_proxy.log 2>&1 &
PROXY_PID=$!
echo "WebSocket proxy started (PID: $PROXY_PID)"
sleep 2  # Give it a moment to start
if ! kill -0 $PROXY_PID 2>/dev/null; then
    echo "ERROR: WebSocket proxy failed to start!"
    cat /var/log/websocket_proxy.log 2>/dev/null || echo "No log file found"
    exit 1
fi
echo "WebSocket proxy is running"

# Start metadata server in background (using uvicorn for FastAPI)
# Configure for high connection counts: increased backlog for socket accept queue
echo "Starting metadata server..."
python -m uvicorn server:app --host 0.0.0.0 --port ${METADATA_SERVER_PORT:-8081} --backlog ${UVICORN_BACKLOG:-8192} --app-dir /app &
METADATA_PID=$!
echo "Metadata server started (PID: $METADATA_PID)"

# Function to cleanup on exit
cleanup() {
    echo "Shutting down services..."
    [ -n "$LOKI_PID" ] && kill $LOKI_PID 2>/dev/null || true
    kill $RSYNC_PID $PROXY_PID $METADATA_PID 2>/dev/null || true
    wait 2>/dev/null || true
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Wait for any process to exit
wait -n
exit_code=$?

# If one process exits, kill the others
cleanup

exit $exit_code
