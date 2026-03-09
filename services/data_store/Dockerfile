FROM python:3.11-slim

# Prevent Python from writing bytecode files
ENV PYTHONDONTWRITEBYTECODE=1

# Install rsync and other system utilities (Python/pip already included in base image)
RUN apt-get update && \
    apt-get install -y rsync net-tools curl unzip && \
    rm -rf /var/lib/apt/lists/*

# Download and install Loki binary for ephemeral log storage
RUN curl -L https://github.com/grafana/loki/releases/download/v3.5.3/loki-linux-amd64.zip -o /tmp/loki.zip && \
    unzip /tmp/loki.zip -d /usr/local/bin/ && \
    chmod +x /usr/local/bin/loki-linux-amd64 && \
    mv /usr/local/bin/loki-linux-amd64 /usr/local/bin/loki && \
    rm /tmp/loki.zip

# Install Python dependencies
RUN pip install --no-cache-dir websockets fastapi uvicorn pydantic

# Create directories and pre-create log file
RUN mkdir -p /data /var/log /tmp/loki /etc/loki && \
    touch /var/log/rsyncd.log

# Copy the WebSocket proxy script (same as old implementation)
COPY websocket_tunnel_server.py /usr/local/bin/server_proxy.py

# Copy the metadata server
COPY server.py /app/server.py
COPY models.py /app/models.py
COPY locks.py /app/locks.py

# Copy Loki configuration
COPY loki.yaml /etc/loki/loki.yaml

# Copy the entrypoint script
COPY start.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose ports for rsync daemon, WebSocket proxy, metadata server, and Loki
EXPOSE 873 8080 8081 3100

# Use the entrypoint script to start all services
CMD ["/entrypoint.sh"]
