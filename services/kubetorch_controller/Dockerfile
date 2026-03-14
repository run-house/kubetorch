FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies from file (cached unless requirements.txt changes)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Bust cache for code changes only (requirements stay cached)
ARG CACHEBUST=1

# Copy controller code
COPY . /app/

# Version is set at build time and matches the image tag
ARG VERSION=controller-routes
ENV VERSION=${VERSION}

# Unbuffered Python output for better logging
ENV PYTHONUNBUFFERED=1

# Port configuration (can be overridden at runtime via helm values)
ARG PORT=8081
EXPOSE ${PORT}

# Default configuration (can be overridden via helm values)
ENV PORT=${PORT}
ENV CONNECTION_POOL_SIZE=20
ENV BACKLOG=8192
ENV TIMEOUT_KEEP_ALIVE=75

# Run server with configurable workers and connection limits
# All settings can be overridden via environment variables from helm chart
# We only support one worker process, because the processes are not sharing information
# about the websockets each are connecting. To support multiple workers, we'd need to store this
# information in a shared place, like redis.
# Note: We don't set --limit-concurrency to allow maximum concurrent connections.
# The backlog controls the kernel's socket accept queue size.
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT} --workers 1 --backlog ${BACKLOG} --timeout-keep-alive ${TIMEOUT_KEEP_ALIVE}"]
