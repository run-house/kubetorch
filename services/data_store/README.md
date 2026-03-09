# Kubetorch Data Store

This directory contains the source for the data store image deployed by the OSS Helm chart.

The container bundles four pieces that Kubetorch clients depend on:

- rsync for bulk file transfer
- the metadata API in `server.py` for key-to-source discovery
- the WebSocket tunnel in `websocket_tunnel_server.py`
- optional embedded Loki for ephemeral log storage

## Key files

- `Dockerfile`: image build context used by `release/build_images.sh`
- `start.sh`: entrypoint that boots rsync, the metadata API, the WebSocket tunnel, and Loki when enabled
- `server.py`, `models.py`, `locks.py`: metadata service implementation
- `loki.yaml`: embedded Loki configuration
- `build_and_push.sh`: convenience wrapper around the shared release tooling
