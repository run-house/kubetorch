# Kubetorch Controller

This directory contains the source for the controller image deployed by the OSS Helm chart.

## Contents

- `server.py`: FastAPI entrypoint for controller routes and WebSocket endpoints
- `routes/`: Kubernetes-facing API handlers
- `background_tasks.py`, `event_watcher.py`, `ttl_controller.py`: background controllers
- `core/` and `helpers/`: shared models, database, Kubernetes helpers, and apply logic
- `tests/`: controller-focused tests
- `Dockerfile`: image build context used by `release/build_images.sh`
- `build_and_push.sh`: convenience wrapper around the shared release tooling
