# Release

This repo now carries the customer-facing Kubetorch release surface in one place:

- `python_client/` for the Python package
- `charts/kubetorch/` for the Helm chart
- `services/kubetorch_controller/` for the controller image
- `services/data_store/` for the data store image
- `release/default_images/` for the workload base images

`/VERSION` is the single source of truth for the Kubetorch release version. Run `release/sync_version.py <version>` before publishing artifacts if you want to bump it.

## Common commands

Sync repo version metadata:

```bash
python3 release/sync_version.py 0.5.0
```

Build the Python package:

```bash
release/build_python.sh
```

Package the Helm chart:

```bash
release/package_chart.sh
```

Build controller and data store images:

```bash
release/build_images.sh
```

Build all runtime images too:

```bash
release/build_images.sh --all
```

Build and push all release artifacts:

```bash
release/release_all.sh --version 0.5.0 --push-images
```

## Image publishing

By default, images are tagged under `ghcr.io/run-house`. Override that by exporting `IMAGE_NAMESPACE`, for example:

```bash
IMAGE_NAMESPACE=ghcr.io/my-org release/build_images.sh --all --push
```

The OTEL runtime images expect the matching `server` or `ubuntu` base image tag to exist locally or in the target registry.
