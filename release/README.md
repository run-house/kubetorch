# Release

This repo now carries the customer-facing Kubetorch release surface in one place:

- `python_client/` for the Python package
- `charts/kubetorch/` for the Helm chart
- `services/kubetorch_controller/` for the controller image
- `services/data_store/` for the data store image
- `release/default_images/` for the workload base images

`/VERSION` is the single source of truth for the Kubetorch release version. Run `release/sync_version.py <version>` before publishing artifacts if you want to bump it.

The Python client now uses that same version for its default remote job images:

- `ghcr.io/run-house/server:<kubetorch version>`
- `ghcr.io/run-house/server-otel:<kubetorch version>`
- `ghcr.io/run-house/ubuntu:<kubetorch version>`
- `ghcr.io/run-house/ubuntu-otel:<kubetorch version>`

## Common commands

Sync repo version metadata:

```bash
python3 release/sync_version.py <version>
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

Run the local build flow in one command:

```bash
release/release_all.sh --version <version> --push-images
```

## What `release_all.sh` does

`release/release_all.sh` is a convenience wrapper around the commands above. It:

- syncs `VERSION`
- builds the Python package locally
- packages the Helm chart locally into `dist/charts/`
- builds all images
- pushes images only if you pass `--push-images`

If you already ran the individual steps manually, you do not need to run `release/release_all.sh` again unless you want it to rerun them.

It does not publish the Python package to PyPI or push the Helm chart to GHCR.

## Publish commands

Publish just the Python package to PyPI:

```bash
PYPI_TOKEN=... release/publish_python.sh --version <version>
```

Publish just the Helm chart to GHCR:

```bash
GHCR_TOKEN=... GHCR_USERNAME=run-house release/publish_chart.sh --version <version>
```

Build everything, push all images to GHCR, push the chart to GHCR, and publish the Python package to PyPI:

```bash
GHCR_TOKEN=... GHCR_USERNAME=run-house PYPI_TOKEN=... release/publish_all.sh --version <version>
```

If you already ran the build steps manually, use:

```bash
GHCR_TOKEN=... GHCR_USERNAME=run-house release/publish_chart.sh --version <version> --skip-package
PYPI_TOKEN=... release/publish_python.sh --version <version> --skip-build
```

## Image publishing

By default, images are tagged under `ghcr.io/run-house`. Override that by exporting `IMAGE_NAMESPACE`, for example:

```bash
IMAGE_NAMESPACE=ghcr.io/my-org release/build_images.sh --all --push
```

The OTEL runtime images expect the matching `server` or `ubuntu` base image tag to exist locally or in the target registry.
