# Building the kubetorch-data-store Image

The canonical path for image builds is now the shared release tooling in `release/`.

## Build

```bash
release/build_images.sh --component kubetorch-data-store
```

## Build and push

```bash
release/build_images.sh --component kubetorch-data-store --push
```

## Override the target registry

```bash
IMAGE_NAMESPACE=ghcr.io/my-org release/build_images.sh --component kubetorch-data-store --push
```
