#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_command docker
require_env GHCR_TOKEN
require_env PYPI_TOKEN

VERSION="$(read_version)"
GHCR_USERNAME="${GHCR_USERNAME:-run-house}"
IMAGE_NAMESPACE="${IMAGE_NAMESPACE:-ghcr.io/run-house}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      VERSION="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

echo "${GHCR_TOKEN}" | docker login ghcr.io -u "${GHCR_USERNAME}" --password-stdin

IMAGE_NAMESPACE="${IMAGE_NAMESPACE}" "${REPO_ROOT}/release/release_all.sh" --version "${VERSION}" --push-images
"${REPO_ROOT}/release/publish_chart.sh" --version "${VERSION}" --skip-package
"${REPO_ROOT}/release/publish_python.sh" --version "${VERSION}" --skip-build

echo "Published kubetorch ${VERSION} artifacts to GHCR and PyPI"
