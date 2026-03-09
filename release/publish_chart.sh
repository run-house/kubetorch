#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_command helm
require_env GHCR_TOKEN

VERSION="$(read_version)"
SKIP_PACKAGE=false
GHCR_USERNAME="${GHCR_USERNAME:-run-house}"
CHART_REGISTRY="${CHART_REGISTRY:-oci://ghcr.io/run-house/charts}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      VERSION="$2"
      shift 2
      ;;
    --skip-package)
      SKIP_PACKAGE=true
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "${SKIP_PACKAGE}" != "true" ]]; then
  "${REPO_ROOT}/release/package_chart.sh" --version "${VERSION}"
fi

CHART_TGZ="${REPO_ROOT}/dist/charts/kubetorch-${VERSION}.tgz"
if [[ ! -f "${CHART_TGZ}" ]]; then
  echo "Missing chart package: ${CHART_TGZ}" >&2
  exit 1
fi

echo "${GHCR_TOKEN}" | helm registry login ghcr.io --username "${GHCR_USERNAME}" --password-stdin
helm push "${CHART_TGZ}" "${CHART_REGISTRY}"

echo "Published kubetorch chart ${VERSION} to ${CHART_REGISTRY}"
