#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

VERSION="$(read_version)"
PUSH_IMAGES=false
BUILD_IMAGES=true
BUILD_CHART=true
BUILD_PYTHON=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      VERSION="$2"
      shift 2
      ;;
    --push-images)
      PUSH_IMAGES=true
      shift
      ;;
    --skip-images)
      BUILD_IMAGES=false
      shift
      ;;
    --skip-chart)
      BUILD_CHART=false
      shift
      ;;
    --skip-python)
      BUILD_PYTHON=false
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

python3 "${REPO_ROOT}/release/sync_version.py" "${VERSION}" >/dev/null

if [[ "${BUILD_PYTHON}" == "true" ]]; then
  "${REPO_ROOT}/release/build_python.sh" --version "${VERSION}"
fi

if [[ "${BUILD_CHART}" == "true" ]]; then
  "${REPO_ROOT}/release/package_chart.sh" --version "${VERSION}"
fi

if [[ "${BUILD_IMAGES}" == "true" ]]; then
  if [[ "${PUSH_IMAGES}" == "true" ]]; then
    "${REPO_ROOT}/release/build_images.sh" --version "${VERSION}" --all --push
  else
    "${REPO_ROOT}/release/build_images.sh" --version "${VERSION}" --all
  fi
fi
