#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_command poetry
require_env PYPI_TOKEN

VERSION="$(read_version)"
SKIP_BUILD=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      VERSION="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=true
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "${SKIP_BUILD}" != "true" ]]; then
  "${REPO_ROOT}/release/build_python.sh" --version "${VERSION}"
fi

pushd "${REPO_ROOT}/python_client" >/dev/null
poetry config pypi-token.pypi "${PYPI_TOKEN}"
poetry publish
popd >/dev/null

echo "Published kubetorch Python package ${VERSION} to PyPI"
