#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_command python3
require_command poetry

VERSION="$(read_version)"
if [[ $# -ge 2 && "$1" == "--version" ]]; then
  VERSION="$2"
fi

python3 "${REPO_ROOT}/release/sync_version.py" "${VERSION}" >/dev/null

pushd "${REPO_ROOT}/python_client" >/dev/null
poetry check
poetry build
popd >/dev/null

echo "Built kubetorch Python artifacts for ${VERSION}"
