#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_command helm
require_command python3

VERSION="$(read_version)"
if [[ $# -ge 2 && "$1" == "--version" ]]; then
  VERSION="$2"
fi

python3 "${REPO_ROOT}/release/sync_version.py" "${VERSION}" >/dev/null

OUTPUT_DIR="${REPO_ROOT}/dist/charts"
mkdir -p "${OUTPUT_DIR}"

pushd "${REPO_ROOT}/charts/kubetorch" >/dev/null
helm dependency update .
helm package . --destination "${OUTPUT_DIR}"
popd >/dev/null

echo "Packaged kubetorch chart ${VERSION} into ${OUTPUT_DIR}"
