#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [version]" >&2
  exit 1
fi

if [[ $# -eq 1 ]]; then
  "${REPO_ROOT}/release/build_images.sh" --version "$1" --component kubetorch-data-store --push
else
  "${REPO_ROOT}/release/build_images.sh" --component kubetorch-data-store --push
fi
