#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_command docker

VERSION="$(read_version)"
PUSH=false
COMPONENTS=()
IMAGE_NAMESPACE="${IMAGE_NAMESPACE:-ghcr.io/run-house}"
PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      VERSION="$2"
      shift 2
      ;;
    --push)
      PUSH=true
      shift
      ;;
    --component)
      COMPONENTS+=("$2")
      shift 2
      ;;
    --all)
      COMPONENTS=(kubetorch-controller kubetorch-data-store kubetorch server ubuntu server-otel ubuntu-otel)
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ${#COMPONENTS[@]} -eq 0 ]]; then
  COMPONENTS=(kubetorch-controller kubetorch-data-store)
fi

build_image() {
  local component="$1"
  local image_ref dockerfile context
  local -a build_args=()

  case "${component}" in
    kubetorch-controller)
      image_ref="${IMAGE_NAMESPACE}/kubetorch-controller:${VERSION}"
      dockerfile="${REPO_ROOT}/services/kubetorch_controller/Dockerfile"
      context="${REPO_ROOT}/services/kubetorch_controller"
      build_args+=(--build-arg "VERSION=${VERSION}")
      ;;
    kubetorch-data-store)
      image_ref="${IMAGE_NAMESPACE}/kubetorch-data-store:${VERSION}"
      dockerfile="${REPO_ROOT}/services/data_store/Dockerfile"
      context="${REPO_ROOT}/services/data_store"
      ;;
    kubetorch)
      image_ref="${IMAGE_NAMESPACE}/kubetorch:${VERSION}"
      dockerfile="${REPO_ROOT}/release/default_images/kubetorch"
      context="${REPO_ROOT}"
      ;;
    server)
      image_ref="${IMAGE_NAMESPACE}/server:${VERSION}"
      dockerfile="${REPO_ROOT}/release/default_images/server"
      context="${REPO_ROOT}"
      ;;
    ubuntu)
      image_ref="${IMAGE_NAMESPACE}/ubuntu:${VERSION}"
      dockerfile="${REPO_ROOT}/release/default_images/ubuntu"
      context="${REPO_ROOT}"
      ;;
    server-otel)
      image_ref="${IMAGE_NAMESPACE}/server-otel:${VERSION}"
      dockerfile="${REPO_ROOT}/release/default_images/server-otel"
      context="${REPO_ROOT}"
      build_args+=(--build-arg "BASE_IMAGE=${IMAGE_NAMESPACE}/server:${VERSION}")
      ;;
    ubuntu-otel)
      image_ref="${IMAGE_NAMESPACE}/ubuntu-otel:${VERSION}"
      dockerfile="${REPO_ROOT}/release/default_images/ubuntu-otel"
      context="${REPO_ROOT}"
      build_args+=(--build-arg "BASE_IMAGE=${IMAGE_NAMESPACE}/ubuntu:${VERSION}")
      ;;
    *)
      echo "Unknown component: ${component}" >&2
      exit 1
      ;;
  esac

  echo "Building ${image_ref}"
  docker build \
    --platform "${PLATFORM}" \
    "${build_args[@]}" \
    -t "${image_ref}" \
    -f "${dockerfile}" \
    "${context}"

  if [[ "${PUSH}" == "true" ]]; then
    echo "Pushing ${image_ref}"
    docker push "${image_ref}"
  fi
}

for component in "${COMPONENTS[@]}"; do
  build_image "${component}"
done
