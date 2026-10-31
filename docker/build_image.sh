#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE=""
TAG=vla-robotwin:offline
BASE=ubuntu:22.04
MIRROR=http://mirrors.aliyun.com/ubuntu/
INSTALL_APT=1
OUTPUT=""
usage() {
  cat <<'EOF'
Usage: bash docker/build_image.sh --bundle ARCHIVE [--tag IMAGE:TAG] [--base IMAGE] [--apt-mirror URL] [--skip-apt] [--output IMAGE.tar]
Build uses the offline Python bundle. The base image and Ubuntu apt packages must be available separately.
--skip-apt is for a preloaded Ubuntu 22.04 base that already contains all documented system libraries.
--output saves the finished image for transfer; existing output files are not overwritten.
EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle) BUNDLE="${2:?missing bundle path}"; shift 2 ;;
    --tag) TAG="${2:?missing image tag}"; shift 2 ;;
    --base) BASE="${2:?missing base image}"; shift 2 ;;
    --apt-mirror) MIRROR="${2:?missing mirror URL}"; shift 2 ;;
    --skip-apt) INSTALL_APT=0; shift ;;
    --output) OUTPUT="${2:?missing output path}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -f "$BUNDLE" ]] || { echo 'Provide the offline archive with --bundle.' >&2; exit 2; }
if [[ -n "$OUTPUT" && ( -e "$OUTPUT" || -L "$OUTPUT" ) ]]; then
  echo "Image output already exists and will not be overwritten: $OUTPUT" >&2; exit 1
fi
command -v docker >/dev/null || { echo 'Docker CLI is required.' >&2; exit 1; }
docker info >/dev/null || { echo 'Docker daemon unavailable or permission denied.' >&2; exit 1; }
mkdir -p "$ROOT/.runtime/docker-build"
context="$(mktemp -d "$ROOT/.runtime/docker-build/context.XXXXXXXX")"
image_tmp=""
cleanup() {
  if [[ -n "$image_tmp" && -f "$image_tmp" ]]; then rm -f -- "$image_tmp"; fi
  if [[ -n "$context" && -d "$context" ]]; then rm -rf -- "$context"; fi
}
trap cleanup EXIT
mkdir "$context/project"
tar -C "$ROOT" --exclude='__pycache__' --exclude='*.pyc' -cf - \
  README.md THIRD_PARTY.md setup.sh run.sh train.sh train_bc.py check.sh download.sh evaluate.sh \
  configs docs requirements resources scripts vla_mtrl docker | tar -C "$context/project" -xf -
ln -- "$BUNDLE" "$context/offline-bundle.tar.gz" 2>/dev/null || cp -- "$BUNDLE" "$context/offline-bundle.tar.gz"
docker build --platform linux/amd64 --build-arg "BASE_IMAGE=$BASE" \
  --build-arg "APT_MIRROR=$MIRROR" --build-arg "INSTALL_APT=$INSTALL_APT" \
  -f "$context/project/docker/Dockerfile" -t "$TAG" "$context"
if [[ -n "$OUTPUT" ]]; then
  mkdir -p -- "$(dirname -- "$OUTPUT")"
  image_tmp="$(mktemp "${OUTPUT}.partial.XXXXXXXX")"
  docker image save --output "$image_tmp" "$TAG"
  if ! ln -- "$image_tmp" "$OUTPUT"; then echo 'Could not create output without overwriting an existing file.' >&2; exit 1; fi
  rm -f -- "$image_tmp"
  image_tmp=""
  echo "Image saved to $OUTPUT"
fi
