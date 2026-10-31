#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
VLA_DRY_RUN=0
VLA_TRAIN_ONLY=0
VLA_SOURCES_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv) VLA_ENV="${2:?}"; export VLA_ENV VIRTUAL_ENV="$VLA_ENV"; shift 2 ;;
    --train-only) VLA_TRAIN_ONLY=1; shift ;;
    --verify-sources) VLA_SOURCES_ONLY=1; shift ;;
    --dry-run) VLA_DRY_RUN=1; shift ;;
    -h|--help) echo "Usage: ./setup.sh [--venv PATH] [--train-only] [--verify-sources] [--dry-run]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
if [[ "$VLA_DRY_RUN" == 1 ]]; then
  printf 'Repository: %s\nEnvironment: %s\n' "$VLA_ROOT" "$VLA_ENV"
  echo "Plan: pinned RLinf and RoboTwin submodules; Python 3.11.18; CUDA 12.6 PyTorch/OpenPI"
  echo "Plan: default includes SAPIEN/mplib simulator; --train-only skips simulator packages"
  echo "No models, demonstrations or GPU jobs are started."
  exit 0
fi
if [[ "$VLA_SOURCES_ONLY" == 1 ]]; then
  exec python3 "$VLA_ROOT/vla_mtrl/source_identity.py" --root "$VLA_ROOT"
fi
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo "Linux x86_64 required" >&2; exit 1; }
command -v curl >/dev/null
if [[ ! -f "$VLA_ROOT/RLinf/rlinf/__init__.py" || ! -f "$VLA_ROOT/RoboTwin/envs/__init__.py" ]]; then
  if command -v git >/dev/null && [[ "$(git -C "$VLA_ROOT" rev-parse --show-toplevel 2>/dev/null || true)" == "$VLA_ROOT" ]]; then
    git -C "$VLA_ROOT" submodule update --init RLinf RoboTwin
  fi
fi
python3 "$VLA_ROOT/vla_mtrl/source_identity.py" --root "$VLA_ROOT"
if ! command -v uv >/dev/null; then
  if [[ ! -x "$VLA_ROOT/.runtime/tools/bootstrap/bin/uv" ]]; then
    python3 -m venv "$VLA_ROOT/.runtime/tools/bootstrap"
    "$VLA_ROOT/.runtime/tools/bootstrap/bin/python" -m pip install uv==0.13.6
  fi
  export PATH="$VLA_ROOT/.runtime/tools/bootstrap/bin:$PATH"
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$VLA_ROOT/.runtime/cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$VLA_ROOT/.runtime/tools/python}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export CUDA_VISIBLE_DEVICES=""
if [[ -e "$VLA_ENV" ]]; then
  "$VLA_ENV/bin/python" -c 'import sys; assert sys.prefix != sys.base_prefix and sys.version_info[:3] == (3,11,18), sys.version'
else
  uv python install --no-bin 3.11.18
  uv venv --python 3.11.18 "$VLA_ENV"
fi
uv pip install --python "$VLA_ENV/bin/python" --no-deps -r "$VLA_ROOT/requirements/runtime-linux-py311.lock"
uv pip install --python "$VLA_ENV/bin/python" --no-deps --reinstall-package rlinf-transformer-openpi rlinf-transformer-openpi==4.53.2
if [[ "$VLA_TRAIN_ONLY" == 0 ]]; then
  uv pip install --python "$VLA_ENV/bin/python" --no-deps -r "$VLA_ROOT/requirements/simulator-linux-py311.lock"
  "$VLA_ENV/bin/python" "$VLA_ROOT/scripts/patch_simulator.py"
fi
uv pip check --python "$VLA_ENV/bin/python"
VLA_DOCTOR_ARGS=(--imports)
if [[ "$VLA_TRAIN_ONLY" == 0 ]]; then VLA_DOCTOR_ARGS+=(--require-simulator); fi
"$VLA_ENV/bin/python" "$VLA_ROOT/scripts/doctor.py" "${VLA_DOCTOR_ARGS[@]}"
echo "Environment ready. Next: ./download.sh"
