#!/usr/bin/env bash
# Install from the verified offline bundle, or opt in to an online mirror.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECTED_SHA256=f0d3989656be8c507096c04a30b8b5ce4ffad2f9371787edac24fe76a9453054
BUNDLE=""
BUNDLE_DIR="$ROOT/.runtime/offline-bundle"
MODE=offline
INDEX=https://mirrors.aliyun.com/pypi/simple/
ARGS=()
usage() {
  cat <<'EOF'
Usage: bash scripts/install_local.sh --bundle /path/to/robotwin-offline-linux-x86_64-py31118.tar.gz [--bundle-dir DIR] [--venv DIR] [--train-only]
       bash scripts/install_local.sh --bundle /path/to/generated.tar.gz --sha256 SHA256 [--bundle-dir DIR] [--venv DIR]
       bash scripts/install_local.sh --online [--index URL] [--venv DIR] [--train-only]
Offline mode verifies the known bundle SHA-256 and never downloads dependencies.
Online mode uses the given Python package index; Python itself may use another host.
EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle) BUNDLE="${2:?missing bundle path}"; shift 2 ;;
    --sha256) EXPECTED_SHA256="${2:?missing SHA-256}"; shift 2 ;;
    --bundle-dir) BUNDLE_DIR="${2:?missing bundle directory}"; shift 2 ;;
    --online) MODE=online; shift ;;
    --index) INDEX="${2:?missing index URL}"; shift 2 ;;
    --venv) ARGS+=(--venv "${2:?missing environment path}"); shift 2 ;;
    --train-only) ARGS+=(--train-only); shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'Linux x86_64 required.' >&2; exit 1; }
if [[ "$MODE" == online ]]; then
  [[ -z "$BUNDLE" ]] || { echo 'Do not combine --online and --bundle.' >&2; exit 2; }
  unset UV_OFFLINE PIP_NO_INDEX
  export PIP_INDEX_URL="$INDEX" UV_DEFAULT_INDEX="$INDEX"
  exec bash "$ROOT/setup.sh" "${ARGS[@]}"
fi
[[ -f "$BUNDLE" ]] || { echo 'Provide the offline archive with --bundle; use --help for details.' >&2; exit 2; }
[[ "$EXPECTED_SHA256" =~ ^[a-fA-F0-9]{64}$ ]] || { echo 'Invalid SHA-256.' >&2; exit 2; }
EXPECTED_SHA256="${EXPECTED_SHA256,,}"
command -v sha256sum >/dev/null
command -v tar >/dev/null
actual="$(sha256sum -- "$BUNDLE")"; actual="${actual%% *}"
[[ "$actual" == "$EXPECTED_SHA256" ]] || { echo 'Offline archive SHA-256 mismatch; existing files were not modified.' >&2; exit 1; }
mkdir -p -- "$(dirname -- "$BUNDLE_DIR")"
if [[ -e "$BUNDLE_DIR" || -L "$BUNDLE_DIR" ]]; then
  [[ -d "$BUNDLE_DIR" && ! -L "$BUNDLE_DIR" && -f "$BUNDLE_DIR/.vla-offline-sha256" ]] || {
    echo "Bundle directory already exists without the matching marker: $BUNDLE_DIR" >&2; exit 1;
  }
  [[ "$(cat -- "$BUNDLE_DIR/.vla-offline-sha256")" == "$EXPECTED_SHA256" ]] || {
    echo 'Existing offline bundle has a different marker; it was not overwritten.' >&2; exit 1;
  }
else
  staging="$(mktemp -d "${BUNDLE_DIR}.tmp.XXXXXXXX")"
  trap 'if [[ -n "${staging:-}" && -d "$staging" ]]; then rm -rf -- "$staging"; fi' EXIT
  echo 'Extracting verified offline dependencies; this may take several minutes.'
  tar -xzf "$BUNDLE" -C "$staging"
  [[ -x "$staging/tools/bin/uv" && -d "$staging/tools/python" && -d "$staging/cache/uv" ]] || {
    echo 'Offline archive has an unexpected layout.' >&2; exit 1;
  }
  printf '%s\n' "$EXPECTED_SHA256" > "$staging/.vla-offline-sha256"
  mv -T -- "$staging" "$BUNDLE_DIR"
  staging=""
  trap - EXIT
fi
export VLA_OFFLINE_ROOT="$(cd -- "$BUNDLE_DIR" && pwd -P)"
export PATH="$VLA_OFFLINE_ROOT/tools/bin:$PATH"
export UV_PYTHON_INSTALL_DIR="$VLA_OFFLINE_ROOT/tools/python"
export UV_CACHE_DIR="$VLA_OFFLINE_ROOT/cache/uv"
if [[ -f "$VLA_OFFLINE_ROOT/cache-index.txt" ]]; then
  IFS= read -r INDEX < "$VLA_OFFLINE_ROOT/cache-index.txt" || true
  [[ "$INDEX" == https://* || "$INDEX" == http://* ]] || { echo 'Invalid cached index URL.' >&2; exit 1; }
fi
export UV_DEFAULT_INDEX="$INDEX"
export UV_OFFLINE=1 PIP_NO_INDEX=1
exec bash "$ROOT/setup.sh" "${ARGS[@]}"
