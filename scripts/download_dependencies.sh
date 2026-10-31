#!/usr/bin/env bash
# On an online Linux host, download, verify and package relocatable dependencies.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
INDEX=https://mirrors.aliyun.com/pypi/simple/
PYTHON_MIRROR=https://mirrors.ustc.edu.cn/github-release/astral-sh/python-build-standalone/
WORK="$ROOT/.runtime/offline-build"
OUTPUT=""
UV_BIN=""
usage() {
  cat <<'EOF'
Usage: bash scripts/download_dependencies.sh --output ARCHIVE.tar.gz [--work-dir DIR] [--index URL] [--python-mirror URL|official] [--uv-bin PATH]
Run on an internet-connected Linux x86_64 host. It downloads pinned Python dependencies,
checks a new environment in offline mode, and exports a bundle and its .sha256 file.
The work directory is retained for resuming failed downloads. Model/data are separate.
EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT="${2:?missing output path}"; shift 2 ;;
    --work-dir) WORK="${2:?missing work directory}"; shift 2 ;;
    --index) INDEX="${2:?missing index URL}"; shift 2 ;;
    --python-mirror) PYTHON_MIRROR="${2:?missing Python mirror URL}"; shift 2 ;;
    --uv-bin) UV_BIN="${2:?missing uv executable}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'Linux x86_64 required.' >&2; exit 1; }
[[ -n "$OUTPUT" && ! -e "$OUTPUT" && ! -L "$OUTPUT" && ! -e "$OUTPUT.sha256" && ! -L "$OUTPUT.sha256" ]] || {
  echo 'Provide a new --output path; the archive and its .sha256 must not already exist.' >&2; exit 2;
}
[[ "$INDEX" == https://* || "$INDEX" == http://* ]] || { echo 'Invalid package index URL.' >&2; exit 2; }
[[ "$PYTHON_MIRROR" == official || "$PYTHON_MIRROR" == https://* || "$PYTHON_MIRROR" == http://* ]] || {
  echo 'Invalid Python mirror URL.' >&2; exit 2;
}
[[ ! -L "$WORK" ]] || { echo 'Work directory must not be a symlink.' >&2; exit 1; }
mkdir -p "$WORK" "$(dirname -- "$OUTPUT")"
if [[ ! -f "$WORK/.vla-build-config" && -n "$(find "$WORK" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Existing work directory has no project marker: $WORK" >&2; exit 1
fi
runtime_sha="$(sha256sum "$ROOT/requirements/runtime-linux-py311.lock")"; runtime_sha="${runtime_sha%% *}"
simulator_sha="$(sha256sum "$ROOT/requirements/simulator-linux-py311.lock")"; simulator_sha="${simulator_sha%% *}"
config="$(printf '%s\n%s\n%s\n%s' "$INDEX" "$PYTHON_MIRROR" "$runtime_sha" "$simulator_sha")"
if [[ -f "$WORK/.vla-build-config" ]]; then
  [[ "$(cat "$WORK/.vla-build-config")" == "$config" ]] || { echo 'Existing work directory uses a different index, mirror or lockfile; choose another --work-dir.' >&2; exit 1; }
else
  printf '%s\n' "$config" > "$WORK/.vla-build-config"
fi
WORK="$(cd -- "$WORK" && pwd -P)"
unset UV_OFFLINE PIP_NO_INDEX
if [[ -z "$UV_BIN" ]]; then
  if [[ ! -x "$WORK/bootstrap/bin/uv" ]]; then
    python3 -m venv "$WORK/bootstrap"
    PIP_INDEX_URL="$INDEX" "$WORK/bootstrap/bin/python" -m pip install uv==0.13.6
  fi
  UV_BIN="$WORK/bootstrap/bin/uv"
fi
[[ -x "$UV_BIN" && "$("$UV_BIN" --version)" == 'uv 0.13.6'* ]] || { echo 'uv 0.13.6 executable required.' >&2; exit 1; }
bundle="$WORK/bundle"
mkdir -p "$bundle/tools/bin" "$bundle/tools/python" "$bundle/cache/uv"
cp -L -- "$UV_BIN" "$bundle/tools/bin/uv"
printf '%s\n' "$INDEX" > "$bundle/cache-index.txt"
export PATH="$bundle/tools/bin:$PATH" PIP_INDEX_URL="$INDEX" UV_DEFAULT_INDEX="$INDEX"
export UV_PYTHON_INSTALL_DIR="$bundle/tools/python" UV_CACHE_DIR="$bundle/cache/uv" UV_LINK_MODE=copy
unset UV_OFFLINE PIP_NO_INDEX
if [[ "$PYTHON_MIRROR" == official ]]; then unset UV_PYTHON_INSTALL_MIRROR; else export UV_PYTHON_INSTALL_MIRROR="$PYTHON_MIRROR"; fi
echo 'Downloading dependencies and installing a preparation environment.'
bash "$ROOT/setup.sh" --venv "$WORK/online-venv"
verify="$(mktemp -d "$WORK/.verify.XXXXXXXX")"
archive_tmp=""
digest_tmp=""
cleanup() {
  if [[ -n "$verify" && -d "$verify" ]]; then rm -rf -- "$verify"; fi
  if [[ -n "$archive_tmp" && -f "$archive_tmp" ]]; then rm -f -- "$archive_tmp"; fi
  if [[ -n "$digest_tmp" && -f "$digest_tmp" ]]; then rm -f -- "$digest_tmp"; fi
}
trap cleanup EXIT
echo 'Verifying the downloaded cache by installing a fresh environment offline.'
UV_OFFLINE=1 PIP_NO_INDEX=1 bash "$ROOT/setup.sh" --venv "$verify/venv"
archive_tmp="$(mktemp "${OUTPUT}.partial.XXXXXXXX")"
tar -czf "$archive_tmp" -C "$bundle" --exclude='cache/uv/.tmp*' --exclude='cache/uv/.lock' tools cache/uv cache-index.txt
digest="$(sha256sum "$archive_tmp")"; digest="${digest%% *}"
digest_tmp="$(mktemp "${OUTPUT}.sha256.partial.XXXXXXXX")"
printf '%s  %s\n' "$digest" "$(basename -- "$OUTPUT")" > "$digest_tmp"
ln -- "$archive_tmp" "$OUTPUT" || { echo 'Archive output appeared during build; not overwritten.' >&2; exit 1; }
if ! ln -- "$digest_tmp" "$OUTPUT.sha256"; then
  echo "Archive saved at $OUTPUT, but checksum output already exists; SHA-256: $digest" >&2; exit 1
fi
echo "Offline bundle: $OUTPUT"
echo "SHA-256: $digest"
echo "Checksum file: $OUTPUT.sha256"
