#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/env.sh"
VLA_PYTHON="$VLA_ENV/bin/python"
for VLA_ARG in "$@"; do
  case "$VLA_ARG" in --dry-run|--help|-h) VLA_PYTHON=python3 ;; esac
done
[[ "$VLA_PYTHON" == python3 || -x "$VLA_PYTHON" ]] || { echo "Run ./setup.sh first" >&2; exit 1; }
exec "$VLA_PYTHON" "$VLA_ROOT/scripts/download_assets.py" "$@"
