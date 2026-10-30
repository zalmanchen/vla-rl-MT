#!/usr/bin/env bash
set -euo pipefail
VLA_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$VLA_REPO_DIR/scripts/setup_env.sh" "$@"
