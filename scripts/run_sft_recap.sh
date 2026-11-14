#!/usr/bin/env bash
# Thin launcher: all training and labeling execute pinned official RLinf code.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT
VLA_WORKFLOW_PYTHON="${VLA_WORKFLOW_PYTHON:-$PROJECT_ROOT/.runtime/venv-rlinf/bin/python}"
if [[ ! -x "$VLA_WORKFLOW_PYTHON" ]]; then
  echo "Missing environment: $VLA_WORKFLOW_PYTHON; run bootstrap/install.sh first." >&2
  exit 2
fi
source "$PROJECT_ROOT/bootstrap/env.sh"
exec "$VLA_WORKFLOW_PYTHON" "$PROJECT_ROOT/scripts/sft_recap.py" "$@"
