#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/env.sh"
export CUDA_VISIBLE_DEVICES=""
exec "$VLA_ENV/bin/python" "$VLA_ROOT/scripts/doctor.py" --imports --data "$@"
