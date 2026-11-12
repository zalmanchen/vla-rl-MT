#!/usr/bin/env bash
set -euo pipefail
cd /cloud/vla-rl-MT
source bootstrap/env_recap_official.sh
mkdir -p .runtime/bootstrap .runtime/runs
if [[ ! -f .runtime/recap-official/configs/provenance.json ]]; then
  python bootstrap/prepare_recap_official.py configs
fi
exec "$@"
