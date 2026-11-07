#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export UV_CACHE_DIR="$PWD/.runtime/cache/uv"
python bootstrap/prepare_sources.py
if [[ ! -x .runtime/venv-rlinf/bin/python ]]; then
  uv venv --python 3.11.15 .runtime/venv-rlinf
fi
uv pip sync --python .runtime/venv-rlinf/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match bootstrap/environment.freeze.txt
uv pip check --python .runtime/venv-rlinf/bin/python
mkdir -p .runtime/cache/openpi/big_vision
if [[ ! -s .runtime/cache/openpi/big_vision/paligemma_tokenizer.model ]]; then
  curl -fL --retry 3 https://storage.googleapis.com/big_vision/paligemma_tokenizer.model \
    -o .runtime/cache/openpi/big_vision/paligemma_tokenizer.model
fi
