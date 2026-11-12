#!/usr/bin/env bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT
export ROBOTWIN_PATH="$PROJECT_ROOT/third_party/RoboTwin"
export REPO_PATH="$PROJECT_ROOT/third_party/RLinf"
export EMBODIED_PATH="$REPO_PATH/examples/embodiment"
export PYTHONPATH="$PROJECT_ROOT/src:$REPO_PATH:$ROBOTWIN_PATH${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PROJECT_ROOT/.runtime/cache/huggingface"
export HF_LEROBOT_HOME="$PROJECT_ROOT/datasets/lerobot"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export OPENPI_DATA_HOME="$PROJECT_ROOT/.runtime/cache/openpi"
if [[ -z "${CUDA_HOME:-}" ]]; then
  if [[ -d /usr/local/cuda-12.8 ]]; then
    export CUDA_HOME=/usr/local/cuda-12.8
  else
    export CUDA_HOME=/usr/local/cuda
  fi
fi
export PATH="$CUDA_HOME/bin:$PATH"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-$PROJECT_ROOT/.runtime/xdg}"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
export ASSETS_PATH="$ROBOTWIN_PATH"
export ROBOT_PLATFORM=ALOHA
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
source "$PROJECT_ROOT/.runtime/venv-rlinf/bin/activate"
