#!/usr/bin/env bash
# Independent, unpatched LIBERO RECAP runtime. Source from the project checkout.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT
export REPO_PATH="$PROJECT_ROOT/third_party/RLinf-recap-official"
export EMBODIED_PATH="$REPO_PATH/examples/embodiment"
export PYTHONPATH="$REPO_PATH"
export HF_HOME="$PROJECT_ROOT/.runtime/cache/huggingface"
export HF_LEROBOT_HOME="$PROJECT_ROOT/datasets/recap-official"
export HF_HUB_DISABLE_XET=1
export OPENPI_DATA_HOME="$PROJECT_ROOT/.runtime/cache/openpi"
export LIBERO_CONFIG_PATH="$PROJECT_ROOT/.runtime/recap-official/libero-config"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
mkdir -p "$LIBERO_CONFIG_PATH"
unset RLINF_EXT_MODULE ROBOTWIN_PATH ROBOT_PLATFORM ASSETS_PATH
source "$PROJECT_ROOT/.runtime/venv-recap-official/bin/activate"
