#!/usr/bin/env bash
VLA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VLA_ENV="${VLA_ENV:-$VLA_ROOT/.runtime/envs/robotwin}"
export VLA_ROOT VLA_ENV
export VIRTUAL_ENV="$VLA_ENV"
export PATH="$VLA_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$VLA_ROOT:$VLA_ROOT/RLinf:$VLA_ROOT/RoboTwin${PYTHONPATH:+:$PYTHONPATH}"
export REPO_PATH="$VLA_ROOT/RLinf"
export EMBODIED_PATH="$VLA_ROOT/RLinf/examples/sft"
export ROBOTWIN_PATH="$VLA_ROOT/RoboTwin"
export ROBOT_PLATFORM=ALOHA
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$VLA_ROOT/.runtime/cache/openpi}"
export HF_HOME="${HF_HOME:-$VLA_ROOT/.runtime/cache/huggingface}"
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
