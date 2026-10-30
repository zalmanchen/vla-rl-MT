"""Portable paths and immutable experiment identities."""

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TASKS = ["adjust_bottle", "handover_block"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def environment():
    os.environ.update(
        {
            "VLA_ROOT": str(ROOT),
            "REPO_PATH": str(ROOT / "RLinf"),
            "EMBODIED_PATH": str(ROOT / "RLinf/examples/sft"),
            "ROBOTWIN_PATH": str(ROOT / "RoboTwin"),
            "ROBOT_PLATFORM": "ALOHA",
            "JAX_PLATFORMS": "cpu",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "RLINF_EXT_MODULE": "vla_mtrl",
        }
    )
    os.environ.setdefault("OPENPI_DATA_HOME", str(ROOT / ".runtime/cache/openpi"))
    os.environ.setdefault("HF_HOME", str(ROOT / ".runtime/cache/huggingface"))
    sys.path[:0] = [str(ROOT), str(ROOT / "RLinf"), str(ROOT / "RoboTwin")]


def compose(
    steps=1000, tasks=None, seed=0, samples_per_task=8, micro_batch=None, run_dir=None
):
    environment()
    from hydra import compose as hydra_compose, initialize_config_dir
    from omegaconf import OmegaConf

    tasks = tasks or TASKS
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.1"):
        cfg = hydra_compose(config_name="robotwin_bc")
    cfg.runner.max_steps = steps
    cfg.actor.optim.total_training_steps = steps
    cfg.actor.optim.lr_warmup_steps = min(100, max(1, steps // 10))
    cfg.actor.seed = seed
    cfg.actor.micro_batch_size = micro_batch or len(tasks)
    cfg.actor.global_batch_size = samples_per_task * len(tasks)
    cfg.data.tasks = tasks
    cfg.runner.logger.log_path = str(run_dir or ROOT / ".runtime/runs/manual")
    OmegaConf.resolve(cfg)
    return cfg
