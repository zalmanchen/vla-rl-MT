"""Exercise the pinned RLinf LIBERO wrapper with real root/EGL subprocesses."""

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    import torch
    from omegaconf import OmegaConf
    from rlinf.envs.sim.libero.libero_env import LiberoEnv

    source = ROOT / "third_party/RLinf-recap-official"
    cfg = OmegaConf.load(source / "examples/embodiment/config/env/libero_10.yaml")
    cfg = OmegaConf.merge(cfg, {
        "total_num_envs": 2, "task_id_filter": [0], "is_eval": True,
        "auto_reset": False, "ignore_terminations": False,
        "video_cfg": {"save_video": False, "video_base_dir": str(ROOT / ".runtime/recap-official/video")},
    })
    env = LiberoEnv(cfg, num_envs=2, seed_offset=0, total_num_processes=1, worker_info=None)
    try:
        obs, _ = env.reset()
        initial_shapes = {key: list(value.shape) for key, value in obs.items()
                          if hasattr(value, "shape")}
        result = env.chunk_step(torch.zeros(2, 10, 7))
        assert env.task_ids.tolist() == [0, 0]
        assert len(result) == 5
        report = {"status": "passed", "uid": os.getuid(), "mujoco_gl": os.environ.get("MUJOCO_GL"),
                  "source": str(source), "num_envs": 2, "task_ids": env.task_ids.tolist(),
                  "steps_per_env": 10, "initial_shapes": initial_shapes, "checked_unix": time.time()}
    finally:
        env.env.close()
    output = ROOT / ".runtime/recap-official/libero-wrapper.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
