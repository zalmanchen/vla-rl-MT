"""Use the released task-specific PACE checkpoint and its original OpenPI transforms."""

from pathlib import Path
import os

import jax
import numpy as np
import torch
from openpi.policies import policy_config
from openpi.training import config


class PacePolicy:
    def __init__(self, task):
        root = Path(os.environ["PROJECT_ROOT"])
        self.policy = policy_config.create_trained_policy(
            config.get_config(f"pi05_robotwin_{task}"),
            root / "models/pi05-robotwin-pace" / task,
            sample_kwargs={"num_steps": 10},
        )
        self.seed = None

    def predict_action_batch(self, inputs, *, mode, rng):
        seed = rng.initial_seed()
        if seed != self.seed:
            self.policy._rng = jax.random.key(seed)
            self.seed = seed
        obs = {
            "images": {
                "cam_high": inputs["main_images"][0].transpose(2, 0, 1),
                "cam_left_wrist": inputs["wrist_images"][0, 0].transpose(2, 0, 1),
                "cam_right_wrist": inputs["wrist_images"][0, 1].transpose(2, 0, 1),
            },
            "state": np.asarray(inputs["states"][0], dtype=np.float32),
            "prompt": inputs["task_descriptions"][0],
        }
        result = self.policy.infer(obs)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.shape != (50, 14) or not np.isfinite(actions).all():
            raise ValueError(f"Invalid PACE actions: {actions.shape}")
        return torch.from_numpy(actions.copy())[None], result.get("policy_timing", {})
