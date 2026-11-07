"""Evaluate a trainable-only pilot checkpoint through the unchanged E1 sampler."""

from pathlib import Path

import torch

from reference_policy import ReferencePolicy


class PilotPolicy(ReferencePolicy):
    def __init__(self, checkpoint):
        super().__init__()
        saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        missing, unexpected = self.model.load_state_dict(saved["state_dict"], strict=False)
        if unexpected:
            raise ValueError(f"Unexpected adapter weights: {unexpected}")
        if not saved["state_dict"]:
            raise ValueError("Empty policy checkpoint")
        self.conditional = saved["contract"]["mode"] == "recap"

    def predict_action_batch(self, env_obs, mode="eval", rng=None):
        if self.conditional:
            env_obs = dict(env_obs)
            env_obs["task_descriptions"] = [p + "\nAdvantage: positive" for p in env_obs["task_descriptions"]]
        return super().predict_action_batch(env_obs, mode=mode, rng=rng)
