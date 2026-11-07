"""Evaluate the released checkpoint with its original OpenPI PyTorch sampler."""

import os
from pathlib import Path

import torch
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from rlinf.models.embodiment.openpi.env_io import EnvIO
from rlinf.models.embodiment.openpi.transforms.pipeline import build_openpi_transforms
from safetensors.torch import load_model


class ReferencePolicy(EnvIO):
    def __init__(self):
        root = Path(os.environ["PROJECT_ROOT"])
        self.device = torch.device("cuda")
        self.config_name = "pi05_aloha_robotwin"
        self.state_indices = None
        self.action_chunk = 32
        self.model = PI0Pytorch(
            Pi0Config(pi05=True, action_horizon=32, dtype="float32")
        )
        missing, unexpected = load_model(
            self.model,
            str(root / "models/pi0.5_robotwin2/model.safetensors"),
            strict=False,
        )
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint mismatch: {missing}, {unexpected}")
        self.model = self.model.to(self.device).eval()
        inputs, outputs = build_openpi_transforms(
            str(root / "models/pi0.5_robotwin2"),
            self.config_name,
            data_kwargs={
                "norm_stats_path": str(
                    root
                    / "models/pi0.5_robotwin2/assets/pi0.5_clean_randomize_joint_training/norm_stats.json"
                ),
                "adapt_to_pi": True,
                "extra_delta_transform": True,
            },
        )
        self.setup_transforms(inputs, outputs)
        torch.set_float32_matmul_precision("highest")
        self.sample = getattr(
            self.model.sample_actions,
            "_torchdynamo_orig_callable",
            self.model.sample_actions,
        )

    @torch.inference_mode()
    def predict_action_batch(self, env_obs, mode="eval", rng=None):
        if mode != "eval":
            raise ValueError("ReferencePolicy is inference-only")
        observation = self.env_obs_to_observation(env_obs)
        observation.images = {
            k: v.permute(0, 3, 1, 2).contiguous() for k, v in observation.images.items()
        }
        noise = torch.randn(
            observation.state.shape[0], 32, 32, device=self.device, generator=rng
        )
        actions = self.sample(self.device, observation, noise=noise, num_steps=10)
        return self.decode_actions(actions, observation.state), {
            "model_actions": actions
        }
