import os
from pathlib import Path

import torch
from omegaconf import OmegaConf
from rlinf.models.embodiment.openpi import get_model

ROOT = Path(os.environ["PROJECT_ROOT"])


def load_model(precision="bf16"):
    cfg = OmegaConf.load(
        ROOT / "third_party/RLinf/examples/embodiment/config/model/pi0_5.yaml"
    )
    cfg.model_path = str(ROOT / "models/pi0.5_robotwin2")
    cfg.precision = precision
    cfg.num_action_chunks = 32
    cfg.action_dim = 14
    cfg.num_steps = 10
    cfg.openpi.task = "eval"
    cfg.openpi.config_name = "pi05_aloha_robotwin"
    cfg.openpi.action_horizon = 32
    cfg.openpi.num_images_in_input = 3
    cfg.openpi.torch_compile = False
    cfg.openpi_data = {
        "norm_stats_path": str(
            ROOT
            / "models/pi0.5_robotwin2/assets/pi0.5_clean_randomize_joint_training/norm_stats.json"
        ),
        "adapt_to_pi": True,
        "extra_delta_transform": True,
    }
    print("Loading checkpoint with RLinf", flush=True)
    dtype = torch.float32 if precision == "fp32" else torch.bfloat16
    model = get_model(cfg, torch_dtype=dtype).to("cuda").eval()
    print("MODEL LOADED", flush=True)
    return model


if __name__ == "__main__":
    model = load_model()
    print("parameters", sum(p.numel() for p in model.parameters()), flush=True)
