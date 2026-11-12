"""Verify zero-update CFG loading and action parity using fixed noise and inputs."""

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from prepare_recap_data import CAMERAS, JOINT_KEYS, decode_jpeg
from smoke_model import load_model

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = next(
        (ROOT / "datasets/RoboTwin2.0-clean/click_bell").glob(
            "**/data/episode_0000000.hdf5"
        )
    )
    with h5py.File(path) as h:
        image = {k: decode_jpeg(h[f"vision/{v}/colors"][0]) for k, v in CAMERAS.items()}
        state = np.concatenate([h[f"state/{k}"][0] for k in JOINT_KEYS])
        instruction = json.loads(h["instructions"][()].decode())[0]
    obs = {
        "main_images": image["cam_high"][None],
        "wrist_images": np.stack([image["cam_left_wrist"], image["cam_right_wrist"]])[
            None
        ],
        "states": state[None],
        "task_descriptions": [instruction],
    }
    model = load_model()
    torch.manual_seed(17)
    with torch.inference_mode():
        native, _ = model.predict_action_batch(obs, mode="eval")
    native = native.float().cpu().numpy()
    del model
    torch.cuda.empty_cache()
    from rlinf.models.embodiment.openpi_cfg import get_model

    with initialize_config_dir(config_dir=str(ROOT / "configs/robotwin_pilot"), version_base="1.1"):
        cfg = compose(config_name="policy").actor.model
    cfg.openpi.guidance_type = "no_guide"
    model = get_model(cfg).to("cuda").eval()
    torch.manual_seed(17)
    with torch.inference_mode():
        unconditional, _ = model.predict_action_batch(obs, mode="eval")
    object.__setattr__(model.config, "guidance_type", "positive")
    torch.manual_seed(17)
    with torch.inference_mode():
        positive, _ = model.predict_action_batch(obs, mode="eval")
    differences = np.abs(native - unconditional)
    result = {
        "status": "passed"
        if np.isfinite(positive).all() and differences.max() < 0.03
        else "failed",
        "shape": list(positive.shape),
        "native_vs_cfg_no_guide_max_abs": float(differences.max()),
        "native_vs_cfg_no_guide_mean_abs": float(differences.mean()),
        "zero_update_positive_vs_no_guide_mean_abs": float(
            np.abs(positive - unconditional).mean()
        ),
        "note": "Two official implementations have different parameter precision; tolerance is 0.03 in raw 14D joint/gripper outputs. Positive differences are recorded, not interpreted as improvement.",
        "noise_seed": 17,
        "model_path": str(cfg.model_path),
    }
    output = ROOT / ".runtime/validation/recap-policy-parity.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if result["status"] != "passed":
        raise RuntimeError("Zero-update CFG action parity check failed")


if __name__ == "__main__":
    main()
