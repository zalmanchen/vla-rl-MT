"""Canonical episode loading and the unchanged RLinf/OpenPI transform pipeline."""

import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from openpi import transforms
from rlinf.data.datasets.recap.cfg_model import TokenizePromptWithGuidance
from rlinf.models.embodiment.openpi.transforms.pipeline import build_openpi_transforms

ROOT = Path(os.environ["PROJECT_ROOT"])
RUN = ROOT / ".runtime/experiments/e4-offline-rl"
CAMERAS = ["head", "left_wrist", "right_wrist"]


def make_transform():
    inputs, _ = build_openpi_transforms(str(ROOT / "models/pi0.5_robotwin2"), "pi05_aloha_robotwin", {
        "norm_stats_path": str(ROOT / "models/pi0.5_robotwin2/assets/pi0.5_clean_randomize_joint_training/norm_stats.json"),
        "adapt_to_pi": True, "extra_delta_transform": True,
    })
    inputs = [TokenizePromptWithGuidance(t.tokenizer, t.discrete_state_input)
              if isinstance(t, transforms.TokenizePrompt) else t for t in inputs]
    return transforms.compose(inputs)


def decode(encoded):
    bgr = cv2.imdecode(np.asarray(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Corrupt JPEG")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class EpisodeSamples(Dataset):
    def __init__(self, entries, transform=None):
        self.entries = entries
        self.transform = transform or make_transform()

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        with h5py.File(entry["path"], "r") as h:
            t = entry["frame"]
            images = {k: decode(h["images/" + k][t]) for k in CAMERAS}
            state = h["state"][t]
            end = min(t + 32, len(h["action"]))
            actions = h["action"][t:end]
            if len(actions) < 32:
                actions = np.concatenate([actions, np.repeat(actions[-1:], 32 - len(actions), axis=0)])
            sample = {"observation/image": images["head"],
                      "observation/wrist_image": np.stack([images["left_wrist"], images["right_wrist"]]),
                      "observation/state": state, "actions": actions,
                      "prompt": str(h.attrs["instruction"])}
        result = self.transform(sample)
        result["task_id"] = np.int64(entry["task_id"])
        result["advantage"] = np.bool_(entry.get("advantage", False))
        return result


def entries_for(paths, task_ids, stride=8):
    entries = []
    for path in paths:
        with h5py.File(path, "r") as h:
            n = len(h["state"])
            task = str(h.attrs["task"])
            split = str(h.attrs["split"])
            reward = np.asarray(h["reward"])
            returns = np.cumsum(reward[::-1])[::-1]
            source = str(h.attrs.get("source", "rollout"))
        for frame in sorted(set(range(0, n, stride)) | {n - 1}):
            entries.append({"path": str(path), "frame": frame, "length": n,
                            "task": task, "task_id": task_ids[task], "split": split,
                            "source": source, "return": float(returns[frame]),
                            "terminal": frame == n - 1})
    return entries


def observation(batch, device):
    from rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model import Observation
    keys = ["state", "tokenized_prompt", "tokenized_prompt_mask", "tokenized_positive_guidance_prompt",
            "tokenized_positive_guidance_prompt_mask", "tokenized_negative_guidance_prompt",
            "tokenized_negative_guidance_prompt_mask"]
    values = {key: batch[key].to(device) for key in keys}
    values["state"] = values["state"].float()
    return Observation(images={k: (v.permute(0, 3, 1, 2).contiguous().to(device=device, dtype=torch.float32) / 127.5 - 1.0) for k, v in batch["image"].items()},
                       image_masks={k: v.to(device) for k, v in batch["image_mask"].items()}, **values)


def load_cfg_model():
    from safetensors.torch import load_model
    from rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model import OpenPi0Config, OpenPi0ForCFGActionPrediction
    cfg = OpenPi0Config(pi05=True, action_horizon=32, dtype="float32", max_token_len=200,
                       discrete_state_input=True, config_name="pi05_aloha_robotwin", num_images_in_input=3,
                       action_chunk=32, action_env_dim=14, train_expert_only=True,
                       positive_only_conditional=True, unconditional_prob=0.1,
                       cfgrl_guidance_scale=1.0)
    model = OpenPi0ForCFGActionPrediction(cfg)
    missing, unexpected = load_model(model, str(ROOT / "models/pi0.5_robotwin2/model.safetensors"), strict=False)
    if missing or unexpected:
        raise ValueError(f"Checkpoint mismatch: {missing}, {unexpected}")
    model.freeze_vlm()
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    return model.to("cuda")
