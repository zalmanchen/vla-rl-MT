"""One backward-pass integration check, not an RL training result."""

import json
import os
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from offline_recap_data import EpisodeSamples, load_cfg_model, observation


def main():
    root = Path(os.environ["PROJECT_ROOT"])
    path = next((root / ".runtime/experiments/e4-offline-rl/demos/adjust_bottle").glob("*.hdf5"))
    entry = {"path": str(path), "frame": 0, "task_id": 0, "advantage": True}
    batch = next(iter(DataLoader(EpisodeSamples([entry]), batch_size=1, num_workers=0)))
    model = load_cfg_model()
    model.gradient_checkpointing_enable()
    model.train()
    model.freeze_vlm()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    obs = observation(batch, "cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, metrics = model({"observation": obs, "actions": batch["actions"].cuda(),
                               "advantage": batch["advantage"].cuda()})
    if not torch.isfinite(loss):
        raise ValueError("Non-finite flow loss")
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
        raise ValueError("Missing or non-finite gradients")
    result = {"scope": "one-example forward/backward only; synthetic positive label for interface check, no optimizer step or saved weights",
              "loss": loss.item(), "seconds": time.time() - started,
              "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "finite_gradients": True, "metrics": metrics}
    (root / ".runtime/experiments/e4-offline-rl/policy-smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
