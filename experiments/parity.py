"""Compare pinned OpenPI and RLinf samplers using identical inputs and noise."""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(os.environ["PROJECT_ROOT"])
OUT = ROOT / ".runtime/experiments/e0-parity"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    mode = argparse.ArgumentParser()
    mode.add_argument("mode", choices=["native", "reference", "compare"])
    args = mode.parse_args()
    if args.mode == "compare":
        native = torch.load(OUT / "native.pt", weights_only=True)
        ref = torch.load(OUT / "reference.pt", weights_only=True)
        delta = (native - ref).abs()
        report = {
            "max_abs": delta.max().item(),
            "mean_abs": delta.mean().item(),
            "reference_abs_mean": ref.abs().mean().item(),
            "relative_l2": ((native - ref).norm() / ref.norm().clamp_min(1e-8)).item(),
            "max_abs_tolerance": 0.01,
            "relative_l2_tolerance": 0.01,
            "scope": "fp32 sampler parity on deterministic synthetic three-camera input; not an environment performance test",
        }
        report["passed"] = report["max_abs"] <= 0.01 and report["relative_l2"] <= 0.01
        (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        if not report["passed"]:
            sys.exit(1)
        return
    torch.manual_seed(20261104)
    if args.mode == "native":
        sys.path.insert(0, str(ROOT / "bootstrap"))
        from smoke_model import load_model

        model = load_model(precision="fp32")
        from compat import match_reference_activations

        print("SigLIP tanh layers", match_reference_activations(model), flush=True)
        generator = np.random.default_rng(20261104)
        obs = {
            "main_images": generator.integers(0, 256, (1, 240, 320, 3), dtype=np.uint8),
            "wrist_images": generator.integers(
                0, 256, (1, 2, 240, 320, 3), dtype=np.uint8
            ),
            "states": np.zeros((1, 14), dtype=np.float32),
            "task_descriptions": ["Pick up the bottle and place it upright."],
        }
        prepared = model.env_obs_to_observation(obs)
        payload = {
            name: getattr(prepared, name)
            for name in [
                "images",
                "image_masks",
                "state",
                "tokenized_prompt",
                "tokenized_prompt_mask",
            ]
        }
        noise = torch.randn(1, 32, 32, device="cuda")
        torch.save({"observation": payload, "noise": noise}, OUT / "inputs.pt")
        torch.set_float32_matmul_precision("highest")
        with torch.inference_mode():
            actions = model.sample_actions(prepared, num_steps=10, noise=noise)
    else:
        import transformers
        from openpi.models.pi0_config import Pi0Config
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from safetensors.torch import load_model

        print("reference transformers", transformers.__file__, flush=True)
        config = Pi0Config(pi05=True, action_horizon=32, dtype="float32")
        model = PI0Pytorch(config)
        missing, unexpected = load_model(
            model, str(ROOT / "models/pi0.5_robotwin2/model.safetensors"), strict=False
        )
        print("missing", missing, "unexpected", unexpected, flush=True)
        if missing or unexpected:
            raise RuntimeError("Reference checkpoint keys do not match")
        model = model.to("cuda").eval()
        payload = torch.load(OUT / "inputs.pt", weights_only=True, map_location="cuda")
        payload["observation"]["images"] = {
            key: value.permute(0, 3, 1, 2).contiguous()
            for key, value in payload["observation"]["images"].items()
        }
        prepared = SimpleNamespace(
            **payload["observation"], token_ar_mask=None, token_loss_mask=None
        )
        noise = payload["noise"]
        sample = getattr(
            model.sample_actions, "_torchdynamo_orig_callable", model.sample_actions
        )
        torch.set_float32_matmul_precision("highest")
        with torch.inference_mode():
            actions = sample(torch.device("cuda"), prepared, noise=noise, num_steps=10)
    torch.save(actions.cpu(), OUT / (args.mode + ".pt"))
    print(args.mode, "DONE", actions.shape, flush=True)


if __name__ == "__main__":
    main()
