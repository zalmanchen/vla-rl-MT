"""Official RLinf Gemma3/SigLIP2 critic, using the canonical local episode pool."""

import argparse
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from offline_recap_data import CAMERAS, ROOT, RUN, decode, entries_for


class ValueSamples(Dataset):
    def __init__(self, entries, scale):
        self.entries, self.scale = entries, scale

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        with h5py.File(e["path"], "r") as h:
            images = {key: decode(h["images/" + camera][e["frame"]])
                      for key, camera in zip(["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"], CAMERAS)}
            prompt = str(h.attrs["instruction"])
        return {"image": images, "image_mask": {k: True for k in images}, "prompt": prompt,
                "target_values": np.float32(np.clip(e["return"] / self.scale, -1, 0)),
                "return_raw": np.float32(e["return"]), "return_normalized": np.float32(np.clip(e["return"] / self.scale, -1, 0))}


def tree_to(data, device):
    if isinstance(data, dict):
        return {k: tree_to(v, device) for k, v in data.items()}
    return data.to(device) if torch.is_tensor(data) else data


def build_model():
    from transformers import AutoTokenizer
    from rlinf.models.embodiment.value_model.recap.configuration import ValueCriticConfig
    from rlinf.models.embodiment.value_model.recap.modeling_critic import ValueCriticModel
    from rlinf.models.embodiment.value_model.recap.processing import ValueProcessor
    from rlinf.models.embodiment.value_model.recap.data_collator import ValueDataCollator
    gemma = ROOT / "models/gemma-3-270m"
    siglip = ROOT / "models/siglip2-so400m-patch14-224"
    cfg = ValueCriticConfig(siglip_path=str(siglip), gemma3_path=str(gemma),
                           critic_expert_variant="gemma_1m", num_bins=201, v_min=-1.0, v_max=0.0,
                           precision="bfloat16", freeze_vision_encoder=True, freeze_vlm=True,
                           stop_gradient_to_vlm=False, action_dim=14, action_horizon=32,
                           max_token_len=200, max_language_len=50, train_expert_only=True)
    model = ValueCriticModel(cfg).cuda()
    tokenizer = AutoTokenizer.from_pretrained(str(gemma), local_files_only=True, add_bos_token=True)
    processor = ValueProcessor(tokenizer=tokenizer, max_token_len=200)
    return model, ValueDataCollator(processor=processor, max_length=50, train=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    plan = json.loads((RUN / "plan.json").read_text())
    ids = {task: i for i, task in enumerate(plan["nested_tasks"]["50"])}
    paths = sorted((RUN / "demos").glob("*/*.hdf5")) + sorted((RUN / "pool/clean").glob("*/trajectory_*.hdf5"))
    if not args.smoke and len(paths) != 450:
        raise ValueError(f"Incomplete fixed episode pool: {len(paths)}/450")
    entries = entries_for(paths, ids, stride=8)
    train = [e for e in entries if e["split"] == "train"]
    val = [e for e in entries if e["split"] != "train"]
    scale = max(1.0, -min(e["return"] for e in train))
    model, collator = build_model()
    batch_size = 1 if args.smoke else 8
    counts = {t: sum(e["task"] == t for e in train) for t in plan["anchors"]}
    weights = [1.0 / counts[e["task"]] for e in train]
    loader = DataLoader(ValueSamples(train, scale), batch_size=batch_size,
                        sampler=WeightedRandomSampler(weights, max(batch_size * args.steps, batch_size), replacement=True),
                        num_workers=0 if args.smoke else 4, collate_fn=collator, pin_memory=not args.smoke)
    # Keep optimizer master parameters in FP32; frozen backbones retain their BF16 precision.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-10)
    model.train()
    model.value_expert.eval()
    steps = 1 if args.smoke else args.steps
    logs = (RUN / "value-train.jsonl").open("a", buffering=1)
    started = time.time()
    for step, batch in enumerate(loader, 1):
        if step > steps:
            break
        batch = tree_to(batch, "cuda")
        if not all(mask.all().item() for mask in batch["observation"]["image_masks"].values()):
            raise ValueError("All three RoboTwin camera views must be present")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch["observation"], target_values=batch["target_values"])
            loss = output.loss
        if not torch.isfinite(loss):
            raise ValueError("Non-finite value loss")
        loss.backward()
        projection_grad = model.value_expert.multi_modal_proj.weight.grad
        if projection_grad is None or not torch.isfinite(projection_grad).all() or projection_grad.abs().sum().item() == 0:
            diagnostic = {"projection_grad_present": projection_grad is not None, "projection_grad_finite": bool(torch.isfinite(projection_grad).all()) if projection_grad is not None else None, "projection_grad_abs_sum": projection_grad.abs().sum().item() if projection_grad is not None else None, "head_grad_abs_sum": model.value_head.value_proj.weight.grad.abs().sum().item() if model.value_head.value_proj.weight.grad is not None else None, "loss": loss.item()}
            (RUN / "value-gradient-diagnostic.json").write_text(json.dumps(diagnostic, indent=2))
            raise ValueError("Cross-modal gradient diagnostic: " + str(diagnostic))
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        if not args.smoke:
            optimizer.step()
        item = {"phase": "smoke" if args.smoke else "train", "step": step, "loss": loss.item(), "grad_norm": norm.item(), "projection_gradient_nonzero": True, "seconds": time.time() - started}
        logs.write(json.dumps(item) + "\n")
        if step == 1 or step % 50 == 0:
            print(json.dumps(item), flush=True)
    if args.smoke:
        (RUN / "value-smoke.json").write_text(json.dumps(dict(item, scope="forward/backward only; no optimizer update"), indent=2))
        return
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items() if name in trainable_names}
    torch.save({"state_dict": state, "scale": scale}, RUN / "value-trainable.pt")
    # Evaluate held-out episodes before using predictions as actor training labels.
    model.eval()
    all_predictions = []
    inference = DataLoader(ValueSamples(entries, scale), batch_size=32, shuffle=False, num_workers=4, collate_fn=collator)
    with torch.inference_mode():
        for batch in inference:
            batch = tree_to(batch, "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(batch["observation"])
            all_predictions.extend(output.predicted_values.detach().float().cpu().reshape(-1).tolist())
    if len(all_predictions) != len(entries):
        raise ValueError("Prediction count mismatch")
    prediction = {(e["path"], e["frame"]): v for e, v in zip(entries, all_predictions)}
    for e, value in zip(entries, all_predictions):
        t, n = e["frame"], e["length"]
        end = min(t + 32, n)
        with h5py.File(e["path"], "r") as h:
            reward = float(h["reward"][t:end].sum())
            e["success"] = bool(h.attrs["success"])
        following = 0.0 if end >= n else prediction[(e["path"], end)]
        e["value"] = value
        e["advantage_continuous"] = reward / scale + following - value
    threshold = float(np.quantile([e["advantage_continuous"] for e in entries if e["split"] == "train"], 0.7))
    for e in entries:
        # Match RLinf save_advantages_to_dataset: expert/SFT samples are always positive.
        e["advantage"] = e["source"] == "expert" or e["advantage_continuous"] >= threshold
    validation_mse = float(np.mean([(e["value"] - e["return"] / scale) ** 2 for e in entries if e["split"] != "train"]))
    label_file = RUN / "advantage-samples.json"
    label_file.write_text(json.dumps({"scale": scale, "threshold": threshold,
                                    "value_validation_mse": validation_mse, "entries": entries}, indent=2))
    (RUN / "VALUE_AND_ADVANTAGES_READY").write_text(str(time.time()))
    print("value validation MSE", validation_mse, "advantage samples", len(entries), flush=True)


if __name__ == "__main__":
    main()
