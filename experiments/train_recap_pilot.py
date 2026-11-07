"""Single/multi-task pilot using the existing RLinf advantage-conditioned flow loss."""

import argparse
import fcntl
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from offline_recap_data import EpisodeSamples, ROOT, RUN, load_cfg_model, observation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--mode", choices=["recap", "success_replay"], default="recap")
    parser.add_argument("--steps-per-task", type=int, default=50)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--label-file", type=Path, default=RUN / "advantage-samples.json")
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    out = RUN / "jobs" / args.job
    out.mkdir(parents=True, exist_ok=True)
    job_lock = (out / "training.lock").open("a")
    fcntl.flock(job_lock, fcntl.LOCK_EX)
    if (out / "FINISHED").exists():
        return
    labeled = json.loads(args.label_file.read_text())
    entries = [dict(e) for e in labeled["entries"] if e["split"] == "train" and e["task"] in args.tasks]
    if args.mode == "success_replay":
        entries = [e for e in entries if e["success"]]
        for e in entries:
            e["advantage"] = False
    groups = {}
    for e in entries:
        key = (e["task"], e["source"])
        groups[key] = groups.get(key, 0) + 1
    weights = [1.0 / groups[(e["task"], e["source"])] for e in entries]
    total_steps = args.steps_per_task * len(args.tasks)
    global_batch = 32
    if global_batch % args.micro_batch:
        raise ValueError("Micro batch must divide global batch")
    accumulation = global_batch // args.micro_batch
    sampler = WeightedRandomSampler(weights, total_steps * global_batch, replacement=True,
                                    generator=torch.Generator().manual_seed(0))
    loader = DataLoader(EpisodeSamples(entries), batch_size=args.micro_batch, sampler=sampler,
                        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    model = load_cfg_model()
    model.gradient_checkpointing_enable()
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.train()
    model.freeze_vlm()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=5e-6, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10)
    contract = {"algorithm": "RLinf RECAP CFG policy loss" if args.mode == "recap" else "success-only flow-matching replay",
                "tasks": args.tasks, "mode": args.mode, "training_seed": 0,
                "global_batch": global_batch, "micro_batch": args.micro_batch, "steps": total_steps,
                "value_labels": "not used; success-only supervised replay" if args.mode == "success_replay" else "shared identical value model/advantage labels for all single and multi-task jobs",
                "update_scope": "action expert and action/time projections only; VLM frozen",
                "trainable_parameters": sum(p.numel() for p in params),
                "pool_manifest": str(RUN / "pool-manifest.json")}
    (out / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    optimizer.zero_grad(set_to_none=True)
    losses = []
    exposure = {}
    started = time.time()
    step = 0
    with (out / "train.jsonl").open("a", buffering=1) as log:
        for micro, batch in enumerate(loader, 1):
            obs = observation(batch, "cuda")
            for task_id in batch["task_id"].tolist():
                exposure[str(task_id)] = exposure.get(str(task_id), 0) + 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, metrics = model({"observation": obs, "actions": batch["actions"].cuda().float(),
                                       "advantage": batch["advantage"].cuda()})
            if not torch.isfinite(loss):
                raise ValueError("Non-finite policy loss")
            (loss / accumulation).backward()
            losses.append(loss.detach().item())
            if micro % accumulation:
                continue
            step += 1
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            warmup = min(5, max(1, total_steps // 10))
            rate = min(1.0, step / warmup) * (0.1 + 0.9 * (1 + math.cos(math.pi * step / total_steps)) / 2)
            for group in optimizer.param_groups:
                group["lr"] = 5e-6 * rate
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            item = {"step": step, "loss": float(np.mean(losses)), "grad_norm": norm.item(),
                    "seconds": time.time() - started, "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                    "exposure_counts": exposure}
            losses.clear()
            log.write(json.dumps(item) + "\n")
            (out / "progress.json").write_text(json.dumps(item, indent=2))
            if step == 1 or step % 10 == 0:
                print(args.job, json.dumps(item), flush=True)
    if step != total_steps:
        raise RuntimeError(f"Incomplete policy run: {step}/{total_steps}")
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    weights = {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items() if name in trainable}
    torch.save({"state_dict": weights, "contract": contract}, out / "actor-trainable.pt")
    (out / "FINISHED").write_text(str(time.time()))


if __name__ == "__main__":
    main()
