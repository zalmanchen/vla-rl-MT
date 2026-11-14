"""Fixed-noise successful-action fit before/after BC, on frozen train/val episodes."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from vla_rl_mt.expert_policy import build_model
from vla_rl_mt.joint_value import digest, to_observation, write_json
from vla_rl_mt.policy_update import GuidedPool, masked_per_sample

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, help="Explicit checkpoint variants, starting with the unchanged W1 parent")
    parser.add_argument("--pool", type=Path, help="Explicit frozen training/validation view")
    parser.add_argument("--pool-sha256", help="Required manifest SHA256 for an explicit view")
    parser.add_argument("--noise-keys", type=Path, help="Stable episode::decision noise indices across different data views")
    args = parser.parse_args()
    if (args.pool is None) != (args.pool_sha256 is None):
        parser.error("--pool and --pool-sha256 must be supplied together")
    args.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((ROOT/"configs/multitask/method_m4.json").read_text())
    pool = GuidedPool(args.pool.resolve() if args.pool else ROOT/plan["pool"],
                      args.pool_sha256 if args.pool else plan["pool_manifest_sha256"])
    noise_keys = json.loads(args.noise_keys.read_text()) if args.noise_keys else None
    model = build_model(OmegaConf.create({"pool_path": str(pool.path), "mode": "bc",
        "update_scope": "expert_adapter", "adapter_lr": 3e-4, "expert_lr": 1e-5})).eval()
    indices = [i for i, row in enumerate(pool.rows) if row["success"]]
    if noise_keys is not None:
        # Common validation precedes new training in both views; its batches
        # and explicit noise/time therefore remain the same across pool sizes.
        indices.sort(key=lambda i: noise_keys[f"{pool.rows[i]['episode']}::{pool.rows[i]['observation_index']}"])
    variants = [("w1", ROOT/plan["w1_checkpoint"]), *[(scope, args.updates/f"{scope}-bc/checkpoints/global_step_128/actor/addon.pt")
                                                    for scope in ("adapter_only", "expert_adapter")]]
    if args.plan:
        variants = [(row["name"], ROOT/row["checkpoint"]) for row in json.loads(args.plan.read_text())["variants"]]
        if not variants or variants[0] != ("w1", ROOT/plan["w1_checkpoint"]):
            raise ValueError("Explicit fit plans must begin with the original W1 parent")
    report = {"scope": "Offline successful-action FM fit only, not an oracle policy or success proxy",
              "noise": "canonical decision index+3000000, same noise/time across variants",
              "native_expert_requires_grad": True, "gradients_computed": False,
              "success_only": True, "rows": [], "metrics": {}, "checkpoint_sha256": {},
              "source_sha256": digest(Path(__file__)), "complete": False,
              "final_test_used": False, "rollouts_return_to_training": False}
    report["pool_manifest_sha256"] = digest(pool.path/"manifest.json")
    if args.noise_keys:
        report["noise_keys_sha256"] = digest(args.noise_keys)
        report["noise"] = "stable episode::decision key+3000000, shared across views; original validation indices retained"
    if args.plan:
        report["plan_sha256"] = digest(args.plan)
    for name, path in variants:
        before = digest(path)
        report["checkpoint_sha256"][name] = before
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model.load_addon(checkpoint["addon"])
        if name != "w1":
            model.expert.load_state_dict(checkpoint["expert"], strict=True)
        losses = []
        for start in range(0, len(indices), 2):
            selected = indices[start:start+2]
            samples = [pool.sample(i) for i in selected]
            raw, actions, masks = torch.utils.data.default_collate(samples)
            noise, times = [], []
            for index in selected:
                row = pool.rows[index]
                noise_index = noise_keys[f"{row['episode']}::{row['observation_index']}"] if noise_keys is not None else index
                generator = torch.Generator().manual_seed(noise_index+3_000_000)
                noise.append(torch.randn((32, 32), generator=generator))
                times.append(torch.rand((), generator=generator).pow(1/1.5)*0.999+0.001)
            with torch.no_grad():
                elementwise = model.policy.compute_loss(to_observation(raw, "cuda"), actions.to("cuda"), train=False,
                    noise=torch.stack(noise).to("cuda"), time=torch.stack(times).to("cuda"))
                values = masked_per_sample(elementwise, masks.to("cuda")).cpu().tolist()
            losses.extend(values)
            report["rows"].extend({**pool.rows[index], "variant": name, "fm_loss": value}
                                  for index, value in zip(selected, values, strict=True))
        metrics = {}
        for split in ("train", "val"):
            by_task = defaultdict(lambda: defaultdict(list))
            for index, value in zip(indices, losses, strict=True):
                row = pool.rows[index]
                if row["split"] == split:
                    by_task[row["task"]][row["episode"]].append(value)
            per_task = {task: float(np.mean([np.mean(values) for values in episodes.values()]))
                        for task, episodes in by_task.items()}
            metrics[split] = {"per_task": per_task, "macro_episode_then_task": float(np.mean(list(per_task.values()))),
                              "episodes": sum(len(episodes) for episodes in by_task.values())}
        report["metrics"][name] = metrics
        if digest(path) != before:
            raise ValueError("Action-fit evaluation changed checkpoint")
        print(json.dumps({"variant": name, "metrics": metrics}), flush=True)
    report["complete"] = True
    write_json(args.output/"comparison.json", report)
    model.close()


if __name__ == "__main__":
    main()
