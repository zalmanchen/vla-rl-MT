"""Audit completed M4 updates and paired development rollouts without a GPU."""

import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from vla_rl_mt.joint_value import digest, write_json
from vla_rl_mt.policy_update import PolicyPairs

ROOT = Path(__file__).resolve().parents[1]


def paired(rows, reference, tasks):
    result = {}
    for task in [*tasks, "macro"]:
        keys = sorted(key for key in reference if task == "macro" or key[0] == task)
        delta = np.array([int(rows[key]["success"])-int(reference[key]["success"]) for key in keys])
        # Equal 12-scene task cohorts: pooled mean is also the task macro mean.
        generator = np.random.default_rng(20261113)
        draws = []
        for name in tasks if task == "macro" else [task]:
            values = delta[[i for i, key in enumerate(keys) if key[0] == name]]
            draws.append(values[generator.integers(0, len(values), (10000, len(values)))].mean(1))
        interval = np.quantile(np.mean(draws, axis=0), [0.025, 0.975])
        result[task] = {"gained": int((delta > 0).sum()), "lost": int((delta < 0).sum()),
                        "delta": float(delta.mean()), "paired_scene_interval": interval.tolist()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, nargs="+", required=True)
    parser.add_argument("--zero-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = json.loads((ROOT/"configs/multitask/method_m4.json").read_text())
    execution = json.loads((args.updates/"execution.json").read_text())
    if not execution["complete"] or any(job.get("exit_code") != 0 for job in execution["jobs"]):
        raise ValueError("All policy workers must exit successfully")
    parent = torch.load(ROOT/plan["w1_checkpoint"], weights_only=True, map_location="cpu")
    summary = {"scope": "Two-task development comparison, one actor seed; selected-layer results exploratory",
               "training": {}, "successes": {}, "paired": {}, "rows": [],
               "rollouts_return_to_training": False, "final_test_used": False}
    identities = []
    for job in execution["jobs"]:
        name = job["name"]
        directory = args.updates/name
        checkpoint_path = directory/f"checkpoints/global_step_{plan['actor_steps']}/actor/addon.pt"
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        identity = checkpoint["identity"]
        if (identity != json.loads((directory/"live-data.json").read_text())
                or identity["parent_sha256"] != plan["parent_checkpoint_sha256"]["w1"]
                or identity["pool_manifest_sha256"] != plan["pool_manifest_sha256"]
                or checkpoint["step"] != plan["actor_steps"]):
            raise ValueError(f"Wrong training identity: {name}")
        metrics = [json.loads(line) for line in (directory/"joint-metrics.jsonl").read_text().splitlines()]
        if [row["step"] for row in metrics] != list(range(1, plan["actor_steps"]+1)):
            raise ValueError(f"Missing updates: {name}")
        for row in metrics:
            gradients = row["gradients"]
            if (gradients["frozen_policy"]["gradient_tensors"] or gradients["value_head"]["gradient_tensors"]
                    or gradients["adapter"]["trainable"] != 131072
                    or not all(group["finite"] for group in gradients.values())
                    or not np.isfinite(row["sft_loss"])):
                raise ValueError(f"Wrong gradient route: {name}")
        weights = torch.load(checkpoint_path.parent/"model_state_dict/full_weights.pt", weights_only=True, map_location="cpu")
        for group, values in checkpoint["addon"].items():
            for key, value in values.items():
                if not torch.equal(value, weights[f"{group}.{key}"]):
                    raise ValueError(f"Addon differs from official checkpoint: {name}")
                if group == "value_head" and not torch.equal(value, parent["addon"][group][key]):
                    raise ValueError(f"Frozen head changed: {name}")
        if all(torch.equal(value, parent["addon"]["adapter"][key]) for key, value in checkpoint["addon"]["adapter"].items()):
            raise ValueError(f"Policy adapter never changed: {name}")
        summary["training"][name] = {"checkpoint_sha256": digest(checkpoint_path), "identity": identity,
            "exposure": checkpoint["exposure"], "peak_allocated_gib": max(row["peak_allocated_gib"] for row in metrics)}
        if not name.startswith("bc"):
            identities.append(identity["sample_sequence_sha256"])
            path = Path(next(value.split("=", 1)[1] for value in job["overrides"] if value.startswith("+data.label_path=")))
            if digest(path) != identity["label_sha256"]:
                raise ValueError("Frozen policy label source changed")
            rows = [row for row in json.loads(path.read_text())["rows"] if row["use_for_loss"]]
            dataset = PolicyPairs(SimpleNamespace(rows=rows), mode=name, label_path=path,
                steps=plan["actor_steps"], batch_size=plan["actor_global_batch"], seed=plan["actor_seed"])
            composition = Counter()
            for index, draw in enumerate(dataset.draws):
                row = rows[int(draw)]
                generator = torch.Generator().manual_seed(plan["actor_seed"]+2_000_000+index)
                torch.randn((32, 32), generator=generator)
                torch.rand((), generator=generator)
                if bool(dataset.labels[draw]) and float(torch.rand((), generator=generator)) > 0.1:
                    composition[f"{row['task']}/{'success' if row['success'] else 'failure'}"] += 1
            if sum(composition.values()) != checkpoint["exposure"]["positive_conditional"]:
                raise ValueError("Reconstructed conditional-positive exposure differs")
            summary["training"][name]["positive_eligible_composition"] = dict(composition)
    if len(set(identities)) != 1:
        raise ValueError("Critic-label controls sampled different training decisions")
    for first, second in [("c1_cond", "c1_weighted"), ("c2_cond", "c2_weighted"), ("c2_cond", "shuffle_cond")]:
        if summary["training"][first]["exposure"] != summary["training"][second]["exposure"]:
            raise ValueError("Matched positive exposure changed")
    # Quantify how much of the real train pool the constrained shuffle changed.
    label_rows = json.loads((ROOT/plan["label_sources"]["w2"]["path"]).read_text())["rows"]
    pool = SimpleNamespace(rows=[row for row in label_rows if row["use_for_loss"]])
    options = dict(label_path=ROOT/plan["label_sources"]["w2"]["path"], steps=plan["actor_steps"],
                   batch_size=plan["actor_global_batch"], seed=plan["actor_seed"])
    original = PolicyPairs(pool, mode="c2_cond", **options)
    shuffled = PolicyPairs(pool, mode="shuffle_cond", **options)
    changed = original.labels != shuffled.labels
    summary["shuffle_changed_train_decisions"] = {task: {
        "changed": sum(bool(changed[i]) for i, row in enumerate(pool.rows) if row["task"] == task and row["split"] == "train"),
        "total": sum(row["task"] == task and row["split"] == "train" for row in pool.rows)}
        for task in sorted({row["task"] for row in pool.rows})}
    summary["shuffle_changed_scheduled_draws"] = int(changed[original.draws].sum())
    reference = json.loads((ROOT/".runtime/runs/method/m2-joint-v1/comparison.json").read_text())
    reference = {(row["task"], row["seed"]): row for row in reference["rows"] if row["variant"] == "w1"}
    variants = {"w1_plain": reference}
    for path in [args.zero_evaluation, *args.evaluation]:
        report = json.loads(path.read_text())
        if not report["complete"] or report["final_test_used"] or report["rollouts_return_to_training"]:
            raise ValueError("Evaluation incomplete or violates frozen-data scope")
        for variant in report["variants"]:
            name = variant["name"]
            if name in variants:
                raise ValueError("Duplicate evaluation variant")
            rows = [row for row in report["rows"] if row["variant"] == name]
            mapping = {(row["task"], row["seed"]): row for row in rows}
            if len(rows) != len(reference) or set(mapping) != set(reference):
                raise ValueError("Missing or duplicate evaluation scenes")
            checksum = (plan["parent_checkpoint_sha256"]["w1"] if name == "zero_positive"
                        else summary["training"][name]["checkpoint_sha256"])
            prompt = "positive" if name == "zero_positive" else summary["training"][name]["identity"]["deploy_prompt"]
            for key, row in mapping.items():
                if (row["status"] != "complete" or row["noise_seed"] != reference[key]["noise_seed"]
                        or row["observed_init_fingerprint"] != reference[key]["observed_init_fingerprint"]
                        or row["checkpoint_sha256"] != checksum or row["deploy_prompt"] != prompt):
                    raise ValueError("Evaluation identity or completion changed")
            variants[name] = mapping
            summary["rows"].extend(rows)
    if set(variants) != {"w1_plain", "zero_positive", *(job["name"] for job in execution["jobs"])}:
        raise ValueError("Missing policy evaluation")
    tasks = sorted({key[0] for key in reference})
    for name, rows in variants.items():
        summary["successes"][name] = {task: dict(Counter("success" if row["success"] else "failure"
            for key, row in rows.items() if key[0] == task)) for task in tasks}
    for new, old in [("zero_positive", "w1_plain"), ("bc", "w1_plain"), ("bc_positive", "zero_positive"),
                     ("c1_cond", "zero_positive"), ("c2_cond", "c1_cond"), ("shuffle_cond", "c2_cond"),
                     ("c1_weighted", "w1_plain"), ("c2_weighted", "c1_weighted"),
                     ("c1_weighted", "c1_cond"), ("c2_weighted", "c2_cond"),
                     ("layer_weighted", "last_weighted")]:
        summary["paired"][f"{new}-vs-{old}"] = paired(variants[new], variants[old], tasks)
    for name in ("w1", "w2"):
        if digest(ROOT/plan[f"{name}_checkpoint"]) != plan["parent_checkpoint_sha256"][name]:
            raise ValueError("Original critic/actor snapshot changed")
    summary["audit_passed"] = True
    summary["interval_scope"] = "Paired scene bootstrap conditional on one trained actor/head seed; no training uncertainty or selection adjustment"
    write_json(args.output, summary)
    print(json.dumps({key: summary[key] for key in ("successes", "shuffle_changed_train_decisions", "audit_passed")}, indent=2))


if __name__ == "__main__":
    main()
