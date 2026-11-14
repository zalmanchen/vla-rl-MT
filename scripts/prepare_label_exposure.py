"""Freeze mixed outcomes, evaluate an unchanged critic, and seal label controls."""

import argparse
import copy
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"bootstrap"))
from evaluate_multitask import digest, write_json

PROTOCOL = ROOT/"configs/multitask/label_exposure.json"


def matched_permutation(labels, rows, instructions, draws, eligible, seed):
    """Permute fixed labels within outcome/instruction/exposure strata."""
    counts, kept = Counter(map(int, draws)), Counter(int(i) for i, ok in zip(draws, eligible, strict=True) if ok)
    strata = defaultdict(list)
    for i, row in enumerate(rows):
        if row["split"] == "train":
            strata[row["task"], instructions[row["episode"]], bool(row["success"]), counts[i], kept[i],
                   row["next_step"]-row["step"]].append(i)
    output = labels.clone()
    generator = torch.Generator().manual_seed(seed+404)
    for indices in strata.values():
        permutation = torch.randperm(len(indices), generator=generator)
        output[indices] = labels[torch.tensor(indices)[permutation]]
    return output


def exact_subset(indices, weights, target):
    """Choose a fixed observation mask with the requested scheduled exposure."""
    reachable = {0: ()}
    for index in indices:
        weight = weights[index]
        if not weight:
            continue
        for total, chosen in list(reachable.items()):
            if total+weight <= target and total+weight not in reachable:
                reachable[total+weight] = (*chosen, index)
        if target in reachable:
            return reachable[target]
    if target not in reachable:
        raise ValueError(f"Cannot match success-only exposure {target}")
    return reachable[target]


def freeze(config, work, destination):
    raw, old = ROOT/config["raw_pool"], ROOT/config["validation_pool"]
    if digest(old/"manifest.json") != config["validation_manifest_sha256"]:
        raise ValueError("Original validation pool changed")
    source = json.loads((raw/"manifest.json").read_text())
    original = json.loads((old/"manifest.json").read_text())
    rows = source["episodes"]+[r for r in original["episodes"] if r["split"] == "val"]
    if len(source["episodes"]) != 192 or any(r["split"] != "train" for r in source["episodes"]):
        raise ValueError("Expected all192 fixed training attempts")
    destination.mkdir(parents=True, exist_ok=False)
    for row in rows:
        origin = old if row["split"] == "val" else raw
        src, dst = origin/row["file"], destination/row["file"]
        if digest(src) != row["sha256"]:
            raise ValueError("Parent episode changed")
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(src, dst)
    parent_config = copy.deepcopy(source["config"])
    parent_config["scope"] = "All192 frozen Motus training attempts and unchanged original24 validation episodes"
    parent_config["splits"]["val"] = original["config"]["splits"]["val"]
    plan = {"config": parent_config, "protocol": source["protocol"], "cases": rows,
            "source_manifest_sha256": digest(raw/"manifest.json"), "validation_manifest_sha256": digest(old/"manifest.json")}
    write_json(destination/"plan.json", plan)
    counts = Counter((r["task"], r["split"], bool(r["success"])) for r in rows)
    coverage = {f"{task}/{split}": {"success": counts[task, split, True], "failure": counts[task, split, False]}
                for task in source["config"]["tasks"] for split in ("train", "val")}
    manifest = {**plan, "scope": parent_config["scope"], "episodes": rows, "coverage": coverage,
                "plan_sha256": digest(destination/"plan.json"), "bc_only": False,
                "outcome_coverage_ready": all(v["success"] and v["failure"] for v in coverage.values()),
                "scene_split_certified": True, "full_simulator_state_saved": False, "contrastive_pairs_verified": False}
    write_json(destination/"manifest.json", manifest)
    write_json(work/"pool-seal.json", {"manifest_sha256": digest(destination/"manifest.json"),
                                      "protocol_sha256": digest(PROTOCOL), "coverage": coverage})


def evaluate_values(config, destination, work):
    from vla_rl_mt.joint_value import BoundaryPool
    from vla_rl_mt.shared_value import SharedPrefixValue
    from evaluate_method_m2 import audit_values
    from evaluate_multitask import load_model

    seal = json.loads((work/"pool-seal.json").read_text())
    pool = BoundaryPool(destination, seal["manifest_sha256"])
    base = json.loads((ROOT/"configs/multitask/method_m4.json").read_text())
    checkpoint = torch.load(ROOT/base["w1_checkpoint"], map_location="cpu", weights_only=True)
    if digest(ROOT/base["w1_checkpoint"]) != base["parent_checkpoint_sha256"]["w1"]:
        raise ValueError("C1 checkpoint changed")
    protocol = pool.manifest["protocol"]
    if digest(ROOT/protocol["model"]["path"]/"model.safetensors") != protocol["model"]["sha256"]:
        raise ValueError("Base weights changed")
    model, _ = load_model("native", protocol)
    probe = SharedPrefixValue(model, **checkpoint["architecture"]).to("cuda").eval()
    probe.load_addon(checkpoint["addon"])
    audit_values(probe, pool, work)
    probe.close()
    print(json.dumps({"phase": "values_complete", "decisions": len(pool.rows)}), flush=True)


def prepare_jobs(config, destination, work):
    import h5py
    from vla_rl_mt.joint_value import BoundaryPool
    from vla_rl_mt.policy_update import PolicyPairs, label_array

    seal = json.loads((work/"pool-seal.json").read_text())
    pool = BoundaryPool(destination, seal["manifest_sha256"])
    label_path = work/"value-label-audit.json"
    data = PolicyPairs(pool, mode="c1_weighted", label_path=label_path, steps=config["steps"],
                       batch_size=config["global_batch"], seed=config["seed"])
    draws, labels = data.draws, data.labels
    eligible = []
    for index in range(len(draws)):
        generator = torch.Generator().manual_seed(config["seed"]+2_000_000+index)
        torch.randn((32, 32), generator=generator)
        torch.rand((), generator=generator)
        eligible.append(float(torch.rand((), generator=generator)) > .1)
    eligible = torch.tensor(eligible)
    instructions = {}
    for entry in pool.manifest["episodes"]:
        with h5py.File(destination/entry["file"]) as episode:
            instructions[entry["file"]] = str(episode.attrs["instruction"])
    shuffled = matched_permutation(labels, pool.rows, instructions, draws, eligible, config["seed"])
    changed = labels[draws] != shuffled[draws]
    if int((changed & eligible).sum()) < 10:
        raise ValueError("Too few shuffled scheduled labels to test their effect")
    retained_counts = Counter(int(i) for i, ok in zip(draws, eligible, strict=True) if ok)
    matched = torch.zeros_like(labels)
    tasks = sorted({r["task"] for r in pool.rows})
    for task in tasks:
        target = sum(int(labels[int(i)]) for i, ok in zip(draws, eligible, strict=True)
                     if ok and pool.rows[int(i)]["task"] == task)
        candidates = [i for i, r in enumerate(pool.rows) if r["split"] == "train" and r["task"] == task and r["success"]]
        # Seeded ordering avoids taking an arbitrary temporal prefix as the only control.
        generator = torch.Generator().manual_seed(config["seed"]+505)
        candidates = [candidates[int(i)] for i in torch.randperm(len(candidates), generator=generator)]
        matched[list(exact_subset(candidates, retained_counts, target))] = True
    variants = {"real": labels, "shuffled": shuffled, "success_matched": matched}
    source = json.loads(label_path.read_text())
    audit = {"changed_decisions": int((labels != shuffled).sum()), "changed_draws": int(changed.sum()),
             "changed_eligible_draws": int((changed & eligible).sum()), "groups": {}, "complete": False}
    exposures = {}
    for name, bits in variants.items():
        report = copy.deepcopy(source)
        mapping = {(r["episode"], r["observation_index"]): bool(bits[i]) for i, r in enumerate(pool.rows)}
        for row in report["rows"]:
            if row["use_for_loss"]:
                row["positive"] = mapping[row["episode"], row["observation_index"]]
        report["label_control"] = name
        path = work/f"labels-{name}.json"
        write_json(path, report)
        if not torch.equal(label_array(pool, path), bits):
            raise ValueError("Worker label reload differs")
        counters = Counter()
        for index, ok in zip(draws, eligible, strict=True):
            row = pool.rows[int(index)]
            if ok and bits[int(index)]:
                counters[row["task"], instructions[row["episode"]], row["success"]] += 1
        exposures[name] = counters
        audit["groups"][name] = {"positive_train_decisions": sum(int(bits[i]) for i, r in enumerate(pool.rows) if r["split"] == "train"),
            "retained_draws": sum(counters.values()),
            "retained_by_task_instruction_outcome": [{"task": key[0], "instruction": key[1], "success": key[2], "count": count}
                                                      for key, count in sorted(counters.items())],
            "retained_action_steps": sum(pool.rows[int(i)]["next_step"]-pool.rows[int(i)]["step"]
                                         for i, ok in zip(draws, eligible, strict=True) if ok and bits[int(i)])}
    if exposures["real"] != exposures["shuffled"]:
        raise ValueError("Shuffling changed actual task/instruction/outcome exposure")
    if audit["groups"]["real"]["retained_action_steps"] != audit["groups"]["shuffled"]["retained_action_steps"]:
        raise ValueError("Shuffling changed executed action exposure")
    for task in tasks:
        if sum(v for k, v in exposures["real"].items() if k[0] == task) != sum(v for k, v in exposures["success_matched"].items() if k[0] == task):
            raise ValueError("Success reference exposure mismatch")
    jobs, smoke, evaluation = [], [], []
    updates = ROOT/config["training_output"]
    for name in config["groups"]:
        label = work/f"labels-{name}.json"
        mode = "bc" if name == "bc" else "c1_weighted"
        overrides = ["actor.model.model_type=mt_native_expert_policy", f"actor.model.mode={mode}",
            f"data.pool_path={destination}", f"data.pool_manifest_sha256={seal['manifest_sha256']}",
            f"+data.label_path={label if name != 'bc' else 'none'}", f"+data.label_sha256={digest(label) if name != 'bc' else 'none'}",
            "actor.micro_batch_size=2", "actor.global_batch_size=4", "actor.seed=0", "actor.optim.lr=0.0003",
            "+actor.model.adapter_lr=0.0003", "+actor.model.expert_lr=1e-6", "+actor.model.update_scope=expert_adapter",
            "+actor.optim.use_fp32_master_params=true", "+actor.fsdp_config.save_full_model_weights=false",
            "runner.max_steps=128", "runner.save_interval=128", "runner.val_check_interval=-1",
            "+runner.eval_export_steps=[]", f"actor.group_name=label_{name}"]
        job = {"name": f"label_{name}", "entry": "scripts/train_method_joint.py", "config": "joint_value",
               "extension": "vla_rl_mt.bc_stability", "overrides": overrides}
        jobs.append(job)
        smoke.append({**job, "overrides": [s.replace("runner.max_steps=128", "runner.max_steps=4").replace(
            "runner.save_interval=128", "runner.save_interval=4") for s in overrides]+["+runner.trace_final_resumable=false"]})
        actor = updates/job["name"]/"checkpoints/global_step_128/actor"
        evaluation.append({"name": job["name"], "training_job": job["name"], "step": 128,
                           "checkpoint": str(actor/"addon.pt"), "ready_marker": str(actor/"evaluation-ready.json"), "prompt": "plain"})
    audit["complete"] = True
    write_json(work/"exposure-audit.json", audit)
    write_json(work/"jobs.json", {"jobs": jobs, "protocol_sha256": digest(PROTOCOL)})
    write_json(work/"smoke-jobs.json", {"jobs": smoke, "protocol_sha256": digest(PROTOCOL)})
    write_json(work/"evaluation.json", {"variants": evaluation, "protocol_sha256": digest(PROTOCOL)})
    print(json.dumps(audit), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("pool", "values", "jobs"))
    args = parser.parse_args()
    config = json.loads(PROTOCOL.read_text())
    work, destination = ROOT/config["work"], ROOT/config["mixed_pool"]
    work.mkdir(parents=True, exist_ok=True)
    if args.phase == "pool":
        freeze(config, work, destination)
    else:
        {"values": evaluate_values, "jobs": prepare_jobs}[args.phase](config, destination, work)


if __name__ == "__main__":
    main()
