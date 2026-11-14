"""Certify nested BC views, controlled updates and every declared curve point."""

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import h5py
import torch
from summarize_method_expert import equal_nested
from summarize_method_m4 import paired

from vla_rl_mt.expert_policy import state_digest
from vla_rl_mt.joint_value import digest, draw_indices, write_json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"bootstrap"))
from cache_method_m1 import validate_pool
from prepare_data_coverage import select_successes


def read(path):
    return json.loads(Path(path).read_text())


def sampling(pool, manifest):
    rows = []
    for entry in manifest["episodes"]:
        with h5py.File(pool/entry["file"]) as episode:
            rows.extend({"episode": entry["file"], "observation_index": i,
                         "task": entry["task"], "success": entry["success"], "split": entry["split"]}
                        for i in range(len(episode["observation_steps"])-1))
    indices = [i for i, row in enumerate(rows) if row["split"] == "train" and row["success"]]
    draws = draw_indices(rows, indices, 512, 101).tolist()
    keys = [(rows[i]["episode"], rows[i]["observation_index"]) for i in draws]
    task_sequence = [rows[i]["task"] for i in draws]
    return {"sha256": hashlib.sha256(json.dumps(keys).encode()).hexdigest(),
        "task_sequence": task_sequence, "task_exposure": dict(Counter(task_sequence)),
        "unique_chunks": len(set(keys)), "unique_episodes": len({key[0] for key in keys})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--updates", type=Path)
    parser.add_argument("--smoke", type=Path)
    parser.add_argument("--evaluation", type=Path, nargs="*", default=[])
    parser.add_argument("--fit-small", type=Path)
    parser.add_argument("--fit-large", type=Path)
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--training-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = read(ROOT/"configs/multitask/data_coverage.json")
    setting = config["coverage"]
    base = read(ROOT/"configs/multitask/method_m4.json")
    raw = read(args.collection/"manifest.json")
    if len(raw["episodes"]) != 192 or digest(ROOT/config["parent_training_protocol"]) != config["parent_training_sha256"]:
        raise ValueError("Collection/parent protocol changed")
    raw_audit = validate_pool(args.collection, raw)
    for source, expected in read(args.collection/"plan.json")["sources"].items():
        if digest(ROOT/source) != expected:
            raise ValueError(f"Collection source changed: {source}")
    original = read(ROOT/setting["validation_source"]/"manifest.json")
    expected_val = {r["file"]: r["sha256"] for r in original["episodes"] if r["split"] == "val"}
    result = {"scope": "Nested successful-BC data coverage, fixed native Motus behavior, one actor seed/two dev tasks",
              "raw_data_audit": raw_audit, "data": {}, "training": {}, "points": {},
              "rows": [], "successes": {}, "paired_vs_w1": {}, "large_vs_small": {},
              "final_test_used": False, "rollouts_return_to_training": False}
    manifests = {}
    train_sets = {}
    for size, count in [("small", 4), ("large", 12)]:
        pool = args.views/size
        manifest = read(pool/"manifest.json")
        manifests[size] = manifest
        audit = validate_pool(pool, manifest)
        selected = select_successes(raw["episodes"], config["tasks"], count)
        expected = {r["file"]: r["sha256"] for r in selected}
        train = {r["file"]: r["sha256"] for r in manifest["episodes"] if r["split"] == "train"}
        val = {r["file"]: r["sha256"] for r in manifest["episodes"] if r["split"] == "val"}
        if train != expected or val != expected_val or not manifest["bc_only"]:
            raise ValueError("Selected training or fixed validation changed")
        train_sets[size] = set(train)
        sample = sampling(pool, manifest)
        result["data"][size] = {"audit": audit, "manifest_sha256": digest(pool/"manifest.json"),
                                "coverage": manifest["coverage"], "sampling": sample}
    if not train_sets["small"] < train_sets["large"]:
        raise ValueError("Training views are not nested")
    if result["data"]["small"]["sampling"]["task_sequence"] != result["data"]["large"]["sampling"]["task_sequence"]:
        raise ValueError("Task exposure differs; lock a common schedule before training")
    result["data_audit_passed"] = True
    if args.data_only:
        write_json(args.output, result)
        print(json.dumps({"data_audit_passed": True}))
        return
    if args.updates is None or args.smoke is None:
        parser.error("Training audits need --updates and --smoke")
    execution = read(args.updates/"execution.json")
    smoke = read(args.smoke/"execution.json")
    names = {"coverage_small", "coverage_large"}
    for queue in [execution, smoke]:
        if not queue["complete"] or {j["name"] for j in queue["jobs"]} != names or any(j.get("exit_code") != 0 for j in queue["jobs"]):
            raise ValueError("Both worker queues must exit successfully")
    parent = torch.load(ROOT/base["w1_checkpoint"], map_location="cpu", weights_only=True)
    for size in ("small", "large"):
        name = f"coverage_{size}"
        directory = args.updates/name
        identity = read(directory/"live-data.json")
        if (identity["pool_manifest_sha256"] != result["data"][size]["manifest_sha256"]
                or identity["sample_sequence_sha256"] != result["data"][size]["sampling"]["sha256"]
                or identity["parent_sha256"] != base["parent_checkpoint_sha256"]["w1"]
                or identity["mode"] != "bc" or identity["update_scope"] != "expert_adapter"
                or identity["expert_lr"] != 1e-6 or identity["lr"] != 3e-4
                or identity["expert_source_sha256"] != digest(ROOT/"src/vla_rl_mt/expert_policy.py")
                or identity["source_sha256"] != digest(ROOT/"src/vla_rl_mt/policy_update.py")
                or identity["trace_source_sha256"] != digest(ROOT/"src/vla_rl_mt/bc_stability.py")):
            raise ValueError("Wrong actor/data/update identity")
        audit = read(directory/"parameter-audit.json")
        if (not audit["optimizer_coverage_exact"] or audit["optimizer"] != "FP32MasterAdamW"
                or sorted((g["parameters"], g["lr"]) for g in audit["optimizer_groups"]) != [(131072, 3e-4), (430098464, 1e-6)]):
            raise ValueError("Optimizer scope differs")
        metrics = [json.loads(line) for line in (directory/"joint-metrics.jsonl").read_text().splitlines()]
        if [r["step"] for r in metrics] != list(range(1, 129)):
            raise ValueError("Incomplete training curve")
        smoke_metrics = [json.loads(line) for line in (args.smoke/name/"joint-metrics.jsonl").read_text().splitlines()]
        if [r["sft_loss"] for r in metrics[:4]] != [r["sft_loss"] for r in smoke_metrics]:
            raise ValueError("New-data smoke and main update prefix differ")
        for row in metrics:
            g = row["gradients"]
            if (g["frozen_vlm"]["gradient_tensors"] or g["frozen_vlm"]["trainable"] or g["value_head"]["gradient_tensors"]
                    or not all(x["finite"] for x in g.values()) or not math.isfinite(row["sft_loss"])):
                raise ValueError("Invalid gradient route")
        for step in setting["evaluation_steps"]:
            path = directory/f"checkpoints/global_step_{step}/actor/addon.pt"
            c = torch.load(path, map_location="cpu", weights_only=True)
            marker = read(path.parent/"evaluation-ready.json")
            sha = digest(path)
            if c["identity"] != identity or c["step"] != step or marker["checkpoint_sha256"] != sha or marker["resumable"] != (step == 128):
                raise ValueError("Checkpoint/ready identity differs")
            if any(c["exposure"][key] != step*4 for key in ["samples", "success", "positive"]):
                raise ValueError("Actual successful BC exposure differs")
            expert_sha = state_digest(c["expert"])
            if expert_sha != c["expert_sha256"] or expert_sha == audit["initial_expert_sha256"]:
                raise ValueError("Expert checksum/update failed")
            if not equal_nested({"value_head": c["addon"]["value_head"]}, {"value_head": parent["addon"]["value_head"]}):
                raise ValueError("Frozen critic changed")
            dcp = list(path.parent.rglob("*.distcp"))
            if bool(dcp) != (step == 128) or (path.parent/"evaluation-only.json").exists() != (step != 128):
                raise ValueError("Wrong light/full checkpoint kind")
            result["points"][f"{name}_step{step}"] = {"sha256": sha, "step": step, "size": size, "exposure": c["exposure"]}
        result["training"][name] = {"identity": identity, "peak_allocated_gib": max(r["peak_allocated_gib"] for r in metrics)}
    result["training_audit_passed"] = True
    if args.training_only:
        write_json(args.output, result)
        print(json.dumps({"training_audit_passed": True}))
        return
    if not args.evaluation or not args.fit_small or not args.fit_large:
        parser.error("Final audit requires all evaluation/fit reports")
    old = read(ROOT/".runtime/runs/method/m2-joint-v1/comparison.json")
    reference = {(r["task"], r["seed"]): r for r in old["rows"] if r["variant"] == "w1"}
    variants = {"w1_plain": reference}
    for path in args.evaluation:
        report = read(path)
        if not report["complete"] or report["final_test_used"] or report["rollouts_return_to_training"]:
            raise ValueError("Incomplete or invalid evaluation")
        for variant in report["variants"]:
            name = variant["name"]
            selected = [r for r in report["rows"] if r["variant"] == name]
            mapping = {(r["task"], r["seed"]): r for r in selected}
            if name in variants or len(selected) != 24 or set(mapping) != set(reference):
                raise ValueError("Evaluation cohort differs")
            for key, row in mapping.items():
                if (row["status"] != "complete" or row["observed_init_fingerprint"] != reference[key]["observed_init_fingerprint"]
                        or row["noise_seed"] != reference[key]["noise_seed"] or row["deploy_prompt"] != "plain"
                        or row["checkpoint_sha256"] != result["points"][name]["sha256"]):
                    raise ValueError("Evaluation state/noise/checkpoint differs")
            variants[name] = mapping
            result["rows"].extend(selected)
    if set(variants) != set(result["points"])|{"w1_plain"}:
        raise ValueError("Missing predefined curve point")
    tasks = config["tasks"]
    for name, mapping in variants.items():
        result["successes"][name] = {task: sum(r["success"] for (t, _), r in mapping.items() if t == task) for task in tasks}
        result["paired_vs_w1"][name] = paired(mapping, reference, tasks)
    for step in setting["evaluation_steps"]:
        result["large_vs_small"][step] = paired(variants[f"coverage_large_step{step}"], variants[f"coverage_small_step{step}"], tasks)
    result["fit"] = {}
    validation_reference = None
    for size, path in [("small", args.fit_small), ("large", args.fit_large)]:
        fit = read(path)
        expected = {f"coverage_{size}_step{step}": result["points"][f"coverage_{size}_step{step}"]["sha256"] for step in setting["evaluation_steps"]}
        expected["w1"] = base["parent_checkpoint_sha256"]["w1"]
        if not fit["complete"] or fit["checkpoint_sha256"] != expected or fit["pool_manifest_sha256"] != result["data"][size]["manifest_sha256"]:
            raise ValueError("Action-fit data/checkpoint differs")
        val = fit["metrics"]["w1"]["val"]
        if validation_reference is not None and val != validation_reference:
            raise ValueError("W1 validation fit changed across data-size groups")
        validation_reference = val
        result["fit"][size] = fit["metrics"]
    result.update(audit_passed=True, new_rollouts=144)
    write_json(args.output, result)
    print(json.dumps({"successes": result["successes"], "audit_passed": True}, indent=2))


if __name__ == "__main__":
    main()
