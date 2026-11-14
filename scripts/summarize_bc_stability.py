"""Audit the two locked BC traces, all curve points, and endpoint reuse."""

import argparse
import json
import math
from pathlib import Path

import torch
from summarize_method_expert import equal_nested
from summarize_method_m4 import paired

from vla_rl_mt.expert_policy import state_digest
from vla_rl_mt.joint_value import digest, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, nargs="*", default=[])
    parser.add_argument("--fit", type=Path)
    parser.add_argument("--training-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.training_only and (not args.evaluation or not args.fit):
        parser.error("Completed rollout and action-fit reports are required")
    protocol_path = ROOT/"configs/multitask/bc_stability.json"
    protocol = json.loads(protocol_path.read_text())
    if digest(ROOT/protocol["parent_protocol"]) != protocol["parent_protocol_sha256"]:
        raise ValueError("Parent protocol changed")
    base = json.loads((ROOT/"configs/multitask/method_m4.json").read_text())
    parent = torch.load(ROOT/base["w1_checkpoint"], weights_only=True, map_location="cpu")
    old_directory = ROOT/".runtime/runs/method/m4-expert-policy-v1/expert_adapter-bc"
    old_path = old_directory/"checkpoints/global_step_128/actor/addon.pt"
    old = torch.load(old_path, weights_only=True, map_location="cpu")
    execution = json.loads((args.updates/"execution.json").read_text())
    expected_jobs = {"expert_lr_1e5": 1e-5, "expert_lr_1e6": 1e-6}
    if (not execution["complete"] or {job["name"] for job in execution["jobs"]} != set(expected_jobs)
            or any(job.get("exit_code") != 0 for job in execution["jobs"])):
        raise ValueError("Both trace workers must complete successfully")
    result = {"scope": "Two locked native BC traces; one actor seed, two development tasks",
              "protocol_sha256": digest(protocol_path), "training": {}, "points": {},
              "successes": {}, "paired_vs_w1": {}, "rows": [],
              "final_test_used": False, "rollouts_return_to_training": False}
    reference_metrics = [json.loads(line) for line in (old_directory/"joint-metrics.jsonl").read_text().splitlines()]
    for name, lr in expected_jobs.items():
        directory = args.updates/name
        identity = json.loads((directory/"live-data.json").read_text())
        for key, value in old["identity"].items():
            expected = lr if key == "expert_lr" else value
            if identity.get(key) != expected:
                raise ValueError(f"Changed training identity: {name}/{key}")
        if identity["trace_source_sha256"] != digest(ROOT/"src/vla_rl_mt/bc_stability.py"):
            raise ValueError("Trace source changed during this batch")
        if identity["evaluation_export_steps"] != [32, 64] or not identity["final_resumable"]:
            raise ValueError("Changed export schedule")
        audit = json.loads((directory/"parameter-audit.json").read_text())
        groups = audit["optimizer_groups"]
        if (not audit["optimizer_coverage_exact"] or audit["optimizer"] != "FP32MasterAdamW"
                or sum(group["parameters"] for group in groups) != 430229536
                or sorted(group["parameters"] for group in groups) != [131072, 430098464]
                or any(not math.isclose(group["lr"], lr if group["parameters"] == 430098464
                                        else protocol["adapter_lr"], rel_tol=1e-12) for group in groups)):
            raise ValueError("Wrong optimizer scope or learning rate")
        metrics = [json.loads(line) for line in (directory/"joint-metrics.jsonl").read_text().splitlines()]
        if [row["step"] for row in metrics] != list(range(1, 129)):
            raise ValueError("Missing training steps")
        for row in metrics:
            gradient = row["gradients"]
            if (gradient["frozen_vlm"]["gradient_tensors"] or gradient["frozen_vlm"]["trainable"]
                    or gradient["value_head"]["gradient_tensors"]
                    or not all(group["finite"] for group in gradient.values())
                    or not math.isfinite(row["sft_loss"])):
                raise ValueError("Invalid training gradients")
        if metrics[0]["sft_loss"] != reference_metrics[0]["sft_loss"]:
            raise ValueError("Initial loss differs from the reference")
        for step in protocol["evaluation_steps"]:
            path = directory/f"checkpoints/global_step_{step}/actor/addon.pt"
            checkpoint = torch.load(path, weights_only=True, map_location="cpu")
            marker = json.loads((path.parent/"evaluation-ready.json").read_text())
            sha = digest(path)
            if (checkpoint["step"] != step or checkpoint["identity"] != identity
                    or marker["checkpoint_sha256"] != sha or marker["step"] != step
                    or marker["resumable"] != (step == 128)):
                raise ValueError("Wrong checkpoint identity or ready marker")
            files = list(path.parent.rglob("*"))
            has_dcp = any(item.name.endswith(".distcp") for item in files)
            if has_dcp != (step == 128) or (path.parent/"evaluation-only.json").exists() != (step != 128):
                raise ValueError("Intermediate/full checkpoint kinds differ from protocol")
            expert_hash = state_digest(checkpoint["expert"])
            if expert_hash != checkpoint["expert_sha256"] or expert_hash == audit["initial_expert_sha256"]:
                raise ValueError("Expert export checksum or update failed")
            if any(not torch.equal(value, parent["addon"]["value_head"][key])
                   for key, value in checkpoint["addon"]["value_head"].items()):
                raise ValueError("Frozen head changed")
            if all(torch.equal(value, parent["addon"]["adapter"][key])
                   for key, value in checkpoint["addon"]["adapter"].items()):
                raise ValueError("Adapter did not update")
            exposure = checkpoint["exposure"]
            if any(exposure[key] != step*4 for key in ("samples", "success", "positive")):
                raise ValueError("BC exposure differs")
            result["points"][f"{name}_step{step}"] = {"step": step, "expert_lr": lr,
                "checkpoint_sha256": sha, "expert_sha256": expert_hash, "exposure": exposure}
            if name == "expert_lr_1e5" and step == 128:
                if (not equal_nested(checkpoint["addon"], old["addon"])
                        or any(not torch.equal(value, old["expert"][key]) for key, value in checkpoint["expert"].items())
                        or checkpoint["expert"].keys() != old["expert"].keys() or exposure != old["exposure"]):
                    raise ValueError("Reference final tensors do not replay; old rollouts cannot be reused")
                result["endpoint_replay_exact"] = True
        result["training"][name] = {"identity": identity, "initial_loss": metrics[0]["sft_loss"],
            "peak_allocated_gib": max(row["peak_allocated_gib"] for row in metrics)}
    for step in protocol["evaluation_steps"]:
        if result["points"][f"expert_lr_1e5_step{step}"]["exposure"] != result["points"][f"expert_lr_1e6_step{step}"]["exposure"]:
            raise ValueError("Learning-rate controls changed exposure")
    result["training_audit_passed"] = True
    if args.training_only:
        write_json(args.output, result)
        print(json.dumps({"training_audit_passed": True, "endpoint_replay_exact": True}))
        return
    reference = json.loads((ROOT/".runtime/runs/method/m2-joint-v1/comparison.json").read_text())
    reference = {(row["task"], row["seed"]): row for row in reference["rows"] if row["variant"] == "w1"}
    old_report_path = ROOT/".runtime/runs/method/m4-expert-work-v1/final-comparison.json"
    old_report = json.loads(old_report_path.read_text())
    if not old_report["audit_passed"] or not result["endpoint_replay_exact"]:
        raise ValueError("Reference reuse has not passed audit")
    old_name = "expert_adapter-bc"
    if digest(old_path) != old_report["training"][old_name]["checkpoint_sha256"]:
        raise ValueError("Reused endpoint checkpoint changed")
    reused_name = "expert_lr_1e5_step128"
    variants = {"w1_plain": list(reference.values()), reused_name: [dict(row) for row in old_report["rows"] if row["variant"] == old_name]}
    source_hashes = {reused_name: digest(old_path)}
    for path in args.evaluation:
        report = json.loads(path.read_text())
        if not report["complete"] or report["final_test_used"] or report["rollouts_return_to_training"]:
            raise ValueError("Invalid evaluation report")
        for variant in report["variants"]:
            name = variant["name"]
            if name in variants or name not in result["points"]:
                raise ValueError("Unexpected or duplicate evaluation point")
            variants[name] = [row for row in report["rows"] if row["variant"] == name]
            source_hashes[name] = result["points"][name]["checkpoint_sha256"]
    if set(variants) != set(result["points"])|{"w1_plain"}:
        raise ValueError("Missing curve point")
    tasks = sorted({task for task, _ in reference})
    for name, rows in variants.items():
        mapping = {(row["task"], row["seed"]): row for row in rows}
        if len(rows) != 24 or set(mapping) != set(reference):
            raise ValueError("Incomplete fixed scene cohort")
        for key, row in mapping.items():
            if (row["status"] != "complete" or row["noise_seed"] != reference[key]["noise_seed"]
                    or row["observed_init_fingerprint"] != reference[key]["observed_init_fingerprint"]):
                raise ValueError("Evaluation scenes or noise differ")
            if name != "w1_plain" and (row["checkpoint_sha256"] != source_hashes[name] or row["deploy_prompt"] != "plain"):
                raise ValueError("Wrong evaluated weights or prompt")
        result["successes"][name] = {task: sum(row["success"] for key, row in mapping.items() if key[0] == task) for task in tasks}
        result["paired_vs_w1"][name] = paired(mapping, reference, tasks)
        result["rows"].extend({**row, "curve_point": name, "reused": name in ("w1_plain", reused_name)} for row in rows)
    fit = json.loads(args.fit.read_text())
    if not fit["complete"] or set(fit["metrics"]) != set(result["points"])|{"w1"}:
        raise ValueError("Incomplete action-fit curve")
    if fit["final_test_used"] or fit["rollouts_return_to_training"]:
        raise ValueError("Invalid action-fit split")
    expected_fit_hashes = {name: point["checkpoint_sha256"] for name, point in result["points"].items()}
    expected_fit_hashes["w1"] = base["parent_checkpoint_sha256"]["w1"]
    if fit["checkpoint_sha256"] != expected_fit_hashes:
        raise ValueError("Action-fit curve evaluated different weights")
    result.update(fit=fit["metrics"], fit_report_sha256=digest(args.fit), new_rollouts=120,
                  reused_endpoint_report_sha256=digest(old_report_path), audit_passed=True)
    write_json(args.output, result)
    print(json.dumps({"successes": result["successes"], "audit_passed": True}, indent=2))


if __name__ == "__main__":
    main()
