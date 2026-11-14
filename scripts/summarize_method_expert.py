"""Audit matched expert-update controls and their complete development rollouts."""

import argparse
import json
import math
from pathlib import Path

import torch
from summarize_method_m4 import paired

from vla_rl_mt.expert_policy import state_digest
from vla_rl_mt.joint_value import digest, write_json

ROOT = Path(__file__).resolve().parents[1]


def equal_nested(first, second):
    return all(torch.equal(value, second[group][key]) for group, values in first.items() for key, value in values.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, nargs="*", default=[])
    parser.add_argument("--training-only", action="store_true", help="Audit updates/resume while rollouts are still running")
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.training_only and not args.evaluation:
        parser.error("Complete evaluation reports are required unless --training-only is set")
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads((ROOT/"configs/multitask/method_m4_expert.json").read_text())
    base = json.loads((ROOT/protocol["base_protocol"]).read_text())
    if digest(ROOT/protocol["base_protocol"]) != protocol["base_protocol_sha256"]:
        raise ValueError("Parent protocol changed")
    parent = torch.load(ROOT/base["w1_checkpoint"], map_location="cpu", weights_only=True)
    execution = json.loads((args.updates/"execution.json").read_text())
    if not execution["complete"] or any(row.get("exit_code") != 0 for row in execution["jobs"]):
        raise ValueError("All six policy workers must complete successfully")
    result = {"scope": "Matched native action-expert stop-update controls, one actor seed, two development tasks",
              "training": {}, "successes": {}, "paired": {}, "rows": [],
              "final_test_used": False, "rollouts_return_to_training": False}
    for job in execution["jobs"]:
        name = job["name"]
        directory = args.updates/name
        path = directory/f"checkpoints/global_step_{protocol['actor_steps']}/actor/addon.pt"
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
        identity = checkpoint["identity"]
        scope, mode = identity["update_scope"], identity["mode"]
        if (name != f"{scope}-{mode}" or identity != json.loads((directory/"live-data.json").read_text())
                or identity["parent_sha256"] != base["parent_checkpoint_sha256"]["w1"]
                or checkpoint["step"] != protocol["actor_steps"]
                or identity["pool_manifest_sha256"] != base["pool_manifest_sha256"]):
            raise ValueError("Wrong worker identity")
        audit = json.loads((directory/"parameter-audit.json").read_text())
        expert_hash = state_digest(checkpoint["expert"])
        if expert_hash != checkpoint["expert_sha256"]:
            raise ValueError("Exported expert checksum differs")
        changed = expert_hash != audit["initial_expert_sha256"]
        if changed != (scope == "expert_adapter"):
            raise ValueError("Expert stop-update control or real update failed")
        if any(not torch.equal(value, parent["addon"]["value_head"][key])
               for key, value in checkpoint["addon"]["value_head"].items()):
            raise ValueError("Frozen value head changed")
        if all(torch.equal(value, parent["addon"]["adapter"][key])
               for key, value in checkpoint["addon"]["adapter"].items()):
            raise ValueError("Adapter did not update")
        groups = audit["groups"]
        expected = groups["adapter"]["parameters"]+(groups["action_expert"]["parameters"] if changed else 0)
        if (not audit["optimizer_coverage_exact"] or audit["optimizer"] != "FP32MasterAdamW"
                or sum(group["parameters"] for group in audit["optimizer_groups"]) != expected):
            raise ValueError("Optimizer coverage differs")
        lrs = sorted(group["lr"] for group in audit["optimizer_groups"])
        expected_lrs = sorted([protocol["adapter_lr"]]+([protocol["expert_lr"]] if changed else []))
        if len(lrs) != len(expected_lrs) or any(not math.isclose(a, b, rel_tol=1e-12) for a, b in zip(lrs, expected_lrs, strict=True)):
            raise ValueError("Learning rates differ from protocol")
        metrics = [json.loads(line) for line in (directory/"joint-metrics.jsonl").read_text().splitlines()]
        if [row["step"] for row in metrics] != list(range(1, protocol["actor_steps"]+1)):
            raise ValueError("Missing updates")
        for row in metrics:
            if (row["gradients"]["frozen_vlm"]["trainable"]
                    or row["gradients"]["frozen_vlm"]["gradient_tensors"]
                    or row["gradients"]["value_head"]["gradient_tensors"]
                    or not all(group["finite"] for group in row["gradients"].values())):
                raise ValueError("Wrong gradient route")
        result["training"][name] = {"identity": identity, "exposure": checkpoint["exposure"],
            "initial_loss": metrics[0]["sft_loss"],
            "checkpoint_sha256": digest(path), "expert_changed": changed,
            "expert_parameters": groups["action_expert"]["parameters"], "optimized_parameters": expected,
            "peak_allocated_gib": max(row["peak_allocated_gib"] for row in metrics)}
    expected_names = {f"{scope}-{mode}" for scope in protocol["update_scopes"] for mode in protocol["policy_groups"]}
    if set(result["training"]) != expected_names:
        raise ValueError("Missing scope controls")
    for mode in protocol["policy_groups"]:
        a, b = [result["training"][f"{scope}-{mode}"] for scope in protocol["update_scopes"]]
        if (a["exposure"] != b["exposure"] or a["identity"]["sample_sequence_sha256"] != b["identity"]["sample_sequence_sha256"]
                or a["identity"]["label_sha256"] != b["identity"]["label_sha256"] or a["initial_loss"] != b["initial_loss"]):
            raise ValueError("Scope controls changed training samples, labels or exposure")
    smoke = json.loads((args.smoke/"execution.json").read_text())
    if not smoke["complete"] or any(row.get("exit_code") != 0 for row in smoke["jobs"]):
        raise ValueError("Official worker/resume smoke incomplete")
    a, b = [torch.load(args.smoke/name/"checkpoints/global_step_4/actor/addon.pt", weights_only=True, map_location="cpu")
            for name in ("uninterrupted", "resumed")]
    if (not equal_nested(a["addon"], b["addon"]) or state_digest(a["expert"]) != state_digest(b["expert"])
            or a["exposure"] != b["exposure"]):
        raise ValueError("Official DCP resume did not replay exactly")
    result["official_resume_exact"] = True
    del a, b
    if args.training_only:
        result["training_audit_passed"] = True
        result["evaluation_status"] = "pending, not audited"
        write_json(args.output, result)
        print(json.dumps({"training_audit_passed": True, "groups": len(result["training"])}))
        return
    reference = json.loads((ROOT/".runtime/runs/method/m2-joint-v1/comparison.json").read_text())
    reference = {(row["task"], row["seed"]): row for row in reference["rows"] if row["variant"] == "w1"}
    variants = {"w1_plain": reference}
    for path in args.evaluation:
        report = json.loads(path.read_text())
        if not report["complete"] or report["final_test_used"] or report["rollouts_return_to_training"]:
            raise ValueError("Incomplete or invalid evaluation")
        for variant in report["variants"]:
            name = variant["name"]
            if name in variants:
                raise ValueError("Duplicate evaluation variant")
            rows = [row for row in report["rows"] if row["variant"] == name]
            mapping = {(row["task"], row["seed"]): row for row in rows}
            if len(rows) != 24 or set(mapping) != set(reference):
                raise ValueError("Evaluation scene set differs")
            for key, row in mapping.items():
                if (row["status"] != "complete" or row["observed_init_fingerprint"] != reference[key]["observed_init_fingerprint"]
                        or row["noise_seed"] != reference[key]["noise_seed"] or row["deploy_prompt"] != "plain"
                        or row["checkpoint_sha256"] != result["training"][name]["checkpoint_sha256"]):
                    raise ValueError("Evaluation state/noise/checkpoint differs")
            variants[name] = mapping
            result["rows"].extend(rows)
    if set(variants) != expected_names|{"w1_plain"}:
        raise ValueError("Missing evaluated policy")
    tasks = sorted({key[0] for key in reference})
    result["successes"] = {name: {task: sum(row["success"] for key, row in rows.items() if key[0] == task)
        for task in tasks} for name, rows in variants.items()}
    for mode in protocol["policy_groups"]:
        new, old = f"expert_adapter-{mode}", f"adapter_only-{mode}"
        result["paired"][f"{new}-vs-{old}"] = paired(variants[new], variants[old], tasks)
    for scope in protocol["update_scopes"]:
        new, old = f"{scope}-c2_weighted", f"{scope}-c1_weighted"
        result["paired"][f"{new}-vs-{old}"] = paired(variants[new], variants[old], tasks)
    for key in ("w1", "w2"):
        if digest(ROOT/base[f"{key}_checkpoint"]) != base["parent_checkpoint_sha256"][key]:
            raise ValueError("Original actor/critic changed")
    result["audit_passed"] = True
    write_json(args.output, result)
    print(json.dumps({"successes": result["successes"], "audit_passed": True}, indent=2))


if __name__ == "__main__":
    main()
