"""Evaluate a declared set of native policy addons on the full fixed M1 val pool."""

import argparse
import copy
import gc
import json
import time
from pathlib import Path

import torch

from vla_rl_mt.joint_value import digest
from vla_rl_mt.shared_value import SharedPrefixValue

ROOT = Path(__file__).resolve().parents[1]


class PromptPolicy:
    def __init__(self, policy, prompt):
        self.policy, self.prompt = policy, prompt

    def predict_action_batch(self, observation, **kwargs):
        raw = copy.copy(observation)
        if self.prompt != "plain":
            raw["task_descriptions"] = [f"{text}\nAdvantage: {self.prompt}" for text in raw["task_descriptions"]]
        return self.policy.predict_action_batch(raw, **kwargs)


def main():
    from evaluate_multitask import evaluate_case, load_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expert", action="store_true", help="Load exported native action-expert scope checkpoints")
    parser.add_argument("--training-execution", type=Path,
                        help="Wait for this task-owned queue to finish each checkpoint before evaluating")
    args = parser.parse_args()
    output = args.output.resolve()
    training_execution = args.training_execution.resolve() if args.training_execution else None
    output.mkdir(parents=True, exist_ok=False)
    schedule = json.loads(args.plan.read_text())
    plan = json.loads((ROOT/"configs/multitask/method_m4.json").read_text())
    pool = ROOT/plan["pool"]
    if digest(pool/"manifest.json") != plan["pool_manifest_sha256"]:
        raise ValueError("Evaluation pool changed")
    manifest = json.loads((pool/"manifest.json").read_text())
    protocol = manifest["protocol"]
    variants = schedule["variants"][args.shard::args.num_shards]
    cases = [{key: entry[key] for key in ("task", "seed", "seed_index", "noise_seed")}
             for entry in manifest["episodes"] if entry["split"] == "val"]
    reference = json.loads((ROOT/".runtime/runs/method/m2-joint-v1/comparison.json").read_text())
    fingerprints = {(row["task"], row["seed"]): row["observed_init_fingerprint"]
                    for row in reference["rows"] if row["variant"] == "w1"}
    if len(cases) != 24 or set(fingerprints) != {(row["task"], row["seed"]) for row in cases}:
        raise ValueError("Missing fixed validation scenes")
    if digest(ROOT/protocol["model"]["path"]/"model.safetensors") != protocol["model"]["sha256"]:
        raise ValueError("Frozen base changed")
    parent = torch.load(ROOT/plan["w1_checkpoint"], map_location="cpu", weights_only=True)
    model, _ = load_model("native", protocol)
    if args.expert:
        from vla_rl_mt.expert_policy import ExpertPolicyUpdate

        probe = ExpertPolicyUpdate(model, "bc", adapter_lr=3e-4, expert_lr=1e-5,
                                   **parent["architecture"]).to("cuda").eval()
    else:
        probe = SharedPrefixValue(model, **parent["architecture"]).to("cuda").eval()
    probe.requires_grad_(False)
    task_configs = {row["name"]: row for row in protocol["tasks"]}
    result = {"scope": "Fixed offline post-training development evaluation; one policy seed, 12 scenes/task",
              "rows": [], "variants": variants, "cases": cases, "complete": False,
              "plan_sha256": digest(args.plan), "source_sha256": digest(Path(__file__)),
              "rollouts_return_to_training": False, "final_test_used": False}

    def save():
        temporary = output/"evaluation.json.tmp"
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        temporary.replace(output/"evaluation.json")

    save()
    for variant in variants:
        ready_path = (ROOT/variant["ready_marker"]).resolve() if variant.get("ready_marker") else None
        if training_execution or ready_path:
            started_waiting = time.monotonic()
            while True:
                if ready_path and ready_path.exists():
                    break
                if training_execution and training_execution.exists():
                    execution = json.loads(training_execution.read_text())
                    job = next(row for row in execution["jobs"] if row["name"] == variant.get("training_job", variant["name"]))
                    if job["status"] == "complete":
                        break
                    if job["status"] == "failed":
                        raise RuntimeError(f"Training failed for {variant['name']}")
                    if any(row["status"] == "failed" for row in execution["jobs"]):
                        raise RuntimeError("Training queue stopped before this checkpoint completed")
                if time.monotonic()-started_waiting > 18000:
                    raise TimeoutError("Checkpoint queue did not complete within its bounded wait")
                time.sleep(2)
        checkpoint_path = (ROOT/variant["checkpoint"]).resolve()
        before = digest(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if variant.get("step") is not None and checkpoint["step"] != variant["step"]:
            raise ValueError("Checkpoint step differs from the declared learning-curve point")
        if ready_path and json.loads(ready_path.read_text())["checkpoint_sha256"] != before:
            raise ValueError("Checkpoint differs from its atomic ready marker")
        if variant["name"] != "zero_positive" and checkpoint["identity"]["parent_sha256"] != plan["parent_checkpoint_sha256"]["w1"]:
            raise ValueError("Actor has a different parent")
        if variant["name"] == "zero_positive" and before != plan["parent_checkpoint_sha256"]["w1"]:
            raise ValueError("Zero-update parent checkpoint changed")
        if any(not torch.equal(value, parent["addon"]["value_head"][key])
               for key, value in checkpoint["addon"]["value_head"].items()):
            raise ValueError("Frozen critic head changed during policy update")
        probe.load_addon(checkpoint["addon"])
        if args.expert:
            probe.expert.load_state_dict(checkpoint["expert"], strict=True)
        policy = PromptPolicy(model, variant["prompt"])
        for case in cases:
            row = evaluate_case(policy, case, task_configs[case["task"]], protocol)
            row.update(variant=variant["name"], deploy_prompt=variant["prompt"], checkpoint_sha256=before)
            if row["status"] == "complete" and row["observed_init_fingerprint"] != fingerprints[row["task"], row["seed"]]:
                raise ValueError("Observed initial state differs from the fixed W1 reference")
            result["rows"].append(row)
            save()
            print(json.dumps({key: row.get(key) for key in ("variant", "task", "seed", "status", "success", "steps")}), flush=True)
            if row["status"] != "complete":
                raise RuntimeError(f"Setup/policy error, not a task failure: {row.get('error')}")
        if digest(checkpoint_path) != before:
            raise ValueError("Evaluation mutated a checkpoint")
        gc.collect()
        torch.cuda.empty_cache()
    result["complete"] = True
    save()
    probe.close()


if __name__ == "__main__":
    main()
