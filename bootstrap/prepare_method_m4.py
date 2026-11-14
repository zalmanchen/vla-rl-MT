"""Materialize bounded M4 worker/evaluation queues from the locked protocol."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_jobs(stage, config, features, analysis, updates):
    if stage == "layers":
        jobs = []
        for layer in config["layers"]:
            for seed in config["head_seeds"]:
                name = f"layer{layer:02d}-seed{seed}"
                jobs.append({"name": name, "entry": "scripts/train_method_value.py", "config": "value_head",
                    "extension": "vla_rl_mt.layer_value", "overrides": [
                        f"data.feature_path={features/f'layer-{layer:02d}.pt'}",
                        f"data.pool_manifest_sha256={config['pool_manifest_sha256']}",
                        "data.target_scheme=terminal_binary", "data.readout=vision_prompt_concat",
                        f"+data.layer={layer}", f"+data.actor_sha256={config['parent_checkpoint_sha256']['w1']}",
                        "actor.model.model_type=mt_layer_value", "actor.model.feature_width=4096",
                        "actor.model.hidden_width=63", f"+actor.model.loss_scale={config['head_loss_scale']}",
                        f"actor.optim.lr={config['head_lr']}",
                        f"actor.seed={seed}", f"actor.micro_batch_size={config['head_micro_batch']}",
                        f"actor.global_batch_size={config['head_global_batch']}", f"runner.max_steps={config['head_steps']}",
                        f"runner.val_check_interval={config['head_steps']}", f"runner.save_interval={config['head_steps']}",
                        f"actor.group_name={name}"]})
        return {"jobs": jobs}
    if stage == "zero-eval":
        return {"variants": [{"name": "zero_positive", "checkpoint": config["w1_checkpoint"], "prompt": "positive"}]}
    selected = json.loads((analysis/"comparison.json").read_text())["selected_candidate"]
    modes = config["policy_groups"]+(["last_weighted", "layer_weighted"] if selected is not None else [])
    if stage == "eval":
        return {"variants": [{"name": mode, "checkpoint": str(updates/mode/f"checkpoints/global_step_{config['actor_steps']}/actor/addon.pt"),
                              "prompt": "positive" if mode.endswith("cond") or mode == "bc_positive" else "plain"}
                             for mode in modes]}
    jobs = []
    for mode in modes:
        if mode.startswith("bc"):
            label, checksum = "none", "none"
        elif mode in ("last_weighted", "layer_weighted"):
            layer = 18 if mode == "last_weighted" else selected
            label = str(analysis/f"layer{layer:02d}-seed0-labels.json")
            checksum = digest(label)
        else:
            source = config["label_sources"]["w1" if mode.startswith("c1") else "w2"]
            label, checksum = str(ROOT/source["path"]), source["sha256"]
            if digest(label) != checksum:
                raise ValueError("Original frozen critic labels changed")
        jobs.append({"name": mode, "entry": "scripts/train_method_joint.py", "config": "joint_value",
                     "extension": "vla_rl_mt.policy_update", "overrides": [
                         "actor.model.model_type=mt_native_policy", f"actor.model.mode={mode}",
                         f"+data.label_path={label}", f"+data.label_sha256={checksum}",
                         f"actor.micro_batch_size={config['actor_micro_batch']}",
                         f"actor.global_batch_size={config['actor_global_batch']}", f"actor.seed={config['actor_seed']}",
                         f"actor.optim.lr={config['actor_lr']}", f"runner.max_steps={config['actor_steps']}",
                         f"runner.save_interval={config['actor_steps']}", "runner.val_check_interval=-1", f"actor.group_name={mode}"]})
    return {"jobs": jobs, "selected_layer": selected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("layers", "policy", "zero-eval", "eval"))
    parser.add_argument("--features", type=Path, default=ROOT/".runtime/datasets/method-layers-v1")
    parser.add_argument("--analysis", type=Path, default=ROOT/".runtime/runs/method/layer-analysis-v1")
    parser.add_argument("--updates", type=Path, default=ROOT/".runtime/runs/method/m4-policy-v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expert", action="store_true", help="Bounded action-expert update-scope follow-up")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads((ROOT/"configs/multitask/method_m4.json").read_text())
    plan = make_jobs(args.stage, config, args.features.resolve(), args.analysis.resolve(), args.updates.resolve())
    if args.expert:
        if args.stage not in ("policy", "eval"):
            raise ValueError("Expert scope changes only policy training/evaluation")
        path = ROOT/"configs/multitask/method_m4_expert.json"
        expert = json.loads(path.read_text())
        if digest(ROOT/expert["base_protocol"]) != expert["base_protocol_sha256"]:
            raise ValueError("Parent protocol changed")
        key = "jobs" if args.stage == "policy" else "variants"
        plan[key] = [row for name in expert["policy_groups"] for row in plan[key] if row["name"] == name]
        plan["selected_layer"] = None
        plan["expert_protocol_sha256"] = digest(path)
        if key == "jobs":
            for job in plan[key]:
                job["extension"] = "vla_rl_mt.expert_policy"
                job["overrides"] = [value.replace("actor.model.model_type=mt_native_policy",
                    "actor.model.model_type=mt_native_expert_policy") for value in job["overrides"]]
                job["overrides"] += [f"+actor.model.adapter_lr={expert['adapter_lr']}",
                    f"+actor.model.expert_lr={expert['expert_lr']}", "+actor.optim.use_fp32_master_params=true",
                    "+actor.fsdp_config.save_full_model_weights=false"]
        original = plan[key]
        plan[key] = []
        for scope in expert["update_scopes"]:
            for original_row in original:
                row = copy.deepcopy(original_row)
                row["name"] = f"{scope}-{row['name']}"
                if key == "jobs":
                    row["overrides"].append(f"+actor.model.update_scope={scope}")
                else:
                    row["checkpoint"] = str(args.updates.resolve()/row["name"]/f"checkpoints/global_step_{expert['actor_steps']}/actor/addon.pt")
                plan[key].append(row)
    plan["protocol_sha256"] = digest(ROOT/"configs/multitask/method_m4.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2)+"\n")


if __name__ == "__main__":
    main()
