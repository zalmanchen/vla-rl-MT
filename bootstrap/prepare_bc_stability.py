"""Materialize two bounded BC traces, five new rollout points and export smoke."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=Path, required=True)
    args = parser.parse_args()
    output, updates = args.output.resolve(), args.updates.resolve()
    protocol_path = ROOT/"configs/multitask/bc_stability.json"
    protocol = json.loads(protocol_path.read_text())
    if digest(ROOT/protocol["parent_protocol"]) != protocol["parent_protocol_sha256"]:
        raise ValueError("Parent experiment protocol changed")
    base = json.loads((ROOT/"configs/multitask/method_m4.json").read_text())
    reference = ROOT/".runtime/runs/method/m4-expert-policy-v1/expert_adapter-bc/checkpoints/global_step_128/actor"
    full_bytes = sum(path.stat().st_size for path in reference.rglob("*") if path.is_file())
    export_bytes = (reference/"addon.pt").stat().st_size
    required = 2*full_bytes+6*export_bytes
    free = shutil.disk_usage(ROOT).free
    if free-required < 6*1024**3:
        raise ValueError("Insufficient disk for two final DCPs, light curve exports and a6GiB reserve")
    output.mkdir(parents=True, exist_ok=False)
    jobs, variants = [], []
    for index, lr in enumerate(protocol["expert_lrs"]):
        name = "expert_lr_1e5" if index == 0 else "expert_lr_1e6"
        overrides = ["actor.model.model_type=mt_native_expert_policy", "actor.model.mode=bc",
            "+data.label_path=none", "+data.label_sha256=none",
            f"actor.micro_batch_size={protocol['micro_batch']}", f"actor.global_batch_size={protocol['global_batch']}",
            f"actor.seed={protocol['seed']}", f"actor.optim.lr={protocol['adapter_lr']}",
            f"+actor.model.adapter_lr={protocol['adapter_lr']}", f"+actor.model.expert_lr={lr}",
            "+actor.model.update_scope=expert_adapter", "+actor.optim.use_fp32_master_params=true",
            "+actor.fsdp_config.save_full_model_weights=false", f"runner.max_steps={protocol['training_steps']}",
            f"runner.save_interval={protocol['training_steps']}", "runner.val_check_interval=-1",
            "+runner.eval_export_steps=[32,64]", f"actor.group_name={name}"]
        jobs.append({"name": name, "entry": "scripts/train_method_joint.py", "config": "joint_value",
                     "extension": "vla_rl_mt.bc_stability", "overrides": overrides})
        for step in protocol["evaluation_steps"]:
            if index == 0 and step == 128:
                continue  # Actual old rollouts, reused only after exact replay audit.
            actor = updates/name/f"checkpoints/global_step_{step}/actor"
            variants.append({"name": f"{name}_step{step}", "training_job": name, "step": step,
                             "checkpoint": str(actor/"addon.pt"), "ready_marker": str(actor/"evaluation-ready.json"),
                             "prompt": "plain"})
    smoke = {**jobs[0], "name": "light-export-smoke"}
    smoke["overrides"] = [value.replace("actor.model.mode=bc", "actor.model.mode=c1_weighted")
        .replace("+data.label_path=none", f"+data.label_path={ROOT/base['label_sources']['w1']['path']}")
        .replace("+data.label_sha256=none", f"+data.label_sha256={base['label_sources']['w1']['sha256']}")
        .replace("runner.max_steps=128", "runner.max_steps=4").replace("runner.save_interval=128", "runner.save_interval=4")
        .replace("+runner.eval_export_steps=[32,64]", "+runner.eval_export_steps=[2]") for value in jobs[0]["overrides"]]
    smoke["overrides"].append("+runner.trace_final_resumable=false")
    for name, value in [("jobs.json", {"jobs": jobs}), ("evaluation.json", {"variants": variants}),
                        ("smoke-jobs.json", {"jobs": [smoke]})]:
        (output/name).write_text(json.dumps({**value, "protocol_sha256": digest(protocol_path)}, indent=2)+"\n")
    (output/"disk-plan.json").write_text(json.dumps({"free_bytes": free, "expected_bytes": required,
        "reserve_bytes": free-required, "final_dcp_count": 2, "intermediate_weights_only": True}, indent=2)+"\n")
    print(json.dumps({"training_traces": len(jobs), "new_rollout_points": len(variants), "disk_reserve_gib": (free-required)/1024**3}))


if __name__ == "__main__":
    main()
