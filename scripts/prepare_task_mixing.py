"""Seal original-Motus initialization and matched single/mixed BC jobs."""

import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
from evaluate_multitask import digest, write_json
from vla_rl_mt.task_mixing import PROTOCOL, TaskPairs, attach, native_policy
from vla_rl_mt.expert_policy import state_digest
from vla_rl_mt.policy_update import GuidedPool


def host_actions(actions):
    if torch.is_tensor(actions):
        actions = actions.detach().float().cpu().numpy()
    return np.asarray(actions)


def observation(pool, entry):
    import cv2
    import h5py

    with h5py.File(pool/entry["file"]) as episode:
        images = {key: cv2.cvtColor(cv2.imdecode(episode[f"images/{key}"][0], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                  for key in episode["images"]}
        return {"main_images": images["cam_high"][None],
                "wrist_images": np.stack([images["cam_left_wrist"], images["cam_right_wrist"]])[None],
                "states": episode["state"][0][None], "task_descriptions": [str(episode.attrs["instruction"])]}


def main():
    settings = json.loads(PROTOCOL.read_text())
    work = ROOT/settings["work"]
    work.mkdir(parents=True, exist_ok=False)
    pool = GuidedPool(ROOT/settings["pool"], settings["pool_manifest_sha256"])
    sequences, audit = {}, {}
    for group in settings["groups"]:
        pairs = TaskPairs(pool, group, settings["steps"], tasks=settings["tasks"], seed=settings["seed"])
        sequences[group] = pairs.sequence()
        write_json(work/f"sequence-{group}.json", sequences[group])
        audit[group] = {"samples": len(pairs), "tasks": dict(Counter(row["task"] for row in sequences[group])),
            "executed_action_steps": sum(pool.rows[int(i)]["next_step"]-pool.rows[int(i)]["step"] for i in pairs.draws),
            "unique_episodes": len({r["episode"] for r in sequences[group]}),
            "sequence_sha256": hashlib.sha256(json.dumps(sequences[group]).encode()).hexdigest()}
    for group, task in zip(("a_only", "b_only"), settings["tasks"], strict=True):
        mixed = [r for r in sequences["mixed"] if r["task"] == task]
        if mixed != sequences[group][:256]:
            raise ValueError("Per-task example/noise prefixes do not match")
    write_json(work/"sampling-audit.json", {"complete": True, "per_task_prefix_exact": True, "groups": audit})
    model = native_policy(pool.manifest["protocol"])
    model.requires_grad_(False)
    cases = [next(r for r in pool.manifest["episodes"] if r["split"] == "val" and r["task"] == task) for task in settings["tasks"]]
    raw = [observation(pool.path, entry) for entry in cases]
    references = []
    for entry, sample in zip(cases, raw, strict=True):
        torch.manual_seed(entry["noise_seed"])
        with torch.inference_mode():
            actions, _ = model.predict_action_batch(copy.deepcopy(sample), mode="eval")
        references.append(host_actions(actions).copy())
    probe = attach(model, settings).eval()
    probe.requires_grad_(False)
    differences = []
    for entry, sample, reference in zip(cases, raw, references, strict=True):
        torch.manual_seed(entry["noise_seed"])
        with torch.inference_mode():
            actions, _ = model.predict_action_batch(copy.deepcopy(sample), mode="eval")
        differences.append(float(np.max(np.abs(host_actions(actions)-reference))))
    if any(value != 0 for value in differences):
        raise ValueError("Zero adapter changes original Motus inference")
    initial = {"kind": "Original Motus plus zero-output addon; no W1 weights", "architecture": probe.architecture,
               "base_sha256": pool.manifest["protocol"]["model"]["sha256"], "addon": probe.addon_state(),
               "expert_sha256": state_digest(probe.expert_state()), "step": 0}
    torch.save(initial, work/"initial-addon.pt")
    write_json(work/"initialization-audit.json", {"complete": True, "original_motus_action_max_abs": differences,
        "native_model_sha256": initial["base_sha256"], "expert_sha256": initial["expert_sha256"],
        "initial_addon_sha256": digest(work/"initial-addon.pt"), "w1_loaded": False})
    probe.close()
    jobs, smoke, variants = [], [], [{"name": "motus_base", "group": "base", "step": 0,
                                     "checkpoint": str(work/"initial-addon.pt")}]
    updates = ROOT/settings["training_output"]
    for group in settings["groups"]:
        overrides = ["actor.model.model_type=mt_native_task_policy", "actor.model.mode=bc",
            f"data.pool_path={pool.path}", f"data.pool_manifest_sha256={settings['pool_manifest_sha256']}",
            f"+actor.model.initial_addon_path={work/'initial-addon.pt'}",
            f"+actor.model.initial_addon_sha256={digest(work/'initial-addon.pt')}",
            f"+actor.model.protocol_sha256={digest(PROTOCOL)}", f"+actor.model.task_group={group}",
            "actor.micro_batch_size=2", "actor.global_batch_size=4", "actor.seed=0", "actor.optim.lr=0.0003",
            "+actor.model.adapter_lr=0.0003", "+actor.model.expert_lr=1e-6", "+actor.model.update_scope=expert_adapter",
            "+actor.optim.use_fp32_master_params=true", "+actor.fsdp_config.save_full_model_weights=false",
            "runner.max_steps=128", "runner.save_interval=128", "runner.val_check_interval=-1",
            "+runner.eval_export_steps=[64]", f"actor.group_name=task_{group}"]
        job = {"name": f"task_{group}", "entry": "scripts/train_method_joint.py", "config": "joint_value",
               "extension": "vla_rl_mt.task_mixing", "overrides": overrides}
        jobs.append(job)
        smoke.append({**job, "overrides": [v.replace("runner.max_steps=128", "runner.max_steps=4").replace(
            "runner.save_interval=128", "runner.save_interval=4").replace("+runner.eval_export_steps=[64]", "+runner.eval_export_steps=[]")
            for v in overrides]+["+runner.trace_final_resumable=false"]})
        for step in settings["evaluation_steps"]:
            actor = updates/job["name"]/f"checkpoints/global_step_{step}/actor"
            variants.append({"name": f"{group}_{step}", "group": group, "step": step,
                "checkpoint": str(actor/"addon.pt"), "ready_marker": str(actor/"evaluation-ready.json")})
    write_json(work/"jobs.json", {"jobs": jobs, "protocol_sha256": digest(PROTOCOL)})
    write_json(work/"smoke-jobs.json", {"jobs": smoke, "protocol_sha256": digest(PROTOCOL)})
    write_json(work/"evaluation.json", {"variants": variants, "protocol_sha256": digest(PROTOCOL)})
    write_json(work/"selection-seal.json", {"protocol_sha256": digest(PROTOCOL),
        "pool_manifest_sha256": settings["pool_manifest_sha256"], "initial_addon_sha256": digest(work/"initial-addon.pt"),
        "jobs_sha256": digest(work/"jobs.json"), "evaluation_sha256": digest(work/"evaluation.json"),
        "sampling_audit_sha256": digest(work/"sampling-audit.json"), "evaluation_rollouts": 168})
    print(json.dumps({"complete": True, "groups": 3, "steps": 128, "evaluations": 168,
                      "zero_adapter_actions_exact": True, "sampling": audit}), flush=True)


if __name__ == "__main__":
    main()
