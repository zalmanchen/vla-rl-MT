"""Seal head jobs, diagnostics and exposure-matched actor draw weights."""

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
from evaluate_multitask import digest, write_json
from evaluate_method_m1 import value_metrics
from vla_rl_mt.signal_policy import CandidatePairs
from vla_rl_mt.value_head import PrefixValueHead

PROTOCOL = ROOT/"configs/multitask/critic_signal.json"


def top_weights(scores, draws, rows, tasks, count):
    weights = torch.zeros(len(draws), dtype=torch.bool)
    for task in tasks:
        indices = [i for i, row in enumerate(draws) if rows[int(row)]["task"] == task]
        indices.sort(key=lambda i: (-float(scores[int(draws[i])]), i))
        if len(indices) < count:
            raise ValueError("Insufficient task draws for fixed supervision budget")
        weights[indices[:count]] = True
    return weights


def shuffle_weights(weights, draws, rows, instructions, seed):
    groups = defaultdict(list)
    for i, index in enumerate(draws):
        row = rows[int(index)]
        groups[row["task"], instructions[row["episode"]], row["success"], row["next_step"]-row["step"]].append(i)
    shuffled = weights.clone()
    generator = torch.Generator().manual_seed(seed+404)
    for indices in groups.values():
        permutation = torch.randperm(len(indices), generator=generator)
        shuffled[indices] = weights[torch.tensor(indices)[permutation]]
    return shuffled


def advantages(rows, prediction):
    values = np.empty(len(rows), dtype=np.float64)
    for i, row in enumerate(rows):
        last = i+1 == len(rows) or rows[i+1]["episode"] != row["episode"]
        if not last and rows[i+1]["step"] != row["next_step"]:
            raise ValueError("Unaligned true next observation")
        values[i] = float(row["success"] and last)+(0. if last else float(prediction[i+1]))-float(prediction[i])
    return values


def head_jobs():
    setting = json.loads(PROTOCOL.read_text())
    work = ROOT/setting["work"]
    feature = ROOT/setting["feature_cache"]
    if digest(feature) != json.loads((work/"feature-seal.json").read_text())["feature_sha256"]:
        raise ValueError("Frozen cache changed")
    jobs = []
    for seed in setting["head_seeds"]:
        for group, configuration in setting["head_groups"].items():
            steps = configuration["steps"]
            overrides = ["actor.model.model_type=mt_signal_value", f"data.feature_path={feature}",
                f"data.pool_manifest_sha256={setting['pool_manifest_sha256']}", "data.condition=instruction",
                "data.readout=vision_prompt_concat", "data.target_scheme=terminal_binary",
                f"+data.feature_sha256={digest(feature)}", f"+data.study_protocol_sha256={digest(PROTOCOL)}",
                f"+data.head_group={group}", f"actor.seed={seed}", "actor.model.feature_width=4096",
                f"actor.model.hidden_width={configuration['hidden']}", "actor.micro_batch_size=32", "actor.global_batch_size=32",
                "actor.optim.lr=0.0003", f"runner.max_steps={steps}", f"runner.save_interval={steps}",
                "runner.val_check_interval=-1", f"actor.group_name=head_{group}_s{seed}"]
            jobs.append({"name": f"head_{group}_s{seed}", "entry": "scripts/train_method_value.py", "config": "value_head",
                         "extension": "vla_rl_mt.signal_value", "overrides": overrides})
    write_json(work/"head-jobs.json", {"jobs": jobs, "protocol_sha256": digest(PROTOCOL)})


def actor_jobs():
    import h5py

    setting = json.loads(PROTOCOL.read_text())
    work = ROOT/setting["work"]
    cache = torch.load(ROOT/setting["feature_cache"], map_location="cpu", weights_only=True)
    rows, feature = cache["rows"], cache["features"]
    all_predictions, diagnostics = {}, {}
    for seed in setting["head_seeds"]:
        predictions = {}
        for group, configuration in setting["head_groups"].items():
            checkpoint = ROOT/setting["head_output"]/f"head_{group}_s{seed}"/f"checkpoints/global_step_{configuration['steps']}/actor/head.pt"
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if saved["metadata"]["group"] != group or saved["metadata"]["seed"] != seed or saved["step"] != configuration["steps"]:
                raise ValueError("Head identity/step mismatch")
            if saved["metadata"]["feature_sha256"] != digest(ROOT/setting["feature_cache"]):
                raise ValueError("Head trained on different features")
            model = PrefixValueHead(4096, configuration["hidden"]).eval()
            model.load_state_dict(saved["state_dict"], strict=True)
            with torch.no_grad():
                prediction = model.value_head(feature).squeeze(-1).numpy()
            if not np.isfinite(prediction).all():
                raise ValueError("Nonfinite value prediction")
            predictions[group] = prediction
        independent = np.asarray([predictions["task_a" if row["task"] == setting["tasks"][0] else "task_b"][i]
                                  for i, row in enumerate(rows)])
        for group, prediction in {"shared": predictions["shared"], "wide": predictions["wide"], "independent": independent}.items():
            name = f"{group}_s{seed}"
            all_predictions[name] = torch.from_numpy(prediction.copy())
            diagnostics[name] = value_metrics(rows, prediction, (0, 1))
    write_json(work/"critic-diagnostics.json", {"complete": True, "scope": "Behavior-policy binary MC fit, not oracle new-policy value", "groups": diagnostics})
    torch.save(all_predictions, work/"predictions.pt")
    pairs = CandidatePairs(SimpleNamespace(rows=rows), setting["tasks"], 128, [False]*512, setting["actor_seed"])
    instructions = {}
    for episode in sorted({row["episode"] for row in rows}):
        with h5py.File(ROOT/setting["pool"]/episode) as source:
            instructions[episode] = str(source.attrs["instruction"])
    weights = {}
    for group in ("shared", "independent", "wide"):
        values = all_predictions[f"{group}_s{setting['actor_critic_seed']}"].numpy()
        weights[group] = top_weights(advantages(rows, values), pairs.draws, rows, setting["tasks"], setting["retained_per_task"])
    weights["shuffled"] = shuffle_weights(weights["shared"], pairs.draws, rows, instructions, setting["actor_seed"])
    if int((weights["shared"] != weights["shuffled"]).sum()) < 10:
        raise ValueError("Strict shuffled control changes fewer than10 draw weights")
    matched = torch.zeros(512, dtype=torch.bool)
    generator = torch.Generator().manual_seed(setting["actor_seed"]+505)
    for task in setting["tasks"]:
        indices = [i for i, row in enumerate(pairs.draws) if rows[int(row)]["task"] == task and rows[int(row)]["success"]]
        if len(indices) < setting["retained_per_task"]:
            raise ValueError("Insufficient successful draws for matched reference")
        selected = torch.randperm(len(indices), generator=generator)[:setting["retained_per_task"]]
        matched[[indices[int(i)] for i in selected]] = True
    weights["success_matched"] = matched
    sequence = [{"episode": rows[int(row)]["episode"], "observation_index": rows[int(row)]["observation_index"], "noise_key": noise}
                for row, noise in zip(pairs.draws, pairs.noise_keys, strict=True)]
    exposures, physical, audit = {}, {}, {}
    for group, selection in weights.items():
        counts = Counter()
        times = 0
        for i, row_index in enumerate(pairs.draws):
            if not selection[i]: continue
            row = rows[int(row_index)]
            counts[row["task"], instructions[row["episode"]], bool(row["success"])] += 1
            times += row["next_step"]-row["step"]
        for task in setting["tasks"]:
            if sum(v for k, v in counts.items() if k[0] == task) != setting["retained_per_task"]:
                raise ValueError("Per-task supervision quota differs")
        exposures[group], physical[group] = counts, times
        audit[group] = {"supervised_draws": int(selection.sum()), "action_steps": times,
            "outcome_instruction_composition": [{"task": k[0], "instruction": k[1], "success": k[2], "count": v} for k, v in sorted(counts.items())]}
        write_json(work/f"weights-{group}.json", {"weights": selection.tolist(), "sequence": sequence,
            "protocol_sha256": digest(PROTOCOL), "group": group, "scope": setting["actor_extraction"]})
    if exposures["shared"] != exposures["shuffled"] or physical["shared"] != physical["shuffled"]:
        raise ValueError("Shuffling changed outcome/instruction/action-time exposure")
    write_json(work/"exposure-audit.json", {"complete": True, "changed_shuffled_draws": int((weights["shared"] != weights["shuffled"]).sum()), "groups": audit})
    base = json.loads((ROOT/setting["parent_protocol"]).read_text())
    template = next(j for j in json.loads((ROOT/base["work"]/"jobs.json").read_text())["jobs"] if j["name"] == "task_mixed")
    jobs, smoke, variants = [], [], []
    for group in setting["actor_groups"]:
        path = work/f"weights-{group}.json"
        overrides = [v.replace("actor.model.model_type=mt_native_task_policy", "actor.model.model_type=mt_signal_policy")
                     .replace("actor.model.mode=bc", "actor.model.mode=c1_weighted")
                     .replace("+runner.eval_export_steps=[64]", "+runner.eval_export_steps=[]")
                     .replace("actor.group_name=task_mixed", f"actor.group_name=signal_{group}") for v in template["overrides"]]
        overrides += [f"+actor.model.study_protocol_sha256={digest(PROTOCOL)}", f"+actor.model.signal_group={group}",
                      f"+data.draw_weights_path={path}", f"+data.draw_weights_sha256={digest(path)}"]
        job = {**template, "name": f"signal_{group}", "extension": "vla_rl_mt.signal_policy", "overrides": overrides}
        jobs.append(job)
        smoke.append({**job, "overrides": [v.replace("runner.max_steps=128", "runner.max_steps=4")
            .replace("runner.save_interval=128", "runner.save_interval=4") for v in overrides]+["+runner.trace_final_resumable=false"]})
        actor = ROOT/setting["actor_output"]/job["name"]/"checkpoints/global_step_128/actor"
        variants.append({"name": job["name"], "step": 128, "parent": str(ROOT/setting["initial_addon"]),
            "parent_sha256": digest(ROOT/setting["initial_addon"]), "checkpoint": str(actor/"addon.pt"),
            "ready_marker": str(actor/"evaluation-ready.json")})
    write_json(work/"actor-jobs.json", {"jobs": jobs, "protocol_sha256": digest(PROTOCOL)})
    write_json(work/"actor-smoke-jobs.json", {"jobs": smoke, "protocol_sha256": digest(PROTOCOL)})
    write_json(work/"actor-evaluation.json", {"variants": variants, "protocol_sha256": digest(PROTOCOL)})
    write_json(work/"actor-selection-seal.json", {"protocol_sha256": digest(PROTOCOL), "actor_jobs_sha256": digest(work/"actor-jobs.json"),
        "evaluation_sha256": digest(work/"actor-evaluation.json"), "exposure_sha256": digest(work/"exposure-audit.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("heads", "actors"))
    args = parser.parse_args()
    {"heads": head_jobs, "actors": actor_jobs}[args.phase]()


if __name__ == "__main__":
    main()
