"""Seal the fixed critic labels and paired actor-gradient optimizer jobs."""

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
from evaluate_multitask import digest, write_json
from vla_rl_mt.policy_gradients import PROTOCOL, SelectedPairs


def main():
    setting = json.loads(PROTOCOL.read_text())
    work = ROOT/setting["work"]
    work.mkdir(parents=True, exist_ok=False)
    old_work = ROOT/".runtime/runs/method/critic-signal-work-v1"
    if not json.loads((old_work/"task-audit.json").read_text())["complete"]:
        raise ValueError("Parent critic study is incomplete")
    frozen = ROOT/setting["frozen_weights"]
    report = json.loads(frozen.read_text())
    cache = torch.load(ROOT/".runtime/datasets/critic-signal-v1/features.pt", map_location="cpu", weights_only=True)
    template = next(j for j in json.loads((old_work/"actor-jobs.json").read_text())["jobs"] if j["name"] == "signal_shared")
    jobs, variants, smoke, sampling = [], [], [], {}
    for seed in setting["actor_seeds"]:
        parent = ROOT/(".runtime/runs/method/task-mixing-work-v1/initial-addon.pt" if seed == 0
                       else f".runtime/runs/method/critic-signal-work-v1/initial-seed{seed}.pt")
        dataset = SelectedPairs(SimpleNamespace(rows=cache["rows"]), report, setting["tasks"], 128, seed)
        write_json(work/f"sequence-seed{seed}.json", dataset.records)
        sampling[str(seed)] = {"samples": len(dataset), "success": sum(cache["rows"][int(i)]["success"] for i in dataset.draws),
            "by_task": dict(Counter(cache["rows"][int(i)]["task"] for i in dataset.draws)),
            "source_draw_repeats": dict(Counter(str(r["source_draw"]) for r in dataset.records)),
            "parent_sha256": digest(parent)}
        for mode in setting["modes"]:
            name = f"grad_{mode}_s{seed}"
            overrides = []
            for value in template["overrides"]:
                key = value.split("=", 1)[0]
                if key == "actor.model.model_type": value = f"{key}=mt_policy_gradients"
                elif key == "actor.seed": value = f"{key}={seed}"
                elif key == "+actor.model.initial_addon_path": value = f"{key}={parent}"
                elif key == "+actor.model.initial_addon_sha256": value = f"{key}={digest(parent)}"
                elif key == "actor.group_name": value = f"{key}={name}"
                overrides.append(value)
            overrides += [f"+actor.model.gradient_protocol_sha256={digest(PROTOCOL)}", f"+actor.model.gradient_mode={mode}"]
            job = {**template, "name": name, "extension": "vla_rl_mt.policy_gradients", "overrides": overrides}
            jobs.append(job)
            if seed == 0:
                short = {**job, "overrides": [v.replace("runner.max_steps=128", "runner.max_steps=4")
                    .replace("runner.save_interval=128", "runner.save_interval=4") for v in overrides]+["+runner.trace_final_resumable=false"]}
                smoke.append(short)
            actor = ROOT/setting["training_output"]/name/"checkpoints/global_step_128/actor"
            variants.append({"name": name, "mode": mode, "seed": seed, "step": 128, "parent": str(parent),
                "parent_sha256": digest(parent), "checkpoint": str(actor/"addon.pt"), "ready_marker": str(actor/"evaluation-ready.json")})
    replay = dict(smoke[0]); replay["name"] = "grad_native_replay_s0"
    replay["overrides"] = [v.replace("+actor.model.gradient_mode=mean", "+actor.model.gradient_mode=native_replay")
                           .replace("actor.group_name=grad_mean_s0", "actor.group_name=grad_native_replay_s0") for v in replay["overrides"]]
    smoke.insert(0, replay)
    write_json(work/"jobs.json", {"jobs": jobs, "protocol_sha256": digest(PROTOCOL)})
    write_json(work/"smoke-jobs.json", {"jobs": smoke, "protocol_sha256": digest(PROTOCOL)})
    write_json(work/"actor-evaluation.json", {"variants": variants, "protocol_sha256": digest(PROTOCOL)})
    write_json(work/"sampling-audit.json", {"complete": True, "by_seed": sampling, "fixed_critic_seed": 0,
        "frozen_weights_sha256": digest(frozen), "note": "Four cycles of same128 retained draws, no new selection or task/instruction relabeling"})
    write_json(work/"selection-seal.json", {"protocol_sha256": digest(PROTOCOL), "jobs_sha256": digest(work/"jobs.json"),
        "evaluation_sha256": digest(work/"actor-evaluation.json"), "weights_sha256": digest(frozen), "rollouts": 216})
    print(json.dumps({"actors": 9, "smoke_jobs": 4, "steps": 128, "rollouts": 216, "effective_supervision_per_actor": 512}), flush=True)


if __name__ == "__main__": main()
