"""Own parity-gated task-gradient training, evaluation and diagnostics."""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
from evaluate_multitask import digest, write_json
from vla_rl_mt.expert_policy import state_digest
from vla_rl_mt.policy_gradients import PROTOCOL


def read(path): return json.loads(Path(path).read_text())


def audit(output, steps, setting):
    work = ROOT/setting["work"]
    execution = read(output/"execution.json")
    if not execution["complete"] or any(j["status"] != "complete" or j["exit_code"] != 0 for j in execution["jobs"]):
        raise ValueError("Gradient queue incomplete or failed")
    variants = {v["name"]: v for v in read(work/"actor-evaluation.json")["variants"]}
    summary = {}
    for job in execution["jobs"]:
        folder = output/job["name"]
        info = variants["grad_mean_s0" if job["name"] == "grad_native_replay_s0" else job["name"]]
        parent = torch.load(info["parent"], map_location="cpu", weights_only=True)
        path = folder/f"checkpoints/global_step_{steps}/actor/addon.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        identity = checkpoint["identity"]
        if (checkpoint["step"] != steps or identity["parent_sha256"] != info["parent_sha256"]
                or identity["gradient_protocol_sha256"] != digest(PROTOCOL)):
            raise ValueError("Wrong gradient checkpoint step/initialization/protocol")
        if checkpoint["expert_sha256"] != state_digest(checkpoint["expert"]):
            raise ValueError("Expert export checksum mismatch")
        if read(folder/"parameter-audit.json")["initial_expert_sha256"] != parent["expert_sha256"]:
            raise ValueError("Expert did not start from original Motus")
        if any(not torch.equal(v, parent["addon"]["value_head"][k]) for k, v in checkpoint["addon"]["value_head"].items()):
            raise ValueError("Frozen unused critic head changed")
        metrics = [json.loads(line) for line in (folder/"joint-metrics.jsonl").read_text().splitlines()]
        if [r["step"] for r in metrics] != list(range(1, steps+1)):
            raise ValueError("Wrong number/order of optimizer updates")
        for row in metrics:
            g = row["gradients"]
            if row["value_loss"] != 0 or g["value_head"]["gradient_tensors"] or g["frozen_vlm"]["gradient_tensors"] or not all(v["finite"] for v in g.values()):
                raise ValueError("Wrong gradient route or nonfinite values")
        expected = read(work/f"sequence-seed{info['seed']}.json")[:steps*4]
        if read(folder/"selected-sequence.json") != expected:
            raise ValueError("Selected data/noise stream differs")
        exposure = checkpoint["exposure"]
        if exposure["samples"] != steps*4 or exposure["positive_conditional"] != steps*4:
            raise ValueError("Effective supervision differs from fixed selected stream")
        gradient_rows = []
        if identity["gradient_mode"] != "native_replay":
            gradient_rows = [json.loads(line) for line in (folder/"task-gradients.jsonl").read_text().splitlines()]
            if [r["step"] for r in gradient_rows] != list(range(1, steps+1)):
                raise ValueError("Task-gradient diagnostics missing")
            for row in gradient_rows:
                actual = math.sqrt(sum(v["combined_preclip_norm"]**2 for v in row["groups"].values()))
                if not row["degenerate_sum"] and abs(actual-row["ordinary_norm"])/max(row["ordinary_norm"], 1e-12) > 1e-3:
                    raise ValueError("Combined raw-gradient norm control failed")
        if steps == 128 and info["seed"] == 0:
            short = ROOT/setting["smoke_output"]/job["name"]/"joint-metrics.jsonl"
            if [json.loads(l)["sft_loss"] for l in short.read_text().splitlines()] != [r["sft_loss"] for r in metrics[:4]]:
                raise ValueError("Main/smoke first4 losses do not replay")
        summary[job["name"]] = {"identity": identity, "exposure": exposure, "checkpoint_sha256": digest(path),
            "peak_allocated_gib": max(r["peak_allocated_gib"] for r in metrics),
            "task_gradient_steps": len(gradient_rows),
            "conflicting_steps": sum(r["conflict"] for r in gradient_rows),
            "intervention_steps": sum(r["intervention_applied"] for r in gradient_rows)}
    return summary


def parity(output, work):
    paths = [output/name/"checkpoints/global_step_4/actor/addon.pt" for name in ("grad_native_replay_s0", "grad_mean_s0")]
    native, captured = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
    expert_equal = all(torch.equal(value, captured["expert"][key]) for key, value in native["expert"].items())
    addon_equal = all(torch.equal(value, captured["addon"][module][key]) for module, state in native["addon"].items() for key, value in state.items())
    losses = []
    for name in ("grad_native_replay_s0", "grad_mean_s0"):
        losses.append([json.loads(line)["sft_loss"] for line in (output/name/"joint-metrics.jsonl").read_text().splitlines()])
    result = {"complete": True, "expert_weights_exact": expert_equal, "addon_weights_exact": addon_equal,
              "four_losses_exact": losses[0] == losses[1], "scope": "Original accumulation versus isolated-task FP32 sum,4 official worker updates"}
    write_json(work/"native-parity-audit.json", result)
    if not expert_equal or not addon_equal or losses[0] != losses[1]:
        raise ValueError("Gradient collection does not exactly replay native ordinary updates")


def evaluate(shard):
    # Parameterize the existing paired evaluator without altering its source or
    # checkpoint/initialization checks; both study configs expose the same keys.
    import run_critic_signal as evaluator
    evaluator.PROTOCOL = PROTOCOL
    evaluator.evaluate("signal", shard)


def summarize(setting):
    from summarize_method_m4 import paired

    work = ROOT/setting["work"]
    all_rows = []
    for shard in (0, 1):
        report = read(ROOT/setting["evaluation_output"]/"signal"/f"shard{shard}/evaluation.json")
        if not report["complete"]: raise ValueError("Evaluation incomplete")
        all_rows.extend(report["rows"])
    if len(all_rows) != 216 or any(r["status"] != "complete" for r in all_rows):
        raise ValueError("Wrong rollout budget/status")
    mapping = {v["name"]: {(r["task"], r["seed"]): r for r in all_rows if r["variant"] == v["name"]}
               for v in read(work/"actor-evaluation.json")["variants"]}
    if any(len(v) != 24 for v in mapping.values()): raise ValueError("Missing/duplicate evaluation scenes")
    scores = {name: {task: sum(r["success"] for r in cases.values() if r["task"] == task) for task in setting["tasks"]}
              for name, cases in mapping.items()}
    aggregate, effects, geometry = {}, {}, {}
    for mode in setting["modes"]:
        totals = [sum(scores[f"grad_{mode}_s{s}"].values()) for s in setting["actor_seeds"]]
        aggregate[mode] = {"totals_out_of24": totals, "mean": float(np.mean(totals)), "sample_std": float(np.std(totals, ddof=1))}
        if mode != "mean":
            effects[mode] = {str(s): paired(mapping[f"grad_{mode}_s{s}"], mapping[f"grad_mean_s{s}"], setting["tasks"])
                             for s in setting["actor_seeds"]}
        for seed in setting["actor_seeds"]:
            name = f"grad_{mode}_s{seed}"
            rows = [json.loads(line) for line in (ROOT/setting["training_output"]/name/"task-gradients.jsonl").read_text().splitlines()]
            geometry[name] = {"negative_cosine_steps": sum(r["conflict"] for r in rows),
                "zero_task_steps": sum(r["zero_task"] for r in rows), "intervention_steps": sum(r["intervention_applied"] for r in rows),
                "mean_cosine": float(np.mean([r["cosine"] for r in rows if r["cosine"] is not None])) if any(r["cosine"] is not None for r in rows) else None,
                "mean_actual_delta": {group: float(np.mean([r["groups"][group]["actual_delta_norm"] for r in rows])) for group in ("adapter", "expert")},
                "global_positive_first_order_task_steps": {task: sum(sum(v[f"{task}_dot_actual_delta"] for v in r["groups"].values()) > 0 for r in rows)
                    for task in ("a", "b")},
                "positive_local_task_effects": {f"{group}_{task}": sum(r["groups"][group][f"{task}_dot_actual_delta"] > 0 for r in rows)
                    for group in ("adapter", "expert") for task in ("a", "b")}}
    write_json(work/"final-comparison.json", {"complete": True, "scores_out_of12": scores, "aggregate": aggregate,
        "paired_vs_mean": effects, "gradient_geometry": geometry, "scope": setting["limits"],
        "baseline_note": "Fresh selected-only four-cycle ordinary baseline; not a causal comparison to earlier sparse17/24"})
    checks = {name: digest(ROOT/name) == value for name, value in read(work/"preservation-before.json").items()}
    if not all(checks.values()): raise ValueError("Protected input changed")
    write_json(work/"task-audit.json", {"complete": True, "actors": 9, "steps_each": 128, "new_rollouts": 216,
        "preservation": checks, "evaluation_returns_to_training": False, "final_test_used": False})


def run():
    setting = read(PROTOCOL)
    work = ROOT/setting["work"]
    if (work/"controller.json").exists(): raise FileExistsError("Controller already exists; inspect before any restart")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", JAX_PLATFORMS="cpu", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    state = {"pid": os.getpid(), "stage": "wait_smoke", "complete": False, "started_unix": time.time()}
    def save(): write_json(work/"controller.json", state)
    save()
    try:
        deadline = time.monotonic()+3600
        short = ROOT/setting["smoke_output"]
        while True:
            execution = read(short/"execution.json") if (short/"execution.json").exists() else {}
            if any(j["status"] == "failed" for j in execution.get("jobs", [])): raise RuntimeError("Gradient smoke worker failed")
            if execution.get("complete"): break
            if time.monotonic() > deadline: raise TimeoutError("Smoke queue did not finish within1h")
            time.sleep(2)
        snapshots = work/"sources"; snapshots.mkdir(exist_ok=False)
        sources = [PROTOCOL, Path(__file__), ROOT/"scripts/prepare_policy_gradients.py", ROOT/"scripts/run_method_jobs.py",
                   ROOT/"scripts/run_critic_signal.py", ROOT/"scripts/train_method_joint.py", *list((ROOT/"src/vla_rl_mt").glob("*.py"))]
        hashes = {}
        for path in sources:
            name = str(path.relative_to(ROOT)); target = snapshots/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target); hashes[name] = digest(path)
        write_json(work/"source-hashes.json", hashes)
        seal = read(work/"selection-seal.json")
        if digest(PROTOCOL) != seal["protocol_sha256"] or digest(ROOT/setting["frozen_weights"]) != seal["weights_sha256"]:
            raise ValueError("Frozen gradient protocol/labels changed")
        protected = read(ROOT/".runtime/runs/method/critic-signal-work-v1/preservation-before.json")
        protected.update({setting["frozen_weights"]: seal["weights_sha256"], "configs/multitask/critic_signal.json": digest(ROOT/"configs/multitask/critic_signal.json")})
        write_json(work/"preservation-before.json", protected)
        write_json(work/"smoke-audit.json", {"complete": True, "groups": audit(short, 4, setting)})
        parity(short, work)
        state["stage"] = "train9x128"; save()
        for name, expected in hashes.items():
            if digest(ROOT/name) != expected: raise ValueError("Worker source changed after parity check")
        if digest(work/"jobs.json") != seal["jobs_sha256"]: raise ValueError("Training queue changed")
        with (work/"training-controller.log").open("w") as stream:
            child = subprocess.Popen([sys.executable, str(ROOT/"scripts/run_method_jobs.py"), "--plan", str(work/"jobs.json"),
                "--output", str(ROOT/setting["training_output"])], cwd=ROOT, env=os.environ, stdout=stream, stderr=subprocess.STDOUT)
            state["child_pid"] = child.pid; save(); code = child.wait()
        if code: raise RuntimeError(f"Main gradient queue failed:{code}")
        write_json(work/"training-audit.json", {"complete": True, "groups": audit(ROOT/setting["training_output"], 128, setting)})
        state["stage"] = "evaluate216"; save()
        if digest(work/"actor-evaluation.json") != seal["evaluation_sha256"]: raise ValueError("Evaluation plan changed")
        children = []
        for gpu in (0, 1):
            stream = (work/f"evaluation-{gpu}.log").open("w")
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "evaluate", "--shard", str(gpu)],
                cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}, stdout=stream, stderr=subprocess.STDOUT)
            children.append((child, stream))
        state["evaluation_pids"] = [p.pid for p, _ in children]; save()
        codes = []
        for child, stream in children: codes.append(child.wait()); stream.close()
        if any(codes): raise RuntimeError(f"Gradient evaluation failed:{codes}")
        summarize(setting)
        state.update(stage="complete", complete=True, finished_unix=time.time()); save()
    except Exception as error:
        state.update(stage="failed", error=repr(error), finished_unix=time.time()); save(); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "evaluate", "summarize"))
    parser.add_argument("--shard", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if args.phase == "run": run()
    elif args.phase == "evaluate": evaluate(args.shard)
    else: summarize(read(PROTOCOL))


if __name__ == "__main__": main()
