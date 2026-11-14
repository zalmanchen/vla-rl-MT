"""Own the bounded BC replication, critic, signal and evaluation queues."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
from evaluate_multitask import digest, evaluate_case, write_json
from vla_rl_mt.expert_policy import state_digest
from vla_rl_mt.task_mixing import TaskPairs, attach, native_policy

PROTOCOL = ROOT/"configs/multitask/critic_signal.json"


def read(path):
    return json.loads(Path(path).read_text())


def evaluate(phase, shard):
    setting = read(PROTOCOL)
    work = ROOT/setting["work"]
    plan = work/("replica-evaluation.json" if phase == "bc" else "actor-evaluation.json")
    variants = read(plan)["variants"][shard::2]
    output = ROOT/setting["evaluation_output"]/phase/f"shard{shard}"
    output.mkdir(parents=True, exist_ok=False)
    pool = ROOT/setting["pool"]
    if digest(pool/"manifest.json") != setting["pool_manifest_sha256"]:
        raise ValueError("Frozen evaluation pool changed")
    manifest = read(pool/"manifest.json")
    cases = [{k: row[k] for k in ("task", "seed", "seed_index", "noise_seed")} for row in manifest["episodes"] if row["split"] == "val"]
    reference = {(r["task"], r["seed"]): r for r in read(ROOT/".runtime/runs/method/m2-joint-v1/comparison.json")["rows"] if r["variant"] == "base"}
    if len(cases) != 24 or set(reference) != {(r["task"], r["seed"]) for r in cases}:
        raise ValueError("Wrong fixed development cohort")
    base = read(ROOT/setting["parent_protocol"])
    model = native_policy(manifest["protocol"])
    probe = attach(model, base).eval()
    probe.requires_grad_(False)
    task_configs = {r["name"]: r for r in manifest["protocol"]["tasks"]}
    report = {"complete": False, "rows": [], "variants": variants, "phase": phase,
              "evaluation_returns_to_training": False, "final_test_used": False}
    write_json(output/"evaluation.json", report)
    for variant in variants:
        checkpoint_path = Path(variant["checkpoint"])
        parent_path = Path(variant["parent"])
        checksum = digest(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        parent = torch.load(parent_path, map_location="cpu", weights_only=True)
        marker = read(variant["ready_marker"])
        if (digest(parent_path) != variant["parent_sha256"] or checkpoint["identity"]["parent_sha256"] != variant["parent_sha256"]
                or checkpoint["step"] != 128 or marker["checkpoint_sha256"] != checksum
                or checkpoint["expert_sha256"] != state_digest(checkpoint["expert"])):
            raise ValueError("Wrong actor initialization/step/export")
        if any(not torch.equal(value, parent["addon"]["value_head"][key]) for key, value in checkpoint["addon"]["value_head"].items()):
            raise ValueError("Unused actor value head changed")
        probe.load_addon(checkpoint["addon"])
        probe.expert.load_state_dict(checkpoint["expert"], strict=True)
        for case in cases:
            row = evaluate_case(model, case, task_configs[case["task"]], manifest["protocol"])
            row.update(variant=variant["name"], checkpoint_sha256=checksum)
            report["rows"].append(row)
            write_json(output/"evaluation.json", report)
            expected = reference[row["task"], row["seed"]]
            if (row["status"] != "complete" or row["noise_seed"] != expected["noise_seed"]
                    or row["observed_init_fingerprint"] != expected["observed_init_fingerprint"]):
                raise ValueError(f"Invalid paired evaluation: {row.get('error')}")
            print(json.dumps({k: row.get(k) for k in ("variant", "task", "seed", "success", "steps")}), flush=True)
        if digest(checkpoint_path) != checksum:
            raise ValueError("Evaluation changed a checkpoint")
    report["complete"] = True
    write_json(output/"evaluation.json", report)
    probe.close()


def audit_actors(setting, phase, output, steps):
    work = ROOT/setting["work"]
    execution = read(output/"execution.json")
    if not execution["complete"] or any(j["status"] != "complete" or j["exit_code"] != 0 for j in execution["jobs"]):
        raise ValueError("Actor queue has failed/incomplete workers")
    plan = read(work/("replica-evaluation.json" if phase == "bc" else "actor-evaluation.json"))
    parents = {v["name"]: v for v in plan["variants"]}
    cache = torch.load(ROOT/setting["feature_cache"], map_location="cpu", weights_only=True)
    report = {}
    for job in execution["jobs"]:
        folder = output/job["name"]
        parent_info = parents[job["name"]]
        parent = torch.load(parent_info["parent"], map_location="cpu", weights_only=True)
        path = folder/f"checkpoints/global_step_{steps}/actor/addon.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        identity = checkpoint["identity"]
        if checkpoint["step"] != steps or identity["parent_sha256"] != parent_info["parent_sha256"]:
            raise ValueError("Wrong actor step or parent")
        if state_digest(checkpoint["expert"]) != checkpoint["expert_sha256"]:
            raise ValueError("Expert export checksum differs")
        if read(folder/"parameter-audit.json")["initial_expert_sha256"] != parent["expert_sha256"]:
            raise ValueError("Initial expert is not original Motus")
        if any(not torch.equal(value, parent["addon"]["value_head"][key]) for key, value in checkpoint["addon"]["value_head"].items()):
            raise ValueError("Frozen unused actor head changed")
        metrics = [json.loads(line) for line in (folder/"joint-metrics.jsonl").read_text().splitlines()]
        if [r["step"] for r in metrics] != list(range(1, steps+1)):
            raise ValueError("Wrong actual update count")
        for row in metrics:
            g = row["gradients"]
            if row["value_loss"] != 0 or g["value_head"]["gradient_tensors"] or g["frozen_vlm"]["gradient_tensors"] or not all(v["finite"] for v in g.values()):
                raise ValueError("Incorrect gradient route or nonfinite gradients")
        exposure = checkpoint["exposure"]
        if exposure["samples"] != steps*4:
            raise ValueError("Incorrect actual sample count")
        if phase == "bc":
            pairs = TaskPairs(SimpleNamespace(rows=cache["rows"]), "mixed", steps, tasks=setting["tasks"], seed=parent_info["seed"])
            if read(folder/"sample-sequence.json") != pairs.sequence() or exposure["success"] != steps*4:
                raise ValueError("Replica sampler/seed differs from intended stream")
        else:
            group = identity["signal_group"]
            weights = read(work/f"weights-{group}.json")
            if (read(folder/"candidate-sequence.json") != weights["sequence"][:steps*4]
                    or exposure["positive_conditional"] != sum(weights["weights"][:steps*4])):
                raise ValueError("Actual candidate sequence or loss supervision differs")
            if steps == 128:
                smoke = ROOT/".runtime/runs/method/critic-signal-actor-smoke-v1"/job["name"]
                prefix = [json.loads(line)["sft_loss"] for line in (smoke/"joint-metrics.jsonl").read_text().splitlines()]
                if prefix != [row["sft_loss"] for row in metrics[:4]]:
                    raise ValueError("Actor smoke prefix did not exactly replay")
        report[job["name"]] = {"identity": identity, "exposure": exposure,
            "checkpoint_sha256": digest(path), "peak_allocated_gib": max(r["peak_allocated_gib"] for r in metrics)}
    return report


def audit_heads(setting):
    from vla_rl_mt.signal_value import HeadPairs
    from vla_rl_mt.value_head import PrefixValueHead

    output = ROOT/setting["head_output"]
    execution = read(output/"execution.json")
    if not execution["complete"] or any(j["status"] != "complete" or j["exit_code"] != 0 for j in execution["jobs"]):
        raise ValueError("Head queue incomplete")
    cache = torch.load(ROOT/setting["feature_cache"], map_location="cpu", weights_only=True)
    audit = {}
    for seed in setting["head_seeds"]:
        by_group = {}
        for group, cfg in setting["head_groups"].items():
            name = f"head_{group}_s{seed}"
            folder = output/name
            checkpoint_path = folder/f"checkpoints/global_step_{cfg['steps']}/actor/head.pt"
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model = PrefixValueHead(4096, cfg["hidden"])
            model.load_state_dict(saved["state_dict"], strict=True)
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise ValueError("Nonfinite trained head")
            expected = HeadPairs(cache, setting["tasks"], cfg["tasks"], cfg["steps"], seed)
            actual = read(folder/"sample-sequence.json")
            if actual != expected.draws.tolist() or saved["step"] != cfg["steps"]:
                raise ValueError("Head sample/step identity differs")
            by_group[group] = actual
            audit[name] = {"parameters": sum(p.numel() for p in model.parameters()), "draws": len(actual),
                           "checkpoint_sha256": digest(checkpoint_path), "metadata": saved["metadata"]}
        for index, task in enumerate(setting["tasks"]):
            shared = [i for i in by_group["shared"] if cache["rows"][i]["task"] == task]
            if shared != by_group["task_a" if index == 0 else "task_b"] or by_group["wide"] != by_group["shared"]:
                raise ValueError("Per-task critic exposure is not exactly matched")
    return audit


def summarize(setting):
    import numpy as np
    from summarize_method_m4 import paired

    work = ROOT/setting["work"]
    rows, mappings = [], {}
    for phase, expected in (("bc", 48), ("signal", 120)):
        phase_rows = []
        for shard in (0, 1):
            report = read(ROOT/setting["evaluation_output"]/phase/f"shard{shard}/evaluation.json")
            if not report["complete"]:
                raise ValueError("Evaluation incomplete")
            phase_rows.extend(report["rows"])
        if len(phase_rows) != expected or any(r["status"] != "complete" for r in phase_rows):
            raise ValueError("Wrong rollout budget or validity")
        rows.extend(phase_rows)
    for name in sorted({r["variant"] for r in rows}):
        mapping = {(r["task"], r["seed"]): r for r in rows if r["variant"] == name}
        if len(mapping) != 24:
            raise ValueError("Missing/duplicate paired scenes")
        mappings[name] = mapping
    old = []
    for shard in (0, 1):
        old.extend(read(ROOT/f".runtime/runs/method/task-mixing-eval-v1/shard{shard}/evaluation.json")["rows"])
    baseline = {(r["task"], r["seed"]): r for r in old if r["variant"] == "motus_base"}
    bc0 = {(r["task"], r["seed"]): r for r in old if r["variant"] == "mixed_128"}
    score = lambda data: {task: sum(r["success"] for r in data.values() if r["task"] == task) for task in setting["tasks"]}
    bc_scores = {"seed0": score(bc0), **{f"seed{s}": score(mappings[f"bc_seed{s}"]) for s in setting["bc_replica_seeds"]}}
    totals = [sum(v.values()) for v in bc_scores.values()]
    summary = {"complete": True, "bc_scores_out_of12": bc_scores, "bc_total_mean_out_of24": float(np.mean(totals)),
        "bc_total_sample_std": float(np.std(totals, ddof=1)), "signal_scores_out_of12": {n: score(v) for n, v in mappings.items() if n.startswith("signal_")},
        "versus_original_motus": {n: paired(v, baseline, setting["tasks"]) for n, v in mappings.items()},
        "signal_vs_seed0_bc": {n: paired(v, bc0, setting["tasks"]) for n, v in mappings.items() if n.startswith("signal_")},
        "shared_vs_shuffled": paired(mappings["signal_shared"], mappings["signal_shuffled"], setting["tasks"]),
        "independent_vs_shared": paired(mappings["signal_independent"], mappings["signal_shared"], setting["tasks"]),
        "wide_vs_shared": paired(mappings["signal_wide"], mappings["signal_shared"], setting["tasks"]),
        "scope": "BC3 actor seeds; signal1 actor seed/3 diagnostic head seeds, two dev tasks. Per-draw top64 extraction, not original RECAP CFG."}
    write_json(work/"final-comparison.json", summary)
    protected = read(work/"preservation-before.json")
    checks = {name: digest(ROOT/name) == expected for name, expected in protected.items()}
    if not all(checks.values()):
        raise ValueError("Protected input changed")
    write_json(work/"task-audit.json", {"complete": True, "new_policy_rollouts": 168, "head_jobs": 12,
        "new_bc_actor_seeds": [1, 2], "signal_actors": 5, "preservation": checks,
        "evaluation_returns_to_training": False, "final_test_used": False})


def run():
    setting = read(PROTOCOL)
    work = ROOT/setting["work"]
    if (work/"controller.json").exists():
        raise FileExistsError("Controller already exists; inspect it instead of launching again")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", JAX_PLATFORMS="cpu", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    state = {"pid": os.getpid(), "complete": False, "stage": "wait_bc_cache", "started_unix": time.time()}
    def save(): write_json(work/"controller.json", state)
    def command(args, name):
        with (work/f"{name}.log").open("w") as stream:
            child = subprocess.Popen([sys.executable, *args], cwd=ROOT, env=os.environ, stdout=stream, stderr=subprocess.STDOUT)
            state["child_pid"] = child.pid
            save()
            code = child.wait()
        if code: raise RuntimeError(f"{name} failed with exit{code}")
    def queue(plan, output, name):
        command([str(ROOT/"scripts/run_method_jobs.py"), "--plan", str(work/plan), "--output", str(output)], name)
    def evaluate_phase(phase):
        children = []
        for gpu in (0, 1):
            stream = (work/f"evaluation-{phase}-{gpu}.log").open("w")
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "evaluate", "--phase", phase, "--shard", str(gpu)],
                cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}, stdout=stream, stderr=subprocess.STDOUT)
            children.append((child, stream))
        state["evaluation_pids"] = [p.pid for p, _ in children]
        save()
        codes = []
        for child, stream in children:
            codes.append(child.wait()); stream.close()
        if any(codes): raise RuntimeError(f"Evaluation failed:{codes}")
    save()
    try:
        deadline = time.monotonic()+7200
        while True:
            execution_path = ROOT/setting["replica_output"]/"execution.json"
            execution = read(execution_path) if execution_path.exists() else {}
            if any(j["status"] == "failed" for j in execution.get("jobs", [])):
                raise RuntimeError("BC replication worker failed")
            if execution.get("complete") and (work/"feature-seal.json").exists(): break
            if time.monotonic() > deadline: raise TimeoutError("BC replication/cache did not complete within2h")
            time.sleep(2)
        snapshots = work/"sources"
        snapshots.mkdir(exist_ok=False)
        sources = [PROTOCOL, Path(__file__), ROOT/"scripts/prepare_critic_signal.py", ROOT/"scripts/plan_critic_signal.py",
                   ROOT/"scripts/run_method_jobs.py", ROOT/"scripts/train_method_joint.py", ROOT/"scripts/train_method_value.py",
                   *list((ROOT/"src/vla_rl_mt").glob("*.py"))]
        hashes = {}
        for path in sources:
            name = str(path.relative_to(ROOT)); target = snapshots/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target); hashes[name] = digest(path)
        write_json(work/"source-hashes.json", hashes)
        protected = read(ROOT/".runtime/runs/method/task-mixing-work-v1/preservation-before.json")
        protected.update({setting["initial_addon"]: digest(ROOT/setting["initial_addon"]),
                          "configs/multitask/task_mixing.json": digest(ROOT/"configs/multitask/task_mixing.json")})
        for name, expected in protected.items():
            if digest(ROOT/name) != expected: raise ValueError(f"Protected input changed:{name}")
        write_json(work/"preservation-before.json", protected)
        write_json(work/"bc-training-audit.json", {"complete": True, "groups": audit_actors(setting, "bc", ROOT/setting["replica_output"], 128)})
        state["stage"] = "evaluate_bc48"; save(); evaluate_phase("bc")
        state["stage"] = "heads12"; save()
        command([str(ROOT/"scripts/plan_critic_signal.py"), "heads"], "head-planning")
        queue("head-jobs.json", ROOT/setting["head_output"], "head-controller")
        write_json(work/"head-audit.json", {"complete": True, "groups": audit_heads(setting)})
        state["stage"] = "labels_and_exposure"; save()
        command([str(ROOT/"scripts/plan_critic_signal.py"), "actors"], "actor-planning")
        for name, expected in hashes.items():
            if digest(ROOT/name) != expected: raise ValueError("Worker source changed during study")
        state["stage"] = "actor_smoke4"; save()
        smoke = ROOT/".runtime/runs/method/critic-signal-actor-smoke-v1"
        queue("actor-smoke-jobs.json", smoke, "actor-smoke-controller")
        write_json(work/"actor-smoke-audit.json", {"complete": True, "groups": audit_actors(setting, "signal", smoke, 4)})
        state["stage"] = "actor_train128"; save()
        queue("actor-jobs.json", ROOT/setting["actor_output"], "actor-controller")
        write_json(work/"actor-training-audit.json", {"complete": True, "groups": audit_actors(setting, "signal", ROOT/setting["actor_output"], 128)})
        state["stage"] = "evaluate_signal120"; save(); evaluate_phase("signal")
        summarize(setting)
        state.update(stage="complete", complete=True, finished_unix=time.time()); save()
    except Exception as error:
        state.update(stage="failed", error=repr(error), finished_unix=time.time()); save(); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "evaluate", "summarize"))
    parser.add_argument("--phase", choices=("bc", "signal"), default="bc")
    parser.add_argument("--shard", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if args.command == "run": run()
    elif args.command == "evaluate": evaluate(args.phase, args.shard)
    else: summarize(read(PROTOCOL))


if __name__ == "__main__":
    main()
