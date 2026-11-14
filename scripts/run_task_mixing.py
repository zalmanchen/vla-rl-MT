"""Run and audit the sealed original-Motus single/mixed-task diagnosis."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
from evaluate_multitask import digest, write_json
from vla_rl_mt.expert_policy import state_digest
from vla_rl_mt.task_mixing import PROTOCOL


def read(path):
    return json.loads(Path(path).read_text())


def audit_training(output, steps, settings, initial, parent_sha):
    execution = read(output/"execution.json")
    if not execution["complete"] or any(j["status"] != "complete" or j["exit_code"] != 0 for j in execution["jobs"]):
        raise ValueError("Training drivers did not complete successfully")
    summary = {}
    for job in execution["jobs"]:
        folder = output/job["name"]
        path = folder/f"checkpoints/global_step_{steps}/actor/addon.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        identity = checkpoint["identity"]
        if (checkpoint["step"] != steps or identity["parent_sha256"] != parent_sha
                or identity["base_sha256"] != initial["base_sha256"]
                or identity["pool_manifest_sha256"] != settings["pool_manifest_sha256"]):
            raise ValueError("Checkpoint initialization/data/step mismatch")
        if identity["protocol_sha256"] != digest(PROTOCOL) or identity["mode"] != "bc":
            raise ValueError("Different protocol or learning objective")
        if state_digest(checkpoint["expert"]) != checkpoint["expert_sha256"]:
            raise ValueError("Expert export checksum mismatch")
        if any(not torch.equal(value, initial["addon"]["value_head"][key])
               for key, value in checkpoint["addon"]["value_head"].items()):
            raise ValueError("Unused frozen value head changed")
        parameter_audit = read(folder/"parameter-audit.json")
        if parameter_audit["initial_expert_sha256"] != initial["expert_sha256"]:
            raise ValueError("Worker did not start from original Motus expert")
        rows = [json.loads(line) for line in (folder/"joint-metrics.jsonl").read_text().splitlines()]
        if [row["step"] for row in rows] != list(range(1, steps+1)):
            raise ValueError("Missing or duplicate optimizer updates")
        for row in rows:
            gradients = row["gradients"]
            if (row["value_loss"] != 0 or gradients["frozen_vlm"]["gradient_tensors"]
                    or gradients["value_head"]["gradient_tensors"]
                    or not all(g["finite"] for g in gradients.values())):
                raise ValueError("Unexpected value/frozen/nonfinite gradients")
        if checkpoint["exposure"]["samples"] != steps*4 or checkpoint["exposure"]["success"] != steps*4:
            raise ValueError("Actual successful BC exposure differs")
        sequence = read(folder/"sample-sequence.json")
        expected = read(ROOT/settings["work"]/f"sequence-{identity['task_group']}.json")[:steps*4]
        if sequence != expected:
            raise ValueError("Worker sample/noise schedule differs from sealed schedule")
        summary[job["name"]] = {"identity": identity, "checkpoint_sha256": digest(path),
            "exposure": checkpoint["exposure"], "peak_allocated_gib": max(r["peak_allocated_gib"] for r in rows)}
    return summary


def evaluate(shard):
    from evaluate_multitask import evaluate_case
    from vla_rl_mt.task_mixing import attach, native_policy

    settings = read(PROTOCOL)
    work = ROOT/settings["work"]
    seal = read(work/"selection-seal.json")
    initial_path = work/"initial-addon.pt"
    if digest(initial_path) != seal["initial_addon_sha256"] or digest(PROTOCOL) != seal["protocol_sha256"]:
        raise ValueError("Sealed evaluation initialization changed")
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    pool = ROOT/settings["pool"]
    if digest(pool/"manifest.json") != settings["pool_manifest_sha256"]:
        raise ValueError("Evaluation data manifest changed")
    manifest = read(pool/"manifest.json")
    cases = [{k: entry[k] for k in ("task", "seed", "seed_index", "noise_seed")}
             for entry in manifest["episodes"] if entry["split"] == "val"]
    reference = {(r["task"], r["seed"]): r for r in read(ROOT/".runtime/runs/method/m2-joint-v1/comparison.json")["rows"]
                 if r["variant"] == "base"}
    if len(cases) != 24 or set(reference) != {(r["task"], r["seed"]) for r in cases}:
        raise ValueError("Development evaluation cohort changed")
    output = ROOT/settings["evaluation_output"]/f"shard{shard}"
    output.mkdir(parents=True, exist_ok=False)
    schedule_path = work/"evaluation.json"
    if digest(schedule_path) != seal["evaluation_sha256"]:
        raise ValueError("Evaluation schedule changed")
    variants = read(schedule_path)["variants"][shard::2]
    report = {"rows": [], "variants": variants, "complete": False, "scope": settings["scope"],
              "evaluation_returns_to_training": False, "final_test_used": False}
    write_json(output/"evaluation.json", report)
    model = native_policy(manifest["protocol"])
    probe = attach(model, settings).eval()
    probe.requires_grad_(False)
    task_configs = {row["name"]: row for row in manifest["protocol"]["tasks"]}
    for variant in variants:
        path = Path(variant["checkpoint"])
        checksum = digest(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if variant["group"] == "base":
            if checksum != seal["initial_addon_sha256"] or state_digest(probe.expert_state()) != initial["expert_sha256"]:
                raise ValueError("Fresh Motus baseline is not original expert")
        else:
            marker = read(Path(variant["ready_marker"]))
            if (checkpoint["step"] != variant["step"] or marker["checkpoint_sha256"] != checksum
                    or checkpoint["identity"]["parent_sha256"] != seal["initial_addon_sha256"]
                    or checkpoint["identity"]["task_group"] != variant["group"]
                    or state_digest(checkpoint["expert"]) != checkpoint["expert_sha256"]):
                raise ValueError("Wrong step/group/initialization/export")
            probe.expert.load_state_dict(checkpoint["expert"], strict=True)
        probe.load_addon(checkpoint["addon"])
        for case in cases:
            row = evaluate_case(model, case, task_configs[case["task"]], manifest["protocol"])
            row.update(variant=variant["name"], group=variant["group"], checkpoint_step=variant["step"], checkpoint_sha256=checksum)
            report["rows"].append(row)
            write_json(output/"evaluation.json", report)
            expected = reference[row["task"], row["seed"]]
            if (row["status"] != "complete" or row["observed_init_fingerprint"] != expected["observed_init_fingerprint"]
                    or row["noise_seed"] != expected["noise_seed"]):
                raise ValueError(f"Invalid paired evaluation: {row.get('error')}")
            print(json.dumps({k: row.get(k) for k in ("variant", "task", "seed", "success", "steps")}), flush=True)
        if digest(path) != checksum:
            raise ValueError("Evaluation mutated a checkpoint")
    report["complete"] = True
    write_json(output/"evaluation.json", report)
    probe.close()


def summarize(settings, work):
    from summarize_method_m4 import paired

    rows = []
    for shard in (0, 1):
        report = read(ROOT/settings["evaluation_output"]/f"shard{shard}/evaluation.json")
        if not report["complete"]:
            raise ValueError("Evaluation shard incomplete")
        rows.extend(report["rows"])
    if len(rows) != 168 or any(row["status"] != "complete" for row in rows):
        raise ValueError("Rollout count/status differs from168 valid planned cases")
    variants = {v["name"]: {(r["task"], r["seed"]): r for r in rows if r["variant"] == v["name"]}
                for v in read(work/"evaluation.json")["variants"]}
    if any(len(mapping) != 24 for mapping in variants.values()):
        raise ValueError("Missing/duplicate variant cases")
    baseline = variants["motus_base"]
    original = {(r["task"], r["seed"]): r for r in read(ROOT/".runtime/runs/method/m2-joint-v1/comparison.json")["rows"] if r["variant"] == "base"}
    baseline_replay = all(baseline[k]["success"] == original[k]["success"] for k in baseline)
    scores = {name: {task: sum(r["success"] for r in mapping.values() if r["task"] == task) for task in settings["tasks"]}
              for name, mapping in variants.items()}
    comparison = {"complete": True, "scores_out_of12": scores, "original_baseline_outcomes_replay": baseline_replay,
        "versus_base": {name: paired(mapping, baseline, settings["tasks"]) for name, mapping in variants.items() if name != "motus_base"},
        "same_total": {group: paired(variants["mixed_128"], variants[f"{group}_128"], settings["tasks"]) for group in ("a_only", "b_only")},
        "same_per_task": {group: paired(variants["mixed_128"], variants[f"{group}_64"], settings["tasks"]) for group in ("a_only", "b_only")},
        "scope": settings["limits"], "checkpoint_selection": "All predeclared64/128 points; no validation selection"}
    write_json(work/"final-comparison.json", comparison)
    preservation = {name: digest(ROOT/name) == expected for name, expected in read(work/"preservation-before.json").items()}
    if not all(preservation.values()):
        raise ValueError("Protected input changed")
    write_json(work/"task-audit.json", {"complete": True, "preservation": preservation, "actor_groups": 3,
        "steps_per_group": 128, "new_evaluation_rollouts": 168, "evaluation_returns_to_training": False,
        "final_test_used": False, "original_baseline_outcomes_replay": baseline_replay})


def run():
    settings = read(PROTOCOL)
    work = ROOT/settings["work"]
    if (work/"controller.json").exists():
        raise FileExistsError("Task-mixing controller already exists; inspect it instead of launching a duplicate")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", JAX_PLATFORMS="cpu", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    state = {"pid": os.getpid(), "stage": "seal", "complete": False, "started_unix": time.time()}
    def save():
        write_json(work/"controller.json", state)
    save()
    try:
        seal = read(work/"selection-seal.json")
        for path, key in [(PROTOCOL, "protocol_sha256"), (work/"jobs.json", "jobs_sha256"),
                          (work/"evaluation.json", "evaluation_sha256"), (work/"sampling-audit.json", "sampling_audit_sha256")]:
            if digest(path) != seal[key]:
                raise ValueError("Prepared protocol or schedule changed")
        sources = [PROTOCOL, ROOT/"scripts/prepare_task_mixing.py", Path(__file__), ROOT/"scripts/run_method_jobs.py",
                   ROOT/"scripts/train_method_joint.py", ROOT/"scripts/evaluate_multitask.py", *list((ROOT/"src/vla_rl_mt").glob("*.py"))]
        snapshots = work/"sources"
        snapshots.mkdir(exist_ok=False)
        for path in sources:
            target = snapshots/path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
        write_json(work/"source-hashes.json", hashes)
        protected = read(ROOT/".runtime/runs/method/label-exposure-work-v1/preservation-before.json")
        protected.update({settings["pool"]+"/manifest.json": settings["pool_manifest_sha256"]})
        for name in protected:
            if digest(ROOT/name) != protected[name]:
                raise ValueError(f"Protected input changed before training: {name}")
        write_json(work/"preservation-before.json", protected)
        initial = torch.load(work/"initial-addon.pt", map_location="cpu", weights_only=True)
        for stage, plan, destination, steps in [("smoke4", work/"smoke-jobs.json", ROOT/settings["smoke_output"], 4),
                                               ("train128", work/"jobs.json", ROOT/settings["training_output"], 128)]:
            state["stage"] = stage
            save()
            if any(digest(ROOT/name) != value for name, value in hashes.items()):
                raise ValueError("Worker sources changed after snapshot")
            command = [sys.executable, str(ROOT/"scripts/run_method_jobs.py"), "--plan", str(plan), "--output", str(destination)]
            with (work/f"{stage}-controller.log").open("w") as stream:
                child = subprocess.Popen(command, cwd=ROOT, env=os.environ, stdout=stream, stderr=subprocess.STDOUT)
                state["child_pid"] = child.pid
                save()
                code = child.wait()
                if code:
                    raise RuntimeError(f"{stage} queue failed with exit{code}")
            audit = audit_training(destination, steps, settings, initial, seal["initial_addon_sha256"])
            if stage == "train128":
                smoke = ROOT/settings["smoke_output"]
                for group in settings["groups"]:
                    name = f"task_{group}"
                    prefix = [json.loads(line)["sft_loss"] for line in (smoke/name/"joint-metrics.jsonl").read_text().splitlines()]
                    replay = [json.loads(line)["sft_loss"] for line in (destination/name/"joint-metrics.jsonl").read_text().splitlines()[:4]]
                    if prefix != replay:
                        raise ValueError("Smoke/main first4 losses differ")
                audit["smoke_prefix_exact"] = True
            write_json(work/f"{stage}-audit.json", {"complete": True, "groups": audit})
        state["stage"] = "evaluate168"
        save()
        children = []
        for gpu in (0, 1):
            stream = (work/f"evaluation-{gpu}.log").open("w")
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "evaluate", "--shard", str(gpu)],
                cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}, stdout=stream, stderr=subprocess.STDOUT)
            children.append((child, stream))
        state["evaluation_pids"] = [p.pid for p, _ in children]
        save()
        codes = []
        for child, stream in children:
            codes.append(child.wait())
            stream.close()
        if any(codes):
            raise RuntimeError(f"Evaluation exit codes{codes}")
        summarize(settings, work)
        state.update(stage="complete", complete=True, finished_unix=time.time())
        save()
    except Exception as error:
        state.update(stage="failed", error=repr(error), finished_unix=time.time())
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "evaluate", "summarize"))
    parser.add_argument("--shard", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if args.phase == "run":
        run()
    elif args.phase == "evaluate":
        evaluate(args.shard)
    else:
        settings = read(PROTOCOL)
        summarize(settings, ROOT/settings["work"])


if __name__ == "__main__":
    main()
