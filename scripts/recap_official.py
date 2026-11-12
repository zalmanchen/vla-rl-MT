"""Plan, run and inspect the pinned RLinf LIBERO RECAP workflow.

This only orchestrates original entrypoints; training and labeling remain in
RLinf. Each run owns its configs, metadata sidecars, logs and checkpoints.
"""

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("libero10_task0_sft", "libero10_task0_train", "libero10_task0_eval")


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def report_metrics(metrics):
    """Keep raw TensorBoard values; encode nonfinite diagnostics explicitly in JSON."""
    invalid = [name for name, value in metrics.items() if not math.isfinite(value)]
    result = {name: None if name in invalid else value for name, value in metrics.items()}
    if invalid:
        result["_nonfinite_metrics"] = invalid
    return result


def merge(base, changes):
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


def checkpoint(run, name, steps):
    return run / name / "checkpoints" / f"global_step_{steps}" / "actor"


def summarize(run, manifest):
    groups = {}
    for step in manifest["steps"]:
        if "expected_episodes" not in step:
            continue
        path = run / step["name"] / "result.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text())
        count = int(metrics["eval/num_trajectories"])
        if count != step["expected_episodes"]:
            raise ValueError(f"Evaluation count changed: {step['name']}")
        group = groups.setdefault(step["name"].rsplit("-seed", 1)[0],
                                  {"episodes": 0, "success_once": 0, "success_at_end": 0, "runs": []})
        group["episodes"] += count
        for key in ("success_once", "success_at_end"):
            group[key] += round(metrics[f"eval/{key}"] * count)
        group["runs"].append(step["name"])
    for group in groups.values():
        group["success_once_rate"] = group["success_once"] / group["episodes"]
        group["success_at_end_rate"] = group["success_at_end"] / group["episodes"]
    result = {"scope": manifest["protocol"]["scope"], "evaluations": groups,
              "comparison": "CFG unconditional initialization vs CFG positive after training; native baseline separate",
              "limitations": "Environment seeds only; policy RNG is upstream-unseeded. Reused reset states are not distinct initial conditions."}
    if "sft-cfg-unconditional" in groups and "recap-positive" in groups:
        result["observed_change_percentage_points"] = 100 * (
            groups["recap-positive"]["success_once_rate"] - groups["sft-cfg-unconditional"]["success_once_rate"])
    write_json(run / "summary.json", result)
    return result


def compile_plan(run, smoke=False, sft_checkpoint=None, cfg_export=None):
    import yaml

    if run.exists():
        raise FileExistsError(f"Run directory already exists: {run}; use run/status to resume it")
    if cfg_export and not sft_checkpoint:
        raise ValueError("An existing CFG export requires the matching native SFT checkpoint")
    if sft_checkpoint and not smoke:
        raise ValueError("Existing two-step checkpoints may only bypass SFT in engineering smoke mode")
    protocol_path = ROOT / "configs/recap_official/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if smoke:
        protocol = copy.deepcopy(protocol)
        protocol["scope"] = "Engineering smoke only; no performance/reproduction claim"
        for name in ("sft", "value", "policy"):
            protocol[name]["steps"] = 2
        protocol["evaluation"].update(seeds=[0], envs_per_seed=2, rollout_epochs=1, episodes_per_policy=2)
    run.mkdir(parents=True)
    config_dir = run / "configs"
    config_dir.mkdir()
    tag = run.name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", tag):
        raise ValueError("Use a run name containing letters, digits, underscores or hyphens")
    data_root = run / "data"
    steps = []
    py = str(ROOT / ".runtime/venv-recap-official/bin/python")

    def helper(name, script, arguments, outputs=(), reserve_gib=0):
        steps.append({"name": name, "command": [py, str(ROOT / script), *map(str, arguments)],
                      "outputs": list(map(str, outputs)), "reserve_gib": reserve_gib})

    def stage(name, kind, template, changes, outputs=(), reserve_gib=0, nproc=1):
        base = yaml.safe_load((ROOT / f".runtime/recap-official/configs/{template}.yaml").read_text())
        config = config_dir / f"{name}.yaml"
        config.write_text(yaml.safe_dump(merge(base, changes), sort_keys=False))
        command = [py, str(ROOT / "scripts/run.py"), kind, "--profile", "recap-official",
                   "--config", str(config)]
        if nproc > 1:
            command += ["--nproc", str(nproc)]
        steps.append({"name": name, "command": command, "config": str(config),
                      "config_sha256": digest(config), "outputs": list(map(str, outputs)),
                      "reserve_gib": reserve_gib,
                      "training": kind in ("sft", "recap-value", "recap-policy")})

    def logger(name):
        # MetricLogger ignores experiment_name for TensorBoard directories.
        # Give each stage its own log_path; '.' preserves the checkpoint layout.
        return {"log_path": str(run / name), "experiment_name": "."}

    def dataset(name, kind):
        return {"dataset_path": str(data_root / name), "type": kind, "weight": 1.0,
                "robot_type": "libero", "model_type": "pi05"}

    train = [dataset(SPLITS[0], "sft"), dataset(SPLITS[1], "rollout")]
    validation = [dataset(SPLITS[2], "rollout")]
    validation[0]["max_samples"] = 10000  # Retain the public critic-validation cap.
    helper("source-audit", "bootstrap/prepare_recap_official.py", ["sources"])
    if smoke:
        helper("prepare-data", "bootstrap/prepare_recap_smoke.py",
               ["--source", ROOT / "datasets/recap-official", "--output", data_root],
               [data_root / "provenance.json"])
    else:
        # Metadata are copied at planning time; no writer touches verified raw data.
        for name in SPLITS:
            source, dest = ROOT / "datasets/recap-official" / name, data_root / name
            shutil.copytree(source / "meta", dest / "meta")
            for path in (dest / "meta").glob("*.parquet"):
                path.unlink()
            for folder in ("data", "videos"):
                (dest / folder).symlink_to((source / folder).resolve(), target_is_directory=True)
    norm = ROOT / ".runtime/recap-official/assets/physical-intelligence/libero/norm_stats.json"
    norm_copy = run / "assets/physical-intelligence/libero/norm_stats.json"
    norm_copy.parent.mkdir(parents=True)
    shutil.copy2(norm, norm_copy)
    native = sft_checkpoint or checkpoint(run, "sft", protocol["sft"]["steps"])
    exported = cfg_export or run / "cfg-initial"
    value = checkpoint(run, "value", protocol["value"]["steps"])
    policy = checkpoint(run, "policy", protocol["policy"]["steps"])
    if not sft_checkpoint:
        p = protocol["sft"]
        stage("sft", "sft", "sft", {
            "runner": {"max_steps": p["steps"], "max_epochs": -1, "save_interval": p["steps"],
                       "logger": logger("sft")},
            "data": {"train_data_paths": str(ROOT / "datasets/recap-official" / SPLITS[0])},
            "actor": {"global_batch_size": p["global_batch_size"], "micro_batch_size": p["micro_batch_size"],
                      "model": {"openpi_data": {"norm_stats_path": str(norm_copy)}}}},
              [native / "model_state_dict/full_weights.pt"], reserve_gib=35)
    if not cfg_export:
        helper("export-cfg", "bootstrap/export_recap_cfg.py",
               ["--checkpoint", native, "--output", exported, "--norm-stats", norm_copy],
               [exported / "model_state_dict/full_weights.pt", exported / "conversion.json"], reserve_gib=16)
    helper("format-audit", "bootstrap/check_recap_formats.py",
           ["--native", native, "--cfg", exported, "--norm-stats", norm_copy,
            "--report", run / "format-audit.json"], [run / "format-audit.json"])

    def evaluate(prefix, model_path, cfg, guidance=None):
        evaluation = protocol["evaluation"]
        for seed in evaluation["seeds"]:
            name = f"{prefix}-seed{seed}"
            model = {"model_path": str(model_path), "num_steps": evaluation["solver_steps"],
                     "openpi": {"num_steps": evaluation["solver_steps"], "train_expert_only": False},
                     "openpi_data": {"norm_stats_path": str(norm_copy)}}
            if guidance:
                model["openpi"]["guidance_type"] = guidance
            stage(name, "eval", "eval_cfg" if cfg else "eval", {
                "runner": {"logger": logger(name)},
                "env": {"eval": {"seed": seed, "total_num_envs": evaluation["envs_per_seed"],
                                 "rollout_epoch": evaluation["rollout_epochs"]}},
                "rollout": {"model": model}}, [run / name / "metrics.log"])
            steps[-1]["expected_episodes"] = evaluation["envs_per_seed"] * evaluation["rollout_epochs"]

    evaluate("sft-native", native, False)
    evaluate("sft-cfg-unconditional", exported, True, "no_guide")
    for name, paths in (("returns-train", train), ("returns-validation", validation)):
        stage(name, "recap-returns", "returns", {"data": {"tag": tag, "train_data_paths": paths,
                                                          "num_workers": 8}},
              [Path(item["dataset_path"]) / f"meta/returns_{tag}.parquet" for item in paths])
    p = protocol["value"]
    stage("value", "recap-value", "value", {
        "runner": {"max_steps": p["steps"], "max_epochs": -1, "save_interval": p["steps"],
                   "val_check_interval": 2 if smoke else 500, "logger": logger("value")},
        "data": {"tag": tag, "train_data_paths": train, "eval_data_paths": validation,
                 "train_num_workers": 2 if smoke else 8, "eval_num_workers": 2 if smoke else 4},
        "actor": {"global_batch_size": p["global_batch_size"], "micro_batch_size": p["micro_batch_size"]}},
          [value / "model_state_dict/full_weights.pt"], reserve_gib=10)
    stage("advantages", "recap-advantages", "advantages", {
        "advantage": {"value_checkpoint": str(value / "model_state_dict/full_weights.pt"),
                      "tag": tag, "returns_tag": tag, "batch_size": 16,
                      "num_dataloader_workers_per_gpu": 2 if smoke else 12},
        "data": {"train_data_paths": train}},
          [Path(item["dataset_path"]) / f"meta/advantages_{tag}.parquet" for item in train], nproc=2)
    p = protocol["policy"]
    stage("policy", "recap-policy", "policy", {
        "runner": {"max_steps": p["steps"], "max_epochs": -1, "save_interval": p["steps"],
                   "logger": logger("policy")},
        "data": {"advantage_tag": tag, "train_data_paths": train, "num_workers": 2 if smoke else 8},
        "actor": {"global_batch_size": p["global_batch_size"], "micro_batch_size": p["micro_batch_size"],
                  "model": {"model_path": str(exported)}}},
          [policy / "model_state_dict/full_weights.pt"], reserve_gib=35)
    # The legacy eval loader requires statistics next to the trained checkpoint.
    policy_stats = policy / "physical-intelligence/libero/norm_stats.json"
    helper("bundle-policy-assets", "scripts/recap_official.py",
           ["bundle-stats", "--source", norm_copy, "--output", policy_stats], [policy_stats])
    evaluate("recap-positive", policy, True, "positive")
    manifest = {"schema_version": 1, "created_unix": time.time(), "protocol": protocol,
                "protocol_sha256": digest(protocol_path), "smoke": smoke, "tag": tag,
                "native_checkpoint": str(native), "cfg_initial": str(exported), "steps": steps,
                "norm_stats": str(norm_copy), "norm_stats_sha256": digest(norm_copy),
                "source_lock_sha256": digest(ROOT / "bootstrap/recap-official.lock.json")}
    own_sources = ["scripts/recap_official.py", "scripts/run.py", "bootstrap/prepare_recap_official.py",
                   "bootstrap/export_recap_cfg.py", "bootstrap/check_recap_formats.py",
                   "bootstrap/prepare_recap_smoke.py", "configs/recap_official/protocol.json"]
    manifest["orchestration_sha256"] = {name: digest(ROOT / name) for name in own_sources}
    with zipfile.ZipFile(run / "orchestration.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in own_sources:
            archive.write(ROOT / name, name)
    if sft_checkpoint:
        manifest["external_native_sha256"] = digest(native / "model_state_dict/full_weights.pt")
    if cfg_export:
        manifest["external_cfg_sha256"] = digest(exported / "model_state_dict/full_weights.pt")
    write_json(run / "plan.json", manifest)
    write_json(run / "status.json", {"phase": "planned", "completed": {}, "created_unix": time.time()})
    return manifest


def execute(run, until=None):
    manifest = json.loads((run / "plan.json").read_text())
    state = json.loads((run / "status.json").read_text())
    names = [step["name"] for step in manifest["steps"]]
    if until is not None and until not in names:
        raise ValueError(f"Unknown stage boundary: {until}; choose from {names}")
    selected = manifest["steps"][:names.index(until) + 1] if until else manifest["steps"]
    if os.environ.get("REPO_PATH") != str(ROOT / "third_party/RLinf-recap-official"):
        raise RuntimeError("Source bootstrap/env_recap_official.sh before running this workflow")
    if digest(Path(manifest["norm_stats"])) != manifest["norm_stats_sha256"]:
        raise ValueError("Frozen normalization statistics changed")
    if digest(ROOT / "bootstrap/recap-official.lock.json") != manifest["source_lock_sha256"]:
        raise ValueError("Source/data lock changed after planning")
    for name, sha in manifest.get("orchestration_sha256", {}).items():
        if digest(ROOT / name) != sha:
            raise ValueError(f"Orchestration changed after planning: {name}; review before making a new plan")
    for key, path in (("external_native_sha256", Path(manifest["native_checkpoint"])),
                      ("external_cfg_sha256", Path(manifest["cfg_initial"]))):
        if key in manifest and digest(path / "model_state_dict/full_weights.pt") != manifest[key]:
            raise ValueError(f"Bound external checkpoint changed: {path}")
    lock_path = ROOT / ".runtime/recap-official/workflow.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.seek(0)
        lock.truncate()
        lock.write(json.dumps({"run_dir": str(run), "pid": os.getpid()}))
        lock.flush()
        for key in ("returncode", "missing_outputs", "reason"):
            state.pop(key, None)
        state.update(phase="running", pid=os.getpid(), started_unix=time.time())
        write_json(run / "status.json", state)
        for step in selected:
            name = step["name"]
            if "config" in step and digest(Path(step["config"])) != step["config_sha256"]:
                raise ValueError(f"Planned config changed: {name}")
            if name in state["completed"]:
                for item in state["completed"][name]["artifacts"]:
                    if digest(Path(item["path"])) != item["sha256"]:
                        raise ValueError(f"Completed artifact changed: {item['path']}")
                continue
            state.update(active_stage=name, updated_unix=time.time())
            write_json(run / "status.json", state)
            free = shutil.disk_usage(run).free / 2**30
            if free < step["reserve_gib"]:
                state.update(phase="failed", reason=f"{name} needs {step['reserve_gib']} GiB free; have {free:.1f}")
                write_json(run / "status.json", state)
                raise RuntimeError(state["reason"])
            print(f"Starting {name}: {run / (name + '.log')}", flush=True)
            with (run / f"{name}.log").open("a") as log:
                child = subprocess.Popen(step["command"], cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
                state["child_pid"] = child.pid
                write_json(run / "status.json", state)
                result = child.wait()
            missing = [path for path in step["outputs"] if not Path(path).is_file()]
            if result or missing:
                state.update(phase="failed", returncode=result, missing_outputs=missing, updated_unix=time.time())
                write_json(run / "status.json", state)
                raise RuntimeError(f"Stage {name} failed (exit={result}, missing={missing}); inspect its log")
            if "expected_episodes" in step or step.get("training"):
                from tensorboard.backend.event_processing.event_accumulator import (
                    EventAccumulator,
                )

                events = EventAccumulator(str(run / name / "tensorboard"),
                                          size_guidance={"scalars": 0}).Reload()
                metrics = {tag: events.Scalars(tag)[-1].value for tag in events.Tags()["scalars"]}
                if "expected_episodes" in step and metrics.get("eval/num_trajectories") != step["expected_episodes"]:
                    state.update(phase="failed", reason=f"Episode count mismatch: {metrics}")
                    write_json(run / "status.json", state)
                    raise RuntimeError(state["reason"])
                if step.get("training"):
                    loss = events.Scalars("train/loss") if "train/loss" in events.Tags()["scalars"] else []
                    finite = loss and all(math.isfinite(event.value) for tag in events.Tags()["scalars"]
                                          if tag.startswith("train/") for event in events.Scalars(tag))
                    if not finite:
                        state.update(phase="failed", reason=f"Missing/nonfinite training metrics: {name}")
                        write_json(run / "status.json", state)
                        raise RuntimeError(state["reason"])
                    if any(not math.isfinite(value) for tag, value in metrics.items()
                           if tag.startswith("eval/") and tag.endswith("/loss")):
                        state.update(phase="failed", reason=f"Nonfinite validation loss: {name}")
                        write_json(run / "status.json", state)
                        raise RuntimeError(state["reason"])
                write_json(run / name / "result.json", report_metrics(metrics))
                step["outputs"].append(str(run / name / "result.json"))
            artifacts = [{"path": path, "sha256": digest(Path(path))} for path in step["outputs"]]
            state["completed"][name] = {"finished_unix": time.time(), "artifacts": artifacts}
            write_json(run / "status.json", state)
        remaining = names[len(selected):]
        state.update(phase="ready" if remaining else "complete", active_stage=None,
                     remaining_stages=remaining, updated_unix=time.time())
        if not remaining:
            summarize(run, manifest)
        write_json(run / "status.json", state)
        print(f"Workflow {state['phase']}; scope: {manifest['protocol']['scope']}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--run-dir", type=Path, required=True)
    plan.add_argument("--smoke", action="store_true")
    plan.add_argument("--sft-checkpoint", type=Path)
    plan.add_argument("--cfg-export", type=Path)
    for action in ("run", "status", "summary"):
        item = sub.add_parser(action)
        item.add_argument("--run-dir", type=Path, required=True)
        if action == "run":
            item.add_argument("--until", help="Stop at this completed stage; keep later stages pending")
    bundle = sub.add_parser("bundle-stats")
    bundle.add_argument("--source", type=Path, required=True)
    bundle.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "plan":
        manifest = compile_plan(args.run_dir.resolve(), args.smoke,
                                args.sft_checkpoint.resolve() if args.sft_checkpoint else None,
                                args.cfg_export.resolve() if args.cfg_export else None)
        print(json.dumps({"run_dir": str(args.run_dir.resolve()), "scope": manifest["protocol"]["scope"],
                          "stages": [step["name"] for step in manifest["steps"]]}, indent=2))
    elif args.action == "run":
        execute(args.run_dir.resolve(), args.until)
    elif args.action == "summary":
        manifest = json.loads((args.run_dir / "plan.json").read_text())
        print(json.dumps(summarize(args.run_dir.resolve(), manifest), indent=2))
    elif args.action == "status":
        state = json.loads((args.run_dir / "status.json").read_text())
        held = False
        with (ROOT / ".runtime/recap-official/workflow.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                held = True
            lock.seek(0)
            owner = json.loads(lock.read() or "{}")
        state["worker_lock_held"] = held and owner.get("run_dir") == str(args.run_dir.resolve())
        if state["phase"] == "running" and not state["worker_lock_held"]:
            state["effective_phase"] = "stopped"
        print(json.dumps(state, indent=2))
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.source, args.output)


if __name__ == "__main__":
    main()
