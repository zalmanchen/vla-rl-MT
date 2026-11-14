"""Plan, collect and freeze a small scene-traceable M1 development pool."""

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def make_plan(config_path, output):
    config = json.loads(config_path.read_text())
    protocol = json.loads((ROOT / config["initialization_protocol"]).read_text())
    seeds = json.loads((ROOT / protocol["evaluation"]["seed_file"]).read_text())
    tasks = {row["name"]: row for row in protocol["tasks"]}
    cases = []
    for task_index, task in enumerate(config["tasks"]):
        used = set()
        all_seeds = seeds[task]["success_seeds"]
        final_start, final_end = protocol["evaluation"]["reserved_final_seed_indices"]
        calibration_start, calibration_end = protocol["evaluation"]["seed_indices"]
        excluded = set(all_seeds[final_start:final_end] + all_seeds[calibration_start:calibration_end])
        for split_index, (split, bounds) in enumerate(config["splits"].items()):
            start, end = bounds
            chosen = all_seeds[start:end]
            if len(chosen) != end-start or set(chosen) & (used | excluded):
                raise ValueError(f"Overlapping or insufficient {task}/{split} scenes")
            used.update(chosen)
            for offset, seed in enumerate(chosen):
                cases.append({"task": task, "split": split, "seed": int(seed), "seed_index": start+offset,
                              "noise_seed": config["noise_seed_base"]+task_index*10000+split_index*1000+offset})
    sources = ["bootstrap/prepare_method_m1.py", "bootstrap/collect_recap_rollouts.py",
               "scripts/evaluate_multitask.py", "bootstrap/sources.lock.json", "bootstrap/environment.freeze.txt",
               "third_party/sources.manifest.json", protocol["model"]["norm_stats"],
               protocol["evaluation"]["seed_file"]]
    for task in config["tasks"]:
        sources += [tasks[task]["checker"], tasks[task]["instruction_source"]]
    output.mkdir(parents=True, exist_ok=False)
    plan = {"config": config, "config_sha256": digest(config_path), "protocol": protocol,
            "cases": cases, "sources": {name: digest(ROOT/name) for name in sources},
            "scope": config["scope"], "created_unix": time.time(), "final_test_used": False}
    save_json(output/"plan.json", plan)
    return plan


def collect_episode(model, case, task_config, plan, output):
    import cv2
    import h5py
    import numpy as np
    import torch
    from collect_recap_rollouts import CAMERAS, env_observation, make_task
    task = None
    sink = open(os.devnull, "w")
    started = time.time()
    try:
        task = make_task(case["task"], case["seed"], "")
        arm = "right" if getattr(task, "qpose_tag", 0) == 1 else "left"
        instruction = task_config["instruction_template"].format(arm=arm)
        task.set_instruction(instruction)
        torch.manual_seed(case["noise_seed"])
        states, encoded, boundaries, action_list, chunk_ids = [], {key: [] for key in CAMERAS}, [0], [], []

        def record_observation(observation):
            state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
            if state.shape != (14,) or not np.isfinite(state).all():
                raise ValueError("Invalid recorded state")
            states.append(state.copy())
            for camera, key in CAMERAS.items():
                rgb = observation["observation"][key]["rgb"]
                ok, png = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                if not ok:
                    raise ValueError("PNG encoding failed")
                encoded[camera].append(png.reshape(-1))

        observation = task.get_obs()
        record_observation(observation)
        limit = int(task.step_lim)
        while len(action_list) < limit and not task.eval_success:
            with torch.inference_mode():
                chunk, _ = model.predict_action_batch(env_observation(observation, instruction), mode="eval")
            actions = chunk[0].float().cpu().numpy().astype(np.float32)
            if actions.shape != (32, 14) or not np.isfinite(actions).all():
                raise ValueError("Invalid native policy action chunk")
            chunk_id = len(boundaries)-1
            for action in actions:
                with contextlib.redirect_stdout(sink):
                    task.take_action(action, action_type="qpos")
                action_list.append(action.copy())
                chunk_ids.append(chunk_id)
                if task.eval_success or len(action_list) >= limit:
                    break
            observation = task.get_obs()
            boundaries.append(len(action_list))
            record_observation(observation)
        if not action_list or task.take_action_cnt != len(action_list):
            raise ValueError("Simulator and recorded action counts differ")
        # Same reward construction as RLinf's public helper, applied to actual
        # executed action steps; terminal observation is a separate T+1 record.
        sys.path.insert(0, str(ROOT/"third_party/RLinf/examples/offline_rl/advantage_labeling/recap/process"))
        from compute_returns import compute_returns_for_episode

        success = bool(task.eval_success)
        config = plan["config"]
        returns, rewards = compute_returns_for_episode(len(action_list), success,
                                                       config["gamma"], config["failure_reward"])
        if abs(float(returns.min())) > config["return_scale"]:
            raise ValueError("Returns exceed the predeclared global normalization scale")
        boundary_array = np.asarray(boundaries, dtype=np.int64)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".partial.hdf5")
        with h5py.File(temporary, "w") as episode:
            episode.create_dataset("state", data=np.stack(states))
            episode.create_dataset("observation_steps", data=boundary_array)
            episode.create_dataset("actions", data=np.stack(action_list))
            episode.create_dataset("action_chunk_id", data=np.asarray(chunk_ids, dtype=np.int64))
            episode.create_dataset("rewards", data=rewards)
            episode.create_dataset("returns", data=returns)
            episode.create_dataset("decision_returns", data=returns[boundary_array[:-1]])
            episode.create_dataset("chunk_rewards", data=np.asarray([
                rewards[left:right].sum() for left, right in zip(boundaries[:-1], boundaries[1:], strict=True)
            ], dtype=np.float32))
            for camera, frames in encoded.items():
                dataset = episode.create_dataset(f"images/{camera}", (len(frames),),
                                                 dtype=h5py.vlen_dtype(np.dtype("uint8")))
                for index, png in enumerate(frames):
                    dataset[index] = png
            episode.attrs.update(
                **case, instruction=instruction, success=success, checker_success=bool(task.check_success()),
                terminated=success, truncated=not success, bootstrap_terminal=True,
                termination="success" if success else "intrinsic_benchmark_deadline",
                episode_limit=limit, complete=True, final_observation_saved=True,
                full_simulator_state_saved=False, source="frozen_native_motus_pi05", image_codec="PNG",
                model_sha256=plan["protocol"]["model"]["sha256"],
                model_revision=plan["protocol"]["model"]["revision"],
                gamma=config["gamma"], failure_reward=config["failure_reward"], return_scale=config["return_scale"],
                elapsed_seconds=time.time()-started,
            )
        temporary.replace(output)
        return {**case, "instruction": instruction, "success": success, "actions": len(action_list),
                "decisions": len(boundaries)-1, "elapsed_seconds": time.time()-started,
                "final_observation_saved": True, "complete": True}

    finally:
        if task is not None:
            task.close_env()
        sink.close()


def run_collection(output, task_name):
    import torch

    sys.path.insert(0, str(ROOT/"scripts"))
    from evaluate_multitask import load_model

    plan = json.loads((output/"plan.json").read_text())
    for name, expected in plan["sources"].items():
        if digest(ROOT/name) != expected:
            raise ValueError(f"Planned source changed: {name}")
    if digest(ROOT/plan["protocol"]["model"]["path"]/"model.safetensors") != plan["protocol"]["model"]["sha256"]:
        raise ValueError("Frozen behavior model checksum mismatch")
    tasks = {row["name"]: row for row in plan["protocol"]["tasks"]}
    cases = [case for case in plan["cases"] if case["task"] == task_name]
    if not cases:
        raise ValueError("Task is not in the frozen collection plan")
    with (output/f"{task_name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {"status": "running", "task": task_name, "results": [], "pid": os.getpid()}
        model, status["model_config"] = load_model("native", plan["protocol"])
        for case in cases:
            relative = Path(case["split"])/task_name/f"scene-{case['seed']}.hdf5"
            path = output/relative
            if path.exists():
                raise FileExistsError(f"Preserve completed data: {path}")
            result = collect_episode(model, case, tasks[task_name], plan, path)
            status["results"].append({**result, "file": str(relative), "sha256": digest(path)})
            save_json(output/f"{task_name}.json", status)
            print(json.dumps(status["results"][-1]), flush=True)
        del model
        torch.cuda.empty_cache()
        status["status"] = "complete"
        save_json(output/f"{task_name}.json", status)


def freeze_pool(output):
    import h5py

    plan = json.loads((output/"plan.json").read_text())
    rows, coverage = [], {}
    for task in plan["config"]["tasks"]:
        record = json.loads((output/f"{task}.json").read_text())
        if record["status"] != "complete":
            raise ValueError("Incomplete collection; do not freeze")
        for row in record["results"]:
            if digest(output/row["file"]) != row["sha256"]:
                raise ValueError("Recorded trajectory changed")
            with h5py.File(output/row["file"]) as episode:
                boundaries = episode["observation_steps"][:]
                if len(episode["state"]) != len(boundaries) or boundaries[-1] != len(episode["actions"]):
                    raise ValueError("Invalid observation/action alignment")
                if not episode.attrs["final_observation_saved"]:
                    raise ValueError("Missing final observation")
            key = f"{task}/{row['split']}"
            counts = coverage.setdefault(key, {"success": 0, "failure": 0})
            counts["success" if row["success"] else "failure"] += 1
            rows.append(row)
    if len(rows) != len(plan["cases"]):
        raise ValueError("Attempt count differs from predeclared plan")
    manifest = {"scope": "frozen development pool only; no final-test scenes", "plan_sha256": digest(output/"plan.json"),
                "config": plan["config"], "protocol": plan["protocol"], "episodes": rows, "coverage": coverage,
                "scene_split_certified": True, "outcome_coverage_ready": all(min(row.values()) > 0 for row in coverage.values()),
                "full_simulator_state_saved": False, "contrastive_pairs_verified": False}
    target = output/"manifest.json"
    if target.exists():
        raise FileExistsError("Do not replace a frozen pool manifest")
    save_json(target, manifest)
    print(json.dumps({"episodes": len(rows), "coverage": coverage,
                      "outcome_coverage_ready": manifest["outcome_coverage_ready"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "collect", "freeze"))
    parser.add_argument("--config", type=Path, default=ROOT/"configs/multitask/method_m1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "plan":
        plan = make_plan(args.config, output)
        print(json.dumps({"cases": len(plan["cases"]), "output": str(output)}))
    elif args.command == "collect":
        if not args.task:
            parser.error("--task is required for collection")
        run_collection(output, args.task)
    else:
        freeze_pool(output)


if __name__ == "__main__":
    main()
