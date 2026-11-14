"""Seal native index views and collect real control-step RoboTwin 2.0 data.

The frozen policy and simulator helpers are reused. No actor training is
performed here, and old chunk-boundary images are never expanded into frames.
"""

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def select_successes(rows, tasks, count):
    selected = []
    for task in tasks:
        candidates = sorted((r for r in rows if r["task"] == task and r["split"] == "train" and r["success"]),
                            key=lambda r: r["seed_index"])
        if len(candidates) < count or len({r["seed"] for r in candidates}) != len(candidates):
            raise ValueError(f"Insufficient or repeated successes for {task}: {len(candidates)}/{count}")
        selected.extend(candidates[:count])
    return selected


def make_plan(config_path, output):
    config = json.loads(config_path.read_text())
    protocol = json.loads((ROOT / config["initialization_protocol"]).read_text())
    seed_path = ROOT / protocol["evaluation"]["seed_file"]
    if digest(seed_path) != protocol["evaluation"]["seed_file_sha256"]:
        raise ValueError("Seed list changed")
    seeds = json.loads(seed_path.read_text())
    cases = []
    for index, name in enumerate(config["tasks"]):
        all_seeds = seeds[name]["success_seeds"]
        first, last = protocol["evaluation"]["reserved_final_seed_indices"]
        low, high = protocol["evaluation"]["seed_indices"]
        excluded = set(all_seeds[first:last] + all_seeds[low:high])
        used = set()
        for split_index, (split, bounds) in enumerate(config["splits"].items()):
            left, right = bounds
            chosen = all_seeds[left:right]
            if (left < 0 or right <= left or len(chosen) != right-left or len(set(chosen)) != len(chosen)
                    or set(chosen) & (used | excluded)):
                raise ValueError(f"Invalid/overlapping/final-test scenes: {name}/{split}")
            used.update(chosen)
            for offset, seed in enumerate(chosen):
                cases.append({"task": name, "split": split, "seed": int(seed), "seed_index": left+offset,
                              "noise_seed": config["noise_seed_base"]+index*10000+split_index*1000+offset})
    if config["min_free_gib"] < 100:
        raise ValueError("Reserve at least100GiB free")
    if shutil.disk_usage(output.parent if output.parent.exists() else ROOT).free / 2**30 < config["min_free_gib"] + config["estimated_raw_gib"]:
        raise RuntimeError("Insufficient space for the declared collection peak")
    parent = ROOT / config["native_parent"]
    if digest(parent / "manifest.json") != config["native_parent_manifest_sha256"]:
        raise ValueError("Native parent pool changed")
    native = json.loads((parent / "manifest.json").read_text())
    selected = select_successes(native["episodes"], config["tasks"], config["sft_successes_per_task"])
    for row in selected:
        if digest(parent / row["file"]) != row["sha256"]:
            raise ValueError("Selected original trajectory changed")
    sources = ["bootstrap/prepare_sft_recap_data.py", "scripts/run_sft_recap_data.py", "bootstrap/collect_recap_rollouts.py",
               "scripts/evaluate_multitask.py", "bootstrap/sources.lock.json", "bootstrap/environment.freeze.txt",
               "third_party/sources.manifest.json", protocol["runtime"]["compatibility_manifest"],
               protocol["model"]["norm_stats"], protocol["evaluation"]["seed_file"]]
    tasks = {row["name"]: row for row in protocol["tasks"]}
    for task in config["tasks"]:
        sources += [tasks[task]["checker"], tasks[task]["instruction_source"]]
    plan = {"config": config, "config_sha256": digest(config_path), "protocol": protocol, "cases": cases,
            "sources": {name: digest(ROOT/name) for name in sources}, "created_unix": time.time(),
            "scope": config["scope"], "rollouts_return_to_training": True,
            "interaction_accounting": "All new216 attempts count as new training-data collection, including re-runs of old training initial conditions"}
    output.mkdir(parents=True, exist_ok=False)
    save_json(output/"plan.json", plan)
    counts = Counter((r["task"], r["instruction"]) for r in selected)
    save_json(output/"native-index.json", {"scope": "Index-only native development view; not valid for the official step loader",
        "parent": config["native_parent"], "parent_manifest_sha256": config["native_parent_manifest_sha256"],
        "sft": selected, "offline_experience": [r for r in native["episodes"] if r["split"] == "train"],
        "development": [r for r in native["episodes"] if r["split"] == "val"],
        "instruction_counts": [{"task": task, "instruction": instruction, "episodes": n}
                               for (task, instruction), n in sorted(counts.items())], "raw_images_copied": False})
    return plan


def inspect_episode(path, case, plan):
    import h5py
    import numpy as np

    with h5py.File(path) as episode:
        count = len(episode["action"])
        if not episode.attrs.get("complete") or episode.attrs["plan_sha256"] != digest(path.parents[2]/"plan.json"):
            raise ValueError(f"Incomplete or wrong-protocol episode: {path}")
        for key in ("task", "seed", "seed_index", "noise_seed", "split"):
            if episode.attrs[key] != case[key]:
                raise ValueError(f"Wrong trajectory identity: {key}")
        if episode["state"].shape != (count, 14) or episode["action"].shape != (count, 14):
            raise ValueError("Pre-action state/action shape mismatch")
        if not np.array_equal(episode["observation_steps"][:], np.arange(count+1)):
            raise ValueError("Missing real control-step observations")
        boundaries = episode["chunk_boundaries"][:]
        if boundaries[0] != 0 or boundaries[-1] != count or not ((np.diff(boundaries) > 0) & (np.diff(boundaries) <= 32)).all():
            raise ValueError("Invalid executed chunk boundaries")
        mask = episode["executed_mask"][:]
        expected_mask = np.arange(32)[None, :] < np.diff(boundaries)[:, None]
        if not np.array_equal(mask, expected_mask) or not episode.attrs["final_observation_recorded"]:
            raise ValueError("Chunk mask/final observation mismatch")
        requested = episode["requested_action_chunks"][:]
        executed = episode["action"][:]
        for i, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            if not np.array_equal(executed[left:right], requested[i, :right-left]):
                raise ValueError("Executed commands differ from the recorded requested chunk")
        for camera in episode["images"]:
            if len(episode[f"images/{camera}"]) != count or not len(episode[f"final_observation/images/{camera}"]):
                raise ValueError("Missing pre-action or final camera image")
        result = {**case, "instruction": str(episode.attrs["instruction"]), "success": bool(episode.attrs["success"]),
                  "actions": count, "observations": count+1, "decisions": len(boundaries)-1,
                  "termination_reason": str(episode.attrs["termination_reason"]),
                  "initial_observed_fingerprint": str(episode.attrs["initial_observed_fingerprint"]),
                  "elapsed_seconds": float(episode.attrs["elapsed_seconds"]), "status": "complete"}
    return {**result, "sha256": digest(path)}


def collect_episode(model, case, task_config, plan, destination, task_factory=None):
    import cv2
    import h5py
    import numpy as np
    import torch
    from collect_recap_rollouts import CAMERAS, env_observation, make_task

    destination = destination.absolute()
    task = None
    started = time.time()
    states, actions, chunks, chunk_ids = [], [], [], []
    images = {camera: [] for camera in CAMERAS}
    boundaries = [0]
    try:
        task = (task_factory or make_task)(case["task"], case["seed"], "")
        arm = "right" if getattr(task, "qpose_tag", 0) == 1 else "left"
        instruction = task_config["instruction_template"].format(arm=arm)
        task.set_instruction(instruction)
        torch.manual_seed(case["noise_seed"])

        def record(obs):
            state = np.asarray(obs["joint_action"]["vector"], dtype=np.float32)
            if state.shape != (14,) or not np.isfinite(state).all():
                raise ValueError("Invalid observed robot state")
            states.append(state.copy())
            for camera, key in CAMERAS.items():
                ok, encoded = cv2.imencode(".png", cv2.cvtColor(obs["observation"][key]["rgb"], cv2.COLOR_RGB2BGR))
                if not ok:
                    raise RuntimeError("Could not encode real camera frame")
                images[camera].append(encoded.reshape(-1))

        observation = task.get_obs()
        record(observation)
        initial = {"state": states[0].round(6).tolist(), "objects": {}}
        for name in task_config["objects"]:
            pose = getattr(task, name).get_pose()
            initial["objects"][name] = {"p": np.asarray(pose.p).round(6).tolist(), "q": np.asarray(pose.q).round(6).tolist()}
        with open(os.devnull, "w") as sink:
            while len(actions) < task.step_lim and not task.eval_success:
                with torch.inference_mode():
                    requested, _ = model.predict_action_batch(env_observation(observation, instruction), mode="eval")
                if torch.is_tensor(requested):
                    requested = requested.detach().float().cpu().numpy()
                requested = np.asarray(requested)[0].astype(np.float32)
                if requested.shape != (32, 14) or not np.isfinite(requested).all():
                    raise ValueError("Invalid requested action chunk")
                chunks.append(requested.copy())
                for action in requested:
                    with contextlib.redirect_stdout(sink):
                        task.take_action(action, action_type="qpos")
                    actions.append(action.copy())
                    chunk_ids.append(len(chunks)-1)
                    observation = task.get_obs()
                    record(observation)
                    if task.eval_success or len(actions) >= task.step_lim:
                        break
                boundaries.append(len(actions))
                print(json.dumps({"task": case["task"], "split": case["split"], "seed_index": case["seed_index"],
                                  "commands": len(actions), "success_so_far": bool(task.eval_success)}), flush=True)
        if not actions or len(actions) != task.take_action_cnt or len(states) != len(actions)+1:
            raise ValueError("Simulator command/observation alignment failed")
        success = bool(task.eval_success)
        mask = np.zeros((len(chunks), 32), dtype=bool)
        for i, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            mask[i, :right-left] = True
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".partial.hdf5")
        with h5py.File(temporary, "x") as episode:
            episode.create_dataset("state", data=np.stack(states[:-1]))
            episode.create_dataset("action", data=np.stack(actions))
            episode.create_dataset("final_observation/state", data=states[-1])
            episode.create_dataset("observation_steps", data=np.arange(len(states), dtype=np.int64))
            episode.create_dataset("requested_action_chunks", data=np.stack(chunks))
            episode.create_dataset("chunk_boundaries", data=np.asarray(boundaries, dtype=np.int64))
            episode.create_dataset("action_chunk_id", data=np.asarray(chunk_ids, dtype=np.int64))
            episode.create_dataset("executed_mask", data=mask)
            rewards = np.zeros(len(actions), dtype=np.float32)
            rewards[-1] = float(success)
            episode.create_dataset("binary_rewards", data=rewards)
            for camera, frames in images.items():
                data = episode.create_dataset(f"images/{camera}", (len(actions),), dtype=h5py.vlen_dtype(np.dtype("uint8")))
                for i, frame in enumerate(frames[:-1]):
                    data[i] = frame
                episode.create_dataset(f"final_observation/images/{camera}", data=frames[-1])
            episode.attrs.update(task=case["task"], instruction=instruction, seed=case["seed"], seed_index=case["seed_index"],
                noise_seed=case["noise_seed"], split=case["split"], success=success, complete=True,
                fps=plan["config"]["fps"], fps_semantics=plan["config"]["fps_semantics"],
                observation_sampling="control_step", final_observation_recorded=True, alignment_verified=True,
                terminated=success, truncated=not success, termination_reason="success" if success else "benchmark_limit",
                episode_limit=int(task.step_lim), elapsed_seconds=time.time()-started,
                behavior_checkpoint_sha256=plan["protocol"]["model"]["sha256"],
                checker_version=plan["sources"][task_config["checker"]],
                scene_family_id=f"robotwin2:demo_clean:{case['task']}:{case['seed']}",
                scene_id_semantics="Task/seed initialization family; not a full-state equivalence proof or pretraining-overlap certification",
                action_semantics="14D qpos targets passed to take_action; actual pre/post joint observations recorded separately",
                initial_observed_fingerprint=hashlib.sha256(json.dumps(initial, sort_keys=True).encode()).hexdigest(),
                initial_observed_state=json.dumps(initial), plan_sha256=digest(destination.parents[2]/"plan.json"))
        temporary.replace(destination)
        return inspect_episode(destination, case, plan)
    finally:
        if task is not None:
            task.close_env()


def run_collection(output, task_name, smoke=False):
    sys.path.insert(0, str(ROOT/"scripts"))
    from evaluate_multitask import load_model
    plan = json.loads((output/"plan.json").read_text())
    for name, sha in plan["sources"].items():
        if digest(ROOT/name) != sha:
            raise ValueError(f"Frozen source changed: {name}")
    if digest(ROOT/plan["protocol"]["model"]["path"]/"model.safetensors") != plan["protocol"]["model"]["sha256"]:
        raise ValueError("Frozen behavior weights changed")
    tasks = {row["name"]: row for row in plan["protocol"]["tasks"]}
    cases = [case for case in plan["cases"] if case["task"] == task_name]
    if not cases:
        raise ValueError("Task is not in the frozen plan")
    if smoke:
        cases = [case for split in plan["config"]["splits"] for case in
                 [r for r in cases if r["split"] == split][:plan["config"]["smoke_cases_per_split"]]]
    mode = "smoke" if smoke else "full"
    with (output/f"{task_name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {"status": "running", "mode": mode, "pid": os.getpid(), "results": [], "scope": plan["scope"]}
        status_path = output/f"{task_name}-{mode}.json"
        save_json(status_path, status)
        try:
            model, status["model_config"] = load_model("native", plan["protocol"])
            for case in cases:
                config = plan["config"]
                if shutil.disk_usage(output).free / 2**30 < config["min_free_gib"]+config["estimated_raw_gib"]:
                    raise RuntimeError("Collection would violate the storage reserve")
                if sum(p.stat().st_size for p in output.rglob("*.hdf5"))/2**30 > config["max_pool_gib"]:
                    raise RuntimeError("Collection exceeded the fixed data budget")
                relative = Path(case["split"])/task_name/f"scene-{case['seed']}.hdf5"
                path = output/relative
                result = inspect_episode(path, case, plan) if path.exists() else collect_episode(model, case, tasks[task_name], plan, path)
                status["results"].append({**result, "file": str(relative)})
                save_json(status_path, status)
            status.update(status="complete", finished_unix=time.time())
        except BaseException as error:
            status.update(status="failed", error=repr(error), failed_case=case if "case" in locals() else None)
            raise
        finally:
            save_json(status_path, status)


def freeze(output):
    plan = json.loads((output/"plan.json").read_text())
    rows = []
    for case in plan["cases"]:
        relative = Path(case["split"])/case["task"]/f"scene-{case['seed']}.hdf5"
        rows.append({**inspect_episode(output/relative, case, plan), "file": str(relative)})
    selected = select_successes(rows, plan["config"]["tasks"], plan["config"]["sft_successes_per_task"])
    counts = Counter((r["task"], r["split"], r["success"]) for r in rows)
    manifest = {**plan, "plan_sha256": digest(output/"plan.json"), "episodes": rows,
                "coverage": [{"task": t, "split": s, "success": success, "episodes": n}
                             for (t, s, success), n in sorted(counts.items())],
                "sft_episode_ids": [r["file"] for r in selected],
                "lerobot_export_ready": False, "full_simulator_state_saved": False}
    if (output/"manifest.json").exists():
        raise FileExistsError("Do not replace a frozen raw data manifest")
    save_json(output/"manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "collect", "freeze"))
    parser.add_argument("--config", type=Path, default=ROOT/"configs/multitask/sft_recap_data.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output.absolute()
    if args.action == "plan":
        result = make_plan(args.config.absolute(), output)
        print(json.dumps({"cases": len(result["cases"]), "raw_output": str(output), "native_sft_index": str(output/"native-index.json")}))
    elif args.action == "collect":
        if args.task is None:
            parser.error("collect needs --task")
        run_collection(output, args.task, args.smoke)
    else:
        result = freeze(output)
        print(json.dumps({"coverage": result["coverage"], "sft_episodes": len(result["sft_episode_ids"])}))


if __name__ == "__main__":
    main()
