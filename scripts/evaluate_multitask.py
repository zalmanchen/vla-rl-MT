"""Plan and run P0 capability calibration with pinned Motus weights.

No training or collection entrypoint is provided. Evaluation records contain
outcomes and initial-state diagnostics, never a reusable training dataset.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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


def wilson(successes, count):
    if not count:
        return None
    z = 1.959963984540054
    rate, z2 = successes / count, z * z
    middle = (rate + z2 / (2 * count)) / (1 + z2 / count)
    radius = z * math.sqrt(rate * (1 - rate) / count + z2 / (4 * count * count)) / (1 + z2 / count)
    return [middle - radius, middle + radius]


def summarize(rows):
    result = {}
    for row in rows:
        item = result.setdefault(row["task"], {"attempted": 0, "valid_rollouts": 0,
                                             "successes": 0, "setup_errors": 0})
        item["attempted"] += 1
        if row["status"] == "complete":
            item["valid_rollouts"] += 1
            item["successes"] += int(row["success"])
        elif row["status"] == "setup_error":
            item["setup_errors"] += 1
    for item in result.values():
        item["rate_valid"] = item["successes"] / item["valid_rollouts"] if item["valid_rollouts"] else None
        item["operational_rate"] = item["successes"] / item["attempted"]
        item["wilson95_valid"] = wilson(item["successes"], item["valid_rollouts"])
    return result


def compare_backends(native, cfg):
    other = {(row["task"], row["seed"]): row for row in cfg}
    result = {"paired_cases": 0, "observed_context_mismatches": [], "by_task": {},
              "scope": "Observed initial joints/poses and noise checks; not a full simulator-state equivalence proof"}
    for row in native:
        key = (row["task"], row["seed"])
        second = other.get(key)
        if second is None or row["status"] != "complete" or second["status"] != "complete":
            continue
        result["paired_cases"] += 1
        if any(row[name] != second[name] for name in
               ("noise_seed", "instruction", "episode_limit", "observed_init_fingerprint")):
            result["observed_context_mismatches"].append({"task": key[0], "seed": key[1]})
        item = result["by_task"].setdefault(key[0], {"paired": 0, "both_success": 0,
                                                   "native_only": 0, "cfg_only": 0, "both_fail": 0})
        item["paired"] += 1
        item["both_success" if row["success"] and second["success"] else
             "native_only" if row["success"] else "cfg_only" if second["success"] else "both_fail"] += 1
    return result


def make_plan(protocol_path, output, smoke=False):
    protocol = json.loads(protocol_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    seed_file = ROOT / protocol["evaluation"]["seed_file"]
    if digest(seed_file) != protocol["evaluation"]["seed_file_sha256"]:
        raise ValueError("Seed file changed")
    seeds = json.loads(seed_file.read_text())
    first, last = protocol["evaluation"]["seed_indices"]
    final_first, final_last = protocol["evaluation"]["reserved_final_seed_indices"]
    if max(first, final_first) < min(last, final_last):
        raise ValueError("Calibration and reserved final-test seed ranges overlap")
    if last - first != protocol["evaluation"]["episodes_per_task"]:
        raise ValueError("Seed range differs from the declared episode budget")
    if protocol["runtime"]["model_action_horizon"] != 32 or protocol["runtime"]["action_dim"] != 14:
        raise ValueError("The pinned Motus model requires a 32-step horizon and 14 environment actions")
    if not 0 < protocol["runtime"]["execute_prefix"] <= 32:
        raise ValueError("Execution prefix must fit the unchanged model horizon")
    cases = []
    for index, task in enumerate(protocol["tasks"]):
        chosen = seeds[task["name"]]["success_seeds"][first:last]
        if len(chosen) != last - first:
            raise ValueError(f"Insufficient calibration seeds: {task['name']}")
        final_seeds = seeds[task["name"]]["success_seeds"][final_first:final_last]
        if len(set(chosen)) != len(chosen) or set(chosen) & set(final_seeds):
            raise ValueError(f"Duplicate or final-test calibration seed: {task['name']}")
        for offset, seed in enumerate(chosen[:1] if smoke else chosen):
            cases.append({"task": task["name"], "seed": int(seed), "seed_index": first + offset,
                          "noise_seed": protocol["evaluation"]["noise_seed_base"] + index * 1000 + offset})
    sources = ["scripts/evaluate_multitask.py", "bootstrap/collect_recap_rollouts.py",
               "bootstrap/smoke_model.py", protocol["runtime"]["compatibility_manifest"],
               protocol["model"]["norm_stats"], "bootstrap/environment.freeze.txt",
               "third_party/RLinf/examples/embodiment/config/model/pi0_5.yaml",
               "third_party/RoboTwin/envs/_base_task.py", "third_party/RoboTwin/task_config/demo_clean.yml",
               "third_party/RoboTwin/task_config/_camera_config.yml",
               "third_party/RoboTwin/assets/embodiments/aloha-agilex/config.yml"]
    patch = json.loads((ROOT / protocol["runtime"]["compatibility_manifest"]).read_text())
    sources += [str(Path(protocol["runtime"]["source"]) / item["path"]) for item in patch["files"]]
    for package in ("openpi", "openpi_cfg"):
        sources += [str(path.relative_to(ROOT)) for path in
                    (ROOT / protocol["runtime"]["source"] / "rlinf/models/embodiment" / package).rglob("*.py")]
    for task in protocol["tasks"]:
        sources += [task["checker"], task["instruction_source"]]
    plan = {"scope": "engineering smoke" if smoke else protocol["scope"], "protocol": protocol,
            "protocol_sha256": digest(protocol_path), "created_unix": time.time(), "smoke": smoke,
            "cases": cases, "sources": {name: digest(ROOT / name) for name in sources},
            "rollouts_return_to_training": False}
    write_json(output / "plan.json", plan)
    return plan


def load_model(backend, protocol):
    import torch
    from omegaconf import OmegaConf

    config = OmegaConf.load(ROOT / "third_party/RLinf/examples/embodiment/config/model/pi0_5.yaml")
    config.model_type = "openpi" if backend == "native" else "cfg_model"
    config.model_path = str(ROOT / protocol["model"]["path"])
    config.precision = "bf16" if backend == "native" else None
    config.num_action_chunks = protocol["runtime"]["model_action_horizon"]
    config.action_dim = protocol["runtime"]["action_dim"]
    config.num_steps = protocol["runtime"]["solver_steps"]
    config.openpi.update({"task": "eval", "config_name": "pi05_aloha_robotwin", "action_horizon": 32,
                          "action_chunk": 32, "action_env_dim": 14, "num_images_in_input": 3,
                          "discrete_state_input": True, "max_token_len": 200, "torch_compile": False,
                          "train_expert_only": False, "guidance_type": "no_guide",
                          "num_steps": protocol["runtime"]["solver_steps"]})
    config.openpi_data = {"norm_stats_path": str(ROOT / protocol["model"]["norm_stats"]),
                          "adapt_to_pi": True, "extra_delta_transform": True}
    if backend == "native":
        from rlinf.models.embodiment.openpi import get_model

        model = get_model(config, torch_dtype=torch.bfloat16)
    else:
        from rlinf.models.embodiment.openpi_cfg import get_model

        model = get_model(config)
    return model.to("cuda").eval(), OmegaConf.to_container(config, resolve=True)


def evaluate_case(model, case, task_config, protocol):
    import numpy as np
    import torch

    sys.path.insert(0, str(ROOT / "bootstrap"))
    from collect_recap_rollouts import env_observation, make_task

    result = {**case, "status": "setup_error", "success": None, "started_unix": time.time(),
              "rollouts_return_to_training": False, "full_simulator_state_saved": False}
    task, stage = None, "setup"
    try:
        # The canonical instruction is installed after resolving scene-specific arm.
        task = make_task(case["task"], case["seed"], "")
        arm = "right" if getattr(task, "qpose_tag", 0) == 1 else "left"
        instruction = task_config["instruction_template"].format(arm=arm)
        task.set_instruction(instruction)
        obs = task.get_obs()
        initial = {"joints": np.asarray(obs["joint_action"]["vector"]).round(6).tolist(), "objects": {}}
        for name in task_config["objects"]:
            pose = getattr(task, name).get_pose()
            initial["objects"][name] = {"p": np.asarray(pose.p).round(6).tolist(),
                                        "q": np.asarray(pose.q).round(6).tolist()}
        result.update(instruction=instruction, initial_observed_state=initial,
                      observed_init_fingerprint=hashlib.sha256(json.dumps(initial, sort_keys=True).encode()).hexdigest(),
                      episode_limit=int(task.step_lim))
        torch.manual_seed(case["noise_seed"])
        steps, calls = 0, 0
        stage = "policy"
        while steps < task.step_lim and not task.eval_success:
            with torch.inference_mode():
                actions, _ = model.predict_action_batch(env_observation(obs, instruction), mode="eval")
            if torch.is_tensor(actions):
                actions = actions.detach().float().cpu().numpy()
            actions = np.asarray(actions)[0].astype(np.float32, copy=False)
            expected = (protocol["runtime"]["model_action_horizon"], protocol["runtime"]["action_dim"])
            if actions.shape != expected or not np.isfinite(actions).all():
                raise ValueError(f"Invalid actions: {actions.shape}")
            calls += 1
            for action in actions[:protocol["runtime"]["execute_prefix"]]:
                task.take_action(action, action_type="qpos")
                steps += 1
                if task.eval_success or steps >= task.step_lim:
                    break
            obs = task.get_obs()
        result.update(status="complete", success=bool(task.eval_success), steps=steps, model_calls=calls,
                      termination="success" if task.eval_success else "benchmark_deadline")
    except Exception as error:
        result.update(status="setup_error" if stage == "setup" else "policy_error", error=repr(error))
    finally:
        if task is not None:
            task.close_env()
        result["elapsed_seconds"] = time.time() - result["started_unix"]
    return result


def run_backend(output, backend):
    plan = json.loads((output / "plan.json").read_text())
    protocol = plan["protocol"]
    if os.environ.get("REPO_PATH") != str(ROOT / "third_party/RLinf"):
        raise RuntimeError("Source bootstrap/env.sh before this RoboTwin evaluation")
    for name, sha in plan["sources"].items():
        if digest(ROOT / name) != sha:
            raise ValueError(f"Frozen source changed: {name}")
    audit_path = output / "audit.json"
    audit = json.loads(audit_path.read_text())
    if not audit["model"]["weights_verified"] or audit["model"]["actual_sha256"] != protocol["model"]["sha256"]:
        raise ValueError("A verified audit.json must precede model evaluation")
    recorded_scenes = {(row["task"], row["seed"]) for row in audit["episodes"] if row.get("seed") is not None}
    if recorded_scenes & {(case["task"], case["seed"]) for case in plan["cases"]}:
        raise ValueError("Calibration overlaps a recorded training/validation scene")
    directory = output / backend
    directory.mkdir(exist_ok=True)
    with (directory / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result_file = directory / "results.json"
        if result_file.exists():
            raise FileExistsError(f"Preserve the existing evaluation: {result_file}; choose a new output plan")
        record = {"phase": "running", "backend": backend, "pid": os.getpid(), "scope": plan["scope"],
                  "plan_sha256": digest(output / "plan.json"),
                  "rollouts_return_to_training": False, "results": [], "started_unix": time.time()}

        def save():
            record["summary"] = summarize(record["results"])
            write_json(result_file, record)

        save()
        try:
            model, config = load_model(backend, protocol)
            write_json(directory / "model-config.json", config)
            tasks = {task["name"]: task for task in protocol["tasks"]}
            for case in plan["cases"]:
                item = evaluate_case(model, case, tasks[case["task"]], protocol)
                record["results"].append(item)
                save()
                print(json.dumps({key: item.get(key) for key in ("task", "seed", "status", "success", "steps")}), flush=True)
                if item["status"] != "complete":
                    raise RuntimeError(item["error"])
            record.update(phase="complete", finished_unix=time.time())
        except Exception as error:
            record.update(phase="failed", error=repr(error), finished_unix=time.time())
            save()
            raise
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "run", "status"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/multitask/phase0.json")
    parser.add_argument("--backend", choices=["native", "cfg"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == "plan":
        plan = make_plan(args.protocol, output, args.smoke)
        print(json.dumps({"output": str(output), "scope": plan["scope"], "cases_per_backend": len(plan["cases"])}))
    elif args.action == "run":
        if not args.backend:
            parser.error("run requires --backend")
        run_backend(output, args.backend)
    else:
        loaded = {}
        for path in sorted(output.glob("*/results.json")):
            data = json.loads(path.read_text())
            loaded[data["backend"]] = data
            print(json.dumps({"backend": data["backend"], "phase": data["phase"], "summary": data["summary"]}, indent=2))
        if "native" in loaded and "cfg" in loaded:
            paired = compare_backends(loaded["native"]["results"], loaded["cfg"]["results"])
            write_json(output / "paired-summary.json", paired)
            print(json.dumps(paired, indent=2))


if __name__ == "__main__":
    main()
