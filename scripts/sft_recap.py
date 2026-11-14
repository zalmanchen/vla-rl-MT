"""Portable RoboTwin 2.0 SFT -> public RLinf RECAP orchestration.

Training, return calculation, advantage labeling and checkpointing stay in
the pinned upstream programs. This file handles configuration and handoff.
"""

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import runpy
import shutil
import subprocess
import sys
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import yaml
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party/RLinf"
ENTRYPOINTS = {
    "sft": "examples/sft/train_vla_sft.py",
    "returns": "examples/offline_rl/advantage_labeling/recap/process/compute_returns.py",
    "value": "examples/offline_rl/advantage_labeling/recap/train_value.py",
    "advantages": "examples/offline_rl/advantage_labeling/recap/process/compute_advantages.py",
    "policy": "examples/offline_rl/policy_optimization/cfg_rl/train_cfg.py",
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def merge(base, changes):
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def environment():
    # The shell also loads CUDA/Vulkan/cache settings from bootstrap/env.sh.
    os.environ.update(PROJECT_ROOT=str(ROOT), REPO_PATH=str(UPSTREAM),
                      EMBODIED_PATH=str(UPSTREAM / "examples/embodiment"),
                      ROBOTWIN_PATH=str(ROOT / "third_party/RoboTwin"),
                      RLINF_EXT_MODULE="vla_rl_mt.recap_workflow")
    paths = [ROOT / "src", UPSTREAM, ROOT / "third_party/RoboTwin"]
    for path in reversed(paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ["PYTHONPATH"] = os.pathsep.join(map(str, paths))


def composed(config):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(config.parent), version_base="1.1"):
        return compose(config_name=config.stem)


def evaluation_cases(workflow):
    protocol = json.loads(Path(workflow["evaluation"]["protocol"]).read_text())
    tasks = {task["name"]: task for task in protocol["tasks"]}
    if set(workflow["tasks"]) - tasks.keys():
        raise ValueError("Add audited checker/instruction definitions to the evaluation protocol for new tasks")
    first, last = workflow["evaluation"]["seed_indices"]
    held_first, held_last = protocol["evaluation"]["reserved_final_seed_indices"]
    if first < 0 or last <= first or max(first, held_first) < min(last, held_last):
        raise ValueError("Invalid development range or overlap with reserved final-test seeds")
    seed_path = ROOT / protocol["evaluation"]["seed_file"]
    if digest(seed_path) != protocol["evaluation"]["seed_file_sha256"]:
        raise ValueError("Evaluation seed file changed")
    seeds = json.loads(seed_path.read_text())
    cases = []
    for index, name in enumerate(workflow["tasks"]):
        selected = seeds[name]["success_seeds"][first:last]
        held = seeds[name]["success_seeds"][held_first:held_last]
        if len(selected) != last - first or len(set(selected)) != len(selected) or set(selected) & set(held):
            raise ValueError(f"Insufficient, repeated or final-test development seeds: {name}")
        for offset, seed in enumerate(selected):
            cases.append({"task": name, "seed": int(seed), "seed_index": first + offset,
                          "noise_seed": workflow["evaluation"]["noise_seed_base"] + index * 1000 + offset})
    return protocol, tasks, cases


def plan(config, output=None, gpus=None, seed=None, smoke=False):
    environment()
    document = OmegaConf.load(config)
    if document.schema_version != 1:
        raise ValueError("Unsupported pipeline schema")
    document.workflow.project_root = str(ROOT)
    if output is not None:
        document.workflow.run_dir = str(output.absolute())
    if gpus is not None:
        document.workflow.gpu_ids = gpus
    if seed is not None:
        document.workflow.seed = seed
    if smoke:
        for key in document.workflow.budgets:
            document.workflow.budgets[key] = 2
        document.workflow.warmup_steps = 1
        start = document.workflow.evaluation.seed_indices[0]
        document.workflow.evaluation.seed_indices = [start, start + 1]
    spec = OmegaConf.to_container(document, resolve=True)
    w = spec["workflow"]
    run = Path(w["run_dir"])
    if not run.is_absolute() or run.exists():
        raise ValueError(f"Use a new absolute run directory; existing plans use run/status: {run}")
    ids = w["gpu_ids"]
    if not ids or len(set(ids)) != len(ids) or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("gpu_ids must be distinct nonnegative device ranks")
    if w["batch_size"] <= 0 or w["micro_batch_size"] <= 0 or w["batch_size"] % (len(ids) * w["micro_batch_size"]):
        raise ValueError("Global batch must be divisible by GPU count * micro batch")
    if w["min_free_gib"] < 100 or w["estimated_peak_additional_gib"] < 0:
        raise ValueError("Reserve at least 100 GiB plus the declared artifact peak")
    if not w["tasks"] or len(set(w["tasks"])) != len(w["tasks"]):
        raise ValueError("Declare distinct training tasks")
    if w["return_min"] >= 0 or w["return_max"] != 0 or not 0 < w["gamma"] <= 1 or w["failure_reward"] >= 0:
        raise ValueError("This pipeline uses public DynamicReturn with a shared negative-to-zero scale")
    if any(type(n) is not int or n <= 0 for n in w["budgets"].values()):
        raise ValueError("Every training stage needs a positive fixed budget")
    if not 0 <= w["warmup_steps"] < min(w["budgets"].values()):
        raise ValueError("Warmup must be shorter than each stage")
    evaluation_cases(w)
    run.mkdir(parents=True)
    configs = run / "configs"
    configs.mkdir()
    tag = "recap_" + hashlib.sha256(str(run).encode()).hexdigest()[:12]
    paths = {key: str(run / "data" / key) for key in w["datasets"]}
    train = [{"dataset_path": paths[key], "type": kind, "weight": 1.0,
              "robot_type": "robotwin", "model_type": "pi05"}
             for key, kind in (("sft", "sft"), ("rollout", "rollout"))]
    validation = [{"dataset_path": paths["validation"], "robot_type": "robotwin", "model_type": "pi05"}]
    frozen_sources = ["scripts/sft_recap.py", "scripts/run_sft_recap.sh", "scripts/evaluate_multitask.py",
                      "bootstrap/export_recap_cfg.py", "bootstrap/prepare_sources.py",
                      "src/vla_rl_mt/recap_workflow.py", "bootstrap/patches/recap-robotwin.json",
                      "third_party/sources.manifest.json"]
    for name, entry in spec["stages"].items():
        template = UPSTREAM / entry["template"]
        cfg = merge(yaml.safe_load(template.read_text()), entry["overrides"])
        cfg.setdefault("hydra", {}).update(run={"dir": "."}, output_subdir=None, job={"chdir": False})
        if name in ("sft", "value", "policy"):
            steps = w["budgets"][name + "_steps"]
            merge(cfg, {"cluster": {"num_nodes": 1, "component_placement": {
                "actor,env,rollout": ",".join(map(str, ids))}},
                "runner": {"max_steps": steps, "max_epochs": -1, "save_interval": steps,
                           "val_check_interval": steps if name == "value" else -1,
                           "logger": {"log_path": str(run / name), "experiment_name": "."}},
                "actor": {"seed": w["seed"], "global_batch_size": w["batch_size"],
                          "micro_batch_size": w["micro_batch_size"], "optim": {
                              "total_training_steps": steps, "lr_warmup_steps": w["warmup_steps"]}}})
        if name == "returns":
            cfg["data"].update(tag=tag, train_data_paths=train + [dict(validation[0], type="rollout")])
        elif name == "value":
            cfg["data"].update(tag=tag, train_data_paths=train, eval_data_paths=validation, seed=w["seed"])
        elif name == "advantages":
            cfg["data"].update(train_data_paths=train)
            cfg["advantage"].update(tag=tag, returns_tag=tag, value_checkpoint=str(
                run / f"value/checkpoints/global_step_{w['budgets']['value_steps']}/actor/model_state_dict/full_weights.pt"))
        elif name == "policy":
            cfg["data"].update(advantage_tag=tag, train_data_paths=train, seed=w["seed"])
        target = configs / f"{name}.yaml"
        target.write_text(yaml.safe_dump(cfg, sort_keys=False))
        composed(target)  # CPU-only Hydra composition, without importing a trainer.
        frozen_sources.append(str(template.relative_to(ROOT)))
    sequence = ["sft", "export-cfg", "check-cfg", "eval-sft-native", "eval-cfg-unconditional", "eval-cfg-positive",
                "returns", "value", "advantages", "policy", "eval-recap-positive"]
    frozen_sources += [str(Path(w["evaluation"]["protocol"]).relative_to(ROOT))]
    protocol, tasks, _ = evaluation_cases(w)
    frozen_sources += [protocol["evaluation"]["seed_file"]]
    for task in w["tasks"]:
        frozen_sources += [tasks[task]["checker"], tasks[task]["instruction_source"]]
    manifest = {"schema_version": 1, "scope": "engineering smoke" if smoke else "bounded development SFT-to-RECAP",
                "benchmark": "RoboTwin 2.0", "workflow": w, "tag": tag, "sequence": sequence,
                "source_config_sha256": digest(config), "created_unix": time.time(),
                "sources": {name: digest(ROOT / name) for name in frozen_sources},
                "configs": {name: digest(configs / f"{name}.yaml") for name in spec["stages"]}}
    write_json(run / "plan.json", manifest)
    write_json(run / "status.json", {"phase": "planned", "completed": {}})
    return manifest


def audit_dataset(path, role, workflow):
    """Read real episode columns; provenance must come from an audited converter."""
    import pyarrow.parquet as pq

    path = Path(path)
    info = json.loads((path / "meta/info.json").read_text())
    provenance = json.loads((path / "meta/post_training_provenance.json").read_text())
    if provenance.get("observation_sampling") != "control_step" or provenance.get("schema_version") != 1:
        raise ValueError(f"Need genuine control-step data, not chunk-boundary images: {path}")
    features = info["features"]
    for key in ("action", "observation.state"):
        if features.get(key, {}).get("shape") != [14]:
            raise ValueError(f"Expected 14D absolute robot joints in {path}: {key}")
    for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        if f"observation.images.{camera}" not in features:
            raise ValueError(f"Missing camera {camera}: {path}")
    if info.get("fps", 0) <= 0:
        raise ValueError(f"Missing control frequency: {path}")
    tasks = {row["task_index"]: row["task"] for row in
             (json.loads(line) for line in (path / "meta/tasks.jsonl").read_text().splitlines() if line.strip())}
    episodes = provenance["episodes"]
    indexed = {item["episode_index"]: item for item in episodes}
    if len(indexed) != len(episodes) or not indexed:
        raise ValueError(f"Duplicate or empty provenance episode indices: {path}")
    seen, counts, frame_count = set(), Counter(), 0
    for file in sorted((path / "data").rglob("*.parquet")):
        table = pq.read_table(file, columns=["episode_index", "frame_index", "is_success", "task_index"])
        rows = table.to_pylist()
        for episode in sorted({r["episode_index"] for r in rows}):
            frames = [r for r in rows if r["episode_index"] == episode]
            item = indexed[episode]
            if episode in seen or [r["frame_index"] for r in frames] != list(range(len(frames))):
                raise ValueError(f"Episodes must be complete, ordered and unsplit: {file}")
            seen.add(episode)
            frame_count += len(frames)
            outcomes = [r["is_success"][0] if isinstance(r["is_success"], list) and len(r["is_success"]) == 1
                        else r["is_success"] for r in frames]
            if type(item.get("success")) is not bool or any(type(v) is not bool or v != item["success"] for v in outcomes):
                raise ValueError(f"Unknown or inconsistent success label: {file}")
            if role == "sft" and not item["success"]:
                raise ValueError(f"SFT data must be verified successes: {file}")
            required = ("episode_id", "scene_family_id", "task_id", "instruction", "source_type",
                        "checker_version", "termination_reason")
            if any(not item.get(key) for key in required) or type(item.get("simulator_seed")) is not int:
                raise ValueError(f"Missing episode/scene/instruction/behavior/checker provenance: {file}")
            if item["source_type"] not in ("expert", "policy") or (
                item["source_type"] == "policy" and not item.get("behavior_checkpoint_sha256")):
                raise ValueError(f"Record the policy checkpoint or audited expert source: {file}")
            if item["source_type"] == "expert" and not item.get("source_sha256"):
                raise ValueError(f"Record the original expert data hash: {file}")
            if item["task_id"] not in workflow["tasks"] or item.get("split") != ("validation" if role == "validation" else "train"):
                raise ValueError(f"Unexpected task or split: {file}")
            if any(tasks.get(frame["task_index"]) != item["instruction"] for frame in frames):
                raise ValueError(f"Stored instruction differs from task/provenance metadata: {file}")
            if not item.get("alignment_verified") or not item.get("final_observation_recorded"):
                raise ValueError(f"Action alignment and post-action final observation need auditing: {file}")
            if item["termination_reason"] not in ("success", "failure", "benchmark_limit"):
                raise ValueError(f"External truncations/unknown outcomes cannot be treated as terminal failures: {file}")
            if any(type(item.get(key)) is not bool for key in ("terminated", "truncated")) or (
                item["success"] != (item["termination_reason"] == "success")):
                raise ValueError(f"Inconsistent termination/outcome metadata: {file}")
            if 500 <= item.get("seed_index", -1) < 550:
                raise ValueError(f"Reserved final-test scene in data: {file}")
            lower_bound = -(len(frames) - 1) + (0 if item["success"] else workflow["failure_reward"])
            if lower_bound < workflow["return_min"]:
                raise ValueError(f"Declared return scale would clip an episode: {file}")
            counts[(item["task_id"], item["success"])] += 1
    if seen != indexed.keys() or info.get("total_episodes") != len(seen) or info.get("total_frames") != frame_count:
        raise ValueError(f"Dataset and provenance episode sets differ: {path}")
    if {item["task_id"] for item in episodes} != set(workflow["tasks"]):
        raise ValueError(f"Each configured task must appear in {role}: {path}")
    return {"episodes": episodes, "counts": [{"task": task, "success": success, "episodes": n}
                                               for (task, success), n in sorted(counts.items())]}


def check_inputs(manifest):
    w = manifest["workflow"]
    required = [Path(w[key]) for key in ("base_weights", "norm_stats", "python")]
    for key in ("siglip", "gemma"):
        required += [Path(w[key]) / "config.json", Path(w[key]) / "model.safetensors"]
    for value in w["datasets"].values():
        required += [Path(value) / "meta" / name for name in ("info.json", "tasks.jsonl", "post_training_provenance.json")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing prerequisites:\n" + "\n".join(missing))
    for path, expected in ((Path(w["base_weights"]), w["base_sha256"]), (Path(w["norm_stats"]), w["norm_sha256"])):
        if digest(path) != expected:
            raise ValueError(f"Pinned model or normalization changed: {path}")
    datasets = {role: audit_dataset(path, role, w) for role, path in w["datasets"].items()}
    train = datasets["sft"]["episodes"] + datasets["rollout"]["episodes"]
    validation = datasets["validation"]["episodes"]
    if {(r["task_id"], r["simulator_seed"]) for r in train} & {(r["task_id"], r["simulator_seed"]) for r in validation}:
        raise ValueError("Train/validation simulator seed overlap")
    for key in ("episode_id", "scene_family_id"):
        if {r[key] for r in train} & {r[key] for r in validation}:
            raise ValueError(f"Train/validation {key} overlap")
    _, _, cases = evaluation_cases(w)
    if {(r["task_id"], r["simulator_seed"]) for r in train + validation} & {(r["task"], r["seed"]) for r in cases}:
        raise ValueError("Training/critic-validation overlaps development evaluation seeds")
    if not {r["success"] for r in datasets["rollout"]["episodes"]} == {True, False}:
        raise ValueError("Value preparation needs audited success/failure coverage")
    fingerprints = {}
    for folder in map(Path, w["datasets"].values()):
        for file in sorted(folder.rglob("*")):
            relative = file.relative_to(folder)
            if (file.is_file() and relative.parts[0] in ("data", "videos", "images", "meta")
                    and not any(part.startswith(".") for part in relative.parts)
                    and not file.name.startswith(("returns", "advantages"))):
                fingerprints[str(file)] = digest(file)
    return {"datasets": datasets, "files": fingerprints, "model_sha256": w["base_sha256"]}


def expected_outputs(run, manifest, name):
    if name in ("sft", "value", "policy"):
        step = manifest["workflow"]["budgets"][name + "_steps"]
        return [run / name / f"checkpoints/global_step_{step}/actor/model_state_dict/full_weights.pt"]
    if name == "export-cfg":
        return [run / "cfg-initial/model_state_dict/full_weights.pt", run / "cfg-initial/conversion.json"]
    if name == "check-cfg":
        return [run / "cfg-bridge.json"]
    if name.startswith("eval-"):
        return [run / name / "results.json"]
    keys = ("sft", "rollout", "validation") if name == "returns" else ("sft", "rollout")
    return [run / "data" / key / f"meta/{name}_{manifest['tag']}.parquet" for key in keys]


def check_cfg(run, manifest):
    """Compare unchanged backends on one real observation with identical noise."""
    import gc
    import numpy as np
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from rlinf.models.embodiment.openpi import get_model as native_model
    from rlinf.models.embodiment.openpi_cfg import get_model as cfg_model
    from rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model import Observation

    dataset = LeRobotDataset("sft", root=run / "data/sft")
    sample = dataset[0]
    images = {camera: (sample[f"observation.images.{camera}"].permute(1, 2, 0) * 255).round().to(torch.uint8)
              for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist")}
    obs = {"main_images": images["cam_high"][None], "wrist_images": torch.stack([
        images["cam_left_wrist"], images["cam_right_wrist"]])[None],
        "states": sample["observation.state"][None], "task_descriptions": [sample["task"]]}
    cfg = composed(run / "configs/sft.yaml").actor.model
    cfg.model_path = str(expected_outputs(run, manifest, "sft")[0].parents[1])
    cfg.openpi.task = "eval"
    native = native_model(cfg).to("cuda").eval()
    native_obs = native.env_obs_to_observation(obs)
    torch.manual_seed(17)
    noise = torch.randn(1, 32, 32, device="cuda")
    with torch.inference_mode():
        raw = native.sample_actions(native_obs, num_steps=10, noise=noise.clone())
        native_actions = native.decode_actions(raw, native_obs.state).float().cpu()
    inputs = {"tokens": native_obs.tokenized_prompt.cpu(), "mask": native_obs.tokenized_prompt_mask.cpu(),
              "state": native_obs.state.float().cpu(),
              "images": {k: v.float().cpu() for k, v in native_obs.images.items()}}
    del native, native_obs, raw
    gc.collect()
    torch.cuda.empty_cache()
    cfg = composed(run / "configs/policy.yaml").actor.model
    cfg.openpi.guidance_type = "no_guide"
    legacy = cfg_model(cfg).to("cuda").eval()
    transformed = legacy.input_transform(legacy.obs_processor(obs), transpose=False)
    legacy_obs = Observation.from_dict(legacy.precision_processor(transformed))
    torch.testing.assert_close(inputs["tokens"], legacy_obs.tokenized_prompt.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(inputs["mask"], legacy_obs.tokenized_prompt_mask.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(inputs["state"], legacy_obs.state.float().cpu(), rtol=0, atol=2e-7)
    for key, value in inputs["images"].items():
        torch.testing.assert_close(value, legacy_obs.images[key].float().cpu().permute(0, 2, 3, 1), rtol=0, atol=2e-7)
    with torch.inference_mode():
        raw = legacy.sample_actions(legacy_obs, noise=noise.clone())["actions"]
        actions = torch.as_tensor(np.asarray(legacy.output_transform({"actions": raw, "state": legacy_obs.state})["actions"]))
    delta = (native_actions - actions).abs()
    maximum = delta.max().item()
    if not torch.isfinite(delta).all() or maximum > manifest["workflow"]["bridge_max_action_delta"]:
        raise RuntimeError(f"Native-to-CFG action guard failed: max delta={maximum}; diagnose before RL")
    write_json(run / "cfg-bridge.json", {"status": "passed_input_and_action_guard", "noise_seed": 17,
        "identical_tokens_and_noise": True, "matched_transformed_inputs_atol": 2e-7,
        "max_action_delta": maximum, "mean_action_delta": delta.mean().item(),
        "guard": manifest["workflow"]["bridge_max_action_delta"],
        "scope": "One-observation guard; unchanged backends can differ numerically. Closed-loop CFG controls remain separate."})


def evaluate(run, manifest, name):
    import torch
    import evaluate_multitask as evaluator

    w = manifest["workflow"]
    protocol, tasks, cases = evaluation_cases(w)
    native = name == "eval-sft-native"
    cfg = composed(run / "configs" / ("sft.yaml" if native else "policy.yaml")).actor.model
    if native:
        cfg.model_path = str(expected_outputs(run, manifest, "sft")[0].parents[1])
        cfg.openpi.task = "eval"
        from rlinf.models.embodiment.openpi import get_model
    else:
        cfg.model_path = str(expected_outputs(run, manifest, "policy")[0].parents[1]) if name == "eval-recap-positive" else str(run / "cfg-initial")
        cfg.openpi.guidance_type = "no_guide" if name == "eval-cfg-unconditional" else "positive"
        from rlinf.models.embodiment.openpi_cfg import get_model
    model = get_model(cfg).to("cuda").eval()
    directory = run / name
    directory.mkdir(exist_ok=True)
    record = {"scope": manifest["scope"], "backend": "native" if native else "cfg", "model_config": OmegaConf.to_container(cfg, resolve=True),
              "rollouts_return_to_training": False, "results": []}
    for case in cases:
        result = evaluator.evaluate_case(model, case, tasks[case["task"]], protocol)
        record["results"].append(result)
        record["summary"] = evaluator.summarize(record["results"])
        write_json(directory / "results.json", record)
        print(json.dumps({k: result.get(k) for k in ("task", "seed", "status", "success")}), flush=True)
        if result["status"] != "complete":
            raise RuntimeError(result.get("error", "Evaluation did not complete"))


def stage(run, name):
    environment()
    from vla_rl_mt.recap_workflow import register
    register()
    from rlinf.scheduler import Cluster
    Cluster.NAMESPACE = "sft_recap_" + hashlib.sha256(str(run).encode()).hexdigest()[:12] + "_" + name
    manifest = json.loads((run / "plan.json").read_text())
    if name == "check-cfg":
        check_cfg(run, manifest)
        return
    if name.startswith("eval-"):
        evaluate(run, manifest, name)
        return
    if name == "export-cfg":
        entry = ROOT / "bootstrap/export_recap_cfg.py"
        w = manifest["workflow"]
        sys.argv = [str(entry), "--checkpoint", str(expected_outputs(run, manifest, "sft")[0].parents[1]),
                    "--output", str(run / "cfg-initial"), "--norm-stats", str(run / "assets/norm_stats.json"),
                    "--config-name", "pi05_aloha_robotwin_h32", "--reference-weights", w["base_weights"]]
    else:
        entry = UPSTREAM / ENTRYPOINTS[name]
        sys.argv = [str(entry), "--config-path", str(run / "configs"), "--config-name", name]
    runpy.run_path(str(entry), run_name="__main__")


def execute(run, until=None):
    environment()
    manifest = json.loads((run / "plan.json").read_text())
    state = json.loads((run / "status.json").read_text())
    names = manifest["sequence"]
    if until is not None and until not in names:
        raise ValueError(f"Unknown stage: {until}; choose from {names}")
    selected = names[:names.index(until) + 1] if until else names
    w = manifest["workflow"]
    with ExitStack() as locks:
        lock_dir = ROOT / ".runtime/locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        for file in [run / "workflow.lock", *[lock_dir / f"sft-recap-gpu-{i}.lock" for i in sorted(w["gpu_ids"])]]:
            lock = locks.enter_context(file.open("a+"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, sha in manifest["sources"].items():
            if digest(ROOT / name) != sha:
                raise ValueError(f"Frozen source changed: {name}; use a reviewed new plan")
        for name, sha in manifest["configs"].items():
            if digest(run / "configs" / f"{name}.yaml") != sha:
                raise ValueError(f"Frozen stage config changed: {name}")
        print("Checking pinned sources and input data...", flush=True)
        subprocess.run([w["python"], str(ROOT / "bootstrap/prepare_sources.py"), "--verify-only"], cwd=ROOT, check=True)
        binding = check_inputs(manifest)
        binding_path = run / "input-binding.json"
        if binding_path.exists():
            if json.loads(binding_path.read_text()) != binding:
                raise ValueError("Immutable training inputs changed since the first launch")
        else:
            for role, source in w["datasets"].items():
                destination = run / "data" / role
                shutil.copytree(Path(source) / "meta", destination / "meta",
                                ignore=shutil.ignore_patterns("returns*.parquet", "advantages*.parquet"))
                for child in Path(source).iterdir():
                    if child.name != "meta" and not child.name.startswith("."):
                        (destination / child.name).symlink_to(child.resolve(), target_is_directory=child.is_dir())
            (run / "assets").mkdir()
            shutil.copy2(w["norm_stats"], run / "assets/norm_stats.json")
            write_json(binding_path, binding)
        for name in selected:
            if name in state["completed"]:
                for path, sha in state["completed"][name].items():
                    if not Path(path).is_file() or digest(path) != sha:
                        raise ValueError(f"Completed artifact missing/changed: {path}")
                continue
            outputs = expected_outputs(run, manifest, name)
            if any(path.exists() for path in outputs):
                raise FileExistsError(f"Uncommitted/partial artifacts at {name}; preserve them and use a new reviewed plan")
            reserve = w["min_free_gib"] + w["estimated_peak_additional_gib"]
            if shutil.disk_usage(run).free / 2**30 < reserve:
                raise RuntimeError(f"Need at least {reserve} GiB free before the stage")
            state.update(phase="running", active_stage=name, updated_unix=time.time())
            write_json(run / "status.json", state)
            env = os.environ.copy()
            if name in ("export-cfg", "check-cfg", "advantages") or name.startswith("eval-"):
                env["CUDA_VISIBLE_DEVICES"] = str(w["gpu_ids"][0])
            command = [w["python"], str(ROOT / "scripts/sft_recap.py"), "_stage", "--run-dir", str(run), "--name", name]
            print(f"Starting {name}; log: {run / (name + '.log')}", flush=True)
            try:
                with (run / f"{name}.log").open("a") as log:
                    subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                if any(not path.is_file() for path in outputs):
                    raise RuntimeError(f"Successful process did not produce the expected artifacts: {name}")
                if name in ("sft", "value", "policy"):
                    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
                    events = EventAccumulator(str(run / name / "tensorboard"), size_guidance={"scalars": 0}).Reload()
                    tags = events.Tags()["scalars"]
                    if "train/loss" not in tags or not events.Scalars("train/loss") or any(
                        not math.isfinite(event.value) for tag in tags if tag.startswith("train/")
                        for event in events.Scalars(tag)):
                        raise RuntimeError(f"Missing or nonfinite training metrics: {name}")
                state["completed"][name] = {str(path): digest(path) for path in outputs}
                write_json(run / "status.json", state)
            except BaseException as error:
                state.update(phase="failed", error=str(error), updated_unix=time.time())
                write_json(run / "status.json", state)
                raise
        state.update(phase="complete" if len(state["completed"]) == len(names) else "ready",
                     active_stage=None, updated_unix=time.time())
        state.pop("error", None)
        write_json(run / "status.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "check", "run", "status", "_stage"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pipelines/robotwin2_sft_recap.yaml")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--gpus", nargs="+", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--until", help="Stop after a completed stage; later stages remain pending")
    parser.add_argument("--name", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.action == "plan":
        manifest = plan(args.config, args.run_dir, args.gpus, args.seed, args.smoke)
        print(json.dumps({"scope": manifest["scope"], "run_dir": manifest["workflow"]["run_dir"], "sequence": manifest["sequence"]}, indent=2))
        return
    if args.run_dir is None:
        parser.error("--run-dir is required for check/run/status")
    if args.gpus is not None or args.seed is not None or args.smoke:
        parser.error("GPU/seed/smoke overrides are planning-only; run uses frozen settings")
    run = args.run_dir.absolute()
    if args.action == "_stage":
        stage(run, args.name)
    elif args.action == "run":
        execute(run, args.until)
    elif args.action == "status":
        print((run / "status.json").read_text())
    else:
        environment()
        manifest = json.loads((run / "plan.json").read_text())
        result = check_inputs(manifest)
        print(json.dumps({"status": "data_ready", "counts": {k: v["counts"] for k, v in result["datasets"].items()},
                          "scope": "Data audit only; run also verifies pinned sources, storage and GPU execution"}, indent=2))


if __name__ == "__main__":
    main()
