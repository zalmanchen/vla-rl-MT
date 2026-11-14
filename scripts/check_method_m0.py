"""Bounded native-Pi0 gradient/memory/checkpoint probe; never starts M1 or RL."""

import argparse
import copy
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
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_development_samples(config, protocol):
    import h5py
    import numpy as np
    from openpi.transforms import compose
    from rlinf.models.embodiment.openpi.transforms.pipeline import (
        build_openpi_transforms,
    )

    sys.path.insert(0, str(ROOT / "bootstrap"))
    from prepare_recap_data import CAMERAS, JOINT_KEYS, decode_jpeg

    entries = json.loads((ROOT / config["development_manifest"]).read_text())["episodes"]
    transforms, _ = build_openpi_transforms(
        str(ROOT / protocol["model"]["path"]), "pi05_aloha_robotwin",
        {"norm_stats_path": str(ROOT / protocol["model"]["norm_stats"]),
         "adapt_to_pi": True, "extra_delta_transform": True},
    )
    transform = compose(transforms)
    tasks = {task["name"]: task for task in protocol["tasks"]}
    samples, records = [], []
    for task in config["tasks"]:
        entry = next(row for row in entries if row["task"] == task and row["source"] == "sft")
        source = Path(entry["path"])
        with h5py.File(source) as episode:
            states = np.concatenate([episode[f"state/{key}"][:] for key in JOINT_KEYS], axis=-1)
            actions = np.concatenate([episode[f"action/{key}"][:32] for key in JOINT_KEYS], axis=-1)
            raw = {"images": {camera: decode_jpeg(episode[f"vision/{native}/colors"][0]).transpose(2, 0, 1)
                              for camera, native in CAMERAS.items()},
                   "state": states[0].astype(np.float32), "actions": actions.astype(np.float32),
                   "prompt": tasks[task]["instruction_template"]}
        if raw["actions"].shape != (32, 14):
            raise ValueError("M0 requires a complete, unpadded source action chunk")
        sample = transform(copy.deepcopy(raw))
        blank = {**copy.deepcopy(raw), "prompt": ""}
        blank_sample = transform(copy.deepcopy(blank))
        changed = copy.deepcopy(raw)
        changed["state"][0] += 0.2
        blank_changed = copy.deepcopy(blank)
        blank_changed["state"][0] += 0.2
        changed_sample = transform(changed)
        blank_changed_sample = transform(blank_changed)
        checks = {
            "instruction_changes_tokens": not np.array_equal(sample["tokenized_prompt"], blank_sample["tokenized_prompt"]),
            "state_changes_tokens": not np.array_equal(sample["tokenized_prompt"], changed_sample["tokenized_prompt"]),
            "state_still_changes_tokens_without_instruction": not np.array_equal(
                blank_sample["tokenized_prompt"], blank_changed_sample["tokenized_prompt"]),
            "removing_instruction_keeps_state": np.array_equal(sample["state"], blank_sample["state"]),
        }
        if not all(checks.values()):
            raise ValueError(f"Unverified language/state input path: {task}: {checks}")
        records.append({"task": task, "source": str(source.relative_to(ROOT)),
                        "source_sha256": digest(source), "frame": 0, "action_frames": [0, 32],
                        "instruction": raw["prompt"], "input_checks": checks,
                        "scene_split_certified": False,
                        "image_shapes": {key: list(value.shape) for key, value in sample["image"].items()},
                        "action_shape": list(sample["actions"].shape),
                        "valid_prompt_tokens": int(np.asarray(sample["tokenized_prompt_mask"]).sum())})
        samples.append(sample)
    return samples, records


def batches(samples, micro, config):
    import numpy as np
    import torch
    from rlinf.models.embodiment.openpi.modules.model import Observation

    noise = torch.randn(len(samples), 32, 32, device="cuda")
    flow_times = torch.tensor(config["flow_times"], device="cuda")

    def stack(values):
        if isinstance(values[0], dict):
            return {key: stack([value[key] for value in values]) for key in values[0]}
        return torch.as_tensor(np.stack(values)).to("cuda")

    result = []
    for start in range(0, len(samples), micro):
        end = min(start + micro, len(samples))
        data = stack(samples[start:end])
        actions = data.pop("actions")
        result.append({"observation": Observation.from_dict(data), "actions": actions,
                       "noise": noise[start:end], "flow_time": flow_times[start:end],
                       "weight": (end - start) / len(samples)})
    return result


def objective(probe, batch, mode, config):
    import torch

    sft, value = None, None
    if mode == "sft" or mode.startswith("joint"):
        sft = probe.sft_loss(batch["observation"], batch["actions"],
                             noise=batch["noise"], flow_time=batch["flow_time"])
    if mode != "sft":
        prediction = probe.value(batch["observation"], detach_shared=mode.endswith("detached"))
        value = torch.square(prediction - config["value_target"]).mean()
    total = (sft if sft is not None else 0) + ((config["lambda_value"] if mode.startswith("joint") else 1)
                                              * value if value is not None else 0)
    return total, sft, value


def state_difference(first, second, module):
    import torch

    differences = [first[module][key] - second[module][key] for key in first[module]]
    return {"l2": sum(float(value.float().square().sum()) for value in differences) ** 0.5,
            "max_abs": max(float(value.abs().max()) for value in differences),
            "exact": all(torch.equal(first[module][key], second[module][key]) for key in first[module])}


def update(probe, data, optimizer, mode, config):
    import torch

    from vla_rl_mt.shared_value import gradient_inventory, isolated_clip

    optimizer.zero_grad(set_to_none=True)
    losses = {"total": 0.0, "sft": 0.0, "value": 0.0}
    for batch in data:
        total, sft, value = objective(probe, batch, mode, config)
        if not bool(torch.isfinite(total)):
            raise ValueError(f"Nonfinite objective: {mode}")
        (total * batch["weight"]).backward()
        for name, item in (("total", total), ("sft", sft), ("value", value)):
            if item is not None:
                losses[name] += float(item.detach()) * batch["weight"]
    gradients = gradient_inventory(probe)
    if not all(row["finite"] for row in gradients.values()):
        raise ValueError(f"Nonfinite gradients: {mode}")
    if gradients["frozen_policy"]["gradient_tensors"]:
        raise ValueError("Frozen policy received gradients")
    isolated_clip(probe)
    optimizer.step()
    return {"losses": losses, "gradients": gradients}


def run(args, report):
    import torch
    from evaluate_multitask import load_model

    from vla_rl_mt.shared_value import SharedPrefixValue, isolated_optimizer

    config = json.loads(args.config.read_text())
    protocol = json.loads((ROOT / config["initialization_protocol"]).read_text())
    if args.micro_batch_size not in (1, 2) or config["global_batch_size"] != len(config["tasks"]):
        raise ValueError("M0 supports only micro=1/2 and the declared two-example global batch")
    weights = ROOT / protocol["model"]["path"] / protocol["model"]["weights"]
    if digest(weights) != protocol["model"]["sha256"]:
        raise ValueError("Base model checksum mismatch")
    if digest(ROOT / protocol["model"]["norm_stats"]) != protocol["model"]["norm_stats_sha256"]:
        raise ValueError("Normalization checksum mismatch")
    torch.manual_seed(config["seed"])
    report.update(config=config, config_sha256=digest(args.config),
                  model_sha256=protocol["model"]["sha256"],
                  micro_batch_size=args.micro_batch_size, global_batch_size=config["global_batch_size"],
                  gradient_accumulation_steps=config["global_batch_size"] // args.micro_batch_size,
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  distributed_training=False, source_lock=json.loads((ROOT / "bootstrap/sources.lock.json").read_text()),
                  source_manifest_sha256=digest(ROOT / "third_party/sources.manifest.json"),
                  addon_source_sha256=digest(ROOT / "src/vla_rl_mt/shared_value.py"),
                  probe_source_sha256=digest(Path(__file__)),
                  value_target_is_synthetic=True, formal_data_ready=False)
    samples, report["samples"] = load_development_samples(config, protocol)
    save_json(args.output / "report.json", report)
    print("Loading pinned native Motus policy", flush=True)
    policy, report["native_model_config"] = load_model("native", protocol)
    # Compare the adapter intervention with identical parameter/forward flags.
    # In particular, native eval and checkpointed training are different paths.
    policy.requires_grad_(False).train()
    policy.img.eval()
    if config["gradient_checkpointing"]:
        policy.gradient_checkpointing_enable({"use_reentrant": False})
    data = batches(samples, args.micro_batch_size, config)
    with torch.no_grad():
        reference = [policy.compute_loss(batch["observation"], batch["actions"], train=False,
                                         noise=batch["noise"], time=batch["flow_time"]).cpu() for batch in data]
    probe = SharedPrefixValue(policy, rank=config["adapter_rank"], head_width=config["value_head_width"]).to("cuda").train()
    with torch.no_grad():
        adapted = [policy.compute_loss(batch["observation"], batch["actions"], train=False,
                                       noise=batch["noise"], time=batch["flow_time"]).cpu() for batch in data]
    report["zero_adapter_native_fm_max_abs"] = max(float((a-b).abs().max()) for a, b in zip(reference, adapted, strict=True))
    if report["zero_adapter_native_fm_max_abs"] != 0:
        raise ValueError("Zero adapter changed native flow-matching output")
    initial = probe.addon_state()
    report["parameters"] = {"frozen_base": sum(p.numel() for p in policy.parameters()),
                            "adapter": sum(p.numel() for p in probe.adapter.parameters()),
                            "value_head": sum(p.numel() for p in probe.value_head.parameters()),
                            "action_expert_trainable": False, "vlm_trainable": False}
    report["modes"] = {}
    states = {}
    for mode in config["modes"]:
        probe.load_addon(initial)
        optimizer = isolated_optimizer(probe, config["learning_rate"])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        item = update(probe, data, optimizer, mode, config)
        torch.cuda.synchronize()
        item.update(seconds=time.monotonic()-started,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
        states[mode] = probe.addon_state()
        item["parameter_changes"] = {name: state_difference(states[mode], initial, name)
                                     for name in ("adapter", "value_head")}
        gradients = item["gradients"]
        expected_adapter = mode != "value_detached"
        expected_head = mode != "sft"
        for name, expected in (("adapter", expected_adapter), ("value_head", expected_head)):
            if (gradients[name]["gradient_l2"] > 0) != expected:
                raise ValueError(f"Incorrect gradient route: {mode}/{name}")
            if (item["parameter_changes"][name]["l2"] > 0) != expected:
                raise ValueError(f"Incorrect parameter update: {mode}/{name}")
        report["modes"][mode] = item
        save_json(args.output / "report.json", report)
        print(json.dumps({"mode": mode, "seconds": item["seconds"],
                          "peak_allocated_gib": item["peak_allocated_gib"],
                          "adapter_gradient_l2": gradients["adapter"]["gradient_l2"]}), flush=True)
    report["sft_vs_joint_detached_adapter"] = state_difference(states["sft"], states["joint_detached"], "adapter")
    if not report["sft_vs_joint_detached_adapter"]["exact"]:
        raise ValueError("Detached value fitting changed the SFT adapter update")
    checkpoint = {"kind": "M0 addon, requires pinned frozen base", "base_sha256": report["model_sha256"],
                  "architecture": probe.architecture, "addon": probe.addon_state(), "optimizer": optimizer.state_dict()}
    checkpoint_path = args.output / "addon.pt"
    torch.save(checkpoint, checkpoint_path)
    with torch.no_grad():
        before_reload = [float(objective(probe, batch, "joint_shared", config)[0]) for batch in data]
    update(probe, data, optimizer, "joint_shared", config)
    expected_next_state = probe.addon_state()
    probe.close()
    reloaded = SharedPrefixValue(policy, **checkpoint["architecture"]).to("cuda").train()
    stored = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if stored["base_sha256"] != report["model_sha256"]:
        raise ValueError("Wrong frozen base for addon checkpoint")
    reloaded.load_addon(stored["addon"])
    resumed_optimizer = isolated_optimizer(reloaded, config["learning_rate"])
    resumed_optimizer.load_state_dict(stored["optimizer"])
    with torch.no_grad():
        after_reload = [float(objective(reloaded, batch, "joint_shared", config)[0]) for batch in data]
    update(reloaded, data, resumed_optimizer, "joint_shared", config)
    resumed_state = reloaded.addon_state()
    report["checkpoint"] = {"bytes": checkpoint_path.stat().st_size, "sha256": digest(checkpoint_path),
                            "reload_objective_max_abs": max(abs(a-b) for a, b in zip(before_reload, after_reload, strict=True)),
                            "next_update": {name: state_difference(expected_next_state, resumed_state, name)
                                            for name in ("adapter", "value_head")}}
    if report["checkpoint"]["reload_objective_max_abs"] > 1e-6 or any(
            row["max_abs"] > 1e-6 for row in report["checkpoint"]["next_update"].values()):
        raise ValueError("Checkpoint reload/resumed optimizer update mismatch")
    reloaded.close()
    report.update(status="passed", finished_unix=time.time(),
                  limitations=["Synthetic value target; not critic quality or policy benefit.",
                               "Frozen original action expert and VLM; addon-only checkpoint.",
                               "No augmentation; fixed noise/time; input data already resident, not end-to-end training throughput.",
                               "Standalone single-GPU probe, not distributed RLinf training or RoboTwin rollout validation."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/multitask/method_m0.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--micro-batch-size", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if os.environ.get("REPO_PATH") != str(ROOT / "third_party/RLinf"):
        parser.error("Source bootstrap/env.sh before running this probe")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "scope": "M0 engineering only", "started_unix": time.time()}
    try:
        run(args, report)
    except Exception as error:
        report.update(status="failed", error=repr(error), finished_unix=time.time())
        raise
    finally:
        save_json(args.output / "report.json", report)
    print(json.dumps({"status": report["status"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
