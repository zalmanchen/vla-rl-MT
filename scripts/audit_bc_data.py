"""Independently audit frozen BC commands, behavior provenance and simulator replay."""

import argparse
import contextlib
import copy
import json
import os
import sys
from itertools import pairwise
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"bootstrap"))
from cache_method_m1 import validate_pool
from collect_recap_rollouts import CAMERAS, make_task

from vla_rl_mt.joint_value import digest, executed_chunk, write_json


def images_at(episode, index):
    return {key: cv2.cvtColor(cv2.imdecode(episode[f"images/{key}"][index], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            for key in CAMERAS}


def environment_input(images, state, instruction):
    return {"main_images": images["cam_high"][None],
            "wrist_images": np.stack([images["cam_left_wrist"], images["cam_right_wrist"]])[None],
            "states": state[None], "task_descriptions": [instruction]}


def recompute(pool, manifest, output, tolerance):
    from evaluate_multitask import load_model
    from openpi.transforms import compose
    from rlinf.models.embodiment.openpi.transforms.pipeline import (
        build_openpi_transforms,
    )

    report = {"scope": "All frozen episodes, teacher-forced recorded observations; no training/collection",
              "manifest_sha256": digest(pool/"manifest.json"), "structural": validate_pool(pool, manifest),
              "behavior": "Original native Motus; same model flags, per-episode noise seed, all recorded decisions",
              "command_tolerance": tolerance, "rows": [], "complete": False}
    protocol = manifest["protocol"]
    model, _ = load_model("native", protocol)
    if digest(ROOT/protocol["model"]["path"]/"model.safetensors") != protocol["model"]["sha256"]:
        raise ValueError("Behavior checkpoint changed")
    transforms, inverse = build_openpi_transforms(str(ROOT/protocol["model"]["path"]), "pi05_aloha_robotwin",
        {"norm_stats_path": str(ROOT/protocol["model"]["norm_stats"]), "adapt_to_pi": True, "extra_delta_transform": True})
    transform, inverse = compose(transforms), compose(inverse)
    for entry in manifest["episodes"]:
        with h5py.File(pool/entry["file"]) as episode:
            if episode.attrs["source"] != "frozen_native_motus_pi05":
                raise ValueError("Unknown behavior source")
            boundaries = episode["observation_steps"][:]
            row = {"episode": entry["file"], "task": entry["task"], "split": entry["split"],
                   "success": entry["success"], "noise_seed": int(episode.attrs["noise_seed"]),
                   "command_max_abs": 0., "roundtrip_max_abs": 0., "decisions": len(boundaries)-1,
                   "max_boundary_tracking_error": 0.}
            torch.manual_seed(row["noise_seed"])
            for index, (left, right) in enumerate(pairwise(boundaries)):
                images = images_at(episode, index)
                state = episode["state"][index].astype(np.float32)
                expected = episode["actions"][left:right]
                if not np.isfinite(state).all() or not np.isfinite(expected).all():
                    raise ValueError("Nonfinite state or executed command")
                with torch.inference_mode():
                    prediction, _ = model.predict_action_batch(environment_input(images, state, episode.attrs["instruction"]), mode="eval")
                actual = prediction[0, :right-left].float().cpu().numpy()
                row["command_max_abs"] = max(row["command_max_abs"], float(np.abs(actual-expected).max()))
                padded, mask = executed_chunk(expected)
                raw = {"images": {key: value.transpose(2, 0, 1) for key, value in images.items()},
                       "state": state, "actions": padded, "prompt": str(episode.attrs["instruction"])}
                normalized = transform(copy.deepcopy(raw))
                recovered = inverse({"state": normalized["state"], "actions": normalized["actions"]})["actions"]
                row["roundtrip_max_abs"] = max(row["roundtrip_max_abs"], float(np.abs(recovered[:right-left, :14]-expected).max()))
                if int(mask.sum()) != int(right-left)*14:
                    raise ValueError("Physical action mask differs")
                # Tracking is descriptive: qpos commands are not measured joint states.
                row["max_boundary_tracking_error"] = max(row["max_boundary_tracking_error"],
                    float(np.abs(episode["state"][index+1]-expected[-1]).max()))
            report["rows"].append(row)
        write_json(output/"report.json", report)
        print(json.dumps(row), flush=True)
    report["passed"] = all(row["command_max_abs"] <= tolerance and row["roundtrip_max_abs"] <= tolerance for row in report["rows"])
    report["complete"] = True
    write_json(output/"report.json", report)
    if not report["passed"]:
        raise ValueError("Behavior or action roundtrip audit failed; preserve report and diagnose before new data")


def replay(pool, manifest, output, comparison):
    cohort = json.loads(comparison.read_text())
    rows = cohort["rows"]
    reference = {(r["task"], r["seed"]): r for r in rows if r["curve_point"] == "w1_plain"}
    new = {(r["task"], r["seed"]): r for r in rows if r["curve_point"] == "expert_lr_1e6_step128"}
    lost = {key for key in reference if reference[key]["success"] and not new[key]["success"]}
    selected = [r for r in manifest["episodes"] if (r["task"], r["seed"]) in lost]
    for task in sorted({r["task"] for r in manifest["episodes"]}):
        selected.append(next(r for r in manifest["episodes"] if r["task"] == task and r["split"] == "train" and r["success"]))
    report = {"scope": "Original executed commands replay, four lost scenes plus two fixed successful anchors",
              "manifest_sha256": digest(pool/"manifest.json"), "comparison_sha256": digest(comparison),
              "full_simulator_state_restored": False, "rows": [], "complete": False,
              "rollouts_return_to_training": False}
    for entry in selected:
        task = None
        with h5py.File(pool/entry["file"]) as episode, open(os.devnull, "w") as sink:
            try:
                task = make_task(entry["task"], entry["seed"], str(episode.attrs["instruction"]))
                row = {"episode": entry["file"], "task": entry["task"], "seed": entry["seed"],
                       "original_success": entry["success"], "status": "running", "boundary_qpos_max_abs": 0.,
                       "image_mae_max": 0., "image_max_abs": 0, "boundary_count": 0}
                boundaries = episode["observation_steps"][:]
                for index, step in enumerate(boundaries):
                    obs = task.get_obs()
                    q = np.asarray(obs["joint_action"]["vector"], dtype=np.float32)
                    row["boundary_qpos_max_abs"] = max(row["boundary_qpos_max_abs"], float(np.abs(q-episode["state"][index]).max()))
                    expected_images = images_at(episode, index)
                    for key, camera in CAMERAS.items():
                        difference = np.abs(np.asarray(obs["observation"][camera]["rgb"]).astype(np.int16)-expected_images[key].astype(np.int16))
                        row["image_mae_max"] = max(row["image_mae_max"], float(difference.mean()))
                        row["image_max_abs"] = max(row["image_max_abs"], int(difference.max()))
                    row["boundary_count"] += 1
                    if index == len(boundaries)-1:
                        break
                    for action in episode["actions"][step:boundaries[index+1]]:
                        with contextlib.redirect_stdout(sink):
                            task.take_action(action, action_type="qpos")
                row.update(status="complete", success=bool(task.eval_success), executed_actions=int(task.take_action_cnt),
                           original_actions=len(episode["actions"]), checker_success=bool(task.check_success()))
                row["outcome_and_count_match"] = row["success"] == entry["success"] and row["executed_actions"] == row["original_actions"]
                report["rows"].append(row)
            finally:
                if task is not None:
                    task.close_env()
        write_json(output/"report.json", report)
        print(json.dumps(row), flush=True)
    report["complete"] = True
    report["all_outcomes_and_counts_match"] = all(r["outcome_and_count_match"] for r in report["rows"])
    write_json(output/"report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("recompute", "replay"))
    parser.add_argument("--pool", type=Path, default=ROOT/".runtime/datasets/method-m1-v1")
    parser.add_argument("--comparison", type=Path, default=ROOT/".runtime/runs/method/bc-stability-work-v1/final-comparison.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    pool, output, comparison = args.pool.resolve(), args.output.resolve(), args.comparison.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((pool/"manifest.json").read_text())
    if args.mode == "recompute":
        recompute(pool, manifest, output, args.tolerance)
    else:
        replay(pool, manifest, output, comparison)


if __name__ == "__main__":
    main()
