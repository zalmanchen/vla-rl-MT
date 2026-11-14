"""Paired native-policy development evaluation before any RL update."""

import argparse
import copy
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from vla_rl_mt.joint_value import BoundaryPool, digest, to_observation, write_json
from vla_rl_mt.shared_value import SharedPrefixValue

ROOT = Path(__file__).resolve().parents[1]


def audit_values(probe, pool, destination):
    from evaluate_method_m1 import decision_advantages, value_metrics

    predictions = []
    with torch.no_grad():
        for start in range(0, len(pool.rows), 4):
            samples = [pool.sample(index)[0] for index in range(start, min(start+4, len(pool.rows)))]
            batch = torch.utils.data.default_collate(samples)
            predictions.extend(probe.value(to_observation(batch, "cuda"), detach_shared=True).cpu().tolist())
    rows, values = [], []
    for index, (row, value) in enumerate(zip(pool.rows, predictions, strict=True)):
        final_chunk = index+1 == len(pool.rows) or pool.rows[index+1]["episode"] != row["episode"]
        rows.append({**row, "episode_limit": 400, "terminal": False, "use_for_loss": True,
                     "chunk_reward": float(row["success"] and final_chunk)})
        values.append(value)
        if final_chunk:
            rows.append({**rows[-1], "step": row["next_step"], "terminal": True,
                         "use_for_loss": False, "chunk_reward": 0.0, "target": 0.0})
            values.append(0.0)  # Prescribed post-terminal bootstrap, not a head prediction.
    values = np.asarray(values)
    advantages = decision_advantages(rows, values)
    train = [i for i, row in enumerate(rows) if row["split"] == "train" and row["use_for_loss"]]
    threshold = float(np.quantile(advantages[train], 0.7))
    report = {"scope": "Prospective binary MC value/one-chunk TD labels; no actor update or oracle action quality",
              "value_batch_size": 4, "metrics": value_metrics(rows, values, (0, 1)),
              "threshold": threshold, "threshold_rule": "M1 rule: unweighted global training-decision quantile 0.7",
              "terminal_values": "fixed zero bootstrap, no terminal loss or head evaluation", "rows": []}
    for index, row in enumerate(rows):
        report["rows"].append({**row, "prediction": float(values[index]),
                              "advantage": float(advantages[index]) if row["use_for_loss"] else None,
                              "positive": bool(advantages[index] >= threshold) if row["use_for_loss"] else None})
    write_json(destination/"value-label-audit.json", report)


def main():
    import cv2
    from evaluate_multitask import evaluate_case, load_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-task", choices=("adjust_bottle", "click_bell"))
    parser.add_argument("--remaining-validation", action="store_true",
                        help="Supplementary diagnostic: remaining fixed M1 val scenes, separate from primary four")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # RoboTwin changes cwd during setup; all later output/checkpoint paths must
    # be pinned before the first simulator call.
    args.output = args.output.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT/"configs/multitask/method_m2.json").read_text())
    pool = BoundaryPool(ROOT/config["pool"], config["pool_manifest_sha256"])
    protocol = pool.manifest["protocol"]
    if digest(ROOT/protocol["model"]["path"]/"model.safetensors") != protocol["model"]["sha256"]:
        raise ValueError("Pinned base model changed")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    mode = checkpoint["identity"]["mode"]
    if (checkpoint["identity"]["pool_manifest_sha256"] != config["pool_manifest_sha256"]
            or checkpoint["identity"]["base_sha256"] != protocol["model"]["sha256"]
            or checkpoint["step"] != config["training_steps"] or mode not in ("w1", "w2")):
        raise ValueError("Wrong M2 checkpoint/pool/step")
    start, end = config["actor_evaluation"]["seed_indices"]
    if args.remaining_validation:
        start, end = end, pool.manifest["config"]["splits"]["val"][1]
    cases = [{key: row[key] for key in ("task", "seed", "seed_index", "noise_seed", "file")}
             for row in pool.manifest["episodes"] if row["split"] == "val" and start <= row["seed_index"] < end]
    if len(cases) != len(config["tasks"])*(end-start):
        raise ValueError("Missing predeclared development scene")
    train_scenes = {(row["task"], row["seed"]) for row in pool.manifest["episodes"] if row["split"] == "train"}
    if train_scenes & {(row["task"], row["seed"]) for row in cases}:
        raise ValueError("Evaluation overlaps training")
    report = {"scope": "M2 pre-RL development evaluation; one policy/noise seed",
              "phase": "supplementary remaining fixed validation scenes" if args.remaining_validation else "primary four scenes/task",
              "scene_indices": [start, end], "supplementary": bool(args.remaining_validation),
              "config": config, "config_sha256": digest(ROOT/"configs/multitask/method_m2.json"),
              "checkpoint_sha256": digest(args.checkpoint), "checkpoint_identity": checkpoint["identity"],
              "cases": cases, "rows": [], "complete": False, "rollouts_return_to_training": False}
    write_json(args.output/"evaluation.json", report)
    model, _ = load_model("native", protocol)
    model.requires_grad_(False)
    task_configs = {row["name"]: row for row in protocol["tasks"]}

    def evaluate(variant, chosen):
        for case in chosen:
            row = evaluate_case(model, case, task_configs[case["task"]], protocol)
            row["variant"] = variant
            if row["status"] == "complete":
                with h5py.File(pool.path/case["file"]) as episode:
                    maximum = float(np.max(np.abs(np.asarray(row["initial_observed_state"]["joints"])-episode["state"][0])))
                row["initial_q_max_abs_vs_pool"] = maximum
                if maximum > 1e-5:
                    raise ValueError("Evaluation initial q differs from the frozen scene")
            report["rows"].append(row)
            write_json(args.output/"evaluation.json", report)
            print(json.dumps({key: row.get(key) for key in ("variant", "task", "seed", "status", "success", "steps")}), flush=True)
            if row["status"] != "complete":
                raise RuntimeError(f"Evaluation failed: {row.get('error')}")

    if args.base_task:
        evaluate("base", [case for case in cases if case["task"] == args.base_task])
    with h5py.File(pool.path/cases[0]["file"]) as episode:
        images = {key: cv2.cvtColor(cv2.imdecode(episode[f"images/{key}"][0], cv2.IMREAD_COLOR),
                                   cv2.COLOR_BGR2RGB) for key in episode["images"]}
        parity_input = {"main_images": images["cam_high"][None],
                        "wrist_images": np.stack([images["cam_left_wrist"], images["cam_right_wrist"]])[None],
                        "states": episode["state"][0][None], "task_descriptions": [str(episode.attrs["instruction"])]}
    torch.manual_seed(cases[0]["noise_seed"])
    with torch.inference_mode():
        reference, _ = model.predict_action_batch(copy.deepcopy(parity_input), mode="eval")
    probe = SharedPrefixValue(model, **checkpoint["architecture"]).to("cuda").eval()
    torch.manual_seed(cases[0]["noise_seed"])
    with torch.inference_mode():
        zero_addon, _ = model.predict_action_batch(copy.deepcopy(parity_input), mode="eval")
    report["zero_adapter_native_action_max_abs"] = float(torch.as_tensor(reference-zero_addon).abs().max())
    if report["zero_adapter_native_action_max_abs"] != 0:
        raise ValueError("Zero addon changes native action inference")
    probe.load_addon(checkpoint["addon"])
    audit_values(probe, pool, args.output)
    evaluate(mode, cases)
    report["complete"] = True
    write_json(args.output/"evaluation.json", report)
    probe.close()


if __name__ == "__main__":
    main()
