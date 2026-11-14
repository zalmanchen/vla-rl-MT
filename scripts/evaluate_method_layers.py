"""Audit four frozen-feature probes, certify W0 and export one candidate's labels."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from vla_rl_mt.joint_value import write_json
from vla_rl_mt.layer_value import LayerValueHead

ROOT = Path(__file__).resolve().parents[1]


def main():
    from evaluate_method_m1 import decision_advantages, value_metrics

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT/"configs/multitask/method_m4.json").read_text())
    execution = json.loads((args.runs/"execution.json").read_text())
    if not execution["complete"] or any(row["exit_code"] for row in execution["jobs"]):
        raise ValueError("All declared heads must finish before selecting a layer")
    result = {"scope": "bounded layer diagnosis, frozen W1 actor; development selection, no policy/RL conclusion",
              "models": {}, "selected_candidate": None, "all_samples_matched_by_seed": True}
    reference_sequences = {}
    for layer in config["layers"]:
        cache = torch.load(args.features/f"layer-{layer:02d}.pt", weights_only=True, map_location="cpu")
        rows, features = cache["rows"], cache["features"]
        for seed in config["head_seeds"]:
            name = f"layer{layer:02d}-seed{seed}"
            directory = args.runs/name
            checkpoint = next((directory/"checkpoints/global_step_128/actor").rglob("full_weights.pt"))
            model = LayerValueHead(4096, 63, 0.1)
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
            model.eval()
            data = json.loads((directory/"feature-data.json").read_text())
            reference_sequences.setdefault(seed, data["sample_sequence_sha256"])
            if reference_sequences[seed] != data["sample_sequence_sha256"]:
                raise ValueError("Different sampling across readout layers")
            with torch.no_grad():
                prediction = model(features).predicted_values.numpy()
            metrics = value_metrics(rows, prediction, (0, 1))
            advantage = decision_advantages(rows, prediction)
            train = [i for i, row in enumerate(rows) if row["split"] == "train" and row["use_for_loss"]]
            threshold = float(np.quantile(advantage[train], 0.7))
            labels = [{**row, "prediction": float(prediction[i]),
                       "advantage": float(advantage[i]) if row["use_for_loss"] else None,
                       "positive": bool(advantage[i] >= threshold) if row["use_for_loss"] else None}
                      for i, row in enumerate(rows)]
            write_json(args.output/f"{name}-labels.json", {"rows": labels, "metrics": metrics, "threshold": threshold,
                "source_checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "scope": "Frozen W1, fresh sequential head; prospective TD labels, not oracle action quality"})
            result["models"][name] = {"mae": metrics["episode_balanced_mae"]["macro"],
                                       "mse": metrics["episode_balanced_mse"]["macro"],
                                       "per_task": metrics["episode_balanced_mse"]["per_task"],
                                       "initial_auc": metrics["success_diagnostics"]["views"]["initial"],
                                       "raw_out_of_range_mass": metrics["binary_probability"]["raw_out_of_range_mass"]}
    result["mean_mse_by_layer"] = {str(layer): float(np.mean([result["models"][f"layer{layer:02d}-seed{seed}"]["mse"]
        for seed in config["head_seeds"]])) for layer in config["layers"]}
    winner = min(config["layers"], key=lambda layer: result["mean_mse_by_layer"][str(layer)])
    wins = sum(result["models"][f"layer{winner:02d}-seed{seed}"]["mse"] < result["models"][f"layer18-seed{seed}"]["mse"]
               for seed in config["head_seeds"])
    if winner != 18 and wins >= 2:
        result["selected_candidate"] = winner
    parity = json.loads((ROOT/".runtime/runs/method/w0-actor-v1/parity.json").read_text())
    if not parity["actor_exact"] or not parity["fresh_head_init_matches_m2_exact"]:
        raise ValueError("W0 actor/init parity failed")
    result["w0_actor_full_128_step_exact"] = True
    result["w0_sequential_head_complete"] = True
    result["candidate_seed_for_policy"] = 0  # Never choose the best head seed.
    write_json(args.output/"comparison.json", result)
    print(json.dumps({key: result[key] for key in ("mean_mse_by_layer", "selected_candidate", "w0_sequential_head_complete")}), flush=True)


if __name__ == "__main__":
    main()
