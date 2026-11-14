"""Independent episode-balanced M1 diagnostics, not actor/grounding evaluation."""

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

from vla_rl_mt.value_head import PrefixValueHead, task_episode_weights


def safe_spearman(first, second):
    if len(first) < 2 or np.ptp(first) == 0 or np.ptp(second) == 0:
        return None
    return float(spearmanr(first, second).statistic)


def outcome_auc(predictions, success):
    labels = np.asarray(success, dtype=bool)
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if not positives or not negatives:
        return None
    ranks = rankdata(predictions)
    return float((ranks[labels].sum()-positives*(positives+1)/2)/(positives*negatives))


def episode_balanced_error(rows, prediction, power):
    grouped = defaultdict(list)
    for row, value in zip(rows, prediction, strict=True):
        grouped[(row["task"], row["episode"])].append(abs(float(value)-row["target"])**power)
    by_task = defaultdict(list)
    for (task, _), errors in grouped.items():
        by_task[task].append(float(np.mean(errors)))
    per_task = {task: float(np.mean(errors)) for task, errors in by_task.items()}
    return {"macro": float(np.mean(list(per_task.values()))), "per_task": per_task}


def episode_balanced_mae(rows, prediction):
    return episode_balanced_error(rows, prediction, 1)


def fixed_ridge_predictions(rows, features, regularization=1.0):
    """Exploratory linear capacity control; all statistics use training only."""
    indices = [index for index, row in enumerate(rows) if row["split"] == "train" and row["use_for_loss"]]
    selected_rows = [rows[index] for index in indices]
    weights = task_episode_weights(selected_rows).numpy()
    weights /= weights.sum()
    features = np.asarray(features, dtype=np.float64)
    center = weights@features[indices]
    scale = np.sqrt(weights@((features[indices]-center)**2))
    scale[scale < 1e-6] = 1.0
    design = (features-center)/scale
    targets = np.asarray([row["target"] for row in selected_rows])
    intercept = weights@targets
    weighted = design[indices]*np.sqrt(weights[:, None])
    dual = np.linalg.solve(weighted@weighted.T+regularization*np.eye(len(indices)),
                           (targets-intercept)*np.sqrt(weights))
    return intercept+design@weighted.T@dual


def decision_advantages(rows, values):
    advantages = np.full(len(rows), np.nan)
    for index, row in enumerate(rows):
        if row["terminal"]:
            continue
        if index+1 >= len(rows) or rows[index+1]["episode"] != row["episode"]:
            raise ValueError("Missing true successor or final observation")
        next_value = 0.0 if rows[index+1]["terminal"] else float(values[index+1])
        advantages[index] = row["chunk_reward"]+next_value-float(values[index])
    return advantages


def probability_metrics(rows, predictions):
    """Report clipping explicitly; never mutate values used for TD labels."""
    weights = task_episode_weights(rows).numpy()
    weights /= weights.sum()
    target = np.asarray([float(row["success"]) for row in rows])
    raw = np.asarray(predictions, dtype=np.float64)
    probability = np.clip(raw, 0, 1)
    bins = np.minimum((probability*5).astype(int), 4)
    calibration, ece = [], 0.0
    for group in sorted(set(bins)):
        selected = bins == group
        mass = float(weights[selected].sum())
        predicted = float(weights[selected]@probability[selected]/mass)
        observed = float(weights[selected]@target[selected]/mass)
        ece += mass*abs(predicted-observed)
        calibration.append({"bin": int(group), "mass": mass, "predicted": predicted, "observed": observed})
    return {"clipped_brier": float(weights@((probability-target)**2)),
            "raw_out_of_range_mass": float(weights@((raw < 0) | (raw > 1))),
            "clipped_ece_5_fixed_bins": ece, "clipped_calibration": calibration,
            "scope": "clip to [0,1] for probability reporting only; training/advantage retain raw predictions"}


def success_diagnostics(rows, predictions):
    """Use the same episode outcomes for both reward families, without calibration fits."""
    episodes = defaultdict(list)
    for index, row in enumerate(rows):
        if row["split"] == "val" and row["use_for_loss"]:
            episodes[(row["task"], row["episode"])].append(index)
    views = {"initial": (None, None), "elapsed_0_25": (0.0, 0.25), "elapsed_25_50": (0.25, 0.5),
             "elapsed_50_75": (0.5, 0.75), "elapsed_75_100": (0.75, 1.000001)}
    result = {"views": {}, "initial_cases": [],
              "scope": "one score per episode/view; elapsed views condition on surviving episodes, not independent samples"}
    for view, (left, right) in views.items():
        by_task = defaultdict(list)
        for (task, episode), indices in episodes.items():
            selected = indices[:1] if left is None else [index for index in indices
                        if left <= rows[index]["step"]/rows[index]["episode_limit"] < right]
            if not selected:
                continue
            case = {"task": task, "episode": episode, "success": bool(rows[indices[0]]["success"]),
                    "score": float(np.mean(predictions[selected]))}
            by_task[task].append(case)
            if view == "initial":
                result["initial_cases"].append(case)
        result["views"][view] = {task: {
            "episodes": len(cases), "successes": sum(case["success"] for case in cases),
            "auc": outcome_auc([case["score"] for case in cases], [case["success"] for case in cases])}
            for task, cases in by_task.items()}
    return result


def value_metrics(rows, predictions, target_range=(-1, 0)):
    indices = [index for index, row in enumerate(rows) if row["split"] == "val" and row["use_for_loss"]]
    selected_rows = [rows[index] for index in indices]
    selected = predictions[indices]
    summary = {"episode_balanced_mae": episode_balanced_mae(selected_rows, selected),
               "episode_balanced_mse": episode_balanced_error(selected_rows, selected, 2), "tasks": {}}
    train = [index for index, row in enumerate(rows) if row["split"] == "train" and row["use_for_loss"]]
    edges = np.unique(np.quantile(predictions[train], [0.2, 0.4, 0.6, 0.8]))
    bins = np.searchsorted(edges, selected, side="right")
    weights = task_episode_weights(selected_rows).numpy()
    weights /= weights.sum()
    targets = np.asarray([row["target"] for row in selected_rows])
    calibration, calibration_error = [], 0.0
    for group in np.unique(bins):
        mask = bins == group
        mass = float(weights[mask].sum())
        prediction_mean = float(weights[mask]@selected[mask]/mass)
        target_mean = float(weights[mask]@targets[mask]/mass)
        calibration.append({"bin": int(group), "mass": mass,
                            "prediction_mean": prediction_mean, "target_mean": target_mean})
        calibration_error += mass*abs(prediction_mean-target_mean)
    summary["value_calibration_bins"] = calibration
    summary["value_bin_calibration_error"] = calibration_error
    summary["success_diagnostics"] = success_diagnostics(rows, predictions)
    if target_range == (0, 1):
        summary["binary_probability"] = probability_metrics(selected_rows, selected)
    for task in sorted({row["task"] for row in selected_rows}):
        positions = [index for index in indices if rows[index]["task"] == task]
        target = np.asarray([rows[index]["target"] for index in positions])
        predicted = predictions[positions]
        episode_groups = defaultdict(list)
        for index in positions:
            episode_groups[rows[index]["episode"]].append(index)
        correlations = []
        starts, successes = [], []
        for group in episode_groups.values():
            starts.append(float(predictions[group[0]]))
            successes.append(rows[group[0]]["success"])
            corr = safe_spearman(predictions[group], [rows[index]["target"] for index in group])
            if corr is not None:
                correlations.append(corr)
        summary["tasks"][task] = {
            "episodes": len(episode_groups), "decisions": len(positions),
            "success_episodes": sum(successes), "failure_episodes": len(successes)-sum(successes),
            "bias": float(np.mean(predicted-target)), "prediction_std": float(np.std(predicted)),
            "prediction_min": float(predicted.min()), "prediction_max": float(predicted.max()),
            "outside_normalized_range_fraction": float(((predicted < target_range[0])
                                                         | (predicted > target_range[1])).mean()),
            "target_std": float(np.std(target)), "row_spearman": safe_spearman(predicted, target),
            "mean_within_episode_spearman": float(np.mean(correlations)) if correlations else None,
            "episodes_with_rank_defined": len(correlations), "start_value_outcome_auc": outcome_auc(starts, successes),
        }
    return summary


def compare_labels(first, second, indices):
    a, b = first[indices], second[indices]
    union = int((a | b).sum())
    return {"decisions": len(indices), "agreement": float((a == b).mean()),
            "positive_jaccard": float((a & b).sum()/union) if union else None,
            "both_positive": int((a & b).sum()), "first_only": int((a & ~b).sum()),
            "second_only": int((~a & b).sum()), "both_negative": int((~a & ~b).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, help="Read a new readout cache without changing the frozen pool")
    parser.add_argument("--hidden-width", type=int, help="Explicit head width for a readout study")
    parser.add_argument("--linear-probe", action="store_true",
                        help="Add an explicitly exploratory fixed-alpha=1 linear capacity control")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    feature_dir = args.feature_dir or args.pool
    caches = {condition: torch.load(feature_dir/f"features-{condition}.pt", map_location="cpu", weights_only=True)
              for condition in ("instruction", "no_instruction")}
    first, second = caches.values()
    def identity(row):
        return tuple(row[key] for key in ("episode", "task", "split", "seed", "step", "terminal", "target", "chunk_reward"))
    if [identity(row) for row in first["rows"]] != [identity(row) for row in second["rows"]]:
        raise ValueError("Feature conditions do not share identical sample identities and targets")
    if first["pool_manifest_sha256"] != second["pool_manifest_sha256"]:
        raise ValueError("Different frozen pools across conditions")
    if first.get("readout", "global_mean") != second.get("readout", "global_mean"):
        raise ValueError("Different readouts across instruction conditions")
    target_scheme = first.get("target_scheme", "recap_time_cost")
    if second.get("target_scheme", "recap_time_cost") != target_scheme:
        raise ValueError("Different reward/target schemes across instruction conditions")
    target_range = (0, 1) if target_scheme == "terminal_binary" else (-1, 0)
    rows, config = first["rows"], first["config"]
    hidden_width = args.hidden_width or config["value_head_width"]
    feature_width = first["features"].shape[1]
    train = [index for index, row in enumerate(rows) if row["split"] == "train" and row["use_for_loss"]]
    val = [index for index, row in enumerate(rows) if row["split"] == "val" and row["use_for_loss"]]
    train_rows = [rows[index] for index in train]
    weights = task_episode_weights(train_rows).numpy()
    weights /= weights.sum()
    train_targets = np.asarray([row["target"] for row in train_rows])
    global_mean = float(weights@train_targets)
    task_means = {}
    tasks = sorted({row["task"] for row in rows})
    for task in tasks:
        mask = np.asarray([row["task"] == task for row in train_rows])
        task_means[task] = float(weights[mask]@train_targets[mask]/weights[mask].sum())
    baselines = {"global_mean": np.full(len(rows), global_mean),
                 "task_mean": np.asarray([task_means[row["task"]] for row in rows])}

    def time_features(items):
        return np.asarray([[1.0, row["step"]/row["episode_limit"],
                            *[float(row["task"] == task) for task in tasks[1:]]] for row in items])

    design = time_features(train_rows)
    coefficient = np.linalg.lstsq(design*np.sqrt(weights[:, None]),
                                  train_targets*np.sqrt(weights), rcond=None)[0]
    baselines["task_and_elapsed_time"] = time_features(rows)@coefficient
    result = {"scope": "M1 frozen-feature development diagnostic, no actor update or goal-switch evidence",
              "pool_manifest_sha256": first["pool_manifest_sha256"], "config": config,
              "baselines": {name: value_metrics(rows, prediction, target_range) for name, prediction in baselines.items()},
              "target_scheme": target_scheme, "target_range": target_range,
              "models": {}, "label_comparisons": {}, "actor_updated": False,
              "head": {"feature_width": feature_width, "hidden_width": hidden_width,
                       "readout": first.get("readout", "global_mean")},
              "threshold_rule": "unweighted global quantile of training decision advantages only",
              "primary_checkpoint": "fixed final step; development loss does not select checkpoint"}
    labels, model_predictions = {}, {}
    for condition, cache in caches.items():
        for seed in config["training_seeds"]:
            name = f"{condition}-seed{seed}"
            checkpoint = (args.runs/name/"checkpoints"/f"global_step_{config['training_steps']}"
                          /"actor/model_state_dict/full_weights.pt")
            model = PrefixValueHead(width=feature_width, hidden=hidden_width).eval()
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            with torch.no_grad():
                prediction = model(cache["features"]).predicted_values.numpy()
            if not np.isfinite(prediction).all():
                raise ValueError("Nonfinite value predictions")
            metrics = value_metrics(rows, prediction, target_range)
            metrics["validation_episode_errors"] = [
                {"task": task, "episode": episode,
                 "mae": float(np.mean([abs(float(prediction[index])-row["target"])
                                       for index, row in enumerate(rows)
                                       if row["episode"] == episode and row["use_for_loss"]])),
                 "mse": float(np.mean([(float(prediction[index])-row["target"])**2
                                       for index, row in enumerate(rows)
                                       if row["episode"] == episode and row["use_for_loss"]]))}
                for task, episode in sorted({(rows[index]["task"], rows[index]["episode"]) for index in val})]
            model_predictions[name] = prediction
            advantage = decision_advantages(rows, prediction)
            threshold = float(np.quantile(advantage[train], config["advantage_quantile"]))
            positive = np.nan_to_num(advantage, nan=-np.inf) >= threshold
            labels[name] = positive
            metrics["threshold"] = threshold
            metrics["advantage_by_task_split"] = {}
            for task, split in itertools.product(tasks, ("train", "val")):
                selected = [index for index, row in enumerate(rows)
                            if row["task"] == task and row["split"] == split and row["use_for_loss"]]
                metrics["advantage_by_task_split"][f"{task}/{split}"] = {
                    "decisions": len(selected), "mean": float(advantage[selected].mean()),
                    "std": float(advantage[selected].std()), "positive_fraction": float(positive[selected].mean())}
            metrics["labels_by_outcome_and_boundary"] = {}
            for task, split, success, terminal_next in itertools.product(tasks, ("train", "val"), (False, True), (False, True)):
                indices = [index for index, row in enumerate(rows)
                           if row["task"] == task and row["split"] == split and row["use_for_loss"]
                           and row["success"] == success and rows[index+1]["terminal"] == terminal_next]
                key = f"{task}/{split}/{'success' if success else 'failure'}/{'last_chunk' if terminal_next else 'interior'}"
                metrics["labels_by_outcome_and_boundary"][key] = {
                    "decisions": len(indices), "positive_fraction": float(positive[indices].mean()) if indices else None}
            metrics["checkpoint"] = str(checkpoint)
            metrics["validation_task_mean_relative_mae_reduction"] = (
                1-metrics["episode_balanced_mae"]["macro"]
                /result["baselines"]["task_mean"]["episode_balanced_mae"]["macro"])
            result["models"][name] = metrics
    for a, b in itertools.combinations(labels, 2):
        result["label_comparisons"][f"{a}/{b}"] = compare_labels(labels[a], labels[b], val)
    result["seed_summaries"] = {}
    for condition in caches:
        values = [result["models"][f"{condition}-seed{seed}"]["episode_balanced_mae"]["macro"]
                  for seed in config["training_seeds"]]
        result["seed_summaries"][condition] = {"mae_mean": float(np.mean(values)),
                                                "mae_seed_std": float(np.std(values)), "values": values}
    # Paired scene uncertainty, conditional on these fitted heads and this pool.
    # It does not substitute for independent data-collection/training replications.
    mean_predictions = {condition: [model_predictions[f"{condition}-seed{seed}"]
                                   for seed in config["training_seeds"]] for condition in caches}
    episode_differences = {}
    for task in tasks:
        episodes = sorted({rows[index]["episode"] for index in val if rows[index]["task"] == task})
        differences = []
        for episode in episodes:
            indices = [index for index in val if rows[index]["episode"] == episode]
            target = np.asarray([rows[index]["target"] for index in indices])
            errors = {condition: float(np.mean([np.abs(prediction[indices]-target).mean()
                                               for prediction in predictions]))
                      for condition, predictions in mean_predictions.items()}
            differences.append(errors["instruction"]-errors["no_instruction"])
        episode_differences[task] = np.asarray(differences)
    generator = np.random.default_rng(911)
    draws = np.mean([generator.choice(values, size=(2000, len(values)), replace=True).mean(1)
                     for values in episode_differences.values()], axis=0)
    result["paired_instruction_minus_no_instruction_mae"] = {
        "mean": float(np.mean([values.mean() for values in episode_differences.values()])),
        "scene_bootstrap_ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "scope": "paired development episodes, conditional on three fitted heads per condition"}
    empirical = decision_advantages(rows, np.asarray([row["target"] for row in rows]))
    result["empirical_mc_advantage_max_abs"] = float(np.nanmax(np.abs(empirical)))
    if args.linear_probe:
        result["exploratory_linear_probe"] = {
            "scope": "Post-primary capacity diagnostic; alpha fixed at 1, no validation tuning, not replacing primary heads",
            "regularization": 1.0,
            "conditions": {condition: value_metrics(rows, fixed_ridge_predictions(rows, cache["features"].numpy()), target_range)
                           for condition, cache in caches.items()}}
    result["limitations"] = [
        "Two tasks, 16 training and 12 development-validation scenes per task; unknown overlap with Motus pretraining.",
        "Feature/input ablation does not establish instruction grounding; scenes can identify tasks.",
        "MC returns reflect recorded behavior; fitted Bellman residuals are not action-quality ground truth.",
        "Head-only scalar MSE on frozen policy features differs from the original RECAP distributional critic.",
        "Elapsed-time baseline is an explicit diagnostic; clock is not fed to the feature heads.",
        "Only three head-training seeds, not three independent data-collection seeds or final tests.",
        "Raw error across reward units is not comparable; binary trajectory targets do not provide within-trajectory progress ranks.",
        "Elapsed-time views condition on episode survival; binary rewards do not remove horizon/time dependence.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"baselines": {key: value["episode_balanced_mae"] for key, value in result["baselines"].items()},
                      "seed_summaries": result["seed_summaries"]}, indent=2))


if __name__ == "__main__":
    main()
