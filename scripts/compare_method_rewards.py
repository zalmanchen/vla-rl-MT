"""Compare success evidence across reward targets without ranking raw return MAE."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def auc(scores, labels):
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    negatives = len(labels)-positives
    if not positives or not negatives:
        return None
    return float((rankdata(scores)[labels].sum()-positives*(positives+1)/2)/(positives*negatives))


def paired_auc(first, second, seeds=(0, 1, 2)):
    """Initial scene scores; resample the same task/outcome cases for both fits."""
    first_maps = [{(r["task"], r["episode"]): r for r in item} for item in first]
    second_maps = [{(r["task"], r["episode"]): r for r in item} for item in second]
    keys = sorted(first_maps[0])
    if any(sorted(mapping) != keys for mapping in first_maps+second_maps):
        raise ValueError("Initial success cases are not paired")
    labels = np.asarray([first_maps[0][key]["success"] for key in keys])
    if any([mapping[key]["success"] for key in keys] != labels.tolist() for mapping in first_maps+second_maps):
        raise ValueError("Paired success labels changed")
    tasks = np.asarray([key[0] for key in keys])
    predictions = [[np.asarray([mapping[key]["score"] for key in keys]) for mapping in maps]
                   for maps in (first_maps, second_maps)]

    def difference(indices):
        return float(np.mean([auc(a[indices], labels[indices])-auc(b[indices], labels[indices])
                              for a, b in zip(*predictions, strict=True)]))

    result = {}
    for task in sorted(set(tasks)):
        indices = np.flatnonzero(tasks == task)
        strata = [indices[labels[indices] == outcome] for outcome in (False, True)]
        if any(not len(group) for group in strata):
            result[task] = {"mean": None, "ci95": None}
            continue
        generator = np.random.default_rng(911)
        draws = [difference(np.concatenate([generator.choice(group, len(group), replace=True) for group in strata]))
                 for _ in range(2000)]
        result[task] = {"mean": difference(indices), "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
                        "per_seed": [float(auc(predictions[0][i][indices], labels[indices])
                                           - auc(predictions[1][i][indices], labels[indices])) for i in range(len(seeds))]}
    return result


def initial_cases(evaluation, condition):
    return [evaluation["models"][f"{condition}-seed{seed}"]["success_diagnostics"]["initial_cases"]
            for seed in (0, 1, 2)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    binary = json.loads(args.binary.read_text())
    reference = json.loads(args.reference.read_text())
    if (binary["target_scheme"] != "terminal_binary" or reference["target_scheme"] != "recap_time_cost"
            or binary["head"] != reference["head"]
            or binary["pool_manifest_sha256"] != reference["pool_manifest_sha256"]):
        raise ValueError("Reward comparison lacks the matched feature/head/pool contract")
    summary = {"scope": "M1 binary versus RECAP targets, development only, no actor update",
               "pool_manifest_sha256": binary["pool_manifest_sha256"], "results": {},
               "initial_auc_binary_minus_recap": {}, "binary_mse_minus_global_constant": {},
               "actor_updated": False,
               "uncertainty": "paired scenes conditional on these heads; AUC bootstrap fixes observed task/outcome counts"}
    for family, evaluation in (("binary", binary), ("recap", reference)):
        base_mse = evaluation["baselines"]["global_mean"]["episode_balanced_mse"]["macro"]
        result = {"own_target_baseline_mse": {key: value["episode_balanced_mse"]["macro"]
                                            for key, value in evaluation["baselines"].items()}}
        for condition in ("instruction", "no_instruction"):
            models = [evaluation["models"][f"{condition}-seed{seed}"] for seed in (0, 1, 2)]
            mse = [model["episode_balanced_mse"]["macro"] for model in models]
            result[condition] = {"own_target_mse_mean": float(np.mean(mse)), "own_target_mse_seed_std": float(np.std(mse)),
                "own_target_mse_per_seed": mse, "own_target_skill_vs_global": 1-float(np.mean(mse))/base_mse,
                "initial_auc_per_task": {task: [model["success_diagnostics"]["views"]["initial"][task]["auc"]
                                                for model in models]
                                         for task in models[0]["tasks"]}}
            if family == "binary":
                result[condition]["clipped_brier_mean"] = float(np.mean([m["binary_probability"]["clipped_brier"] for m in models]))
                result[condition]["raw_out_of_range_mass_mean"] = float(np.mean([m["binary_probability"]["raw_out_of_range_mass"] for m in models]))
                result[condition]["clipped_ece_mean"] = float(np.mean([m["binary_probability"]["clipped_ece_5_fixed_bins"] for m in models]))
        summary["results"][family] = result
    for condition in ("instruction", "no_instruction"):
        summary["initial_auc_binary_minus_recap"][condition] = paired_auc(initial_cases(binary, condition), initial_cases(reference, condition))
        baseline = {(r["task"], r["episode"]): r for r in binary["baselines"]["global_mean"]["success_diagnostics"]["initial_cases"]}
        errors = [{(r["task"], r["episode"]): r["mse"] for r in binary["models"][f"{condition}-seed{seed}"]["validation_episode_errors"]}
                  for seed in (0, 1, 2)]
        by_task = {task: np.asarray([np.mean([model[key] for model in errors])
                                     - (float(value["success"])-value["score"])**2
                                     for key, value in baseline.items() if key[0] == task])
                   for task in sorted({key[0] for key in baseline})}
        generator = np.random.default_rng(911)
        draws = np.mean([generator.choice(values, (2000, len(values)), replace=True).mean(1) for values in by_task.values()], axis=0)
        summary["binary_mse_minus_global_constant"][condition] = {
            "mean": float(np.mean([v.mean() for v in by_task.values()])),
            "scene_bootstrap_ci95": np.quantile(draws, [0.025, 0.975]).tolist()}
    summary["binary_initial_auc_instruction_minus_no_instruction"] = paired_auc(initial_cases(binary, "instruction"), initial_cases(binary, "no_instruction"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
