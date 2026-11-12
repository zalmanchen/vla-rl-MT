"""Audit actual RECAP labels, checkpoints, parameter updates and paired evaluation."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROLLOUT_INSTRUCTIONS = {
    "click_bell": "Press the top center of the bell.",
    "stack_blocks_two": "Stack the green block on top of the red block.",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    run = args.run.resolve()
    state = json.loads((run / "status.json").read_text())
    required = {"returns", "value", "advantages", "policy"}
    if not required.issubset(state["completed_stages"]):
        raise ValueError(f"Four stages incomplete: {state['completed_stages']}")
    report = {
        "status": "passed",
        "run_dir": str(run),
        "tag": state["tag"],
        "scope": "execution and data correctness; no claim of policy improvement or paper reproduction",
        "datasets": {},
        "checks": {},
        "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    tag = state["tag"]
    total_positive = 0
    total_negative = 0
    collection_scenarios = set()
    train_labels = {}
    for split in ["sft-train", "sft-val", "rollout-train", "rollout-val"]:
        meta = ROOT / "datasets/lerobot/recap-pilot-v2" / split / "meta"
        manifest = json.loads((meta / "conversion.json").read_text())
        if split.startswith("rollout"):
            collection_scenarios.update(
                (entry["task"], int(Path(entry["path"]).stem.split("_")[-1]))
                for entry in manifest["episodes"]
            )
        table = pq.read_table(meta / f"returns_{tag}.parquet").to_pandas()
        if (
            len(table) != manifest["frame_count"]
            or table.duplicated(["episode_index", "frame_index"]).any()
        ):
            raise ValueError(f"Return sidecar coverage mismatch: {split}")
        for index, entry in enumerate(manifest["episodes"]):
            if (
                split.startswith("rollout")
                and entry["instruction"] != EXPECTED_ROLLOUT_INSTRUCTIONS[entry["task"]]
            ):
                raise ValueError(
                    f"Unverified rollout instruction: {split} episode {index}"
                )
            episode = table[table.episode_index == index].sort_values("frame_index")
            n = entry["frames"]
            np.testing.assert_array_equal(episode.frame_index.to_numpy(), np.arange(n))
            reward = episode.reward.to_numpy()
            np.testing.assert_array_equal(reward[:-1], np.full(n - 1, -1.0))
            expected_terminal = 0.0 if entry["success"] else -300.0
            if reward[-1] != expected_terminal:
                raise ValueError(f"Incorrect terminal reward: {split} episode {index}")
            np.testing.assert_allclose(
                episode["return"].to_numpy(), np.cumsum(reward[::-1])[::-1]
            )
        if table["return"].min() < -1200 or table["return"].max() > 0:
            raise ValueError(f"Return outside configured value range: {split}")
        stats = {
            "episodes": manifest["episode_count"],
            "frames": len(table),
            "successes": sum(x["success"] for x in manifest["episodes"]),
        }
        if split.endswith("train"):
            advantages = pq.read_table(meta / f"advantages_{tag}.parquet").to_pandas()
            if (
                len(advantages) != len(table)
                or advantages.duplicated(["episode_index", "frame_index"]).any()
            ):
                raise ValueError(f"Advantage sidecar coverage mismatch: {split}")
            keys = ["episode_index", "frame_index"]
            if set(map(tuple, advantages[keys].to_numpy())) != set(
                map(tuple, table[keys].to_numpy())
            ):
                raise ValueError(f"Advantage indices differ: {split}")
            if not np.isfinite(
                advantages.select_dtypes(include="number").to_numpy()
            ).all():
                raise ValueError(f"Nonfinite advantages: {split}")
            if not advantages.advantage.isin([True, False]).all():
                raise ValueError(f"Nonboolean labels: {split}")
            np.testing.assert_allclose(
                advantages.advantage_continuous,
                advantages.reward_sum
                + advantages.value_next
                - advantages.value_current,
                atol=1e-6,
            )
            task_counts = {}
            for index, entry in enumerate(manifest["episodes"]):
                episode = advantages[advantages.episode_index == index].sort_values(
                    "frame_index"
                )
                rewards = (
                    table[table.episode_index == index]
                    .sort_values("frame_index")
                    .reward.to_numpy()
                )
                expected_sums = np.array(
                    [rewards[i : i + 32].sum() for i in range(len(rewards))]
                )
                np.testing.assert_allclose(episode.reward_sum_raw, expected_sums)
                np.testing.assert_allclose(
                    episode.reward_sum, expected_sums / 1200, atol=1e-6
                )
                expected_next = np.concatenate(
                    [
                        episode.value_current.to_numpy()[32:],
                        np.zeros(min(32, len(episode))),
                    ]
                )
                np.testing.assert_allclose(episode.value_next, expected_next, atol=1e-6)
                counts = task_counts.setdefault(
                    entry["task"],
                    {
                        "frames": 0,
                        "positive_labels": 0,
                        "last_chunk_frames": 0,
                        "last_chunk_positive_labels": 0,
                    },
                )
                counts["frames"] += len(episode)
                counts["positive_labels"] += int(episode.advantage.sum())
                counts["last_chunk_frames"] += min(32, len(episode))
                counts["last_chunk_positive_labels"] += int(
                    episode.advantage.iloc[-32:].sum()
                )
            stats["by_task"] = task_counts
            train_labels[split] = advantages
            positives = int(advantages.advantage.sum())
            if split.startswith("sft") and positives != len(advantages):
                raise ValueError(
                    "SFT labels do not follow the official positive-only rule"
                )
            total_positive += positives
            total_negative += len(advantages) - positives
            stats.update(
                positive_labels=positives, negative_labels=len(advantages) - positives
            )
        report["datasets"][split] = stats
        destination = run / "dataset-sidecars" / split
        destination.mkdir(parents=True, exist_ok=True)
        for name in [
            f"returns_{tag}.parquet",
            f"advantages_{tag}.parquet",
            "conversion.json",
            "mixture_config.yaml",
            "info.json",
            "stats.json",
            "tasks.jsonl",
        ]:
            if (meta / name).is_file():
                shutil.copy2(meta / name, destination / name)
    if total_positive == 0 or total_negative == 0:
        raise ValueError("Pilot does not exercise both training label classes")
    threshold = float(
        np.percentile(
            np.concatenate(
                [t.advantage_continuous.to_numpy() for t in train_labels.values()]
            ),
            70,
        )
    )
    np.testing.assert_array_equal(
        train_labels["rollout-train"].advantage,
        train_labels["rollout-train"].advantage_continuous >= threshold,
    )
    report["unified_advantage_threshold"] = threshold
    torch.set_num_threads(2)
    policy_file = Path(state["policy_checkpoint"]) / "model_state_dict/full_weights.pt"
    value_file = Path(state["value_checkpoint"]) / "model_state_dict/full_weights.pt"
    policy = torch.load(policy_file, map_location="cpu", weights_only=True, mmap=True)
    value = torch.load(value_file, map_location="cpu", weights_only=True, mmap=True)
    checked = 0
    for name, tensor in policy.items():
        if name.startswith(
            ("action_", "time_mlp_", "paligemma_with_expert.gemma_expert.")
        ):
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError(f"Nonfinite policy tensor: {name}")
            checked += tensor.numel()
    for name, tensor in value.items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite critic tensor: {name}")
    changes = {}
    with safe_open(
        ROOT / "models/pi0.5_robotwin2/model.safetensors", framework="pt", device="cpu"
    ) as base:
        for name in [
            "action_out_proj.weight",
            "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.weight",
        ]:
            delta = (base.get_tensor(name).float() - policy[name].float()).abs()
            changes[name] = {
                "max_abs": float(delta.max()),
                "mean_abs": float(delta.mean()),
            }
    if changes["action_out_proj.weight"]["max_abs"] <= 0:
        raise ValueError("No observed action-expert parameter update")
    if (
        changes[
            "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.weight"
        ]["max_abs"]
        != 0
    ):
        raise ValueError("Sampled frozen VLM tensor changed")
    evaluation = json.loads((run / "evaluation/results.json").read_text())
    if evaluation["status"] != "completed":
        raise ValueError("Paired development evaluation incomplete")
    branches = {}
    for episode in evaluation["results"]:
        branches.setdefault(episode["branch"], set()).add(
            (episode["task"], episode["seed"])
        )
        if (
            not Path(episode["video"]).is_file()
            or not 0 < episode["steps"] <= episode["episode_limit"]
        ):
            raise ValueError("Missing video or invalid evaluation episode")
    expected = {"base_no_guide", "base_positive", "trained_positive"}
    seed_file = (
        ROOT / "third_party/RLinf/rlinf/envs/sim/robotwin/seeds/train_seeds.json"
    )
    seed_pool = json.loads(seed_file.read_text())
    count = evaluation["episodes_per_task"]
    expected_scenarios = {
        (task, int(seed))
        for task in EXPECTED_ROLLOUT_INSTRUCTIONS
        for seed in seed_pool[task]["success_seeds"][200 : 200 + count]
    }
    if (
        len(evaluation["results"]) != 3 * len(expected_scenarios)
        or branches.get("base_no_guide") != expected_scenarios
        or collection_scenarios & expected_scenarios
    ):
        raise ValueError("Evaluation coverage differs or overlaps the collected pool")
    if set(branches) != expected or any(
        x != branches["base_no_guide"] for x in branches.values()
    ):
        raise ValueError("Evaluation scenarios are not paired across all conditions")
    patch = json.loads((ROOT / "bootstrap/patches/recap-robotwin.json").read_text())
    for item in patch["files"]:
        path = ROOT / "third_party/RLinf" / item["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["patched_sha256"]:
            raise ValueError(f"Unrecorded source change: {path}")
    report.update(
        checks={
            "returns_and_outcomes": "passed",
            "advantage_coverage_formula_boundaries_and_finiteness": "passed",
            "policy_trainable_elements_checked": checked,
            "critic_tensors_finite": "passed",
            "parameter_changes": changes,
            "paired_checkpoint_evaluation": "passed",
            "recorded_source_patch": "passed",
        },
        development_evaluation=evaluation["summary"],
        policy_checkpoint=str(policy_file),
        value_checkpoint=str(value_file),
        checkpoint_sha256={
            name: hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
            for name, path in [("policy", policy_file), ("value", value_file)]
        },
    )
    output = ROOT / ".runtime/validation/recap-readiness.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
