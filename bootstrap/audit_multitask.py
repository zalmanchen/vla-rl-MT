"""Read-only audit of the Motus checkpoint and the existing offline pilot pool.

Does not collect trajectories, rewrite datasets, or declare critic readiness from
file existence. Unknown scene/outcome provenance remains unknown.
"""

import argparse
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def summarize_episodes(rows):
    grouped = {}
    for row in rows:
        key = (row["task"], row["split"], row["source"])
        item = grouped.setdefault(key, {"episodes": 0, "frames": 0, "successes": 0,
                                        "failures": 0, "unknown_outcomes": 0,
                                        "known_scene_ids": 0})
        item["episodes"] += 1
        item["frames"] += row["frames"]
        item["successes" if row["success"] is True else
             "failures" if row["success"] is False else "unknown_outcomes"] += 1
        item["known_scene_ids"] += row.get("seed") is not None
    return [{"task": key[0], "split": key[1], "source": key[2], **value}
            for key, value in sorted(grouped.items())]


def split_collisions(rows, field):
    seen = collections.defaultdict(set)
    for row in rows:
        value = row.get(field)
        if value is not None:
            seen[(row["task"], value)].add(row["split"])
    return [{"task": key[0], "id": key[1], "splits": sorted(splits)}
            for key, splits in seen.items() if len(splits) > 1]


def audit(protocol, verify_model=False):
    import h5py

    model = protocol["model"]
    path = ROOT / model["path"]
    metadata = path / ".cache/huggingface/download" / (model["weights"] + ".metadata")
    cached = metadata.read_text().splitlines()
    if cached[:2] != [model["revision"], model["sha256"]]:
        raise ValueError("Model download provenance differs from the protocol")
    if digest(ROOT / model["norm_stats"]) != model["norm_stats_sha256"]:
        raise ValueError("Normalization statistics changed")
    weights_sha = digest(path / model["weights"]) if verify_model else None
    if verify_model and weights_sha != model["sha256"]:
        raise ValueError("Model weights checksum mismatch")
    if digest(ROOT / protocol["critic_contract"]["source_file"]) != protocol["critic_contract"]["source_sha256"]:
        raise ValueError("Critic implementation changed; re-audit its actual input contract")
    patch_path = ROOT / protocol["runtime"]["compatibility_manifest"]
    patch = json.loads(patch_path.read_text())
    patch_files = {}
    for item in patch["files"]:
        actual = digest(ROOT / protocol["runtime"]["source"] / item["path"])
        if actual != item["patched_sha256"]:
            raise ValueError(f"Unknown source patch: {item['path']}")
        patch_files[item["path"]] = actual
    rows = []
    for manifest in sorted((ROOT / "datasets/lerobot/recap-pilot-v2").glob("*/meta/conversion.json")):
        data = json.loads(manifest.read_text())
        for episode, entry in enumerate(data["episodes"]):
            source = Path(entry["path"])
            row = {**entry, "episode_index": episode, "source_sha256": digest(source),
                   "seed": None, "final_observation_saved": None,
                   "full_simulator_state_saved": None, "recovery_annotation": "unknown",
                   "success_label_origin": "conversion assumption for expert demonstrations"}
            with h5py.File(source) as h:
                if entry["source"] == "rollout":
                    row.update(seed=int(h.attrs["seed"]),
                               model_revision=str(h.attrs["model_revision"]),
                               termination=str(h.attrs["termination"]),
                               episode_limit=int(h.attrs["episode_limit"]),
                               success_label_origin="recorded simulator outcome",
                               terminated_truncated_separate=("terminated" in h and "truncated" in h),
                               recorded_chunk_ids="chunk_id" in h)
                    row["final_observation_saved"] = "final_observation" in h
                    row["full_simulator_state_saved"] = "simulator_state" in h
                    row["interior_next_observations_by_shift"] = entry["frames"] - 1
                else:
                    row["termination"] = "unrecorded"
            rows.append(row)
    inventory = {}
    for task in sorted((ROOT / "datasets/RoboTwin2.0-clean").iterdir()):
        if task.is_dir():
            inventory[task.name] = len(list(task.rglob("*.hdf5")))
    cross_source = collections.defaultdict(lambda: collections.Counter())
    for row in rows:
        cross_source[row["instruction"]][row["source"]] += 1
    rows_train = [row for row in rows if row["split"] == "train" and row["source"] == "rollout"]
    train_tasks = sorted({row["task"] for row in rows_train})
    risks = [
        "Only two tasks and eight autonomous training episodes in the converted pilot pool.",
        "No successful autonomous validation episode; source-stratified success/failure diagnostics are insufficient.",
        "Demonstration scene seeds are unrecorded: scene-level split separation is not certified.",
        "Autonomous recordings lack post-action final observations and full restorable simulator state.",
        "Expert demonstration success is assumed by conversion, not independently verified here.",
        "Instruction phrasing differs by source; report source-stratified metrics and audit shortcut learning.",
        "No verified same-state alternative-goal labels: contrastive auxiliary training is ineligible.",
    ]
    return {"scope": "resource/data audit, not evidence of model capability or RL benefit",
            "model": {"repo": model["repo"], "revision": model["revision"],
                      "expected_sha256": model["sha256"], "actual_sha256": weights_sha,
                      "weights_verified": verify_model, "norm_stats_sha256": model["norm_stats_sha256"]},
            "patch_files": patch_files,
            "raw_clean_demonstrations": {"tasks": len(inventory), "episodes_by_task": inventory},
            "converted_pool": summarize_episodes(rows), "episodes": rows,
            "known_seed_split_collisions": split_collisions(rows, "seed"),
            "source_content_split_collisions": split_collisions(rows, "source_sha256"),
            "scene_split_certified": False, "autonomous_train_tasks": train_tasks,
            "instructions_by_source": {text: dict(value) for text, value in cross_source.items()},
            "critic_contract": protocol["critic_contract"],
            "research_training_ready": False, "contrastive_pairs_verified": False,
            "limitations": risks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/multitask/phase0.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-model", action="store_true")
    args = parser.parse_args()
    result = audit(json.loads(args.protocol.read_text()), args.verify_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "weights_verified": result["model"]["weights_verified"],
                      "converted_pool": result["converted_pool"],
                      "research_training_ready": result["research_training_ready"]}, indent=2))


if __name__ == "__main__":
    main()
