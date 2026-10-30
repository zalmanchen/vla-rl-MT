#!/usr/bin/env python3
"""Audit raw HDF5, freeze episode splits, and fit shared train-only normalization."""

import argparse
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vla_mtrl.common import ROOT, TASKS, digest, environment, write_json


def build_manifest(raw_root, tasks, seed=0):
    from vla_mtrl.data import inspect_episode

    rows = []
    for task in tasks:
        files = sorted((raw_root / task).rglob("*.hdf5")) + sorted(
            (raw_root / task).rglob("*.h5")
        )
        if len(files) < 2:
            raise ValueError(
                f"Need at least two episodes for {task}; run ./download.sh --component data"
            )
        hashes = [digest(path) for path in files]
        groups = sorted(set(hashes))
        if len(groups) < 2:
            raise ValueError("Need distinct episode groups")
        random.Random(seed).shuffle(groups)
        validation = set(groups[: max(1, len(groups) // 5)])
        for path, sha in zip(files, hashes):
            rows.append(
                {
                    "file": str(path.relative_to(raw_root)),
                    "task": task,
                    "instruction": task.replace("_", " "),
                    "frames": inspect_episode(path),
                    "sha256": sha,
                    "bytes": path.stat().st_size,
                    "split": "validation" if sha in validation else "train",
                }
            )
    return {
        "schema": 1,
        "tasks": tasks,
        "seed": seed,
        "horizon": 50,
        "episodes": rows,
        "action_semantics": "next 50 demonstrated absolute joint positions; arm deltas applied by native OpenPI transforms",
        "split_scope": "episode/content grouped; physical reset and upstream pretraining disjointness not claimed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=ROOT / ".data/robotwin/manifest.json"
    )
    args = parser.parse_args()
    environment()
    import h5py
    import numpy as np
    from openpi.shared.normalize import RunningStats, serialize_json
    from vla_mtrl.data import DELTA_MASK, HORIZON

    raw = ROOT / ".data/robotwin/raw"
    output = args.output.resolve()
    manifest = build_manifest(raw, args.tasks, args.seed)
    manifest["raw_root"] = os.path.relpath(raw, output.parent)
    if output.exists() and json.loads(output.read_text()) != manifest:
        raise ValueError(
            "Frozen manifest differs; choose --output for a new data protocol"
        )
    state_stats, action_stats = RunningStats(), RunningStats()
    for row in manifest["episodes"]:
        if row["split"] != "train":
            continue
        with h5py.File(raw / row["file"], "r") as file:
            qpos = file["joint_action/vector"][:].astype(np.float64)
        states = qpos[:-HORIZON]
        windows = np.lib.stride_tricks.sliding_window_view(
            qpos[1:], HORIZON, axis=0
        ).transpose(0, 2, 1)
        actions = windows - states[:, None, :] * DELTA_MASK
        state_stats.update(states)
        action_stats.update(actions)
    norm = output.with_suffix(".norm_stats.json")
    text = serialize_json(
        {
            "state": state_stats.get_statistics(),
            "actions": action_stats.get_statistics(),
        }
    )
    if norm.exists() and norm.read_text() != text:
        raise ValueError("Frozen normalization differs; choose a new manifest output")
    output.parent.mkdir(parents=True, exist_ok=True)
    norm.write_text(text)
    write_json(output, manifest)
    print(
        json.dumps(
            {
                "manifest": str(output),
                "norm_stats": str(norm),
                "episodes": len(manifest["episodes"]),
                "sha256": digest(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
