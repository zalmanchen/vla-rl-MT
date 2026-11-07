"""Audit train/eval separation and freeze the complete first-stage experience pool."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".runtime/experiments/e4-offline-rl"


def main():
    plan = json.loads((RUN / "plan.json").read_text())
    tests = json.loads((ROOT / "third_party/RLinf/rlinf/envs/sim/robotwin/seeds/eval_seeds.json").read_text())
    paths = sorted((RUN / "demos").glob("*/*.hdf5")) + sorted((RUN / "pool/clean").glob("*/trajectory_*.hdf5"))
    if len(paths) != 450:
        raise RuntimeError(f"Expected 450 complete episodes, got {len(paths)}")
    rows = []
    seen = set()
    for path in paths:
        with h5py.File(path) as h:
            task = str(h.attrs["task"])
            n = len(h["state"])
            if h["state"].shape != (n, 14) or h["action"].shape != (n, 14):
                raise ValueError("Invalid state/action shape")
            if not np.isfinite(h["state"][:]).all() or not np.isfinite(h["action"][:]).all():
                raise ValueError("Non-finite states/actions")
            for camera in ["head", "left_wrist", "right_wrist"]:
                if len(h["images/" + camera]) != n:
                    raise ValueError("Camera/frame mismatch")
            if h["terminated"][:].sum() != 1 or not h["terminated"][-1]:
                raise ValueError("Invalid terminal mask")
            seed = int(h.attrs["seed"]) if "seed" in h.attrs else None
            if seed is not None:
                if seed in tests[task]["success_seeds"]:
                    raise ValueError("Training/test seed overlap")
                if seed not in plan["rollout_seeds"][task]["train"] + plan["rollout_seeds"][task]["value_validation"]:
                    raise ValueError("Unknown collection seed")
                if (task, seed) in seen:
                    raise ValueError("Duplicate trajectory seed")
                seen.add((task, seed))
            row = {"path": str(path.relative_to(ROOT)), "task": task, "frames": n,
                   "split": str(h.attrs["split"]), "success": bool(h.attrs["success"]), "seed": seed}
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        row["sha256"] = digest.hexdigest()
        rows.append(row)
    manifest = {"frozen": True, "episodes": len(rows), "frames": sum(r["frames"] for r in rows),
                "expert_episodes": 250, "rollout_episodes": 200, "test_seed_overlap": False,
                "rows": rows}
    (RUN / "pool-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("POOL_FROZEN", manifest["episodes"], manifest["frames"], flush=True)


if __name__ == "__main__":
    main()
