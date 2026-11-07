"""Convert the existing five-task demonstrations to the fixed-pool episode schema."""

import concurrent.futures
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".runtime/experiments/e4-offline-rl"


def convert_task(task):
    sources = sorted((ROOT / "datasets/RoboTwin2.0-clean" / task).glob("**/episode_*.hdf5"))
    if len(sources) != 50:
        raise ValueError(f"Expected 50 demonstrations: {task}: {len(sources)}")
    dest = RUN / "demos" / task
    dest.mkdir(parents=True, exist_ok=True)
    for i, source in enumerate(sources):
        output = dest / source.name
        if output.exists():
            continue
        tmp = output.with_suffix(".tmp")
        with h5py.File(source) as src, h5py.File(tmp, "w") as dst:
            instructions = json.loads(src["instructions"][()].decode())
            dst.attrs.update(task=task, success=True, instruction=instructions[0],
                             split="train" if i < 45 else "value_validation",
                             source="expert", source_path=str(source), source_frequency=int(src["additional_info/frequency"][()]))
            for key in ["state", "action"]:
                arrays = [src[f"{key}/{part}"][:] for part in ["left_arm_joint_states", "left_ee_joint_states", "right_arm_joint_states", "right_ee_joint_states"]]
                dst.create_dataset(key, data=np.concatenate(arrays, axis=-1), compression="gzip")
            n = len(dst["state"])
            reward = np.full(n, -1, dtype=np.float32)
            reward[-1] = 0
            dst.create_dataset("reward", data=reward)
            dst.create_dataset("terminated", data=np.arange(n) == n - 1)
            # Expert files have no post-terminal observation. Never bootstrap their terminal value.
            dst.attrs["terminal_observation_available"] = False
            for original, camera in [("cam_head", "head"), ("cam_left_wrist", "left_wrist"), ("cam_right_wrist", "right_wrist")]:
                images = src[f"vision/{original}/colors"]
                target = dst.create_dataset("images/" + camera, (n,), dtype=h5py.vlen_dtype(np.dtype("uint8")))
                for t in range(n):
                    target[t] = np.frombuffer(bytes(images[t]), dtype=np.uint8)
        tmp.replace(output)
    print("demo conversion complete", task, len(sources), flush=True)


def main():
    plan = json.loads((RUN / "plan.json").read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        for _ in pool.map(convert_task, plan["anchors"]):
            pass
    (RUN / "DEMOS_READY").write_text("250 episodes; 45 train and 5 value-validation per task\n")


if __name__ == "__main__":
    main()
