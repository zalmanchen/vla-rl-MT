"""Convert RoboTwin demonstrations and recorded policy episodes to LeRobot v2.1."""

import argparse
import fcntl
import hashlib
import json
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = {
    "cam_high": "cam_head",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}
JOINT_KEYS = [
    "left_arm_joint_states",
    "left_ee_joint_states",
    "right_arm_joint_states",
    "right_ee_joint_states",
]


def decode_jpeg(value):
    encoded = np.frombuffer(bytes(value), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode RoboTwin camera JPEG")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def episode_info(path, task, split, source):
    with h5py.File(path) as h:
        if source == "sft":
            instructions = json.loads(h["instructions"][()].decode())
            instruction = instructions[0]
            n = len(h["state/left_arm_joint_states"])
            fps = int(h["additional_info/frequency"][()])
            success = True
        else:
            instruction = str(h.attrs["instruction"])
            n = len(h["state"])
            fps = int(h.attrs["fps"])
            success = bool(h.attrs["success"])
            if not h.attrs.get("complete", False):
                raise ValueError(f"Incomplete rollout: {path}")
    if n < 2 or not instruction:
        raise ValueError(f"Invalid episode: {path}")
    return {
        "path": str(path.resolve()),
        "task": task,
        "split": split,
        "source": source,
        "frames": n,
        "fps": fps,
        "success": success,
        "instruction": instruction,
        "source_bytes": path.stat().st_size,
    }


def read_frames(entry):
    with h5py.File(entry["path"]) as h:
        if entry["source"] == "sft":
            states = np.concatenate([h[f"state/{k}"][:] for k in JOINT_KEYS], axis=-1)
            actions = np.concatenate([h[f"action/{k}"][:] for k in JOINT_KEYS], axis=-1)
            camera_paths = {k: f"vision/{v}/colors" for k, v in CAMERAS.items()}
        else:
            states, actions = h["state"][:], h["action"][:]
            camera_paths = {k: f"images/{k}" for k in CAMERAS}
        if states.shape != actions.shape or states.shape != (entry["frames"], 14):
            raise ValueError(f"Invalid state/action dimensions: {entry['path']}")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError(f"Nonfinite joints: {entry['path']}")
        for i in range(entry["frames"]):
            yield {
                "observation.state": states[i].astype(np.float32),
                "action": actions[i].astype(np.float32),
                "is_success": np.array([entry["success"]], dtype=bool),
                **{
                    f"observation.images.{k}": decode_jpeg(h[p][i])
                    for k, p in camera_paths.items()
                },
            }


def convert(entries, destination):
    if not entries:
        return
    signature = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    manifest = destination / "meta/conversion.json"
    if destination.exists():
        if (
            manifest.exists()
            and json.loads(manifest.read_text())["signature"] == signature
        ):
            print("ALREADY_CONVERTED", destination, flush=True)
            return
        raise FileExistsError(
            f"Dataset exists with a different or incomplete manifest: {destination}"
        )
    staging = destination.with_name(destination.name + ".building")
    if staging.exists():
        quarantine = staging.with_name(
            staging.name + f".interrupted-{int(time.time())}"
        )
        staging.rename(quarantine)
    fps_values = {e["fps"] for e in entries}
    if len(fps_values) != 1:
        raise ValueError(f"Mixed frame rates: {fps_values}")
    example = next(read_frames(entries[0]))
    features = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": ["joints"]},
        "action": {"dtype": "float32", "shape": (14,), "names": ["joints"]},
        "is_success": {"dtype": "bool", "shape": (1,), "names": ["episode_success"]},
    }
    for camera in CAMERAS:
        key = f"observation.images.{camera}"
        features[key] = {
            "dtype": "image",
            "shape": example[key].shape,
            "names": ["height", "width", "channels"],
        }
    dataset = LeRobotDataset.create(
        repo_id=destination.name,
        root=staging,
        fps=fps_values.pop(),
        features=features,
        robot_type="aloha_agilex",
        use_videos=False,
        image_writer_threads=4,
    )
    for episode, entry in enumerate(entries):
        for frame in read_frames(entry):
            dataset.add_frame(frame, task=entry["instruction"])
        dataset.save_episode()
        print(
            "CONVERTED",
            destination.name,
            episode,
            entry["task"],
            entry["frames"],
            entry["success"],
            flush=True,
        )
    dataset.stop_image_writer()
    record = {
        "signature": signature,
        "episodes": entries,
        "episode_count": len(entries),
        "frame_count": sum(e["frames"] for e in entries),
        "source_format": "RoboTwin native HDF5 / recorded RLinf policy rollout",
        "action_semantics": "absolute 14D joint targets; no resampling",
        "split_unit": "complete episode; disjoint source episodes and simulator seeds",
        "dataset_revision": "3dc3b798668feb99ac61cc9086d84cbcc3d79186",
    }
    (staging / "meta/conversion.json").write_text(json.dumps(record, indent=2) + "\n")
    staging.rename(destination)
    print("DATASET_READY", destination, record["frame_count"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", nargs="+", default=["click_bell", "stack_blocks_two"]
    )
    parser.add_argument("--train-episodes", type=int, default=8)
    parser.add_argument("--val-episodes", type=int, default=2)
    parser.add_argument("--source", choices=["sft", "rollout", "all"], default="all")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "datasets/lerobot/recap-pilot-v2"
    )
    parser.add_argument(
        "--rollouts", type=Path, default=ROOT / ".runtime/recap-rollouts-v2"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".conversion.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for source in ["sft", "rollout"] if args.source == "all" else [args.source]:
            splits = {"train": [], "val": []}
            for task in args.tasks:
                if source == "sft":
                    paths = sorted(
                        (ROOT / "datasets/RoboTwin2.0-clean" / task).glob(
                            "**/data/episode_*.hdf5"
                        )
                    )
                    required = args.train_episodes + args.val_episodes
                    if len(paths) < required:
                        raise ValueError(
                            f"Need {required} demonstrations for {task}, found {len(paths)}"
                        )
                    for i, path in enumerate(paths[:required]):
                        split = "train" if i < args.train_episodes else "val"
                        splits[split].append(episode_info(path, task, split, source))
                else:
                    for split in splits:
                        paths = sorted(
                            (args.rollouts / split / task).glob("episode_*.hdf5")
                        )
                        splits[split].extend(
                            episode_info(p, task, split, source) for p in paths
                        )
            for split, entries in splits.items():
                convert(entries, args.output / f"{source}-{split}")


if __name__ == "__main__":
    main()
