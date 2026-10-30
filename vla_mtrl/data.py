"""Read RoboTwin's recorded joint trajectories and JPEG camera observations."""

import json
import math
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import Dataset, Sampler

CAMERAS = {
    "cam_high": "head_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}
HORIZON = 50
DELTA_MASK = np.array([True] * 6 + [False] + [True] * 6 + [False])


def decode_image(value):
    import cv2

    array = np.asarray(value)
    if array.ndim == 3 and array.shape[-1] == 3:
        return array.astype(np.uint8)
    encoded = (
        np.frombuffer(value, dtype=np.uint8)
        if isinstance(value, (bytes, np.bytes_))
        else array.astype(np.uint8).ravel()
    )
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Invalid RoboTwin JPEG observation")
    # imencode/imdecode preserves the upstream camera array's channel order.
    return decoded


def inspect_episode(path, horizon=HORIZON):
    with h5py.File(path, "r") as file:
        qpos = file["joint_action/vector"][:]
        if qpos.ndim != 2 or qpos.shape[1] != 14 or not np.isfinite(qpos).all():
            raise ValueError(f"Expected finite [T,14] ALOHA qpos: {path}")
        if len(qpos) <= horizon:
            raise ValueError(f"Episode shorter than future-action horizon: {path}")
        for camera in CAMERAS.values():
            images = file[f"observation/{camera}/rgb"]
            if len(images) != len(qpos):
                raise ValueError(f"Camera/state length mismatch: {path}")
            decode_image(images[0])
    return len(qpos)


class RoboTwinWindows(Dataset):
    def __init__(self, manifest, tasks, transform, horizon=HORIZON):
        self.manifest_path = Path(manifest)
        info = json.loads(self.manifest_path.read_text())
        self.raw_root = self.manifest_path.parent / info["raw_root"]
        self.rows = [
            r for r in info["episodes"] if r["split"] == "train" and r["task"] in tasks
        ]
        self.transform, self.horizon = transform, horizon
        self.windows = [
            (i, t)
            for i, row in enumerate(self.rows)
            for t in range(row["frames"] - horizon)
        ]
        self.by_task = {task: [] for task in tasks}
        for index, (episode, _) in enumerate(self.windows):
            self.by_task[self.rows[episode]["task"]].append(index)
        if any(not indices for indices in self.by_task.values()):
            raise ValueError("Every requested task needs usable training windows")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        episode, frame = self.windows[index]
        row = self.rows[episode]
        with h5py.File(self.raw_root / row["file"], "r") as file:
            qpos = file["joint_action/vector"]
            sample = {
                "state": qpos[frame].astype(np.float32),
                "actions": qpos[frame + 1 : frame + self.horizon + 1].astype(
                    np.float32
                ),
                "images": {
                    name: decode_image(file[f"observation/{camera}/rgb"][frame])
                    for name, camera in CAMERAS.items()
                },
                "prompt": row["instruction"],
            }
        if self.transform is None:
            return sample
        sample = self.transform(sample)
        return {
            "observation": {
                key: sample[key]
                for key in [
                    "image",
                    "image_mask",
                    "state",
                    "tokenized_prompt",
                    "tokenized_prompt_mask",
                ]
            },
            "actions": sample["actions"],
            "task_label": row["task"],
        }


class BalancedSampler(Sampler):
    def __init__(self, by_task, batch_size, seed):
        self.by_task = by_task
        self.batch_size, self.seed = batch_size, seed
        if batch_size % len(by_task):
            raise ValueError("Microbatch must be divisible by the number of tasks")
        self.length = (
            math.ceil(sum(map(len, by_task.values())) / batch_size) * batch_size
        )
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.epoch, self.position = epoch, 0
        rng = np.random.default_rng(self.seed + epoch)
        indices = []
        for _ in range(self.length // self.batch_size):
            batch = np.concatenate(
                [
                    rng.choice(
                        values, self.batch_size // len(self.by_task), replace=True
                    )
                    for values in self.by_task.values()
                ]
            )
            rng.shuffle(batch)
            indices.extend(batch.tolist())
        self.indices = indices

    def __len__(self):
        return self.length

    def __iter__(self):
        while self.position < self.length:
            value = self.indices[self.position]
            self.position += 1
            yield value

    def state_dict(self):
        return {"epoch": self.epoch, "position": self.position}

    def load_state_dict(self, state):
        self.set_epoch(state["epoch"])
        self.position = state["position"]
