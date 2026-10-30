"""Meaningful CPU regressions for raw data, task budgets and download safety."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "RLinf"), str(ROOT / "scripts")]
from vla_mtrl.common import compose, environment, write_json
from vla_mtrl.data import BalancedSampler, RoboTwinWindows, inspect_episode
from download_assets import safe_extract, verify
from prepare_data import build_manifest


def episode(path, frames=55, offset=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.arange(frames * 14, dtype=np.float32).reshape(frames, 14) / 100 + offset
    with h5py.File(path, "w") as file:
        file.create_dataset("joint_action/vector", data=values)
        image = np.zeros((12, 12, 3), dtype=np.uint8)
        image[:, :, 0] = 100
        ok, encoded = cv2.imencode(".jpg", image)
        assert ok
        for camera in ["head_camera", "left_camera", "right_camera"]:
            file.create_dataset(
                f"observation/{camera}/rgb",
                data=[encoded.tobytes()] * frames,
                dtype=f"S{len(encoded.tobytes())}",
            )
    return values


class RawDataTests(unittest.TestCase):
    def test_native_transforms_recover_recorded_future_joint_targets(self):
        environment()
        import openpi.transforms as transforms
        from rlinf.models.embodiment.openpi.transforms.pipeline import (
            build_openpi_transforms,
        )

        class OfflineTokenizer:
            def __init__(self, *args, **kwargs):
                pass

            def tokenize(self, *args, **kwargs):
                return np.zeros(48, dtype=np.int32), np.ones(48, dtype=bool)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode(root / "raw/task/a.hdf5")
            manifest = {
                "raw_root": "raw",
                "episodes": [
                    {
                        "file": "task/a.hdf5",
                        "frames": 55,
                        "task": "task",
                        "split": "train",
                        "instruction": "task",
                    }
                ],
            }
            write_json(root / "manifest.json", manifest)
            write_json(
                root / "norm_stats.json",
                {
                    "norm_stats": {
                        key: {"mean": [0.0] * 14, "std": [1.0] * 14}
                        for key in ["state", "actions"]
                    }
                },
            )
            with patch("openpi.models.tokenizer.PaligemmaTokenizer", OfflineTokenizer):
                inputs, outputs = build_openpi_transforms(
                    str(root),
                    "pi0_aloha_robotwin",
                    {"norm_stats_path": str(root / "norm_stats.json")},
                )
            sample = RoboTwinWindows(root / "manifest.json", ["task"], None)[0]
            expected = sample["actions"].copy()
            encoded = transforms.compose(inputs)(sample)
            self.assertEqual(encoded["actions"].shape, (50, 32))
            decoded = transforms.compose(outputs)(
                {"actions": encoded["actions"], "state": encoded["state"]}
            )
            np.testing.assert_allclose(decoded["actions"], expected, atol=1e-5)

    def test_future_actions_stay_inside_the_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = episode(root / "raw/task/episode.hdf5")
            manifest = {
                "raw_root": "raw",
                "episodes": [
                    {
                        "file": "task/episode.hdf5",
                        "frames": 55,
                        "task": "task",
                        "split": "train",
                        "instruction": "task",
                    }
                ],
            }
            write_json(root / "manifest.json", manifest)
            dataset = RoboTwinWindows(root / "manifest.json", ["task"], None)
            self.assertEqual(len(dataset), 5)
            np.testing.assert_array_equal(dataset[0]["actions"], values[1:51])
            np.testing.assert_array_equal(dataset[4]["actions"], values[5:55])
            self.assertEqual(dataset[0]["images"]["cam_high"].shape, (12, 12, 3))

    def test_invalid_joint_dimension_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.hdf5"
            with h5py.File(path, "w") as file:
                file.create_dataset("joint_action/vector", data=np.zeros((100, 7)))
            with self.assertRaisesRegex(ValueError, "T,14"):
                inspect_episode(path)

    def test_duplicate_content_cannot_cross_split(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            episode(raw / "task/a.hdf5", offset=1)
            (raw / "task/duplicate.hdf5").write_bytes(
                (raw / "task/a.hdf5").read_bytes()
            )
            episode(raw / "task/b.hdf5", offset=2)
            rows = build_manifest(raw, ["task"])["episodes"]
            groups = {}
            for row in rows:
                groups.setdefault(row["sha256"], set()).add(row["split"])
            self.assertTrue(all(len(split) == 1 for split in groups.values()))

    def test_every_microbatch_has_equal_task_exposure_and_resume_is_exact(self):
        sampler = BalancedSampler({"a": [0, 1], "b": [2, 3, 4, 5]}, 2, 0)
        iterator = iter(sampler)
        next(iterator)
        state = sampler.state_dict()
        remaining = list(iterator)
        resumed = BalancedSampler(sampler.by_task, 2, 0)
        resumed.load_state_dict(state)
        self.assertEqual(list(resumed), remaining)
        sampler.set_epoch(0)
        values = list(sampler)
        for start in range(0, len(values), 2):
            self.assertEqual(sum(v < 2 for v in values[start : start + 2]), 1)


class DownloadTests(unittest.TestCase):
    def test_zip_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as file:
                file.writestr("../outside", "bad")
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                safe_extract(archive, root / "target")
            self.assertFalse((root / "outside").exists())

    def test_same_size_corruption_fails_hash_check(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weight"
            path.write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                verify(
                    path,
                    {"rfilename": "weight", "size": 3, "lfs": {"sha256": "0" * 64}},
                )


class PlanTests(unittest.TestCase):
    def test_paired_plan_does_not_require_data_or_gpu(self):
        result = subprocess.run(
            ["bash", str(ROOT / "train.sh"), "--mode", "paired", "--smoke", "--plan"],
            cwd="/tmp",
            capture_output=True,
            text=True,
            check=True,
        )
        plan = json.loads(result.stdout)
        self.assertEqual(len(plan["training_groups"]), 3)
        self.assertEqual(plan["steps"], 20)
        self.assertFalse(plan["gpu_query_or_allocation"])
        self.assertEqual(Path(plan["root"]), ROOT)

    def test_native_config_has_14d_actions_and_matched_exposure(self):
        environment()
        mt = compose(20, ["a", "b"], samples_per_task=8)
        st = compose(20, ["a"], samples_per_task=8)
        self.assertEqual(mt.actor.model.action_dim, 14)
        self.assertEqual(mt.actor.global_batch_size // 2, st.actor.global_batch_size)
        self.assertLess(mt.actor.optim.lr_warmup_steps, mt.runner.max_steps)
        self.assertEqual(mt.actor.model.openpi.config_name, "pi0_aloha_robotwin")


if __name__ == "__main__":
    unittest.main()
