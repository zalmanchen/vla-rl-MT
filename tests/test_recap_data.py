"""Check episode boundaries, outcome labels, and matching ALOHA value transforms."""
# ruff: noqa: E402

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

if not all(importlib.util.find_spec(name) for name in ["torch", "h5py", "lerobot"]):
    raise unittest.SkipTest("RECAP data tests require the prepared main environment")

import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


converter = load_script("recap_converter", ROOT / "bootstrap/prepare_recap_data.py")
returns = load_script(
    "recap_returns",
    ROOT
    / "third_party/RLinf/examples/offline_rl/advantage_labeling/recap/process/compute_returns.py",
)


class RecapDataTests(unittest.TestCase):
    def test_failed_one_element_success_column_receives_failure_penalty(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "episode.parquet"
            pq.write_table(
                pa.table(
                    {
                        "episode_index": [0, 0, 1, 1],
                        "frame_index": [0, 1, 0, 1],
                        "is_success": [[False], [False], [True], [True]],
                        "task_index": [0, 0, 0, 0],
                    }
                ),
                path,
            )
            result = returns._process_single_parquet(
                str(path), "rollout", 1.0, -300, {0: "task"}
            )
            self.assertEqual(result["return"].to_pylist(), [-301, -300, -1, 0])

    def test_real_lerobot_conversion_keeps_actions_and_episode_boundaries(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = []
            for episode in range(2):
                path = root / f"source-{episode}.hdf5"
                joints = np.arange(42, dtype=np.float32).reshape(3, 14) / 100 + episode
                with h5py.File(path, "w") as h:
                    h.create_dataset("state", data=joints)
                    h.create_dataset("action", data=joints + 0.02)
                    for camera in converter.CAMERAS:
                        frames = h.create_dataset(
                            f"images/{camera}",
                            (3,),
                            dtype=h5py.vlen_dtype(np.dtype("uint8")),
                        )
                        for i in range(3):
                            ok, image = cv2.imencode(
                                ".jpg",
                                np.full(
                                    (24, 32, 3), 20 + 80 * episode + i, dtype=np.uint8
                                ),
                            )
                            self.assertTrue(ok)
                            frames[i] = image.reshape(-1)
                    h.attrs.update(
                        instruction="test task",
                        success=bool(episode),
                        complete=True,
                        fps=15,
                    )
                entries.append(converter.episode_info(path, "test", "train", "rollout"))
            destination = root / "converted"
            converter.convert(entries, destination)
            dataset = LeRobotDataset(
                destination.name,
                root=destination,
                delta_timestamps={"action": [i / 15 for i in range(4)]},
            )
            chunk = dataset[2]["action"].numpy()
            expected = np.arange(42, dtype=np.float32).reshape(3, 14)[2] / 100 + 0.02
            np.testing.assert_allclose(chunk, np.repeat(expected[None], 4, axis=0))
            self.assertFalse(bool(dataset[2]["is_success"].item()))
            self.assertTrue(bool(dataset[3]["is_success"].item()))
            record = json.loads((destination / "meta/conversion.json").read_text())
            self.assertEqual(record["episode_count"], 2)
            self.assertEqual(record["frame_count"], 6)

    def test_value_training_and_inference_preserve_the_same_three_cameras(self):
        from openpi.transforms import compose
        from rlinf.data.datasets.recap.value_dataset import ValueDataset
        from rlinf.models.embodiment.value_model.recap.checkpoint_utils import (
            build_input_transforms,
        )

        state = np.zeros(14, dtype=np.float32)
        state[[6, 13]] = 0.5
        sample = {"observation.state": state, "prompt": "press bell"}
        for index, camera in enumerate(converter.CAMERAS):
            sample[f"observation.images.{camera}"] = np.full(
                (24, 32, 3), index * 60, dtype=np.uint8
            )
        train = ValueDataset._build_transform("robotwin", "pi05", 32, None)
        infer = compose(
            build_input_transforms("robotwin", "pi05", 32, None, None, True)
        )
        a, b = train(dict(sample)), infer(dict(sample))
        expected = {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
        self.assertEqual(set(a["image"]), expected)
        for key in expected:
            np.testing.assert_array_equal(a["image"][key], b["image"][key])
            self.assertTrue(a["image_mask"][key])
        np.testing.assert_allclose(a["state"], b["state"])

    def test_guidance_inference_uses_the_training_state_tokenization(self):
        import torch
        from openpi import transforms
        from rlinf.data.datasets.recap.cfg_model import TokenizePromptWithGuidance
        from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
        from rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model import (
            OpenPi0ForCFGActionPrediction,
        )

        config = get_openpi_config(
            "pi05_aloha_robotwin",
            model_path=str(ROOT / "models/pi0.5_robotwin2"),
            data_kwargs={
                "norm_stats_path": str(
                    ROOT
                    / "models/pi0.5_robotwin2/assets/pi0.5_clean_randomize_joint_training/norm_stats.json"
                )
            },
        )
        data = config.data.create(config.assets_dirs, config.model)
        prefix = [
            *data.data_transforms.inputs,
            transforms.Normalize(data.norm_stats, use_quantiles=data.use_quantile_norm),
        ]
        model = OpenPi0ForCFGActionPrediction.__new__(OpenPi0ForCFGActionPrediction)
        torch.nn.Module.__init__(model)
        model.setup_wrappers([*prefix, *data.model_transforms.inputs])
        state = np.zeros(14, dtype=np.float32)
        state[[6, 13]] = 0.5
        sample = {
            "observation/state": state,
            "observation/image": np.zeros((24, 32, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((2, 24, 32, 3), dtype=np.uint8),
            "prompt": "press bell",
        }
        train_transforms = [
            TokenizePromptWithGuidance(t.tokenizer, t.discrete_state_input)
            if isinstance(t, transforms.TokenizePrompt)
            else t
            for t in data.model_transforms.inputs
        ]
        expected = transforms.compose([*prefix, *train_transforms])(dict(sample))
        batch = {k: np.asarray(v)[None] for k, v in sample.items() if k != "prompt"}
        batch.update(
            prompt=["press bell"],
            positive_guidance_prompt=["press bell\nAdvantage: positive"],
            negative_guidance_prompt=["press bell\nAdvantage: negative"],
        )
        actual = model.input_transform(batch, transpose=False)
        for key in [
            "tokenized_prompt",
            "tokenized_positive_guidance_prompt",
            "tokenized_negative_guidance_prompt",
        ]:
            np.testing.assert_array_equal(actual[key][0].numpy(), expected[key])


if __name__ == "__main__":
    unittest.main()
