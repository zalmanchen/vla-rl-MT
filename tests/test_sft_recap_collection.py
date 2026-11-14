"""Control-step alignment, true final frames and held-out scene protection."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"bootstrap"))
spec = importlib.util.spec_from_file_location("control_data", ROOT/"bootstrap/prepare_sft_recap_data.py")
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


class FakeTask:
    def __init__(self, limit=5, success_at=5):
        self.step_lim, self.success_at = limit, success_at
        self.take_action_cnt = 0
        self.closed = False

    @property
    def eval_success(self):
        return self.success_at is not None and self.take_action_cnt >= self.success_at

    def set_instruction(self, instruction):
        self.instruction = instruction

    def get_obs(self):
        return {"joint_action": {"vector": np.full(14, self.take_action_cnt)}, "observation": {
            name: {"rgb": np.full((24, 32, 3), self.take_action_cnt, dtype=np.uint8)}
            for name in ("head_camera", "left_camera", "right_camera")}}

    def take_action(self, action, action_type):
        self.take_action_cnt += 1

    def close_env(self):
        self.closed = True


class FakeModel:
    def predict_action_batch(self, obs, mode):
        return torch.ones((1, 32, 14)), {}


class ControlDataTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name)
        self.config = json.loads((ROOT/"configs/multitask/sft_recap_data.json").read_text())
        self.plan = {"config": self.config, "protocol": {"model": {"sha256": "a"*64}}, "sources": {"checker.py": "b"*64}}
        data.save_json(self.output/"plan.json", self.plan)
        self.case = {"task": "click_bell", "seed": 10, "seed_index": 200, "noise_seed": 123, "split": "train"}
        self.task_config = {"instruction_template": "press the bell", "objects": [], "checker": "checker.py"}
        self.path = self.output/"train/click_bell/scene-10.hdf5"

    def test_real_observation_for_each_command_and_true_final_frame(self):
        task = FakeTask()
        result = data.collect_episode(FakeModel(), self.case, self.task_config, self.plan, self.path,
                                      task_factory=lambda *args: task)
        self.assertEqual((result["actions"], result["observations"]), (5, 6))
        self.assertTrue(task.closed)
        with h5py.File(self.path) as episode:
            np.testing.assert_array_equal(episode["state"][:, 0], np.arange(5))
            np.testing.assert_array_equal(episode["final_observation/state"][:], np.full(14, 5))
            final = cv2.imdecode(episode["final_observation/images/cam_high"][:], cv2.IMREAD_COLOR)
            self.assertTrue((final == 5).all())
            for i in range(5):
                frame = cv2.imdecode(episode["images/cam_high"][i], cv2.IMREAD_COLOR)
                self.assertTrue((frame == i).all())
            self.assertEqual(int(episode["executed_mask"][:].sum()), 5)
            np.testing.assert_array_equal(episode["binary_rewards"][:], [0, 0, 0, 0, 1])
        with h5py.File(self.path, "r+") as episode:
            episode["executed_mask"][0, 31] = True
        with self.assertRaisesRegex(ValueError, "Chunk mask"):
            data.inspect_episode(self.path, self.case, self.plan)

    def test_benchmark_timeout_is_recorded_as_truncation(self):
        task = FakeTask(limit=3, success_at=None)
        result = data.collect_episode(FakeModel(), self.case, self.task_config, self.plan, self.path,
                                      task_factory=lambda *args: task)
        self.assertFalse(result["success"])
        with h5py.File(self.path) as episode:
            self.assertTrue(episode.attrs["truncated"])
            self.assertFalse(episode.attrs["terminated"])
            self.assertEqual(episode.attrs["termination_reason"], "benchmark_limit")
            self.assertEqual(episode["binary_rewards"][:].sum(), 0)

    def test_final_scenes_cannot_become_training_data(self):
        self.config["splits"]["train"] = [500, 501]
        path = self.output/"bad.json"
        data.save_json(path, self.config)
        with self.assertRaisesRegex(ValueError, "final-test"):
            data.make_plan(path, self.output/"must-not-exist")
        self.assertFalse((self.output/"must-not-exist").exists())

    def test_success_quota_does_not_trigger_extra_collection(self):
        rows = [{"task": "a", "split": "train", "success": True, "seed_index": 4, "seed": 5}]
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            data.select_successes(rows, ["a"], 2)


if __name__ == "__main__":
    unittest.main()
