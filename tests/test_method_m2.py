"""Check real-chunk masking, reproducible sampling and the M2 intervention."""

import importlib.util
import json
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("M2 tests require torch")

import numpy as np
import torch
from torch import nn

from vla_rl_mt.joint_value import (
    JointPairs,
    JointPrefixModel,
    executed_chunk,
    make_worker_class,
)
from vla_rl_mt.shared_value import masked_flow_mse

ROOT = Path(__file__).resolve().parents[1]


class MethodJointTests(unittest.TestCase):
    def test_eval_records_the_duration_consumed_by_the_official_runner(self):
        events = []

        @contextmanager
        def timer():
            events.append("start")
            yield
            events.append("recorded")

        worker = SimpleNamespace(worker_timer=timer, _evaluate_values=lambda: {"mse": 0.2})
        self.assertEqual(make_worker_class().run_eval(worker), {"mse": 0.2})
        self.assertEqual(events, ["start", "recorded"])

    def test_joint_config_preserves_the_frozen_m1_pool(self):
        from omegaconf import OmegaConf

        joint = OmegaConf.load(ROOT/"configs/multitask/joint_value.yaml")
        reference = json.loads((ROOT/"configs/multitask/method_m1_binary.json").read_text())
        self.assertEqual(joint.data.pool_manifest_sha256, reference["pool_manifest_sha256"])

    def test_temporal_and_dimension_padding_have_no_direct_loss_or_gradient(self):
        real = np.arange(42, dtype=np.float32).reshape(3, 14)
        padded, mask = executed_chunk(real)
        np.testing.assert_array_equal(padded[:3], real)
        np.testing.assert_array_equal(padded[3:], np.repeat(real[-1:], 29, axis=0))
        self.assertEqual(int(mask.sum()), 3*14)
        loss = torch.ones((1, 32, 32), requires_grad=True)
        valid = torch.from_numpy(mask)[None]
        with torch.no_grad():
            loss[~valid] = 1e20
        result = masked_flow_mse(loss, valid)
        self.assertEqual(float(result), 1.0)
        result.backward()
        self.assertEqual(float(loss.grad[~valid].abs().sum()), 0.0)
        self.assertAlmostEqual(float(loss.grad[valid].sum()), 1.0, places=6)
        with self.assertRaises(ValueError):
            masked_flow_mse(loss, torch.zeros_like(valid))

    def test_each_decision_has_equal_weight_despite_chunk_length(self):
        loss = torch.tensor([[[2.0], [99.0]], [[4.0], [4.0]]])
        mask = torch.tensor([[[True], [False]], [[True], [True]]])
        self.assertEqual(float(masked_flow_mse(loss, mask)), 3.0)

    def test_sft_never_samples_failure_or_validation_and_rng_is_indexed(self):
        rows = [{"task": task, "episode": f"{task}-{split}-{success}", "split": split,
                 "success": success, "target": float(success)}
                for task in ("a", "b") for split in ("train", "val") for success in (True, False)]
        pool = SimpleNamespace(rows=rows, sample=lambda i: ({"state": torch.tensor([i])},
                                                          torch.zeros(32, 32), torch.ones(32, 32, dtype=torch.bool)))
        first, second = JointPairs(pool, 10, 2, 0), JointPairs(pool, 10, 2, 0)
        self.assertTrue(torch.equal(first.sft, second.sft))
        self.assertTrue(torch.equal(first.value, second.value))
        self.assertTrue(all(rows[int(i)]["split"] == "train" and rows[int(i)]["success"] for i in first.sft))
        self.assertTrue(all(rows[int(i)]["split"] == "train" for i in first.value))
        self.assertTrue(any(not rows[int(i)]["success"] for i in first.value))
        torch.randn(100)
        self.assertTrue(torch.equal(first[5]["observation"]["noise"], second[5]["observation"]["noise"]))
        first.offset = 5
        self.assertTrue(torch.equal(first[0]["observation"]["noise"], second[5]["observation"]["noise"]))

    def test_frozen_policy_is_excluded_from_addon_state_but_follows_train_flags(self):
        policy = nn.Module()
        policy.llm, policy.img = nn.Linear(4, 4), nn.Identity()
        model = JointPrefixModel(policy, mode="w2", value_weight=0.1, width=4, rank=2,
                                 readout="vision_prompt_concat", head_width=3)
        self.addCleanup(model.close)
        self.assertFalse(any(name.startswith("policy") for name in model.state_dict()))
        self.assertFalse(any(parameter.requires_grad for parameter in policy.parameters()))
        self.assertEqual(model.value_head[0].normalized_shape, (8,))
        model.train()
        self.assertTrue(policy.llm.training)
        self.assertFalse(policy.img.training)
        model.eval()
        self.assertFalse(policy.llm.training)


if __name__ == "__main__":
    unittest.main()
