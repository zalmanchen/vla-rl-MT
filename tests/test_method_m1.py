"""Check scene/return boundaries and unbiased M1 diagnostics, without simulation."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

if any(importlib.util.find_spec(name) is None for name in ("torch", "h5py", "scipy")):
    raise unittest.SkipTest("M1 tests require the prepared environment")

import h5py
import numpy as np
import torch

from vla_rl_mt.shared_value import pool_prefix
from vla_rl_mt.value_head import FeatureDataset, PrefixValueHead, task_episode_weights

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


metrics = load("m1_metrics", "scripts/evaluate_method_m1.py")
cache_tool = load("m1_cache", "bootstrap/cache_method_m1.py")


class MethodValueTests(unittest.TestCase):
    def test_grouped_readout_respects_padding_and_preserves_both_groups(self):
        hidden = torch.tensor([[[1., 2.], [3., 4.], [999., 999.],
                                [10., 20.], [30., 40.], [999., 999.]]], requires_grad=True)
        mask = torch.tensor([[True, True, False, True, True, False]])
        prompt_mask = torch.tensor([[True, True, False]])
        grouped = pool_prefix(hidden, mask, prompt_mask, readout="vision_prompt_concat")
        torch.testing.assert_close(grouped, torch.tensor([[2., 3., 20., 30.]]))
        global_mean = pool_prefix(hidden, mask)
        torch.testing.assert_close(global_mean, torch.tensor([[11., 16.5]]))
        duplicate = pool_prefix(hidden, mask, readout="global_duplicate")
        torch.testing.assert_close(duplicate, torch.cat([global_mean, global_mean], dim=-1))
        grouped.sum().backward()
        self.assertEqual(float(hidden.grad[:, [2, 5]].abs().sum()), 0)
        self.assertGreater(float(hidden.grad[:, [0, 1, 3, 4]].abs().sum()), 0)

    def test_grouped_readout_rejects_missing_or_misaligned_prompt(self):
        hidden = torch.zeros(1, 4, 2)
        mask = torch.tensor([[True, True, True, False]])
        with self.assertRaisesRegex(ValueError, "requires the actual"):
            pool_prefix(hidden, mask, readout="vision_prompt_concat")
        with self.assertRaisesRegex(ValueError, "verified suffix"):
            pool_prefix(hidden, mask, torch.tensor([[True, True]]), readout="vision_prompt_concat")
        with self.assertRaisesRegex(ValueError, "both visual and prompt"):
            pool_prefix(hidden, torch.tensor([[True, True, False, False]]),
                        torch.tensor([[False, False]]), readout="vision_prompt_concat")

    def test_readout_control_heads_have_identical_capacity_and_initialization(self):
        torch.manual_seed(7)
        control = PrefixValueHead(width=4096, hidden=63)
        torch.manual_seed(7)
        grouped = PrefixValueHead(width=4096, hidden=63)
        self.assertEqual(sum(p.numel() for p in grouped.parameters()), 266367)
        self.assertLess(abs(266367/266497-1), 0.001)
        for left, right in zip(control.parameters(), grouped.parameters(), strict=True):
            self.assertTrue(torch.equal(left, right))

    def test_sampling_balances_tasks_and_episodes_despite_different_lengths(self):
        rows = [{"task": "a", "episode": "a1"}]
        rows += [{"task": "a", "episode": "a2"}]*3
        rows += [{"task": "b", "episode": "b1"}]*2
        weights = task_episode_weights(rows).numpy()
        self.assertAlmostEqual(weights[:4].sum(), 0.5)
        self.assertAlmostEqual(weights[4:].sum(), 0.5)
        self.assertAlmostEqual(weights[0], 0.25)
        self.assertAlmostEqual(weights[1:4].sum(), 0.25)

    def test_feature_dataset_excludes_terminal_and_validation_rows(self):
        rows = [{"split": "train", "use_for_loss": True, "target": -0.4},
                {"split": "train", "use_for_loss": False, "target": 0},
                {"split": "val", "use_for_loss": True, "target": -0.2}]
        dataset = FeatureDataset({"rows": rows, "features": torch.arange(6).reshape(3, 2)}, "train")
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]["observation"].tolist(), [0, 1])
        self.assertAlmostEqual(float(dataset[0]["target_values"]), -0.4)

    def test_terminal_value_never_bootstraps_and_mc_residual_is_not_action_truth(self):
        rows = [{"episode": "a", "terminal": False, "chunk_reward": -0.2},
                {"episode": "a", "terminal": False, "chunk_reward": -0.1},
                {"episode": "a", "terminal": True, "chunk_reward": 0}]
        advantage = metrics.decision_advantages(rows, np.array([-0.3, -0.1, 99]))
        np.testing.assert_allclose(advantage[:2], 0, atol=1e-7)
        self.assertTrue(np.isnan(advantage[-1]))
        with self.assertRaises(ValueError):
            metrics.decision_advantages(rows[:1], np.array([-0.3]))

    def test_episode_mae_does_not_overweight_long_failures(self):
        rows = [{"task": "a", "episode": "a1", "target": 0}]
        rows += [{"task": "a", "episode": "a2", "target": 0}]*3
        rows += [{"task": "b", "episode": "b1", "target": 0}]*2
        self.assertEqual(metrics.episode_balanced_mae(rows, [10, 0, 0, 0, 0, 0])["macro"], 2.5)
        self.assertIsNone(metrics.safe_spearman([1, 1], [0, 1]))
        self.assertEqual(metrics.outcome_auc([0, 0], [True, False]), 0.5)
        self.assertIsNone(metrics.outcome_auc([0, 1], [True, True]))

    def test_return_audit_checks_values_even_if_file_hash_is_updated(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = Path(directory)
            path = pool/"episode.hdf5"
            with h5py.File(path, "w") as episode:
                episode.create_dataset("observation_steps", data=[0, 2, 4])
                episode.create_dataset("state", data=np.zeros((3, 14)))
                episode.create_dataset("actions", data=np.zeros((4, 14)))
                episode.create_dataset("action_chunk_id", data=[0, 0, 1, 1])
                episode.create_dataset("rewards", data=[-1, -1, -1, -300])
                episode.create_dataset("returns", data=[-303, -302, -301, -300])
                episode.create_dataset("decision_returns", data=[-303, -301])
                episode.create_dataset("chunk_rewards", data=[-2, -301])
                episode.create_dataset("images/cam_high", data=[0, 0, 0])
                episode.attrs.update(task="a", split="train", seed=1, success=False,
                                     model_sha256="model", final_observation_saved=True, complete=True,
                                     terminated=False, truncated=True, episode_limit=4)
            case = {"task": "a", "split": "train", "seed": 1}
            (pool/"plan.json").write_text(json.dumps({"cases": [case]}))
            row = {**case, "file": path.name, "sha256": cache_tool.digest(path), "success": False}
            manifest = {"plan_sha256": cache_tool.digest(pool/"plan.json"), "episodes": [row],
                        "config": {"gamma": 1.0, "failure_reward": -300},
                        "protocol": {"model": {"sha256": "model"}}}
            self.assertTrue(cache_tool.validate_pool(pool, manifest)["chunk_returns_verified"])
            with h5py.File(path, "r+") as episode:
                episode["decision_returns"][0] = -302
            row["sha256"] = cache_tool.digest(path)
            with self.assertRaisesRegex(ValueError, "Reward/return"):
                cache_tool.validate_pool(pool, manifest)

    def test_linear_probe_does_not_use_validation_targets_for_fit_or_scaling(self):
        rows = [{"task": "a", "episode": "a1", "split": "train", "use_for_loss": True, "target": -1},
                {"task": "a", "episode": "a2", "split": "train", "use_for_loss": True, "target": 0},
                {"task": "a", "episode": "a3", "split": "val", "use_for_loss": True, "target": 100}]
        features = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=float)
        first = metrics.fixed_ridge_predictions(rows, features)
        rows[-1]["target"] = -100
        second = metrics.fixed_ridge_predictions(rows, features)
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
