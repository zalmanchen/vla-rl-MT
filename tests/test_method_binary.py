"""Check reward derivation, immutable raw scores and episode-level diagnostics."""

import copy
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if any(importlib.util.find_spec(name) is None for name in ("torch", "scipy")):
    raise unittest.SkipTest("Binary value tests require the prepared environment")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


derive = load("binary_derive", "bootstrap/prepare_method_binary.py")
metrics = load("binary_metrics", "scripts/evaluate_method_m1.py")
compare = load("binary_compare", "scripts/compare_method_rewards.py")


class BinaryValueTests(unittest.TestCase):
    def rows(self):
        rows = []
        for episode, success in (("success", True), ("failure", False)):
            for step in (0, 2, 4):
                rows.append({"task": "a", "episode": episode, "split": "val", "success": success,
                             "step": step, "episode_limit": 4, "terminal": step == 4,
                             "use_for_loss": step != 4, "target": -0.5, "chunk_reward": -0.1})
        return rows

    def test_binary_targets_do_not_mutate_parent_and_pay_success_only_once(self):
        parent = self.rows()
        saved = copy.deepcopy(parent)
        derived = derive.binary_rows(parent)
        self.assertEqual(parent, saved)
        self.assertEqual([r["target"] for r in derived], [1, 1, 0, 0, 0, 0])
        self.assertEqual([r["chunk_reward"] for r in derived], [0, 1, 0, 0, 0, 0])
        residual = metrics.decision_advantages(derived, np.array([r["target"] for r in derived]))
        np.testing.assert_array_equal(residual[[0, 1, 3, 4]], 0)
        with self.assertRaisesRegex(ValueError, "Missing"):
            derive.binary_rows(parent[:-1])

    def test_probability_clipping_is_report_only_and_never_changes_td_values(self):
        rows = derive.binary_rows(self.rows())
        selected = [r for r in rows if r["use_for_loss"]]
        raw = np.array([1.5, 1.5, -0.2, -0.2])
        saved = raw.copy()
        result = metrics.probability_metrics(selected, raw)
        self.assertEqual(result["clipped_brier"], 0)
        self.assertEqual(result["raw_out_of_range_mass"], 1)
        np.testing.assert_array_equal(raw, saved)
        self.assertGreater(metrics.episode_balanced_error(selected, raw, 2)["macro"], 0)

    def test_success_views_count_episodes_and_report_missing_outcome_classes(self):
        rows = self.rows()
        rows[4]["step"] = 1  # Failure has no decision in the 50-75% window.
        result = metrics.success_diagnostics(rows, np.array([0.9, 0.8, 99, 0.1, 0.2, 99]))
        self.assertEqual(result["views"]["initial"]["a"]["episodes"], 2)
        self.assertEqual(result["views"]["initial"]["a"]["auc"], 1)
        self.assertEqual(result["views"]["elapsed_50_75"]["a"]["episodes"], 1)
        self.assertIsNone(result["views"]["elapsed_50_75"]["a"]["auc"])

    def test_paired_auc_uses_matched_scene_outcomes_and_ties(self):
        first = [[{"task": "a", "episode": "s", "success": True, "score": 0.9},
                  {"task": "a", "episode": "f", "success": False, "score": 0.1}]]
        second = copy.deepcopy(first)
        second[0][0]["score"] = second[0][1]["score"] = 0.5
        result = compare.paired_auc(first, second, seeds=(0,))
        self.assertEqual(result["a"]["mean"], 0.5)
        second[0][0]["success"] = False
        with self.assertRaisesRegex(ValueError, "labels changed"):
            compare.paired_auc(first, second, seeds=(0,))

    def test_namespace_is_selected_before_validation_initializes_the_cluster(self):
        import rlinf.config
        import rlinf.runners.sft_runner
        import rlinf.scheduler
        import rlinf.utils.placement

        observed = []

        class FakeCluster:
            NAMESPACE = "default"

            def __init__(self, **kwargs):
                observed.append(self.NAMESPACE)
                if self.NAMESPACE != "binary_test":
                    raise AssertionError("Validation created managers in the wrong namespace")

        def validate(cfg):
            FakeCluster(cluster_cfg=cfg.cluster)  # Match the official validator's side effect.
            return cfg

        class Placement:
            def __init__(self, *args):
                pass

            def get_strategy(self, name):
                return None

        class Worker:
            @classmethod
            def create_group(cls, cfg):
                return cls()

            def launch(self, *args, **kwargs):
                return self

        class Runner:
            def __init__(self, **kwargs):
                pass

            def init_workers(self):
                pass

            def run(self):
                observed.append("trained")

        cfg = SimpleNamespace(runner={"ray_namespace": "binary_test"}, cluster={},
                              actor=SimpleNamespace(group_name="head"))
        for name, path in (("binary_training_entry", "scripts/train_method_value.py"),
                           ("joint_training_entry", "scripts/train_method_joint.py")):
            with self.subTest(entry=path):
                entry = load(name, path)
                observed.clear()
                FakeCluster.NAMESPACE = "default"
                with (patch.object(rlinf.config, "validate_cfg", validate),
                      patch.object(rlinf.scheduler, "Cluster", FakeCluster),
                      patch.object(rlinf.utils.placement, "HybridComponentPlacement", Placement),
                      patch.object(rlinf.runners.sft_runner, "SFTRunner", Runner),
                      patch.object(entry, "make_worker_class", lambda: Worker)):
                    entry.main.__wrapped__(cfg)
                self.assertEqual(observed, ["binary_test", "binary_test", "trained"])


if __name__ == "__main__":
    unittest.main()
