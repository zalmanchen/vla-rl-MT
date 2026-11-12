"""Check calibration accounting and offline split guards without loading a model."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluation = load("multitask_eval", "scripts/evaluate_multitask.py")
audit = load("multitask_audit", "bootstrap/audit_multitask.py")


class MultitaskTests(unittest.TestCase):
    def test_setup_failures_are_separate_from_policy_outcomes(self):
        result = evaluation.summarize([
            {"task": "a", "status": "complete", "success": True},
            {"task": "a", "status": "complete", "success": False},
            {"task": "a", "status": "setup_error", "success": None},
        ])["a"]
        self.assertEqual(result["rate_valid"], 0.5)
        self.assertEqual(result["operational_rate"], 1 / 3)
        self.assertEqual(result["setup_errors"], 1)
        self.assertIsNone(evaluation.wilson(0, 0))

    def test_unknown_scenes_do_not_become_certified_nonoverlap(self):
        rows = [{"task": "a", "split": split, "seed": None} for split in ("train", "val")]
        self.assertEqual(audit.split_collisions(rows, "seed"), [])
        rows.extend({"task": "a", "split": split, "seed": 7} for split in ("train", "val"))
        self.assertEqual(audit.split_collisions(rows, "seed")[0]["id"], 7)

    def test_failure_and_unknown_outcomes_stay_distinct(self):
        rows = [{"task": "a", "split": "train", "source": "rollout", "frames": 2,
                 "success": outcome} for outcome in (True, False, None)]
        item = audit.summarize_episodes(rows)[0]
        self.assertEqual([item[key] for key in ("successes", "failures", "unknown_outcomes")], [1, 1, 1])
        self.assertEqual(item["known_scene_ids"], 0)

    def test_final_test_range_is_rejected_before_evaluation(self):
        protocol = json.loads((ROOT / "configs/multitask/phase0.json").read_text())
        protocol["evaluation"]["seed_indices"] = [500, 520]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, "final-test"):
                evaluation.make_plan(path, Path(directory) / "run")

    def test_same_seed_does_not_hide_different_initial_observations(self):
        row = {"task": "a", "seed": 3, "status": "complete", "success": True,
               "noise_seed": 8, "instruction": "goal", "episode_limit": 10,
               "observed_init_fingerprint": "first"}
        second = {**row, "success": False, "observed_init_fingerprint": "different"}
        result = evaluation.compare_backends([row], [second])
        self.assertEqual(result["paired_cases"], 1)
        self.assertEqual(len(result["observed_context_mismatches"]), 1)
        self.assertEqual(result["by_task"]["a"]["native_only"], 1)


if __name__ == "__main__":
    unittest.main()
