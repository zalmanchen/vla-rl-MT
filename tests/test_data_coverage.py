"""Protect the nested success-data comparison against selection and split errors."""

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"bootstrap"))
spec = importlib.util.spec_from_file_location("coverage_preparation", ROOT/"bootstrap/prepare_data_coverage.py")
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


class CoverageSelectionTests(unittest.TestCase):
    def test_nested_selection_uses_declared_order_and_excludes_failures_and_validation(self):
        rows = [{"task": "a", "seed": i, "seed_index": i, "split": "train", "success": i != 1}
                for i in reversed(range(6))]
        rows.append({"task": "a", "seed": 99, "seed_index": -1, "split": "val", "success": True})
        small = preparation.select_successes(rows, ["a"], 2)
        large = preparation.select_successes(rows, ["a"], 4)
        self.assertEqual([r["seed"] for r in small], [0, 2])
        self.assertEqual(small, large[:2])

    def test_insufficient_successes_cannot_silently_change_the_budget(self):
        rows = [{"task": "a", "seed": 0, "seed_index": 0, "split": "train", "success": True}]
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            preparation.select_successes(rows, ["a"], 2)

    def test_duplicate_scenes_cannot_count_as_extra_coverage(self):
        row = {"task": "a", "seed": 0, "seed_index": 0, "split": "train", "success": True}
        with self.assertRaisesRegex(ValueError, "duplicated"):
            preparation.select_successes([row, dict(row)], ["a"], 2)


if __name__ == "__main__":
    unittest.main()
