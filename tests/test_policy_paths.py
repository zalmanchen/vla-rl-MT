"""The numerical audit must expose both physical and padded-output errors."""

import importlib.util
import unittest
from pathlib import Path

import torch


class PolicyPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("check_policy_paths", Path(__file__).resolve().parents[1]/"scripts/check_policy_paths.py")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_masked_error_still_reports_padding_difference(self):
        reference = torch.ones(1, 2, 3)
        actual = reference.clone()
        actual[0, 1, 2] += 7
        mask = torch.ones_like(reference, dtype=torch.bool)
        mask[0, 1, 2] = False
        error = self.module.velocity_error(reference, actual, mask)
        self.assertEqual(error["max_abs"], 0)
        self.assertEqual(error["all_model_max_abs"], 7)
        self.assertEqual(error["relative_rmse"], 0)

    def test_empty_and_wrong_shape_masks_fail(self):
        value = torch.zeros(1, 2, 3)
        for mask in [torch.zeros_like(value, dtype=torch.bool), torch.ones(1, 1, 3, dtype=torch.bool)]:
            with self.assertRaises(ValueError):
                self.module.velocity_error(value, value, mask)


if __name__ == "__main__":
    unittest.main()
