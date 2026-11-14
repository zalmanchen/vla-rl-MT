"""Gradient geometry and selected-data controls must be exact and finite."""

import unittest
from collections import Counter
from types import SimpleNamespace

import torch

from vla_rl_mt.policy_gradients import SelectedPairs, coefficients


class PolicyGradientTests(unittest.TestCase):
    def combine(self, a, b, mode):
        ca, cb, stats = coefficients(float(a@a), float(b@b), float(a@b), float((a+b).square().sum()), mode)
        return ca*a+cb*b, stats

    def test_pcgrad_nonconflicting_unchanged_and_conflict_projection(self):
        a, b = torch.tensor([1., 0.]), torch.tensor([1., 1.])
        result, stats = self.combine(a, b, "pcgrad")
        self.assertTrue(torch.equal(result, a+b)); self.assertFalse(stats["conflict"])
        b = torch.tensor([-1., 1.])
        result, stats = self.combine(a, b, "pcgrad")
        self.assertTrue(stats["conflict"])
        self.assertGreaterEqual(float(result@a), 0.); self.assertGreaterEqual(float(result@b), 0.)
        self.assertTrue(torch.allclose(result.norm(), (a+b).norm()))

    def test_balance_equal_task_contributions_and_ordinary_norm(self):
        a, b = torch.tensor([4., 0.]), torch.tensor([0., 1.])
        result, stats = self.combine(a, b, "balance")
        self.assertAlmostEqual(float(result[0]), float(result[1]))
        self.assertTrue(torch.allclose(result.norm(), (a+b).norm()))
        self.assertFalse(stats["conflict"])

    def test_zero_and_antiparallel_are_finite_and_explicit(self):
        for mode in ("mean", "balance", "pcgrad"):
            result, stats = self.combine(torch.zeros(2), torch.tensor([1., 2.]), mode)
            self.assertTrue(torch.equal(result, torch.tensor([1., 2.]))); self.assertTrue(stats["zero_task"])
        result, stats = self.combine(torch.tensor([2., 0.]), torch.tensor([-1., 0.]), "pcgrad")
        self.assertTrue(torch.equal(result, torch.zeros(2))); self.assertTrue(stats["degenerate_sum"])

    def test_cycles_preserve_selected_draws_tasks_and_noise_pairing(self):
        rows = [{"task": task, "episode": f"{task}-{i}", "observation_index": 0, "success": bool(i%2)}
                for task in ("a", "b") for i in range(64)]
        report = {"weights": [True]*128, "sequence": [{"episode": r["episode"], "observation_index": 0, "noise_key": i+1000}
                                                        for i, r in enumerate(rows)]}
        pool = SimpleNamespace(rows=rows, sample=lambda i: ({"state": torch.tensor([i])}, torch.zeros(32, 32), torch.ones(32, 32, dtype=torch.bool)))
        first = SelectedPairs(pool, report, ["a", "b"], 128, 0)
        second = SelectedPairs(pool, report, ["a", "b"], 128, 0)
        smoke = SelectedPairs(pool, report, ["a", "b"], 4, 0)
        self.assertEqual(len(first), 512); self.assertEqual(first.records[:16], smoke.records)
        self.assertEqual(set(Counter(r["source_draw"] for r in first.records).values()), {4})
        for start in range(0, len(first), 4):
            self.assertEqual([r["task_id"] for r in first.records[start:start+4]], [0, 0, 1, 1])
        self.assertTrue(torch.equal(first[17]["noise"], second[17]["noise"]))
        self.assertFalse(torch.equal(first[0]["noise"], first[128]["noise"]))


if __name__ == "__main__": unittest.main()
