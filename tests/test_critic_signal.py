"""Critic and actor comparisons must preserve their actual supervision budgets."""

import unittest
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch

from vla_rl_mt.signal_value import HeadPairs
from vla_rl_mt.signal_policy import CandidatePairs
from vla_rl_mt.value_head import PrefixValueHead

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from plan_critic_signal import advantages, shuffle_weights, top_weights


class CriticSignalTests(unittest.TestCase):
    def rows(self):
        return [{"task": task, "episode": f"{task}-{split}-{episode}", "split": split,
                 "success": bool(episode % 2), "target": float(episode % 2), "observation_index": index}
                for task in ("a", "b") for split in ("train", "val") for episode in range(6) for index in range(episode+1)]

    def test_head_capacity_control_is_equal_to_three_parameters(self):
        count = lambda h: sum(p.numel() for p in PrefixValueHead(4096, h).parameters())
        self.assertEqual(count(63), 266367)
        self.assertEqual(count(128)-2*count(63), 3)

    def test_shared_and_independent_heads_use_identical_per_task_draws(self):
        rows = self.rows()
        cache = {"rows": rows, "features": torch.zeros(len(rows), 4)}
        shared = HeadPairs(cache, ["a", "b"], [0, 1], 1000, 1)
        for task_index, task in enumerate(("a", "b")):
            independent = HeadPairs(cache, ["a", "b"], [task_index], 500, 1)
            selected = [int(i) for i in shared.draws if rows[int(i)]["task"] == task]
            self.assertEqual(selected, independent.draws.tolist())
            self.assertEqual(len(selected), 16000)
            self.assertTrue(all(rows[i]["split"] == "train" for i in selected))

    def test_actor_masks_change_only_loss_selection_and_preserve_smoke_prefix(self):
        rows = self.rows()
        pool = SimpleNamespace(rows=rows, sample=lambda i: ({"state": torch.tensor([i])}, torch.zeros(32, 32),
                                                           torch.ones(32, 32, dtype=torch.bool)))
        positive = CandidatePairs(pool, ["a", "b"], 128, [True]*512)
        negative = CandidatePairs(pool, ["a", "b"], 128, [False]*512)
        smoke = CandidatePairs(pool, ["a", "b"], 4, [True]*512)
        self.assertTrue(torch.equal(positive.draws, negative.draws))
        self.assertEqual(positive.noise_keys[:16], smoke.noise_keys)
        for index in (0, 15, 511):
            for key in ("actions", "action_mask", "noise", "flow_time"):
                self.assertTrue(torch.equal(positive[index][key], negative[index][key]))
            self.assertEqual(float(positive[index]["routing_random"]), .5)
            self.assertNotEqual(bool(positive[index]["positive"]), bool(negative[index]["positive"]))
            self.assertEqual(rows[int(positive.draws[index])]["split"], "train")

    def test_top_quota_and_strict_permutation_preserve_actual_composition(self):
        rows = [{"task": "a" if i < 32 else "b", "episode": str(i), "success": bool(i % 2),
                 "step": 0, "next_step": 32} for i in range(64)]
        draws = torch.tensor(list(range(64))*8)
        instructions = {str(i): rows[i]["task"] for i in range(64)}
        weights = top_weights(torch.arange(64).float()%16, draws, rows, ["a", "b"], 64)
        shuffled = shuffle_weights(weights, draws, rows, instructions, 0)
        def counts(bits):
            return Counter((rows[int(i)]["task"], rows[int(i)]["success"])
                           for index, i in enumerate(draws) if bits[index])
        self.assertEqual(int(weights.sum()), 128)
        self.assertEqual(counts(weights), counts(shuffled))
        self.assertGreaterEqual(int((weights != shuffled).sum()), 10)

    def test_td_labels_use_true_successor_and_zero_terminal_bootstrap(self):
        rows = [{"episode": "a", "step": 0, "next_step": 32, "success": True},
                {"episode": "a", "step": 32, "next_step": 64, "success": True},
                {"episode": "b", "step": 0, "next_step": 32, "success": False}]
        values = advantages(rows, [.2, .6, .7])
        self.assertTrue(torch.allclose(torch.tensor(values), torch.tensor([.4, .4, -.7], dtype=torch.float64)))
        rows[1]["step"] = 31
        with self.assertRaises(ValueError):
            advantages(rows, [.2, .6, .7])


if __name__ == "__main__":
    unittest.main()
