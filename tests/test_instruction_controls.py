"""Goal specificity and label randomization must preserve declared controls."""

import sys
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from evaluate_instruction_response import goal_predicates
from prepare_label_exposure import exact_subset, matched_permutation


class Actor:
    def __init__(self, position):
        self.position = np.asarray(position)

    def get_pose(self):
        return SimpleNamespace(p=self.position)

    def get_functional_point(self, index):
        return self.position


class InstructionControlTests(unittest.TestCase):
    def test_exclusive_bottle_goals_and_both_control(self):
        task = SimpleNamespace(bottle1=Actor([-.06, -.105, .95]), bottle2=Actor([.16, .1, .8]),
                               left_target_pose=[-.06, -.105, 1], right_target_pose=[.06, -.105, 1])
        self.assertEqual(goal_predicates(task, "pick_dual_bottles"), {"left": True, "right": False, "both": False})
        task.bottle2 = Actor([.06, -.105, .95])
        self.assertEqual(goal_predicates(task, "pick_dual_bottles"), {"left": False, "right": False, "both": True})

    def test_stack_order_and_open_grippers(self):
        task = SimpleNamespace(block1=Actor([0, 0, .77]), block2=Actor([0, 0, .82]),
                               is_left_gripper_open=lambda: True, is_right_gripper_open=lambda: True)
        self.assertEqual(goal_predicates(task, "stack_blocks_two"), {"green_on_red": True, "red_on_green": False})
        task.block1, task.block2 = task.block2, task.block1
        self.assertEqual(goal_predicates(task, "stack_blocks_two"), {"green_on_red": False, "red_on_green": True})
        task.is_left_gripper_open = lambda: False
        self.assertFalse(any(goal_predicates(task, "stack_blocks_two").values()))

    def test_permutation_preserves_outcomes_instructions_and_exposure(self):
        rows = [{"episode": str(i//8), "task": "a", "success": bool(i//16), "split": "train",
                 "step": 0, "next_step": 32} for i in range(32)]
        instructions = {str(i): str(i%2) for i in range(4)}
        labels = torch.tensor([False, True]*16)
        draws = torch.tensor(list(range(32))*2)
        eligible = torch.tensor([True]*32+[False]*32)
        changed = matched_permutation(labels, rows, instructions, draws, eligible, 0)
        def counts(bits):
            return Counter((rows[int(i)]["success"], instructions[rows[int(i)]["episode"]])
                           for i, ok in zip(draws, eligible) if ok and bits[int(i)])
        self.assertEqual(counts(labels), counts(changed))
        self.assertTrue(bool((changed != labels).any()))
        self.assertTrue(torch.equal(changed, matched_permutation(labels, rows, instructions, draws, eligible, 0)))

    def test_exact_exposure_subset_or_explicit_failure(self):
        weights = {0: 2, 1: 3, 2: 7}
        chosen = exact_subset(list(weights), weights, 5)
        self.assertEqual(sum(weights[i] for i in chosen), 5)
        with self.assertRaises(ValueError):
            exact_subset(list(weights), weights, 1)


if __name__ == "__main__":
    unittest.main()
