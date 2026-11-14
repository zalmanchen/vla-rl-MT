"""Cross-task controls must share exact examples and stochastic FM targets."""

import unittest
from types import SimpleNamespace

import torch

from vla_rl_mt.task_mixing import TaskPairs


class TaskMixingTests(unittest.TestCase):
    def pool(self):
        rows = [{"episode": f"{task}-{split}-{episode}", "task": task, "split": split,
                 "success": episode != 0, "observation_index": decision}
                for task in ("a", "b") for split in ("train", "val")
                for episode in range(4) for decision in range(episode+1)]
        return SimpleNamespace(rows=rows, sample=lambda i: ({"state": torch.tensor([i])}, torch.zeros(32, 32),
                                                           torch.ones(32, 32, dtype=torch.bool)))

    def test_exact_per_task_data_and_noise_exposure(self):
        pool = self.pool()
        joint = TaskPairs(pool, "mixed", 128, tasks=["a", "b"])
        for group, task_index in (("a_only", 0), ("b_only", 1)):
            single = TaskPairs(pool, group, 64, tasks=["a", "b"])
            positions = [i for i, task in enumerate(joint.task_indices) if task == task_index]
            self.assertEqual(len(positions), 256)
            self.assertEqual([joint.sequence()[i] for i in positions], single.sequence())
            for i in (0, 13, 255):
                for key in ("noise", "flow_time", "actions", "action_mask", "routing_random"):
                    self.assertTrue(torch.equal(joint[positions[i]][key], single[i][key]))

    def test_fixed_batch_balance_success_only_and_no_validation(self):
        pool = self.pool()
        pairs = TaskPairs(pool, "mixed", 128, tasks=["a", "b"])
        for start in range(0, len(pairs), 4):
            self.assertEqual(pairs.task_indices[start:start+4], [0, 0, 1, 1])
        for index in pairs.draws:
            self.assertEqual(pool.rows[int(index)]["split"], "train")
            self.assertTrue(pool.rows[int(index)]["success"])

    def test_smoke_prefix_and_total_budget(self):
        pool = self.pool()
        for group in ("a_only", "b_only", "mixed"):
            main = TaskPairs(pool, group, 128, tasks=["a", "b"])
            smoke = TaskPairs(pool, group, 4, tasks=["a", "b"])
            self.assertEqual(len(main), 512)
            self.assertEqual(main.sequence()[:16], smoke.sequence())


if __name__ == "__main__":
    unittest.main()
