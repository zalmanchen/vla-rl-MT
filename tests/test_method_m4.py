"""M4 mask reduction, train-only label controls and sequence isolation."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from vla_rl_mt.policy_update import PolicyPairs, label_array, masked_per_sample


class MethodM4Tests(unittest.TestCase):
    def test_queue_rejects_dependencies_that_could_leave_it_waiting_forever(self):
        path = Path(__file__).resolve().parents[1]/"scripts/run_method_jobs.py"
        spec = importlib.util.spec_from_file_location("method_jobs", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.validate_jobs([{"name": "a"}, {"name": "b", "depends_on": ["a"]}])
        for jobs in [[{"name": "a"}, {"name": "a"}], [{"name": "a", "depends_on": ["missing"]}],
                     [{"name": "a", "depends_on": ["b"]}, {"name": "b", "depends_on": ["a"]}]]:
            with self.assertRaises(ValueError):
                module.validate_jobs(jobs)

    def pool(self):
        rows = [{"episode": f"{task}-{split}-{episode}", "task": task, "split": split,
                 "step": 0, "observation_index": 0, "success": episode % 2 == 0, "target": float(episode % 2 == 0)}
                for task in ("a", "b") for split in ("train", "val") for episode in range(20)]
        pool = SimpleNamespace(rows=rows, sample=lambda index: ({"state": torch.tensor([index])}, torch.zeros(32, 32),
                                                               torch.ones(32, 32, dtype=torch.bool)))
        return pool

    def test_masks_have_no_padding_gradient_and_weighted_zero_is_finite(self):
        losses = torch.tensor([[[2.0], [100.0]], [[4.0], [4.0]]], requires_grad=True)
        mask = torch.tensor([[[True], [False]], [[True], [True]]])
        per_sample = masked_per_sample(losses, mask)
        self.assertTrue(torch.equal(per_sample, torch.tensor([2.0, 4.0])))
        total = (per_sample*torch.tensor([1.0, 0.0])).mean()
        self.assertEqual(float(total), 1.0)
        total.backward()
        self.assertEqual(float(losses.grad[0, 1]), 0)
        self.assertEqual(float(losses.grad[1].abs().sum()), 0)

    def test_shuffled_labels_preserve_scheduled_exposure_and_never_shuffle_val(self):
        pool = self.pool()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"labels.json"
            path.write_text(json.dumps({"rows": [{**r, "use_for_loss": True, "positive": i % 3 == 0}
                                                 for i, r in enumerate(pool.rows)]}))
            normal = PolicyPairs(pool, mode="c2_cond", label_path=path, steps=128, batch_size=4, seed=0)
            shuffled = PolicyPairs(pool, mode="shuffle_cond", label_path=path, steps=128, batch_size=4, seed=0)
            self.assertTrue(torch.equal(normal.draws, shuffled.draws))
            self.assertTrue(any(normal.labels != shuffled.labels))
            for task in ("a", "b"):
                indices = [i for i, r in enumerate(pool.rows) if r["task"] == task and r["split"] == "train"]
                self.assertEqual(int(normal.labels[indices].sum()), int(shuffled.labels[indices].sum()))
                for split in ("train", "val"):
                    indices = [i for i, r in enumerate(pool.rows) if r["task"] == task and r["split"] == split]
                    if split == "val":
                        self.assertTrue(torch.equal(normal.labels[indices], shuffled.labels[indices]))
                for conditional in (False, True):
                    totals = []
                    for dataset in (normal, shuffled):
                        count = 0
                        for i, draw in enumerate(dataset.draws):
                            if pool.rows[int(draw)]["task"] != task:
                                continue
                            sample = dataset[i]
                            count += bool(sample["positive"]) and (not conditional or sample["routing_random"] > 0.1)
                        totals.append(int(count))
                    self.assertEqual(totals[0], totals[1])
            first, second = normal[3], shuffled[3]
            self.assertTrue(torch.equal(first["noise"], second["noise"]))

    def test_label_identity_change_is_rejected_and_bc_uses_only_success_train(self):
        pool = self.pool()
        bc = PolicyPairs(pool, mode="bc", label_path="none", steps=10, batch_size=4, seed=0)
        self.assertTrue(all(pool.rows[int(i)]["split"] == "train" and pool.rows[int(i)]["success"] for i in bc.draws))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"labels.json"
            rows = [{**row, "use_for_loss": True, "positive": True} for row in pool.rows]
            rows[0]["target"] = 0.5
            path.write_text(json.dumps({"rows": rows}))
            with self.assertRaisesRegex(ValueError, "identity"):
                label_array(pool, path)


if __name__ == "__main__":
    unittest.main()
