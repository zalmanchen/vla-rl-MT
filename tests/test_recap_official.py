"""Check the official workflow's data boundaries, checkpoint binding and fail-stop."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pipeline = load("official_pipeline", "scripts/recap_official.py")
subset = load("official_subset", "bootstrap/prepare_recap_smoke.py")


class OfficialWorkflowTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.addCleanup(patch.stopall)
        patch.object(pipeline, "ROOT", self.root).start()
        patch.dict(os.environ, {"REPO_PATH": str(self.root / "third_party/RLinf-recap-official")}).start()
        for path in ("bootstrap", "scripts", "configs/recap_official", ".runtime/recap-official/configs",
                     ".runtime/recap-official/assets/physical-intelligence/libero"):
            (self.root / path).mkdir(parents=True)
        (self.root / "bootstrap/recap-official.lock.json").write_text("{}")
        stats = self.root / ".runtime/recap-official/assets/physical-intelligence/libero/norm_stats.json"
        stats.write_text("{}")
        (self.root / "configs/recap_official/protocol.json").write_text(
            (ROOT / "configs/recap_official/protocol.json").read_text())
        for name in ("sft", "value", "advantages", "policy", "eval", "eval_cfg", "returns"):
            (self.root / f".runtime/recap-official/configs/{name}.yaml").write_text("{}\n")
        for source in ("scripts/run.py", "scripts/recap_official.py", "bootstrap/prepare_recap_official.py",
                       "bootstrap/export_recap_cfg.py", "bootstrap/check_recap_formats.py",
                       "bootstrap/prepare_recap_smoke.py"):
            (self.root / source).write_text("# test fixture\n")
        for name in pipeline.SPLITS:
            meta = self.root / "datasets/recap-official" / name / "meta"
            meta.mkdir(parents=True)
            (meta / "info.json").write_text("{}")

    def test_full_plan_preserves_budgets_and_excludes_validation_from_training(self):
        import yaml

        run = self.root / "full-one"
        plan = pipeline.compile_plan(run)
        stages = {step["name"]: step for step in plan["steps"]}
        for name, steps, batch in (("sft", 1000, 128), ("value", 18000, 256), ("policy", 30000, 512)):
            config = yaml.safe_load(Path(stages[name]["config"]).read_text())
            self.assertEqual(config["runner"]["max_steps"], steps)
            self.assertEqual(config["actor"]["global_batch_size"], batch)
        value = yaml.safe_load(Path(stages["value"]["config"]).read_text())
        advantages = yaml.safe_load(Path(stages["advantages"]["config"]).read_text())
        policy = yaml.safe_load(Path(stages["policy"]["config"]).read_text())
        self.assertEqual(value["data"]["train_data_paths"], advantages["data"]["train_data_paths"])
        self.assertEqual(value["data"]["train_data_paths"], policy["data"]["train_data_paths"])
        self.assertNotIn(value["data"]["eval_data_paths"][0], policy["data"]["train_data_paths"])
        self.assertEqual(value["data"]["eval_data_paths"][0]["max_samples"], 10000)
        self.assertEqual(value["data"]["train_num_workers"], 8)
        self.assertEqual(advantages["advantage"]["num_dataloader_workers_per_gpu"], 12)
        self.assertEqual(value["data"]["tag"], policy["data"]["advantage_tag"])
        self.assertEqual(advantages["advantage"]["returns_tag"], value["data"]["tag"])
        self.assertEqual(policy["actor"]["model"]["model_path"], plan["cfg_initial"])
        self.assertEqual(stages["recap-positive-seed1"]["expected_episodes"], 50)
        self.assertTrue((run / "orchestration.zip").is_file())
        self.assertTrue((run / "data/libero10_task0_train/data").is_symlink())

    def fake_run(self, steps):
        run = self.root / "process-test"
        run.mkdir()
        stats = self.root / ".runtime/recap-official/assets/physical-intelligence/libero/norm_stats.json"
        plan = {"steps": steps, "protocol": {"scope": "test"}, "norm_stats": str(stats),
                "norm_stats_sha256": pipeline.digest(stats),
                "source_lock_sha256": pipeline.digest(self.root / "bootstrap/recap-official.lock.json"),
                "native_checkpoint": str(run / "native"), "cfg_initial": str(run / "cfg")}
        pipeline.write_json(run / "plan.json", plan)
        pipeline.write_json(run / "status.json", {"phase": "planned", "completed": {}})
        return run

    def step(self, name, code, outputs=()):
        return {"name": name, "command": [sys.executable, "-c", code], "outputs": list(map(str, outputs)),
                "reserve_gib": 0}

    def test_failed_child_stops_following_stages(self):
        marker = self.root / "must-not-run"
        run = self.fake_run([self.step("failed", "raise SystemExit(7)"),
                             self.step("later", f"open({str(marker)!r}, 'w').close()")])
        with self.assertRaisesRegex(RuntimeError, "exit=7"):
            pipeline.execute(run)
        self.assertFalse(marker.exists())
        self.assertEqual(json.loads((run / "status.json").read_text())["phase"], "failed")

    def test_zero_exit_without_checkpoint_does_not_pass(self):
        run = self.fake_run([self.step("no-checkpoint", "pass", [self.root / "missing.pt"])])
        with self.assertRaisesRegex(RuntimeError, "missing="):
            pipeline.execute(run)
        self.assertEqual(json.loads((run / "status.json").read_text())["completed"], {})

    def test_stage_boundary_preserves_pending_work(self):
        marker = self.root / "pending"
        run = self.fake_run([self.step("first", "pass"),
                             self.step("later", f"open({str(marker)!r}, 'w').close()")])
        pipeline.execute(run, until="first")
        state = json.loads((run / "status.json").read_text())
        self.assertEqual(state["phase"], "ready")
        self.assertEqual(state["remaining_stages"], ["later"])
        self.assertFalse(marker.exists())
        pipeline.execute(run)
        self.assertTrue(marker.exists())

    def test_completed_artifact_is_not_overwritten_and_mutation_is_rejected(self):
        output = self.root / "weights.pt"
        run = self.fake_run([self.step("model", f"open({str(output)!r}, 'x').write('weights')", [output])])
        pipeline.execute(run)
        pipeline.execute(run)  # A second launch must skip the exclusive writer.
        output.write_text("different weights")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            pipeline.execute(run)

    def test_smoke_matches_language_instead_of_sft_task_index(self):
        episodes = [{"episode_index": 0, "tasks": ["unrelated"], "is_success": True}]
        episodes += [{"episode_index": index, "tasks": [subset.TASK], "is_success": True}
                     for index in (15, 16, 17)]
        self.assertEqual([ep["episode_index"] for ep in subset.select_episodes(episodes, True)], [15, 16])
        rollout = [{"episode_index": 8, "tasks": [subset.TASK], "is_success": False}, episodes[-1]]
        self.assertEqual([ep["is_success"] for ep in subset.select_episodes(rollout, False)], [False, True])

    def test_singleton_standard_deviation_is_explicit_in_valid_json(self):
        result = pipeline.report_metrics({"eval/loss": 5.8, "eval/std": float("nan")})
        encoded = json.dumps(result, allow_nan=False)
        self.assertEqual(json.loads(encoded)["eval/std"], None)
        self.assertEqual(result["_nonfinite_metrics"], ["eval/std"])

    def test_summary_uses_episode_counts_and_separates_backends(self):
        run = self.root / "summary-test"
        steps = []
        for name, count, rate in (("sft-native-seed0", 2, 1.0),
                                  ("sft-cfg-unconditional-seed0", 2, 1.0),
                                  ("sft-cfg-unconditional-seed1", 8, 0.25),
                                  ("recap-positive-seed0", 10, 0.6)):
            path = run / name
            path.mkdir(parents=True)
            (path / "result.json").write_text(json.dumps({"eval/num_trajectories": count,
                "eval/success_once": rate, "eval/success_at_end": rate}))
            steps.append({"name": name, "expected_episodes": count})
        result = pipeline.summarize(run, {"steps": steps, "protocol": {"scope": "test"}})
        self.assertEqual(result["evaluations"]["sft-cfg-unconditional"]["success_once_rate"], 0.4)
        self.assertAlmostEqual(result["observed_change_percentage_points"], 20)


if __name__ == "__main__":
    unittest.main()
