"""Protect the official handoff, dataset eligibility and workflow fail-stop."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sft_recap", ROOT / "scripts/sft_recap.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class SftRecapTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        self.config = ROOT / "configs/pipelines/robotwin2_sft_recap.yaml"

    def test_smoke_binds_sft_value_labels_and_actor_without_validation_leak(self):
        run = self.folder / "plan"
        manifest = pipeline.plan(self.config, run, [0, 1], smoke=True)
        sft, value, advantages, policy = [pipeline.composed(run / f"configs/{s}.yaml") for s in
                                         ("sft", "value", "advantages", "policy")]
        self.assertEqual(manifest["benchmark"], "RoboTwin 2.0")
        self.assertEqual(sft.actor.model.openpi.config_name, "pi05_aloha_robotwin_h32")
        self.assertEqual(sft.actor.model.num_action_chunks, 32)
        self.assertFalse(sft.actor.model.openpi.train_expert_only)
        self.assertFalse(policy.actor.model.openpi.train_expert_only)
        for cfg in (sft, value, policy):
            self.assertEqual(cfg.runner.max_steps, 2)
            self.assertEqual(cfg.runner.save_interval, 2)
        self.assertEqual(value.data.train_data_paths, advantages.data.train_data_paths)
        self.assertEqual(value.data.train_data_paths, policy.data.train_data_paths)
        self.assertNotIn(value.data.eval_data_paths[0].dataset_path,
                         [v.dataset_path for v in policy.data.train_data_paths])
        self.assertEqual(advantages.advantage.returns_tag, value.data.tag)
        self.assertEqual(policy.data.advantage_tag, value.data.tag)
        self.assertEqual(policy.actor.model.model_path, str(run / "cfg-initial"))
        self.assertIn("global_step_2/actor/model_state_dict/full_weights.pt", advantages.advantage.value_checkpoint)
        with self.assertRaisesRegex(ValueError, "existing plans"):
            pipeline.plan(self.config, run)

    def test_final_test_range_and_bad_accumulation_are_rejected_before_plan_creation(self):
        document = OmegaConf.load(self.config)
        document.workflow.evaluation.seed_indices = [500, 501]
        path = self.folder / "bad.yaml"
        OmegaConf.save(document, path)
        with self.assertRaisesRegex(ValueError, "final-test"):
            pipeline.plan(path, self.folder / "must-not-exist")
        self.assertFalse((self.folder / "must-not-exist").exists())
        with self.assertRaisesRegex(ValueError, "divisible"):
            pipeline.plan(self.config, self.folder / "other", [0, 1, 2])

    def dataset(self, name, successes, role="rollout"):
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = self.folder / name
        (path / "meta").mkdir(parents=True)
        (path / "data/chunk-000").mkdir(parents=True)
        info = {"fps": 15, "total_episodes": len(successes), "total_frames": 2 * len(successes),
                "features": {key: {"shape": [14]} for key in ("action", "observation.state")}}
        for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            info["features"]["observation.images." + camera] = {"dtype": "image"}
        episodes = []
        for i, success in enumerate(successes):
            episode = {"episode_index": i, "episode_id": name + str(i), "task_id": "click_bell",
                       "scene_family_id": name + str(i), "instruction": "press the bell",
                       "simulator_seed": 100000 + i, "source_type": "policy", "behavior_checkpoint_sha256": "a" * 64,
                       "success": success, "split": "validation" if role == "validation" else "train",
                       "checker_version": "fixture", "alignment_verified": True, "final_observation_recorded": True,
                       "termination_reason": "success" if success else "benchmark_limit",
                       "terminated": success, "truncated": not success}
            episodes.append(episode)
            pq.write_table(pa.table({"episode_index": [i, i], "frame_index": [0, 1],
                                     "task_index": [0, 0], "is_success": [[success], [success]]}), path / f"data/chunk-000/episode_{i:06d}.parquet")
        pipeline.write_json(path / "meta/info.json", info)
        (path / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "press the bell"}) + "\n")
        pipeline.write_json(path / "meta/post_training_provenance.json", {
            "schema_version": 1, "observation_sampling": "control_step", "episodes": episodes})
        return path

    def test_real_parquet_labels_and_control_step_provenance_are_checked(self):
        path = self.dataset("mixed", [True, False])
        workflow = {"tasks": ["click_bell"], "return_min": -1000, "failure_reward": -300}
        result = pipeline.audit_dataset(path, "rollout", workflow)
        self.assertEqual(len(result["episodes"]), 2)
        with self.assertRaisesRegex(ValueError, "verified successes"):
            pipeline.audit_dataset(path, "sft", workflow)
        provenance = json.loads((path / "meta/post_training_provenance.json").read_text())
        provenance["observation_sampling"] = "chunk_boundary"
        pipeline.write_json(path / "meta/post_training_provenance.json", provenance)
        with self.assertRaisesRegex(ValueError, "genuine control-step"):
            pipeline.audit_dataset(path, "rollout", workflow)

    def test_failure_halts_pipeline_and_records_the_failing_stage(self):
        run = self.folder / "run"
        run.mkdir()
        pipeline.write_json(run / "plan.json", {"sequence": ["returns", "advantages"], "tag": "test",
            "sources": {}, "configs": {}, "workflow": {"gpu_ids": [0], "python": sys.executable,
                "min_free_gib": 0, "estimated_peak_additional_gib": 0}})
        pipeline.write_json(run / "status.json", {"phase": "planned", "completed": {}})
        pipeline.write_json(run / "input-binding.json", {})
        with patch.object(pipeline, "check_inputs", return_value={}), patch.object(
            pipeline.subprocess, "run", side_effect=[subprocess.CompletedProcess([], 0), subprocess.CalledProcessError(7, [])]) as child:
            with self.assertRaises(subprocess.CalledProcessError):
                pipeline.execute(run)
        self.assertEqual(child.call_count, 2)  # source audit, then the failed first stage
        state = json.loads((run / "status.json").read_text())
        self.assertEqual(state["phase"], "failed")
        self.assertEqual(state["active_stage"], "returns")
        self.assertEqual(state["completed"], {})

    def test_audited_scene_family_cannot_cross_training_and_validation(self):
        paths = {"sft": self.dataset("sft", [True], "sft"),
                 "rollout": self.dataset("rollout", [True, False]),
                 "validation": self.dataset("validation", [True, False], "validation")}
        provenance_path = paths["validation"] / "meta/post_training_provenance.json"
        provenance = json.loads(provenance_path.read_text())
        for i, episode in enumerate(provenance["episodes"]):
            episode["simulator_seed"] = 200000 + i
        pipeline.write_json(provenance_path, provenance)
        for key in ("base", "norm", "siglip/config.json", "siglip/model.safetensors", "gemma/config.json", "gemma/model.safetensors"):
            path = self.folder / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture")
        workflow = {"tasks": ["click_bell"], "return_min": -1000, "failure_reward": -300,
                    "datasets": {k: str(v) for k, v in paths.items()}, "python": sys.executable,
                    "base_weights": str(self.folder / "base"), "norm_stats": str(self.folder / "norm"),
                    "base_sha256": pipeline.digest(self.folder / "base"), "norm_sha256": pipeline.digest(self.folder / "norm"),
                    "siglip": str(self.folder / "siglip"), "gemma": str(self.folder / "gemma"),
                    "evaluation": {"protocol": str(ROOT / "configs/multitask/phase0.json"),
                                   "seed_indices": [300, 304], "noise_seed_base": 7}}
        self.assertEqual(len(pipeline.check_inputs({"workflow": workflow})["datasets"]), 3)
        provenance["episodes"][0]["scene_family_id"] = "sft0"
        pipeline.write_json(provenance_path, provenance)
        with self.assertRaisesRegex(ValueError, "scene_family_id overlap"):
            pipeline.check_inputs({"workflow": workflow})

    def test_registered_window_matches_the_real_official_sft_loader(self):
        pipeline.environment()
        from vla_rl_mt.recap_workflow import register
        from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
        from rlinf.data.datasets.openpi.official_sft_data_loader import _validate_openpi_model_shape

        original = get_openpi_config("pi05_aloha_robotwin")
        self.assertEqual(original.model.action_horizon, 50)
        register()
        register()  # Driver and workers both call the extension.
        config = get_openpi_config("pi05_aloha_robotwin_h32")
        model = OmegaConf.create({"openpi": {"action_horizon": 32, "model_action_dim": 32, "config_name": "pi05_aloha_robotwin_h32"}})
        _validate_openpi_model_shape(model, config)
        with self.assertRaisesRegex(ValueError, "different network"):
            _validate_openpi_model_shape(model, original)
        self.assertEqual(get_openpi_config("pi05_aloha_robotwin").model.action_horizon, 50)


if __name__ == "__main__":
    unittest.main()
