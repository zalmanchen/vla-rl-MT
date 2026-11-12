"""Check pipeline failure reporting and experiment-label isolation at process edges."""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "recap_pipeline", ROOT / "scripts/recap.py"
)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class RecapPipelineTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        for path in ["bootstrap", "scripts", "configs/robotwin_pilot", ".runtime/bootstrap"]:
            (self.root / path).mkdir(parents=True, exist_ok=True)
        for config in (ROOT / "configs/robotwin_pilot").glob("*.yaml"):
            shutil.copy2(config, self.root / "configs/robotwin_pilot" / config.name)
        for split in ["sft-train", "sft-val", "rollout-train", "rollout-val"]:
            p = self.root / "datasets/lerobot/recap-pilot-v2" / split / "meta"
            p.mkdir(parents=True)
            (p / "conversion.json").write_text("{}")
        (self.root / "bootstrap/apply_recap_patch.py").write_text("pass\n")
        (self.root / "scripts/run.py").write_text(
            "import os, sys\nfrom pathlib import Path\n"
            "config=Path(sys.argv[sys.argv.index('--config')+1])\n"
            "if not config.is_file(): raise FileNotFoundError(config)\n"
            "stage=sys.argv[1].removeprefix('recap-')\n"
            "with open('stages.txt','a') as f: f.write(stage+'\\n')\n"
            "if os.environ.get('PIPELINE_TEST_FAIL')==stage: raise SystemExit(9)\n"
            "overrides=dict(x.split('=',1) for x in sys.argv[sys.argv.index('--')+1:])\n"
            "if stage in ['value','policy']:\n"
            " p=Path(overrides['runner.logger.log_path'])/stage/'checkpoints'/('global_step_'+overrides['runner.max_steps'])/'actor/model_state_dict/full_weights.pt'\n"
            " p.parent.mkdir(parents=True); p.write_bytes(b'process-edge fixture')\n"
        )
        self.state = self.root / ".runtime/bootstrap/recap-run.json"
        self.addCleanup(patch.stopall)
        patch.object(pipeline, "ROOT", self.root).start()
        patch.object(pipeline, "STATE", self.state).start()
        patch.object(pipeline, "PYTHON", Path(sys.executable)).start()

    def args(self, name):
        return argparse.Namespace(
            name=name,
            command="smoke",
            value_steps=2,
            policy_steps=2,
            gpu=0,
            eval_episodes=0,
        )

    def test_separate_runs_keep_unique_tags_and_source_snapshots(self):
        pipeline.run_pipeline(self.args("one"))
        first = json.loads(self.state.read_text())
        pipeline.run_pipeline(self.args("two"))
        second = json.loads(self.state.read_text())
        self.assertNotEqual(first["tag"], second["tag"])
        self.assertEqual(
            second["completed_stages"], ["returns", "value", "advantages", "policy"]
        )
        self.assertEqual(second["status"], "completed")
        self.assertTrue((Path(first["run_dir"]) / "source.zip").is_file())
        self.assertTrue((Path(first["run_dir"]) / "status.json").is_file())

    def test_child_failure_stops_later_stages_and_preserves_log(self):
        with patch.dict(os.environ, {"PIPELINE_TEST_FAIL": "value"}):
            with self.assertRaises(subprocess.CalledProcessError):
                pipeline.run_pipeline(self.args("failure"))
        state = json.loads(self.state.read_text())
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["stage"], "value")
        self.assertEqual(state["completed_stages"], ["returns"])
        self.assertEqual(
            (self.root / "stages.txt").read_text().splitlines(), ["returns", "value"]
        )
        self.assertTrue(Path(state["current_log"]).is_file())


if __name__ == "__main__":
    unittest.main()
