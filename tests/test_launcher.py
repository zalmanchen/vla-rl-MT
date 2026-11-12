"""Check launcher forwarding, environment discovery, and child failure reporting."""
import importlib.util
import io
import json
import os
import shlex
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('launcher', Path(__file__).resolve().parents[1] / 'scripts/run.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.entry = self.root / 'third_party/RLinf/probe.py'
        self.entry.parent.mkdir(parents=True)
        self.config = self.root / 'config.yaml'
        self.config.write_text('data: {}\n')
        self.entry.write_text(
            'import json, os, sys\nfrom pathlib import Path\n'
            'Path("captured.json").write_text(json.dumps({"argv": sys.argv[1:], '
            '"root": os.environ["PROJECT_ROOT"], "cwd": os.getcwd()}))\n'
            'raise SystemExit(7)\n')
        self.addCleanup(patch.stopall)
        patch.object(launcher, 'ROOT', self.root).start()
        patch.dict(launcher.ENTRYPOINTS, {'probe': 'probe.py'}).start()

    def test_forwards_hydra_arguments_and_child_exit(self):
        overrides = ['data.path=a b', 'label=$(literal)', '--cfg', 'job']
        result = launcher.main(['probe', '--config', str(self.config), '--python', sys.executable, '--', *overrides])
        self.assertEqual(result, 7)
        capture = json.loads((self.root / 'captured.json').read_text())
        self.assertEqual(capture['argv'], ['--config-path', str(self.root), '--config-name', 'config', *overrides])
        self.assertEqual(capture['root'], str(self.root))
        self.assertEqual(capture['cwd'], str(self.root))

    def test_dry_run_does_not_require_environment_or_launch_child(self):
        self.assertEqual(launcher.main(['probe', '--config', str(self.config), '--dry-run']), 0)
        self.assertFalse((self.root / 'captured.json').exists())

    def test_preserves_virtualenv_python_symlink(self):
        python = self.root / 'venv/bin/python'
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        output = io.StringIO()
        with redirect_stdout(output):
            launcher.main(['probe', '--config', str(self.config), '--python', str(python), '--dry-run'])
        self.assertEqual(shlex.split(output.getvalue())[0], str(python))

    def test_multi_gpu_advantages_uses_official_torchrun_module(self):
        entry = self.root / 'third_party/RLinf' / launcher.ENTRYPOINTS['recap-advantages']
        entry.parent.mkdir(parents=True)
        entry.touch()
        output = io.StringIO()
        with redirect_stdout(output):
            launcher.main(['recap-advantages', '--config', str(self.config), '--nproc', '2', '--dry-run'])
        command = shlex.split(output.getvalue())
        self.assertEqual(command[1:6], ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', '2'])
        self.assertEqual(command[6], str(entry))

    def test_ray_stages_reject_torchrun_process_count(self):
        with self.assertRaises(SystemExit) as error:
            launcher.main(['train', '--config', str(self.config), '--nproc', '2', '--dry-run'])
        self.assertEqual(error.exception.code, 2)

    def test_missing_config_fails_before_launch(self):
        with self.assertRaises(SystemExit) as error:
            launcher.main(['probe', '--config', str(self.root / 'missing.yaml')])
        self.assertEqual(error.exception.code, 2)
        self.assertFalse((self.root / 'captured.json').exists())

    def test_extensions_are_discoverable_and_environment_is_preserved(self):
        with patch.dict(os.environ, {'PYTHONPATH': '/custom', 'HF_ENDPOINT': 'https://example.org'}):
            env = launcher.environment(self.root)
        self.assertEqual(env['PYTHONPATH'].split(os.pathsep)[0], str(self.root / 'src'))
        self.assertEqual(env['PYTHONPATH'].split(os.pathsep)[-1], '/custom')
        self.assertEqual(env['HF_ENDPOINT'], 'https://example.org')

    def test_official_profile_uses_isolated_source_and_environment(self):
        upstream = self.root / 'third_party/RLinf-recap-official'
        upstream.mkdir(parents=True)
        (upstream / 'probe.py').write_text(self.entry.read_text())
        output = io.StringIO()
        with redirect_stdout(output):
            launcher.main(['probe', '--profile', 'recap-official', '--config', str(self.config), '--dry-run'])
        command = shlex.split(output.getvalue())
        self.assertEqual(command[0], str(self.root / '.runtime/venv-recap-official/bin/python'))
        self.assertEqual(command[1], str(upstream / 'probe.py'))
        with patch.dict(os.environ, {'PYTHONPATH': '/old', 'RLINF_EXT_MODULE': 'custom',
                                     'ROBOT_PLATFORM': 'ALOHA'}):
            env = launcher.environment(self.root, 'recap-official')
        self.assertEqual(env['PYTHONPATH'], str(upstream))
        self.assertEqual(env['REPO_PATH'], str(upstream))
        self.assertEqual(env['MUJOCO_GL'], 'egl')
        self.assertNotIn('RLINF_EXT_MODULE', env)
        self.assertNotIn('ROBOT_PLATFORM', env)


if __name__ == '__main__':
    unittest.main()
