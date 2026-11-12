"""Launch official RLinf Python entrypoints with explicit Hydra configs."""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = {
    'sft': 'examples/sft/train_vla_sft.py',
    'recap-returns': 'examples/offline_rl/advantage_labeling/recap/process/compute_returns.py',
    'recap-value': 'examples/offline_rl/advantage_labeling/recap/train_value.py',
    'recap-advantages': 'examples/offline_rl/advantage_labeling/recap/process/compute_advantages.py',
    'recap-policy': 'examples/offline_rl/policy_optimization/cfg_rl/train_cfg.py',
    'train': 'examples/embodiment/train_embodied_agent.py',
    'eval': 'evaluations/eval_embodied_agent.py',
}


def environment(root, profile='robotwin'):
    env = os.environ.copy()
    official = profile == 'recap-official'
    upstream = root / ('third_party/RLinf-recap-official' if official else 'third_party/RLinf')
    if official:
        for key in ('RLINF_EXT_MODULE', 'ROBOTWIN_PATH', 'ROBOT_PLATFORM', 'ASSETS_PATH'):
            env.pop(key, None)
        env.update(PROJECT_ROOT=str(root), REPO_PATH=str(upstream),
                   EMBODIED_PATH=str(upstream / 'examples/embodiment'),
                   PYTHONPATH=str(upstream), HF_HOME=str(root / '.runtime/cache/huggingface'),
                   HF_LEROBOT_HOME=str(root / 'datasets/recap-official'),
                   OPENPI_DATA_HOME=str(root / '.runtime/cache/openpi'),
                   LIBERO_CONFIG_PATH=str(root / '.runtime/recap-official/libero-config'),
                   MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl')
        return env
    robotwin = root / 'third_party/RoboTwin'
    env.update(PROJECT_ROOT=str(root), REPO_PATH=str(upstream),
               EMBODIED_PATH=str(upstream / 'examples/embodiment'), ROBOTWIN_PATH=str(robotwin))
    paths = [str(root / 'src'), str(upstream), str(robotwin)]
    if env.get('PYTHONPATH'):
        paths.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = os.pathsep.join(paths)
    env.setdefault('ROBOT_PLATFORM', 'ALOHA')
    return env


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    overrides = []
    if '--' in argv:
        split = argv.index('--')
        argv, overrides = argv[:split], argv[split + 1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=ENTRYPOINTS)
    parser.add_argument('--config', required=True, type=Path, help='Project or upstream Hydra YAML file')
    parser.add_argument('--profile', choices=['robotwin', 'recap-official'], default='robotwin',
                        help='Select the source tree and default Python environment')
    parser.add_argument('--python', type=Path)
    parser.add_argument('--nproc', type=int, default=1, help='Local GPU processes for recap-advantages only')
    parser.add_argument('--dry-run', action='store_true', help='Print command without importing RLinf or training')
    args = parser.parse_args(argv)
    official = args.profile == 'recap-official'
    python = args.python or ROOT / ('.runtime/venv-recap-official/bin/python' if official
                                   else '.runtime/venv-rlinf/bin/python')
    if args.nproc < 1 or (args.nproc != 1 and args.stage != 'recap-advantages'):
        parser.error('--nproc > 1 is only supported for recap-advantages; other distributed stages use RLinf placement')
    config = args.config.resolve()
    upstream = ROOT / ('third_party/RLinf-recap-official' if official else 'third_party/RLinf')
    entry = upstream / ENTRYPOINTS[args.stage]
    if not config.is_file() or config.suffix != '.yaml':
        parser.error(f'Config must be an existing .yaml file: {config}')
    if not entry.is_file():
        parser.error(f'RLinf source missing: {entry}; run bootstrap/prepare_sources.py')
    command = [str(python.absolute())]
    if args.nproc > 1:
        command += ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', str(args.nproc)]
    command += [str(entry), '--config-path', str(config.parent), '--config-name', config.stem, *overrides]
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    if not python.is_file():
        parser.error(f'Python environment missing: {python}')
    if official:
        (ROOT / '.runtime/recap-official/libero-config').mkdir(parents=True, exist_ok=True)
    return subprocess.run(command, cwd=ROOT, env=environment(ROOT, args.profile), check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
