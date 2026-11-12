"""Read-only inventory for a fresh RECAP host; does not claim runtime validation."""
import fnmatch
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    checks = {}
    for tool in ('uv', 'nvcc', 'ffmpeg', 'nvidia-smi'):
        checks['tool:' + tool] = shutil.which(tool) is not None
    for path in ('third_party/RLinf/rlinf', 'third_party/RoboTwin/envs',
                 '.runtime/venv-rlinf/bin/python',
                 '.runtime/cache/openpi/big_vision/paligemma_tokenizer.model',
                 'third_party/RoboTwin/assets/objects',
                 'datasets/RoboTwin2.0-clean'):
        checks[path] = (ROOT / path).exists()
    for lock_name in ('downloads.lock.json', 'recap-models.lock.json'):
        for model in json.loads((ROOT / 'bootstrap' / lock_name).read_text()):
            if not model['destination'].startswith('models/'):
                continue
            dest = ROOT / model['destination']
            # Inventory only: full file hashes are checked by the download verifiers.
            selected = [f for f in model['files'] if any(
                fnmatch.fnmatchcase(f['path'], pattern)
                for pattern in model.get('patterns', ['*']))]
            checks[model['destination'] + ':files'] = bool(selected) and all(
                (dest / f['path']).is_file() for f in selected)
            marker = dest / 'DOWNLOAD_COMPLETE'
            checks[model['destination'] + ':revision'] = (
                marker.is_file() and marker.read_text().strip() == model['revision'])
    for key, ready in checks.items():
        print(('PRESENT ' if ready else 'MISSING ') + key)
    usage = shutil.disk_usage(ROOT)
    print(f'Free disk: {usage.free / 2**30:.1f} GiB')
    if shutil.which('nvidia-smi'):
        subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                        '--format=csv'], check=False)
    print('Inventory only; package compatibility, CUDA, rendering, data and model loading require smoke checks.')
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
