"""Finish validation after the already-running installers/downloaders complete."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / '.runtime/bootstrap/recap-preparation.json'
PYTHON = ROOT / '.runtime/venv-rlinf/bin/python'
TOOLS = ROOT / '.runtime/venv-tools/bin/python'


def state(phase, **extra):
    STATE.write_text(json.dumps(dict(phase=phase, pid=os.getpid(), updated_unix=time.time(), **extra), indent=2) + '\n')
    print(phase, flush=True)


def wait_for(test, phase):
    state(phase)
    deadline = time.monotonic() + 24 * 3600
    while not test():
        if time.monotonic() >= deadline:
            raise TimeoutError(phase)
        time.sleep(30)


def run(command, timeout=3600):
    subprocess.run(command, cwd=ROOT, check=True, timeout=timeout)


def env_ready():
    if not PYTHON.is_file():
        return False
    return subprocess.run([str(PYTHON), '-c', 'import torch, sapien, openpi, lerobot'],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60).returncode == 0


def main():
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with (STATE.parent / 'finish-recap-setup.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            wait_for(env_ready, 'waiting_for_base_environment')
            state('validating_headless_render')
            run(['bash', '-c', 'source bootstrap/env.sh && python bootstrap/smoke_render.py'])
            wait_for(lambda: all((ROOT / d / 'DOWNLOAD_COMPLETE').is_file() for d in
                                ['models/pi0.5_robotwin2', 'datasets/RoboTwin2.0']), 'waiting_for_policy_and_clean_data')
            state('verifying_downloads')
            run([str(TOOLS), 'bootstrap/verify_downloads.py'], timeout=12 * 3600)
            state('extracting_assets_and_clean_data')
            run([str(TOOLS), 'bootstrap/extract_assets.py'], timeout=12 * 3600)
            state('validating_policy_and_episode')
            run(['bash', '-c', 'source bootstrap/env.sh && python bootstrap/smoke_model.py && python bootstrap/smoke_episode.py'])
            # Critic compatibility and RoboTwin->LeRobot adaptation are separate acceptance steps.
            state('base_validated', critic_validated=False, official_recap_data_adapter_ready=False)
        except Exception as exc:
            state('error', error_type=type(exc).__name__)
            raise


if __name__ == '__main__':
    main()
