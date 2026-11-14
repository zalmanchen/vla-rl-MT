"""Own a bounded two-GPU collection queue: real smoke, full pool, frozen index.

No policy optimization is started by this data controller.
"""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def save(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def execute(pool, output, gpus, smoke_only=False):
    plan = json.loads((pool/"plan.json").read_text())
    tasks = plan["config"]["tasks"]
    if len(gpus) != len(tasks) or len(set(gpus)) != len(gpus):
        raise ValueError("One distinct GPU per collection task is required")
    output.mkdir(parents=True, exist_ok=False)
    state = {"pid": os.getpid(), "phase": "starting", "pool": str(pool), "jobs": [],
             "policy_training_started": False, "created_unix": time.time()}
    active = []
    with ExitStack() as stack:
        locks = ROOT/".runtime/locks"
        locks.mkdir(parents=True, exist_ok=True)
        for gpu in sorted(gpus):
            lock = stack.enter_context((locks/f"sft-recap-gpu-{gpu}.lock").open("a+"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for phase in (["smoke"] if smoke_only else ["smoke", "full"]):
                state.update(phase=phase, updated_unix=time.time())
                active = []
                for task, gpu in zip(tasks, gpus):
                    command = [sys.executable, str(ROOT/"bootstrap/prepare_sft_recap_data.py"), "collect",
                               "--output", str(pool), "--task", task]
                    if phase == "smoke":
                        command.append("--smoke")
                    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "VLA_COLLECTION_GPU": str(gpu),
                           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "JAX_PLATFORMS": "cpu",
                           "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
                    log_path = output/f"{task}-{phase}.log"
                    stream = stack.enter_context(log_path.open("x"))
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                    row = {"task": task, "gpu": gpu, "phase": phase, "pid": child.pid,
                           "status": "running", "log": str(log_path), "command": command}
                    state["jobs"].append(row)
                    active.append((child, row))
                save(output/"status.json", state)
                while any(child.poll() is None for child, _ in active):
                    for child, row in active:
                        if child.poll() is not None:
                            row.update(status="complete" if child.returncode == 0 else "failed", exit_code=child.returncode)
                    if any(row["status"] == "failed" for _, row in active):
                        raise RuntimeError(f"{phase} collector failed; stop the owned queue and inspect its log")
                    save(output/"status.json", state)
                    time.sleep(2)
                for child, row in active:
                    row.update(status="complete" if child.returncode == 0 else "failed", exit_code=child.returncode)
                    if child.returncode:
                        raise RuntimeError(f"{phase} collector failed: {row['task']}")
                save(output/"status.json", state)
            if not smoke_only:
                state["phase"] = "freezing_raw_pool"
                save(output/"status.json", state)
                with (output/"freeze.log").open("x") as stream:
                    subprocess.run([sys.executable, str(ROOT/"bootstrap/prepare_sft_recap_data.py"), "freeze",
                                    "--output", str(pool)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
            state.update(phase="smoke_complete" if smoke_only else "raw_pool_complete", finished_unix=time.time(),
                         next_gate="LeRobot export and official-loader audit before SFT; no frame interpolation")
        except BaseException as error:
            state.update(phase="failed", error=repr(error), updated_unix=time.time())
            raise
        finally:
            for child, row in active:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    row.update(status="stopped_owned_worker", exit_code=child.returncode)
            save(output/"status.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    execute(args.pool.absolute(), args.output.absolute(), args.gpus, args.smoke_only)


if __name__ == "__main__":
    main()
