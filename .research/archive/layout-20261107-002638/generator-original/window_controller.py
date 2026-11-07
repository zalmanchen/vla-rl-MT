"""Small paired execution-window diagnostic; the frozen E1 results remain untouched."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".runtime/experiments/e3-motus-window-diagnostic"
TASKS = ["adjust_bottle", "click_bell", "stack_blocks_two"]


def collect(phase):
    seeds = json.loads((ROOT / ".runtime/experiments/e1-reference-clean/seed_manifest.json").read_text())
    rows = []
    for task in TASKS:
        valid = seeds["tasks"][task]["valid_seeds"][:20]
        base = ROOT / ".runtime/experiments/e1-reference-clean/clean" / task
        original = [json.loads((base / f"episode_{seed}.json").read_text()) for seed in valid]
        original_by_seed = {r["seed"]: r for r in original}
        for horizon in [8, 16, 32]:
            records = original if horizon == 32 else [json.loads(p.read_text()) for p in (RUN / f"h{horizon}" / "clean" / task).glob("episode_*.json")]
            if not {r["seed"] for r in records}.issubset(valid):
                raise ValueError("Unexpected diagnostic seed")
            rows.append({"task": task, "execution_horizon": horizon, "completed": len(records),
                         "target": 20, "successes": sum(r["success"] for r in records),
                         "success_rate": sum(r["success"] for r in records) / len(records) if records else None,
                         "new_wins_vs_h32": sum(r["success"] and not original_by_seed[r["seed"]]["success"] for r in records),
                         "new_losses_vs_h32": sum(not r["success"] and original_by_seed[r["seed"]]["success"] for r in records)})
    data = {"phase": phase, "updated_unix": time.time(), "episodes_per_cell": 20,
            "scope": "diagnostic only: first 20 frozen E1 test scenes; do not use these to tune final test parameters",
            "h32_source": "original E1 results, reused without rerunning", "tasks": rows}
    output = ROOT / ".research/motus-window-diagnostic.json"
    tmp = output.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(output)


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "controller.json").write_text(json.dumps({"pid": os.getpid(), "started_unix": time.time()}))
    (RUN / "contract.json").write_text(json.dumps({"tasks": TASKS, "horizons": [8, 16, 32], "episodes_per_cell": 20,
        "weights": "motus-robotics/pi0.5_robotwin2@effc6e1a99bd4d0edf9447e8ba313def67649cfe",
        "backend": "same FP32 reference inference as E1", "denoising_steps": 10, "prediction_horizon": 32,
        "source_manifest_sha256": hashlib.sha256((ROOT / ".runtime/experiments/e1-reference-clean/seed_manifest.json").read_bytes()).hexdigest(),
        "only_changed_variable": "number of actions executed before acquiring a new observation",
        "h32": "cached original E1 results", "new_episodes": 120, "training": False}, indent=2))
    lock = (ROOT / ".runtime/experiments/gpu-evaluation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    for horizon in [8, 16]:
        for task in TASKS:
            phase = f"evaluating {task}/h{horizon}"
            collect(phase)
            command = ["bash", "-c", 'source bootstrap/env.sh; export PYTHONPATH="$PROJECT_ROOT/.runtime/reference-transformers:$PROJECT_ROOT/experiments:$PYTHONPATH"; exec python -u experiments/window_eval_worker.py "$@"', "--",
                       "--task", task, "--regime", "clean", "--episodes", "20", "--run", str(RUN / f"h{horizon}"),
                       "--backend", "reference", "--seed-protocol", "fixed_eval", "--execution-horizon", str(horizon)]
            with (RUN / f"{task}-h{horizon}.log").open("a") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                print(phase, process.pid, flush=True)
                launched = time.time()
                heartbeat = RUN / f"h{horizon}" / "clean" / task / "heartbeat.json"
                while process.poll() is None:
                    collect(phase)
                    last_progress = max(launched, heartbeat.stat().st_mtime if heartbeat.exists() else launched)
                    if time.time() - last_progress > 1200:
                        process.terminate()
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        raise TimeoutError(phase)
                    time.sleep(20)
                if process.returncode:
                    raise RuntimeError(f"Worker failed: {phase}")
                n = len(list((RUN / f"h{horizon}" / "clean" / task).glob("episode_*.json")))
                if n != 20:
                    raise RuntimeError(f"Incomplete diagnostic: {phase}, n={n}")
    collect("complete")
    (RUN / "FINISHED").write_text(str(time.time()))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        collect("error: " + str(exc))
        (RUN / "ERROR.json").write_text(json.dumps({"error": str(exc)}))
        raise
