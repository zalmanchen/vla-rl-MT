"""Wait for verified PACE checkpoints, then run frozen-scene paired evaluations."""

import hashlib
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".runtime/experiments/e2-pace-single-task"
TASKS = ["adjust_bottle", "click_bell", "stack_blocks_two", "handover_block"]


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def summary(phase):
    base = json.loads((ROOT / ".research/e1-results.json").read_text())
    rows = []
    for task in TASKS:
        records = [json.loads(p.read_text()) for p in (RUN / "clean" / task).glob("episode_*.json")]
        old = next(r for r in base["tasks"] if r["task"] == task)
        rows.append({"task": task, "completed": len(records), "target": 100,
                     "successes": sum(r["success"] for r in records),
                     "single_task_success": sum(r["success"] for r in records) / len(records) if records else None,
                     "multi_task_success": old["success_rate"]})
    data = {"phase": phase, "updated_unix": time.time(), "run": str(RUN.relative_to(ROOT)),
            "model": "niejunnan25/pi05-robotwin", "prediction_horizon": 50, "execution_horizon": 32,
            "comparison": "released single-task vs released multi-task SFT checkpoints; not a controlled training ablation",
            "tasks": rows}
    atomic(ROOT / ".research/pace-comparison.json", data)
    return data


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    started = time.time()
    lock = json.loads((ROOT / "bootstrap/pace-downloads.lock.json").read_text())
    contract = {"tasks": TASKS, "episodes_per_task": 100, "execution_horizon": 32,
                "prediction_horizon": 50, "denoising_steps": 10, "regime": "clean",
                "model_revision": lock["revision"], "openpi_revision": "62e44c30c49df06897168cc5081ecd71afd088b2",
                "source_seed_manifest": ".runtime/experiments/e1-reference-clean/seed_manifest.json",
                "source_seed_manifest_sha256": sha256(ROOT / ".runtime/experiments/e1-reference-clean/seed_manifest.json"),
                "backend": "original OpenPI JAX; checkpoint-specific transforms and normalization",
                "rendering": "same RoboTwin snapshot and OIDN-off renderer as E1",
                "training": "none", "started_unix": started}
    atomic(RUN / "contract.json", contract)
    atomic(RUN / "controller.json", {"pid": os.getpid(), "started_unix": started})
    os.environ["PROJECT_ROOT"] = str(ROOT)
    for task in TASKS:
        target = ROOT / "models/pi05-robotwin-pace"
        files = [f for f in lock["files"] if f["path"].startswith(task + "/") or f["path"] == "tokenizer.model"]
        print("waiting for checkpoint", task, flush=True)
        while not all((target / f["path"]).exists() and (target / f["path"]).stat().st_size == f["bytes"] for f in files):
            summary("downloading: " + task)
            if time.time() - started > 48 * 3600:
                raise TimeoutError("Checkpoint download exceeded 48 hours")
            time.sleep(30)
        for f in files:
            if sha256(target / f["path"]) != f["sha256"]:
                raise ValueError("Checkpoint checksum mismatch: " + f["path"])
        atomic(RUN / (task + "-weights-verified.json"), {"revision": lock["revision"], "files": len(files)})
        # Preserve the existing tokenizer cache; verify it is the same upstream tokenizer.
        tokenizer = ROOT / ".runtime/cache/openpi/big_vision/paligemma_tokenizer.model"
        if not tokenizer.exists() or sha256(tokenizer) != next(f["sha256"] for f in files if f["path"] == "tokenizer.model"):
            raise ValueError("OpenPI tokenizer cache does not match the released tokenizer")
        gpu_lock = (ROOT / ".runtime/experiments/gpu-evaluation.lock").open("a")
        print("waiting for GPU evaluation lock", task, flush=True)
        fcntl.flock(gpu_lock, fcntl.LOCK_EX)
        for episodes, stage in [(1, "pilot"), (100, "full")]:
            logpath = RUN / f"{task}-{stage}.log"
            cmd = ["bash", "-c", 'source bootstrap/env.sh; export XLA_PYTHON_CLIENT_PREALLOCATE=false; export XLA_PYTHON_CLIENT_MEM_FRACTION=0.65; exec .runtime/venv-pace/bin/python -u experiments/pace_eval_worker.py "$@"', "--",
                   "--task", task, "--regime", "clean", "--episodes", str(episodes), "--run", str(RUN), "--seed-protocol", "fixed_eval", "--execution-horizon", "32"]
            with logpath.open("a") as log:
                proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                print("started", task, stage, proc.pid, flush=True)
                heartbeat = RUN / "clean" / task / "heartbeat.json"
                launch = time.time()
                last_progress = launch
                while proc.poll() is None:
                    if heartbeat.exists() and heartbeat.stat().st_mtime > launch:
                        last_progress = max(last_progress, heartbeat.stat().st_mtime)
                    summary("evaluating: " + task + "/" + stage)
                    # First load/JIT compilation can be slow; later stalls remain bounded.
                    timeout = 1800 if last_progress == launch else 300
                    if time.time() - last_progress > timeout:
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                        raise TimeoutError(f"Worker stalled: {task}/{stage}")
                    time.sleep(20)
                if proc.returncode:
                    raise RuntimeError(f"Worker failed: {task}/{stage}; see {logpath}")
            rows = summary("evaluating: " + task)["tasks"]
            row = next(r for r in rows if r["task"] == task)
            if row["completed"] < episodes:
                raise RuntimeError(f"Incomplete worker output: {task}/{stage}")
        fcntl.flock(gpu_lock, fcntl.LOCK_UN)
        gpu_lock.close()
    summary("complete")
    (RUN / "FINISHED").write_text(str(time.time()) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        atomic(RUN / "ERROR.json", {"error": str(exc), "unix_time": time.time()})
        summary("error: " + str(exc))
        raise
