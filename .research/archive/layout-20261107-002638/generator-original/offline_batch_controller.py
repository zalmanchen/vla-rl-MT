"""Run the first fixed-pool offline-RL preparation batch with bounded GPU sharing."""

import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".runtime/experiments/e4-offline-rl"


def write_status(phase):
    plan = json.loads((RUN / "plan.json").read_text())
    tasks = []
    for task in plan["anchors"]:
        records = [json.loads(p.read_text()) for p in (RUN / "pool/clean" / task).glob("episode_*.json")]
        tasks.append({"task": task, "episodes": len(records), "target": 40,
                      "successes": sum(r["success"] for r in records),
                      "failures": sum(not r["success"] for r in records)})
    status = {"phase": phase, "updated_unix": time.time(), "algorithm": plan["algorithm"],
              "policy_training_started": False, "tasks": tasks}
    p = ROOT / ".research/offline-batch-status.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2) + "\n")
    tmp.replace(p)


def collect_once(task):
    command = ["bash", "-c", 'source bootstrap/env.sh; export OMP_NUM_THREADS=4; export PYTHONPATH="$PROJECT_ROOT/.runtime/reference-transformers:$PROJECT_ROOT/experiments:$PYTHONPATH"; exec python -u experiments/collect_offline_worker.py "$@"', "--",
               "--task", task, "--regime", "clean", "--episodes", "40", "--run", str(RUN / "pool"),
               "--backend", "reference", "--seed-protocol", "fixed_eval"]
    with (RUN / f"collect-{task}.log").open("a") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print("collecting", task, process.pid, flush=True)
        heartbeat = RUN / "pool/clean" / task / "heartbeat.json"
        started = time.time()
        while process.poll() is None:
            progress = max(started, heartbeat.stat().st_mtime if heartbeat.exists() else started)
            timeout = 600 if progress == started else 300
            if time.time() - progress > timeout:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                return {"task": task, "exit_code": process.returncode, "error": "heartbeat timeout"}
            time.sleep(10)
        return {"task": task, "exit_code": process.returncode}



def collect(task):
    failures = 0
    while True:
        directory = RUN / "pool/clean" / task
        before = len(list(directory.glob("episode_*.json")))
        if before >= 40:
            return {"task": task, "exit_code": 0, "episodes": before}
        result = collect_once(task)
        after = len(list(directory.glob("episode_*.json")))
        if after > before:
            failures = 0
        else:
            failures += 1
        if failures >= 3:
            return dict(result, error="three consecutive attempts without a completed episode", episodes=after)
        print("collector refreshed", task, "episodes", after, "return", result["exit_code"], flush=True)


def main():
    plan = json.loads((RUN / "plan.json").read_text())
    (RUN / "controller.json").write_text(json.dumps({"pid": os.getpid(), "started_unix": time.time(), "collectors": 3}))
    lock = (ROOT / ".runtime/experiments/gpu-evaluation.lock").open("a")
    write_status("waiting for GPU lock")
    fcntl.flock(lock, fcntl.LOCK_EX)
    jobs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(collect, task): task for task in plan["anchors"]}
        while pending:
            done, _ = concurrent.futures.wait(pending, timeout=20, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                result = future.result()
                jobs.append(result)
                pending.pop(future)
                (RUN / "collection-jobs.json").write_text(json.dumps(jobs, indent=2))
            write_status("collecting and freezing offline data")
    if any(job["exit_code"] for job in jobs):
        write_status("collection needs retry; inspect collection-jobs.json")
        raise RuntimeError("One or more offline collection workers failed")
    write_status("collection complete; preparing value and policy training")
    (RUN / "COLLECTION_FINISHED").write_text(str(time.time()))


if __name__ == "__main__":
    main()
