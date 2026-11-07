"""Persistent first-stage workflow: freeze data, official critic, CFG pilots, held-out eval."""
from research_artifacts import artifact_path, relocate_html

import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".runtime/experiments/e4-offline-rl"
STATUS = artifact_path('offline-rl-pipeline-status.json')


def status(phase, **extra):
    data = {"phase": phase, "updated_unix": time.time(), "pid": os.getpid(), **extra}
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(STATUS)
    try:
        subprocess.run([str(ROOT / ".runtime/venv-tools/bin/python"), str(ROOT / "experiments/render_offline_progress.py")], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
    except Exception:
        pass


def run_script(script, arguments, logfile):
    environment = 'source bootstrap/env.sh; export OMP_NUM_THREADS=4; export PYTHONPATH="$PROJECT_ROOT/.runtime/reference-transformers:$PROJECT_ROOT/experiments:$PYTHONPATH"; exec python -u "$@"'
    if script == "recap_value_hdf5.py":
        environment = 'source bootstrap/env.sh; export OMP_NUM_THREADS=4; export PYTHONPATH="$PROJECT_ROOT/experiments:$REPO_PATH:$ROBOTWIN_PATH"; exec "$PROJECT_ROOT/.runtime/venv-pace/bin/python" -u "$@"'
    command = ["bash", "-c", environment, "--", "experiments/" + script, *arguments]
    with logfile.open("a") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print("started", script, arguments, "pid", process.pid, flush=True)
        try:
            return process.wait(timeout=12 * 3600)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


def train_eval(job):
    out = RUN / "jobs" / job["name"]
    out.mkdir(parents=True, exist_ok=True)
    rc = None
    for micro in [4, 2, 1]:
        log = out / f"train-micro{micro}.log"
        rc = run_script("train_recap_pilot.py", ["--tasks", *job["tasks"], "--job", job["name"],
                         "--mode", job["mode"], "--steps-per-task", "50", "--micro-batch", str(micro)], log)
        if rc == 0:
            break
        text = log.read_text(errors="replace")[-10000:]
        if "out of memory" not in text.lower():
            return dict(job, status="training_error", exit_code=rc, log=str(log.relative_to(ROOT)))
    if rc:
        return dict(job, status="training_error", exit_code=rc)
    for task in job["tasks"]:
        log = out / f"eval-{task}.log"
        rc = run_script("eval_recap_pilot.py", ["--task", task, "--regime", "clean", "--episodes", "20",
                        "--run", str(out / "evaluation"), "--seed-protocol", "fixed_eval", "--backend", "reference",
                        "--checkpoint", str(out / "actor-trainable.pt")], log)
        if rc:
            return dict(job, status="evaluation_error", exit_code=rc, log=str(log.relative_to(ROOT)))
    results = []
    for task in job["tasks"]:
        records = [json.loads(p.read_text()) for p in (out / "evaluation/clean" / task).glob("episode_*.json")]
        if len(records) != 20:
            return dict(job, status="evaluation_incomplete", task=task, count=len(records))
        source = ROOT / ".runtime/experiments/e1-reference-clean/clean" / task
        base = [json.loads((source / f"episode_{r['seed']}.json").read_text()) for r in records]
        results.append({"task": task, "episodes": 20, "successes": sum(r["success"] for r in records),
                        "baseline_successes_same_seeds": sum(r["success"] for r in base),
                        "new_wins": sum(r["success"] and not b["success"] for r, b in zip(records, base)),
                        "new_losses": sum(not r["success"] and b["success"] for r, b in zip(records, base))})
    result = dict(job, status="complete", results=results)
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    (RUN / "workflow-controller.json").write_text(json.dumps({"pid": os.getpid(), "started_unix": time.time()}))
    deadline = time.time() + 48 * 3600
    while True:
        data_ready = (RUN / "COLLECTION_FINISHED").exists() and (RUN / "DEMOS_READY").exists()
        models_ready = all((ROOT / "models" / name / "DOWNLOAD_COMPLETE").exists()
                           for name in ["gemma-3-270m", "siglip2-so400m-patch14-224"])
        if data_ready and models_ready:
            break
        status("waiting for fixed pool and official backbones", data_ready=data_ready, models_ready=models_ready,
               policy_training_started=False)
        if time.time() > deadline:
            raise TimeoutError("Preparation exceeded 48 hours; inspect download and collection logs")
        time.sleep(30)
    if run_script("freeze_offline_pool.py", [], RUN / "pool-audit.log"):
        raise RuntimeError("Offline-pool audit failed")
    lock = (ROOT / ".runtime/experiments/gpu-evaluation.lock").open("a")
    status("waiting for GPU lock", policy_training_started=False)
    fcntl.flock(lock, fcntl.LOCK_EX)
    status("official Gemma3/SigLIP2 critic smoke test", policy_training_started=False)
    smoke = json.loads((RUN / "value-smoke.json").read_text()) if (RUN / "value-smoke.json").exists() else {}
    if not smoke.get("projection_gradient_nonzero"):
        if run_script("recap_value_hdf5.py", ["--smoke"], RUN / "value-smoke.log"):
            raise RuntimeError("Official value-model smoke test failed")
    status("training official value model and labeling advantages", policy_training_started=False)
    if not (RUN / "VALUE_AND_ADVANTAGES_READY").exists():
        if run_script("recap_value_hdf5.py", ["--steps", "1000"], RUN / "value-workflow.log"):
            raise RuntimeError("Value learning or advantage labeling failed")
    plan = json.loads((RUN / "plan.json").read_text())
    jobs = [{"name": "st-" + task, "tasks": [task], "mode": "recap"} for task in plan["anchors"]]
    jobs = [jobs[0], {"name": "mt5-recap", "tasks": plan["anchors"], "mode": "recap"}, *jobs[1:],
            {"name": "mt5-success-replay", "tasks": plan["anchors"], "mode": "success_replay"}]
    (RUN / "pilot-jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(train_eval, job): job for job in jobs}
        while pending:
            done, _ = concurrent.futures.wait(pending, timeout=30, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(dict(job, status="error", error=str(exc)))
                (RUN / "pilot-results.json").write_text(json.dumps(results, indent=2) + "\n")
            status("pilot policy training and held-out evaluation", policy_training_started=True,
                   completed_jobs=len(results), total_jobs=len(jobs), results=results)
    phase = "complete" if all(r["status"] == "complete" for r in results) else "pilot has incomplete jobs"
    status(phase, policy_training_started=True, results=results)
    if phase == "complete":
        (RUN / "PILOT_FINISHED").write_text(str(time.time()))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        status("error", error=str(exc))
        raise
