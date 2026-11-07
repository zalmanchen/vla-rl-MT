"""Finish incomplete baseline tasks and continuously regenerate the research record."""

import concurrent.futures
import fcntl
import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

LOGGER = logging.getLogger(__name__)
ROOT = Path(os.environ["PROJECT_ROOT"])
RUN = ROOT / ".runtime/experiments/e1-reference-clean"


def records(task):
    return [
        json.loads(p.read_text()) for p in (RUN / "clean" / task).glob("episode_*.json")
    ]


def refresh():
    for name in ["analyze_results.py", "render_progress.py"]:
        result = subprocess.run(
            [sys.executable, str(ROOT / "experiments" / name)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            cwd=ROOT,
        )
        if result.returncode:
            print("report_refresh_error", name, result.stderr, flush=True)


def execute(task):
    no_progress = 0
    batch = 0
    while len(records(task)) < 100 and no_progress < 3:
        before = len(records(task))
        batch += 1
        log_path = RUN / "logs" / f"resume-{task}-{batch:03d}.log"
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            str(ROOT / ".runtime/reference-transformers")
            + os.pathsep
            + env.get("PYTHONPATH", "")
        )
        cmd = [
            sys.executable,
            "-u",
            str(ROOT / "experiments/resume_worker.py"),
            "--backend",
            "reference",
            "--seed-protocol",
            "fixed_eval",
            "--task",
            task,
            "--regime",
            "clean",
            "--episodes",
            "100",
            "--max-new-episodes",
            "20",
            "--run",
            str(RUN),
        ]
        timeout = False
        start = time.time()
        with log_path.open("a") as log:
            child = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
                cwd=ROOT,
            )
            while child.poll() is None:
                time.sleep(5)
                hb = RUN / "clean" / task / "heartbeat.json"
                last = max(start, hb.stat().st_mtime if hb.exists() else start)
                if time.time() - last > 240:
                    timeout = True
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    break
        after = len(records(task))
        no_progress = no_progress + 1 if after == before else 0
        result = {
            "task": task,
            "batch": batch,
            "exit_code": child.returncode,
            "watchdog_timeout": timeout,
            "before": before,
            "after": after,
            "seconds": time.time() - start,
            "log": str(log_path.relative_to(ROOT)),
        }
        with (RUN / "resume-jobs.jsonl").open("a", buffering=1) as output:
            output.write(json.dumps(result) + "\n")
        print(json.dumps(result), flush=True)
    return {
        "task": task,
        "completed": len(records(task)),
        "status": "complete"
        if len(records(task)) == 100
        else "blocked_after_three_no_progress_retries",
    }


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    lock = (RUN / "resume.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tasks = [
        "handover_mic",
        "move_can_pot",
        "shake_bottle",
        "put_bottles_dustbin",
        "place_a2b_right",
        "place_a2b_left",
    ]
    source = ROOT / "experiments/resume_worker.py"
    amendment = {
        "reason": "Finish six incomplete Clean baseline tasks; no training or policy changes",
        "original_contract_sha256": hashlib.sha256(
            (RUN / "contract.json").read_bytes()
        ).hexdigest(),
        "resume_worker_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "tasks": tasks,
        "max_new_episodes_per_process": 20,
        "setup_attempts_per_seed": 3,
        "seed_policy": "Retry identical scene seed three times; only UnStableError during pre-policy initialization may exclude seed. Use subsequent candidates from same published task list to reach 100 valid episodes. Never replace policy failures. Record final seed manifest.",
        "assets_path": str(ROOT / "third_party/RoboTwin"),
        "watchdog_seconds": 240,
        "no_progress_restart_limit": 3,
        "original_results_preserved": True,
    }
    (RUN / "resume-amendment.json").write_text(json.dumps(amendment, indent=2) + "\n")
    (RUN / "resume-controller.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "started_unix": time.time(),
                "tasks": tasks,
                "training": False,
            },
            indent=2,
        )
        + "\n"
    )
    refresh()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        pending = {
            pool.submit(execute, task) for task in tasks if len(records(task)) < 100
        }
        while pending:
            done, pending = concurrent.futures.wait(
                pending, timeout=60, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                try:
                    results.append(future.result())
                except Exception as exc:
                    LOGGER.exception("Resume job failed")
                    results.append({"status": "controller_error", "error": repr(exc)})
            refresh()
    (RUN / "resume-results.json").write_text(json.dumps(results, indent=2) + "\n")
    contract = json.loads((RUN / "contract.json").read_text())
    candidate = json.loads(
        (
            ROOT / "third_party/RLinf/rlinf/envs/sim/robotwin/seeds/eval_seeds.json"
        ).read_text()
    )
    manifest = {}
    for task in contract["tasks"]:
        used = {x["seed"] for x in records(task)}
        invalid = RUN / "clean" / task / "invalid_initializations.jsonl"
        exclusions = (
            [json.loads(l) for l in invalid.read_text().splitlines() if l]
            if invalid.exists()
            else []
        )
        manifest[task] = {
            "valid_seeds": [s for s in candidate[task]["success_seeds"] if s in used],
            "invalid_initializations": exclusions,
            "target": 100,
        }
    all_complete = all(len(v["valid_seeds"]) == 100 for v in manifest.values())
    (RUN / "seed_manifest.json").write_text(
        json.dumps({"frozen": all_complete, "tasks": manifest}, indent=2) + "\n"
    )
    (RUN / "RESUME_FINISHED").write_text(str(time.time()) + "\n")
    refresh()
    print(json.dumps({"all_complete": all_complete, "results": results}), flush=True)


if __name__ == "__main__":
    main()
