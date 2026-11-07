"""Run the frozen SFT evaluation protocol with bounded GPU concurrency."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ["PROJECT_ROOT"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--tasks", nargs="*")
    p.add_argument(
        "--regimes",
        nargs="+",
        choices=["clean", "randomized"],
        default=["clean", "randomized"],
    )
    p.add_argument(
        "--seed-protocol",
        choices=["expert_filter", "fixed_eval"],
        default="expert_filter",
    )
    p.add_argument("--backend", choices=["native", "reference"], default="native")
    args = p.parse_args()
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    parity = json.loads(
        (ROOT / ".runtime/experiments/e0-parity/report.json").read_text()
    )
    if args.backend == "native" and not parity["passed"]:
        raise RuntimeError("E0 sampler parity has not passed")
    tasks = args.tasks or sorted(
        p.parent.name
        for p in (ROOT / "datasets/RoboTwin2.0/dataset").glob("*/demo_clean.zip")
    )
    contract = {
        "model": "motus-robotics/pi0.5_robotwin2",
        "model_revision": "effc6e1a99bd4d0edf9447e8ba313def67649cfe",
        "tasks": tasks,
        "regimes": args.regimes,
        "episodes_per_task_regime": args.episodes,
        "workers": args.workers,
        "action_horizon": 32,
        "action_execution": 32,
        "denoising_steps": 10,
        "precision": "fp32",
        "backend": args.backend,
        "activation_alignment": "original OpenPI"
        if args.backend == "reference"
        else "SigLIP and Gemma tanh GELU adaptation",
        "seed_protocol_mode": args.seed_protocol,
        "denoiser": "none",
        "planner": "mplib",
        "language": "official seen templates; deterministic seed",
        "seed_protocol": "first N published RLinf eval seeds; no replacement"
        if args.seed_protocol == "fixed_eval"
        else "expert filtered candidate seeds",
        "parity": parity,
        "script_sha256": hashlib.sha256(
            (ROOT / "experiments/eval_worker.py").read_bytes()
        ).hexdigest(),
    }
    contract_path = run / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise RuntimeError("Existing run has a different protocol")
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    jobs = [(task, regime) for regime in contract["regimes"] for task in tasks]

    def execute(job):
        task, regime = job
        logs = run / "logs"
        logs.mkdir(exist_ok=True)
        cmd = [
            sys.executable,
            "-u",
            str(ROOT / "experiments/eval_worker.py"),
            "--task",
            task,
            "--regime",
            regime,
            "--episodes",
            str(args.episodes),
            "--run",
            str(run),
            "--seed-protocol",
            args.seed_protocol,
            "--backend",
            args.backend,
        ]
        with (logs / f"{regime}-{task}.log").open("a") as log:
            child_env = os.environ.copy()
            if args.backend == "reference":
                child_env["PYTHONPATH"] = (
                    str(ROOT / ".runtime/reference-transformers")
                    + os.pathsep
                    + child_env.get("PYTHONPATH", "")
                )
            child = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=child_env,
            )
            start = time.time()
            while child.poll() is None:
                time.sleep(5)
                hb = run / regime / task / "heartbeat.json"
                active = max(start, hb.stat().st_mtime if hb.exists() else start)
                if time.time() - active > 1800:
                    import signal

                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=30)
                    return {
                        "task": task,
                        "regime": regime,
                        "exit_code": child.returncode,
                        "error": "worker heartbeat timeout",
                    }
            return {
                "task": task,
                "regime": regime,
                "exit_code": child.returncode,
                "seconds": time.time() - start,
            }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute, job) for job in jobs]
        with (run / "jobs.jsonl").open("a", buffering=1) as output:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                output.write(json.dumps(result) + "\n")
                print(json.dumps(result), flush=True)
    (run / "QUEUE_FINISHED").write_text(str(time.time()) + "\n")


if __name__ == "__main__":
    main()
