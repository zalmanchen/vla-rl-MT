"""Evaluate the completed replay control while RL backbone downloads continue."""
from research_artifacts import artifact_path, relocate_html

import concurrent.futures
import fcntl
import json
from pathlib import Path
import time

from launch_recap_workflow import ROOT, RUN, run_script


def main():
    job = RUN / "jobs/mt5-success-replay"
    if not (job / "FINISHED").exists():
        raise RuntimeError("Replay training has not finished")
    plan = json.loads((RUN / "plan.json").read_text())
    lock = (ROOT / ".runtime/experiments/gpu-evaluation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    progress = artifact_path('replay-evaluation-status.json')

    def evaluate(task):
        rc = run_script("eval_recap_pilot.py", ["--task", task, "--regime", "clean", "--episodes", "20",
                        "--run", str(job / "evaluation"), "--seed-protocol", "fixed_eval", "--backend", "reference",
                        "--checkpoint", str(job / "actor-trainable.pt")], job / f"eval-{task}.log")
        return {"task": task, "exit_code": rc}

    jobs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(evaluate, task): task for task in plan["anchors"]}
        while pending:
            done, _ = concurrent.futures.wait(pending, timeout=20, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                jobs.append(future.result())
                pending.pop(future)
            counts = {task: len(list((job / "evaluation/clean" / task).glob("episode_*.json"))) for task in plan["anchors"]}
            tmp = progress.with_suffix(".tmp")
            tmp.write_text(json.dumps({"phase": "evaluating" if pending else "finished", "updated_unix": time.time(),
                                       "episode_counts": counts, "jobs": jobs}, indent=2))
            tmp.replace(progress)
    if any(item["exit_code"] for item in jobs):
        raise RuntimeError("Replay evaluation has incomplete tasks")
    results = []
    for task in plan["anchors"]:
        records = [json.loads(p.read_text()) for p in (job / "evaluation/clean" / task).glob("episode_*.json")]
        if len(records) != 20:
            raise RuntimeError("Incomplete evaluation")
        source = ROOT / ".runtime/experiments/e1-reference-clean/clean" / task
        baseline = [json.loads((source / f"episode_{r['seed']}.json").read_text()) for r in records]
        results.append({"task": task, "episodes": 20, "successes": sum(r["success"] for r in records),
                        "baseline_successes_same_seeds": sum(r["success"] for r in baseline),
                        "new_wins": sum(r["success"] and not b["success"] for r, b in zip(records, baseline)),
                        "new_losses": sum(not r["success"] and b["success"] for r, b in zip(records, baseline))})
    (job / "result.json").write_text(json.dumps({"name": "mt5-success-replay", "mode": "success_replay",
                                                "status": "complete", "results": results}, indent=2))


if __name__ == "__main__":
    main()
