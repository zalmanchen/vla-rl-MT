"""Summarize the frozen-checkpoint evaluation without treating errors as failures."""
from research_artifacts import artifact_path, relocate_html

import csv
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".runtime/experiments/e1-reference-clean"
OUT = ROOT / ".research"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def wilson(successes, n):
    if not n:
        return None, None
    z = 1.959963984540054
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0, center - half), min(1, center + half)


def main():
    contract = json.loads((RUN / "contract.json").read_text())
    controller = json.loads((RUN / "controller.json").read_text())
    jobs = [json.loads(x) for x in (RUN / "jobs.jsonl").read_text().splitlines() if x]
    jobmap = {x["task"]: x for x in jobs}
    seeds = json.loads(
        (
            ROOT / "third_party/RLinf/rlinf/envs/sim/robotwin/seeds/eval_seeds.json"
        ).read_text()
    )
    rows = []
    all_records = []
    audits = []
    for task in contract["tasks"]:
        d = RUN / "clean" / task
        records = [json.loads(p.read_text()) for p in d.glob("episode_*.json")]
        records = [r for r in records if r["status"] == "completed"]
        unique = {r["seed"] for r in records}
        if len(unique) != len(records):
            raise ValueError(f"Duplicate seed: {task}")
        expected = seeds[task]["success_seeds"][:100]
        if not unique.issubset(set(seeds[task]["success_seeds"])):
            raise ValueError(f"Unexpected seed: {task}")
        for r in records:
            if not r["success"] and r["steps"] != r["limit"]:
                audits.append(
                    {
                        "task": task,
                        "seed": r["seed"],
                        "problem": "failure before episode limit",
                    }
                )
        events = []
        if (d / "events.jsonl").exists():
            events = [
                json.loads(x)
                for x in (d / "events.jsonl").read_text().splitlines()
                if x
            ]
        errors = [x for x in events if x.get("status") == "error"]
        n = len(records)
        k = sum(r["success"] for r in records)
        low, high = wilson(k, n)
        if n == 100:
            status = "complete"
            reason = ""
        elif any("ASSETS_PATH" in e["error"] for e in errors):
            status = "blocked"
            reason = "Missing ASSETS_PATH environment variable"
        elif any("unstable" in e["error"].lower() for e in errors):
            status = "partial"
            reason = "Unstable object during scene initialization"
        elif jobmap[task].get("error") == "worker heartbeat timeout":
            status = "partial"
            reason = "Watchdog timeout; root cause not established"
        else:
            status = "partial"
            reason = "See worker log"
        row = {
            "task": task,
            "completed": n,
            "target": 100,
            "successes": k,
            "success_rate": k / n if n else None,
            "wilson95_low": low,
            "wilson95_high": high,
            "status": status,
            "infrastructure_error": reason,
            "job_exit_code": jobmap[task]["exit_code"],
            "mean_seconds": statistics.mean(r["seconds"] for r in records)
            if records
            else None,
            "mean_steps": statistics.mean(r["steps"] for r in records)
            if records
            else None,
            "missing_seed_count": max(0, 100 - n),
            "replacement_seeds": sorted(unique - set(expected)),
            "original_seeds_not_completed": sorted(set(expected) - unique),
            "errors": [
                {"seed": e["seed"], "error": e["error"], "phase_in_log": e.get("phase")}
                for e in errors
            ],
        }
        rows.append(row)
        all_records.extend(records)
    complete = [r for r in rows if r["status"] == "complete"]
    end = float((RUN / "QUEUE_FINISHED").read_text())
    original_end = end
    resume_receipt = RUN / "resume-controller.json"
    resume_started = (
        json.loads(resume_receipt.read_text())["started_unix"]
        if resume_receipt.exists()
        else None
    )
    resume_done = RUN / "RESUME_FINISHED"
    resume_end = float(resume_done.read_text()) if resume_done.exists() else None
    if resume_end is not None:
        end = resume_end
    tz = ZoneInfo("Asia/Shanghai")
    summary = {
        "run": str(RUN.relative_to(ROOT)),
        "stage": "preliminary evaluation and released-checkpoint baseline; no training",
        "model": contract["model"],
        "model_revision": contract["model_revision"],
        "queue_finished": True,
        "evaluation_complete": len(complete) == len(rows),
        "planned_tasks": len(rows),
        "complete_tasks": len(complete),
        "partial_or_blocked_tasks": len(rows) - len(complete),
        "planned_episodes": 2200,
        "completed_episodes": len(all_records),
        "coverage": len(all_records) / 2200,
        "successes_all_observed": sum(r["success"] for r in all_records),
        "complete_subset_macro_success": statistics.mean(
            r["success_rate"] for r in complete
        ),
        "complete_subset_successes": sum(r["successes"] for r in complete),
        "complete_subset_episodes": sum(r["completed"] for r in complete),
        "all_observed_descriptive_success": sum(r["success"] for r in all_records)
        / len(all_records),
        "started_shanghai": datetime.fromtimestamp(
            controller["started_unix"], tz
        ).isoformat(),
        "finished_shanghai": datetime.fromtimestamp(end, tz).isoformat(),
        "wall_hours": (original_end - controller["started_unix"]) / 3600,
        "original_finished_shanghai": datetime.fromtimestamp(original_end, tz).isoformat(),
        "elapsed_hours_including_pause": (end - controller["started_unix"]) / 3600,
        "resume_wall_hours": (resume_end - resume_started) / 3600
        if resume_end is not None and resume_started is not None
        else None,
        "resume_started_unix": resume_started,
        "resume_finished_unix": resume_end,
        "resume_running": resume_started is not None and resume_end is None,
        "worker_matches_contract": hashlib.sha256(
            (ROOT / "experiments/eval_worker.py").read_bytes()
        ).hexdigest()
        == contract["script_sha256"],
        "audit_findings": audits,
        "contract": contract,
        "tasks": rows,
    }
    (artifact_path('e1-results.json')).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    with (artifact_path('e1-results.csv')).open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                k
                for k in rows[0]
                if k
                not in ["errors", "replacement_seeds", "original_seeds_not_completed"]
            ],
        )
        writer.writeheader()
        writer.writerows(
            {
                k: v
                for k, v in r.items()
                if k
                not in ["errors", "replacement_seeds", "original_seeds_not_completed"]
            }
            for r in rows
        )
    observed = sorted(
        (r for r in rows if r["completed"]), key=lambda r: r["success_rate"]
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, r in enumerate(observed):
        p = r["success_rate"] * 100
        color = "#2563eb" if r["status"] == "complete" else "#d97706"
        ax.barh(i, p, color=color, alpha=0.88)
        ax.errorbar(
            p,
            i,
            xerr=[
                [max(0, p - r["wilson95_low"] * 100)],
                [max(0, r["wilson95_high"] * 100 - p)],
            ],
            fmt="none",
            ecolor="#334155",
            capsize=3,
            linewidth=1,
        )
        ax.text(
            104,
            i,
            f"{r['successes']}/{r['completed']}"
            + (" *" if r["status"] != "complete" else ""),
            va="center",
            fontsize=9,
        )
    ax.set_yticks(range(len(observed)), [r["task"] for r in observed])
    ax.set_xlim(-1, 122)
    ax.set_xlabel("Success rate (%) — 95% Wilson intervals")
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_title(
        "Released multi-task checkpoint · RoboTwin Clean\nBlue: 100 episodes complete | Orange*: incomplete, descriptive only",
        loc="left",
        fontsize=12,
    )
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "e1-success-rates.svg", bbox_inches="tight")
    fig.savefig(FIG / "e1-success-rates.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    samples = [
        ("handover_block", False),
        ("lift_pot", False),
        ("adjust_bottle", True),
        ("stack_bowls_two", True),
    ]
    sheet = Image.new("RGB", (960, 4 * 270), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (task, success) in enumerate(samples):
        candidates = []
        for p in sorted((RUN / "clean" / task).glob("*.mp4")):
            record = json.loads(p.with_suffix(".json").read_text())
            if record["success"] == success:
                candidates.append((p, record))
        p, r = candidates[0]
        reader = imageio.get_reader(p)
        frames = []
        try:
            count = reader.count_frames()
            for t in [0, count // 2, count - 1]:
                frames.append(reader.get_data(t))
        finally:
            reader.close()
        for col, frame in enumerate(frames):
            sheet.paste(
                Image.fromarray(frame).resize((320, 240)), (col * 320, row * 270 + 30)
            )
        draw.text(
            (8, row * 270 + 8),
            f"{task} | seed={r['seed']} | success={success} | start / middle / end",
            fill="black",
        )
    sheet.save(FIG / "e1-rollout-contact-sheet.jpg", quality=92)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ["tasks", "contract"]},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
