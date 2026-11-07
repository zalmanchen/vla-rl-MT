"""Start the published-checkpoint Clean baseline without any training."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ["PROJECT_ROOT"])
run = ROOT / ".runtime/experiments/e1-reference-clean"
run.mkdir(parents=True, exist_ok=True)
seeds = json.loads(
    (
        ROOT / "third_party/RLinf/rlinf/envs/sim/robotwin/seeds/eval_seeds.json"
    ).read_text()
)
tasks = sorted(seeds)
assert len(tasks) == 22
receipt = run / "controller.json"
if receipt.exists():
    previous = json.loads(receipt.read_text())
    try:
        os.kill(previous["pid"], 0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError(f"Controller still running: {previous['pid']}")
cmd = [
    sys.executable,
    "-u",
    str(ROOT / "experiments/supervisor.py"),
    "--backend",
    "reference",
    "--seed-protocol",
    "fixed_eval",
    "--regimes",
    "clean",
    "--workers",
    "2",
    "--episodes",
    "100",
    "--run",
    str(run),
    "--tasks",
    *tasks,
]
with (run / "controller.log").open("a") as log:
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd=ROOT,
        env=os.environ.copy(),
    )
receipt.write_text(
    json.dumps(
        {
            "pid": process.pid,
            "started_unix": time.time(),
            "command": cmd,
            "training": False,
            "checkpoint_updated": False,
        },
        indent=2,
    )
    + "\n"
)
print(
    json.dumps(
        {"pid": process.pid, "run": str(run), "tasks": len(tasks), "episodes": 2200}
    ),
    flush=True,
)
