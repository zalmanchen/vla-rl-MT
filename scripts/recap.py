"""Run and inspect the official four-stage RoboTwin RECAP pipeline."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".runtime/venv-rlinf/bin/python"
STATE = ROOT / ".runtime/bootstrap/recap-run.json"


def run_pipeline(args):
    run_name = args.name or time.strftime("recap-%Y%m%d-%H%M%S", time.gmtime())
    run = ROOT / ".runtime/runs" / run_name
    run.mkdir(parents=True, exist_ok=False)
    tag = ("smoke-" if args.command == "smoke" else "pilot-") + hashlib.sha256(
        run_name.encode()
    ).hexdigest()[:8]
    value_steps = args.value_steps or (2 if args.command == "smoke" else 500)
    policy_steps = args.policy_steps or (2 if args.command == "smoke" else 500)
    env = os.environ.copy()
    env.update(
        PROJECT_ROOT=str(ROOT),
        REPO_PATH=str(ROOT / "third_party/RLinf"),
        RECAP_GPU=str(args.gpu),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    paths = [
        str(ROOT / "src"),
        str(ROOT / "third_party/RLinf"),
        str(ROOT / "third_party/RoboTwin"),
    ]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    record = {
        "status": "running",
        "pid": os.getpid(),
        "run_dir": str(run),
        "value_steps": value_steps,
        "policy_steps": policy_steps,
        "gpu": args.gpu,
        "tag": tag,
        "completed_stages": [],
        "started_unix": time.time(),
        "rlinf_revision": "e1a82ebf0c1af69da6ab9b1519f5f3692fe33566",
        "scope": "RLinf official algorithm with recorded RoboTwin input compatibility patch; pilot, not paper reproduction",
    }

    def save():
        record["updated_unix"] = time.time()
        value = json.dumps(record, indent=2) + "\n"
        (run / "status.json").write_text(value)
        temporary = STATE.with_suffix(".tmp")
        temporary.write_text(value)
        temporary.replace(STATE)

    def stage(name, config, overrides):
        tensorboard = run / "tensorboard"
        previous_events = set(tensorboard.glob("events.*"))
        record["stage"] = name
        log = run / f"{name}.log"
        record["current_log"] = str(log)
        command = [
            str(PYTHON),
            str(ROOT / "scripts/run.py"),
            f"recap-{name}",
            "--config",
            str(ROOT / "configs/robotwin_pilot" / config),
            "--",
            *overrides,
        ]
        (run / f"{name}-command.json").write_text(json.dumps(command, indent=2) + "\n")
        save()
        print("STAGE", name, "LOG", log, flush=True)
        stage_env = env.copy()
        if name == "advantages":
            stage_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        with log.open("w") as stream:
            subprocess.run(
                command,
                cwd=ROOT,
                env=stage_env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        if name in {"value", "policy"} and tensorboard.is_dir():
            destination = run / name / "tensorboard"
            destination.mkdir(parents=True, exist_ok=True)
            for path in tensorboard.iterdir():
                if path.name == "config.yaml" or (
                    path.name.startswith("events.") and path not in previous_events
                ):
                    shutil.copy2(path, destination / path.name)
        record["completed_stages"].append(name)
        save()

    try:
        for split in ["sft-train", "sft-val", "rollout-train", "rollout-val"]:
            p = (
                ROOT
                / "datasets/lerobot/recap-pilot-v2"
                / split
                / "meta/conversion.json"
            )
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing converted dataset {p}; prepare the frozen pool first"
                )
        subprocess.run(
            [sys.executable, str(ROOT / "bootstrap/apply_recap_patch.py")], check=True
        )
        subprocess.run(
            ["git", "diff", "--binary", "--output", str(run / "project.diff")],
            cwd=ROOT,
            check=True,
        )
        sources = {}
        for directory in ["bootstrap", "configs", "scripts", "src", "tests"]:
            for path in sorted((ROOT / directory).rglob("*")):
                if path.is_file() and path.suffix in [
                    ".py",
                    ".yaml",
                    ".json",
                    ".patch",
                    ".txt",
                    ".sh",
                ]:
                    sources[str(path.relative_to(ROOT))] = hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
        (run / "source-files.json").write_text(json.dumps(sources, indent=2) + "\n")
        with zipfile.ZipFile(run / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for name in sources:
                archive.write(ROOT / name, name)
        common = [f"runner.logger.log_path={run}"]
        stage("returns", "returns.yaml", [f"data.tag={tag}"])
        stage(
            "value",
            "value.yaml",
            [
                *common,
                "runner.logger.experiment_name=value",
                f"runner.max_steps={value_steps}",
                f"runner.save_interval={min(100, value_steps)}",
                "runner.val_check_interval=-1"
                if args.command == "smoke"
                else "runner.val_check_interval=100",
                f"data.tag={tag}",
                "actor.optim.lr_warmup_steps=0"
                if args.command == "smoke"
                else "actor.optim.lr_warmup_steps=20",
            ],
        )
        checkpoint = run / f"value/checkpoints/global_step_{value_steps}/actor"
        if not (checkpoint / "model_state_dict/full_weights.pt").is_file():
            raise FileNotFoundError(f"Value checkpoint missing: {checkpoint}")
        record["value_checkpoint"] = str(checkpoint)
        stage(
            "advantages",
            "advantages.yaml",
            [
                f"advantage.value_checkpoint={checkpoint}",
                f"advantage.tag={tag}",
                f"advantage.returns_tag={tag}",
            ],
        )
        for split in ["sft-train", "sft-val", "rollout-train", "rollout-val"]:
            meta = ROOT / "datasets/lerobot/recap-pilot-v2" / split / "meta"
            destination = run / "dataset-sidecars" / split
            destination.mkdir(parents=True)
            for name in [
                f"returns_{tag}.parquet",
                f"advantages_{tag}.parquet",
                "conversion.json",
                "mixture_config.yaml",
                "info.json",
                "stats.json",
                "tasks.jsonl",
            ]:
                if (meta / name).is_file():
                    shutil.copy2(meta / name, destination / name)
        stage(
            "policy",
            "policy.yaml",
            [
                *common,
                "runner.logger.experiment_name=policy",
                f"runner.max_steps={policy_steps}",
                f"runner.save_interval={min(100, policy_steps)}",
                f"data.advantage_tag={tag}",
                "actor.optim.lr_warmup_steps=0"
                if args.command == "smoke"
                else "actor.optim.lr_warmup_steps=20",
            ],
        )
        checkpoint = run / f"policy/checkpoints/global_step_{policy_steps}/actor"
        if not (checkpoint / "model_state_dict/full_weights.pt").is_file():
            raise FileNotFoundError(f"Policy checkpoint missing: {checkpoint}")
        record.update(
            status="completed",
            stage="four_stages_passed",
            policy_checkpoint=str(checkpoint),
            policy_success_rate_validated=False,
        )
        save()
        if args.eval_episodes:
            record.update(
                status="running",
                stage="evaluation",
                current_log=str(run / "evaluation.log"),
            )
            save()
            eval_env = env.copy()
            eval_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            with (run / "evaluation.log").open("w") as stream:
                subprocess.run(
                    [
                        str(PYTHON),
                        str(ROOT / "bootstrap/evaluate_recap.py"),
                        "--checkpoint",
                        str(checkpoint),
                        "--episodes",
                        str(args.eval_episodes),
                        "--output",
                        str(run / "evaluation"),
                    ],
                    cwd=ROOT,
                    env=eval_env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            evaluation = json.loads((run / "evaluation/results.json").read_text())
            record.update(
                status="completed",
                stage="four_stages_and_development_eval_passed",
                development_evaluation=evaluation["summary"],
            )
            save()
        print("FOUR_STAGES_PASSED", run, flush=True)
    except Exception as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        save()
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["smoke", "train", "status"])
    p.add_argument("--value-steps", type=int)
    p.add_argument("--policy-steps", type=int)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--name")
    p.add_argument(
        "--eval-episodes",
        type=int,
        default=0,
        help="Paired development episodes per task and condition after training (0 skips)",
    )
    args = p.parse_args()
    if args.command == "status":
        if not STATE.exists():
            print("No four-stage run recorded")
            return
        state = json.loads(STATE.read_text())
        command_line = Path(f"/proc/{state['pid']}/cmdline")
        state["controller_alive"] = (
            command_line.exists() and b"scripts/recap.py" in command_line.read_bytes()
        )
        end = (
            state["updated_unix"]
            if state["status"] in ["completed", "failed"]
            else time.time()
        )
        state["elapsed_minutes"] = round((end - state["started_unix"]) / 60, 1)
        log = Path(state.get("current_log", ""))
        if log.is_file():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 65536))
                tail = stream.read().decode(errors="replace")
            matches = re.findall(
                r"Global Step:\s*\d+%\|[^\n\r]*?\|\s*(\d+)/(\d+)", tail
            )
            if matches:
                state["stage_progress"] = {
                    "completed": int(matches[-1][0]),
                    "total": int(matches[-1][1]),
                }
        print(json.dumps(state, indent=2))
        return
    if any(x is not None and x <= 0 for x in [args.value_steps, args.policy_steps]):
        p.error("Training steps must be positive")
    if args.gpu < 0 or not 0 <= args.eval_episodes <= 100:
        p.error("--gpu must be nonnegative and --eval-episodes must be in [0, 100]")
    if args.name and (Path(args.name).name != args.name or args.name in [".", ".."]):
        p.error("--name must be a single directory name")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with (STATE.parent / "recap-pipeline.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_pipeline(args)


if __name__ == "__main__":
    main()
