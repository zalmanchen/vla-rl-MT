#!/usr/bin/env python3
"""Plan/run task-balanced BC experiments and evaluate frozen checkpoints."""

import argparse
from datetime import datetime
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vla_mtrl.common import ROOT, TASKS, compose, digest, environment, write_json


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["train", "eval", "status"])
    p.add_argument("--tasks", nargs="+", default=TASKS)
    p.add_argument(
        "--gpu",
        "--gpus",
        type=int,
        default=0,
        help="One physical GPU per model; paired jobs run sequentially",
    )
    p.add_argument("--mode", choices=["mt", "st", "paired"], default="mt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int)
    p.add_argument("--samples-per-task", type=int, default=8)
    p.add_argument("--micro-batch", type=int)
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--manifest", type=Path, default=ROOT / ".data/robotwin/manifest.json"
    )
    p.add_argument(
        "--run",
        default="latest",
        help="Campaign path; train creates a new one unless supplied",
    )
    p.add_argument("--resume-checkpoint", type=Path)
    p.add_argument(
        "--checkpoint", type=Path, help="Optional explicit checkpoint to evaluate"
    )
    p.add_argument("--episodes", type=int)
    p.add_argument("--max-control-steps", type=int, default=500)
    p.add_argument("--plan", "--dry-run", action="store_true")
    p.add_argument("--detach", action="store_true")
    return p


def resolve_run(value):
    if value == "latest":
        return Path(json.loads((ROOT / ".runtime/latest.json").read_text())["run"])
    return Path(value).resolve()


def gpu_uuid(index):
    result = subprocess.check_output(
        ["nvidia-smi", "-i", str(index), "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if not result.startswith("GPU-") or "\n" in result:
        raise ValueError("Cannot identify the requested GPU")
    return result


def main():
    args = parser().parse_args()
    if len(set(args.tasks)) != len(args.tasks) or not args.tasks or args.gpu < 0:
        raise ValueError("Use unique task names and a nonnegative GPU index")
    args.steps = args.steps or (20 if args.smoke else 1000)
    args.episodes = args.episodes or (2 if args.smoke else 20)
    if args.steps < 2 or args.episodes < 1 or args.samples_per_task < 1:
        raise ValueError("Steps >=2, episodes >0, samples per task >0 required")
    groups = [args.tasks] if args.mode == "mt" else [[t] for t in args.tasks]
    if args.mode == "paired":
        groups = [args.tasks] + [[t] for t in args.tasks]
    if args.plan:
        print(
            json.dumps(
                {
                    "action": args.action,
                    "tasks": args.tasks,
                    "training_groups": groups,
                    "steps": args.steps,
                    "seed": args.seed,
                    "episodes_per_task": args.episodes,
                    "gpu": args.gpu,
                    "manifest": str(args.manifest),
                    "root": str(ROOT),
                    "gpu_query_or_allocation": False,
                },
                indent=2,
            )
        )
        return
    environment()
    if args.action == "status":
        run = resolve_run(args.run)
        print((run / "status.json").read_text())
        return
    if args.detach:
        log_dir = ROOT / ".runtime/controllers"
        log_dir.mkdir(parents=True, exist_ok=True)
        logfile = log_dir / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".log")
        argv = [value for value in sys.argv[1:] if value != "--detach"]
        # Freeze 'latest' before dispatching a detached evaluation.
        if args.action == "eval" and args.run == "latest":
            if "--run" in argv:
                argv[argv.index("--run") + 1] = str(resolve_run("latest"))
            else:
                argv += ["--run", str(resolve_run("latest"))]
        with logfile.open("a") as stream:
            child = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), *argv],
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        print(json.dumps({"pid": child.pid, "logfile": str(logfile)}, indent=2))
        return
    lock_dir = ROOT / ".runtime/locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / f"gpu{args.gpu}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid(args.gpu)
        if args.action == "eval":
            from evaluate_robotwin import evaluate

            evaluate(args, resolve_run(args.run))
            return
        from omegaconf import OmegaConf
        from vla_mtrl.data import inspect_episode

        manifest_path = args.manifest.resolve()
        manifest = json.loads(manifest_path.read_text())
        if not set(args.tasks) <= set(manifest["tasks"]):
            raise ValueError("Training tasks are missing from the frozen manifest")
        norm = manifest_path.with_suffix(".norm_stats.json")
        if not norm.is_file():
            raise FileNotFoundError("Run scripts/prepare_data.py first")
        raw_root = manifest_path.parent / manifest["raw_root"]
        for row in manifest["episodes"]:
            path = raw_root / row["file"]
            if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
                raise ValueError(f"Frozen demonstration changed: {path}")
            inspect_episode(path)
        model = ROOT / ".model/pi0_robotwin/model.safetensors"
        release = json.loads((ROOT / "resources/releases.json").read_text())["model"]
        expected_model = next(
            row for row in release["files"] if row["rfilename"] == "model.safetensors"
        )
        if not model.is_file() or model.stat().st_size != expected_model["size"]:
            raise FileNotFoundError(
                "Model is incomplete; run ./download.sh --component model"
            )
        run = (
            ROOT
            / ".runtime/experiments"
            / datetime.now().strftime("robotwin_%Y%m%d_%H%M%S_%f")
            if args.run == "latest"
            else Path(args.run).resolve()
        )
        run.mkdir(parents=True, exist_ok=True)
        source_files = [
            *sorted((ROOT / "vla_mtrl").glob("*.py")),
            ROOT / "train_bc.py",
            ROOT / "configs/robotwin_bc.yaml",
            Path(__file__).resolve(),
        ]
        contract = {
            "tasks": args.tasks,
            "mode": args.mode,
            "seed": args.seed,
            "steps": args.steps,
            "samples_per_task": args.samples_per_task,
            "micro_batch_override": args.micro_batch,
            "manifest_sha256": digest(manifest_path),
            "norm_sha256": digest(norm),
            "model_sha256": expected_model["lfs"]["sha256"],
            "source_sha256": {
                str(p.relative_to(ROOT)): digest(p) for p in source_files
            },
            "environment": {
                n: importlib.metadata.version(n)
                for n in ["torch", "rlinf-openpi", "ray", "transformers", "h5py"]
            },
            "submodules": {
                name: subprocess.check_output(
                    ["git", "-C", str(ROOT / name), "rev-parse", "HEAD"], text=True
                ).strip()
                for name in ["RLinf", "RoboTwin"]
            },
        }
        contract_path = run / "contract.json"
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError("Run contract changed; create a new run")
        write_json(contract_path, contract)
        shutil.copyfile(norm, run / "norm_stats.json")
        write_json(ROOT / ".runtime/latest.json", {"run": str(run)})
        jobs = {}
        write_json(
            run / "status.json", {"phase": "training", "run": str(run), "jobs": jobs}
        )
        for tasks in groups:
            name = "mt" if len(tasks) > 1 else "st_" + tasks[0]
            job = run / name
            job.mkdir(exist_ok=True)
            cfg = compose(
                args.steps,
                tasks,
                args.seed,
                args.samples_per_task,
                args.micro_batch,
                job,
            )
            if (
                cfg.actor.global_batch_size % cfg.actor.micro_batch_size
                or cfg.actor.micro_batch_size % len(tasks)
            ):
                raise ValueError(
                    "Microbatch must balance tasks and divide the global batch"
                )
            cfg.data.train_data_paths = str(manifest_path)
            cfg.actor.model.openpi_data.norm_stats_path = str(run / "norm_stats.json")
            cfg.runner.save_interval = max(1, args.steps // 2) if args.smoke else 250
            if args.resume_checkpoint:
                cfg.runner.resume_dir = str(args.resume_checkpoint.resolve())
            serialized = OmegaConf.to_container(cfg, resolve=True)
            serialized["hydra"] = {"run": {"dir": str(job)}, "output_subdir": None}
            OmegaConf.save(OmegaConf.create(serialized), job / "train_config.yaml")
            jobs[name] = {"phase": "training", "tasks": tasks}
            write_json(
                run / "status.json",
                {"phase": "training", "run": str(run), "jobs": jobs},
            )
            with (job / "stdout.log").open("a") as stream:
                try:
                    subprocess.run(
                        [
                            sys.executable,
                            str(ROOT / "train_bc.py"),
                            "--config-path",
                            str(job),
                            "--config-name",
                            "train_config",
                        ],
                        cwd=ROOT,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                except subprocess.CalledProcessError:
                    jobs[name] = {
                        "phase": "failed",
                        "tasks": tasks,
                        "log": str(job / "stdout.log"),
                    }
                    write_json(
                        run / "status.json",
                        {"phase": "failed", "run": str(run), "jobs": jobs},
                    )
                    raise
            matches = list(
                job.rglob(
                    f"global_step_{args.steps}/actor/model_state_dict/full_weights.pt"
                )
            )
            if len(matches) != 1:
                raise RuntimeError(
                    "Training did not produce exactly one final-budget checkpoint"
                )
            jobs[name] = {
                "phase": "completed",
                "tasks": tasks,
                "checkpoint": str(matches[0]),
            }
            write_json(job / "result.json", jobs[name])
            write_json(
                run / "status.json",
                {"phase": "training", "run": str(run), "jobs": jobs},
            )
        write_json(
            run / "status.json",
            {"phase": "training_completed", "run": str(run), "jobs": jobs},
        )
        print(json.dumps({"run": str(run), "jobs": jobs}, indent=2))


if __name__ == "__main__":
    main()
