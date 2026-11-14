"""Launch matched single-GPU W1/W2 jobs on a task-owned local Ray cluster."""

import argparse
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, default=2)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--value-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--modes", nargs="+", choices=("w0", "w1", "w2"), default=["w1", "w2"])
    parser.add_argument("--skip-value-eval", action="store_true", help="Interface checks only")
    args = parser.parse_args()
    if len(args.modes) > 2 or len(set(args.modes)) != len(args.modes):
        parser.error("Specify one or two distinct single-GPU modes")
    if (args.steps <= 0 or args.global_batch_size <= 0 or args.value_weight <= 0
            or args.micro_batch_size <= 0 or args.global_batch_size % args.micro_batch_size):
        parser.error("Steps, batch size and value weight must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    private = Path(tempfile.mkdtemp(prefix="m2-", dir="/tmp"))
    token = private/"token"
    token.write_text(secrets.token_hex(32))
    token.chmod(0o600)
    os.environ.update(RAY_AUTH_MODE="token", RAY_AUTH_TOKEN_PATH=str(token))
    import ray

    children, streams = [], []
    state = {"scope": "M2 paired continued-SFT mechanism probe; not original RECAP or final performance",
             "steps": args.steps, "global_batch_size": args.global_batch_size,
             "micro_batch_size": args.micro_batch_size, "seed": args.seed, "value_weight": args.value_weight,
             "jobs": [], "complete": False}

    def save():
        (output/"execution.json").write_text(json.dumps(state, indent=2)+"\n")

    try:
        context = ray.init(address="local", num_cpus=32, num_gpus=2, include_dashboard=False,
                           object_store_memory=1024**3, _temp_dir=str(private/"ray"),
                           namespace="m2-owner", logging_level="ERROR")
        address = context.address_info["gcs_address"]
        state["ray_address"] = address
        for gpu, mode in enumerate(args.modes):
            stream = (output/f"{mode}.log").open("w")
            streams.append(stream)
            env = {**os.environ, "RAY_ADDRESS": address, "M2_GPU": str(gpu),
                   "RLINF_EXT_MODULE": "vla_rl_mt.joint_value", "HF_HUB_OFFLINE": "1",
                   "TRANSFORMERS_OFFLINE": "1", "JAX_PLATFORMS": "cpu",
                   "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
            command = [sys.executable, str(ROOT/"scripts/train_method_joint.py"),
                       "--config-path", str(ROOT/"configs/multitask"), "--config-name", "joint_value",
                       f"actor.model.mode={mode}", f"actor.seed={args.seed}",
                       f"actor.micro_batch_size={args.micro_batch_size}",
                       f"actor.global_batch_size={args.global_batch_size}", f"runner.max_steps={args.steps}",
                       f"runner.val_check_interval={-1 if args.skip_value_eval else args.steps}",
                       f"runner.save_interval={args.steps}", f"actor.model.value_weight={args.value_weight}",
                       f"runner.logger.log_path={output}", f"+runner.ray_namespace=m2-{output.name}-{mode}"]
            if args.skip_value_eval:
                command.append("+runner.skip_value_eval=true")
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            children.append(child)
            state["jobs"].append({"mode": mode, "gpu": gpu, "pid": child.pid, "command": command,
                                  "status": "running", "exit_code": None})
        save()
        while any(child.poll() is None for child in children):
            for row, child in zip(state["jobs"], children, strict=True):
                if child.poll() is not None:
                    row.update(status="complete" if child.returncode == 0 else "failed", exit_code=child.returncode)
            save()
            time.sleep(1)
        for row, child in zip(state["jobs"], children, strict=True):
            row.update(status="complete" if child.returncode == 0 else "failed", exit_code=child.returncode)
        state["complete"] = all(row["exit_code"] == 0 for row in state["jobs"])
        save()
        if not state["complete"]:
            raise RuntimeError(f"A paired M2 job failed; inspect {output}")
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        for stream in streams:
            stream.close()
        ray.shutdown()
        token.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
