"""Run a bounded queue of official-worker jobs on owned GPU 0/1 Ray slots."""

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


def validate_jobs(jobs):
    names = {job["name"] for job in jobs}
    if not jobs or len(names) != len(jobs):
        raise ValueError("Worker names must be distinct and nonempty")
    pending = {job["name"]: set(job.get("depends_on", [])) for job in jobs}
    if any(not dependencies <= names for dependencies in pending.values()):
        raise ValueError("Unknown worker dependency")
    finished = set()
    while pending:
        ready = {name for name, dependencies in pending.items() if dependencies <= finished}
        if not ready:
            raise ValueError("Cyclic worker dependencies")
        finished.update(ready)
        pending = {name: dependencies for name, dependencies in pending.items() if name not in ready}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", choices=(0, 1), default=[0, 1],
                        help="Owned physical GPU slots; select one while the other runs independent diagnostics")
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus):
        parser.error("GPU slots must be unique")
    jobs = json.loads(args.plan.read_text())["jobs"]
    validate_jobs(jobs)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    private = Path(tempfile.mkdtemp(prefix="mtj-", dir="/tmp"))
    token = private/"token"
    token.write_text(secrets.token_hex(32))
    token.chmod(0o600)
    os.environ.update(RAY_AUTH_MODE="token", RAY_AUTH_TOKEN_PATH=str(token))
    import ray

    state = {"complete": False, "jobs": [{**job, "status": "pending"} for job in jobs]}
    active = {}

    def save():
        temp = output/"execution.json.tmp"
        temp.write_text(json.dumps(state, indent=2)+"\n")
        temp.replace(output/"execution.json")

    try:
        context = ray.init(address="local", num_cpus=32, num_gpus=2, include_dashboard=False,
                           object_store_memory=1024**3, _temp_dir=str(private/"ray"),
                           namespace="method-job-owner", logging_level="ERROR")
        address = context.address_info["gcs_address"]
        state["ray_address"] = address
        while any(row["status"] in ("pending", "running") for row in state["jobs"]):
            for gpu, (child, stream, row) in list(active.items()):
                if child.poll() is not None:
                    row.update(status="complete" if child.returncode == 0 else "failed", exit_code=child.returncode)
                    stream.close()
                    del active[gpu]
            if any(row["status"] == "failed" for row in state["jobs"]):
                save()
                raise RuntimeError("Worker failure; remaining queue stopped, inspect logs")
            for gpu in args.gpus:
                finished = {row["name"] for row in state["jobs"] if row["status"] == "complete"}
                pending = next((row for row in state["jobs"] if row["status"] == "pending"
                                and set(row.get("depends_on", [])) <= finished), None)
                if gpu in active or pending is None:
                    continue
                env = {**os.environ, "RAY_ADDRESS": address, "M1_GPU": str(gpu), "M2_GPU": str(gpu),
                       "RLINF_EXT_MODULE": pending["extension"], "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "JAX_PLATFORMS": "cpu", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
                command = [sys.executable, str(ROOT/pending["entry"]), "--config-path", str(ROOT/"configs/multitask"),
                           "--config-name", pending["config"], *pending["overrides"],
                           f"runner.logger.log_path={output}", f"runner.logger.experiment_name={pending['name']}",
                           f"+runner.ray_namespace=mt-{output.name}-{pending['name']}"]
                stream = (output/f"{pending['name']}.log").open("w")
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                pending.update(status="running", gpu=gpu, pid=child.pid, command=command)
                active[gpu] = child, stream, pending
            save()
            if active:
                time.sleep(1)
        state["complete"] = True
        save()
    finally:
        for child, stream, _ in active.values():
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            stream.close()
        ray.shutdown()
        token.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
