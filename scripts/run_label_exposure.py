"""Complete the sealed label-control queue, evaluations and runtime audit."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from evaluate_multitask import digest, write_json


def read(path):
    return json.loads(path.read_text())


def audit_training(output, steps, parent, expected_pool):
    import torch
    from vla_rl_mt.expert_policy import state_digest

    execution = read(output/"execution.json")
    if not execution["complete"] or any(j["status"] != "complete" or j["exit_code"] != 0 for j in execution["jobs"]):
        raise ValueError("Training drivers did not all complete successfully")
    result = {}
    for job in execution["jobs"]:
        path = output/job["name"]/f"checkpoints/global_step_{steps}/actor/addon.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        identity = checkpoint["identity"]
        if checkpoint["step"] != steps or identity["pool_manifest_sha256"] != expected_pool:
            raise ValueError("Checkpoint step or pool changed")
        if identity["update_scope"] != "expert_adapter" or identity["expert_lr"] != 1e-6:
            raise ValueError("Unexpected trainable scope or expert learning rate")
        if checkpoint["expert_sha256"] != state_digest(checkpoint["expert"]):
            raise ValueError("Exported expert digest differs")
        if any(not torch.equal(value, parent["addon"]["value_head"][key])
               for key, value in checkpoint["addon"]["value_head"].items()):
            raise ValueError("Frozen critic head changed")
        rows = [json.loads(line) for line in (output/job["name"]/"joint-metrics.jsonl").read_text().splitlines()]
        if len(rows) != steps or [r["step"] for r in rows] != list(range(1, steps+1)):
            raise ValueError("Missing or repeated optimizer updates")
        for row in rows:
            gradients = row["gradients"]
            if not all(g["finite"] for g in gradients.values()) or gradients["frozen_vlm"]["gradient_tensors"] or gradients["value_head"]["gradient_tensors"]:
                raise ValueError("Invalid gradients in the declared update scope")
        result[job["name"]] = {"identity": identity, "exposure": checkpoint["exposure"],
                               "checkpoint_sha256": digest(path), "peak_allocated_gib": max(r["peak_allocated_gib"] for r in rows)}
    sequences = [result[f"label_{name}"]["identity"]["sample_sequence_sha256"] for name in ("real", "shuffled", "success_matched")]
    if len(set(sequences)) != 1:
        raise ValueError("Mixed-data draw schedules differ")
    return result


def main():
    import torch

    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", JAX_PLATFORMS="cpu",
                      XLA_PYTHON_CLIENT_PREALLOCATE="false")
    config = read(ROOT/"configs/multitask/label_exposure.json")
    work, updates = ROOT/config["work"], ROOT/config["training_output"]
    state = {"stage": "wait_smoke", "complete": False, "pid": os.getpid(), "started_unix": time.time()}
    def save():
        write_json(work/"controller.json", state)
    save()
    try:
        smoke = ROOT/".runtime/runs/method/label-exposure-smoke-v1"
        deadline = time.monotonic()+3600
        while True:
            execution = read(smoke/"execution.json") if (smoke/"execution.json").exists() else {}
            if any(j["status"] == "failed" for j in execution.get("jobs", [])):
                raise RuntimeError("Short-run worker failed; main queue not launched")
            if execution.get("complete"):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Short-run queue did not finish within1h")
            time.sleep(2)
        sealed = read(work/"actor-protocol-seal.json")
        if digest(ROOT/"configs/multitask/label_exposure.json") != sealed["protocol_sha256"]:
            raise ValueError("Actor protocol changed after exposure audit")
        for name in ("jobs", "evaluation"):
            if digest(work/f"{name}.json") != sealed[f"{name}_sha256"]:
                raise ValueError("Sealed queue changed")
        for name, expected in read(work/"source-hashes.json").items():
            if digest(ROOT/name) != expected:
                raise ValueError(f"Worker source changed: {name}")
        base = read(ROOT/"configs/multitask/method_m4.json")
        parent = torch.load(ROOT/base["w1_checkpoint"], map_location="cpu", weights_only=True)
        audit = audit_training(smoke, 4, parent, sealed["pool_sha256"])
        write_json(work/"smoke-audit.json", audit)
        state["stage"] = "train128"
        save()
        command = [sys.executable, str(ROOT/"scripts/run_method_jobs.py"), "--plan", str(work/"jobs.json"), "--output", str(updates)]
        with (work/"training-controller.log").open("w") as stream:
            subprocess.run(command, cwd=ROOT, check=True, stdout=stream, stderr=subprocess.STDOUT)
        training = audit_training(updates, config["steps"], parent, sealed["pool_sha256"])
        # The short smoke is an exact prefix replay, including stochastic draws.
        for name in training:
            prefix = [json.loads(line)["sft_loss"] for line in (smoke/name/"joint-metrics.jsonl").read_text().splitlines()]
            main_losses = [json.loads(line)["sft_loss"] for line in (updates/name/"joint-metrics.jsonl").read_text().splitlines()[:4]]
            if prefix != main_losses:
                raise ValueError(f"Short-run prefix does not replay: {name}")
        exposure = read(work/"exposure-audit.json")
        for name in ("real", "shuffled", "success_matched"):
            if training[f"label_{name}"]["exposure"]["positive_conditional"] != exposure["groups"][name]["retained_draws"]:
                raise ValueError("Actual worker exposure differs from predeclared control")
        write_json(work/"training-audit.json", {"complete": True, "groups": training, "smoke_prefix_exact": True})
        state["stage"] = "evaluate96"
        save()
        children = []
        for gpu in (0, 1):
            command = [sys.executable, str(ROOT/"scripts/evaluate_method_updates.py"), "--plan", str(work/"evaluation.json"),
                       "--expert", "--shard", str(gpu), "--num-shards", "2", "--output",
                       str(ROOT/f".runtime/runs/method/label-exposure-eval-v1/shard{gpu}")]
            stream = (work/f"evaluation-{gpu}.log").open("w")
            child = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}, stdout=stream, stderr=subprocess.STDOUT)
            children.append((child, stream))
        failed = []
        for child, stream in children:
            code = child.wait()
            stream.close()
            if code:
                failed.append(code)
        if failed:
            raise RuntimeError(f"Evaluation workers failed: {failed}")
        rows = []
        for gpu in (0, 1):
            result = read(ROOT/f".runtime/runs/method/label-exposure-eval-v1/shard{gpu}/evaluation.json")
            if not result["complete"]:
                raise ValueError("Evaluation incomplete")
            rows.extend(result["rows"])
        if len(rows) != 96 or any(r["status"] != "complete" for r in rows):
            raise ValueError("Evaluation budget or operational validity differs")
        from summarize_method_m4 import paired
        scores = {}
        for variant in sorted({r["variant"] for r in rows}):
            scores[variant] = {task: sum(r["success"] for r in rows if r["variant"] == variant and r["task"] == task)
                               for task in ("adjust_bottle", "click_bell")}
        mappings = {name: {(r["task"], r["seed"]): r for r in rows if r["variant"] == name} for name in scores}
        if any(len(mapping) != 24 for mapping in mappings.values()):
            raise ValueError("Duplicate or absent paired scenes")
        reference = {(r["task"], r["seed"]): r for r in read(ROOT/".runtime/runs/method/m2-joint-v1/comparison.json")["rows"] if r["variant"] == "w1"}
        for mapping in mappings.values():
            if set(mapping) != set(reference):
                raise ValueError("Paired cohort differs from W1")
            if any(mapping[k]["noise_seed"] != reference[k]["noise_seed"] or mapping[k]["observed_init_fingerprint"] != reference[k]["observed_init_fingerprint"] for k in reference):
                raise ValueError("Noise or observed initialization differs from W1")
        comparison = {"complete": True, "scores_out_of12": scores,
                      "w1_reference_out_of12": {task: sum(r["success"] for r in reference.values() if r["task"] == task) for task in ("adjust_bottle", "click_bell")},
                      "real_vs_shuffled": paired(mappings["label_real"], mappings["label_shuffled"], ["adjust_bottle", "click_bell"]),
                      "versus_w1": {name: paired(mapping, reference, ["adjust_bottle", "click_bell"]) for name, mapping in mappings.items()},
                      "scope": "Single actor seed and original24 development scenes; fixed final128, no final test"}
        write_json(work/"final-comparison.json", comparison)
        preservation = {name: digest(ROOT/name) == expected for name, expected in read(work/"preservation-before.json").items()}
        if not all(preservation.values()):
            raise ValueError("Protected input changed")
        write_json(work/"task-audit.json", {"complete": True, "preservation": preservation, "new_rollouts": 96,
                   "actor_groups": 4, "steps_per_group": 128, "evaluation_returns_to_training": False, "final_test_used": False})
        state.update(stage="complete", complete=True, finished_unix=time.time())
        save()
    except Exception as error:
        state.update(stage="failed", error=repr(error), finished_unix=time.time())
        save()
        raise


if __name__ == "__main__":
    main()
