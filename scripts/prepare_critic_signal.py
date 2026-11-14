"""Prepare immutable BC replications and frozen critic-signal controls."""

import argparse
import copy
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
from evaluate_multitask import digest, write_json
from vla_rl_mt.shared_value import PrefixAdapter

PROTOCOL = ROOT/"configs/multitask/critic_signal.json"


def replicas():
    setting = json.loads(PROTOCOL.read_text())
    work = ROOT/setting["work"]
    work.mkdir(parents=True, exist_ok=False)
    original = ROOT/setting["initial_addon"]
    parent = torch.load(original, map_location="cpu", weights_only=True)
    base = json.loads((ROOT/setting["parent_protocol"]).read_text())
    template = next(j for j in json.loads((ROOT/base["work"]/"jobs.json").read_text())["jobs"] if j["name"] == "task_mixed")
    jobs, variants = [], []
    for seed in setting["bc_replica_seeds"]:
        initial = copy.deepcopy(parent)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            adapter = PrefixAdapter(parent["architecture"]["width"], parent["architecture"]["rank"])
        initial["addon"]["adapter"] = adapter.state_dict()
        initial["initialization_note"] = f"Actor seed{seed}: resampled zero-output adapter down weights, unused frozen head kept common; base policy unchanged"
        if bool(initial["addon"]["adapter"]["up.weight"].any()):
            raise ValueError("Replication adapter is not zero output")
        initial_path = work/f"initial-seed{seed}.pt"
        torch.save(initial, initial_path)
        overrides = []
        for value in template["overrides"]:
            key = value.split("=", 1)[0]
            if key == "actor.seed": value = f"actor.seed={seed}"
            elif key == "+actor.model.initial_addon_path": value = f"{key}={initial_path}"
            elif key == "+actor.model.initial_addon_sha256": value = f"{key}={digest(initial_path)}"
            elif key == "+runner.eval_export_steps": value = f"{key}=[]"
            elif key == "actor.group_name": value = f"{key}=bc_seed{seed}"
            overrides.append(value)
        job = {**template, "name": f"bc_seed{seed}", "overrides": overrides}
        jobs.append(job)
        actor = ROOT/setting["replica_output"]/job["name"]/"checkpoints/global_step_128/actor"
        variants.append({"name": job["name"], "seed": seed, "step": 128, "parent": str(initial_path),
                         "parent_sha256": digest(initial_path), "checkpoint": str(actor/"addon.pt"),
                         "ready_marker": str(actor/"evaluation-ready.json")})
    write_json(work/"replica-jobs.json", {"jobs": jobs, "protocol_sha256": digest(PROTOCOL)})
    write_json(work/"replica-evaluation.json", {"variants": variants, "protocol_sha256": digest(PROTOCOL)})
    write_json(work/"initial-seal.json", {"protocol_sha256": digest(PROTOCOL), "original_addon_sha256": digest(original),
        "pool_manifest_sha256": setting["pool_manifest_sha256"], "replica_jobs_sha256": digest(work/"replica-jobs.json")})
    print(json.dumps({"replicas": len(jobs), "steps_each": 128, "replica_rollouts": 48}), flush=True)


def cache_features():
    from vla_rl_mt.joint_value import BoundaryPool, to_observation
    from vla_rl_mt.shared_value import SharedPrefixValue, pool_prefix
    from vla_rl_mt.task_mixing import native_policy

    setting = json.loads(PROTOCOL.read_text())
    work = ROOT/setting["work"]
    path = ROOT/setting["feature_cache"]
    if path.exists():
        raise FileExistsError("Frozen feature cache already exists")
    pool = BoundaryPool(ROOT/setting["pool"], setting["pool_manifest_sha256"])
    initial = torch.load(ROOT/setting["initial_addon"], map_location="cpu", weights_only=True)
    model = native_policy(pool.manifest["protocol"])
    probe = SharedPrefixValue(model, **initial["architecture"]).to("cuda").eval()
    probe.load_addon(initial["addon"])
    probe.requires_grad_(False)
    features = []
    for start in range(0, len(pool.rows), 4):
        samples = [pool.sample(i)[0] for i in range(start, min(start+4, len(pool.rows)))]
        observation = to_observation(torch.utils.data.default_collate(samples), "cuda")
        with torch.no_grad():
            hidden, mask = probe.prefix_tokens(observation, detach_shared=True)
            feature = pool_prefix(hidden, mask, observation.tokenized_prompt_mask, readout="vision_prompt_concat")
        features.append(feature.cpu())
        if start % 100 == 0:
            print(json.dumps({"cached": start+len(samples), "total": len(pool.rows)}), flush=True)
    probe.close()
    tensor = torch.cat(features)
    if tensor.shape != (len(pool.rows), 4096) or not bool(torch.isfinite(tensor).all()):
        raise ValueError("Nonfinite or wrong-shape frozen critic features")
    rows = [{**row, "use_for_loss": True, "terminal": False, "episode_limit": 400} for row in pool.rows]
    cache = {"features": tensor, "rows": rows, "condition": "instruction", "readout": "vision_prompt_concat",
        "target_scheme": "terminal_binary", "pool_manifest_sha256": setting["pool_manifest_sha256"],
        "base_sha256": initial["base_sha256"], "initial_addon_sha256": digest(ROOT/setting["initial_addon"]),
        "scope": "Original frozen Motus prefix, no W1 or action suffix; only real decision rows, terminal bootstrap prescribed zero"}
    path.parent.mkdir(parents=True, exist_ok=False)
    torch.save(cache, path)
    write_json(work/"feature-seal.json", {"feature_sha256": digest(path), "rows": len(rows), "width": 4096,
        "pool_manifest_sha256": setting["pool_manifest_sha256"], "initial_addon_sha256": cache["initial_addon_sha256"],
        "protocol_sha256": digest(PROTOCOL), "source_sha256": digest(Path(__file__))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("replicas", "cache"))
    args = parser.parse_args()
    {"replicas": replicas, "cache": cache_features}[args.phase]()


if __name__ == "__main__":
    main()
