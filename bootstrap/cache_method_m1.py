"""Extract both declared M1 feature views from an already frozen offline pool."""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_pool(pool, manifest):
    """Certify actual files/splits/returns, not just metadata readiness flags."""
    import h5py
    import numpy as np

    plan = json.loads((pool/"plan.json").read_text())
    if digest(pool/"plan.json") != manifest["plan_sha256"]:
        raise ValueError("Frozen plan changed")
    planned = {(row["task"], row["split"], row["seed"]) for row in plan["cases"]}
    recorded = [(row["task"], row["split"], row["seed"]) for row in manifest["episodes"]]
    if len(recorded) != len(set(recorded)) or set(recorded) != planned:
        raise ValueError("Recorded scenes differ from the complete plan")
    if len({(task, seed) for task, _, seed in recorded}) != len(recorded):
        raise ValueError("Scene reused across train/validation")
    if manifest["config"]["gamma"] != 1.0:
        raise ValueError("This M1 protocol fixes undiscounted returns")
    for row in manifest["episodes"]:
        path = pool/row["file"]
        if not path.resolve().is_relative_to(pool.resolve()) or digest(path) != row["sha256"]:
            raise ValueError("Episode escaped the pool or changed")
        with h5py.File(path) as episode:
            for key in ("task", "split", "seed", "success"):
                if episode.attrs[key] != row[key]:
                    raise ValueError(f"Episode identity/outcome mismatch: {key}")
            if episode.attrs["model_sha256"] != manifest["protocol"]["model"]["sha256"]:
                raise ValueError("Different behavior model in frozen pool")
            steps = episode["observation_steps"][:]
            length = len(episode["actions"])
            if (len(steps) < 2 or steps[0] != 0 or steps[-1] != length
                    or (np.diff(steps) <= 0).any() or (np.diff(steps) > 32).any()
                    or len(episode["state"]) != len(steps)):
                raise ValueError("Invalid action/chunk/next-observation alignment")
            if not np.array_equal(episode["action_chunk_id"][:], np.repeat(np.arange(len(steps)-1), np.diff(steps))):
                raise ValueError("Action chunk IDs disagree with observation boundaries")
            if not episode.attrs["final_observation_saved"] or not episode.attrs["complete"]:
                raise ValueError("Incomplete episode or absent final observation")
            if bool(episode.attrs["terminated"]) != row["success"] or bool(episode.attrs["truncated"]) == row["success"]:
                raise ValueError("Terminal/deadline flags disagree with simulator outcome")
            if not row["success"] and length != episode.attrs["episode_limit"]:
                raise ValueError("External/incomplete truncation cannot be labeled a deadline failure")
            for key in episode["images"]:
                if len(episode[f"images/{key}"]) != len(steps):
                    raise ValueError("Camera is missing a boundary/final observation")
            rewards = np.full(length, -1, dtype=np.float32)
            rewards[-1] = 0 if row["success"] else manifest["config"]["failure_reward"]
            returns = np.cumsum(rewards[::-1])[::-1]
            chunk_rewards = np.asarray([rewards[left:right].sum()
                                        for left, right in zip(steps[:-1], steps[1:], strict=True)])
            for name, expected in (("rewards", rewards), ("returns", returns),
                                   ("decision_returns", returns[steps[:-1]]), ("chunk_rewards", chunk_rewards)):
                if not np.array_equal(episode[name][:], expected):
                    raise ValueError(f"Reward/return boundary mismatch: {name}")
            continuation = np.append(returns[steps[1:-1]], 0)
            if not np.array_equal(chunk_rewards+continuation, returns[steps[:-1]]):
                raise ValueError("Chunk MC Bellman identity failed")
    return {"episodes": len(recorded), "scenes_disjoint": True,
            "post_action_final_observations_verified": True, "chunk_returns_verified": True}


def main():
    import cv2
    import h5py
    import numpy as np
    import torch
    from openpi.transforms import compose
    from rlinf.models.embodiment.openpi.modules.model import Observation
    from rlinf.models.embodiment.openpi.transforms.pipeline import (
        build_openpi_transforms,
    )

    from vla_rl_mt.shared_value import SharedPrefixValue, pool_prefix

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--condition", choices=("instruction", "no_instruction"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--readout", choices=("global_mean", "vision_prompt_concat"), default="global_mean")
    parser.add_argument("--global-control-output", type=Path,
                        help="Also save duplicated global means from the identical forward passes")
    parser.add_argument("--reference-cache", type=Path,
                        help="Require exact global-feature and row parity with an existing M1 cache")
    args = parser.parse_args()
    pool = args.pool.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    control_output = args.global_control_output.resolve() if args.global_control_output else None
    if control_output and (control_output == output or control_output.exists()):
        raise FileExistsError(control_output)
    if control_output and args.readout != "vision_prompt_concat":
        raise ValueError("The matched global control accompanies the grouped readout")
    manifest = json.loads((pool/"manifest.json").read_text())
    if not manifest["scene_split_certified"] or not manifest["outcome_coverage_ready"]:
        raise ValueError("M1 requires frozen traceable scenes and per-task/split outcome coverage")
    data_audit = validate_pool(pool, manifest)
    protocol, config = manifest["protocol"], manifest["config"]
    if digest(ROOT/protocol["model"]["path"]/"model.safetensors") != protocol["model"]["sha256"]:
        raise ValueError("Base model checksum mismatch")
    inputs, _ = build_openpi_transforms(str(ROOT/protocol["model"]["path"]), "pi05_aloha_robotwin",
                                       {"norm_stats_path": str(ROOT/protocol["model"]["norm_stats"]),
                                        "adapt_to_pi": True, "extra_delta_transform": True})
    transform = compose(inputs)
    sys.path.insert(0, str(ROOT/"scripts"))
    from evaluate_multitask import load_model

    torch.manual_seed(20261112)
    model, model_config = load_model("native", protocol)
    probe = SharedPrefixValue(model, head_width=config["value_head_width"]).to("cuda").train()
    probe.requires_grad_(False)
    model.gradient_checkpointing_enable({"use_reentrant": False})
    if args.readout == "vision_prompt_concat" and getattr(model, "pcd", None):
        raise ValueError("Grouped readout requires image+prompt prefix ordering")
    features, global_features, rows, pending = [], [], [], []

    def flush():
        if not pending:
            return

        def stack(values):
            if isinstance(values[0], dict):
                return {key: stack([value[key] for value in values]) for key in values[0]}
            return torch.as_tensor(np.stack(values)).to("cuda")

        observation = Observation.from_dict(stack(pending))
        with torch.no_grad():
            hidden, mask = probe.prefix_tokens(observation, detach_shared=True)
            features.append(pool_prefix(hidden, mask, observation.tokenized_prompt_mask,
                                        readout=args.readout).cpu())
            if control_output or args.reference_cache:
                global_features.append(pool_prefix(hidden, mask).cpu())
        pending.clear()

    for entry in manifest["episodes"]:
        source = pool/entry["file"]
        if digest(source) != entry["sha256"]:
            raise ValueError(f"Frozen episode changed: {source}")
        with h5py.File(source) as episode:
            boundaries = episode["observation_steps"][:]
            decision_returns = episode["decision_returns"][:]
            chunk_rewards = episode["chunk_rewards"][:]
            for index, step in enumerate(boundaries):
                terminal = index == len(boundaries)-1
                images = {}
                for camera in episode["images"]:
                    image = cv2.imdecode(episode[f"images/{camera}"][index], cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError("Invalid frozen PNG observation")
                    images[camera] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
                raw = {"images": images, "state": episode["state"][index],
                       "prompt": episode.attrs["instruction"] if args.condition == "instruction" else ""}
                sample = transform(copy.deepcopy(raw))
                pending.append(sample)
                rows.append({"episode": entry["file"], "task": entry["task"], "split": entry["split"],
                             "seed": entry["seed"], "observation_index": index, "step": int(step),
                             "episode_limit": int(episode.attrs["episode_limit"]), "terminal": terminal,
                             "success": entry["success"], "use_for_loss": not terminal,
                             "target": 0.0 if terminal else float(decision_returns[index]/config["return_scale"]),
                             "chunk_reward": 0.0 if terminal else float(chunk_rewards[index]/config["return_scale"]),
                             "valid_prompt_tokens": int(np.asarray(sample["tokenized_prompt_mask"]).sum())})
                if len(pending) == args.batch_size:
                    flush()
        print(json.dumps({"condition": args.condition, "episode": entry["file"], "observations": len(rows)}), flush=True)
    flush()
    probe.close()
    tensor = torch.cat(features)
    if len(tensor) != len(rows) or not torch.isfinite(tensor).all():
        raise ValueError("Invalid extracted features")
    output.parent.mkdir(parents=True, exist_ok=True)
    reference_audit = None
    if args.reference_cache:
        reference = torch.load(args.reference_cache, map_location="cpu", weights_only=True)
        if (reference["condition"] != args.condition or reference["rows"] != rows
                or reference["pool_manifest_sha256"] != digest(pool/"manifest.json")
                or reference.get("readout", "global_mean") != "global_mean"):
            raise ValueError("Reference cache identities or feature contract differ")
        regenerated = torch.cat(global_features)
        if not torch.equal(regenerated, reference["features"]):
            difference = float((regenerated-reference["features"]).abs().max())
            raise ValueError(f"Global features changed from the original cache: max_abs={difference}")
        reference_audit = {"sha256": digest(args.reference_cache), "global_max_abs_difference": 0.0}
    record = {"features": tensor, "rows": rows, "condition": args.condition, "data_audit": data_audit,
              "readout": args.readout, "reference_audit": reference_audit,
              "pool_manifest_sha256": digest(pool/"manifest.json"), "config": config,
              "model_sha256": protocol["model"]["sha256"], "model_config": model_config,
              "shared_value_source_sha256": digest(ROOT/"src/vla_rl_mt/shared_value.py"),
              "cache_source_sha256": digest(Path(__file__)), "batch_size": args.batch_size,
              "scope": "frozen native pi0.5, zero adapter, prefix-only value features; development only"}
    temporary = output.with_suffix(".tmp")
    torch.save(record, temporary)
    temporary.replace(output)
    if control_output:
        control = {**record, "features": torch.cat([torch.cat(global_features)]*2, dim=-1),
                   "readout": "global_duplicate"}
        control_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = control_output.with_suffix(".tmp")
        torch.save(control, temporary)
        temporary.replace(control_output)
    print(json.dumps({"condition": args.condition, "shape": list(tensor.shape), "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
