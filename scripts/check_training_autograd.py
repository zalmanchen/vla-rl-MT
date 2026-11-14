"""Check real grad-enabled native training against the no-grad path audit."""

import argparse
import json
from pathlib import Path

import torch
from check_policy_paths import actual_training_velocity, velocity_error
from omegaconf import OmegaConf

from vla_rl_mt.expert_policy import build_model
from vla_rl_mt.joint_value import digest, to_observation, write_json
from vla_rl_mt.policy_update import GuidedPool

ROOT = Path(__file__).resolve().parents[1]


def main():
    from rlinf.models.embodiment.openpi.modules.model import (
        _observation_to_dtype,
        preprocess_observation,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("w1", "adapter_bc", "expert_bc"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = ROOT/"configs/multitask/native_path_check.json"
    plan = json.loads(protocol.read_text())
    base = json.loads((ROOT/"configs/multitask/method_m4.json").read_text())
    source = plan["variants"][args.variant]
    path = ROOT/source["path"]
    if digest(path) != source["sha256"]:
        raise ValueError("Checkpoint changed")
    pool = GuidedPool(ROOT/base["pool"], base["pool_manifest_sha256"])
    selected = []
    for task in sorted({row["task"] for row in pool.rows}):
        for split in ("train", "val"):
            for success in (False, True):
                selected.append(next(i for i, row in enumerate(pool.rows)
                    if row["task"] == task and row["split"] == split and row["success"] == success))
    model = build_model(OmegaConf.create({"pool_path": str(pool.path), "mode": "bc",
        "update_scope": "expert_adapter", "adapter_lr": 3e-4, "expert_lr": 1e-5})).train()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_addon(checkpoint["addon"])
    if args.variant != "w1":
        model.expert.load_state_dict(checkpoint["expert"], strict=True)
    policy = model.policy
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = {"scope": "Actual grad-enabled compute_loss; original activation checkpointing enabled; no backward/update",
              "variant": args.variant, "checkpoint_sha256": digest(path), "protocol_sha256": digest(protocol),
              "source_sha256": digest(Path(__file__)), "cases": [pool.rows[i] for i in selected],
              "rows": [], "complete": False, "final_test_used": False}
    for precision in plan["precisions"]:
        if precision == "fp32":
            policy.to_bfloat16_for_selected_params("float32")
            policy.embed_dtype = policy.llm.embed_dtype = policy.img.dtype_mm = torch.float32
        for start in range(0, len(selected), 2):
            indices = selected[start:start+2]
            raw, actions, mask = torch.utils.data.default_collate([pool.sample(i) for i in indices])
            observation = to_observation(raw, "cuda")
            actions, mask = actions.to("cuda"), mask.to("cuda")
            noise = torch.stack([torch.randn((32, 32), generator=torch.Generator().manual_seed(plan["noise_seed"]+i))
                                 for i in indices]).to("cuda")
            with torch.no_grad():
                obs = _observation_to_dtype(preprocess_observation(observation, train=False), policy.embed_dtype)
                _, prefix_mask, cache = policy.build_prefix_cache(obs)
            for value in plan["flow_times"]:
                time = torch.full((len(indices),), value, device="cuda")
                with torch.enable_grad():
                    actual = actual_training_velocity(policy, observation, actions, noise, time)
                with torch.no_grad():
                    no_grad = actual_training_velocity(policy, observation, actions, noise, time)
                    x_t = time[:, None, None]*noise+(1-time[:, None, None])*actions
                    cached = policy.velocity_from_suffix(policy.run_suffix(obs, x_t, time, cache, prefix_mask))
                    errors = {"grad_vs_no_grad": velocity_error(actual, no_grad, mask),
                              "grad_vs_cache": velocity_error(actual, cached, mask)}
                report["rows"].append({"precision": precision, "indices": indices, "time": value, "errors": errors})
            write_json(args.output/"report.json", report)
    report["fp32_within_locked_limits"] = all(
        error["all_model_max_abs"] <= plan["fp32_max_abs"] and error["relative_rmse"] <= plan["fp32_relative_rmse"]
        for row in report["rows"] if row["precision"] == "fp32" for error in row["errors"].values())
    report["checkpoint_unchanged"] = digest(path) == source["sha256"]
    report["complete"] = True
    write_json(args.output/"report.json", report)
    model.close()
    print(json.dumps({"variant": args.variant, "fp32_within_locked_limits": report["fp32_within_locked_limits"]}))


if __name__ == "__main__":
    main()
