"""Compare actual native training velocity, prefix-cache velocity and Euler paths."""

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from vla_rl_mt.expert_policy import build_model
from vla_rl_mt.joint_value import digest, to_observation, write_json
from vla_rl_mt.policy_update import GuidedPool

ROOT = Path(__file__).resolve().parents[1]


def velocity_error(reference, actual, mask):
    if reference.shape != actual.shape or mask.shape != reference.shape or mask.dtype != torch.bool or not bool(mask.any()):
        raise ValueError("Expected matching velocities and a nonempty physical-action mask")
    full_difference = (actual-reference).float()
    difference = full_difference[mask]
    absolute = difference.abs()
    rmse = float(difference.square().mean().sqrt())
    return {"max_abs": float(absolute.max()), "p99_abs": float(torch.quantile(absolute, 0.99)),
            "rmse": rmse, "all_model_max_abs": float(full_difference.abs().max()),
            "relative_rmse": rmse/max(float(reference.float()[mask].square().mean().sqrt()), 1e-12)}


def joint_velocity(policy, observation, x_t, time):
    from rlinf.models.embodiment.openpi.pi0 import make_attn_mask

    prefix, prefix_mask, prefix_ar = policy.embed_prefix(observation)
    suffix, suffix_mask, suffix_ar, cond = policy.embed_suffix(observation, x_t, time)
    mask = torch.cat((prefix_mask, suffix_mask), dim=1)
    attention = make_attn_mask(mask, torch.cat((prefix_ar, suffix_ar)))
    _, suffix = policy.llm([prefix, suffix], positions=mask.int().cumsum(1)-1, mask=attention,
                          adarms_cond=[None, cond], return_prefix=False)[0]
    return policy.velocity_from_suffix(suffix[:, -policy.action_horizon:])


def actual_training_velocity(policy, observation, actions, noise, time):
    captured = []
    hook = policy.action_out_proj.register_forward_hook(lambda module, args, out: captured.append(out.detach().clone()))
    try:
        loss = policy.compute_loss(observation, actions, train=False, noise=noise, time=time)
    finally:
        hook.remove()
    if len(captured) != 1 or not torch.equal(loss, (captured[0]-(noise-actions)).square()):
        raise ValueError("Captured velocity does not reproduce the actual native FM loss")
    return captured[0]


def euler_joint(policy, observation, noise, steps):
    value, time, dt = noise, 1.0, -1/steps
    while time >= -dt/2:
        t = torch.full((len(noise),), time, device=noise.device, dtype=torch.float32)
        value = value+dt*joint_velocity(policy, observation, value, t)
        time += dt
    return value


def main():
    from rlinf.models.embodiment.openpi.modules.model import (
        _observation_to_dtype,
        preprocess_observation,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("w1", "adapter_bc", "expert_bc"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(args.protocol.read_text())
    base = json.loads((ROOT/plan["base_protocol"]).read_text())
    source = plan["variants"][args.variant]
    path = ROOT/source["path"]
    if digest(path) != source["sha256"]:
        raise ValueError("Pinned diagnostic checkpoint changed")
    pool = GuidedPool(ROOT/base["pool"], base["pool_manifest_sha256"])
    selected = [next(i for i, row in enumerate(pool.rows) if (row["task"], row["split"], row["success"]) == (task, split, success))
                for task in sorted({row["task"] for row in pool.rows})
                for split in ("train", "val") for success in (False, True)]
    model = build_model(OmegaConf.create({"pool_path": str(pool.path), "mode": "bc", "update_scope": "expert_adapter",
                                        "adapter_lr": 3e-4, "expert_lr": 1e-5}))
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_addon(checkpoint["addon"])
    if "expert" in checkpoint:
        model.expert.load_state_dict(checkpoint["expert"], strict=True)
    policy = model.policy
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = {"scope": "Fixed offline numerical diagnosis; no training or rollout", "variant": args.variant,
              "checkpoint_sha256": digest(path), "protocol_sha256": digest(args.protocol),
              "source_sha256": digest(Path(__file__)), "cases": [pool.rows[i] for i in selected],
              "rows": [], "euler": [], "complete": False, "final_test_used": False}
    for precision in plan["precisions"]:
        if precision == "fp32":
            policy.to_bfloat16_for_selected_params("float32")
            policy.embed_dtype = policy.llm.embed_dtype = policy.img.dtype_mm = torch.float32
        for flags in ("training", "inference"):
            model.train(flags == "training")
            model.adapter.requires_grad_(flags == "training")
            model.expert.requires_grad_(flags == "training")
            for start in range(0, len(selected), 2):
                indices = selected[start:start+2]
                samples = [pool.sample(i) for i in indices]
                raw, actions, mask = torch.utils.data.default_collate(samples)
                observation = to_observation(raw, "cuda")
                actions, mask = actions.to("cuda"), mask.to("cuda")
                noise = torch.stack([torch.randn((32, 32), generator=torch.Generator().manual_seed(plan["noise_seed"]+i))
                                     for i in indices]).to("cuda")
                with torch.no_grad():
                    sample_obs = preprocess_observation(observation, train=False)
                    train_obs = _observation_to_dtype(sample_obs, policy.embed_dtype)
                    _, pm_sample, cache_sample = policy.build_prefix_cache(sample_obs)
                    _, pm_train, cache_train = policy.build_prefix_cache(train_obs)
                    for value in plan["flow_times"]:
                        time = torch.full((len(indices),), value, device="cuda")
                        x_t = time[:, None, None]*noise+(1-time[:, None, None])*actions
                        actual = actual_training_velocity(policy, observation, actions, noise, time)
                        joint_train = joint_velocity(policy, train_obs, x_t, time)
                        joint_sample = joint_velocity(policy, sample_obs, x_t, time)
                        cached_train = policy.velocity_from_suffix(policy.run_suffix(train_obs, x_t, time, cache_train, pm_train))
                        cached_sample = policy.velocity_from_suffix(policy.run_suffix(sample_obs, x_t, time, cache_sample, pm_sample))
                        comparisons = {"actual_vs_joint_copy": (actual, joint_train),
                            "joint_vs_cache_same_train_input": (joint_train, cached_train),
                            "joint_vs_cache_same_sample_input": (joint_sample, cached_sample),
                            "actual_train_vs_actual_cache": (actual, cached_sample),
                            "cache_input_cast_only": (cached_train, cached_sample)}
                        report["rows"].append({"precision": precision, "flags": flags, "indices": indices, "time": value,
                            "errors": {name: velocity_error(a, b, mask) for name, (a, b) in comparisons.items()}})
                    if flags == "inference":
                        cached = policy.sample_actions(observation, noise=noise, num_steps=plan["euler_steps"])
                        joint = euler_joint(policy, sample_obs, noise, plan["euler_steps"])
                        report["euler"].append({"precision": precision, "indices": indices,
                            "error": velocity_error(joint, cached, mask)})
                write_json(args.output/"report.json", report)
                print(json.dumps({"precision": precision, "flags": flags, "indices": indices}), flush=True)
    report["fp32_joint_cache_within_locked_limits"] = all(
        row["errors"]["joint_vs_cache_same_sample_input"]["max_abs"] <= plan["fp32_max_abs"]
        and row["errors"]["joint_vs_cache_same_sample_input"]["all_model_max_abs"] <= plan["fp32_max_abs"]
        and row["errors"]["joint_vs_cache_same_sample_input"]["relative_rmse"] <= plan["fp32_relative_rmse"]
        for row in report["rows"] if row["precision"] == "fp32")
    report["actual_joint_copy_exact"] = all(row["errors"]["actual_vs_joint_copy"]["all_model_max_abs"] == 0 for row in report["rows"])
    report["checkpoint_unchanged"] = digest(path) == source["sha256"]
    report["complete"] = True
    write_json(args.output/"report.json", report)
    model.close()


if __name__ == "__main__":
    main()
