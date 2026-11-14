"""Compare exported native inference with the actual official DCP model state."""

import argparse
import json
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from omegaconf import OmegaConf
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

from vla_rl_mt.expert_policy import build_model, register, state_digest
from vla_rl_mt.joint_value import to_observation, write_json
from vla_rl_mt.policy_update import GuidedPool, PolicyPairs

ROOT = Path(__file__).resolve().parents[1]


def main():
    from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy

    register()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    identity = checkpoint["identity"]
    plan = json.loads((ROOT/"configs/multitask/method_m4.json").read_text())
    pool = GuidedPool(ROOT/plan["pool"], plan["pool_manifest_sha256"])
    label = ROOT/plan["label_sources"]["w1" if identity["mode"].startswith("c1") else "w2"]["path"]
    pairs = PolicyPairs(pool, mode=identity["mode"], label_path=label, steps=128, batch_size=4, seed=0)
    batch = torch.utils.data.default_collate([pairs[0], pairs[1]])
    observation = to_observation(batch["native"], "cuda")
    noise = batch["noise"].to("cuda")
    model = build_model(OmegaConf.create({"pool_path": str(pool.path), "mode": identity["mode"],
        "update_scope": identity["update_scope"], "adapter_lr": 3e-4, "expert_lr": 1e-5})).eval()
    model.load_addon(checkpoint["addon"])
    model.expert.load_state_dict(checkpoint["expert"], strict=True)
    model.requires_grad_(False)

    def sample(obs):
        return model.policy.sample_actions(obs, noise=noise, num_steps=10)

    model.forward = sample
    with torch.inference_mode():
        reference = model(observation).detach().cpu()
    # Prove the DCP read actually restores weights, rather than comparing two
    # objects that both still contain the already-loaded export.
    with torch.no_grad():
        next(model.expert.parameters()).view(-1)[0].add_(0.03125)
    with tempfile.TemporaryDirectory(prefix="mt-export-", dir="/tmp") as temporary:
        dist.init_process_group("nccl", init_method=f"file://{temporary}/group", rank=0, world_size=1)
        try:
            wrapped = FSDP(model, auto_wrap_policy=get_fsdp_wrap_policy(model, config=OmegaConf.create({}),
                model_type="mt_native_expert_policy"),
                sharding_strategy=ShardingStrategy.NO_SHARD, use_orig_params=True, device_id=0, sync_module_states=True)
            model.bind_expert()
            options = StateDictOptions(full_state_dict=False, strict=True)
            state = {"fsdp_checkpoint": {"model": get_model_state_dict(wrapped, options=options)}}
            dcp.load(state, checkpoint_id=checkpoint_path.parent/"dcp_checkpoint")
            set_model_state_dict(wrapped, state["fsdp_checkpoint"]["model"], options=options)
            actual_expert = model.expert_state()
            actual_addon = model.addon_state()
            with torch.inference_mode():
                actual = wrapped(observation).detach().cpu()
            report = {"scope": "Actual official DCP state versus exported standalone native inference; no rollout",
                "expert_exact": state_digest(actual_expert) == state_digest(checkpoint["expert"]),
                "addon_exact": all(torch.equal(v, actual_addon[g][k]) for g, values in checkpoint["addon"].items() for k, v in values.items()),
                "native_action_max_abs_delta": float((reference-actual).abs().max()),
                "dcp_model_keys": len(state["fsdp_checkpoint"]["model"]),
                "native_actions_exact": torch.equal(reference, actual)}
            report["passed"] = report["expert_exact"] and report["addon_exact"] and report["native_actions_exact"]
            write_json(args.output/"report.json", report)
            print(json.dumps(report, indent=2))
            if not report["passed"]:
                raise ValueError("DCP/export/native inference differ")
        finally:
            dist.destroy_process_group()
    model.close()


if __name__ == "__main__":
    main()
