"""Real expert-scope probe: unchanged forward, exact ownership and replay."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from vla_rl_mt.expert_policy import ExpertPolicyUpdate, clip_actor, inventory
from vla_rl_mt.joint_value import to_observation, write_json
from vla_rl_mt.policy_update import GuidedPool, PolicyPairs, build_model

ROOT = Path(__file__).resolve().parents[1]


def frozen_digest(model):
    result = hashlib.sha256()
    expert_ids = {id(param) for param in model.expert.parameters()}
    for name, param in model.policy.named_parameters():
        if id(param) not in expert_ids:
            result.update(name.encode())
            result.update(param.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def main():
    from rlinf.hybrid_engines.fsdp.optim import FP32MasterAdamW

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((ROOT/"configs/multitask/method_m4.json").read_text())
    pool = GuidedPool(ROOT/plan["pool"], plan["pool_manifest_sha256"])
    pairs = PolicyPairs(pool, mode="c1_weighted", label_path=ROOT/plan["label_sources"]["w1"]["path"],
                        steps=128, batch_size=4, seed=0)
    indices = [next(i for i, row in enumerate(pairs.draws) if bool(pairs.labels[int(row)])),
               next(i for i, row in enumerate(pairs.draws) if not bool(pairs.labels[int(row)]))]
    batch = torch.utils.data.default_collate([pairs[i] for i in indices])
    raw = batch.pop("native")
    data = {key: value.to("cuda") for key, value in batch.items()}
    data.update(obs=to_observation(raw, "cuda"),
                positive_tokens=raw["tokenized_positive_guidance_prompt"].to("cuda"),
                positive_mask=raw["tokenized_positive_guidance_prompt_mask"].to("cuda"))
    data["routing_random"] = torch.ones(2, device="cuda")
    old = build_model(OmegaConf.create({"pool_path": str(pool.path), "mode": "c1_weighted"}))
    initial = old.addon_state()
    with torch.no_grad():
        reference = old(data).loss.detach()
        repeated_reference = old(data).loss.detach()
    policy = old.policy
    old.close()
    model = ExpertPolicyUpdate(policy, "c1_weighted", adapter_lr=3e-4, expert_lr=1e-5,
                              update_scope="adapter_only", **old.architecture).to("cuda").train()
    model.load_addon(initial)
    del old
    before = frozen_digest(model)
    expert_initial = model.expert_state()
    with torch.no_grad():
        adapter_only = model(data).loss.detach()
    model.expert.requires_grad_(False)
    with torch.no_grad():
        frozen_expert = model(data).loss.detach()
    model.expert.requires_grad_(True)
    model.update_scope = "expert_adapter"
    with torch.no_grad():
        actual = model(data).loss.detach()
    report = {"scope": "Action-expert update engineering, two real positive/negative train decisions",
              "plain_weighted_fm_abs_delta": float((actual-reference).abs()), "updates": {}}
    report.update(reference_loss=float(reference), repeated_reference_loss=float(repeated_reference),
                  adapter_only_loss=float(adapter_only), expert_loss=float(actual),
                  frozen_expert_loss=float(frozen_expert),
                  matched_scope_fm_abs_delta=float((actual-adapter_only).abs()),
                  frozen_expert_native_fm_abs_delta=float((frozen_expert-reference).abs()))
    write_json(args.output/"forward-audit.json", report)
    if report["matched_scope_fm_abs_delta"] != 0 or report["frozen_expert_native_fm_abs_delta"] != 0:
        raise ValueError("Scope controls or frozen native reference changed computation")
    model.step_metrics.clear()
    optimizer = FP32MasterAdamW([{"params": model.adapter.parameters(), "lr": 3e-4},
                                {"params": model.expert.parameters(), "lr": 1e-5}],
                               betas=(0.9, 0.95), eps=1e-8, weight_decay=0)

    def update():
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        output.loss.backward()
        groups = inventory(model)
        if (groups["frozen_vlm"]["gradient_tensors"] or groups["value_head"]["gradient_tensors"]
                or groups["action_expert"]["gradient_l2"] <= 0 or groups["adapter"]["gradient_l2"] <= 0
                or not all(row["finite"] for row in groups.values())):
            raise ValueError("Wrong real expert-scope gradient route")
        clip_actor(model)
        optimizer.step()
        model.step_metrics.clear()
        return {"loss": float(output.loss.detach()), "gradients": groups}

    report["updates"]["first"] = update()
    saved = {"addon": model.addon_state(), "expert": model.expert_state(),
             "optimizer": copy.deepcopy(optimizer.state_dict())}
    # Offload the replay copy only; the live optimizer retains original dtypes.
    for state in saved["optimizer"]["state"].values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.cpu()
    update()
    expected_addon, expected_expert = model.addon_state(), model.expert_state()
    model.load_addon(saved["addon"])
    model.expert.load_state_dict(saved["expert"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    report["updates"]["replayed"] = update()
    actual_expert = model.expert_state()
    report["next_update_reload_exact"] = all(torch.equal(value, actual_expert[key])
        for key, value in expected_expert.items())
    actual_addon = model.addon_state()
    report["next_update_reload_exact"] &= all(torch.equal(value, actual_addon[group][key])
        for group, values in expected_addon.items() for key, value in values.items())
    report["expert_changed"] = any(not torch.equal(value, expected_expert[key]) for key, value in expert_initial.items())
    report["frozen_vlm_exact"] = frozen_digest(model) == before
    report["head_exact"] = all(torch.equal(value, actual_addon["value_head"][key]) for key, value in initial["value_head"].items())
    report["peak_allocated_gib"] = torch.cuda.max_memory_allocated()/1024**3
    report["passed"] = all(report[key] for key in ("next_update_reload_exact", "expert_changed", "frozen_vlm_exact", "head_exact"))
    write_json(args.output/"report.json", report)
    if not report["passed"]:
        raise ValueError("Expert update/replay/frozen preservation failed")
    model.close()
    print(json.dumps({key: report[key] for key in ("plain_weighted_fm_abs_delta", "peak_allocated_gib", "passed")}), flush=True)


if __name__ == "__main__":
    main()
