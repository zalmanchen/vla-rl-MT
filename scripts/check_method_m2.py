"""Engineering checks on live, audited chunks before the bounded M2 comparison."""

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from vla_rl_mt.joint_value import (
    BoundaryPool,
    JointPairs,
    build_joint_model,
    to_observation,
    write_json,
)
from vla_rl_mt.shared_value import gradient_inventory, isolated_clip, isolated_optimizer

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = OmegaConf.load(ROOT/"configs/multitask/joint_value.yaml")
    pool_path = ROOT/".runtime/datasets/method-m1-v1"
    pool = BoundaryPool(pool_path, config.data.pool_manifest_sha256)
    pairs = JointPairs(pool, 2, 2, 0)
    batch = torch.utils.data.default_collate([pairs[0], pairs[1]])
    raw = batch["observation"]
    data = {key: to_observation(raw[key], "cuda") for key in ("sft", "value")}
    data.update({key: raw[key].to("cuda") for key in ("actions", "action_mask", "noise", "flow_time")})
    targets = batch["target_values"].to("cuda")
    model_config = OmegaConf.create({"pool_path": str(pool_path), "init_seed": 0, "mode": "w1", "value_weight": 0.1,
                                    "adapter_rank": 32, "head_width": 63, "readout": "vision_prompt_concat"})
    model = build_joint_model(model_config)
    initial = model.addon_state()
    report = {"scope": "M2 engineering only; two real success SFT chunks and mixed-outcome value chunks",
              "pool_audit": pool.audit, "sft_rows": [pool.rows[int(i)] for i in pairs.sft[:2]],
              "value_rows": [pool.rows[int(i)] for i in pairs.value[:2]], "modes": {},
              "architecture": model.architecture, "base_excluded_from_state": not any(
                  key.startswith("policy.") for key in model.state_dict()),
              "valid_action_elements": raw["action_mask"].sum((1, 2)).tolist()}

    def update(mode):
        model.mode = mode
        optimizer.zero_grad(set_to_none=True)
        output = model(data, targets)
        output.loss.backward()
        gradients = gradient_inventory(model)
        isolated_clip(model)
        optimizer.step()
        losses = model.step_metrics.pop()
        return {"loss": float(output.loss.detach()), **losses, "gradients": gradients}

    states = {}
    for mode in ("w0", "w1", "w2"):
        model.load_addon(initial)
        optimizer = isolated_optimizer(model, 3e-4)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        item = update(mode)
        torch.cuda.synchronize()
        item.update(seconds=time.monotonic()-started, peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
        states[mode] = model.addon_state()
        report["modes"][mode] = item
        write_json(args.output/"report.json", report)
        print(json.dumps({"mode": mode, **{key: item[key] for key in ("loss", "seconds", "peak_allocated_gib")}}), flush=True)
    report["w0_w1_adapter_exact"] = all(torch.equal(states["w0"]["adapter"][key], value)
                                         for key, value in states["w1"]["adapter"].items())
    report["w1_w2_adapter_max_abs"] = max(float((value-states["w1"]["adapter"][key]).abs().max())
                                           for key, value in states["w2"]["adapter"].items())
    if not report["w0_w1_adapter_exact"] or report["w1_w2_adapter_max_abs"] == 0:
        raise ValueError("The real-chunk actor intervention is not isolated")
    for detach in (True, False):
        model.load_addon(initial)
        model.zero_grad(set_to_none=True)
        (model.value(data["value"], detach_shared=detach)-targets).square().mean().backward()
        gradients = gradient_inventory(model)
        report[f"value_only_detach_{detach}"] = gradients
        if (gradients["adapter"]["gradient_l2"] > 0) == detach or gradients["frozen_policy"]["gradient_tensors"]:
            raise ValueError("Value-only gradient route is incorrect")
    # Restore both addon and optimizer; identical next live-batch update required.
    model.load_addon(states["w2"])
    checkpoint = {"addon": model.addon_state(), "optimizer": copy.deepcopy(optimizer.state_dict())}
    torch.save(checkpoint, args.output/"probe-checkpoint.pt")
    update("w2")
    expected = model.addon_state()
    checkpoint = torch.load(args.output/"probe-checkpoint.pt", weights_only=True)
    model.load_addon(checkpoint["addon"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    update("w2")
    actual = model.addon_state()
    report["checkpoint_next_update_exact"] = all(torch.equal(expected[name][key], value)
        for name, values in actual.items() for key, value in values.items())
    if not report["checkpoint_next_update_exact"]:
        raise ValueError("M2 checkpoint does not replay the same next update")
    report["passed"] = True
    write_json(args.output/"report.json", report)
    model.close()


if __name__ == "__main__":
    main()
