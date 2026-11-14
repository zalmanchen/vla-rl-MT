"""Live native-policy bridge: token parity, masks, gradients and next-step replay."""

import argparse
import copy
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from vla_rl_mt.joint_value import to_observation, write_json
from vla_rl_mt.policy_update import GuidedPool, PolicyPairs, build_model
from vla_rl_mt.shared_value import gradient_inventory, isolated_clip, isolated_optimizer

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((ROOT/"configs/multitask/method_m4.json").read_text())
    pool = GuidedPool(ROOT/plan["pool"], plan["pool_manifest_sha256"])
    label_path = ROOT/".runtime/runs/method/m2-joint-v1/actor-w1-v2/value-label-audit.json"
    pairs = PolicyPairs(pool, mode="c1_cond", label_path=label_path, steps=128, batch_size=4, seed=0)
    selected = [next(i for i, draw in enumerate(pairs.draws) if bool(pairs.labels[int(draw)])),
                next(i for i, draw in enumerate(pairs.draws) if not bool(pairs.labels[int(draw)]))]
    batch = torch.utils.data.default_collate([pairs[i] for i in selected])
    raw = batch["native"]
    data = {key: value.to("cuda") for key, value in batch.items() if key != "native"}
    data.update(obs=to_observation(copy.deepcopy(raw), "cuda"),
                positive_tokens=raw["tokenized_positive_guidance_prompt"].to("cuda"),
                positive_mask=raw["tokenized_positive_guidance_prompt_mask"].to("cuda"))
    data["routing_random"] = torch.ones(2, device="cuda")  # Both branches represented in this interface check.
    model = build_model(OmegaConf.create({"pool_path": str(pool.path), "mode": "bc"}))
    initial = model.addon_state()
    row = pool.rows[int(pairs.draws[selected[0]])]
    with h5py.File(pool.path/row["episode"]) as episode:
        index = row["observation_index"]
        images = {key: cv2.cvtColor(cv2.imdecode(episode[f"images/{key}"][index], cv2.IMREAD_COLOR),
                                   cv2.COLOR_BGR2RGB) for key in episode["images"]}
        env = {"main_images": images["cam_high"][None],
               "wrist_images": np.stack([images["cam_left_wrist"], images["cam_right_wrist"]])[None],
               "states": episode["state"][index][None], "task_descriptions": [str(episode.attrs["instruction"])]}
    report = {"scope": "M4 real-interface engineering, not reward/advantage correctness", "training_rows": selected,
              "positive_and_negative_present": True, "token_parity": {}, "zero_update_actions": {}}
    model.eval()
    samples = [pairs[i] for i in selected]
    actions = {}
    for mode, suffix, field in (("plain", "", "tokenized_prompt"),
                                ("positive", "\nAdvantage: positive", "tokenized_positive_guidance_prompt"),
                                ("negative", "\nAdvantage: negative", "tokenized_negative_guidance_prompt"),
                                ("neutral", "\nContext: ordinary", None)):
        changed = copy.deepcopy(env)
        changed["task_descriptions"] = [env["task_descriptions"][0]+suffix]
        observation = model.policy.env_obs_to_observation(changed)
        if field:
            exact = torch.equal(observation.tokenized_prompt.cpu()[0], torch.as_tensor(samples[0]["native"][field]))
            report["token_parity"][mode] = bool(exact)
            if not exact:
                raise ValueError(f"Train/deploy token mismatch: {mode}")
        torch.manual_seed(911)
        with torch.inference_mode():
            result, _ = model.policy.predict_action_batch(changed, mode="eval")
        actions[mode] = torch.as_tensor(result).detach().cpu()
        report["zero_update_actions"][mode] = {"max_abs_delta_vs_plain": float((actions[mode]-actions["plain"]).abs().max()),
                                               "l2_delta_vs_plain": float((actions[mode]-actions["plain"]).square().sum().sqrt())}
    model.train()
    with torch.no_grad():
        reference = model.sft_loss(data["obs"], data["actions"], noise=data["noise"], flow_time=data["flow_time"],
                                   action_mask=data["action_mask"])
        actual = model(data).loss
    report["plain_native_fm_loss_abs_delta"] = float((reference-actual).abs())
    if report["plain_native_fm_loss_abs_delta"] != 0:
        raise ValueError("Native plain FM changed at bridge")
    model.step_metrics.clear()

    def update(mode):
        model.mode = mode
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        output.loss.backward()
        inventory = gradient_inventory(model)
        if inventory["frozen_policy"]["gradient_tensors"] or inventory["value_head"]["gradient_tensors"]:
            raise ValueError("Frozen base or critic head received gradients")
        if inventory["adapter"]["gradient_l2"] <= 0:
            raise ValueError("Actor adapter did not receive the real update")
        isolated_clip(model)
        optimizer.step()
        model.step_metrics.clear()
        return {"loss": float(output.loss.detach()), "gradients": inventory}

    report["updates"] = {}
    for mode in ("c1_cond", "c1_weighted"):
        model.load_addon(initial)
        optimizer = isolated_optimizer(model, 3e-4)
        report["updates"][mode] = update(mode)
    checkpoint = {"addon": model.addon_state(), "optimizer": copy.deepcopy(optimizer.state_dict())}
    torch.save(checkpoint, args.output/"probe-checkpoint.pt")
    update("c1_weighted")
    expected = model.addon_state()
    checkpoint = torch.load(args.output/"probe-checkpoint.pt", weights_only=True, map_location="cpu")
    model.load_addon(checkpoint["addon"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    update("c1_weighted")
    actual = model.addon_state()
    report["next_update_reload_exact"] = all(torch.equal(expected[name][key], value)
        for name, values in actual.items() for key, value in values.items())
    report["head_unchanged"] = all(torch.equal(actual["value_head"][key], value) for key, value in initial["value_head"].items())
    if not report["next_update_reload_exact"] or not report["head_unchanged"]:
        raise ValueError("M4 next-step replay or frozen critic preservation failed")
    report["passed"] = True
    write_json(args.output/"report.json", report)
    model.close()
    print(json.dumps({key: report[key] for key in ("token_parity", "plain_native_fm_loss_abs_delta", "next_update_reload_exact", "passed")}), flush=True)


if __name__ == "__main__":
    main()
