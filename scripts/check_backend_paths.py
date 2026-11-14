"""Independent native/official-CFG math diagnosis on real RoboTwin inputs.

Factory BF16 is reported first. FP32 stages share exactly mapped core weights;
erf-GELU/RoPE changes are disposable diagnostics, never production patches.
"""

import argparse
import json
import sys
import types
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from check_policy_paths import velocity_error

from vla_rl_mt.joint_value import digest, to_observation, write_json
from vla_rl_mt.policy_update import GuidedPool

ROOT = Path(__file__).resolve().parents[1]


def cfg_velocity(model, observation, actions, noise, time):
    fields = model._preprocess_observation(observation, train=False)
    captured = []
    hook = model.action_out_proj.register_forward_hook(lambda module, args, out: captured.append(out.detach().clone()))
    try:
        model._compute_flow_losses(fields[0], fields[1], fields[-1], actions, fields[2], fields[3],
                                   actions.device, time=time, noise=noise)
    finally:
        hook.remove()
    if len(captured) != 1:
        raise ValueError("CFG did not produce one velocity")
    return captured[0]


def cfg_cache(model, observation):
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    # The official sampler forces eager attention before its prefix-only HF
    # call. Mirror that here; the joint two-expert path uses manual attention.
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
    fields = model._preprocess_observation(observation, train=False)
    prefix, mask, ar = model.embed_prefix(fields[0], fields[1], fields[2], fields[3])
    _, cache = model.paligemma_with_expert.forward(attention_mask=model._prepare_attention_masks_4d(make_att_2d_masks(mask, ar)),
        position_ids=mask.int().cumsum(1)-1, past_key_values=None, inputs_embeds=[prefix, None], use_cache=True)
    return fields[-1], mask, cache


def main():
    sys.path.insert(0, str(ROOT/"scripts"))
    from evaluate_multitask import load_model
    from rlinf.models.embodiment.openpi.modules.model import (
        _observation_to_dtype,
        preprocess_observation,
    )
    from rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model import Observation
    from rlinf.utils.ckpt_convertor.openpi.openpi_to_openpi_pytorch import (
        ACTION_EXPERT_LM_HEAD,
        new_to_old_state_dict,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.protocol.read_text())
    base = json.loads((ROOT/config["base_protocol"]).read_text())
    pool = GuidedPool(ROOT/base["pool"], base["pool_manifest_sha256"])
    model_file = ROOT/pool.manifest["protocol"]["model"]["path"]/"model.safetensors"
    if digest(model_file) != pool.manifest["protocol"]["model"]["sha256"]:
        raise ValueError("Pinned base model changed")
    selected = [next(i for i, row in enumerate(pool.rows) if (row["task"], row["split"], row["success"]) == (task, "val", outcome))
                for task in sorted({row["task"] for row in pool.rows}) for outcome in (False, True)]
    native, _ = load_model("native", pool.manifest["protocol"])
    cfg, _ = load_model("cfg", pool.manifest["protocol"])
    native.requires_grad_(False)
    cfg.requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    mapped = new_to_old_state_dict({key: value.detach().cpu() for key, value in native.state_dict().items()})
    embedding = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    mapped[embedding] = mapped["paligemma_with_expert.paligemma.lm_head.weight"]
    target = cfg.state_dict()
    if set(target)-set(mapped) != {ACTION_EXPERT_LM_HEAD} or set(mapped)-set(target):
        raise ValueError("Unexpected core weight-layout difference")
    mismatched = [key for key, value in mapped.items() if not torch.equal(value, target[key].detach().cpu())]
    del target, mapped
    report = {"scope": "Four real validation observations; base Motus only, no adapters/training/rollouts",
        "factory_core_weight_mismatches": mismatched, "unused_head_excluded": ACTION_EXPERT_LM_HEAD,
        "cases": [pool.rows[i] for i in selected], "input_checks": [], "rows": [], "complete": False,
        "production_modified": False, "source_sha256": digest(Path(__file__)), "protocol_sha256": digest(args.protocol)}
    inputs = []
    for index in selected:
        row = pool.rows[index]
        with h5py.File(pool.path/row["episode"]) as episode:
            frame = row["observation_index"]
            images = {camera: cv2.cvtColor(cv2.imdecode(episode[f"images/{camera}"][frame], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                      for camera in episode["images"]}
            env = {"main_images": images["cam_high"][None],
                   "wrist_images": np.stack([images["cam_left_wrist"], images["cam_right_wrist"]])[None],
                   "states": episode["state"][frame][None], "task_descriptions": [str(episode.attrs["instruction"])]}
        raw, actions, mask = pool.sample(index)
        raw = torch.utils.data.default_collate([raw])
        observation = native.env_obs_to_observation(env)
        cfg_observation = Observation.from_dict(cfg.precision_processor(cfg.input_transform(cfg.obs_processor(env), transpose=False)))
        pool_observation = to_observation(raw, "cuda")
        image_error = max(float((observation.images[k]-cfg_observation.images[k].permute(0, 2, 3, 1)).abs().max())
                          for k in observation.images)
        check = {"index": index, "images_max_abs": image_error,
            "state_max_abs": float((observation.state-cfg_observation.state).abs().max()),
            "tokens_exact": torch.equal(observation.tokenized_prompt, cfg_observation.tokenized_prompt),
            "prompt_mask_exact": torch.equal(observation.tokenized_prompt_mask, cfg_observation.tokenized_prompt_mask),
            "pool_tokens_exact": torch.equal(observation.tokenized_prompt, pool_observation.tokenized_prompt)}
        if (image_error > 2e-7 or check["state_max_abs"] > 2e-7
                or not all(check[key] for key in ("tokens_exact", "prompt_mask_exact", "pool_tokens_exact"))):
            raise ValueError(f"Backend input mismatch: {check}")
        report["input_checks"].append(check)
        noise = torch.randn((1, 32, 32), generator=torch.Generator().manual_seed(config["noise_seed"]+index)).to("cuda")
        inputs.append((index, observation, cfg_observation, actions[None].to("cuda"), mask[None].to("cuda"), noise))
    for stage in ("factory_bf16", "common_fp32", "erf_gelu_diagnostic", "erf_gelu_native_rope_diagnostic"):
        if stage == "common_fp32":
            native.to_bfloat16_for_selected_params("float32")
            native.embed_dtype = native.llm.embed_dtype = native.img.dtype_mm = torch.float32
            cfg.float()
            common = new_to_old_state_dict({key: value.detach().cpu() for key, value in native.state_dict().items()})
            common[embedding] = common["paligemma_with_expert.paligemma.lm_head.weight"]
            missing, unexpected = cfg.load_state_dict(common, strict=False)
            if missing != [ACTION_EXPERT_LM_HEAD] or unexpected:
                raise ValueError("Common FP32 weight transfer failed")
            del common
        if stage == "erf_gelu_diagnostic":
            count = 0
            for module in cfg.modules():
                if type(module).__name__ == "SiglipMLP":
                    module.activation_fn = torch.nn.GELU(approximate="none")
                    count += 1
                elif type(module).__name__ == "GemmaMLP":
                    module.act_fn = torch.nn.GELU(approximate="none")
                    count += 1
            report["temporary_gelu_modules"] = count
        if stage == "erf_gelu_native_rope_diagnostic":
            def rotary(self, x, position_ids):
                exponents = (2/self.config.head_dim)*torch.arange(self.config.head_dim//2, dtype=torch.float32, device=x.device)
                radians = position_ids[..., None].float()/(self.config.rope_theta**exponents)[None, None, :]
                embedding = torch.cat((radians, radians), dim=-1)
                return embedding.cos().to(x.dtype), embedding.sin().to(x.dtype)

            for module in (cfg.paligemma_with_expert.paligemma.language_model.rotary_emb,
                           cfg.paligemma_with_expert.gemma_expert.model.rotary_emb):
                module.forward = types.MethodType(rotary, module)
        with torch.no_grad():
            for index, observation, cfg_observation, actions, mask, noise in inputs:
                train_obs = _observation_to_dtype(preprocess_observation(observation, train=False), native.embed_dtype)
                _, native_mask, native_cache = native.build_prefix_cache(train_obs)
                state, cfg_mask, cache = cfg_cache(cfg, cfg_observation)
                for value in config["flow_times"]:
                    time = torch.full((1,), value, device="cuda")
                    x_t = time[:, None, None]*noise+(1-time[:, None, None])*actions
                    native_velocity = native.velocity_from_suffix(native.run_suffix(train_obs, x_t, time, native_cache, native_mask))
                    cfg_train = cfg_velocity(cfg, cfg_observation, actions, noise, time)
                    cfg_sample = cfg.denoise_step(state, cfg_mask, cache, x_t, time)
                    report["rows"].append({"stage": stage, "index": index, "time": value,
                        "native_vs_cfg": velocity_error(native_velocity, cfg_sample, mask),
                        "cfg_joint_vs_cache": velocity_error(cfg_train, cfg_sample, mask)})
                native_actions = native.sample_actions(observation, noise=noise, num_steps=config["euler_steps"])
                cfg_actions = cfg.sample_actions(cfg_observation, noise=noise)["actions"]
                native_env = native.decode_actions(native_actions, observation.state)
                cfg_env = cfg.output_transform({"actions": cfg_actions, "state": cfg_observation.state})["actions"].to("cuda")
                report["rows"].append({"stage": stage, "index": index,
                    "actions": velocity_error(native_actions, cfg_actions, mask),
                    "environment_actions": velocity_error(native_env, cfg_env, torch.ones_like(native_env, dtype=torch.bool))})
        write_json(args.output/"report.json", report)
        print(json.dumps({"stage": stage, "rows": len(report["rows"])}), flush=True)
    report["complete"] = True
    write_json(args.output/"report.json", report)


if __name__ == "__main__":
    main()
