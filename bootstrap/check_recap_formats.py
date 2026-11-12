"""Compare native SFT and CFG-unconditional inference on released observations.

Both implementations receive identical transformed observations and Gaussian
noise. This checks the format bridge; it does not measure policy success.
"""

import argparse
import gc
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--cfg", required=True, type=Path)
    parser.add_argument("--norm-stats", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--native-evidence", type=Path,
                        help="Engineering diagnostic only: reuse an explicitly supplied native evidence file")
    args = parser.parse_args()
    import av
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from omegaconf import OmegaConf
    from rlinf.models.embodiment.openpi import get_model as native_model
    from rlinf.models.embodiment.openpi_cfg import get_model as cfg_model
    from rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model import Observation

    root = Path(__file__).resolve().parents[1]
    data = root / "datasets/recap-official/libero10_task0_sft"
    state = pq.read_table(data / "data/chunk-000/episode_000015.parquet", columns=["state"])["state"][0].as_py()
    images = {}
    for key in ("image", "wrist_image"):
        with av.open(str(data / f"videos/chunk-000/{key}/episode_000015.mp4")) as container:
            images[key] = next(container.decode(video=0)).to_ndarray(format="rgb24")
    env_obs = {"main_images": torch.from_numpy(images["image"])[None],
               "wrist_images": torch.from_numpy(images["wrist_image"])[None],
               "states": torch.tensor([state]),
               "task_descriptions": ["put both the alphabet soup and the tomato sauce in the basket"]}
    base = {"model_type": "openpi", "precision": "fp32", "pi05": True,
            "action_dim": 7, "num_action_chunks": 10, "num_steps": 5,
            "openpi": {"task": "eval", "config_name": "pi05_libero", "train_expert_only": False,
                       "model_action_dim": 32, "paligemma_variant": "gemma_2b",
                       "action_expert_variant": "gemma_300m", "discrete_state_input": False,
                       "max_token_len": 200, "torch_compile": False, "embed_dtype": "fp32",
                       "num_steps": 5, "action_chunk": 10, "action_horizon": 10,
                       "guidance_type": "no_guide"},
            "openpi_data": {"norm_stats_path": str(args.norm_stats)}}
    torch.manual_seed(0)
    noise = torch.randn(1, 10, 32, device="cuda")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.create({**base, "model_path": str(args.native)})
    if args.native_evidence:
        reference = torch.load(args.native_evidence, map_location="cpu", weights_only=True)
        native_raw, native_actions, common = reference["native_raw"], reference["native_actions"], reference["inputs"]
        noise = reference["noise"].to("cuda")
    else:
        native = native_model(cfg).to("cuda").eval()
        # Native construction enables TF32 globally. Disable it for this
        # numerical diagnostic; production evaluation retains upstream dtypes.
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        native_obs = native.env_obs_to_observation(env_obs)
        with torch.inference_mode():
            native_raw = native.sample_actions(native_obs, num_steps=5, noise=noise.clone()).cpu()
            native_actions = native.decode_actions(native_raw.to("cuda"), native_obs.state).cpu()
        common = {"state": native_obs.state.cpu(), "tokens": native_obs.tokenized_prompt.cpu(),
                  "mask": native_obs.tokenized_prompt_mask.cpu(),
                  "images": {key: value.cpu() for key, value in native_obs.images.items()}}
        torch.save({"native_raw": native_raw, "native_actions": native_actions, "inputs": common,
                    "noise": noise.cpu()}, args.report.with_suffix(".native.pt"))
        del native_obs, native
        gc.collect()
        torch.cuda.empty_cache()
    cfg.model_path = str(args.cfg)
    legacy = cfg_model(cfg).float()
    # The public CFG factory unconditionally quantizes its backbone to BF16.
    # Reload the verified FP32 export to isolate layout/math parity from that
    # lossy storage conversion. Production evaluation keeps the public dtype.
    legacy.load_state_dict(torch.load(args.cfg / "model_state_dict/full_weights.pt",
                                     map_location="cpu", mmap=True, weights_only=True), strict=True)
    legacy = legacy.to("cuda").eval()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    processed = legacy.precision_processor(legacy.input_transform(legacy.obs_processor(env_obs), transpose=False))
    legacy_obs = Observation.from_dict(processed)
    for key, value in common["images"].items():
        # The native implementation consumes NHWC; the legacy HF backbone NCHW.
        legacy_image = legacy_obs.images[key].cpu().permute(0, 2, 3, 1)
        # Elementwise normalization follows different operation orders.
        torch.testing.assert_close(value, legacy_image, rtol=0, atol=2e-7)
    torch.testing.assert_close(common["state"], legacy_obs.state.float().cpu(), rtol=0, atol=2e-7)
    torch.testing.assert_close(common["tokens"], legacy_obs.tokenized_prompt.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(common["mask"], legacy_obs.tokenized_prompt_mask.cpu(), rtol=0, atol=0)
    with torch.inference_mode():
        legacy_raw = legacy.sample_actions(legacy_obs, noise=noise.clone())["actions"]
        legacy_actions = legacy.output_transform({"actions": legacy_raw, "state": legacy_obs.state})["actions"]
    legacy_raw = legacy_raw.cpu()
    legacy_actions = torch.as_tensor(np.asarray(legacy_actions)).cpu()
    raw_delta = (native_raw - legacy_raw).abs()
    action_delta = (native_actions - legacy_actions).abs()
    equivalent = torch.allclose(native_raw, legacy_raw, rtol=0.0001, atol=0.0001)
    report = {"status": "equivalent" if equivalent else "unexplained_difference",
              "numerically_equivalent": equivalent, "precision": "fp32", "tf32": False,
              "observations": "released SFT episode 15 frame 0, both cameras and proprioception",
              "matched_inputs_within_atol": 2e-7, "identical_tokens_and_noise": True, "noise_seed": 0, "solver_steps": 5,
              "raw_max_abs": raw_delta.max().item(), "raw_rmse": raw_delta.square().mean().sqrt().item(),
              "env_actions_max_abs": action_delta.max().item(),
              "tolerance": {"rtol": 0.0001, "atol": 0.0001},
              "scope": "one-observation numerical conversion check; not closed-loop performance"}
    if args.native_evidence:
        import hashlib

        report["reused_native_evidence"] = {"path": str(args.native_evidence),
                                            "sha256": hashlib.sha256(args.native_evidence.read_bytes()).hexdigest(),
                                            "scope": "explicit engineering reference; formal runs compute fresh native outputs"}
    if not equivalent:
        # Pinned native SigLIP and Gemma use erf GELU; the legacy backend uses
        # tanh GELU. Test that explanation in this disposable diagnostic model.
        # No production module, checkpoint or source file is changed.
        count = 0
        for module in legacy.modules():
            if type(module).__name__ == "SiglipMLP":
                module.activation_fn = torch.nn.GELU(approximate="none")
                count += 1
            elif type(module).__name__ == "GemmaMLP":
                module.act_fn = torch.nn.GELU(approximate="none")
                count += 1
        with torch.inference_mode():
            matched = legacy.sample_actions(legacy_obs, noise=noise.clone())["actions"].cpu()
        delta = (native_raw - matched).abs()
        explained = torch.allclose(native_raw, matched, rtol=0.0001, atol=0.0001)
        report["activation_diagnostic"] = {"modules": count, "legacy_gelu_temporarily": "erf",
                                            "max_abs": delta.max().item(),
                                            "rmse": delta.square().mean().sqrt().item(),
                                            "explains_difference": explained,
                                            "production_modified": False}
        if not explained:
            import types

            def native_rotary(self, x, position_ids):
                dim = self.config.head_dim
                exponents = (2.0 / dim) * torch.arange(dim // 2, dtype=torch.float32, device=x.device)
                timescale = self.config.rope_theta**exponents
                radians = position_ids[..., None].float() / timescale[None, None, :]
                emb = torch.cat((radians, radians), dim=-1)
                return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

            for rotary in (legacy.paligemma_with_expert.paligemma.language_model.rotary_emb,
                           legacy.paligemma_with_expert.gemma_expert.model.rotary_emb):
                rotary.forward = types.MethodType(native_rotary, rotary)
            with torch.inference_mode():
                matched = legacy.sample_actions(legacy_obs, noise=noise.clone())["actions"].cpu()
            delta = (native_raw - matched).abs()
            explained = torch.allclose(native_raw, matched, rtol=0.0001, atol=0.0001)
            report["gelu_and_rope_diagnostic"] = {
                "legacy_temporarily": "erf GELU plus native divide-based RoPE frequency computation",
                "max_abs": delta.max().item(), "rmse": delta.square().mean().sqrt().item(),
                "explains_difference": explained, "production_modified": False}
        if explained:
            report["status"] = "known_upstream_forward_difference"
            report["scope"] = ("Not numerically equivalent. Pinned upstream forward mismatch diagnosed; "
                               "compare unmodified CFG unconditional baseline against CFG post-training, "
                               "and report native baseline separately.")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] == "unexplained_difference":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
