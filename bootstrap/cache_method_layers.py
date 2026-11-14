"""Cache at most four prefix depths of the frozen W1 policy; no actor changes."""

import argparse
import copy
import json
import sys
from pathlib import Path

import cv2
import h5py
import torch

from vla_rl_mt.joint_value import BoundaryPool, digest, to_observation, write_json
from vla_rl_mt.shared_value import SharedPrefixValue, pool_prefix

ROOT = Path(__file__).resolve().parents[1]


def main():
    sys.path.insert(0, str(ROOT/"scripts"))
    from evaluate_multitask import load_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.protocol.read_text())
    pool = BoundaryPool(ROOT/config["pool"], config["pool_manifest_sha256"])
    checkpoint_path = ROOT/config["w1_checkpoint"]
    actor_sha = digest(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    if (actor_sha != config["parent_checkpoint_sha256"]["w1"]
            or checkpoint["identity"]["base_sha256"] != pool.manifest["protocol"]["model"]["sha256"]
            or digest(ROOT/config["binary_row_cache"]) != config["binary_row_cache_sha256"]
            or digest(ROOT/pool.manifest["protocol"]["model"]["path"]/"model.safetensors")
                != pool.manifest["protocol"]["model"]["sha256"]):
        raise ValueError("Pinned actor/base/row-cache identity mismatch")
    parent = torch.load(ROOT/config["binary_row_cache"], weights_only=True, map_location="cpu")
    if parent["pool_manifest_sha256"] != config["pool_manifest_sha256"]:
        raise ValueError("Binary row cache is from another pool")
    policy, _ = load_model("native", pool.manifest["protocol"])
    probe = SharedPrefixValue(policy, **checkpoint["architecture"]).to("cuda").train()
    probe.load_addon(checkpoint["addon"])
    probe.requires_grad_(False)
    policy.gradient_checkpointing_enable({"use_reentrant": False})
    layers = config["layers"]
    if len(layers) > 4 or len(set(layers)) != len(layers) or max(layers) != len(policy.llm.layers):
        raise ValueError("At most four distinct layers, including final layer, are required")
    captured, handles = {}, []
    for layer in layers:
        def capture(module, inputs, output, layer=layer):
            captured[layer] = output[0][0]
        handles.append(policy.llm.layers[layer-1].register_forward_hook(capture))
    features = {layer: [] for layer in layers}
    pending = []
    final_parity = 0.0

    def flush():
        nonlocal final_parity
        if not pending:
            return
        observation = to_observation(torch.utils.data.default_collate(pending), "cuda")
        with torch.no_grad():
            final, mask = probe.prefix_tokens(observation, detach_shared=True)
            reference = pool_prefix(final, mask, observation.tokenized_prompt_mask, readout="vision_prompt_concat")
            for layer in layers:
                # A common frozen native final RMSNorm at every depth keeps the
                # normalization operation fixed and reproduces the old readout.
                normalized, _ = policy.llm.final_norms[0](captured[layer], None)
                result = pool_prefix(normalized, mask, observation.tokenized_prompt_mask, readout="vision_prompt_concat")
                features[layer].append(result.cpu())
                if layer == len(policy.llm.layers):
                    final_parity = max(final_parity, float((result-reference).abs().max()))
        captured.clear()
        pending.clear()

    try:
        for row in parent["rows"]:
            with h5py.File(pool.path/row["episode"]) as episode:
                index = row["observation_index"]
                images = {camera: cv2.cvtColor(cv2.imdecode(episode[f"images/{camera}"][index], cv2.IMREAD_COLOR),
                                               cv2.COLOR_BGR2RGB).transpose(2, 0, 1) for camera in episode["images"]}
                raw = {"images": images, "state": episode["state"][index], "prompt": str(episode.attrs["instruction"])}
            pending.append(pool.transform(copy.deepcopy(raw)))
            if len(pending) == 2:
                flush()
        flush()
    finally:
        for handle in handles:
            handle.remove()
        probe.close()
    if final_parity != 0 or digest(checkpoint_path) != actor_sha:
        raise ValueError("Final readout parity or frozen-actor identity failed")
    report = {"scope": "Frozen W1 prefix readout only; no actor/value gradients", "layers": layers,
              "pool_audit": pool.audit, "actor_sha256": actor_sha,
              "normalization": "post-block states at each depth followed by the same frozen final RMSNorm",
              "last_layer_native_readout_max_abs": final_parity, "files": {}}
    for layer in layers:
        tensor = torch.cat(features[layer]).float()
        if tensor.shape != (len(parent["rows"]), 4096) or not bool(torch.isfinite(tensor).all()):
            raise ValueError("Invalid layer feature shape/value")
        cache = {"features": tensor, "rows": parent["rows"], "layer": layer,
                 "pool_manifest_sha256": config["pool_manifest_sha256"], "actor_sha256": actor_sha,
                 "condition": "instruction", "readout": "vision_prompt_concat", "target_scheme": "terminal_binary",
                 "normalization": report["normalization"]}
        path = args.output/f"layer-{layer:02d}.pt"
        torch.save(cache, path)
        report["files"][str(layer)] = {"path": str(path), "sha256": digest(path)}
    write_json(args.output/"audit.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
