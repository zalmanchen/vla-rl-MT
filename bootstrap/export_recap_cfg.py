"""Export a native RLinf SFT checkpoint for the official RECAP CFG loader.

Uses RLinf's own tensor-layout conversions. The language embedding alias is
explicit; the unused action-expert LM head is zero-filled, never borrowed from
another policy. Every policy tensor must round-trip exactly.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--config-name", default="pi05_libero")
    parser.add_argument("--reference-weights", type=Path,
                        help="Native safetensors architecture reference; default: LIBERO base")
    args = parser.parse_args()
    import torch
    from rlinf.models.embodiment.openpi.checkpoint import (
        _convert_official_openpi_keys,
        _normalize_state_dict,
        resolve_full_weights,
    )
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model import (
        OpenPi0Config,
        OpenPi0ForCFGActionPrediction,
    )
    from rlinf.utils.ckpt_convertor.openpi.openpi_pytorch_to_openpi import (
        old_to_new_state_dict,
    )
    from rlinf.utils.ckpt_convertor.openpi.openpi_to_openpi_pytorch import (
        ACTION_EXPERT_LM_HEAD,
        new_to_old_state_dict,
    )
    from safetensors import safe_open

    source = resolve_full_weights(args.checkpoint)
    if source is None:
        raise FileNotFoundError(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=False)
    native = _normalize_state_dict(torch.load(source, map_location="cpu", weights_only=True, mmap=True))
    root = Path(__file__).resolve().parents[1]
    reference = args.reference_weights or root / "models/pi05_base_openpi/model.safetensors"
    with safe_open(reference, framework="pt", device="cpu") as base:
        # Motus ships OpenPI PyTorch keys; native SFT saves RLinf Pi0 keys.
        # Run the official converter on meta tensors to compare architecture
        # without loading another full model or confusing the two layouts.
        shapes = {key: torch.empty(tuple(base.get_slice(key).get_shape()), device="meta") for key in base.keys()}
    shapes = _convert_official_openpi_keys(_normalize_state_dict(shapes), reference)
    expected = {key: tuple(value.shape) for key, value in shapes.items()}
    if set(native) != set(expected) or any(tuple(native[key].shape) != shape for key, shape in expected.items()):
        raise ValueError("SFT checkpoint is not the complete expected native Pi0.5 architecture")

    mapped = new_to_old_state_dict(native)
    embedding = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    mapped[embedding] = mapped["paligemma_with_expert.paligemma.lm_head.weight"]
    cfg = OpenPi0Config(**get_openpi_config(args.config_name).model.__dict__)
    with torch.device("meta"):
        target = OpenPi0ForCFGActionPrediction(cfg).state_dict()
    if set(target) - set(mapped) != {ACTION_EXPERT_LM_HEAD} or set(mapped) - set(target):
        raise ValueError("Unexpected missing/extra keys in native-to-CFG conversion")
    mapped[ACTION_EXPERT_LM_HEAD] = torch.zeros(target[ACTION_EXPERT_LM_HEAD].shape, dtype=torch.float32)
    if any(mapped[key].shape != target[key].shape for key in target):
        raise ValueError("CFG target tensor shape mismatch")
    restored = old_to_new_state_dict(mapped)
    if set(restored) != set(native):
        raise ValueError("Round-trip conversion changed the native key set")
    for key in native:
        if not torch.equal(native[key], restored[key]):
            raise ValueError(f"Round-trip conversion changed policy tensor {key}")

    path = args.output / "model_state_dict/full_weights.pt"
    path.parent.mkdir()
    temporary = path.with_suffix(".pt.tmp")
    torch.save(mapped, temporary)
    temporary.replace(path)
    environment = "robotwin" if "robotwin" in args.config_name else "libero"
    stats = args.output / f"physical-intelligence/{environment}/norm_stats.json"
    stats.parent.mkdir(parents=True)
    shutil.copy2(args.norm_stats, stats)
    report = {"status": "passed", "source": str(source), "source_sha256": digest(source),
              "checkpoint": str(path), "sha256": digest(path), "norm_stats_sha256": digest(stats),
              "native_policy_tensors": len(native), "cfg_tensors": len(mapped),
              "round_trip": "all native tensors bitwise identical", "embedding_alias": embedding,
              "unused_zero_head": ACTION_EXPERT_LM_HEAD,
              "config_name": args.config_name, "reference_weights": str(reference),
              "scope": "official format conversion; requires separate numerical inference parity validation"}
    (args.output / "conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
