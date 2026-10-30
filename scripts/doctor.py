#!/usr/bin/env python3
"""Check CPU dependencies, demonstration integrity and native action transforms."""

import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vla_mtrl.common import ROOT, digest, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imports", action="store_true")
    parser.add_argument("--data", action="store_true")
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--require-simulator", action="store_true")
    args = parser.parse_args()
    environment()
    report = {
        "root": str(ROOT),
        "python": sys.executable,
        "simulator": {
            p: importlib.util.find_spec(p) is not None
            for p in ["sapien", "mplib", "open3d", "toppra", "transforms3d"]
        },
    }
    if args.require_simulator and not all(report["simulator"].values()):
        raise RuntimeError(
            "Simulator dependencies missing; run ./setup.sh without --train-only"
        )
    if args.imports or args.data:
        import torch
        import transformers
        from rlinf.models.embodiment.openpi import get_model

        del get_model
        if (
            not torch.__version__.startswith("2.7.1")
            or torch.version.cuda != "12.6"
            or transformers.__version__ != "4.53.2"
        ):
            raise RuntimeError(
                "Expected torch 2.7.1/CUDA 12.6 and OpenPI transformer fork 4.53.2"
            )
        report["versions"] = {
            p: importlib.metadata.version(p)
            for p in ["torch", "rlinf-openpi", "ray", "transformers", "tokenizers"]
        }
        report["cuda_initialized"] = torch.cuda.is_initialized()
        if report["cuda_initialized"]:
            raise RuntimeError("CPU check unexpectedly initialized CUDA")
    if args.data:
        import numpy as np
        import openpi.transforms as transforms
        from rlinf.models.embodiment.openpi.transforms.pipeline import (
            build_openpi_transforms,
        )
        from vla_mtrl.data import RoboTwinWindows, inspect_episode

        path = ROOT / ".data/robotwin/manifest.json"
        manifest = json.loads(path.read_text())
        for row in manifest["episodes"]:
            source = path.parent / manifest["raw_root"] / row["file"]
            inspect_episode(source)
            if args.deep and digest(source) != row["sha256"]:
                raise ValueError(f"Demonstration changed: {source}")
        inputs, outputs = build_openpi_transforms(
            str(ROOT / ".model/pi0_robotwin"),
            "pi0_aloha_robotwin",
            {"norm_stats_path": str(path.with_suffix(".norm_stats.json"))},
        )
        dataset = RoboTwinWindows(path, manifest["tasks"], None)
        sample = dataset[0]
        expected = sample["actions"].copy()
        encoded = transforms.compose(inputs)(sample)
        decoded = transforms.compose(outputs)(
            {"actions": encoded["actions"], "state": encoded["state"]}
        )
        error = float(np.max(np.abs(decoded["actions"] - expected)))
        if error > 1e-4:
            raise ValueError(f"Action roundtrip error: {error}")
        report["data"] = {
            "episodes": len(manifest["episodes"]),
            "windows": len(dataset),
            "roundtrip_error": error,
        }
    report["scope"] = (
        "CPU only; GPU training and SAPIEN rendering need target-server validation"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
