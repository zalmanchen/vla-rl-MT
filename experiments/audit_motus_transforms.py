"""Compare E1 adapters with the model-card OpenPI pipeline on saved real observations."""
from research_artifacts import artifact_path, relocate_html

import copy
import json
import os
from pathlib import Path

import numpy as np
from openpi.models.pi0_config import Pi0Config
from openpi.shared import normalize
from openpi.training.config import LeRobotAlohaDataConfig
import openpi.transforms as transforms
from rlinf.models.embodiment.openpi.transforms.pipeline import build_openpi_transforms

ROOT = Path(os.environ["PROJECT_ROOT"])


def compare_tree(left, right, prefix=""):
    differences = []
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return [{"key": prefix, "issue": "different keys"}]
        for key in left:
            differences.extend(compare_tree(left[key], right[key], prefix + "/" + key))
        return differences
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape:
        return [{"key": prefix, "shapes": [list(a.shape), list(b.shape)]}]
    if a.dtype.kind in "bOUS":
        equal = np.array_equal(a, b)
        delta = None
    else:
        delta = float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.size else 0.0
        equal = delta <= 1e-6
    return [] if equal else [{"key": prefix, "max_abs": delta}]


def main():
    checkpoint = ROOT / "models/pi0.5_robotwin2"
    normdir = checkpoint / "assets/pi0.5_clean_randomize_joint_training"
    norm = normalize.load(normdir)
    model = Pi0Config(pi05=True, action_horizon=32, dtype="float32")
    # Construct independently using the exact class/options shown in the model card.
    from openpi.training.config import AssetsConfig

    upstream = LeRobotAlohaDataConfig(
        repo_id="unused", adapt_to_pi=True, use_delta_joint_actions=True,
        assets=AssetsConfig(assets_dir=str(normdir.parent), asset_id=normdir.name),
    ).create(normdir.parent, model)
    forward = transforms.compose([
        transforms.InjectDefaultPrompt(None), *upstream.data_transforms.inputs,
        transforms.Normalize(norm, use_quantiles=upstream.use_quantile_norm),
        *upstream.model_transforms.inputs,
    ])
    backward = transforms.compose([
        *upstream.model_transforms.outputs,
        transforms.Unnormalize(norm, use_quantiles=upstream.use_quantile_norm),
        *upstream.data_transforms.outputs,
    ])
    ins, outs = build_openpi_transforms(str(checkpoint), "pi05_aloha_robotwin", {
        "norm_stats_path": str(normdir / "norm_stats.json"),
        "adapt_to_pi": True, "extra_delta_transform": True,
    })
    local_forward, local_backward = transforms.compose(ins), transforms.compose(outs)
    rng = np.random.default_rng(20261105)
    rows = []
    for path in sorted((ROOT / ".runtime/experiments/e1-reference-clean/clean").glob("*/input_*.npz")):
        with np.load(path) as data:
            common = {"state": data["state"].copy(), "prompt": str(data["instruction"])}
            direct = dict(common, images={
                "cam_high": data["head"].transpose(2, 0, 1),
                "cam_left_wrist": data["wrist"][0].transpose(2, 0, 1),
                "cam_right_wrist": data["wrist"][1].transpose(2, 0, 1),
            })
            wrapped = {"observation/image": data["head"], "observation/wrist_image": data["wrist"],
                       "observation/state": common["state"], "prompt": common["prompt"]}
            a, b = forward(copy.deepcopy(direct)), local_forward(copy.deepcopy(wrapped))
            inputs_diff = compare_tree(a, b)
            actions = rng.standard_normal((32, 32)).astype(np.float32)
            da = backward({"state": a["state"].copy(), "actions": actions.copy()})
            db = local_backward({"state": b["state"].copy(), "actions": actions.copy()})
            rows.append({"task": path.parent.name, "snapshot": str(path.relative_to(ROOT)),
                         "input_differences": inputs_diff, "output_differences": compare_tree(da, db)})
    report = {"scope": "E1 RLinf transforms vs independently constructed model-card upstream OpenPI transforms; no network inference",
              "snapshots": len(rows), "tasks": len({r["task"] for r in rows}),
              "all_input_output_equal_at_1e_minus_6": all(not r["input_differences"] and not r["output_differences"] for r in rows),
              "checkpoint_vs_paper_identity": "unconfirmed: public model card does not identify the paper table checkpoint",
              "rows": rows}
    output = artifact_path('motus-transform-audit.json')
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
