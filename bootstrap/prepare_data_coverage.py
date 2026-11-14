"""Prepare a bounded collection and nested success-only BC data views."""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from prepare_method_m1 import digest, freeze_pool, make_plan, save_json

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/"configs/multitask/data_coverage.json"


def select_successes(rows, tasks, count):
    selected = []
    for task in tasks:
        candidates = sorted((r for r in rows if r["task"] == task and r["split"] == "train" and r["success"]),
                            key=lambda r: r["seed_index"])
        if len({r["seed"] for r in candidates}) != len(candidates) or len(candidates) < count:
            raise ValueError(f"Insufficient or duplicated successes for {task}: {len(candidates)}/{count}")
        selected.extend(candidates[:count])
    return selected


def prepare_collection(collection, work):
    for mode, key in [("recompute", "passed"), ("replay", "all_outcomes_and_counts_match")]:
        path = ROOT/f".runtime/runs/method/bc-data-audit-v1/{mode}/report.json"
        report = json.loads(path.read_text())
        if not report["complete"] or not report[key]:
            raise ValueError(f"Initial data audit incomplete: {mode}")
    plan = make_plan(PROTOCOL, collection)
    old = ROOT/plan["config"]["coverage"]["validation_source"]
    old_manifest = json.loads((old/"manifest.json").read_text())
    used = {(r["task"], r["seed"]) for r in old_manifest["episodes"]}
    if any((r["task"], r["seed"]) in used for r in plan["cases"]):
        raise ValueError("New training scene overlaps the original pool")
    for name in ["scripts/audit_bc_data.py", "bootstrap/prepare_data_coverage.py"]:
        plan["sources"][name] = digest(ROOT/name)
    plan["audit_reports"] = {mode: digest(ROOT/f".runtime/runs/method/bc-data-audit-v1/{mode}/report.json")
                             for mode in ("recompute", "replay")}
    save_json(collection/"plan.json", plan)
    save_json(work/"collection-contract.json", {"plan_sha256": digest(collection/"plan.json"),
        "protocol_sha256": digest(PROTOCOL), "attempts": len(plan["cases"]), "new_training_only": True})


def build_view(source, old, destination, rows, config, selection):
    destination.mkdir(parents=True, exist_ok=False)
    episodes = []
    for row in rows:
        origin = old if row["split"] == "val" else source
        src, dst = origin/row["file"], destination/row["file"]
        if digest(src) != row["sha256"]:
            raise ValueError("Source episode changed")
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(src, dst)
        episodes.append(dict(row))
    parent = json.loads((source/"manifest.json").read_text())
    plan = {"config": config, "protocol": parent["protocol"], "cases": episodes,
            "selection": selection, "source_manifest_sha256": digest(source/"manifest.json"),
            "validation_manifest_sha256": digest(old/"manifest.json")}
    save_json(destination/"plan.json", plan)
    counts = Counter((r["task"], r["split"], r["success"]) for r in episodes)
    coverage = {f"{task}/{split}": {"success": counts[task, split, True], "failure": counts[task, split, False]}
                for task in config["tasks"] for split in ("train", "val")}
    manifest = {"scope": "Derived success-only BC training view and unchanged original development validation",
        "plan_sha256": digest(destination/"plan.json"), "config": config, "protocol": parent["protocol"],
        "episodes": episodes, "coverage": coverage, "scene_split_certified": True,
        "outcome_coverage_ready": False, "bc_only": True, "full_simulator_state_saved": False,
        "contrastive_pairs_verified": False, "source_manifest_sha256": plan["source_manifest_sha256"],
        "validation_manifest_sha256": plan["validation_manifest_sha256"]}
    save_json(destination/"manifest.json", manifest)
    return manifest


def prepare_training(collection, views, updates, work):
    import h5py

    if not (collection/"manifest.json").exists():
        freeze_pool(collection)
    config = json.loads(PROTOCOL.read_text())
    setting = config["coverage"]
    if digest(ROOT/config["parent_training_protocol"]) != config["parent_training_sha256"]:
        raise ValueError("Parent BC protocol changed")
    raw = json.loads((collection/"manifest.json").read_text())
    if len(raw["episodes"]) != 192:
        raise ValueError("Collection attempt budget changed")
    old = ROOT/setting["validation_source"]
    if digest(old/"manifest.json") != setting["validation_manifest_sha256"]:
        raise ValueError("Original pool changed")
    original = json.loads((old/"manifest.json").read_text())
    validation = [r for r in original["episodes"] if r["split"] == "val"]
    pools = {}
    for name, count in [("small", setting["small_successes_per_task"]), ("large", setting["large_successes_per_task"])]:
        selected = select_successes(raw["episodes"], config["tasks"], count)
        pools[name] = build_view(collection, old, views/name, selected+validation, config,
            {"count_per_task": count, "rule": setting["selection"]})
    small = {r["file"] for r in pools["small"]["episodes"] if r["split"] == "train"}
    large = {r["file"] for r in pools["large"]["episodes"] if r["split"] == "train"}
    if not small < large:
        raise ValueError("Small training pool must be a strict subset")
    # Preserve old validation noise exactly; give new training decisions stable
    # IDs from the entire raw collection, independent of the selected view.
    noise = {}
    for origin, manifest, offset in [(old, original, 0), (collection, raw, 100000)]:
        index = offset
        for row in manifest["episodes"]:
            with h5py.File(origin/row["file"]) as episode:
                for decision in range(len(episode["observation_steps"])-1):
                    noise[f"{row['file']}::{decision}"] = index
                    index += 1
    save_json(work/"fit-noise.json", noise)
    base = json.loads((ROOT/"configs/multitask/method_m4.json").read_text())
    jobs, variants, smoke = [], [], []
    for name in ("small", "large"):
        job = f"coverage_{name}"
        pool = views/name
        overrides = ["actor.model.model_type=mt_native_expert_policy", "actor.model.mode=bc",
            f"data.pool_path={pool}", f"data.pool_manifest_sha256={digest(pool/'manifest.json')}",
            "+data.label_path=none", "+data.label_sha256=none", "actor.micro_batch_size=2", "actor.global_batch_size=4",
            "actor.seed=0", "actor.optim.lr=0.0003", "+actor.model.adapter_lr=0.0003", "+actor.model.expert_lr=1e-6",
            "+actor.model.update_scope=expert_adapter", "+actor.optim.use_fp32_master_params=true",
            "+actor.fsdp_config.save_full_model_weights=false", "runner.max_steps=128", "runner.save_interval=128",
            "runner.val_check_interval=-1", "+runner.eval_export_steps=[32,64]", f"actor.group_name={job}"]
        jobs.append({"name": job, "entry": "scripts/train_method_joint.py", "config": "joint_value",
                     "extension": "vla_rl_mt.bc_stability", "overrides": overrides})
        smoke.append({**jobs[-1], "overrides": [v.replace("runner.max_steps=128", "runner.max_steps=4")
            .replace("runner.save_interval=128", "runner.save_interval=4").replace("+runner.eval_export_steps=[32,64]", "+runner.eval_export_steps=[2]")
            for v in overrides]+["+runner.trace_final_resumable=false"]})
        fit = [{"name": "w1", "checkpoint": base["w1_checkpoint"]}]
        for step in setting["evaluation_steps"]:
            actor = updates/job/f"checkpoints/global_step_{step}/actor"
            variants.append({"name": f"{job}_step{step}", "training_job": job, "step": step,
                "checkpoint": str(actor/"addon.pt"), "ready_marker": str(actor/"evaluation-ready.json"), "prompt": "plain"})
            fit.append({"name": f"{job}_step{step}", "checkpoint": str(actor/"addon.pt")})
        save_json(work/f"fit-{name}.json", {"variants": fit, "pool": str(pool), "pool_manifest_sha256": digest(pool/"manifest.json")})
    variants.sort(key=lambda v: (v["step"], v["name"]))
    save_json(work/"jobs.json", {"jobs": jobs, "protocol_sha256": digest(PROTOCOL)})
    save_json(work/"smoke-jobs.json", {"jobs": smoke, "protocol_sha256": digest(PROTOCOL)})
    save_json(work/"evaluation.json", {"variants": variants, "protocol_sha256": digest(PROTOCOL)})
    save_json(work/"views.json", {name: {"path": str(views/name), "manifest_sha256": digest(views/name/"manifest.json"),
                                      "coverage": manifest["coverage"]} for name, manifest in pools.items()})
    print(json.dumps({"training_views": 2, "new_rollout_variants": len(variants), "nested": True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "training"))
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--views", type=Path)
    parser.add_argument("--updates", type=Path)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    collection, work = args.collection.resolve(), args.work.resolve()
    if args.command == "plan":
        prepare_collection(collection, work)
    else:
        if args.views is None or args.updates is None:
            parser.error("training requires --views and --updates")
        prepare_training(collection, args.views.resolve(), args.updates.resolve(), work)


if __name__ == "__main__":
    main()
