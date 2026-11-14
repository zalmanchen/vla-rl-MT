"""Derive terminal-binary targets without changing frozen trajectories/features."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def binary_rows(rows):
    """One success reward on the final executed chunk, zero terminal bootstrap."""
    result = copy.deepcopy(rows)
    for index, row in enumerate(result):
        if row["terminal"]:
            if row["use_for_loss"]:
                raise ValueError("Terminal observation cannot be a supervised decision")
            row.update(target=0.0, chunk_reward=0.0)
            continue
        if not row["use_for_loss"] or index+1 >= len(result):
            raise ValueError("Missing supervised decision/successor")
        successor = result[index+1]
        if successor["episode"] != row["episode"] or successor["success"] != row["success"]:
            raise ValueError("Missing aligned final observation or inconsistent outcome")
        row.update(target=float(row["success"]),
                   chunk_reward=float(row["success"] and successor["terminal"]))
    # This certifies the reward bookkeeping; it is not oracle action advantage.
    for index, row in enumerate(result):
        if not row["terminal"]:
            if row["target"] != row["chunk_reward"]+result[index+1]["target"]:
                raise ValueError("Binary chunk return identity failed")
    return result


def main():
    import torch
    from cache_method_m1 import validate_pool

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/multitask/method_m1_binary.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.config.read_text())
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    pool = ROOT/protocol["pool"]
    fingerprint = digest(pool/"manifest.json")
    if fingerprint != protocol["pool_manifest_sha256"]:
        raise ValueError("Raw trajectory pool differs from declared protocol")
    manifest = json.loads((pool/"manifest.json").read_text())
    audit = validate_pool(pool, manifest)
    reference_plan = json.loads(((ROOT/protocol["reference_runs"]).parent/"training-plan.json").read_text())
    caches = {}
    for condition in protocol["feature_conditions"]:
        source = ROOT/protocol["parent_feature_dir"]/f"features-{condition}.pt"
        original = torch.load(source, map_location="cpu", weights_only=True)
        if (original["pool_manifest_sha256"] != fingerprint or original["condition"] != condition
                or original["readout"] != protocol["readout"]
                or original["features"].shape != (len(original["rows"]), protocol["feature_width"])):
            raise ValueError("Wrong parent feature contract")
        expected_hashes = {job["feature_sha256"] for job in reference_plan["jobs"]
                           if job["condition"] == condition and job["readout"] == protocol["readout"]}
        if expected_hashes != {digest(source)}:
            raise ValueError("Parent feature cache differs from the locked reference training plan")
        if original.get("target_scheme", "recap_time_cost") != "recap_time_cost":
            raise ValueError("Expected the unchanged RECAP parent targets")
        episodes = {row["file"]: row for row in manifest["episodes"]}
        for row in original["rows"]:
            expected = episodes[row["episode"]]
            if any(row[key] != expected[key] for key in ("task", "split", "seed", "success")):
                raise ValueError("Cached outcome/identity differs from audited raw trajectory")
        cache = {**original, "rows": binary_rows(original["rows"]), "target_scheme": "terminal_binary",
                 "target_protocol": protocol, "target_protocol_sha256": digest(args.config),
                 "parent_feature_sha256": digest(source), "data_audit": audit,
                 "derivation_source_sha256": digest(Path(__file__)),
                 "scope": protocol["scope"]}
        cache["config"] = {**original["config"], "scope": protocol["scope"],
                           "gamma": 1.0, "failure_reward": 0.0, "ordinary_reward": 0.0,
                           "success_reward": 1.0, "return_scale": 1.0,
                           "value_head_width": protocol["head_hidden_width"]}
        assert torch.equal(cache["features"], original["features"])
        caches[condition] = cache
    output.mkdir(parents=True, exist_ok=False)
    records = {}
    for condition, cache in caches.items():
        destination = output/f"features-{condition}.pt"
        torch.save(cache, destination)
        records[condition] = {"file": destination.name, "sha256": digest(destination),
                              "parent_feature_sha256": cache["parent_feature_sha256"],
                              "features_identical": True, "rows": len(cache["rows"])}
    (output/"manifest.json").write_text(json.dumps({"protocol": protocol, "data_audit": audit,
        "raw_pool_manifest_sha256": fingerprint, "target_protocol_sha256": digest(args.config),
        "caches": records, "raw_pool_modified": False}, indent=2)+"\n")
    print(json.dumps({"derived": str(output), "features_identical": True, "caches": records}, indent=2))


if __name__ == "__main__":
    main()
