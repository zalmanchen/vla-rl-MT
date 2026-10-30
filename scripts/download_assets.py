#!/usr/bin/env python3
"""Download verified RoboTwin models, trajectories and simulator assets."""

import argparse
import json
from pathlib import Path
import stat
import subprocess
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vla_mtrl.common import ROOT, TASKS, digest, environment, write_json


def safe_extract(archive, target):
    target = Path(target).resolve()
    with zipfile.ZipFile(archive) as file:
        for member in file.infolist():
            destination = (target / member.filename).resolve()
            if not destination.is_relative_to(target) or stat.S_ISLNK(
                member.external_attr >> 16
            ):
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        file.extractall(target)


def verify(path, row):
    if not path.is_file() or path.stat().st_size != row["size"]:
        raise ValueError(f"Missing/truncated download: {path}")
    sha = digest(path)
    expected = (row.get("lfs") or {}).get("sha256")
    if expected and sha != expected:
        raise ValueError(f"SHA256 mismatch: {path}")
    if not expected and row.get("blobId"):
        import hashlib

        contents = path.read_bytes()
        if (
            hashlib.sha1(f"blob {len(contents)}\0".encode() + contents).hexdigest()
            != row["blobId"]
        ):
            raise ValueError(f"Git blob mismatch: {path}")
    return {"file": row["rfilename"], "bytes": row["size"], "sha256": sha}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component",
        choices=["all", "model", "data", "assets", "tokenizer", "dexmimicgen"],
        default="all",
    )
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-prepare",
        action="store_true",
        help="Download archives without building the training manifest",
    )
    args = parser.parse_args()
    environment()
    releases = json.loads((ROOT / "resources/releases.json").read_text())
    parts = (
        ["model", "data", "assets", "tokenizer"]
        if args.component == "all"
        else [args.component]
    )
    requested = [f"dataset/{task}/demo_clean.zip" for task in args.tasks]
    available = {r["rfilename"] for r in releases["data"]["files"]}
    if args.workers < 1 or any(name not in available for name in requested):
        parser.error(
            "Choose supported task names from resources/releases.json and positive workers"
        )
    plan = {
        "components": parts,
        "tasks": args.tasks,
        "model_repo": releases["model"]["repo"],
        "model_revision": releases["model"]["revision"],
        "data_revision": releases["data"]["revision"],
        "root": str(ROOT),
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    from huggingface_hub import snapshot_download

    for part in parts:
        if part == "tokenizer":
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/prepare_tokenizer.py")], check=True
            )
            continue
        if part == "dexmimicgen":
            from huggingface_hub import HfApi

            target = ROOT / ".data/dexmimicgen"
            receipt = target / "download_complete.json"
            revision = (
                json.loads(receipt.read_text())["revision"]
                if receipt.exists()
                else "main"
            )
            info = HfApi().repo_info(
                "MimicGen/dexmimicgen_datasets",
                repo_type="dataset",
                revision=revision,
                files_metadata=True,
            )
            patterns = [
                f"generated/two_arm_{task}.hdf5"
                for task in ["box_cleanup", "drawer_cleanup", "lift_tray"]
            ]
            snapshot_download(
                info.id,
                revision=info.sha,
                repo_type="dataset",
                local_dir=target,
                allow_patterns=patterns,
                max_workers=args.workers,
            )
            rows = []
            for source in info.siblings:
                if source.rfilename in patterns:
                    lfs = source.lfs
                    sha = lfs["sha256"] if isinstance(lfs, dict) else lfs.sha256
                    rows.append(
                        verify(
                            target / source.rfilename,
                            {
                                "rfilename": source.rfilename,
                                "size": source.size,
                                "lfs": {"sha256": sha},
                            },
                        )
                    )
            if len(rows) != len(patterns):
                raise ValueError("Incomplete DexMimicGen task set")
            write_json(
                receipt,
                {"revision": info.sha, "verified": rows, "training_supported": False},
            )
            continue
        spec = releases["model" if part == "model" else "data"]
        rows = (
            spec["files"]
            if part == "model"
            else [
                r
                for r in spec["files"]
                if r["rfilename"]
                in (
                    requested
                    if part == "data"
                    else ["objects.zip", "embodiments.zip", "background_texture.zip"]
                )
            ]
        )
        target = ROOT / (
            ".model/pi0_robotwin" if part == "model" else ".data/downloads/robotwin"
        )
        snapshot_download(
            spec["repo"],
            revision=spec["revision"],
            repo_type="model" if part == "model" else "dataset",
            local_dir=target,
            allow_patterns=[r["rfilename"] for r in rows],
            max_workers=args.workers,
        )
        verified = [verify(target / row["rfilename"], row) for row in rows]
        write_json(
            target / f"{part}_complete.json",
            {"revision": spec["revision"], "verified": verified},
        )
        if part == "data":
            for task in args.tasks:
                safe_extract(
                    target / f"dataset/{task}/demo_clean.zip",
                    ROOT / ".data/robotwin/raw" / task,
                )
        elif part == "assets":
            for row in rows:
                safe_extract(target / row["rfilename"], ROOT / "RoboTwin/assets")
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "RoboTwin/script/update_embodiment_config_path.py"),
                ],
                cwd=ROOT / "RoboTwin",
                check=True,
            )
    if "data" in parts and not args.no_prepare:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/prepare_data.py"),
                "--tasks",
                *args.tasks,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
