#!/usr/bin/env python3
"""Verify pinned GitHub source ZIPs and install them into empty directories."""

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vla_mtrl.source_identity import EXPECTED_REVISIONS, verify_sources


def prepare_sources(root, archives, manifest_path):
    root, manifest_path = Path(root).resolve(), Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if not (root / "setup.sh").is_file():
        raise ValueError("Project directory must contain setup.sh")
    if manifest.get("schema") != 1 or set(manifest.get("sources", {})) != set(EXPECTED_REVISIONS):
        raise ValueError("Invalid pinned source manifest")
    for name, revision in EXPECTED_REVISIONS.items():
        if manifest["sources"][name].get("revision") != revision:
            raise ValueError("Unexpected source revision for " + name)
        dest = root / name
        if dest.is_symlink() or (dest.exists() and (not dest.is_dir() or any(dest.iterdir()))):
            raise ValueError("{} is not an empty directory; existing files were not touched".format(dest))
    with tempfile.TemporaryDirectory(prefix=".vla-source-", dir=str(root)) as tmp:
        staging = Path(tmp)
        for name, revision in EXPECTED_REVISIONS.items():
            spec = manifest["sources"][name]
            prefix = name + "-" + revision
            expected = {item["path"] for item in spec["files"]}
            if not expected:
                raise ValueError("Empty source manifest for " + name)
            with zipfile.ZipFile(archives[name]) as archive:
                actual, total_size = set(), 0
                for info in archive.infolist():
                    path = PurePosixPath(info.filename)
                    if (path.is_absolute() or ".." in path.parts or "\\" in info.filename
                            or not path.parts or path.parts[0] != prefix):
                        raise ValueError("Unexpected path or version in {}: {}".format(name, info.filename))
                    if not info.is_dir():
                        rel = PurePosixPath(*path.parts[1:]).as_posix()
                        if rel in actual:
                            raise ValueError("Duplicate ZIP file entry: " + info.filename)
                        actual.add(rel)
                        total_size += info.file_size
                if total_size > 512 * 1024 * 1024:
                    raise ValueError("Unexpectedly large source ZIP: " + name)
                if actual != expected:
                    raise ValueError("{} ZIP file list differs from pinned revision: missing={}, extra={}".format(
                        name, sorted(expected - actual)[:5], sorted(actual - expected)[:5]))
                archive.extractall(staging)
            (staging / prefix).rename(staging / name)
        identities = verify_sources(staging, manifest_path)
        installed = []
        try:
            for name in EXPECTED_REVISIONS:
                dest = root / name
                if dest.is_symlink() or (dest.exists() and (not dest.is_dir() or any(dest.iterdir()))):
                    raise ValueError("{} changed while preparing; not overwritten".format(dest))
                os.replace(str(staging / name), str(dest))
                installed.append(name)
        except (OSError, ValueError) as error:
            raise ValueError("Could not place both sources. Already placed: {}. "
                             "Existing nonempty directories were not removed. {}".format(installed, error))
    return identities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--rlinf", type=Path, required=True, help="Pinned RLinf ZIP from GitHub")
    parser.add_argument("--robotwin", type=Path, required=True, help="Pinned RoboTwin ZIP from GitHub")
    args = parser.parse_args()
    try:
        identities = prepare_sources(
            args.repo, {"RLinf": args.rlinf, "RoboTwin": args.robotwin},
            Path(args.repo) / "resources/source_manifest.json",
        )
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print("Sources not prepared: {}".format(error), file=sys.stderr)
        return 1
    print(json.dumps(identities, indent=2, ensure_ascii=False))
    print("Verified sources are ready. Next: bash ./setup.sh --verify-sources")
    return 0


if __name__ == "__main__":
    sys.exit(main())
