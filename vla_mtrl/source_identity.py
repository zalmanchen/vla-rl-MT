"""Verify pinned third-party source content with or without Git metadata."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_REVISIONS = {
    "RLinf": "e1a82ebf0c1af69da6ab9b1519f5f3692fe33566",
    "RoboTwin": "dca9ec682688821c42944eacd7f5dd9bfb15396f",
}


def _git_head_at_root(path):
    """Do not mistake an enclosing superproject for a submodule repository."""
    dotgit = path / ".git"
    if not (dotgit.exists() or dotgit.is_symlink()) or not shutil.which("git"):
        return None
    env = os.environ.copy()
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"):
        env.pop(name, None)
    command = ["git", "-C", str(path), "rev-parse"]
    root = subprocess.run(command + ["--show-toplevel"], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, universal_newlines=True, env=env, timeout=15)
    if root.returncode or Path(root.stdout.strip()).resolve() != path.resolve():
        return None
    head = subprocess.run(command + ["HEAD"], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, universal_newlines=True, env=env, timeout=15)
    return head.stdout.strip() if head.returncode == 0 else None


def verify_sources(root=ROOT, manifest_path=None):
    """Verify every tracked blob. Extra runtime files are outside this check."""
    root = Path(root).resolve()
    manifest_path = Path(manifest_path or root / "resources/source_manifest.json")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema") != 1 or set(manifest.get("sources", {})) != set(EXPECTED_REVISIONS):
        raise ValueError("Invalid pinned source manifest")
    identities = {}
    for name, expected in EXPECTED_REVISIONS.items():
        spec = manifest["sources"][name]
        if spec.get("revision") != expected or not spec.get("files"):
            raise ValueError("Invalid manifest revision or empty file list for " + name)
        source = root / name
        head = _git_head_at_root(source) if source.is_dir() else None
        if head is not None and head != expected:
            raise ValueError("{} Git HEAD is {}, expected {}".format(name, head, expected))
        problems, materialized = [], []
        for entry in spec["files"]:
            relative = PurePosixPath(entry["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe source manifest path: " + entry["path"])
            path = source.joinpath(*relative.parts)
            try:
                if entry["mode"] == "120000":
                    if path.is_symlink():
                        content = os.fsencode(os.readlink(path))
                    elif path.is_file():
                        # Some ZIP tools materialize a symlink as its target text.
                        content = path.read_bytes()
                        materialized.append(entry["path"])
                    else:
                        raise FileNotFoundError(str(path))
                else:
                    if path.is_symlink() or not path.is_file():
                        raise FileNotFoundError(str(path))
                    if path.stat().st_size != entry["size"]:
                        problems.append(entry["path"] + " (size differs)")
                        if len(problems) >= 5:
                            break
                        continue
                    content = path.read_bytes()
                if len(content) != entry["size"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
                    problems.append(entry["path"] + " (content differs)")
            except (OSError, ValueError):
                problems.append(entry["path"] + " (missing or unreadable)")
            if len(problems) >= 5:
                break
        if problems:
            raise ValueError(
                "{} source files do not match pinned revision {}: {}. "
                "Download {}, extract its contents into {}/ and retry. "
                "Do not overwrite customized files without backing them up."
                .format(name, expected, "; ".join(problems), spec["archive_url"], name)
            )
        identities[name] = {
            "revision": expected,
            "method": "git_head_and_content" if head is not None else "archive_content_verified",
            "git_head_verified": head is not None,
            "tracked_files_verified": len(spec["files"]),
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "materialized_symlinks": materialized,
            "note": "Tracked file contents verified; permissions and extra files are not covered.",
        }
    return identities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(verify_sources(args.root, args.manifest), indent=2, ensure_ascii=False))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print("Source verification failed: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
