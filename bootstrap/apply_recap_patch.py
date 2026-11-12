"""Apply the recorded RoboTwin compatibility patch to the pinned RLinf snapshot."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def apply_patch():
    manifest_path = ROOT / "bootstrap/patches/recap-robotwin.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    source = ROOT / "third_party/RLinf"
    pending = []
    for item in manifest["files"]:
        digest = hashlib.sha256((source / item["path"]).read_bytes()).hexdigest()
        if digest == item["patched_sha256"]:
            continue
        if digest != item["upstream_sha256"]:
            raise ValueError(
                f"Unrecognized local changes in {item['path']}; preserve and review them before applying the patch"
            )
        pending.append(item["path"])
    if pending:
        if len(pending) != len(manifest["files"]):
            raise ValueError(
                "Partially applied RECAP patch; inspect the source before continuing"
            )
        patch = ROOT / "bootstrap/patches/recap-robotwin.patch"
        if hashlib.sha256(patch.read_bytes()).hexdigest() != manifest["patch_sha256"]:
            raise ValueError("RECAP patch checksum mismatch")
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=source, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=source, check=True)
    for item in manifest["files"]:
        actual = hashlib.sha256((source / item["path"]).read_bytes()).hexdigest()
        if actual != item["patched_sha256"]:
            raise ValueError(f"Patched source checksum mismatch: {item['path']}")
    print("RLinf RoboTwin RECAP compatibility patch verified")


if __name__ == "__main__":
    apply_patch()
