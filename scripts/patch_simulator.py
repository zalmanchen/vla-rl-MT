"""Apply the compatibility changes used by pinned RLinf's RoboTwin installer."""

import importlib.util
from pathlib import Path
import re

for package, relative in [
    ("sapien", "wrapper/urdf_loader.py"),
    ("mplib", "planner.py"),
]:
    spec = importlib.util.find_spec(package)
    path = Path(spec.origin).parent / relative
    original = path.read_text()
    if package == "sapien":
        updated = re.sub(r"(\"r\")(\))( as)", r'\1, encoding="utf-8") as', original)
    else:
        updated = original.replace(
            "if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:",
            "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:",
        )
    if updated != original:
        # Replace the file rather than modifying a hardlinked uv cache entry.
        temporary = path.with_suffix(path.suffix + ".vla-tmp")
        temporary.write_text(updated)
        temporary.replace(path)
    print(f"Verified simulator compatibility patch: {path}")
