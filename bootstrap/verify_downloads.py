"""Verify every selected download against its Hugging Face content ETag."""

import argparse
import concurrent.futures
import fnmatch
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(job):
    base, revision, relative = job
    path = base / relative
    meta = base / ".cache/huggingface/download" / (relative + ".metadata")
    lines = meta.read_text().splitlines()
    if lines[0] != revision:
        raise ValueError(f"Revision mismatch: {path}")
    etag = lines[1].strip('"')
    if re.fullmatch("[0-9a-f]{64}", etag):
        digest = hashlib.sha256()
    elif re.fullmatch("[0-9a-f]{40}", etag):
        digest = hashlib.sha1()
        digest.update(f"blob {path.stat().st_size}\0".encode())
    else:
        raise ValueError(f"Unknown content ETag for {path}: {etag}")
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != etag:
        raise ValueError(f"Checksum mismatch: {path}")
    print("verified", relative, flush=True)
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "etag": etag,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-randomized", action="store_true")
    args = parser.parse_args()
    jobs = []
    roots = [
        "models/pi0.5_robotwin2",
        "datasets/RoboTwin2.0",
    ]
    if args.include_randomized:
        roots.append("datasets/RoboTwin2.0-randomized")
    for rel in roots:
        base = ROOT / rel
        manifest = json.loads((base / "download-manifest.json").read_text())
        for item in manifest["files"]:
            if any(
                fnmatch.fnmatchcase(item["path"], pat) for pat in manifest["patterns"]
            ):
                jobs.append((base, manifest["revision"], item["path"]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(verify, jobs))
    output = ROOT / ".runtime/validation"
    output.mkdir(parents=True, exist_ok=True)
    (output / "downloads.json").write_text(
        json.dumps(
            {
                "status": "verified",
                "file_count": len(results),
                "total_bytes": sum(x["bytes"] for x in results),
                "files": results,
            },
            indent=2,
        )
        + "\n"
    )
    print("ALL DOWNLOADS VERIFIED", len(results), flush=True)
