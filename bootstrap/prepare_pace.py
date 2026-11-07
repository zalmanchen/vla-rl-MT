"""Fetch pinned PACE sources and four task checkpoints, verifying upstream hashes."""

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import requests
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
REVISION = "17ef545270cf26ed81169893b60f845f80ce6557"
TASKS = ["adjust_bottle", "click_bell", "stack_blocks_two", "handover_block"]
SOURCES = {
    "OpenPI-PACE": ("niejunnan25/Openpi-PACE", "62e44c30c49df06897168cc5081ecd71afd088b2"),
    "PACE": ("niejunnan25/PACE", "5d0f4a92415f8899963cba7f0f929f1b9c5f75db"),
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    cache = ROOT / ".runtime/bootstrap"
    source_receipts = {}
    for name, (repo, revision) in ([] if os.environ.get("PACE_MODELS_ONLY") == "1" else SOURCES.items()):
        target = ROOT / "third_party" / name
        archive = cache / f"{name}-{revision}.zip"
        if not archive.exists():
            response = requests.get(f"https://codeload.github.com/{repo}/zip/{revision}", timeout=180)
            response.raise_for_status()
            archive.write_bytes(response.content)
        if not target.exists():
            with zipfile.ZipFile(archive) as z:
                target.mkdir(parents=True)
                for entry in z.infolist():
                    rel = Path(*Path(entry.filename).parts[1:])
                    if not rel.parts:
                        continue
                    dest = target / rel
                    if not dest.resolve().is_relative_to(target.resolve()):
                        raise ValueError(entry.filename)
                    if entry.is_dir():
                        dest.mkdir(parents=True, exist_ok=True)
                    else:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(z.read(entry))
                        mode = (entry.external_attr >> 16) & 0o777
                        if mode:
                            dest.chmod(mode)
        source_receipts[name] = {"repository": repo, "revision": revision, "archive_sha256": digest(archive)}
        print("source ready", name, revision, flush=True)
    (cache / "pace-sources.lock.json").write_text(json.dumps(source_receipts, indent=2) + "\n")
    manifest = json.loads((cache / "pace-manifest.remote.json").read_text())
    dest = ROOT / "models/pi05-robotwin-pace"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    files = [manifest["tokenizer"]]
    for model in manifest["models"]:
        if model["task"] in TASKS:
            files.extend(model["files"])
    lock = {"repo": "niejunnan25/pi05-robotwin", "revision": REVISION, "tasks": TASKS, "sources": source_receipts, "files": files}
    (ROOT / "bootstrap/pace-downloads.lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    print("selected", len(files), "files", sum(f["bytes"] for f in files), "bytes", flush=True)

    def fetch(item):
        target = dest / item["path"]
        if target.exists() and target.stat().st_size == item["bytes"] and digest(target) == item["sha256"]:
            return item["path"]
        for attempt in range(6):
            try:
                downloaded = Path(hf_hub_download(lock["repo"], item["path"], revision=REVISION, local_dir=dest, endpoint="https://huggingface.co"))
                if downloaded.stat().st_size != item["bytes"] or digest(downloaded) != item["sha256"]:
                    raise ValueError(f"Checksum mismatch: {item['path']}")
                print("verified", item["path"], flush=True)
                return item["path"]
            except Exception as exc:
                print("retry", item["path"], attempt + 1, type(exc).__name__, flush=True)
                if attempt == 5:
                    raise
                time.sleep(5 * (attempt + 1))

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for _ in pool.map(fetch, files):
            pass
    for name in ["README.md", "TERMS.md", "NOTICE"]:
        hf_hub_download(lock["repo"], name, revision=REVISION, local_dir=dest, endpoint="https://huggingface.co")
    (dest / "DOWNLOAD_COMPLETE").write_text(REVISION + "\n")
    print("ALL FOUR TASK CHECKPOINTS VERIFIED", flush=True)


if __name__ == "__main__":
    main()
