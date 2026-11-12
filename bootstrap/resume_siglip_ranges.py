"""Resume the public SigLIP2 file with bounded parallel HTTP ranges and final SHA256."""

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models/siglip2-so400m-patch14-224"
REVISION = "78e403963a4f6a3640d07803284752326fdf4edf"


def main():
    lock = json.loads((ROOT / "bootstrap/recap-models.lock.json").read_text())
    model = next(m for m in lock if m["repo"].endswith("siglip2-so400m-patch14-224"))
    item = next(f for f in model["files"] if f["path"] == "model.safetensors")
    if (DEST / "DOWNLOAD_COMPLETE").exists():
        return
    part = DEST / "model.safetensors.parallel.partial"
    receipt = ROOT / ".runtime/bootstrap/siglip-ranges.json"
    if receipt.exists() and part.exists():
        state = json.loads(receipt.read_text())
    else:
        candidates = list((DEST / ".cache/huggingface/download").glob("*.incomplete"))
        prefix = max(candidates, key=lambda p: p.stat().st_size) if candidates else None
        if prefix:
            shutil.copyfile(prefix, part)
        else:
            part.touch()
        state = {"revision": REVISION, "prefix_bytes": part.stat().st_size, "done": [], "total_bytes": item["bytes"]}
        receipt.write_text(json.dumps(state))
    mutex = threading.Lock()
    chunk = 8 * 1024 * 1024
    pending = [(start, min(start + chunk, item["bytes"])) for start in range(state["prefix_bytes"], item["bytes"], chunk) if start not in state["done"]]
    fd = os.open(part, os.O_RDWR)
    os.ftruncate(fd, item["bytes"])
    print("range resume", "prefix", state["prefix_bytes"], "remaining chunks", len(pending), flush=True)

    thread_local = threading.local()

    def fetch(bounds):
        if not hasattr(thread_local, "session"):
            thread_local.session = requests.Session()
            thread_local.session.trust_env = True
        start, end = bounds
        for attempt in range(8):
            try:
                host = "https://hf-mirror.com"
                thread_local.session.trust_env = True
                url = f"{host}/google/siglip2-so400m-patch14-224/resolve/{REVISION}/model.safetensors?download=true"
                with thread_local.session.get(url, headers={"Range": f"bytes={start}-{end-1}"}, timeout=(20, 60), stream=True) as response:
                    response.raise_for_status()
                    expected = f"bytes {start}-{end-1}/{item['bytes']}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("Server did not honor the requested byte range")
                    data = b"".join(response.iter_content(1024 * 1024))
                    if len(data) != end - start:
                        raise ValueError("Incomplete byte range")
                offset = 0
                while offset < len(data):
                    offset += os.pwrite(fd, data[offset:], start + offset)
                with mutex:
                    state["done"].append(start)
                    tmp = receipt.with_suffix(".tmp")
                    tmp.write_text(json.dumps(state))
                    tmp.replace(receipt)
                    print("verified range length", end - start, "chunks", len(state["done"]), flush=True)
                return
            except Exception as exc:
                print("range retry", start, attempt + 1, type(exc).__name__, flush=True)
                if attempt == 7:
                    raise
                time.sleep(min(20, attempt * 3 + 1))
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for _ in pool.map(fetch, pending):
                pass
        os.fsync(fd)
    finally:
        os.close(fd)
    digest = hashlib.sha256()
    with part.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != item["sha256"]:
        raise ValueError("Full model SHA256 mismatch; do not use this checkpoint")
    part.replace(DEST / "model.safetensors")
    for other in model["files"]:
        target = DEST / other["path"]
        if not target.exists() or target.stat().st_size != other["bytes"]:
            raise ValueError("Missing companion file: " + other["path"])
    (DEST / "DOWNLOAD_COMPLETE").write_text(REVISION + "\n")
    print("SIGLIP_COMPLETE_SHA256_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
