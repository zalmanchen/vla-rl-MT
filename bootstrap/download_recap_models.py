"""Download pinned official RECAP backbones using the existing HF login."""

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
from huggingface_hub import HfApi, get_token, hf_hub_download
from huggingface_hub.errors import GatedRepoError

ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    ("google/gemma-3-270m", "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1", "gemma-3-270m"),
    ("google/siglip2-so400m-patch14-224", "78e403963a4f6a3640d07803284752326fdf4edf", "siglip2-so400m-patch14-224"),
]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    (ROOT / ".runtime/bootstrap").mkdir(parents=True, exist_ok=True)
    token = get_token()
    for p in [ROOT / ".runtime/cache/huggingface/token", Path.home() / ".cache/huggingface/token"]:
        if token is None and p.exists():
            token = p.read_text().strip()
    api = HfApi(endpoint="https://huggingface.co", token=token)
    lock = []
    selected = [m for m in MODELS if not os.environ.get("RECAP_MODEL_FILTER") or m[2] == os.environ["RECAP_MODEL_FILTER"]]
    for repo, revision, directory in selected:
        info = api.model_info(repo, revision=revision, files_metadata=True)
        files = []
        for f in info.siblings:
            if f.rfilename.endswith((".json", ".model", ".safetensors")):
                files.append({"path": f.rfilename, "bytes": f.size,
                              "sha256": f.lfs.sha256 if f.lfs else None})
        lock.append({"repo": repo, "revision": revision, "destination": "models/" + directory, "files": files})
    (ROOT / ("bootstrap/recap-models" + ("-" + os.environ["RECAP_MODEL_FILTER"] if os.environ.get("RECAP_MODEL_FILTER") else "") + ".lock.json")).write_text(json.dumps(lock, indent=2) + "\n")
    state_path = ROOT / (".runtime/bootstrap/recap-model-download-status" + ("-" + os.environ["RECAP_MODEL_FILTER"] if os.environ.get("RECAP_MODEL_FILTER") else "") + ".json")

    def download(model):
        dest = ROOT / model["destination"]
        dest.mkdir(parents=True, exist_ok=True)
        print("starting", model["repo"], model["revision"], flush=True)

        def fetch(item):
            for attempt in range(8):
                try:
                    target = Path(hf_hub_download(model["repo"], item["path"], revision=model["revision"],
                                                token=token, endpoint="https://huggingface.co", local_dir=dest))
                    if target.stat().st_size != item["bytes"]:
                        raise ValueError("Size mismatch")
                    if item["sha256"] and digest(target) != item["sha256"]:
                        raise ValueError("Checksum mismatch")
                    print("verified", model["repo"], item["path"], flush=True)
                    return
                except GatedRepoError:
                    raise
                except Exception as exc:
                    # Do not print URLs, authorization headers or tokens.
                    print("retry", model["repo"], item["path"], attempt + 1, type(exc).__name__, flush=True)
                    if attempt == 7:
                        raise
                    time.sleep(min(30, 5 * (attempt + 1)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            for _ in pool.map(fetch, model["files"]):
                pass
        (dest / "DOWNLOAD_COMPLETE").write_text(model["revision"] + "\n")
        print("MODEL_COMPLETE", model["repo"], flush=True)

    state_path.write_text(json.dumps({"phase": "downloading", "started_unix": time.time(), "pid": os.getpid()}))
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for _ in pool.map(download, lock):
                pass
    except Exception as exc:
        state_path.write_text(json.dumps({"phase": "error", "type": type(exc).__name__, "unix_time": time.time()}))
        raise
    state_path.write_text(json.dumps({"phase": "complete", "unix_time": time.time()}))


if __name__ == "__main__":
    main()
