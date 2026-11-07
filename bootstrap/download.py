import concurrent.futures
import fnmatch
import json
import pathlib
import time

from huggingface_hub import hf_hub_download

ROOT = pathlib.Path(__file__).resolve().parents[1]


def download(repo, kind, dest, patterns):
    import types

    locked = json.loads((ROOT / "bootstrap/downloads.lock.json").read_text())
    manifest = next(m for m in locked if m["destination"] == dest)
    if manifest["repo"] != repo or manifest["type"] != kind:
        raise ValueError("Download does not match pinned manifest")
    info = types.SimpleNamespace(sha=manifest["revision"])
    files = manifest["files"]
    manifest = {
        "repo": repo,
        "type": kind,
        "revision": info.sha,
        "files": files,
        "patterns": patterns,
    }
    path = ROOT / dest
    path.mkdir(parents=True, exist_ok=True)
    (path / "download-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(repo, info.sha, "starting", flush=True)
    selected = [
        f["path"]
        for f in files
        if any(fnmatch.fnmatchcase(f["path"], pat) for pat in patterns)
    ]
    if not selected:
        raise RuntimeError(f"No files matched {repo}")
    print(repo, len(selected), "files selected", flush=True)

    def fetch(name):
        for attempt in range(8):
            try:
                result = hf_hub_download(
                    repo, name, repo_type=kind, revision=info.sha, local_dir=path
                )
                print("downloaded", repo, name, flush=True)
                return result
            except Exception as exc:
                print("retry", name, attempt + 1, str(exc)[:200], flush=True)
                if attempt == 7:
                    raise
                time.sleep(min(30 * (attempt + 1), 120))

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as downloads:
        for result in downloads.map(fetch, selected):
            pass
    (path / "DOWNLOAD_COMPLETE").write_text(info.sha + "\n")
    print(repo, "COMPLETE", flush=True)


if __name__ == "__main__":
    jobs = [
        (
            "motus-robotics/pi0.5_robotwin2",
            "model",
            "models/pi0.5_robotwin2",
            ["README.md", "model.safetensors", "assets/**", "metadata.pt"],
        ),
        (
            "TianxingChen/RoboTwin2.0",
            "dataset",
            "datasets/RoboTwin2.0",
            [
                "README.md",
                "dataset/*/demo_clean.zip",
                "embodiments.zip",
                "objects.zip",
                "background_texture.zip",
            ],
        ),
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(download, *j) for j in jobs]
        for f in futures:
            f.result()
