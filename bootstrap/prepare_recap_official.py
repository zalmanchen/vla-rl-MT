"""Prepare the pinned public RLinf RECAP experiment without RoboTwin patches.

This prepares resources, not a claim that the unpublished experiment is reproduced.
Run downloads with .runtime/venv-tools/bin/python.
"""

import argparse
import base64
import concurrent.futures
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "bootstrap/recap-official.lock.json"
CACHE = ROOT / ".runtime/bootstrap"
STATE = CACHE / "recap-official-download.json"
FILES = CACHE / "recap-official-files.json"
VERIFIED = CACHE / "recap-official-verified.jsonl"
TEMPLATES = {
    "returns": "examples/offline_rl/config/recap_compute_returns.yaml",
    "value": "examples/offline_rl/config/recap_value_model_sft.yaml",
    "advantages": "examples/offline_rl/config/recap_compute_advantages.yaml",
    "policy": "examples/offline_rl/config/cfg_rl_openpi.yaml",
    "sft": "examples/sft/config/libero_sft_openpi.yaml",
    "eval": "evaluations/libero/libero_10_openpi_pi05_eval.yaml",
    "eval_cfg": "evaluations/libero/libero_10_openpi_pi05_eval.yaml",
}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def file_ok(path, item):
    if not path.is_file() or path.stat().st_size != item["bytes"]:
        return False
    if item["sha256"]:
        return digest(path) == item["sha256"]
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(header + payload).hexdigest() == item["git_blob"]


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def live_state(path):
    """A saved 'downloading' field alone does not establish a live worker."""
    state = json.loads(path.read_text())
    lock_path = path.with_suffix(".lock")
    held = False
    if lock_path.exists():
        with lock_path.open("rb") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                held = True
    state["worker_lock_held"] = held
    state["recorded_pid_alive"] = Path(f"/proc/{state.get('pid')}").exists()
    if state["phase"] not in ("complete", "converted", "error") and not held:
        state["effective_phase"] = "stopped"
    else:
        state["effective_phase"] = state["phase"]
    return state


def prepare_sources(lock):
    source = lock["upstream"]
    archive = CACHE / "RLinf-upstream.zip"
    if not archive.exists():
        urllib.request.urlretrieve(
            f"{source['repository']}/archive/{source['revision']}.zip", archive
        )
    if digest(archive) != source["archive_sha256"]:
        raise ValueError("Official RLinf archive checksum mismatch")
    target = ROOT / source["path"]
    count = 0
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            relative = Path(*Path(entry.filename).parts[1:])
            if not relative.parts or entry.is_dir():
                continue
            dest = target / relative
            if not dest.resolve().is_relative_to(target.resolve()):
                raise ValueError("Unsafe archive path")
            original = bundle.read(entry)
            if dest.exists():
                if dest.read_bytes() != original:
                    raise ValueError(f"Official source was modified: {relative}")
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(original)
                permissions = (entry.external_attr >> 16) & 0o777
                if permissions:
                    dest.chmod(permissions)
            count += 1
    report = {"revision": source["revision"], "files_verified": count,
              "patches": [], "path": source["path"], "checked_unix": time.time()}
    atomic_json(CACHE / "recap-official-source.json", report)
    print(json.dumps(report), flush=True)


def prepare_configs(lock):
    # Hydra forbids importing another primary config's hydra.searchpath. Merge
    # path overrides into generated primary copies, preserving the source files.
    import yaml

    def merge(base, override):
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                merge(base[key], value)
            else:
                base[key] = value
        return base

    directory = ROOT / ".runtime/recap-official/configs"
    directory.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for stage, relative in TEMPLATES.items():
        source = ROOT / lock["upstream"]["path"] / relative
        overrides = ROOT / "configs/recap_official" / f"{stage}.yaml"
        dest = directory / f"{stage}.yaml"
        config = merge(yaml.safe_load(source.read_text()), yaml.safe_load(overrides.read_text()))
        dest.write_text(yaml.safe_dump(config, sort_keys=False))
        provenance[stage] = {"template": relative, "template_sha256": digest(source),
                             "overrides_sha256": digest(overrides), "generated_sha256": digest(dest)}
    atomic_json(directory / "provenance.json", provenance)
    print("Prepared primary configs:", directory, flush=True)


def prepare_work_data(lock):
    """Keep released metadata intact when official stages write sidecars/stats."""
    dataset = lock["dataset"]
    files = load_manifest(lock)
    source_root = ROOT / dataset["destination"]
    work_root = ROOT / ".runtime/recap-official/data"
    provenance = {}
    for split in dataset["splits"].values():
        source = source_root / split
        target = work_root / split
        metadata = [f for f in files if f["path"].startswith(split + "/meta/")]
        if not metadata or not all(file_ok(source_root / f["path"], f) for f in metadata):
            raise ValueError(f"Released metadata not fully verified: {split}")
        (target / "meta").mkdir(parents=True, exist_ok=True)
        for item in metadata:
            original = source_root / item["path"]
            dest = target / "meta" / original.name
            if not dest.exists():
                shutil.copy2(original, dest)
        for name in ("data", "videos"):
            link = target / name
            if link.is_symlink():
                if link.resolve() != (source / name).resolve():
                    raise ValueError(f"Working dataset points at another source: {link}")
            elif link.exists():
                raise ValueError(f"Working data link would overwrite a directory: {link}")
            else:
                link.symlink_to(source / name, target_is_directory=True)
        provenance[split] = {"revision": dataset["revision"], "source": str(source),
                             "original_metadata_sha256": {Path(f["path"]).name: digest(source_root / f["path"])
                                                          for f in metadata}}
    atomic_json(work_root / "provenance.json", provenance)
    print("Prepared working metadata; media/parquet remain shared:", work_root, flush=True)


def convert_base(lock, wait):
    """Convert the original JAX base using RLinf's converter, after verification."""
    base_lock = ROOT / "bootstrap/recap-official-base.lock.json"
    marker = ROOT / "models/pi05_base_jax/DOWNLOAD_COMPLETE"
    norm = ROOT / lock["initial_policy"]["norm_stats"]["path"]
    target = ROOT / "models/pi05_base_openpi"
    state_path = CACHE / "recap-official-base-conversion.json"
    with (CACHE / "recap-official-base-conversion.lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not marker.is_file():
            if not wait:
                raise RuntimeError("Original JAX base is not fully downloaded/verified")
            atomic_json(state_path, {"phase": "waiting_for_verified_base", "pid": os.getpid(),
                                     "updated_unix": time.time()})
            time.sleep(30)
        if marker.read_text().strip() != digest(base_lock):
            raise ValueError("Base completion marker differs from source lock")
        if digest(norm) != lock["initial_policy"]["norm_stats"]["sha256"]:
            raise ValueError("SFT normalization statistics changed")
        if (target / "CONVERSION_COMPLETE.json").exists():
            report = json.loads((target / "CONVERSION_COMPLETE.json").read_text())
            if report["model_sha256"] != digest(target / "model.safetensors"):
                raise ValueError("Previously converted model changed")
            print("Converted base already verified", flush=True)
            return
        env = os.environ.copy()
        env.update(PYTHONPATH=str(ROOT / lock["upstream"]["path"]), JAX_PLATFORMS="cpu",
                   XLA_PYTHON_CLIENT_PREALLOCATE="false")
        command = [sys.executable, "-m", "rlinf.utils.ckpt_convertor.openpi.convert",
                   "--mode", "jax_to_openpi", "--input-model", str(marker.parent),
                   "--input-norm-stats", str(norm), "--output-model", str(target),
                   "--output-norm-stats", str(target / "physical-intelligence/libero/norm_stats.json"),
                   "--action-dim", "32", "--action-horizon", "10", "--max-token-len", "200"]
        atomic_json(state_path, {"phase": "converting", "pid": os.getpid(), "command": command,
                                 "updated_unix": time.time()})
        try:
            subprocess.run(command, cwd=ROOT, env=env, check=True)
            report = {"phase": "converted", "base_lock_sha256": digest(base_lock),
                      "norm_stats_sha256": digest(norm), "command": command,
                      "model_sha256": digest(target / "model.safetensors"),
                      "updated_unix": time.time(), "scope": "original base conversion; not SFT or RECAP training"}
            atomic_json(target / "CONVERSION_COMPLETE.json", report)
            atomic_json(state_path, report)
        except Exception as exc:
            atomic_json(state_path, {"phase": "error", "error_type": type(exc).__name__,
                                     "updated_unix": time.time()})
            raise


def load_manifest(lock, files_path=FILES):
    dataset = lock["dataset"]
    if not files_path.exists():
        url = (f"https://huggingface.co/api/datasets/{dataset['repo']}/revision/"
               f"{dataset['revision']}?blobs=true")
        with urllib.request.urlopen(url, timeout=120) as response:
            info = json.load(response)
        files = [{"path": f["rfilename"], "bytes": f["size"],
                  "sha256": f.get("lfs", {}).get("sha256"), "git_blob": f["blobId"]}
                 for f in info["siblings"]]
        files_path.write_text(json.dumps(files, sort_keys=True, separators=(",", ":")) + "\n")
    if digest(files_path) != dataset["file_manifest_sha256"]:
        raise ValueError("Dataset file manifest differs from pinned lock")
    return json.loads(files_path.read_text())


def download(lock, workers, endpoint, prefix="recap-official"):
    # The OS lock covers retries and completion records, including resumed runs.
    with (CACHE / f"{prefix}-download.lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return download_locked(lock, workers, endpoint, prefix)


def download_locked(lock, workers, endpoint, prefix):
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    from huggingface_hub import hf_hub_download

    state_path = CACHE / f"{prefix}-download.json"
    verified_path = CACHE / f"{prefix}-verified.jsonl"
    files = load_manifest(lock, CACHE / f"{prefix}-files.json")
    dataset = lock["dataset"]
    dest = ROOT / dataset["destination"]
    dest.mkdir(parents=True, exist_ok=True)
    complete = {}
    if verified_path.exists():
        for line in verified_path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            complete[row["path"]] = row

    def unchanged(item):
        record = complete.get(item["path"], {})
        path = dest / item["path"]
        if not path.is_file():
            return False
        stat = path.stat()
        return (record.get("revision") == dataset["revision"]
                and record.get("mtime_ns") == stat.st_mtime_ns
                and record.get("bytes") == stat.st_size == item["bytes"]
                and record.get("sha256") == item["sha256"])

    pending = [item for item in files if not unchanged(item)]
    # SFT and held-out data first; within a split, metadata precedes episodes.
    pending.sort(key=lambda x: ("_train/" in x["path"],
                                "/meta/" not in x["path"], x["path"]))
    verified_bytes = sum(x["bytes"] for x in files if unchanged(x))
    state = {"phase": "downloading", "pid": os.getpid(), "revision": dataset["revision"],
             "started_unix": time.time(), "files_total": len(files),
             "files_verified": len(files) - len(pending), "bytes_total": dataset["bytes"],
             "bytes_verified": verified_bytes, "workers": workers, "endpoint": endpoint}
    atomic_json(state_path, state)
    print(json.dumps(state), flush=True)

    def fetch(item):
        for attempt in range(12):
            try:
                path = dest / item["path"]
                if not file_ok(path, item):
                    if path.exists():
                        # Retain unexpected files for inspection; do not trust the HF cache marker.
                        path.rename(path.with_name(path.name + f".invalid-{time.time_ns()}"))
                        hf_meta = dest / ".cache/huggingface/download" / (item["path"] + ".metadata")
                        hf_meta.unlink(missing_ok=True)
                    path = Path(hf_hub_download(
                        dataset["repo"], item["path"], repo_type="dataset",
                        revision=dataset["revision"], local_dir=dest,
                        endpoint=endpoint))
                if not file_ok(path, item):
                    raise ValueError("Downloaded file checksum mismatch")
                return {"path": item["path"], "bytes": item["bytes"],
                        "sha256": item["sha256"], "revision": dataset["revision"],
                        "mtime_ns": path.stat().st_mtime_ns}
            except Exception as exc:
                print("retry", item["path"], attempt + 1, type(exc).__name__, flush=True)
                if attempt == 11:
                    raise
                time.sleep(min(30, 2 * (attempt + 1)))

    try:
        with verified_path.open("a") as journal, concurrent.futures.ThreadPoolExecutor(workers) as pool:
            futures = {pool.submit(fetch, item): item for item in pending}
            last = 0.0
            for future in concurrent.futures.as_completed(futures):
                record = future.result()
                journal.write(json.dumps(record) + "\n")
                journal.flush()
                state["files_verified"] += 1
                state["bytes_verified"] += record["bytes"]
                if time.time() - last >= 5:
                    state["updated_unix"] = time.time()
                    atomic_json(state_path, state)
                    print(f"verified {state['files_verified']}/{len(files)} files; "
                          f"{state['bytes_verified'] / 2**30:.2f}/{dataset['bytes'] / 2**30:.2f} GiB",
                          flush=True)
                    last = time.time()
    except Exception as exc:
        state.update(phase="error", error_type=type(exc).__name__, updated_unix=time.time())
        atomic_json(state_path, state)
        raise
    state.update(phase="complete", updated_unix=time.time())
    atomic_json(state_path, state)
    (dest / "DOWNLOAD_COMPLETE").write_text(dataset["revision"] + "\n")
    print(json.dumps(state), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["sources", "configs", "work-data", "download", "assets", "base", "convert-base", "status"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--wait", action="store_true", help="For convert-base: wait for verified base download")
    parser.add_argument("--endpoint", choices=["https://huggingface.co", "https://hf-mirror.com"],
                        default="https://huggingface.co")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")
    CACHE.mkdir(parents=True, exist_ok=True)
    lock = json.loads(LOCK_PATH.read_text())
    if args.action == "sources":
        prepare_sources(lock)
    elif args.action == "configs":
        prepare_configs(lock)
    elif args.action == "work-data":
        prepare_work_data(lock)
    elif args.action == "convert-base":
        convert_base(lock, args.wait)
    elif args.action == "download":
        download(lock, args.workers, args.endpoint)
    elif args.action == "base":
        download_base(min(args.workers, 4))
    elif args.action == "assets":
        assets_lock = json.loads((ROOT / "bootstrap/recap-official-assets.lock.json").read_text())
        download(assets_lock, args.workers, args.endpoint, prefix="recap-official-assets")
    else:
        print(json.dumps(live_state(STATE), indent=2) if STATE.exists() else "Download not started")
        base_state = CACHE / "recap-official-base-download.json"
        if base_state.exists():
            print("Base checkpoint:", json.dumps(live_state(base_state), indent=2))
            partials = list((ROOT / "models/pi05_base_jax").rglob("*.incomplete"))
            print("Base bytes in progress:", sum(p.stat().st_size for p in partials))
        conversion = CACHE / "recap-official-base-conversion.json"
        if conversion.exists():
            print("Base conversion:", json.dumps(live_state(conversion), indent=2))
        print("Unresolved experiment details:", json.dumps(lock["unresolved"], ensure_ascii=False))


def download_base(workers):
    """Resume pinned GCS generations; verify CRC32C (and MD5 when provided)."""
    import google_crc32c
    lock = json.loads((ROOT / "bootstrap/recap-official-base.lock.json").read_text())
    state_path = CACHE / "recap-official-base-download.json"
    with (CACHE / "recap-official-base-download.lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dest = ROOT / lock["destination"]
        dest.mkdir(parents=True, exist_ok=True)
        available = 0
        for item in lock["files"]:
            for candidate in (dest / item["path"], dest / (item["path"] + ".incomplete")):
                if candidate.is_file():
                    available += candidate.stat().st_size
        state = {"phase": "downloading", "pid": os.getpid(), "started_unix": time.time(),
                 "files_total": len(lock["files"]), "bytes_total": sum(f["bytes"] for f in lock["files"]),
                 "files_verified": 0, "bytes_verified": 0, "bytes_available": available,
                 "range_bytes": 2 * 1024 * 1024}
        progress_lock = threading.Lock()

        def progress(received=0, force=False):
            with progress_lock:
                state["bytes_available"] += received
                now = time.time()
                if force or now - state.get("updated_unix", 0) >= 5:
                    state["updated_unix"] = now
                    atomic_json(state_path, state)

        progress(force=True)

        def valid(path, item):
            if not path.is_file() or path.stat().st_size != item["bytes"]:
                return False
            md5 = hashlib.md5()
            crc = google_crc32c.Checksum()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    md5.update(block)
                    crc.update(block)
            return (base64.b64encode(crc.digest()).decode() == item["crc32c"]
                    and (not item["md5"] or base64.b64encode(md5.digest()).decode() == item["md5"]))

        def fetch(item):
            path = dest / item["path"]
            if valid(path, item):
                return item
            if path.exists():
                raise ValueError(f"Existing base checkpoint differs from lock: {item['path']}")
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + ".incomplete")
            if item["bytes"] == 0:
                partial.touch(exist_ok=True)
            obj = urllib.parse.quote(item["object"], safe="/")
            url = (f"https://storage.googleapis.com/{lock['bucket']}/{obj}"
                   f"?generation={item['generation']}")
            failures = 0
            while not partial.exists() or partial.stat().st_size < item["bytes"]:
                start = partial.stat().st_size if partial.exists() else 0
                end = min(start + state["range_bytes"], item["bytes"]) - 1
                try:
                    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(request, timeout=30) as response:
                        if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{item['bytes']}":
                            raise ValueError("GCS response did not match requested range")
                        data = response.read(end - start + 2)
                    if len(data) != end - start + 1:
                        raise ValueError("Incomplete GCS response")
                    with partial.open("ab") as output:
                        output.write(data)
                    progress(len(data))
                    failures = 0
                except Exception as exc:
                    failures += 1
                    print("retry", item["path"], start, failures, type(exc).__name__, flush=True)
                    if failures >= 12:
                        raise
                    time.sleep(min(30, failures * 2))
            if not valid(partial, item):
                raise ValueError(f"Base checkpoint checksum mismatch: {item['path']}")
            partial.replace(path)
            print("verified", item["path"], item["bytes"], flush=True)
            return item

        try:
            with concurrent.futures.ThreadPoolExecutor(workers) as pool:
                for future in concurrent.futures.as_completed([pool.submit(fetch, x) for x in lock["files"]]):
                    item = future.result()
                    with progress_lock:
                        state["files_verified"] += 1
                        state["bytes_verified"] += item["bytes"]
                    progress(force=True)
        except Exception as exc:
            state.update(phase="error", error_type=type(exc).__name__, updated_unix=time.time())
            atomic_json(state_path, state)
            raise
        state.update(phase="complete", updated_unix=time.time())
        atomic_json(state_path, state)
        (dest / "DOWNLOAD_COMPLETE").write_text(digest(ROOT / "bootstrap/recap-official-base.lock.json") + "\n")


if __name__ == "__main__":
    main()
