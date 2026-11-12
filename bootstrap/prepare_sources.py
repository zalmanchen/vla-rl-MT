"""Verify vendored source snapshots; download a locked archive if sources are absent."""

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify_files(expected, target, patches=()):
    """Verify listed file hashes, accepting only recorded compatibility patches."""
    overrides = {item["path"]: item for item in patches}
    patched, differences = [], []
    if not expected:
        raise ValueError("Source snapshot contains no files")
    for name, upstream_sha in expected.items():
        relative = Path(name)
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid source member path: {name}")
        dest = target / relative
        if not dest.resolve().is_relative_to(target.resolve()):
            raise ValueError(f"Source member resolves outside target: {relative}")
        if not dest.is_file():
            differences.append(name)
            continue
        actual = hashlib.sha256(dest.read_bytes()).hexdigest()
        if actual == upstream_sha:
            continue
        override = overrides.get(name)
        if (override is not None and upstream_sha == override["upstream_sha256"]
                and actual == override["patched_sha256"]):
            patched.append(name)
        else:
            differences.append(name)
    if differences:
        raise ValueError(f"Source mismatch ({len(differences)} files): {differences[:10]}")
    return {"checked_files": len(expected), "recorded_patch_files": patched,
            "scope": "listed source members; added assets/caches are not certified",
            "status": "verified"}


def verify_snapshot(source, archive, target, patches=()):
    """Verify the downloaded archive and its extracted source files."""
    with archive.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != source["archive_sha256"]:
        raise ValueError(f"Source archive checksum mismatch: {archive.name}")
    with zipfile.ZipFile(archive) as snapshot:
        expected = {Path(*Path(entry.filename).parts[1:]).as_posix():
                    hashlib.sha256(snapshot.read(entry)).hexdigest()
                    for entry in snapshot.infolist() if not entry.is_dir()}
    return {"revision": source["revision"], "archive_sha256": actual,
            **verify_files(expected, target, patches)}


def verify_vendored(source, manifest, target, patches=()):
    """Verify a Git-distributed snapshot without requiring a cached ZIP."""
    for key in ("repository", "revision", "archive_sha256", "path"):
        if manifest[key] != source[key]:
            raise ValueError(f"Vendored source manifest differs from source lock: {key}")
    return {"revision": source["revision"], "archive_sha256": source["archive_sha256"],
            **verify_files(manifest["files"], target, patches)}


def verify_source(name, source, cache):
    patches = []
    if name == "RLinf":
        patch = json.loads((ROOT / "bootstrap/patches/recap-robotwin.json").read_text())
        if patch["upstream_revision"] != source["revision"]:
            raise ValueError("Compatibility patch revision differs from source lock")
        patches = patch["files"]
    target = ROOT / source["path"]
    manifest_path = ROOT / "third_party/sources.manifest.json"
    result = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest["schema_version"] != 1:
            raise ValueError("Unsupported vendored source manifest schema")
        result = verify_vendored(source, manifest["sources"][name], target, patches)
    archive = cache / (name + "-upstream.zip")
    if archive.is_file():
        result = verify_snapshot(source, archive, target, patches)
    if result is None:
        raise FileNotFoundError(f"No vendored manifest or cached archive for {name}")
    return {"source": name, **result}


def main():
    locked = json.loads((ROOT / "bootstrap/sources.lock.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=sorted(locked), help="Limit to one locked source")
    parser.add_argument("--verify-only", action="store_true",
                        help="Verify vendored hashes (and ZIP if cached); do not download or patch")
    args = parser.parse_args()
    selected = {args.source: locked[args.source]} if args.source else locked
    cache = ROOT / ".runtime/bootstrap"
    if args.verify_only:
        for name, source in selected.items():
            print(json.dumps(verify_source(name, source, cache)), flush=True)
        return
    cache.mkdir(parents=True, exist_ok=True)
    for name, source in selected.items():
        target = ROOT / source["path"]
        if target.exists():
            print(json.dumps(verify_source(name, source, cache)), flush=True)
            continue
        url = source["repository"] + "/archive/" + source["revision"] + ".zip"
        archive = cache / (name + "-upstream.zip")
        urllib.request.urlretrieve(url, archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != source["archive_sha256"]:
            raise ValueError(f"Source archive checksum mismatch: {name}")
        target.mkdir(parents=True)
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                relative = Path(*Path(entry.filename).parts[1:])
                if not relative.parts:
                    continue
                dest = target / relative
                if not dest.resolve().is_relative_to(target.resolve()):
                    raise ValueError(entry.filename)
                if entry.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(z.read(entry))
                    permissions = (entry.external_attr >> 16) & 0o777
                    if permissions:
                        dest.chmod(permissions)
        print("Prepared:", target)

    if "RLinf" in selected:
        from apply_recap_patch import apply_patch

        apply_patch()


if __name__ == "__main__":
    main()
