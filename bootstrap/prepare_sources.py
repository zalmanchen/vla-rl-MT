"""Download pinned snapshots, or verify their cached archives and source files."""

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify_snapshot(source, archive, target, patches=()):
    """Read-only verification; added assets/caches are outside this check."""
    with archive.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != source["archive_sha256"]:
        raise ValueError(f"Source archive checksum mismatch: {archive.name}")
    overrides = {item["path"]: item for item in patches}
    checked, patched, differences = 0, [], []
    with zipfile.ZipFile(archive) as snapshot:
        for entry in snapshot.infolist():
            if entry.is_dir():
                continue
            relative = Path(*Path(entry.filename).parts[1:])
            if not relative.parts or ".." in relative.parts:
                raise ValueError(f"Invalid source archive member: {entry.filename}")
            dest = target / relative
            if not dest.resolve().is_relative_to(target.resolve()):
                raise ValueError(f"Source member resolves outside target: {relative}")
            checked += 1
            if not dest.is_file():
                differences.append(str(relative))
                continue
            expected = snapshot.read(entry)
            content = dest.read_bytes()
            if content == expected:
                continue
            override = overrides.get(relative.as_posix())
            if (override is not None
                    and hashlib.sha256(expected).hexdigest() == override["upstream_sha256"]
                    and hashlib.sha256(content).hexdigest() == override["patched_sha256"]):
                patched.append(str(relative))
            else:
                differences.append(str(relative))
    if not checked:
        raise ValueError("Source snapshot contains no files")
    if differences:
        raise ValueError(f"Source mismatch ({len(differences)} files): {differences[:10]}")
    return {"revision": source["revision"], "archive_sha256": actual,
            "checked_files": checked, "recorded_patch_files": patched,
            "scope": "locked archive members; added assets/caches are not certified",
            "status": "verified"}


def main():
    locked = json.loads((ROOT / "bootstrap/sources.lock.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=sorted(locked), help="Limit to one locked source")
    parser.add_argument("--verify-only", action="store_true",
                        help="Compare cached archive and existing files; do not download or patch")
    args = parser.parse_args()
    selected = {args.source: locked[args.source]} if args.source else locked
    cache = ROOT / ".runtime/bootstrap"
    if args.verify_only:
        for name, source in selected.items():
            patches = []
            if name == "RLinf":
                manifest = json.loads((ROOT / "bootstrap/patches/recap-robotwin.json").read_text())
                if manifest["upstream_revision"] != source["revision"]:
                    raise ValueError("Compatibility patch revision differs from source lock")
                patches = manifest["files"]
            result = verify_snapshot(source, cache / (name + "-upstream.zip"),
                                     ROOT / source["path"], patches)
            print(json.dumps({"source": name, **result}), flush=True)
        return
    cache.mkdir(parents=True, exist_ok=True)
    for name, source in selected.items():
        target = ROOT / source["path"]
        if target.exists():
            print("Already exists:", target)
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
