"""Download pinned upstream source snapshots without nested Git repositories."""

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    locked = json.loads((ROOT / "bootstrap/sources.lock.json").read_text())
    cache = ROOT / ".runtime/bootstrap"
    cache.mkdir(parents=True, exist_ok=True)
    for name, source in locked.items():
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


if __name__ == "__main__":
    main()
