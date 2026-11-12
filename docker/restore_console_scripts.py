"""Restore missing console entrypoints after snapshotting a virtualenv."""

import argparse
import base64
import hashlib
import importlib.metadata
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, default=Path(sys.prefix) / "bin")
    parser.add_argument("--native-bin-source", type=Path, action="append", default=[])
    args = parser.parse_args()
    args.bin_dir.mkdir(parents=True, exist_ok=True)
    existing = Path(sys.prefix) / "bin"
    restored = []
    for distribution in importlib.metadata.distributions():
        for entry in distribution.entry_points:
            if entry.group != "console_scripts" or (existing / entry.name).exists():
                continue
            if Path(entry.name).name != entry.name:
                raise ValueError("Unexpected console entrypoint name")
            target = args.bin_dir / entry.name
            if target.exists():
                continue
            code = (f"#!{sys.prefix}/bin/python\n"
                    "import sys\nfrom importlib.metadata import distribution\n"
                    f"entry = next(e for e in distribution({distribution.metadata['Name']!r}).entry_points "
                    f"if e.group == 'console_scripts' and e.name == {entry.name!r})\n"
                    "sys.exit(entry.load()())\n")
            target.write_text(code)
            target.chmod(0o755)
            restored.append(entry.name)
        for item in distribution.files or []:
            if not str(item).startswith("../../../bin/") or not item.hash or item.hash.mode != "sha256":
                continue
            target = args.bin_dir / item.name
            if target.exists() or (existing / item.name).exists():
                continue
            for source_dir in args.native_bin_source:
                source = source_dir / item.name
                if not source.is_file():
                    continue
                data = source.read_bytes()
                checksum = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
                if data.startswith(b"\x7fELF") and checksum == item.hash.value:
                    shutil.copy2(source, target)
                    restored.append(item.name)
                    break
    print("Restored console entrypoints:", ", ".join(sorted(restored)) or "none")


if __name__ == "__main__":
    main()
