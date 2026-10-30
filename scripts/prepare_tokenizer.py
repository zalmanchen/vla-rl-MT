#!/usr/bin/env python3
"""Prepare the PaliGemma tokenizer required by the pinned OpenPI runtime."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

SOURCE = "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
EXPECTED_SHA256 = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
EXPECTED_BYTES = 4264023


def verify(path):
    if (
        path.stat().st_size != EXPECTED_BYTES
        or hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED_SHA256
    ):
        raise ValueError(f"PaliGemma tokenizer size/SHA256 mismatch: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    # Match OpenPI's own cache lookup, including the optional environment override.
    target = (
        Path(os.environ.get("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser()
        / "big_vision/paligemma_tokenizer.model"
    )
    if args.dry_run:
        print(
            json.dumps(
                {"source": SOURCE, "target": str(target), "sha256": EXPECTED_SHA256},
                indent=2,
            )
        )
        return
    if target.exists():
        verify(target)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix="tokenizer.", delete=False
        ) as stream:
            temporary = Path(stream.name)
        try:
            subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--retry",
                    "3",
                    "--connect-timeout",
                    "15",
                    "--max-time",
                    "300",
                    SOURCE,
                    "--output",
                    str(temporary),
                ],
                check=True,
            )
            verify(temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "source": SOURCE,
                "path": str(target),
                "sha256": EXPECTED_SHA256,
                "bytes": EXPECTED_BYTES,
                "verified": True,
            }
        )
    )


if __name__ == "__main__":
    main()
