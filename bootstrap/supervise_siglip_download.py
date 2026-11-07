"""Restart resumable public-model downloads after transient network failures."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def alive(pid):
    try:
        return b"resume_siglip_ranges.py" in (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except FileNotFoundError:
        return False


def main():
    pid = 539994
    attempts = 0
    complete = ROOT / "models/siglip2-so400m-patch14-224/DOWNLOAD_COMPLETE"
    receipt = ROOT / ".runtime/bootstrap/siglip-download-supervisor.json"
    while not complete.exists():
        receipt.write_text(json.dumps({"pid": os.getpid(), "download_pid": pid, "restarts": attempts,
                                       "updated_unix": time.time(), "phase": "downloading"}))
        if alive(pid):
            time.sleep(30)
            continue
        if attempts >= 6:
            raise RuntimeError("Download retry budget exhausted; see siglip-range-download.log")
        attempts += 1
        log = (ROOT / ".runtime/logs/siglip-range-download.log").open("a")
        process = subprocess.Popen([sys.executable, "-u", "bootstrap/resume_siglip_ranges.py"],
                                   stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, start_new_session=True)
        log.close()
        pid = process.pid
        time.sleep(30)
    receipt.write_text(json.dumps({"phase": "complete", "updated_unix": time.time()}))


if __name__ == "__main__":
    main()
