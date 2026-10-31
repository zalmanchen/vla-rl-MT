"""CPU-only checks for Git and GitHub ZIP source provenance."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vla_mtrl.source_identity import EXPECTED_REVISIONS, verify_sources


class SourceIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="source-identity-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        sources = {}
        for name, revision in EXPECTED_REVISIONS.items():
            source = self.root / name
            source.mkdir()
            data = (name + "-source").encode()
            (source / "code.py").write_bytes(data)
            sources[name] = {
                "revision": revision,
                "archive_url": "https://github.com/example/" + name,
                "files": [{"path": "code.py", "mode": "100644", "size": len(data),
                           "sha256": hashlib.sha256(data).hexdigest()}],
            }
        self.manifest = self.root / "manifest.json"
        self.data = {"schema": 1, "sources": sources}
        self.write_manifest()

    def write_manifest(self):
        self.manifest.write_text(json.dumps(self.data))

    def test_zip_files_verify_without_git_and_ignore_extra_runtime_assets(self):
        (self.root / "RoboTwin" / "downloaded_asset.bin").write_bytes(b"asset")
        with patch("vla_mtrl.source_identity.shutil.which", return_value=None):
            result = verify_sources(self.root, self.manifest)
        self.assertEqual(result["RLinf"]["method"], "archive_content_verified")
        self.assertFalse(result["RLinf"]["git_head_verified"])
        self.assertEqual(result["RoboTwin"]["tracked_files_verified"], 1)

    def test_same_size_modified_file_is_rejected(self):
        path = self.root / "RLinf/code.py"
        path.write_bytes(b"x" * path.stat().st_size)
        with self.assertRaisesRegex(ValueError, "RLinf.*content differs"):
            verify_sources(self.root, self.manifest)

    def test_missing_source_gives_archive_guidance(self):
        (self.root / "RoboTwin/code.py").unlink()
        with self.assertRaisesRegex(ValueError, "RoboTwin.*missing.*Download"):
            verify_sources(self.root, self.manifest)

    def test_symlink_materialized_by_zip_is_recorded(self):
        data = b"../target"
        (self.root / "RLinf/doclink").write_bytes(data)
        self.data["sources"]["RLinf"]["files"].append(
            {"path": "doclink", "mode": "120000", "size": len(data),
             "sha256": hashlib.sha256(data).hexdigest()}
        )
        self.write_manifest()
        result = verify_sources(self.root, self.manifest)
        self.assertEqual(result["RLinf"]["materialized_symlinks"], ["doclink"])

    def test_manifest_path_cannot_escape_source(self):
        self.data["sources"]["RLinf"]["files"][0]["path"] = "../outside"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Unsafe source manifest path"):
            verify_sources(self.root, self.manifest)

    @unittest.skipUnless(shutil.which("git"), "requires Git to test incorrect HEAD")
    def test_git_checkout_at_wrong_commit_is_not_mislabeled_as_pinned(self):
        repo = self.root / "RLinf"
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "add", "code.py"], check=True)
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=test", "-c",
                        "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
        with self.assertRaisesRegex(ValueError, "RLinf Git HEAD"):
            verify_sources(self.root, self.manifest)


if __name__ == "__main__":
    unittest.main()
