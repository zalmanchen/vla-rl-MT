"""Exercise ZIP preparation without downloading files or requiring a GPU."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_sources import EXPECTED_REVISIONS, prepare_sources


class PrepareSourcesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "setup.sh").write_text("fixture")
        self.archives = {}
        self.manifest = self.root / "manifest.json"
        sources = {}
        for name, revision in EXPECTED_REVISIONS.items():
            (self.root / name).mkdir()
            data = (name + "-fixture").encode()
            sources[name] = {"revision": revision, "archive_url": "https://example.invalid/",
                             "files": [{"path": "code.py", "mode": "100644", "size": len(data),
                                        "sha256": hashlib.sha256(data).hexdigest()}]}
            self.archives[name] = self.root / (name + ".zip")
            with zipfile.ZipFile(self.archives[name], "w") as archive:
                archive.writestr(name + "-" + revision + "/code.py", data)
        self.manifest.write_text(json.dumps({"schema": 1, "sources": sources}))

    def prepare(self):
        return prepare_sources(self.root, self.archives, self.manifest)

    def assert_empty_sources(self):
        for name in EXPECTED_REVISIONS:
            self.assertEqual(list((self.root / name).iterdir()), [])

    def test_valid_archives_are_verified_and_installed(self):
        result = self.prepare()
        for name in EXPECTED_REVISIONS:
            self.assertEqual((self.root / name / "code.py").read_bytes(), (name + "-fixture").encode())
            self.assertEqual(result[name]["method"], "archive_content_verified")

    def test_existing_files_are_not_overwritten(self):
        sentinel = self.root / "RLinf" / "mine"
        sentinel.write_text("keep")
        with self.assertRaisesRegex(ValueError, "not an empty directory"):
            self.prepare()
        self.assertEqual(sentinel.read_text(), "keep")
        self.assertEqual(list((self.root / "RoboTwin").iterdir()), [])

    def test_wrong_revision_leaves_both_directories_empty(self):
        with zipfile.ZipFile(self.archives["RoboTwin"], "w") as archive:
            archive.writestr("RoboTwin-wrong/code.py", "content")
        with self.assertRaisesRegex(ValueError, "Unexpected path or version"):
            self.prepare()
        self.assert_empty_sources()

    def test_traversal_is_rejected(self):
        revision = EXPECTED_REVISIONS["RLinf"]
        with zipfile.ZipFile(self.archives["RLinf"], "w") as archive:
            archive.writestr("RLinf-" + revision + "/../../outside", "content")
        with self.assertRaisesRegex(ValueError, "Unexpected path or version"):
            self.prepare()
        self.assertFalse((self.root / "outside").exists())
        self.assert_empty_sources()

    def test_corrupted_archive_leaves_both_directories_empty(self):
        revision = EXPECTED_REVISIONS["RLinf"]
        with zipfile.ZipFile(self.archives["RLinf"], "w") as archive:
            archive.writestr("RLinf-" + revision + "/code.py", b"x" * len(b"RLinf-fixture"))
        with self.assertRaisesRegex(ValueError, "content differs"):
            self.prepare()
        self.assert_empty_sources()

    def test_invalid_manifest_is_rejected(self):
        self.manifest.write_text('{}')
        with self.assertRaisesRegex(ValueError, "Invalid pinned source manifest"):
            self.prepare()
        self.assert_empty_sources()

    def test_bundled_mode_installs_then_preserves_nonempty_directories(self):
        resources = self.root / "resources"
        archives = resources / "source_archives"
        archives.mkdir(parents=True)
        shutil.copyfile(str(self.manifest), str(resources / "source_manifest.json"))
        for name, filename in (("RLinf", "RLinf_e1a82ebf.zip"), ("RoboTwin", "RoboTwin_dca9ec68.zip")):
            shutil.copyfile(str(self.archives[name]), str(archives / filename))
        command = [sys.executable, str(ROOT / "scripts/prepare_sources.py"),
                   "--repo", str(self.root), "--bundled-if-empty"]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        sentinel = self.root / "RLinf" / "mine"
        sentinel.write_text("keep")
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(sentinel.read_text(), "keep")
        self.assertTrue((self.root / "RoboTwin" / "code.py").is_file())


if __name__ == "__main__":
    unittest.main()
