"""Source verification must reject drift without repairing the snapshot."""

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_sources", ROOT / "bootstrap/prepare_sources.py")
sources = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sources)


class SourceSnapshotTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        archive = root / "snapshot.zip"
        with zipfile.ZipFile(archive, "w") as snapshot:
            snapshot.writestr("upstream-revision/module.py", b"original source\n")
        target = root / "source"
        target.mkdir()
        (target / "module.py").write_bytes(b"original source\n")
        source = {"repository": "https://example.invalid/locked-source",
                  "path": "third_party/Example", "revision": "revision",
                  "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
        return source, archive, target

    def test_wrong_source_is_rejected_and_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            source, archive, target = self.fixture(directory)
            changed = b"another revision or a local edit\n"
            (target / "module.py").write_bytes(changed)
            with self.assertRaisesRegex(ValueError, "Source mismatch"):
                sources.verify_snapshot(source, archive, target)
            self.assertEqual((target / "module.py").read_bytes(), changed)
            (target / "module.py").unlink()
            with self.assertRaisesRegex(ValueError, "Source mismatch"):
                sources.verify_snapshot(source, archive, target)
            self.assertFalse((target / "module.py").exists())

    def test_archive_corruption_cannot_certify_a_matching_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            source, archive, target = self.fixture(directory)
            with archive.open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "archive checksum"):
                sources.verify_snapshot(source, archive, target)

    def test_only_recorded_patch_preimage_and_postimage_are_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            source, archive, target = self.fixture(directory)
            patched = b"recorded compatibility patch\n"
            (target / "module.py").write_bytes(patched)
            patch = {"path": "module.py",
                     "upstream_sha256": hashlib.sha256(b"original source\n").hexdigest(),
                     "patched_sha256": hashlib.sha256(patched).hexdigest()}
            result = sources.verify_snapshot(source, archive, target, [patch])
            self.assertEqual(result["recorded_patch_files"], ["module.py"])
            with self.assertRaisesRegex(ValueError, "Source mismatch"):
                sources.verify_snapshot(source, archive, target,
                                        [{**patch, "upstream_sha256": "wrong revision"}])

    def test_fresh_clone_verifies_offline_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            source, archive, target = self.fixture(directory)
            root = Path(directory)
            vendored = root / source["path"]
            vendored.parent.mkdir()
            target.rename(vendored)
            archive.unlink()
            manifest = {"schema_version": 1, "sources": {
                "Example": {**source, "files": {
                    "module.py": hashlib.sha256(b"original source\n").hexdigest()}}}}
            (root / "third_party/sources.manifest.json").write_text(json.dumps(manifest))
            bootstrap = root / "bootstrap"
            bootstrap.mkdir()
            (bootstrap / "sources.lock.json").write_text(json.dumps({"Example": source}))
            script = bootstrap / "prepare_sources.py"
            script.write_bytes((ROOT / "bootstrap/prepare_sources.py").read_bytes())
            command = [sys.executable, str(script), "--verify-only", "--source", "Example"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["checked_files"], 1)
            self.assertFalse((root / ".runtime").exists())
            (vendored / "module.py").write_text("unrecorded change\n")
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("Source mismatch", failed.stderr)
            self.assertEqual((vendored / "module.py").read_text(), "unrecorded change\n")

    def test_vendored_manifest_cannot_override_lock_or_escape_root(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _, target = self.fixture(directory)
            manifest = {**source, "revision": "another revision", "files": {
                "module.py": hashlib.sha256(b"original source\n").hexdigest()}}
            with self.assertRaisesRegex(ValueError, "differs from source lock"):
                sources.verify_vendored(source, manifest, target)
            manifest.update(revision=source["revision"], files={"../outside.py": "hash"})
            with self.assertRaisesRegex(ValueError, "Invalid source member path"):
                sources.verify_vendored(source, manifest, target)


if __name__ == "__main__":
    unittest.main()
