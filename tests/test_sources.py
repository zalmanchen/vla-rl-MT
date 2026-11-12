"""Source verification must reject drift without repairing the snapshot."""

import hashlib
import importlib.util
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
        source = {"revision": "revision",
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


if __name__ == "__main__":
    unittest.main()
