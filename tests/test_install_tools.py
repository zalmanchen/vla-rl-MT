"""Exercise installation safety and Docker build orchestration without a daemon."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SHA = "f0d3989656be8c507096c04a30b8b5ce4ffad2f9371787edac24fe76a9453054"


def executable(path, content):
    path.write_text("#!/bin/bash\nset -e\n" + content)
    path.chmod(0o755)


class InstallToolsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.project = self.root / "project"
        (self.project / "scripts").mkdir(parents=True)
        (self.project / "docker").mkdir()
        shutil.copy2(ROOT / "scripts/install_local.sh", self.project / "scripts")
        shutil.copy2(ROOT / "scripts/download_dependencies.sh", self.project / "scripts")
        shutil.copy2(ROOT / "docker/build_image.sh", self.project / "docker")
        executable(self.project / "setup.sh", 'printf "%s|%s|%s|%s\\n" "${UV_OFFLINE:-}" "${PIP_NO_INDEX:-}" "${UV_DEFAULT_INDEX:-}" "$*" > "$TEST_CAPTURE"\n')
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.bundle = self.root / "input.tar.gz"
        self.bundle.write_bytes(b"fixture")
        executable(self.bin / "sha256sum", 'printf "%s  %s\\n" "${FAKE_SHA}" "$1"\n')
        executable(self.bin / "tar", 'if [[ "$1" == -xzf ]]; then d="$4"; mkdir -p "$d/tools/bin" "$d/tools/python" "$d/cache/uv"; touch "$d/tools/bin/uv"; chmod +x "$d/tools/bin/uv"; if [[ -n "${FAKE_INDEX:-}" ]]; then printf "%s\\n" "$FAKE_INDEX" > "$d/cache-index.txt"; fi; else exec /usr/bin/tar "$@"; fi\n')
        self.capture = self.root / "capture"
        self.env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
                        TEST_CAPTURE=str(self.capture), FAKE_SHA=SHA)

    def install(self, *args):
        return subprocess.run(["bash", str(self.project / "scripts/install_local.sh"), *args],
                              env=self.env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_offline_install_and_reuse(self):
        destination = self.root / "bundle"
        args = ("--bundle", str(self.bundle), "--bundle-dir", str(destination), "--venv", "/tmp/example")
        self.assertEqual(self.install(*args).returncode, 0)
        self.assertIn("1|1|https://mirrors.aliyun.com/pypi/simple/|--venv /tmp/example", self.capture.read_text())
        sentinel = destination / "user-data"
        sentinel.write_text("keep")
        self.assertEqual(self.install(*args).returncode, 0)
        self.assertEqual(sentinel.read_text(), "keep")

    def test_bad_digest_or_existing_directory_never_overwrites(self):
        destination = self.root / "bundle"
        destination.mkdir()
        (destination / "mine").write_text("keep")
        self.env["FAKE_SHA"] = "0" * 64
        bad = self.install("--bundle", str(self.bundle), "--bundle-dir", str(destination))
        self.assertNotEqual(bad.returncode, 0)
        self.env["FAKE_SHA"] = SHA
        existing = self.install("--bundle", str(self.bundle), "--bundle-dir", str(destination))
        self.assertNotEqual(existing.returncode, 0)
        self.assertEqual((destination / "mine").read_text(), "keep")

    def test_online_mode_configures_index_and_clears_offline_flags(self):
        self.env.update(UV_OFFLINE="1", PIP_NO_INDEX="1")
        result = self.install("--online", "--index", "https://example.invalid/simple/", "--train-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.capture.read_text().strip(), "||https://example.invalid/simple/|--train-only")

    def test_generated_bundle_uses_supplied_digest_and_recorded_index(self):
        self.env.update(FAKE_SHA="a" * 64, FAKE_INDEX="https://example.invalid/simple/")
        result = self.install("--bundle", str(self.bundle), "--sha256", "a" * 64)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("1|1|https://example.invalid/simple/|", self.capture.read_text())

    def test_tokenizer_allows_an_alternative_url_and_keeps_the_pinned_hash(self):
        env = dict(os.environ, VLA_TOKENIZER_URL="https://example.invalid/tokenizer.model")
        result = subprocess.run(["python3", str(ROOT / "scripts/prepare_tokenizer.py"), "--dry-run"],
                                env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["source"], "https://example.invalid/tokenizer.model")
        self.assertEqual(plan["sha256"], "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6")

    def test_docker_build_context_and_export_without_daemon(self):
        for name in ("README.md", "THIRD_PARTY.md", "run.sh", "train.sh", "train_bc.py", "check.sh", "download.sh", "evaluate.sh"):
            (self.project / name).write_text("fixture")
        for name in ("configs", "docs", "requirements", "resources", "vla_mtrl"):
            (self.project / name).mkdir()
            (self.project / name / "fixture").write_text("fixture")
        (self.project / "docker/Dockerfile").write_text("FROM fixture")
        executable(self.bin / "docker", 'case "$1" in info) exit 0;; build) for last; do :; done; test -f "$last/offline-bundle.tar.gz"; test -f "$last/project/scripts/install_local.sh";; image) test "$2" = save; printf image > "$4";; *) exit 1;; esac\n')
        output = self.root / "image.tar"
        cmd = ["bash", str(self.project / "docker/build_image.sh"), "--bundle", str(self.bundle), "--output", str(output)]
        result = subprocess.run(cmd, env=self.env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(output.read_text(), "image")
        result = subprocess.run(cmd, env=self.env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(output.read_text(), "image")

    def test_download_bundle_records_index_and_checks_offline(self):
        requirements = self.project / "requirements"
        requirements.mkdir()
        for name in ("runtime-linux-py311.lock", "simulator-linux-py311.lock"):
            (requirements / name).write_text("fixture\n")
        executable(self.project / "setup.sh", 'mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"; if [[ "${UV_OFFLINE:-}" == 1 ]]; then test -f "$UV_CACHE_DIR/fixture"; printf offline >> "$TEST_CAPTURE"; else printf cache > "$UV_CACHE_DIR/fixture"; printf python > "$UV_PYTHON_INSTALL_DIR/fixture"; fi\n')
        uv = self.root / "uv"
        executable(uv, 'echo "uv 0.13.6 (test)"\n')
        output = self.root / "portable.tar.gz"
        env = dict(os.environ, TEST_CAPTURE=str(self.capture))
        cmd = ["bash", str(self.project / "scripts/download_dependencies.sh"), "--uv-bin", str(uv),
               "--work-dir", str(self.root / "work"), "--index", "https://example.invalid/simple/",
               "--python-mirror", "official", "--output", str(output)]
        result = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.capture.read_text(), "offline")
        with tarfile.open(output, "r:gz") as archive:
            self.assertEqual(archive.extractfile("cache-index.txt").read(), b"https://example.invalid/simple/\n")
            self.assertEqual(archive.extractfile("cache/uv/fixture").read(), b"cache")
        self.assertTrue(Path(str(output) + ".sha256").is_file())
        self.assertNotEqual(subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode, 0)


if __name__ == "__main__":
    unittest.main()
