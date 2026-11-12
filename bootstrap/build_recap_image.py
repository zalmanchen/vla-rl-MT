"""Snapshot the verified RECAP environment into a Docker-loadable image.

`context` prepares an allowlisted Docker build context. `build` also builds
without a daemon using a CUDA rootfs, chroot and the Docker archive format.
Model weights, training data, credentials and host GPU drivers are excluded.
"""

import argparse
import datetime
import errno
import fcntl
import hashlib
import io
import json
import os
import shutil
import stat
import subprocess
import tarfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".runtime/docker/recap-official"
CONTEXT = WORK / "context"
PAYLOAD = CONTEXT / "payload"
# Use the local overlay for chroot/device nodes; /cloud is a FUSE workspace.
ROOTFS = Path("/tmp/vla-rl-mt-recap-image-rootfs")
CACHE = ROOT / ".runtime/docker"
VENV = ROOT / ".runtime/venv-recap-official"
RUNTIME_PACKAGES = ["build-essential", "git", "pkg-config", "ca-certificates", "ffmpeg", "libegl1", "libgl1", "libglib2.0-0",
                    "libgomp1", "libosmesa6", "libvulkan1", "libsm6", "libxext6", "libxrender1"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def progress(phase, **kwargs):
    data = {"phase": phase, "pid": os.getpid(), "updated_unix": time.time(), **kwargs}
    write_json(WORK / "status.json", data)
    print(json.dumps(data), flush=True)


def link_or_copy(source, target):
    # Snapshot files are read-only inputs. Only staging symlinks are replaced.
    try:
        os.link(source, target)
    except OSError as exc:
        if exc.errno == errno.EXDEV:
            shutil.copy2(source, target)
        else:
            raise


def copy_tree(source, target, *, links=False):
    shutil.copytree(source, target, symlinks=True,
                    ignore=shutil.ignore_patterns("__pycache__", ".cache", ".git"),
                    copy_function=link_or_copy if links else shutil.copy2)


def prepare_context():
    if ROOT != Path("/cloud/vla-rl-MT"):
        raise ValueError("This snapshot preserves the verified environment's /cloud/vla-rl-MT prefix")
    if (CONTEXT / "snapshot.json").exists():
        print("Using existing snapshot context", CONTEXT, flush=True)
        return json.loads((CONTEXT / "snapshot.json").read_text())
    if PAYLOAD.exists():
        raise ValueError(f"Incomplete context exists; inspect it before restarting: {PAYLOAD}")
    progress("snapshotting_environment")
    payload_root = PAYLOAD / ROOT.relative_to("/")
    payload_root.mkdir(parents=True)
    copy_tree(VENV, payload_root / ".runtime/venv-recap-official", links=True)
    probe = subprocess.check_output([str(VENV / "bin/python"), "-c",
        "import sys,json,torch,transformers,importlib.metadata as m;"
        "print(json.dumps({'python_prefix':sys.base_prefix,'python_version':sys.version.split()[0],"
        "'versions':{'torch':torch.__version__,'transformers':transformers.__version__,"
        "'mujoco':m.version('mujoco'),'rlinf-libero':m.version('rlinf-libero'),"
        "'robosuite':m.version('robosuite'),'torchcodec':m.version('torchcodec')},"
        "'packages':len(list(m.distributions()))}))"], text=True)
    runtime = json.loads(probe)
    python_prefix = Path(runtime["python_prefix"])
    target_prefix = PAYLOAD / python_prefix.relative_to("/")
    target_prefix.parent.mkdir(parents=True, exist_ok=True)
    copy_tree(python_prefix, target_prefix, links=True)
    alias = target_prefix.parent / "cpython-3.11-linux-x86_64-gnu"
    alias.symlink_to(target_prefix.name, target_is_directory=True)
    linker_config = PAYLOAD / "etc/ld.so.conf.d/recap-python.conf"
    linker_config.parent.mkdir(parents=True, exist_ok=True)
    linker_config.write_text(str(python_prefix / "lib") + "\n")

    for name in ("bootstrap", "scripts", "configs", "src", "tests", "docker", "docs"):
        copy_tree(ROOT / name, payload_root / name)
    subprocess.run([str(VENV / "bin/python"), str(ROOT / "docker/restore_console_scripts.py"),
                    "--bin-dir", str(payload_root / ".runtime/venv-recap-official/bin"),
                    "--native-bin-source", str(ROOT / ".runtime/venv-rlinf/bin"),
                    "--native-bin-source", str(ROOT / ".runtime/venv-bootstrap/bin")], check=True)
    for name in ("README.md", "AGENTS.md", "THIRD_PARTY.md", "pyproject.toml"):
        if (ROOT / name).is_file():
            shutil.copy2(ROOT / name, payload_root / name)
    archive = ROOT / ".runtime/bootstrap/RLinf-upstream.zip"
    lock = json.loads((ROOT / "bootstrap/recap-official.lock.json").read_text())
    assert digest(archive) == lock["upstream"]["archive_sha256"]
    upstream = payload_root / "third_party/RLinf-recap-official"
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            relative = Path(*Path(member.filename).parts[1:])
            if not relative.parts or member.is_dir():
                continue
            dest = upstream / relative
            if not dest.resolve().is_relative_to(upstream.resolve()):
                raise ValueError("Unsafe source archive member")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(bundle.read(member))
            dest.chmod((member.external_attr >> 16) & 0o777 or 0o644)
    assets = PAYLOAD / "opt/libero-assets"
    assets.parent.mkdir(parents=True, exist_ok=True)
    copy_tree(ROOT / "datasets/LIBERO-assets", assets, links=True)
    asset_link = payload_root / ".runtime/venv-recap-official/lib/python3.11/site-packages/libero/libero/assets"
    asset_link.unlink()
    asset_link.symlink_to("/opt/libero-assets", target_is_directory=True)
    for relative in (".runtime/cache/openpi/big_vision/paligemma_tokenizer.model",
                     ".runtime/recap-official/assets/physical-intelligence/libero/norm_stats.json",
                     ".runtime/bootstrap/RLinf-upstream.zip",
                     ".runtime/bootstrap/recap-official-files.json",
                     ".runtime/bootstrap/recap-official-assets-files.json"):
        source = ROOT / relative
        if source.is_file():
            dest = payload_root / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
    for excluded in ("models", "datasets", ".git", ".runtime/cache/huggingface"):
        assert not (payload_root / excluded).exists(), excluded
    (payload_root / "docker/entrypoint.sh").chmod(0o755)
    runtime.update(rlinf_revision=lock["upstream"]["revision"],
                   environment_freeze_sha256=digest(ROOT / "bootstrap/recap-official-environment.freeze.txt"),
                   exclusions=["model weights", "training datasets", "HF tokens", "host GPU driver"])
    write_json(PAYLOAD / "opt/recap-image/runtime.json", runtime)
    shutil.copy2(ROOT / "docker/Dockerfile.recap", CONTEXT / "Dockerfile")
    (CONTEXT / ".dockerignore").write_text("*\n!payload/\n!payload/**\n!Dockerfile\n")
    write_json(CONTEXT / "snapshot.json", runtime)
    return runtime


def build_rootfs():
    progress("preparing_cuda_rootfs")
    if not ROOTFS.exists():
        ROOTFS.mkdir()
        # This is the pinned NVIDIA rootfs exported by crane, not a host rootfs.
        with tarfile.open(CACHE / "cuda-base-rootfs.tar") as archive:
            archive.extractall(ROOTFS, filter="tar")
    if not (ROOTFS / "opt/recap-image/system-ready.json").exists():
        for name in ("dev", "dev/shm", "proc", "sys", "tmp"):
            (ROOTFS / name).mkdir(parents=True, exist_ok=True)
        (ROOTFS / "tmp").chmod(0o1777)
        for name, minor in (("null", 3), ("zero", 5), ("random", 8), ("urandom", 9)):
            device = ROOTFS / "dev" / name
            if not device.exists():
                os.mknod(device, stat.S_IFCHR | 0o666, os.makedev(1, minor))
            device.chmod(0o666)
        resolv = ROOTFS / "etc/resolv.conf"
        resolv.unlink(missing_ok=True)
        resolv.write_text(Path("/etc/resolv.conf").read_text())
        sources = ROOTFS / "etc/apt/sources.list"
        sources.write_text("deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main universe restricted multiverse\n"
                           "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main universe restricted multiverse\n"
                           "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-security main universe restricted multiverse\n")
        for cuda_repo in (ROOTFS / "etc/apt/sources.list.d").glob("cuda*.list"):
            cuda_repo.unlink()
        policy = ROOTFS / "usr/sbin/policy-rc.d"
        policy.write_text("#!/bin/sh\nexit 101\n")
        policy.chmod(0o755)
        progress("installing_system_libraries")
        clean = os.environ.copy()
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
            clean.pop(key, None)
        clean.update(DEBIAN_FRONTEND="noninteractive", LC_ALL="C.UTF-8")
        subprocess.run(["chroot", str(ROOTFS), "apt-get", "update"], env=clean, check=True)
        subprocess.run(["chroot", str(ROOTFS), "apt-get", "install", "-y", "--no-install-recommends",
                        *RUNTIME_PACKAGES], env=clean, check=True)
        subprocess.run(["chroot", str(ROOTFS), "apt-get", "clean"], env=clean, check=True)
        shutil.rmtree(ROOTFS / "var/lib/apt/lists")
        (ROOTFS / "var/lib/apt/lists").mkdir()
        resolv.write_text("")
        packages = subprocess.check_output(["chroot", str(ROOTFS), "dpkg-query", "-W"], text=True)
        system_dir = ROOTFS / "opt/recap-image"
        system_dir.mkdir(parents=True, exist_ok=True)
        (system_dir / "system-packages.txt").write_text(packages)
        write_json(system_dir / "system-ready.json", {"packages": RUNTIME_PACKAGES, "checked_unix": time.time()})
    if not (ROOTFS / "opt/recap-image/payload-ready.json").exists():
        progress("applying_environment_snapshot")
        shutil.copytree(PAYLOAD, ROOTFS, dirs_exist_ok=True, symlinks=True, copy_function=link_or_copy)
        write_json(ROOTFS / "opt/recap-image/payload-ready.json", {"checked_unix": time.time()})
    subprocess.run(["chroot", str(ROOTFS), "ldconfig"], check=True)
    progress("validating_userland")
    with (WORK / "cpu-smoke.log").open("w") as output:
        subprocess.run(["chroot", str(ROOTFS), "/usr/bin/env", "PYTHONDONTWRITEBYTECODE=1",
                        str(ROOT / "docker/entrypoint.sh"), "python", "docker/check_image.py"],
                       stdout=output, stderr=subprocess.STDOUT, check=True)
    write_json(WORK / "cpu-smoke.json", {"status": "passed", "gpu_container_verified": False,
                                        "renderer": "osmesa", "checked_unix": time.time()})


def pack_image(tag):
    smoke = json.loads((WORK / "cpu-smoke.json").read_text())
    if smoke.get("status") != "passed":
        raise RuntimeError("Userland validation must pass before creating the image")
    progress("compressing_layer")
    layer = WORK / "filesystem.tar.gz"
    sha = hashlib.sha256()
    tar_command = ["tar", "--numeric-owner", "--owner=0", "--group=0", "--sort=name",
                   "--exclude=./dev/*", "--exclude=./proc/*", "--exclude=./sys/*",
                   "--exclude=./run/*", "--exclude=./tmp/*", "--exclude=./root/.cache",
                   "-C", str(ROOTFS), "-cf", "-", "."]
    count = 0
    last = time.time()
    with layer.open("wb") as output:
        tar = subprocess.Popen(tar_command, stdout=subprocess.PIPE)
        gzip = subprocess.Popen(["gzip", "-1", "-n"], stdin=subprocess.PIPE, stdout=output)
        try:
            while block := tar.stdout.read(4 * 1024 * 1024):
                sha.update(block)
                gzip.stdin.write(block)
                count += len(block)
                if time.time() - last > 10:
                    progress("compressing_layer", uncompressed_bytes=count)
                    last = time.time()
            gzip.stdin.close()
            assert tar.wait() == 0 and gzip.wait() == 0
        except BaseException:
            tar.terminate()
            gzip.terminate()
            raise
    base = json.loads((CACHE / "base-config.json").read_text())
    config = base["config"]
    values = dict(item.split("=", 1) for item in config.get("Env", []))
    values.update(NVIDIA_DRIVER_CAPABILITIES="compute,utility,graphics", NVIDIA_VISIBLE_DEVICES="all",
                  PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false", MUJOCO_GL="egl",
                  PYOPENGL_PLATFORM="egl")
    config.update(Env=[f"{k}={v}" for k, v in values.items()], WorkingDir=str(ROOT),
                  Entrypoint=[str(ROOT / "docker/entrypoint.sh")], Cmd=["bash"], User="0")
    config.setdefault("Labels", {}).update({"org.opencontainers.image.title": "RLinf RECAP verified environment",
        "org.opencontainers.image.base.digest": (CACHE / "base-digest.txt").read_text().strip(),
        "org.opencontainers.image.description": "LIBERO / pi0.5 / RECAP userland snapshot; model weights and training data mounted externally"})
    metadata = {"architecture": "amd64", "os": "linux", "config": config,
                "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "rootfs": {"type": "layers", "diff_ids": ["sha256:" + sha.hexdigest()]},
                "history": [{"created_by": "verified RECAP environment snapshot"}]}
    blob = json.dumps(metadata, separators=(",", ":")).encode()
    image_id = hashlib.sha256(blob).hexdigest()
    config_name = image_id + ".json"
    layer_name = digest(layer) + ".tar.gz"
    manifest = [{"Config": config_name, "RepoTags": [tag], "Layers": [layer_name]}]
    dest = WORK / "vla-rl-mt-recap-official.tar"
    progress("writing_docker_archive", layer_bytes=layer.stat().st_size)
    with tarfile.open(dest.with_suffix(".tar.tmp"), "w") as archive:
        for name, body in [(config_name, blob), ("manifest.json", json.dumps(manifest).encode())]:
            member = tarfile.TarInfo(name)
            member.size = len(body)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(body))
        archive.add(layer, arcname=layer_name, recursive=False)
    dest.with_suffix(".tar.tmp").replace(dest)
    progress("validating_archive")
    subprocess.run([str(CACHE / "crane"), "validate", "--tarball", str(dest)], check=True)
    report = {"phase": "complete", "tag": tag, "image_id": "sha256:" + image_id,
              "archive": str(dest), "archive_bytes": dest.stat().st_size, "archive_sha256": digest(dest),
              "uncompressed_layer_bytes": count, "diff_id": metadata["rootfs"]["diff_ids"][0],
              "base_digest": (CACHE / "base-digest.txt").read_text().strip(),
              "validation": {"docker_archive": "crane validate passed", "cpu_physics_video": "passed",
                             "gpu_container": "not run: no Docker daemon/socket in this container"}}
    write_json(WORK / "image.json", report)
    (WORK / "SHA256SUMS").write_text(report["archive_sha256"] + "  " + dest.name + "\n")
    progress("complete", **{k: v for k, v in report.items() if k != "phase"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["context", "build", "pack"])
    parser.add_argument("--tag", default="vla-rl-mt:recap-official-20261111")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / "build.lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "pack":
            pack_image(args.tag)
            return
        prepare_context()
        if args.action == "build":
            if os.getuid() != 0:
                raise RuntimeError("Daemonless build needs root with CAP_SYS_CHROOT; use docker build with the context otherwise")
            build_rootfs()
            pack_image(args.tag)


if __name__ == "__main__":
    main()
