import concurrent.futures
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def extract(archive, target):
    marker = target / (".extracted-" + archive.name)
    if marker.exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            dest = (target / entry.filename).resolve()
            if not dest.is_relative_to(target.resolve()):
                raise ValueError(entry.filename)
        z.extractall(target)
    marker.write_text(str(archive.stat().st_size) + "\n")
    print("extracted", archive, flush=True)


if __name__ == "__main__":
    base = ROOT / "datasets/RoboTwin2.0"
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(
            lambda name: extract(base / name, ROOT / "third_party/RoboTwin/assets"),
            ["embodiments.zip", "objects.zip", "background_texture.zip"],
        ):
            pass
    print("ASSETS COMPLETE", flush=True)
    for archive in sorted((base / "dataset").glob("*/demo_clean.zip")):
        extract(archive, ROOT / "datasets/RoboTwin2.0-clean" / archive.parent.name)
    print("CLEAN DATA COMPLETE", flush=True)
