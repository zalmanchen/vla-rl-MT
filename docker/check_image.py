"""Check the packaged runtime; --gpu additionally requires CUDA and EGL."""

import argparse
import importlib.metadata
import json
import os
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    # CPU validation also exercises LIBERO physics/assets through OSMesa.
    # GPU validation uses the image's normal EGL path.
    if not args.gpu:
        os.environ["MUJOCO_GL"] = "osmesa"
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    import av
    import numpy as np
    import rlinf
    import torch
    import transformers
    from torchcodec.decoders import VideoDecoder

    root = Path("/cloud/vla-rl-MT")
    expected = json.loads(Path("/opt/recap-image/runtime.json").read_text())
    actual = {"torch": torch.__version__, "transformers": transformers.__version__}
    for key, value in actual.items():
        assert value == expected["versions"][key], (key, value, expected["versions"][key])
    assert str(root / "third_party/RLinf-recap-official") in rlinf.__file__, rlinf.__file__
    assert importlib.metadata.version("mujoco") == expected["versions"]["mujoco"]
    subprocess.run([str(root / ".runtime/venv-recap-official/bin/python"), "-m", "pip", "check"], check=True)
    assert torch.isfinite(torch.ones(4, 4) @ torch.ones(4, 4)).all()
    if args.gpu:
        assert torch.cuda.is_available(), "CUDA unavailable: run with --gpus all and NVIDIA Container Toolkit"
        x = torch.ones(16, 16, device="cuda", dtype=torch.bfloat16)
        assert torch.isfinite(x @ x).all()

    with tempfile.TemporaryDirectory() as directory:
        video = Path(directory) / "codec.mp4"
        with av.open(str(video), "w") as container:
            stream = container.add_stream("libx264", rate=10)
            stream.width = stream.height = 32
            stream.pix_fmt = "yuv420p"
            for index in range(3):
                frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), 32 + index * 32, dtype=np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        assert tuple(VideoDecoder(str(video))[0].shape) == (3, 32, 32)

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(0)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128)
    try:
        env.seed(0)
        env.reset()
        env.set_init_state(suite.get_task_init_states(0)[0])
        for _ in range(4):
            observation, *_ = env.step(np.zeros(7, dtype=np.float32))
        images = {key: list(value.shape) for key, value in observation.items() if "image" in key}
        assert len(images) == 2
        assert all(observation[key].std() > 1 for key in images)
    finally:
        env.close()
    print(json.dumps({"status": "passed", "gpu_verified": args.gpu,
                      "renderer": os.environ["MUJOCO_GL"], "versions": actual,
                      "task": task.language, "images": images}, indent=2))


if __name__ == "__main__":
    main()
