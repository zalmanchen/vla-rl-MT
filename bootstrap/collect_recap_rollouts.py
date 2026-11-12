"""Record a bounded, frozen rollout pool using the official RLinf Pi0.5 model."""

import argparse
import importlib
import json
import os
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import yaml
from smoke_model import load_model

ROOT = Path(__file__).resolve().parents[1]
ROBOTWIN = ROOT / "third_party/RoboTwin"
CAMERAS = {
    "cam_high": "head_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}
TASK_INSTRUCTIONS = {
    "click_bell": "Press the top center of the bell.",
    "stack_blocks_two": "Stack the green block on top of the red block.",
}


def make_task(name, seed, instruction):
    os.chdir(ROBOTWIN)
    import sapien.render

    if not getattr(sapien.render.set_ray_tracing_denoiser, "_recap", False):
        original = sapien.render.set_ray_tracing_denoiser

        def denoiser(value):
            return original("none" if value == "oidn" else value)

        denoiser._recap = True
        sapien.render.set_ray_tracing_denoiser = denoiser
    args = yaml.safe_load((ROBOTWIN / "task_config/demo_clean.yml").read_text())
    robot = ROBOTWIN / "assets/embodiments/aloha-agilex"
    args.update(
        task_name=name,
        task_config="demo_clean",
        planner_backend="mplib",
        left_robot_file=str(robot),
        right_robot_file=str(robot),
        dual_arm_embodied=True,
        eval_mode=True,
        eval_video_log=False,
        collect_data=False,
        save_data=False,
        render_freq=0,
    )
    args["left_embodiment_config"] = yaml.safe_load((robot / "config.yml").read_text())
    args["right_embodiment_config"] = args["left_embodiment_config"]
    camera = yaml.safe_load((ROBOTWIN / "task_config/_camera_config.yml").read_text())[
        "D435"
    ]
    args.update(head_camera_h=camera["h"], head_camera_w=camera["w"])
    task = getattr(importlib.import_module(f"envs.{name}"), name)()
    task.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
    task.set_instruction(instruction)
    return task


def env_observation(obs, instruction):
    return {
        "main_images": obs["observation"]["head_camera"]["rgb"][None],
        "wrist_images": np.stack(
            [
                obs["observation"]["left_camera"]["rgb"],
                obs["observation"]["right_camera"]["rgb"],
            ]
        )[None],
        "states": np.asarray(obs["joint_action"]["vector"])[None],
        "task_descriptions": [instruction],
    }


def collect(model, name, seed, split, output):
    destination = output / split / name / f"episode_{seed:09d}.hdf5"
    if destination.exists():
        with h5py.File(destination) as h:
            if h.attrs.get("complete", False):
                print("ROLLOUT_EXISTS", str(destination), flush=True)
                return
        raise ValueError(f"Incomplete final rollout: {destination}")
    instruction = TASK_INSTRUCTIONS[name]
    task = None
    started = time.time()
    torch.manual_seed(seed)
    states, actions, images = [], [], {k: [] for k in CAMERAS}
    try:
        task = make_task(name, seed, instruction)
        obs = task.get_obs()
        limit = int(task.step_lim)
        print("ROLLOUT_START", name, split, seed, "limit", limit, flush=True)
        while len(actions) < limit and not task.eval_success:
            with torch.inference_mode():
                chunk, _ = model.predict_action_batch(
                    env_observation(obs, instruction), mode="eval"
                )
            chunk = chunk[0].float().cpu().numpy()
            if chunk.shape != (32, 14) or not np.isfinite(chunk).all():
                raise ValueError(f"Invalid model actions: {chunk.shape}")
            for action in chunk:
                states.append(
                    np.asarray(obs["joint_action"]["vector"], dtype=np.float32)
                )
                actions.append(action.copy())
                for camera, key in CAMERAS.items():
                    rgb = obs["observation"][key]["rgb"]
                    ok, jpeg = cv2.imencode(
                        ".jpg",
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95],
                    )
                    if not ok:
                        raise ValueError("Camera JPEG encoding failed")
                    images[camera].append(jpeg.reshape(-1))
                task.take_action(action, action_type="qpos")
                obs = task.get_obs()
                if task.eval_success or len(actions) >= limit:
                    break
            print(
                "ROLLOUT_STEPS",
                name,
                seed,
                len(actions),
                bool(task.eval_success),
                flush=True,
            )
        if not actions:
            raise ValueError("Empty rollout")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".partial.hdf5")
        with h5py.File(temporary, "w") as h:
            h.create_dataset("state", data=np.stack(states))
            h.create_dataset("action", data=np.stack(actions))
            for camera, frames in images.items():
                d = h.create_dataset(
                    f"images/{camera}",
                    (len(frames),),
                    dtype=h5py.vlen_dtype(np.dtype("uint8")),
                )
                for i, jpeg in enumerate(frames):
                    d[i] = jpeg
            h.attrs.update(
                task=name,
                instruction=instruction,
                seed=seed,
                split=split,
                source="rlinf_native_pi05_rollout",
                fps=15,
                success=bool(task.eval_success),
                complete=True,
                episode_limit=limit,
                termination="success" if task.eval_success else "task_time_limit",
                elapsed_seconds=time.time() - started,
                model_revision="effc6e1a99bd4d0edf9447e8ba313def67649cfe",
                rlinf_revision="e1a82ebf0c1af69da6ab9b1519f5f3692fe33566",
            )
        temporary.replace(destination)
        print(
            "ROLLOUT_SAVED",
            str(destination),
            "success",
            bool(task.eval_success),
            "seconds",
            round(time.time() - started, 1),
            flush=True,
        )
    finally:
        if task is not None:
            task.close_env()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", nargs="+", default=["click_bell", "stack_blocks_two"])
    p.add_argument("--train-episodes", type=int, default=4)
    p.add_argument("--val-episodes", type=int, default=1)
    p.add_argument("--output", type=Path, default=ROOT / ".runtime/recap-rollouts-v2")
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.train_episodes < 1 or args.val_episodes < 1:
        p.error("Both train and validation episode counts must be positive")
    if not set(args.tasks).issubset(TASK_INSTRUCTIONS):
        p.error(
            "Each task needs a source-verified task-level instruction before collection"
        )
    seeds = json.loads(
        (
            ROOT / "third_party/RLinf/rlinf/envs/sim/robotwin/seeds/train_seeds.json"
        ).read_text()
    )
    model = load_model()
    for task in args.tasks:
        valid = seeds[task]["success_seeds"]
        if set(valid[16 : 16 + args.train_episodes]) & set(
            valid[100 : 100 + args.val_episodes]
        ):
            raise ValueError("Requested training and validation seed ranges overlap")
        for split, selected in [
            ("train", valid[16 : 16 + args.train_episodes]),
            ("val", valid[100 : 100 + args.val_episodes]),
        ]:
            for seed in selected:
                collect(model, task, int(seed), split, args.output)
    print("ROLLOUT_POOL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
