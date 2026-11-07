"""Resume-safe single-task SFT evaluation with explicit expert seed filtering."""

import argparse
import h5py
import cv2
import importlib
import json
import logging
import os
import random
import signal
import sys
import time
import traceback
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml

LOGGER = logging.getLogger(__name__)
ROOT = Path(os.environ["PROJECT_ROOT"])
sys.path.insert(0, str(ROOT / "bootstrap"))
from smoke_model import load_model


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def timeout_handler(signum, frame):
    raise TimeoutError("Episode exceeded 900 seconds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--regime", choices=["clean", "randomized"], required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--seed-protocol",
        choices=["expert_filter", "fixed_eval"],
        default="expert_filter",
    )
    parser.add_argument("--backend", choices=["native", "reference"], default="native")
    parser.add_argument("--max-new-episodes", type=int, default=8)
    args = parser.parse_args()
    pool = json.loads((ROOT / ".runtime/experiments/e4-offline-rl/plan.json").read_text())["rollout_seeds"][args.task]
    run = args.run.resolve()
    out = run / args.regime / args.task
    out.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGALRM, timeout_handler)
    os.chdir(ROOT / "third_party/RoboTwin")
    import sapien.render

    original = sapien.render.set_ray_tracing_denoiser
    sapien.render.set_ray_tracing_denoiser = lambda name: original(
        "none" if name == "oidn" else name
    )
    from description.utils.generate_episode_instructions import (
        generate_episode_descriptions,
    )

    cls = getattr(importlib.import_module("envs." + args.task), args.task)
    config = yaml.safe_load(
        Path("task_config/demo_" + args.regime + ".yml").read_text()
    )
    robot = ROOT / "third_party/RoboTwin/assets/embodiments/aloha-agilex"
    robot_config = yaml.safe_load((robot / "config.yml").read_text())
    cam = yaml.safe_load(Path("task_config/_camera_config.yml").read_text())["D435"]
    config.update(
        task_name=args.task,
        task_config="demo_" + args.regime,
        planner_backend="mplib",
        left_robot_file=str(robot),
        right_robot_file=str(robot),
        left_embodiment_config=robot_config,
        right_embodiment_config=robot_config,
        dual_arm_embodied=True,
        head_camera_h=cam["h"],
        head_camera_w=cam["w"],
        eval_mode=True,
        eval_video_log=False,
        collect_data=False,
        save_data=False,
        render_freq=0,
        save_path=str(out / "simulator"),
    )
    episode_paths = list(out.glob("episode_*.json"))
    completed = [json.loads(p.read_text()) for p in episode_paths]
    accepted = {x["seed"] for x in completed if x["status"] == "completed"}
    if len(accepted) >= args.episodes:
        return
    candidates = pool["train"] + pool["value_validation"]
    if args.episodes > len(candidates):
        raise ValueError("Not enough locked training seeds")
    if args.backend == "reference":
        from reference_policy import ReferencePolicy

        model = ReferencePolicy()
    else:
        model = load_model(precision="fp32")
        from compat import match_reference_activations

        match_reference_activations(model)
    initial_count = len(accepted)
    errors = 0
    rejections = 0
    with (out / "events.jsonl").open("a", buffering=1) as events:
        for seed in candidates:
            if seed in accepted:
                continue
            if len(accepted) >= args.episodes or len(accepted) - initial_count >= args.max_new_episodes:
                break
            started = time.time()
            task = cls()
            phase = "expert_validation"
            try:
                signal.alarm(900)
                random.seed(seed)
                np.random.seed(seed)
                task.setup_demo(now_ep_num=0, seed=seed, is_test=True, **config)
                if args.seed_protocol == "expert_filter":
                    expert = task.play_once()
                    valid = bool(task.plan_success and task.check_success())
                    metadata = (
                        expert["info"]
                        if isinstance(expert, dict) and "info" in expert
                        else task.get_info()
                    )
                    task.close_env()
                    task = cls()
                    if not valid:
                        rejections += 1
                        events.write(
                            json.dumps(
                                {
                                    "seed": seed,
                                    "status": "seed_rejected",
                                    "reason": "expert_did_not_succeed",
                                    "seconds": time.time() - started,
                                }
                            )
                            + "\n"
                        )
                        if rejections >= 25:
                            raise RuntimeError(
                                "25 consecutive expert rejections; inspect planner before continuing"
                            )
                        continue
                    rejections = 0
                else:
                    metadata = task.get_info()
                descriptions = generate_episode_descriptions(
                    args.task, [metadata], 1, seed=seed
                )
                if not descriptions or not descriptions[0]["seen"]:
                    raise ValueError("No valid seen instruction")
                instruction = descriptions[0]["seen"][0]
                phase = "policy_rollout"
                random.seed(seed)
                np.random.seed(seed)
                if args.seed_protocol == "expert_filter":
                    task.setup_demo(now_ep_num=0, seed=seed, is_test=True, **config)
                task.set_instruction(instruction)
                rng = torch.Generator(device="cuda").manual_seed(seed)
                steps = 0
                frames = []
                trajectory = []
                record = len(accepted) < 3 or len(accepted) % 25 == 0
                action_norms = []
                while steps < task.step_lim and not task.eval_success:
                    obs = task.get_obs()
                    image = obs["observation"]["head_camera"]["rgb"]
                    wrist = np.stack(
                        [
                            obs["observation"]["left_camera"]["rgb"],
                            obs["observation"]["right_camera"]["rgb"],
                        ]
                    )
                    state = np.asarray(obs["joint_action"]["vector"])
                    if steps == 0:
                        if image.std() < 2 or wrist.std() < 2:
                            raise ValueError("Near-constant camera observation")
                        if record:
                            np.savez_compressed(
                                out / f"input_{seed}.npz",
                                head=image,
                                wrist=wrist,
                                state=state,
                                instruction=instruction,
                            )
                    if record:
                        frames.append(image.copy())
                    inputs = {
                        "main_images": image[None],
                        "wrist_images": wrist[None],
                        "states": state[None],
                        "task_descriptions": [instruction],
                    }
                    with torch.inference_mode():
                        actions, _ = model.predict_action_batch(
                            inputs, mode="eval", rng=rng
                        )
                    actions = actions[0].float().cpu().numpy()
                    if actions.shape != (32, 14) or not np.isfinite(actions).all():
                        raise ValueError("Invalid policy actions")
                    action_norms.append(float(np.linalg.norm(actions)))
                    for action_index, action in enumerate(actions):
                        current = obs if action_index == 0 else task.get_obs()
                        encoded = []
                        for camera in ["head_camera", "left_camera", "right_camera"]:
                            rgb = current["observation"][camera]["rgb"]
                            ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
                            if not ok:
                                raise ValueError("JPEG encoding failed")
                            encoded.append(jpeg)
                        trajectory.append((np.asarray(current["joint_action"]["vector"], dtype=np.float32).copy(), action.copy(), encoded))
                        task.take_action(action, action_type="qpos")
                        steps += 1
                        if task.eval_success or steps >= task.step_lim:
                            break
                    atomic_json(
                        out / "heartbeat.json",
                        {
                            "seed": seed,
                            "phase": phase,
                            "steps": steps,
                            "unix_time": time.time(),
                        },
                    )
                result = {
                    "status": "completed",
                    "task": args.task,
                    "regime": args.regime,
                    "seed": seed,
                    "instruction": instruction,
                    "success": bool(task.eval_success),
                    "steps": steps,
                    "limit": task.step_lim,
                    "seconds": time.time() - started,
                    "action_norm_mean": float(np.mean(action_norms)),
                }
                final_observation = task.get_obs()
                trajectory_path = out / f"trajectory_{seed}.hdf5"
                tmp_trajectory = trajectory_path.with_suffix(".tmp")
                with h5py.File(tmp_trajectory, "w") as h:
                    h.attrs.update(task=args.task, seed=seed, instruction=instruction, success=bool(task.eval_success),
                                   split="train" if seed in pool["train"] else "value_validation", prediction_horizon=32, execution_horizon=32)
                    h.create_dataset("state", data=np.stack([r[0] for r in trajectory]), compression="gzip")
                    h.create_dataset("action", data=np.stack([r[1] for r in trajectory]), compression="gzip")
                    rewards = np.full(len(trajectory), -1, dtype=np.float32)
                    rewards[-1] = 0 if task.eval_success else -300
                    h.create_dataset("reward", data=rewards)
                    h.create_dataset("terminated", data=np.arange(len(trajectory)) == len(trajectory)-1)
                    h.create_dataset("terminal_state", data=np.asarray(final_observation["joint_action"]["vector"], dtype=np.float32))
                    for camera_index, camera in enumerate(["head", "left_wrist", "right_wrist"]):
                        dataset = h.create_dataset("images/" + camera, (len(trajectory),), dtype=h5py.vlen_dtype(np.dtype("uint8")))
                        for frame_index, item in enumerate(trajectory):
                            dataset[frame_index] = item[2][camera_index].ravel()
                    for source, camera in [("head_camera", "head"),("left_camera", "left_wrist"),("right_camera", "right_wrist")]:
                        ok,jpeg=cv2.imencode(".jpg",cv2.cvtColor(final_observation["observation"][source]["rgb"],cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,95])
                        if not ok:raise ValueError("Terminal image encoding failed")
                        h.create_dataset("terminal_images/"+camera,data=jpeg)
                tmp_trajectory.replace(trajectory_path)
                result["trajectory"] = str(trajectory_path)
                result["dataset_split"] = "train" if seed in pool["train"] else "value_validation"
                if record:
                    imageio.mimsave(out / f"episode_{seed}.mp4", frames, fps=5)
                atomic_json(out / f"episode_{seed}.json", result)
                accepted.add(seed)
                completed.append(result)
                errors = 0
                events.write(json.dumps(result) + "\n")
                atomic_json(
                    out / "summary.json",
                    {
                        "task": args.task,
                        "regime": args.regime,
                        "episodes": len(accepted),
                        "successes": sum(
                            x["success"]
                            for x in completed
                            if x["status"] == "completed"
                        ),
                        "target": args.episodes,
                    },
                )
            except Exception as exc:
                errors += 1
                event = {
                    "status": "error",
                    "seed": seed,
                    "phase": phase,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "seconds": time.time() - started,
                }
                events.write(json.dumps(event) + "\n")
                print(json.dumps(event), flush=True)
                if (
                    args.seed_protocol == "fixed_eval"
                    or errors >= 3
                    or rejections >= 25
                ):
                    raise
            finally:
                signal.alarm(0)
                if hasattr(task, "scene"):
                    try:
                        task.close_env()
                    except Exception:
                        LOGGER.exception("Simulator cleanup failed")
                atomic_json(
                    out / "heartbeat.json",
                    {
                        "seed": seed,
                        "phase": "between_episodes",
                        "unix_time": time.time(),
                    },
                )
    if len(accepted) < args.episodes and len(accepted) == initial_count:
        raise RuntimeError("No new completed episodes")


if __name__ == "__main__":
    main()
