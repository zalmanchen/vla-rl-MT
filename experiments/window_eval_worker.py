"""Resume-safe single-task SFT evaluation with explicit expert seed filtering."""

import argparse
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
    parser.add_argument("--execution-horizon", type=int, choices=[8, 16, 32], required=True)
    args = parser.parse_args()
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
    seed_file = ROOT / "third_party/RLinf/rlinf/envs/sim/robotwin/seeds/eval_seeds.json"
    known = (
        json.loads(seed_file.read_text()).get(args.task, {}).get("success_seeds", [])
    )
    if args.seed_protocol == "fixed_eval":
        if args.regime != "clean" or len(known) < args.episodes:
            raise ValueError("Fixed eval requires Clean and enough published seeds")
        frozen = json.loads((ROOT / ".runtime/experiments/e1-reference-clean/seed_manifest.json").read_text())
        candidates = frozen["tasks"][args.task]["valid_seeds"][:args.episodes]
    else:
        candidates = list(known) + list(range(200000000, 200010000))
    candidates = list(dict.fromkeys(candidates))
    if args.backend == "reference":
        from reference_policy import ReferencePolicy

        model = ReferencePolicy()
    else:
        model = load_model(precision="fp32")
        from compat import match_reference_activations

        match_reference_activations(model)
    errors = 0
    rejections = 0
    with (out / "events.jsonl").open("a", buffering=1) as events:
        for seed in candidates:
            if seed in accepted:
                continue
            if len(accepted) >= args.episodes:
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
                    for action in actions[:args.execution_horizon]:
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
                    "execution_horizon": args.execution_horizon,
                }
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
    if len(accepted) < args.episodes:
        raise RuntimeError("Candidate seeds exhausted")


if __name__ == "__main__":
    main()
