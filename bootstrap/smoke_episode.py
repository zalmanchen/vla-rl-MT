"""Run one bounded ALOHA handover episode using the RLinf Pi0.5 factory."""

import importlib
import json
import os
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from smoke_model import load_model

ROOT = Path(os.environ["PROJECT_ROOT"])
ROBOTWIN = ROOT / "third_party/RoboTwin"


def main():
    os.chdir(ROBOTWIN)
    import sapien.render

    set_denoiser = sapien.render.set_ray_tracing_denoiser
    sapien.render.set_ray_tracing_denoiser = lambda name: set_denoiser(
        "none" if name == "oidn" else name
    )
    args = yaml.safe_load((ROBOTWIN / "task_config/demo_clean.yml").read_text())
    robot = ROBOTWIN / "assets/embodiments/aloha-agilex"
    args.update(
        task_name="handover_block",
        task_config="demo_clean",
        planner_backend="mplib",
        left_robot_file=str(robot),
        right_robot_file=str(robot),
        dual_arm_embodied=True,
        eval_mode=True,
        eval_video_log=False,
        collect_data=False,
        save_data=False,
        step_lim=320,
        render_freq=0,
    )
    args["left_embodiment_config"] = yaml.safe_load((robot / "config.yml").read_text())
    args["right_embodiment_config"] = args["left_embodiment_config"]
    camera = yaml.safe_load((ROBOTWIN / "task_config/_camera_config.yml").read_text())[
        "D435"
    ]
    args.update(head_camera_h=camera["h"], head_camera_w=camera["w"])
    task = importlib.import_module("envs.handover_block").handover_block()
    instruction = (
        "Use the left arm to pick up the block and hand it over to the right arm."
    )
    started = time.time()
    try:
        task.setup_demo(now_ep_num=0, seed=100000, is_test=True, **args)
        task.set_instruction(instruction)
        limit = task.step_lim
        print("ENV READY; limit", limit, flush=True)
        model = load_model()
        frames = []
        steps = 0
        while steps < limit and not task.eval_success:
            obs = task.get_obs()
            image = obs["observation"]["head_camera"]["rgb"]
            frames.append(image.copy())
            env_obs = {
                "main_images": image[None],
                "wrist_images": np.stack(
                    [
                        obs["observation"]["left_camera"]["rgb"],
                        obs["observation"]["right_camera"]["rgb"],
                    ]
                )[None],
                "states": np.asarray(obs["joint_action"]["vector"])[None],
                "task_descriptions": [instruction],
            }
            with torch.inference_mode():
                actions, _ = model.predict_action_batch(env_obs, mode="eval")
            actions = actions[0].float().cpu().numpy()
            if actions.shape != (32, 14) or not np.isfinite(actions).all():
                raise RuntimeError(f"Invalid actions: {actions.shape}")
            for action in actions:
                task.take_action(action, action_type="qpos")
                steps += 1
                if task.eval_success or steps >= limit:
                    break
            print("steps", steps, "success", bool(task.eval_success), flush=True)
        output = ROOT / ".runtime/validation"
        output.mkdir(exist_ok=True)
        imageio.mimsave(output / "handover_block.mp4", frames, fps=5)
        result = {
            "task": "handover_block",
            "seed": 100000,
            "model": "motus-robotics/pi0.5_robotwin2",
            "model_revision": "effc6e1a99bd4d0edf9447e8ba313def67649cfe",
            "rlinf_revision": "e1a82ebf0c1af69da6ab9b1519f5f3692fe33566",
            "action_horizon": 32,
            "steps": steps,
            "success": bool(task.eval_success),
            "episode_limit": limit,
            "render_denoiser": "none",
            "elapsed_seconds": time.time() - started,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "status": "passed_integration_smoke",
        }
        (output / "handover_block.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        if hasattr(task, "scene"):
            task.close_env()


if __name__ == "__main__":
    main()
