"""Single-environment closed-loop RoboTwin evaluation with per-episode records."""

import json
import os
from pathlib import Path
import random

from vla_mtrl.common import ROOT, compose, digest, write_json


def evaluate(args, run):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv
    from rlinf.models.embodiment.openpi import get_model

    status = json.loads((run / "status.json").read_text())
    tasks = json.loads((run / "contract.json").read_text())["tasks"]
    policies = {
        "b0": {"checkpoint": str(ROOT / ".model/pi0_robotwin"), "tasks": tasks},
        **status["jobs"],
    }
    if args.checkpoint:
        policies = {
            "custom": {
                "checkpoint": str(args.checkpoint.resolve()),
                "tasks": args.tasks,
            }
        }
    if any(info.get("phase", "completed") != "completed" for info in policies.values()):
        raise ValueError("Finish the selected training jobs before evaluating")
    seeds = json.loads(
        (ROOT / "RLinf/rlinf/envs/sim/robotwin/seeds/eval_seeds.json").read_text()
    )
    records = []
    protocol = {
        "episodes_per_task": args.episodes,
        "max_control_steps": args.max_control_steps,
        "tasks": tasks,
        "seed_file_sha256": digest(
            ROOT / "RLinf/rlinf/envs/sim/robotwin/seeds/eval_seeds.json"
        ),
    }
    protocol_path = run / "eval/protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Evaluation budget changed; use a new run for a new protocol")
    write_json(protocol_path, protocol)
    os.chdir(ROOT / "RoboTwin")
    for policy, info in policies.items():
        model_cfg = compose().actor.model
        model_cfg.model_type = "openpi"
        model_cfg.model_path = info["checkpoint"]
        model_cfg.openpi.task = "eval"
        norm = (
            ROOT
            / ".model/pi0_robotwin/physical-intelligence/robotwin/adjust_bottle/norm_stats.json"
            if policy == "b0"
            else run / "norm_stats.json"
        )
        model_cfg.openpi_data.norm_stats_path = str(norm)
        model = get_model(model_cfg).to("cuda").eval()
        checkpoint = Path(info["checkpoint"])
        model_sha = digest(
            checkpoint / "model.safetensors" if checkpoint.is_dir() else checkpoint
        )
        for task in info["tasks"]:
            cfg = OmegaConf.load(
                ROOT
                / "RLinf/examples/embodiment/config/env/robotwin_adjust_bottle.yaml"
            )
            cfg.assets_path = str(ROOT / "RoboTwin")
            cfg.seeds_path = str(
                ROOT / "RLinf/rlinf/envs/sim/robotwin/seeds/eval_seeds.json"
            )
            cfg.total_num_envs = 1
            cfg.auto_reset = False
            cfg.ignore_terminations = False
            cfg.use_custom_reward = False
            cfg.use_fixed_reset_state_ids = True
            cfg.is_eval = True
            cfg.center_crop = False
            cfg.max_episode_steps = args.max_control_steps
            cfg.task_config.task_name = task
            cfg.task_config.step_lim = args.max_control_steps
            cfg.task_config.planner_backend = "mplib"
            cfg.task_config.embodiment = ["aloha-agilex"]
            cfg.task_config.camera.collect_wrist_camera = True
            cfg.task_config.collect_data = False
            cfg.task_config.eval_video_log = False
            cfg.task_config.domain_randomization = {
                "random_background": False,
                "cluttered_table": False,
                "clean_background_rate": 1,
                "random_head_camera_dis": 0,
                "random_table_height": 0,
                "random_light": False,
                "crazy_random_light_rate": 0,
            }
            trial_seeds = seeds[task]["success_seeds"][: args.episodes]
            if len(trial_seeds) != args.episodes:
                raise ValueError("Not enough fixed evaluation seeds")
            env = RoboTwinEnv(cfg, 1, 0, 1, None)
            try:
                for seed in trial_seeds:
                    output = run / "eval" / policy / task / f"seed_{seed}.json"
                    identity = {
                        "policy": policy,
                        "task": task,
                        "requested_seed": seed,
                        "model_sha256": model_sha,
                        "norm_sha256": digest(norm),
                        "max_control_steps": args.max_control_steps,
                    }
                    if output.exists():
                        saved = json.loads(output.read_text())
                        if any(
                            saved.get(key) != value for key, value in identity.items()
                        ):
                            raise ValueError(
                                "Cannot mix evaluation policies or protocols"
                            )
                        if saved["status"] == "completed":
                            records.append(saved)
                            continue
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    steps, success = 0, False
                    try:
                        obs, _ = env.reset(env_seeds=[seed])
                        rng = torch.Generator(device="cuda").manual_seed(seed)
                        while steps < args.max_control_steps:
                            obs["task_descriptions"] = [task.replace("_", " ")]
                            with (
                                torch.inference_mode(),
                                torch.autocast("cuda", dtype=torch.bfloat16),
                            ):
                                actions, _ = model.predict_action_batch(
                                    obs, mode="eval", rng=rng
                                )
                            action_array = actions.float().cpu().numpy()
                            if (
                                action_array.ndim != 3
                                or action_array.shape[-1] != 14
                                or not np.isfinite(action_array).all()
                            ):
                                raise ValueError("Invalid policy action")
                            done = False
                            for index in range(
                                min(
                                    action_array.shape[1],
                                    args.max_control_steps - steps,
                                )
                            ):
                                obs, _, terminated, truncated, _ = env.step(
                                    action_array[:, index], auto_reset=False
                                )
                                steps += 1
                                success = bool(env.success_once[0].item())
                                done = (
                                    success or bool(terminated[0]) or bool(truncated[0])
                                )
                                if done:
                                    break
                            if done:
                                break
                        row = {
                            **identity,
                            "status": "completed",
                            "success": success,
                            "control_steps": steps,
                        }
                    except Exception as error:
                        row = {
                            **identity,
                            "status": "error",
                            "error": repr(error),
                            "control_steps": steps,
                        }
                    write_json(output, row)
                    records.append(row)
            finally:
                env.offload()
        del model
        torch.cuda.empty_cache()
    summary = {}
    for policy in policies:
        per_task = {}
        for task in policies[policy]["tasks"]:
            rows = [r for r in records if r["policy"] == policy and r["task"] == task]
            valid = [r for r in rows if r["status"] == "completed"]
            per_task[task] = {
                "successes": sum(r["success"] for r in valid),
                "completed": len(valid),
                "errors": len(rows) - len(valid),
                "planned": args.episodes,
                "success_rate": sum(r["success"] for r in valid) / len(valid)
                if valid
                else None,
            }
        rates = [r["success_rate"] for r in per_task.values()]
        summary[policy] = {
            "tasks": per_task,
            "macro_success_rate": sum(rates) / len(rates)
            if all(r is not None for r in rates)
            else None,
        }
    write_json(run / "eval/summary.json", summary)
    print(json.dumps(summary, indent=2))
    if any(r["status"] == "error" for r in records):
        raise RuntimeError("Evaluation has runtime errors; see per-episode records")
