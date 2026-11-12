"""Evaluate base/no-guide, base/positive and trained/positive on paired development seeds."""

import argparse
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from collect_recap_rollouts import ROOT, TASK_INSTRUCTIONS, env_observation, make_task
from hydra import compose, initialize_config_dir


def evaluate(model, branch, task_name, seed, instruction, output):
    task = None
    started = time.time()
    frames = []
    torch.manual_seed(seed)
    try:
        task = make_task(task_name, seed, instruction)
        steps = 0
        while steps < task.step_lim and not task.eval_success:
            obs = task.get_obs()
            frames.append(obs["observation"]["head_camera"]["rgb"].copy())
            with torch.inference_mode():
                actions, _ = model.predict_action_batch(
                    env_observation(obs, instruction), mode="eval"
                )
            actions = np.asarray(actions)[0]
            if actions.shape != (32, 14) or not np.isfinite(actions).all():
                raise ValueError(f"Invalid evaluation actions: {actions.shape}")
            for action in actions:
                task.take_action(action, action_type="qpos")
                steps += 1
                if task.eval_success or steps >= task.step_lim:
                    break
        video = output / f"{branch}-{task_name}-{seed}.mp4"
        imageio.mimsave(video, frames, fps=5)
        result = {
            "branch": branch,
            "task": task_name,
            "seed": seed,
            "success": bool(task.eval_success),
            "steps": steps,
            "episode_limit": int(task.step_lim),
            "instruction": instruction,
            "elapsed_seconds": time.time() - started,
            "video": str(video),
        }
        print("EVALUATED", json.dumps(result), flush=True)
        return result
    finally:
        if task is not None:
            task.close_env()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--tasks", nargs="+", default=["click_bell", "stack_blocks_two"]
    )
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument(
        "--trained-guidance",
        choices=["positive", "no_guide"],
        default="positive",
        help="Deploy the trained checkpoint with the condition used in its training",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    if args.episodes < 1 or args.episodes > 100:
        parser.error(
            "--episodes must be in [1, 100] for this bounded development evaluation"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    seed_file = (
        ROOT / "third_party/RLinf/rlinf/envs/sim/robotwin/seeds/train_seeds.json"
    )
    seeds = json.loads(seed_file.read_text())
    from rlinf.models.embodiment.openpi_cfg import get_model

    with initialize_config_dir(config_dir=str(ROOT / "configs/robotwin_pilot"), version_base="1.1"):
        config = compose(config_name="policy").actor.model
    episodes = []
    for task in args.tasks:
        if task not in TASK_INSTRUCTIONS:
            parser.error(f"Task requires an audited instruction: {task}")
        if len(seeds[task]["success_seeds"]) < 200 + args.episodes:
            parser.error(f"Not enough reserved development seeds for {task}")
        instruction = TASK_INSTRUCTIONS[task]
        for seed in seeds[task]["success_seeds"][200 : 200 + args.episodes]:
            episodes.append((task, int(seed), instruction))
    trained_branch = f"trained_{args.trained_guidance}"
    branches_to_evaluate = ["base_no_guide", "base_positive", trained_branch]
    record = {
        "status": "running",
        "scope": "paired development integration evaluation, not an independent final test",
        "checkpoint": str(args.checkpoint.resolve()),
        "seed_selection": "official training-success seed list, indices [200,200+episodes); excluded from pilot train/validation",
        "episodes_per_task": args.episodes,
        "trained_guidance": args.trained_guidance,
        "tasks": args.tasks,
        "rollouts_return_to_training": False,
        "results": [],
    }
    report = args.output / "results.json"

    def save():
        report.write_text(json.dumps(record, indent=2) + "\n")

    save()
    for checkpoint, branches in [
        (None, ["base_no_guide", "base_positive"]),
        (args.checkpoint.resolve(), [trained_branch]),
    ]:
        if checkpoint is not None:
            config.model_path = str(checkpoint)
        print("EVAL_LOAD", config.model_path, flush=True)
        model = get_model(config).to("cuda").eval()
        try:
            for branch in branches:
                object.__setattr__(
                    model.config,
                    "guidance_type",
                    "no_guide" if branch.endswith("no_guide") else "positive",
                )
                for task, seed, instruction in episodes:
                    record["results"].append(
                        evaluate(model, branch, task, seed, instruction, args.output)
                    )
                    save()
        finally:
            del model
            torch.cuda.empty_cache()
    summary = {}
    for branch in branches_to_evaluate:
        subset = [x for x in record["results"] if x["branch"] == branch]
        summary[branch] = {
            "successes": sum(x["success"] for x in subset),
            "episodes": len(subset),
        }
    record.update(status="completed", summary=summary)
    save()
    print("EVALUATION_COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
