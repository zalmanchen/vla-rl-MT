"""Verify feasible same-scene goals before paired instruction interventions."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"bootstrap"))
from evaluate_multitask import digest, write_json


def goal_predicates(task, family):
    if family == "pick_dual_bottles":
        poses = [np.asarray(actor.get_functional_point(0)) for actor in (task.bottle1, task.bottle2)]
        targets = [task.left_target_pose, task.right_target_pose]
        hit = [bool(np.all(np.abs(p[:2]-np.asarray(t[:2])) < .1) and p[2] > .89)
               for p, t in zip(poses, targets, strict=True)]
        return {"left": hit[0] and bool(poses[1][2] <= .84),
                "right": hit[1] and bool(poses[0][2] <= .84), "both": all(hit)}
    red, green = np.asarray(task.block1.get_pose().p), np.asarray(task.block2.get_pose().p)
    opened = task.is_left_gripper_open() and task.is_right_gripper_open()
    def stacked(top, bottom):
        return bool(opened and np.all(np.abs(top-(bottom+[0, 0, .05])) < [.025, .025, .012]))
    return {"green_on_red": stacked(green, red), "red_on_green": stacked(red, green)}


def initial_state(task, family, obs):
    names = ("bottle1", "bottle2") if family == "pick_dual_bottles" else ("block1", "block2")
    state = {"joints": np.asarray(obs["joint_action"]["vector"]).round(6).tolist(), "objects": {}, "images": {}}
    for name in names:
        pose = getattr(task, name).get_pose()
        state["objects"][name] = {"p": np.asarray(pose.p).round(6).tolist(), "q": np.asarray(pose.q).round(6).tolist()}
    for name in ("head_camera", "left_camera", "right_camera"):
        state["images"][name] = hashlib.sha256(obs["observation"][name]["rgb"].tobytes()).hexdigest()
    return state


def expert(task, family, goal):
    if family == "stack_blocks_two":
        task.last_gripper, task.last_actor = None, None
        order = (task.block1, task.block2) if goal == "green_on_red" else (task.block2, task.block1)
        for block in order:
            task.pick_and_place_block(block)
    elif goal == "both":
        task.play_once()
    else:
        from envs.utils import ArmTag
        arm = ArmTag(goal)
        actor = task.bottle1 if goal == "left" else task.bottle2
        target = task.left_target_pose if goal == "left" else task.right_target_pose
        task.move(task.grasp_actor(actor, arm_tag=arm, pre_grasp_dis=.08))
        task.move(task.move_by_displacement(arm_tag=arm, z=.1))
        task.move(task.place_actor(actor, target_pose=target, arm_tag=arm, functional_point_id=0,
                                   pre_dis=0., dis=0., is_open=False))


def rollout(case, instruction, model=None):
    from collect_recap_rollouts import env_observation, make_task
    import torch

    task = None
    row = {**case, "instruction": instruction, "status": "setup_error", "started_unix": time.time()}
    try:
        family, goal = case["family"], case["goal"]
        task = make_task(family, case["seed"], instruction)
        canonical = task.check_success
        canonical_goal = "both" if family == "pick_dual_bottles" else "green_on_red"
        obs = task.get_obs()
        row.update(initial_state=initial_state(task, family, obs), initial_predicates=goal_predicates(task, family),
                   initial_canonical_equivalent=bool(canonical()) == goal_predicates(task, family)[canonical_goal],
                   episode_limit=int(task.step_lim))
        # Only this instance is overridden; source tasks and reward trees remain untouched.
        task.check_success = lambda: goal_predicates(task, family)[goal]
        row["status"] = "running"
        if model is None:
            expert(task, family, goal)
            steps = None
            row["planner_success"] = bool(task.plan_success)
        else:
            torch.manual_seed(case["noise_seed"])
            steps = 0
            while steps < task.step_lim and not task.eval_success:
                with torch.inference_mode():
                    actions, _ = model.predict_action_batch(env_observation(obs, instruction), mode="eval")
                if torch.is_tensor(actions):
                    actions = actions.detach().float().cpu().numpy()
                actions = np.asarray(actions)[0].astype(np.float32)
                if actions.shape != (32, 14) or not np.isfinite(actions).all():
                    raise ValueError("Invalid action chunk")
                if steps == 0:
                    row["first_chunk"] = actions.tolist()
                for action in actions:
                    task.take_action(action, action_type="qpos")
                    steps += 1
                    if task.eval_success or steps >= task.step_lim:
                        break
                obs = task.get_obs()
        predicates = goal_predicates(task, family)
        row.update(status="complete", success=bool(predicates[goal]), final_predicates=predicates, steps=steps,
                   final_canonical_equivalent=bool(canonical()) == predicates[canonical_goal])
    except Exception as error:
        row.update(status="setup_error" if row["status"] == "setup_error" else "execution_error", error=repr(error))
    finally:
        if task is not None:
            task.close_env()
        row["elapsed_seconds"] = time.time()-row["started_unix"]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("expert", "policy"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--policy", choices=("base", "w1", "w2"))
    parser.add_argument("--gate", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    path = ROOT/"configs/multitask/instruction_response.json"
    config = json.loads(path.read_text())
    protocol = json.loads((ROOT/config["parent_protocol"]).read_text())
    seed_path = ROOT/protocol["evaluation"]["seed_file"]
    if digest(seed_path) != protocol["evaluation"]["seed_file_sha256"]:
        raise ValueError("Scene seed source changed")
    seeds = json.loads(seed_path.read_text())
    first, last = config["seed_indices"]
    report = {"config": config, "config_sha256": digest(path), "rows": [], "complete": False,
              "full_simulator_state_saved": False, "evaluation_returns_to_training": False}
    write_json(output/"report.json", report)
    model, probe = None, None
    if args.phase == "policy":
        if args.policy is None or args.gate is None:
            parser.error("policy requires --policy and --gate")
        gate = json.loads(args.gate.resolve().read_text())
        if not gate["complete"] or gate["config_sha256"] != digest(path):
            raise ValueError("Expert gate incomplete or different protocol")
        from evaluate_multitask import load_model
        import torch
        if digest(ROOT/protocol["model"]["path"]/"model.safetensors") != protocol["model"]["sha256"]:
            raise ValueError("Base weights changed")
        model, _ = load_model("native", protocol)
        model.requires_grad_(False)
        if args.policy != "base":
            from vla_rl_mt.shared_value import SharedPrefixValue
            checkpoint_path = ROOT/f".runtime/runs/method/m2-joint-v1/{args.policy}-seed0/checkpoints/global_step_128/actor/addon.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            probe = SharedPrefixValue(model, **checkpoint["architecture"]).to("cuda").eval()
            probe.load_addon(checkpoint["addon"])
            report["checkpoint_sha256"] = digest(checkpoint_path)
        report["gate_sha256"] = digest(args.gate.resolve())
    for family_index, (family, goals) in enumerate(config["families"].items()):
        for seed_index in range(first, last):
            seed = seeds[family]["success_seeds"][seed_index]
            if args.phase == "policy" and not gate["eligible"][f"{family}/{seed}"]:
                continue
            for goal, instructions in goals.items():
                for paraphrase, instruction in enumerate(instructions[:1] if args.phase == "expert" else instructions):
                    case = {"family": family, "seed": seed, "seed_index": seed_index, "goal": goal,
                            "paraphrase": paraphrase, "policy": args.policy or "expert",
                            "noise_seed": config["noise_seed_base"]+family_index*10000+seed_index}
                    row = rollout(case, instruction, model)
                    report["rows"].append(row)
                    write_json(output/"report.json", report)
                    print(json.dumps({k: row.get(k) for k in ("policy", "family", "seed", "goal", "status", "success", "error")}), flush=True)
    if args.phase == "expert":
        report["eligible"] = {}
        for family in config["families"]:
            for seed_index in range(first, last):
                seed = seeds[family]["success_seeds"][seed_index]
                rows = [r for r in report["rows"] if r["family"] == family and r["seed"] == seed]
                good = len(rows) == len(config["families"][family]) and all(
                    r["status"] == "complete" and r["success"] and r["planner_success"]
                    and not any(r["initial_predicates"].values()) and r["initial_canonical_equivalent"]
                    and r["final_canonical_equivalent"] for r in rows)
                good = good and all(r.get("initial_state") == rows[0].get("initial_state") for r in rows)
                report["eligible"][f"{family}/{seed}"] = bool(good)
    report["complete"] = True
    write_json(output/"report.json", report)
    if probe is not None:
        probe.close()


if __name__ == "__main__":
    main()
