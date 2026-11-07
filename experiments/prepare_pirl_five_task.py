"""Freeze the bounded E7 single-task / shared-policy learning-curve experiment."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / '.runtime/experiments/e7-five-task-curves'
TASKS = ['adjust_bottle', 'click_bell', 'stack_blocks_two', 'place_empty_cup', 'place_container_plate']
STEPS = [5, 10, 20, 50, 100]


def main():
    if (RUN / 'plan.json').exists():
        raise SystemExit('E7 plan exists; do not overwrite an active experiment.')
    base_path = ROOT / '.runtime/experiments/e6-posttraining-diagnostic/configs/ppo-update1-adjust_bottle-seed0.yaml'
    base = yaml.safe_load(base_path.read_text())
    manifest = json.loads((ROOT / '.runtime/experiments/e1-reference-clean/seed_manifest.json').read_text())
    train_seeds = json.loads((ROOT / 'third_party/RLinf/rlinf/envs/sim/robotwin/seeds/train_seeds.json').read_text())
    limits, reference = {}, {}
    for task in TASKS:
        seeds = manifest['tasks'][task]['valid_seeds'][:20]
        assert not set(train_seeds[task]['success_seeds']) & set(manifest['tasks'][task]['valid_seeds'])
        rows = [json.loads((ROOT / f'.runtime/experiments/e1-reference-clean/clean/{task}/episode_{s}.json').read_text()) for s in seeds]
        assert len(rows) == 20 and all(x['status'] == 'completed' for x in rows)
        limits[task] = rows[0]['limit']
        reference[task] = {'seeds': seeds, 'successes': sum(x['success'] for x in rows), 'episodes': 20}
    for audit in ['multitask-env-smoke.json', 'weight-roundtrip-audit.json', 'reward-mask-audit.json', 'terminal-integration-audit.json', 'seed-separation-audit.json']:
        assert json.loads((ROOT / '.runtime/experiments/e5-pirl' / audit).read_text())['status'] == 'pass', audit
    (RUN / 'configs').mkdir(parents=True)
    jobs = []
    # Shared curve is produced first; all six policies start independently from Motus.
    groups = [('multi-mt5-seed0', TASKS)] + [(f'single-{t}-seed0', [t]) for t in TASKS]
    for name, tasks in groups:
        cfg = copy.deepcopy(base)
        folder = RUN / 'jobs' / name
        cfg['runner'].update(max_epochs=100, max_steps=100, save_interval=-1, resume_dir=None)
        cfg['runner']['logger'].update(log_path=str(folder), experiment_name=name)
        cfg['actor']['pilot_checkpoint_steps'] = [1, *STEPS]
        cfg['actor']['pilot_save_first_update'] = False
        cfg['actor']['seed'] = 0
        for split in ['train', 'eval']:
            env = cfg['env'][split]
            env.update(total_num_envs=2 * len(tasks), seed=0, max_episode_steps=max(limits[t] for t in tasks))
            tc = env['task_config']
            tc.update(task_name=tasks[0], step_lim=limits[tasks[0]], pilot_tasks=tasks,
                      pilot_task_limits={t: limits[t] for t in tasks}, save_path=str(folder / 'simulator'))
            tc.pop('pilot_record_dir', None)
            if split == 'train':
                tc['pilot_record_dir'] = str(folder / 'online-chunks')
        path = RUN / 'configs' / f'{name}.yaml'
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        jobs.append({'name': name, 'tasks': tasks, 'kind': 'multi' if len(tasks) > 1 else 'single',
                     'seed': 0, 'iterations': 100, 'config': str(path),
                     'nominal_actions_per_iteration': 1024 * len(tasks),
                     'expected_optimizer_steps': 100 * len(tasks)})
    source = RUN / 'source'; source.mkdir()
    names = ['run_pirl_diagnostic_job.py', 'pirl_diagnostic_extension.py', 'pirl_diagnostic_actor.py',
             'pirl_extension.py', 'pirl_rewards.py', 'compat.py', 'eval_pirl.py', 'pirl_policy.py', 'reference_policy.py']
    hashes = {}
    for name in names:
        original = ROOT / 'experiments' / name
        shutil.copy2(original, source / name)
        hashes[name] = hashlib.sha256(original.read_bytes()).hexdigest()
    plan = {'created_unix': time.time(), 'purpose': 'Five specialists versus one shared policy, bounded preliminary learning curves',
            'algorithm': 'Flow-SDE + PPO; strict FP32, one epoch, frozen VLM, no critic warmup',
            'starting_checkpoint': str(ROOT / 'models/pi0.5_robotwin2'), 'training_seeds': [0], 'tasks': TASKS,
            'jobs': jobs, 'evaluation_steps': STEPS, 'evaluation_episodes_per_task': 20,
            'reference': reference, 'source_sha256': hashes, 'base_config': str(base_path),
            'budget': {'per_task_per_iteration_nominal_actions': 1024,
                       'per_task_final_nominal_actions': 102400,
                       'five_specialists_aggregate_nominal_actions': 512000,
                       'shared_policy_total_nominal_actions': 512000,
                       'total_primary_nominal_actions': 1024000},
            'evaluation': 'Frozen E1 first20 scenes per task; original reference ODE,32 actions,10 denoising steps. Report every scheduled checkpoint.',
            'resource_policy': {'training_processes': 1, 'evaluation_only_processes': 2,
                                'overlap_eval_processes': 1, 'eval_reserve_gib': 28,
                                'minimum_headroom_gib': 12, 'overlap_after_training_iterations': 2},
            'limits': ['Single training seed and reused development scenes; not final paper evidence.',
                       'Critic instability is a hypothesis, not an established cause.',
                       'Longer training may improve or regress; do not select only the best checkpoint.',
                       'Per-task nominal interactions are matched. Actual interactions and update counts are reported separately.',
                       'Shared policy has five optimizer steps per collection iteration; each specialist has one. Aggregate steps are matched.',
                       'No success-replay, extra seeds,22-task retention or task-count scaling is auto-launched.'],
            'formal_queue_e5_remains_disabled': True}
    (RUN / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    print(json.dumps({'run': str(RUN), 'jobs': len(jobs), 'budget': plan['budget']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
