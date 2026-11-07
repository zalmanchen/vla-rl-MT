"""Bounded, resumable diagnostic of existing checkpoints; never starts training."""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from pirl_batch_controller import ROOT, RUN, checkpoint, evaluate, write

JOB_NAME = 'single-adjust_bottle-seed0'


def process_args(pid):
    path = Path(f'/proc/{pid}/cmdline')
    return path.read_bytes().decode().split('\0') if path.exists() else []


def alive(pid):
    path = Path(f'/proc/{pid}/stat')
    return path.exists() and path.read_text().split(') ', 1)[1][0] != 'Z'


def takeover():
    """Stop only the verified old scheduler and its resumable eval workers."""
    pid = int(json.loads((RUN / 'controller.json').read_text())['pid'])
    args = process_args(pid)
    if not any(Path(a).name == 'pirl_batch_controller.py' for a in args):
        raise RuntimeError('Expected batch controller is not running; inspect before takeover')
    if Path(f'/proc/{pid}/cwd').resolve() != ROOT:
        raise RuntimeError('Controller working directory mismatch')
    os.kill(pid, signal.SIGSTOP)
    terminated = False
    try:
        children = set()
        for p in Path(f'/proc/{pid}/task').glob('*/children'):
            children.update(map(int, p.read_text().split()))
        workers = []
        for child in children:
            child_args = process_args(child)
            if not child_args:
                continue
            if any(Path(a).name == 'eval_pirl.py' for a in child_args):
                if os.getpgid(child) != child:
                    raise RuntimeError('Unexpected eval process group')
                workers.append(child)
            elif not any(Path(a).name == 'render_pirl_progress.py' for a in child_args):
                raise RuntimeError(f'Unexpected child {child}; no training will be interrupted')
        archive = RUN / 'pilot-handoff' / str(int(time.time()))
        archive.mkdir(parents=True)
        for name in ['plan.json', 'status.json', 'controller.json']:
            (archive / name).write_bytes((RUN / name).read_bytes())
        write(RUN / 'pilot-plan.json', {
            'active': True, 'formal_queue_enabled': False,
            'created_unix': time.time(), 'original_plan': str(archive / 'plan.json'),
            'reason': 'User requested fast preliminary validation; first completed run regressed',
            'diagnostic': {'job': JOB_NAME, 'steps': [10, 20], 'episodes_per_step': 20,
                           'parallel_evaluators': 2, 'new_training_jobs': 0},
            'next_small_matrix': {
                'status': 'planned; not automatically launched by diagnostic controller',
                'tasks': ['adjust_bottle', 'stack_blocks_two', 'place_container_plate'],
                'training_seeds': [0], 'iterations': 20, 'envs_per_task': 2,
                'nominal_actions_per_env_per_iteration': 512,
                'single_policies': 3, 'shared_policies': 1,
                'additional_ppo_jobs_if_existing_ST20_is_reusable': 3,
                'aggregate_single_and_shared_nominal_budget_each': 61440,
                'evaluate_episodes_per_task': 20,
                'reference_paired_successes': 30, 'reference_paired_episodes': 60,
                'optional_matched_shared_success_replay_jobs': 1,
                'gate': 'Diagnose regression before expansion; keep same training contract for ST/MT',
            },
            'deferred': ['remaining 17 full PPO jobs', '18 full replay jobs',
                         '3 training seeds', '100-scene final evaluation', '22-task retention',
                         'task-count scaling and long-horizon applications'],
            'evaluation_use': 'Diagnostic only; report all planned checkpoints. Final claims need an independent test set after pilot decisions.',
        })
        write(archive / 'handoff.json', {'old_controller_pid': pid, 'eval_pids': workers,
              'action': 'SIGTERM scheduler and resumable eval workers; preserve completed episode files and every checkpoint',
              'time_unix': time.time()})
        os.kill(pid, signal.SIGTERM)
        os.kill(pid, signal.SIGCONT)
        terminated = True
        for child in workers:
            if alive(child):
                os.killpg(child, signal.SIGTERM)
        deadline = time.time() + 15
        while any(alive(p) for p in [pid, *workers]):
            if time.time() > deadline:
                raise RuntimeError('Old worker did not exit; refusing to start competing evaluators')
            time.sleep(0.2)
    finally:
        if not terminated and alive(pid):
            os.kill(pid, signal.SIGCONT)


def status(phase, detail):
    write(RUN / 'status.json', {'phase': phase, 'detail': detail,
          'updated_unix': time.time(), 'pid': os.getpid(), 'mode': 'quick_pilot'})
    subprocess.run([sys.executable, str(ROOT / 'experiments/render_pirl_progress.py')], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--take-over', action='store_true')
    args = parser.parse_args()
    plan = json.loads((RUN / 'plan.json').read_text())
    job = next(j for j in plan['jobs'] if j['name'] == JOB_NAME)
    for step in [10, 20]:
        if not (checkpoint(job, step) / 'model_state_dict/full_weights.pt').exists():
            raise RuntimeError(f'Missing checkpoint {step}')
    pilot_lock = (RUN / 'pilot-controller.lock').open('a')
    fcntl.flock(pilot_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.take_over:
        takeover()
    if not (RUN / 'pilot-plan.json').exists():
        raise RuntimeError('Pilot plan missing; an explicit handoff is required')
    lock = (RUN / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu = (ROOT / '.runtime/experiments/gpu-evaluation.lock').open('a')
    status('快速前置诊断：等待 GPU', '仅评估已有第10、20轮权重；完整训练矩阵暂不启动。')
    fcntl.flock(gpu, fcntl.LOCK_EX)
    write(RUN / 'pilot-controller.json', {'pid': os.getpid(), 'started_unix': time.time()})
    status('快速前置诊断：两个检查点并行评估', 'adjust_bottle，第10、20轮各20个相同场景；原始和第100轮结果直接复用。')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(evaluate, job, job['tasks'], RUN / 'jobs' / JOB_NAME / f'curve-{step}',
                               checkpoint(job, step), episodes=20) for step in [10, 20]]
        for future in futures:
            future.result()
    status('快速前置诊断完成', '已补齐第10、20轮，各20场景。先分析退化再决定短程ST／MT实验；不会自动扩跑完整矩阵。')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if (RUN / 'pilot-controller.json').exists() and json.loads((RUN / 'pilot-controller.json').read_text()).get('pid') == os.getpid():
            status('快速前置诊断需检查', str(exc))
        raise
