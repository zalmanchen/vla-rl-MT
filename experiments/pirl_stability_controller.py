"""Small E6 queue: zero-update SDE, critic-only, then one-epoch PPO."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / '.runtime/experiments/e6-posttraining-diagnostic'
E5 = ROOT / '.runtime/experiments/e5-pirl'


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(data, ensure_ascii=False, indent=2)); temp.replace(path)


def render():
    subprocess.run([sys.executable, str(ROOT / 'experiments/render_pirl_stability.py')], cwd=ROOT, check=True)


def status(phase, detail):
    write(RUN / 'status.json', {'phase': phase, 'detail': detail, 'pid': os.getpid(), 'updated_unix': time.time()})
    render()


def stop_process_group(process):
    """Release a child session even when native code ignores Python alarms."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def evaluation_stall(logpath, started, now):
    heartbeat_path = logpath.parent / 'clean/adjust_bottle/heartbeat.json'
    heartbeat = json.loads(heartbeat_path.read_text()) if heartbeat_path.exists() else {}
    # A resumed worker first loads its model. Ignore the previous worker's timestamp.
    last_progress = max(started, heartbeat.get('unix_time', started))
    if now - last_progress <= 900:
        return None
    return {'reason': 'evaluation heartbeat stale for more than900seconds',
            'heartbeat': heartbeat, 'idle_seconds': now - last_progress,
            'time_unix': now, 'policy_failure': False,
            'recovery': 'Preserve completed episodes; retry the same frozen seeds.'}


def launch(script, args, logpath, reference=False):
    prefix = '$PROJECT_ROOT/.runtime/reference-transformers:' if reference else ''
    shell = 'source bootstrap/env.sh; export PYTHONPATH="' + prefix + '$PROJECT_ROOT/experiments:$PYTHONPATH"; export RLINF_EXT_MODULE=pirl_diagnostic_extension; export OMP_NUM_THREADS=4; export HYDRA_FULL_ERROR=1; exec python -u "$@"'
    logpath.parent.mkdir(parents=True, exist_ok=True)
    with logpath.open('a') as log:
        process = subprocess.Popen(['bash', '-c', shell, '--', 'experiments/' + script, *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        started = time.time()
        write(logpath.with_suffix('.process.json'), {'pid': process.pid, 'started_unix': started, 'script': script, 'args': args})
        deadline = started + 3 * 3600
        while process.poll() is None:
            if script == 'eval_pirl.py':
                stalled = evaluation_stall(logpath, started, time.time())
                if stalled:
                    stalled['pid'] = process.pid
                    write(logpath.parent / f'watchdog-{time.time_ns()}.json', stalled)
                    stop_process_group(process)
                    raise TimeoutError(f'Evaluation heartbeat timeout: {logpath}')
            if time.time() > deadline:
                stop_process_group(process)
                raise TimeoutError(str(logpath))
            render(); time.sleep(15)
        if process.returncode:
            raise RuntimeError(f'{script} exited {process.returncode}: {logpath}')


def evaluate(config, checkpoint, dest, mode='eval'):
    records = list((dest / 'clean/adjust_bottle').glob('episode_*.json'))
    if len(records) < 20:
        for attempt in range(2):
            try:
                launch('eval_pirl.py', ['--task', 'adjust_bottle', '--regime', 'clean', '--episodes', '20',
                       '--run', str(dest), '--seed-protocol', 'fixed_eval', '--backend', 'reference',
                       '--config', str(config), '--checkpoint', str(checkpoint), '--policy-mode', mode],
                       dest / 'evaluation.log', reference=True)
                break
            except TimeoutError:
                # One retry per controller invocation; all timeout receipts remain.
                if attempt:
                    raise
                status('评估超时后续跑', f'{dest.name}：保留已完成场景，按原种子重试一次。')
    records = [json.loads(p.read_text()) for p in (dest / 'clean/adjust_bottle').glob('episode_*.json')]
    assert len(records) == 20 and all(x['status'] == 'completed' for x in records)
    base = ROOT / '.runtime/experiments/e1-reference-clean/clean/adjust_bottle'
    before = [json.loads((base / f"episode_{x['seed']}.json").read_text()) for x in records]
    result = {'episodes': 20, 'successes': sum(x['success'] for x in records),
              'base_successes': sum(x['success'] for x in before), 'policy_mode': mode,
              'new_wins': sum(x['success'] and not b['success'] for x, b in zip(records, before)),
              'new_losses': sum(not x['success'] and b['success'] for x, b in zip(records, before))}
    write(dest / 'results.json', result)


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    lock = (RUN / 'controller.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu = (ROOT / '.runtime/experiments/gpu-evaluation.lock').open('a')
    status('等待GPU', 'E6仅包含两个短训练作业，正式矩阵保持关闭。')
    fcntl.flock(gpu, fcntl.LOCK_EX)
    write(RUN / 'controller.json', {'pid': os.getpid(), 'started_unix': time.time()})
    plan = json.loads((RUN / 'plan.json').read_text())
    config = E5 / 'configs/single-adjust_bottle-seed0.yaml'
    status('D0：无更新SDE闭环评估', '同权重、同20场景；复用已有参考ODE与原生ODE结果。')
    evaluate(config, 'base', RUN / 'zero-sde', mode='train')
    for job in plan['jobs']:
        folder = RUN / 'jobs' / job['name']
        if not (folder / 'TRAINING_FINISHED').exists():
            status('D1：只训练critic' if job['kind'] == 'critic_only' else 'D2：每批只更新一次PPO', job['name'])
            checkpoint_root = folder / job['name'] / 'checkpoints'
            existing = sorted((p for p in checkpoint_root.glob('global_step_*')
                               if (p / 'actor/model_state_dict/full_weights.pt').exists()
                               and (not job.get('checkpoint_receipts_required', False)
                                    or (p / 'actor/diagnostic-state.json').exists())),
                              key=lambda p: int(p.name.split('_')[-1]))
            args = ['--config-path', str(RUN / 'configs'), '--config-name', job['name']]
            if existing:
                args.append('runner.resume_dir=' + str(existing[-1]))
            launch('run_pirl_diagnostic_job.py', args, folder / 'training.log')
            checkpoint = checkpoint_root / f"global_step_{job['iterations']}/actor/model_state_dict/full_weights.pt"
            if not checkpoint.exists():
                raise RuntimeError('Diagnostic checkpoint missing')
            audit = [json.loads(line) for line in (folder / 'batch-audit.jsonl').read_text().splitlines()]
            if not audit or any(x['logprob_gap_max_abs'] > .01 for x in audit):
                raise RuntimeError('Missing or failed batch audit')
            if job['kind'] == 'critic_only' and not all(x['only_value_head_trainable'] for x in audit):
                raise RuntimeError('Critic-only actor freeze check failed')
            (folder / 'TRAINING_FINISHED').write_text(str(time.time()))
        if job['kind'] != 'critic_only':
            for step in [5, 10]:
                status('D2：短程PPO配对评估', f"{job['name']}，第{step}轮，20场景")
                checkpoint = folder / job['name'] / f'checkpoints/global_step_{step}/actor'
                evaluate(job['config'], checkpoint, folder / f'evaluation-{step}')
    status('E6短诊断完成', '无更新SDE、critic-only、更新一次的PPO已完成；下一步根据证据选择稳定配置。')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('E6需检查', str(exc)); raise
