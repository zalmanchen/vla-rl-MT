"""E7: one training job, memory-gated concurrent evaluation, frozen paired curves."""
import fcntl
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

from pirl_stability_controller import stop_process_group
from pirl_runtime_guard import (memory_sample, evaluation_allowed, should_defer, FatalLogScanner,
                                lineage_records, process_tree, identity_alive)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / '.runtime/experiments/e7-five-task-curves'


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def lines(path):
    if not path.exists():
        return []
    return [json.loads(s) for s in path.read_text().splitlines(keepends=True) if s.endswith('\n')]


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2)); tmp.replace(path)


def event(kind, **data):
    with (RUN / 'events.jsonl').open('a') as f:
        f.write(json.dumps({'time_unix': time.time(), 'kind': kind, **data}, ensure_ascii=False) + '\n')


def gpu_sample():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                             '--format=csv,noheader,nounits', '-i', '0'], capture_output=True, text=True, check=True, timeout=10)
    used, total, utilization = map(float, result.stdout.strip().split(','))
    return {'time_unix': time.time(), 'used_gib': used / 1024, 'total_gib': total / 1024,
            'utilization_percent': utilization}


def checkpoint(job, step):
    return RUN / 'jobs' / job['name'] / job['name'] / f'checkpoints/global_step_{step}/actor'


def checkpoint_ready(job, step):
    p = checkpoint(job, step)
    receipt = read(p / 'diagnostic-state.json', {})
    return ((p / 'model_state_dict/full_weights.pt').exists()
            and (p / 'dcp_checkpoint/.metadata').is_file()
            and any(x.stat().st_size > 0 for x in (p / 'dcp_checkpoint').glob('*.distcp'))
            and receipt.get('global_step') == step
            and receipt.get('optimizer_steps') == step * len(job['tasks']))


def records(spec, plan):
    folder = spec['dest'] / 'clean' / spec['task']
    result = [read(folder / f'episode_{seed}.json', {}) for seed in plan['reference'][spec['task']]['seeds']]
    return [x for x in result if x.get('status') == 'completed']


def save_evaluation(spec, plan):
    rows = records(spec, plan)
    if len(rows) != 20:
        raise RuntimeError(f'Incomplete evaluation: {spec["id"]}')
    base = ROOT / '.runtime/experiments/e1-reference-clean/clean' / spec['task']
    before = [read(base / f'episode_{x["seed"]}.json') for x in rows]
    write(spec['dest'] / f'results-{spec["task"]}.json', {
        'task': spec['task'], 'step': spec['step'], 'episodes': 20,
        'successes': sum(x['success'] for x in rows), 'base_successes': sum(x['success'] for x in before),
        'new_wins': sum(x['success'] and not b['success'] for x, b in zip(rows, before)),
        'new_losses': sum(not x['success'] and b['success'] for x, b in zip(rows, before))})


def eval_specs(plan):
    for step in plan['evaluation_steps']:
        for job in plan['jobs']:
            if not checkpoint_ready(job, step):
                continue
            for task in job['tasks']:
                yield {'id': f'{job["name"]}/{step}/{task}', 'job': job, 'step': step, 'task': task,
                       'dest': RUN / 'jobs' / job['name'] / f'curve-{step}'}


def spawn(script, args, logpath, reference=False):
    source = RUN / 'source'
    prefix = '$PROJECT_ROOT/.runtime/reference-transformers:' if reference else ''
    shell = ('source bootstrap/env.sh; source_dir="$1"; shift; '
             'export PYTHONPATH="' + prefix + '$source_dir:$PROJECT_ROOT/experiments:$PYTHONPATH"; '
             'export RLINF_EXT_MODULE=pirl_diagnostic_extension; export OMP_NUM_THREADS=4; '
             'export HYDRA_FULL_ERROR=1; exec python -u "$@"')
    logpath.parent.mkdir(parents=True, exist_ok=True)
    policy = read(RUN / 'runtime-policy.json')
    env = os.environ.copy()
    env['RAY_OVERRIDE_RESOURCES'] = json.dumps({'CPU': policy['ray_cpus'], 'object_store_memory': policy['ray_object_store_gib'] * 1024**3})
    env['RAY_memory_usage_threshold'] = '0.95'
    env['RAY_USAGE_STATS_ENABLED'] = '0'
    with logpath.open('a') as log:
        process = subprocess.Popen(['bash', '-c', shell, '--', str(source), str(source / script), *args],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    state = {'pid': process.pid, 'started_unix': time.time(), 'script': script, 'args': args}
    write(logpath.with_suffix('.process.json'), state)
    return {'process': process, 'started': state['started_unix'], 'logpath': logpath}




class Queue:
    def __init__(self, plan):
        self.plan = plan
        self.policy = {**plan['resource_policy'], **read(RUN / 'runtime-policy.json')}
        self.owned = {}
        self.training_gpu_peak = self.training_ram_peak = 0
        self.training_iterations = 0
        self.evaluators = {}
        self.failed = {}
        self.errors = {}
        self.deferred = set()
        self.rendered_at = 0
        self.trainer = None

    def refresh(self, phase, job=None, peak=0):
        sample = gpu_sample()
        ram = memory_sample()
        sample['ram'] = ram
        sample.update(phase=phase, job=job['name'] if job else None,
                      evaluators=len(self.evaluators), sampled_training_only_peak_gib=peak)
        with (RUN / 'gpu-utilization.jsonl').open('a') as f:
            f.write(json.dumps(sample) + '\n')
        write(RUN / 'status.json', {'phase': phase, 'job': job['name'] if job else None,
              'updated_unix': time.time(), 'pid': os.getpid(), 'gpu': sample,
              'evaluators': list(self.evaluators), 'failed_evaluations': self.failed, 'ram': ram,
              'runtime_policy': '.runtime/experiments/e7-five-task-curves/runtime-policy.json'})
        if time.time() - self.rendered_at > 60:
            for name in ['render_pirl_five_task.py', 'render_progress.py']:
                try:
                    with (RUN / 'render.log').open('a') as log:
                        result = subprocess.run([str(ROOT / '.runtime/venv-rlinf/bin/python'), str(ROOT / 'experiments' / name)], cwd=ROOT,
                                                stdout=log, stderr=subprocess.STDOUT, timeout=45)
                    if result.returncode: event('report_error', script=name, returncode=result.returncode)
                except subprocess.TimeoutExpired:
                    event('report_timeout', script=name)
            self.rendered_at = time.time()
        return sample

    def reap_evaluators(self, training=False):
        for ident, active in list(self.evaluators.items()):
            process = active['process']; spec = active['spec']
            heartbeat = read(spec['dest'] / 'clean' / spec['task'] / 'heartbeat.json', {})
            stalled = time.time() - max(active['started'], heartbeat.get('unix_time', active['started'])) > 900
            if stalled and process.poll() is None:
                write(spec['dest'] / 'logs' / f'{spec["task"]}-timeout-{time.time_ns()}.json',
                      {'pid': process.pid, 'heartbeat': heartbeat, 'policy_failure': False, 'time_unix': time.time()})
                stop_process_group(process)
            code = process.poll()
            if code is None:
                continue
            del self.evaluators[ident]
            if code == 0 and len(records(spec, self.plan)) == 20:
                save_evaluation(spec, self.plan); event('evaluation_complete', id=ident)
                continue
            self.errors[ident] = self.errors.get(ident, 0) + 1
            tail = active['logpath'].read_text(errors='replace')[-5000:]
            retryable = stalled or 'out of memory' in tail.lower()
            event('evaluation_error', id=ident, returncode=code, stalled=stalled, retryable=retryable)
            if retryable and self.errors[ident] < 2:
                if training:
                    self.deferred.add(ident)
            else:
                self.failed[ident] = f'exit={code}; stalled={stalled}; see {active["logpath"]}'

    def start_evaluators(self, count, training=False):
        for spec in eval_specs(self.plan):
            if len(self.evaluators) >= count:
                break
            ident = spec['id']
            if ident in self.evaluators or ident in self.failed or (training and ident in self.deferred):
                continue
            if len(records(spec, self.plan)) == 20:
                continue
            gpu, ram = gpu_sample(), memory_sample()
            pending = sum(time.time() - x['started'] < 180 for x in self.evaluators.values())
            if not evaluation_allowed(gpu, ram, self.policy, pending=pending, training=training,
                                      gpu_peak=self.training_gpu_peak, ram_peak=self.training_ram_peak,
                                      iterations=self.training_iterations):
                return
            active = spawn('eval_pirl.py', ['--task', spec['task'], '--regime', 'clean', '--episodes', '20',
                           '--run', str(spec['dest']), '--seed-protocol', 'fixed_eval', '--backend', 'reference',
                           '--config', spec['job']['config'], '--checkpoint', str(checkpoint(spec['job'], spec['step']))],
                           spec['dest'] / 'logs' / f'{spec["task"]}.log', reference=True)
            active['spec'] = spec; self.evaluators[ident] = active
            event('evaluation_started', id=ident, pid=active['process'].pid, overlap=training)

    def drain_active(self):
        while self.evaluators:
            self.reap_evaluators()
            self.refresh('评估衔接')
            time.sleep(5)

    def train(self, job):
        folder = RUN / 'jobs' / job['name']
        if (folder / 'TRAINING_FINISHED').exists() and checkpoint_ready(job, 100):
            return
        existing = [step for step in range(101) if checkpoint_ready(job, step)]
        start = max(existing, default=0)
        args = ['--config-path', str(RUN / 'configs'), '--config-name', job['name']]
        if existing:
            args.append('runner.resume_dir=' + str(checkpoint(job, start).parent))
            event('training_resume', job=job['name'], step=start, note='Environment/RNG continuity is not claimed.')
        previous_metrics_bytes = (folder / 'metrics.log').stat().st_size if (folder / 'metrics.log').exists() else 0
        previous_phases = lineage_records(lines(folder / 'memory-phases.jsonl'), read(folder / 'resume-attempts.json', []))
        updated = [x for x in previous_phases if x['phase'] == 'after_training']
        previous_iteration = max((x['global_step'] + 1 for x in updated), default=0)
        retained_until = max((x['time_unix'] for x in updated if x['global_step'] + 1 <= start), default=0)
        if start and not retained_until:
            raise RuntimeError('Cannot account for resumed checkpoint interactions: no matching phase record')
        if memory_sample()['available_gib'] < self.policy['minimum_training_ram_gib']:
            raise RuntimeError('Insufficient host memory to safely initialize training; no training launched')
        logpath = folder / f'training-{time.time_ns()}.log'
        active = spawn('run_pirl_diagnostic_job.py', args, logpath)
        self.trainer = active['process']
        write(folder / 'active-training.json', {'pid': self.trainer.pid, 'started_unix': active['started'],
                                               'start_step': start, 'log': str(logpath)})
        if previous_metrics_bytes:
            receipts = read(folder / 'resume-attempts.json', [])
            receipts.append({'started_unix': active['started'], 'resume_step': start,
                             'retained_until_unix': retained_until, 'metrics_start_bytes': previous_metrics_bytes,
                             'previous_observed_iteration': previous_iteration,
                             'checkpoint': str(checkpoint(job, start)) if start else 'base',
                             'archive': self.policy['recovery_archive'],
                             'environment_rng_continuity': False})
            write(folder / 'resume-attempts.json', receipts)
        self.training_gpu_peak = self.training_ram_peak = 0
        self.training_iterations = 0
        scanner = FatalLogScanner()
        while self.trainer.poll() is None:
            self.owned.update(process_tree(self.trainer.pid))
            fatal = scanner.scan(logpath)
            if fatal:
                event('training_fatal_error', job=job['name'], evidence=fatal)
                raise RuntimeError(f'Training worker failed: {fatal}; log={logpath}')
            self.reap_evaluators(training=True)
            sample = self.refresh('在线训练', job, self.training_gpu_peak)
            ram = sample['ram']
            if not self.evaluators:
                self.training_gpu_peak = max(self.training_gpu_peak, sample['used_gib'])
                self.training_ram_peak = max(self.training_ram_peak, ram['used_gib'])
            phases = [a for a in lines(folder / 'memory-phases.jsonl')
                      if a['time_unix'] >= active['started'] and a['phase'] == 'after_training']
            self.training_iterations = len(phases)
            if self.evaluators and should_defer(sample, ram, self.policy):
                for ident, evaluation in list(self.evaluators.items()):
                    stop_process_group(evaluation['process']); self.deferred.add(ident)
                    del self.evaluators[ident]
                    event('evaluation_deferred_for_training_memory', id=ident, gpu=sample, ram=ram)
            elif not self.evaluators and ram['pressure_fraction'] >= self.policy['training_hard_pressure_fraction']:
                event('training_memory_guard', job=job['name'], ram=ram)
                raise RuntimeError(f'Training host memory exceeded safe pressure: {ram}')
            else:
                self.start_evaluators(self.policy['overlap_eval_processes'], training=True)
            activity = [logpath, folder / 'batch-audit.jsonl', folder / 'memory-phases.jsonl',
                        folder / 'online-chunks/interactions.jsonl']
            last = max([active['started']] + [p.stat().st_mtime for p in activity if p.exists()])
            if time.time() - last > 1800 or time.time() - active['started'] > 48 * 3600:
                event('training_watchdog', job=job['name'], last_activity=last)
                raise TimeoutError(f'Training stopped making progress: {job["name"]}')
            time.sleep(5)
        code = self.trainer.returncode; self.trainer = None
        if code:
            raise RuntimeError(f'Training exited {code}: {logpath}')
        if not checkpoint_ready(job, 100):
            raise RuntimeError('Missing complete final checkpoint or optimizer-step receipt')
        audits = [a for a in lines(folder / 'batch-audit.jsonl') if a['time_unix'] >= active['started']]
        if ([a['global_step'] for a in audits] != list(range(start, 100))
                or any(a['optimizer_steps'] != a['global_step'] * len(job['tasks']) for a in audits)
                or any(a['logprob_gap_max_abs'] > .01 for a in audits)):
            raise RuntimeError('Current attempt did not pass all scheduled batch audits')
        write(folder / 'TRAINING_FINISHED', {'time_unix': time.time(), 'iterations': 100,
                                            'optimizer_steps': job['expected_optimizer_steps']})
        event('training_complete', job=job['name'])
        self.cleanup_owned()
        self.drain_active()

    def cleanup_owned(self):
        # Do not kill global Ray processes: only the driver's recorded descendants.
        for sig in [signal.SIGTERM, signal.SIGKILL]:
            live = [pid for pid, identity in self.owned.items() if identity_alive(pid, identity)]
            if not live: break
            for pid in live:
                try: os.kill(pid, sig)
                except ProcessLookupError: pass
            deadline = time.time() + 5
            while time.time() < deadline and any(identity_alive(pid, self.owned[pid]) for pid in live):
                time.sleep(.2)
        self.owned = {pid: identity for pid, identity in self.owned.items() if identity_alive(pid, identity)}
        if self.owned: raise RuntimeError(f'Owned training processes did not exit: {list(self.owned)}')

    def cleanup(self):
        if self.trainer is not None:
            stop_process_group(self.trainer)
        for active in self.evaluators.values():
            stop_process_group(active['process'])
        self.cleanup_owned()


def main():
    plan = read(RUN / 'plan.json')
    for name, digest in plan['source_sha256'].items():
        assert hashlib.sha256((RUN / 'source' / name).read_bytes()).hexdigest() == digest, name
    lock = (RUN / 'controller.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write(RUN / 'controller.json', {'pid': os.getpid(), 'started_unix': time.time()})
    gpu_lock = (ROOT / '.runtime/experiments/gpu-evaluation.lock').open('a')
    queue = Queue(plan)
    queue.refresh('等待GPU')
    fcntl.flock(gpu_lock, fcntl.LOCK_EX)
    try:
        order = queue.policy.get('job_order', [x['name'] for x in plan['jobs']])
        assert len(order) == len(plan['jobs']) and set(order) == {x['name'] for x in plan['jobs']}
        for name in order:
            queue.train(next(x for x in plan['jobs'] if x['name'] == name))
        wait_started = time.time()
        while True:
            queue.reap_evaluators()
            queue.start_evaluators(queue.policy['evaluation_only_processes'])
            queue.refresh('配对曲线评估')
            remaining = [s for s in eval_specs(plan) if len(records(s, plan)) != 20 and s['id'] not in queue.failed]
            if not queue.evaluators and not remaining:
                break
            if queue.evaluators: wait_started = time.time()
            elif time.time() - wait_started > 1800:
                raise RuntimeError('Evaluation queue blocked by resource headroom for 30 minutes')
            time.sleep(5)
        queue.refresh('E7完成' if not queue.failed else '训练完成，部分评估需检查')
        if not queue.failed:
            write(RUN / 'FINISHED', {'time_unix': time.time(), 'jobs': 6})
    except BaseException as exc:
        queue.cleanup()
        write(RUN / 'failure.json', {'time_unix': time.time(), 'error': str(exc)})
        queue.rendered_at = 0
        queue.refresh('E7需检查')
        raise


if __name__ == '__main__':
    main()
