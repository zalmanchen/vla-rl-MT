"""CPU-only E7 curves; incomplete evaluations never become performance points."""
import argparse
import fcntl
import html
import json
import os
from pathlib import Path
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / '.runtime/experiments/e7-five-task-curves'
OUT = ROOT / '.research'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.runtime/cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def lines(path):
    if not path.exists():
        return []
    return [json.loads(s) for s in path.read_text().splitlines(keepends=True) if s.endswith('\n')]


def atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f'.{os.getpid()}.tmp'); temp.write_text(text); temp.replace(path)


def save_figure(fig, name):
    for fmt in ['png', 'svg']:
        target = OUT / 'figures' / f'{name}.{fmt}'
        target.parent.mkdir(exist_ok=True)
        temp = target.with_suffix(f'.{os.getpid()}.tmp')
        fig.savefig(temp, format=fmt, dpi=160, bbox_inches='tight'); temp.replace(target)
    plt.close(fig)


def wilson(k, n):
    p = k / n; z = 1.95996398454; d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (center - half), 100 * (center + half)


def collect(plan):
    jobs = []
    for job in plan['jobs']:
        folder = RUN / 'jobs' / job['name']
        audits = lines(folder / 'batch-audit.jsonl')
        metrics = (folder / 'metrics.log').read_text(errors='replace') if (folder / 'metrics.log').exists() else ''
        parts = re.split(r'Global Step:\s*(\d+)\s*/\s*\d+', metrics)
        curve = {}
        for i in range(1, len(parts), 2):
            values = {}
            for key in ['critic/explained_variance', 'critic/value_loss', 'actor/grad_norm', 'actor/clip_fraction']:
                found = re.search(re.escape(key) + r'=([-+\d.eE]+)', parts[i + 1])
                if found:
                    values[key] = float(found.group(1))
            curve[int(parts[i])] = values
        ready, pending = [], []
        for step in [1, *plan['evaluation_steps']]:
            checkpoint = folder / job['name'] / f'checkpoints/global_step_{step}/actor'
            receipt = read(checkpoint / 'diagnostic-state.json', {})
            if ((checkpoint / 'model_state_dict/full_weights.pt').exists()
                    and receipt.get('global_step') == step
                    and receipt.get('optimizer_steps') == step * len(job['tasks'])):
                ready.append(step)
            elif checkpoint.exists():
                pending.append(step)
        phases = lines(folder / 'memory-phases.jsonl')
        updated_steps = [p['global_step'] + 1 for p in phases if p['phase'] == 'after_training']
        interactions = {t: {'nominal_actions': 0, 'executed_actions': 0, 'chunks': 0} for t in job['tasks']}
        for record in lines(folder / 'online-chunks/interactions.jsonl'):
            values = interactions[record['task']]
            values['nominal_actions'] += record['requested_actions']
            values['executed_actions'] += record['executed_actions']
            values['chunks'] += 1
        evaluations = []
        for step in plan['evaluation_steps']:
            for task in job['tasks']:
                records = [read(folder / f'curve-{step}/clean/{task}/episode_{seed}.json', {})
                           for seed in plan['reference'][task]['seeds']]
                records = [x for x in records if x.get('status') == 'completed']
                evaluations.append({'step': step, 'task': task, 'completed': len(records),
                                    'successes': sum(x['success'] for x in records),
                                    'target': 20, 'nominal_actions_per_task': step * 1024})
        jobs.append({**job, 'completed_iterations': max([0, *curve.keys(), *ready, *updated_steps]),
                     'finished': (folder / 'TRAINING_FINISHED').exists(), 'ready_checkpoints': ready, 'pending_checkpoints': pending,
                     'audited_batches': len(audits),
                     'max_logprob_gap': max((a['logprob_gap_max_abs'] for a in audits), default=None),
                     'metrics': curve, 'interactions': interactions, 'evaluations': evaluations})
    return jobs


def plot(plan, jobs, gpu):
    tasks = plan['tasks']; colors = {'single': '#2462a3', 'multi': '#d16b26'}
    lookup = {(j['kind'], e['task'], e['step']): e for j in jobs for e in j['evaluations'] if e['completed'] == 20}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    for task, ax in zip(tasks, axes.flat):
        baseline = plan['reference'][task]['successes'] * 5
        ax.axhline(baseline, color='#777777', linestyle=':', label='Motus SFT')
        for kind, label in [('single', 'Single-task RL'), ('multi', 'Shared 5-task RL')]:
            points = [(step, lookup[kind, task, step]) for step in plan['evaluation_steps'] if (kind, task, step) in lookup]
            xs = [0] + [step * 1.024 for step, _ in points]
            ys = [baseline] + [row['successes'] * 5 for _, row in points]
            ax.plot(xs, ys, 'o-', color=colors[kind], label=label)
            if points:
                bounds = [wilson(plan['reference'][task]['successes'], 20)] + [wilson(row['successes'], 20) for _, row in points]
                ax.fill_between(xs, [b[0] for b in bounds], [b[1] for b in bounds], color=colors[kind], alpha=.10)
        ax.set_title(task); ax.grid(alpha=.2); ax.set_ylim(0, 100); ax.set_xlim(0, 103)
    ax = axes.flat[5]
    baseline = sum(plan['reference'][t]['successes'] for t in tasks)
    ax.axhline(baseline, color='#777777', linestyle=':')
    for kind, label in [('single', 'Mean of 5 specialists'), ('multi', 'One shared policy')]:
        steps = [s for s in plan['evaluation_steps'] if all((kind, t, s) in lookup for t in tasks)]
        values = [sum(lookup[kind, t, s]['successes'] for t in tasks) for s in steps]
        ax.plot([0] + [s * 1.024 for s in steps], [baseline] + values, 'o-', color=colors[kind], label=label)
    ax.set_title('Five-task macro success'); ax.grid(alpha=.2); ax.legend(fontsize=10)
    axes.flat[0].legend(fontsize=10)
    fig.supxlabel('Nominal environment action points per task (thousands)')
    fig.supylabel('Success rate (%)')
    fig.suptitle('Frozen 20 scenes/task | one training seed | only complete checkpoints shown', fontsize=15)
    fig.tight_layout(); save_figure(fig, 'e7-learning-curves')

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    for job in jobs:
        points = sorted((int(step), values['critic/explained_variance']) for step, values in job['metrics'].items()
                        if 'critic/explained_variance' in values)
        if points:
            axes[0].plot([p[0] for p in points], [p[1] for p in points], label=job['name'])
    axes[0].axhline(0, color='grey', linestyle=':'); axes[0].set(xlabel='Collection iteration', ylabel='Critic explained variance')
    axes[0].grid(alpha=.2)
    if axes[0].get_legend_handles_labels()[0]: axes[0].legend(fontsize=9)
    if gpu:
        xs = [(x['time_unix'] - gpu[0]['time_unix']) / 60 for x in gpu]
        axes[1].plot(xs, [x['utilization_percent'] for x in gpu], color='#2462a3', label='GPU utilization')
        mem = axes[1].twinx()
        mem.plot(xs, [x['used_gib'] for x in gpu], color='#d16b26', alpha=.65)
        mem.set(ylabel='GPU memory (GiB)', ylim=(0, gpu[-1]['total_gib']))
    axes[1].set(xlabel='Elapsed wall time (minutes)', ylabel='GPU utilization (%)', ylim=(0, 100))
    axes[1].grid(alpha=.2); fig.tight_layout(); save_figure(fig, 'e7-training-resource-curves')


def main(force=False):
    lock = (RUN / 'render.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    state = read(RUN / 'status.json', {})
    target = OUT / 'five-task-curves.json'
    final = state.get('phase') in ['E7完成', 'E7需检查', '训练完成，部分评估需检查']
    if target.exists() and time.time() - target.stat().st_mtime < 45 and not (force or final):
        return
    plan = read(RUN / 'plan.json'); jobs = collect(plan); gpu = lines(RUN / 'gpu-utilization.jsonl')
    plot(plan, jobs, gpu)
    data = {'updated': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'state': state,
            'reference': plan['reference'], 'budget': plan['budget'], 'jobs': jobs,
            'gpu_samples': len(gpu), 'latest_gpu': gpu[-1] if gpu else None,
            'observed_peak_gib': max((x['used_gib'] for x in gpu), default=0)}
    atomic(target, json.dumps(data, ensure_ascii=False, indent=2))
    esc = html.escape
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="60"><title>五任务单任务与共享RL曲线</title><style>body{max-width:1280px;margin:32px auto;padding:0 24px;font:19px/1.8 system-ui;color:#193245;background:#f3f6f8}section{background:white;padding:24px;margin:20px 0;border-radius:12px}h1{font-size:32px}h2{font-size:25px}table{border-collapse:collapse;width:100%;font-size:18px}th,td{border-bottom:1px solid #d5dfe5;padding:10px;text-align:left;vertical-align:top}.scroll{overflow:auto}img{width:100%;height:auto}a{color:#126b98}code{overflow-wrap:anywhere;font-size:17px}</style></head><body><h1>五个单任务策略 vs 一个五任务共享策略</h1>'''
    page += f'<p>更新：{esc(data["updated"])}。阶段：{esc(state.get("phase", "准备"))}；当前作业：{esc(state.get("job") or "—")}。</p>'
    page += '<section><h2>固定比较口径</h2><p>全部从已做多任务SFT的Motus π0.5独立初始化；Flow-SDE＋PPO、严格FP32、每批1遍更新、冻结VLM、无critic预热。1个训练种子，每个策略最多100轮，评估0／5／10／20／50／100轮。</p><p>每任务每轮2个环境×512个名义动作点。五个单任务策略合计与共享策略的总预算各为512000个名义动作点；实际执行量另列。共享策略每轮5次优化器更新，每个单任务策略每轮1次，五个单任务合计与共享策略的更新次数一致。</p><p>20个冻结开发场景只用于前置诊断；多训练几轮不保证收益，critic指标波动也尚未被证明是成功率波动的原因。</p></section>'
    page += '<section><h2>五任务成功率曲线</h2><p>横轴为每任务名义交互量，虚线为原始模型，蓝线为单任务RL，橙线为共享RL。只绘制跑满20场景的检查点；均值曲线要求五个任务全部评完。浅色区间是逐任务episode的95% Wilson区间，不是多个训练种子的方差。</p><img src="figures/e7-learning-curves.svg" alt="五个任务与宏平均的配对成功率曲线"><p><a href="figures/e7-learning-curves.png">PNG</a> · <a href="figures/e7-learning-curves.svg">SVG</a></p></section>'
    page += '<section><h2>运行进度与实际交互量</h2><div class="scroll"><table><tr><th>策略</th><th>训练轮次</th><th>完整检查点</th><th>审计批次</th><th>完成评估场景</th><th>实际动作点</th></tr>'
    for job in jobs:
        episodes = sum(x['completed'] for x in job['evaluations'])
        actual = sum(x['executed_actions'] for x in job['interactions'].values())
        checkpoints = '已保存：' + (str(job['ready_checkpoints']) if job['ready_checkpoints'] else '无')
        if job['pending_checkpoints']:
            checkpoints += '；写入中：' + str(job['pending_checkpoints'])
        page += f'<tr><td>{esc(job["name"])}</td><td>{job["completed_iterations"]}/100</td><td>{esc(checkpoints)}</td><td>{job["audited_batches"]}</td><td>{episodes}/{100 * len(job["tasks"])}</td><td>{actual}</td></tr>'
    page += '</table></div></section><section><h2>critic与GPU使用曲线</h2><p>critic曲线用于定位价值估计问题，不作为成功率变化的因果证据。GPU利用率为蓝线，显存为橙线；模型初始化、CPU仿真、同步和写盘阶段可能降低GPU利用率。</p>'
    if gpu:
        last = gpu[-1]
        page += f'<p>最近GPU利用率：{last["utilization_percent"]:.0f}%；显存：{last["used_gib"]:.1f}/{last["total_gib"]:.1f} GiB；采样观测峰值：{data["observed_peak_gib"]:.1f} GiB。训练时最多重叠1个评估进程，须先通过至少2轮训练的显存检查并留出余量；纯评估阶段最多2路并行。</p>'
    page += '<img src="figures/e7-training-resource-curves.svg" alt="critic解释方差及GPU利用率与显存曲线"></section>'
    if state.get('failed_evaluations'):
        page += f'<section><h2>需要处理的评估</h2><p>{esc(str(state["failed_evaluations"]))}</p></section>'
    page += '<p><a href="five-task-curves.json">机器可读结果</a> · <a href="../.runtime/experiments/e7-five-task-curves/plan.json">冻结计划</a> · <a href="takeaway.md">经验记录</a> · <a href="progress.html">总进度</a></p></body></html>'
    atomic(OUT / 'five-task-curves.html', page)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--watch', action='store_true'); parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.watch:
        while True:
            main()
            state = read(RUN / 'status.json', {})
            if state.get('phase') in ['E7完成', 'E7需检查', '训练完成，部分评估需检查']:
                main(force=True); break
            time.sleep(30)
    else:
        main(force=args.force)
