"""Render the bounded E6 queue and its paired results."""
import html
import json
import os
from pathlib import Path
import re
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / '.runtime/experiments/e6-posttraining-diagnostic'


def read(path, fallback):
    return json.loads(path.read_text()) if path.exists() else fallback


def main():
    state = read(RUN / 'status.json', {})
    plan = read(RUN / 'plan.json', {})
    e5 = ROOT / '.runtime/experiments/e5-pirl'
    variants = [('原始参考ODE', ROOT / '.runtime/experiments/e1-reference-clean/clean/adjust_bottle'),
                ('原始RLinf ODE', e5 / 'zero-update-evaluation/clean/adjust_bottle'),
                ('原始RLinf SDE，无更新', RUN / 'zero-sde/clean/adjust_bottle'),
                ('原配置PPO，5遍更新，10轮', e5 / 'jobs/single-adjust_bottle-seed0/curve-10/clean/adjust_bottle')]
    seed_ids = read(ROOT / '.runtime/experiments/e1-reference-clean/seed_manifest.json', {})['tasks']['adjust_bottle']['valid_seeds'][:20]
    jobs = []
    for job in plan.get('jobs', []):
        folder = RUN / 'jobs' / job['name']
        log = folder / 'training.log'
        steps = re.findall(r'Global Step:\s*(\d+)\s*/\s*(\d+)', log.read_text(errors='replace')) if log.exists() else []
        actual = max((int(a) for a, b in steps), default=0)
        audit_path = folder / 'batch-audit.jsonl'
        audits = []
        if audit_path.exists():
            # A worker may be appending the last line while the page refreshes.
            for line in audit_path.read_text().splitlines(keepends=True):
                if line.endswith('\n'):
                    audits.append(json.loads(line))
        jobs.append({'name': job['name'], 'completed_iterations': actual, 'target': job['iterations'],
                     'finished': (folder / 'TRAINING_FINISHED').exists(),
                     'audited_batches': len(audits), 'latest_audit': audits[-1] if audits else None})
        if job['kind'] != 'critic_only':
            for step in [5, 10]:
                variants.append((f'PPO，1遍更新，{step}轮', folder / f'evaluation-{step}/clean/adjust_bottle'))
    results = []
    for name, folder in variants:
        records = [read(folder / f'episode_{s}.json', {}) for s in seed_ids]
        records = [x for x in records if x.get('status') == 'completed']
        results.append({'name': name, 'completed': len(records), 'successes': sum(x['success'] for x in records), 'target': 20})
    artifact = {'updated': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'state': state, 'jobs': jobs,
                'paired_results': results, 'aborted_attempts': plan.get('aborted_attempts', [])}
    out = ROOT / '.research'
    target = out / 'stability-progress.json'; tmp = target.with_suffix(f'.{os.getpid()}.tmp'); tmp.write_text(json.dumps(artifact, ensure_ascii=False, indent=2)); tmp.replace(target)
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="30"><title>E6 后训练稳定性诊断</title><style>body{max-width:1100px;margin:36px auto;padding:0 24px;font:19px/1.8 system-ui;color:#193245;background:#f4f7f9}section{padding:24px;background:white;margin:22px 0;border-radius:12px}table{width:100%;border-collapse:collapse}td,th{padding:12px;text-align:left;border-bottom:1px solid #ddd}a{color:#087e80}</style><h1>E6：先确认后训练为何退化</h1>'''
    page += f'<p>{html.escape(artifact["updated"])}</p><section><h2>{html.escape(state.get("phase", "准备"))}</h2><p>{html.escape(state.get("detail", ""))}</p></section>'
    if artifact['aborted_attempts']:
        actions = sum(attempt['nominal_actions'] for attempt in artifact['aborted_attempts'])
        completed = sum(attempt['completed_iterations'] for attempt in artifact['aborted_attempts'])
        page += f'<section><h2>中断尝试与额外消耗</h2><p>已有{len(artifact["aborted_attempts"])}次中断尝试，共完成{completed}轮优化、消耗{actions}个名义动作点，计入实际成本。首次在权重差分同步时显存不足；随后一次默认分桶同步在更新前被概率差异检查拦截。日志、配置和轨迹均单独归档。当前采用官方FP32分桶传输全部状态，从原始Motus权重开始；下表只统计当前尝试。</p></section>'
    page += '<section><h2>本轮范围</h2><p>D0：原始权重SDE采样，20场景。D1：冻结actor，仅训练critic，5轮；另续跑1轮验收批次审计，分开记录。D2：从原始权重开始，每批只更新1遍，训练10轮，评估第5／10轮。D1与D2独立初始化，不混用critic权重。</p><p>每个训练作业每轮2×512名义动作点。D2与旧5遍更新的前10轮匹配名义采样预算，轨迹会因策略更新而不同。重算logprob仅用于诊断，保留原rollout概率作为PPO分母；发生明显不一致时停止队列。</p><table><tr><th>作业</th><th>训练进度</th><th>完成标记</th></tr>'
    page += ''.join(f'<tr><td>{html.escape(j["name"])}</td><td>{j["completed_iterations"]}/{j["target"]}</td><td>{j["finished"]}</td></tr>' for j in jobs)
    page += '</table><h2>更新前批次核验</h2><p>比较采样时记录的概率与训练时重算的概率。两者应在数值误差内一致；重算值只用于核验，不替换PPO分母。critic诊断的前5轮没有审计记录，补跑1轮单独验收。</p><table><tr><th>作业</th><th>已有审计批次</th><th>最近最大概率对数差</th><th>仅价值头可训练</th></tr>'
    for job in jobs:
        audit = job['latest_audit']
        gap = f'{audit["logprob_gap_max_abs"]:.6g}' if audit else '待验收'
        frozen = ('是' if audit['only_value_head_trainable'] else '否') if audit else '待验收'
        page += f'<tr><td>{html.escape(job["name"])}</td><td>{job["audited_batches"]}</td><td>{gap}</td><td>{frozen}</td></tr>'
    page += '</table></section><section><h2>同20场景配对结果</h2><table><tr><th>策略</th><th>进度</th><th>成功数</th></tr>'
    for result in results:
        score = f'{result["successes"]}/{result["completed"]}' if result['completed'] else '待评估'
        page += f'<tr><td>{html.escape(result["name"])}</td><td>{result["completed"]}/20</td><td>{score}</td></tr>'
    page += '</table><p>未完成场景不算失败。这是单任务、单种子诊断，不能当作多任务冲突或论文正式性能证据。</p></section><p><a href="phenomena-paper-audit.html">原论文核对</a> · <a href="../.runtime/experiments/e6-posttraining-diagnostic/plan.json">冻结诊断计划</a> · <a href="progress.html">总记录</a></p></html>'
    target = out / 'stability-progress.html'; tmp = target.with_suffix(f'.{os.getpid()}.tmp'); tmp.write_text(page); tmp.replace(target)


if __name__ == '__main__':
    main()
