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
        jobs.append({'name': job['name'], 'completed_iterations': actual, 'target': job['iterations'], 'finished': (folder / 'TRAINING_FINISHED').exists()})
        if job['kind'] != 'critic_only':
            for step in [5, 10]:
                variants.append((f'PPO，1遍更新，{step}轮', folder / f'evaluation-{step}/clean/adjust_bottle'))
    results = []
    for name, folder in variants:
        records = [read(folder / f'episode_{s}.json', {}) for s in seed_ids]
        records = [x for x in records if x.get('status') == 'completed']
        results.append({'name': name, 'completed': len(records), 'successes': sum(x['success'] for x in records), 'target': 20})
    artifact = {'updated': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'state': state, 'jobs': jobs, 'paired_results': results}
    out = ROOT / '.research'
    target = out / 'stability-progress.json'; tmp = target.with_suffix(f'.{os.getpid()}.tmp'); tmp.write_text(json.dumps(artifact, ensure_ascii=False, indent=2)); tmp.replace(target)
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="30"><title>E6 后训练稳定性诊断</title><style>body{max-width:1100px;margin:36px auto;padding:0 24px;font:19px/1.8 system-ui;color:#193245;background:#f4f7f9}section{padding:24px;background:white;margin:22px 0;border-radius:12px}table{width:100%;border-collapse:collapse}td,th{padding:12px;text-align:left;border-bottom:1px solid #ddd}a{color:#087e80}</style><h1>E6：先确认后训练为何退化</h1>'''
    page += f'<p>{html.escape(artifact["updated"])}</p><section><h2>{html.escape(state.get("phase", "准备"))}</h2><p>{html.escape(state.get("detail", ""))}</p></section>'
    page += '<section><h2>本轮范围</h2><p>D0：原始权重SDE采样，20场景。D1：冻结actor，仅训练critic，5轮。D2：从原始权重开始，每批只更新1遍，训练10轮，评估第5／10轮。D1与D2独立初始化，不混用critic权重。</p><p>每个训练作业每轮2×512名义动作点。D2与旧5遍更新的前10轮匹配名义采样预算，轨迹会因策略更新而不同。重算logprob仅用于诊断，保留原rollout概率作为PPO分母；发生明显不一致时停止队列。</p><table><tr><th>作业</th><th>训练进度</th><th>完成标记</th></tr>'
    page += ''.join(f'<tr><td>{html.escape(j["name"])}</td><td>{j["completed_iterations"]}/{j["target"]}</td><td>{j["finished"]}</td></tr>' for j in jobs)
    page += '</table></section><section><h2>同20场景配对结果</h2><table><tr><th>策略</th><th>进度</th><th>成功数</th></tr>'
    page += ''.join(f'<tr><td>{html.escape(r["name"])}</td><td>{r["completed"]}/20</td><td>{r["successes"]}/{r["completed"]}</td></tr>' for r in results)
    page += '</table><p>未完成场景不算失败。这是单任务、单种子诊断，不能当作多任务冲突或论文正式性能证据。</p></section><p><a href="phenomena-paper-audit.html">原论文核对</a> · <a href="../.runtime/experiments/e6-posttraining-diagnostic/plan.json">冻结诊断计划</a> · <a href="progress.html">总记录</a></p></html>'
    target = out / 'stability-progress.html'; tmp = target.with_suffix(f'.{os.getpid()}.tmp'); tmp.write_text(page); tmp.replace(target)


if __name__ == '__main__':
    main()
