"""Render the bounded E6 queue and its paired results."""
from research_artifacts import artifact_path, relocate_html
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
        ready_checkpoints, pending_checkpoints = [], []
        for checkpoint in (folder / job['name'] / 'checkpoints').glob('global_step_*'):
            ready = ((checkpoint / 'actor/diagnostic-state.json').exists()
                     if job.get('checkpoint_receipts_required', False)
                     else (checkpoint / 'actor/model_state_dict/full_weights.pt').exists())
            (ready_checkpoints if ready else pending_checkpoints).append(int(checkpoint.name.split('_')[-1]))
        jobs.append({'name': job['name'], 'completed_iterations': actual, 'target': job['iterations'],
                     'finished': (folder / 'TRAINING_FINISHED').exists(),
                     'audited_batches': len(audits), 'latest_audit': audits[-1] if audits else None,
                     'ready_checkpoints': sorted(ready_checkpoints), 'pending_checkpoints': sorted(pending_checkpoints)})
        if job['kind'] != 'critic_only':
            for step in [5, 10]:
                variants.append((f'PPO，1遍更新，{step}轮', folder / f'evaluation-{step}/clean/adjust_bottle'))
    results = []
    base_folder = ROOT / '.runtime/experiments/e1-reference-clean/clean/adjust_bottle'
    base_records = {seed: read(base_folder / f'episode_{seed}.json', {}) for seed in seed_ids}
    for name, folder in variants:
        records = [read(folder / f'episode_{s}.json', {}) for s in seed_ids]
        records = [x for x in records if x.get('status') == 'completed']
        paired = [(x, base_records[x['seed']]) for x in records
                  if base_records[x['seed']].get('status') == 'completed']
        results.append({'name': name, 'completed': len(records), 'successes': sum(x['success'] for x in records), 'target': 20,
                        'paired_episodes': len(paired),
                        'base_successes_on_same_seeds': sum(b['success'] for x, b in paired),
                        'new_wins': sum(x['success'] and not b['success'] for x, b in paired),
                        'new_losses': sum(not x['success'] and b['success'] for x, b in paired)})
    numerics = {'default': read(RUN / 'numerical-diagnostic/results.json', {}),
                'strict_fp32': read(RUN / 'numerical-diagnostic-highest/results.json', {})}
    artifact = {'updated': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'state': state, 'jobs': jobs,
                'paired_results': results, 'aborted_attempts': plan.get('aborted_attempts', []),
                'numerical_diagnostics': numerics}
    out = ROOT / '.research'
    target = artifact_path('stability-progress.json'); tmp = target.with_suffix(f'.{os.getpid()}.tmp'); tmp.write_text(json.dumps(artifact, ensure_ascii=False, indent=2)); tmp.replace(target)
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="30"><title>E6 后训练稳定性诊断</title><style>body{max-width:1100px;margin:36px auto;padding:0 24px;font:19px/1.8 system-ui;color:#193245;background:#f4f7f9}section{padding:24px;background:white;margin:22px 0;border-radius:12px}table{width:100%;border-collapse:collapse}td,th{padding:12px;text-align:left;border-bottom:1px solid #ddd}a{color:#087e80}</style><h1>E6：先确认后训练为何退化</h1>'''
    page += f'<p>{html.escape(artifact["updated"])}</p><section><h2>{html.escape(state.get("phase", "准备"))}</h2><p>{html.escape(state.get("detail", ""))}</p></section>'
    if artifact['aborted_attempts']:
        actions = sum(attempt['nominal_actions'] for attempt in artifact['aborted_attempts'])
        completed = sum(attempt['completed_iterations'] for attempt in artifact['aborted_attempts'])
        page += f'<section><h2>中断尝试与额外消耗</h2><p>已有{len(artifact["aborted_attempts"])}次中断尝试，共完成{completed}轮优化、消耗{actions}个名义动作点，计入实际成本。历史故障包括权重同步／模型装载时的显存不足，以及概率差异检查停止；其中一次完成4轮、在第5轮更新前停止。日志、配置和轨迹均单独归档。当前采用CPU分桶传输、同步后清理GPU缓存并禁用TF32；下表只统计当前尝试。第1／5／10轮保存检查点，审计中断也先保存当前进度。</p></section>'
    if all('train2_vs_train4' in data for data in numerics.values()):
        page += '<section><h2>停止原因：默认TF32带来的数值敏感性</h2><p>原来的0.01逐元素阈值在14336项中被1项触发。固定原始Motus权重与真实输入，只改变批大小，复现了数值差异；关闭TF32后明显降低。训练与采样两侧现使用最高FP32计算精度，保留原停止阈值。</p><table><tr><th>同权重对照</th><th>逐元素最大差</th><th>PPO比率最大偏差</th></tr>'
        for key, name in [('default', '默认：允许TF32'), ('strict_fp32', '修复：禁用TF32')]:
            result = numerics[key]['train2_vs_train4']
            page += f'<tr><td>{name}</td><td>{result["component_max_abs"]:.6g}</td><td>{100 * result["chunk_ratio_error_max"]:.4f}%</td></tr>'
        page += '</table><p>这是同权重数值对照，不是丢失的第4轮权重重放，也不能证明TF32解释了此前成功率下降。当前D2还改变了计算精度，与历史5遍PPO比较时，不能单独归因于更新次数。</p></section>'
    page += '<section><h2>本轮范围</h2><p>D0：原始权重SDE采样，20场景。D1：冻结actor，仅训练critic，5轮；另续跑1轮验收批次审计，分开记录。D2：从原始权重开始，每批只更新1遍，训练10轮，评估第5／10轮。D1与D2独立初始化，不混用critic权重。</p><p>每个训练作业每轮2×512名义动作点。D2与旧5遍更新的前10轮匹配名义采样预算，轨迹会因策略更新而不同。重算logprob仅用于诊断，保留原rollout概率作为PPO分母；发生明显不一致时停止队列。</p><table><tr><th>作业</th><th>训练进度</th><th>完成标记</th></tr>'
    for job in jobs:
        checkpoint_text = '已保存：' + ('、'.join(map(str, job['ready_checkpoints'])) or '无')
        if job['pending_checkpoints']:
            checkpoint_text += '；尚未写完：' + '、'.join(map(str, job['pending_checkpoints']))
        page += f'<tr><td>{html.escape(job["name"])}</td><td>{job["completed_iterations"]}/{job["target"]}</td><td>{job["finished"]}<br>{checkpoint_text}</td></tr>'
    page += '</table><p>保存完整检查点可能需要数分钟，写盘时GPU利用率下降属于正常现象；只有完整保存的检查点才计入“已保存”。</p>'
    page += '<h2>更新前批次核验</h2><p>比较采样时记录的概率与训练时重算的概率。两者应在数值误差内一致；重算值只用于核验，不替换PPO分母。critic诊断的前5轮没有审计记录，补跑1轮单独验收。</p><table><tr><th>作业</th><th>已有审计批次</th><th>最近最大概率对数差</th><th>仅价值头可训练</th></tr>'
    for job in jobs:
        audit = job['latest_audit']
        gap = f'{audit["logprob_gap_max_abs"]:.6g}' if audit else '待验收'
        frozen = ('是' if audit['only_value_head_trainable'] else '否') if audit else '待验收'
        page += f'<tr><td>{html.escape(job["name"])}</td><td>{job["audited_batches"]}</td><td>{gap}</td><td>{frozen}</td></tr>'
    page += '</table></section><section><h2>同20场景配对结果</h2><table><tr><th>策略</th><th>进度</th><th>成功数</th><th>相同已完成场景的原始成功数</th><th>新增成功／丢失成功</th></tr>'
    for result in results:
        score = f'{result["successes"]}/{result["completed"]}' if result['completed'] else '待评估'
        paired_score = f'{result["base_successes_on_same_seeds"]}/{result["paired_episodes"]}' if result['paired_episodes'] else '—'
        page += f'<tr><td>{html.escape(result["name"])}</td><td>{result["completed"]}/20</td><td>{score}</td><td>{paired_score}</td><td>{result["new_wins"]}／{result["new_losses"]}</td></tr>'
    page += '</table><p>未完成场景不算失败；部分结果只与相同已完成场景比较。这是单任务、单种子诊断，不能当作多任务冲突或论文正式性能证据。</p>'
    for result in results:
        if result['name'].startswith('PPO，1遍更新，') and result['completed'] == 20:
            change = result['new_wins'] - result['new_losses']
            page += f'<p>{html.escape(result["name"])}：成功{result["successes"]}/20，对比原始{result["base_successes_on_same_seeds"]}/20；新增{result["new_wins"]}、丢失{result["new_losses"]}，净变化{change:+d}个场景。20场景仍不足以确认稳定收益；当前配置同时减少更新次数并禁用TF32，不能单独归因。</p>'
    recovery = RUN / 'jobs/ppo-update1-adjust_bottle-seed0/evaluation-10/recovery'
    if list(recovery.glob('*/recovery.json')):
        page += '<p>第10轮评估曾在第7个场景的第64步停滞。已归档日志和已完成结果哈希，从相同检查点、相同冻结种子恢复，未重跑训练。新增进程外15分钟心跳检查及一次同种子重试；超时记录不计策略失败。</p>'
    page += '</section><p><a href="takeaway.md">训练与排障经验</a> · <a href="phenomena-paper-audit.html">原论文核对</a> · <a href="../.runtime/experiments/e6-posttraining-diagnostic/plan.json">冻结诊断计划</a> · <a href="progress.html">总记录</a></p></html>'
    target = artifact_path('stability-progress.html'); tmp = target.with_suffix(f'.{os.getpid()}.tmp'); tmp.write_text(relocate_html(page, 'stability-progress.html')); tmp.replace(target)


if __name__ == '__main__':
    main()
