"""Show only paired evidence and the bounded pilot queue, preserving full-run history."""
import html
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / '.runtime/experiments/e5-pirl'
JOB = RUN / 'jobs/single-adjust_bottle-seed0'
BASE = ROOT / '.runtime/experiments/e1-reference-clean'


def render():
    state = json.loads((RUN / 'status.json').read_text())
    seeds = json.loads((BASE / 'seed_manifest.json').read_text())['tasks']['adjust_bottle']['valid_seeds'][:20]
    baseline = {s: json.loads((BASE / f'clean/adjust_bottle/episode_{s}.json').read_text()) for s in seeds}
    rows, evidence = [], []
    for label, folder in [('原始多任务SFT', BASE / 'clean/adjust_bottle'),
                          ('单任务RL · 10轮', JOB / 'curve-10/clean/adjust_bottle'),
                          ('单任务RL · 20轮', JOB / 'curve-20/clean/adjust_bottle'),
                          ('单任务RL · 100轮', JOB / 'evaluation/clean/adjust_bottle')]:
        records = []
        for seed in seeds:
            p = folder / f'episode_{seed}.json'
            if p.exists():
                d = json.loads(p.read_text())
                if d.get('status') == 'completed':
                    records.append(d)
        n = len(records)
        wins = sum(x['success'] and not baseline[x['seed']]['success'] for x in records)
        losses = sum(not x['success'] and baseline[x['seed']]['success'] for x in records)
        successes = sum(x['success'] for x in records)
        evidence.append({'label': label, 'source': str(folder), 'completed': n, 'target': 20,
                         'successes': successes, 'paired_original_successes': sum(baseline[x['seed']]['success'] for x in records),
                         'new_wins': wins, 'new_losses': losses, 'seeds': [x['seed'] for x in records]})
        rate = f'{successes / n:.0%}' if n == 20 else '待完成'
        rows.append(f'<tr><td>{label}</td><td>{n}/20</td><td>{successes}/{n}</td><td>{rate}</td><td>{wins} / {losses}</td></tr>')
    updated = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S 北京时间')
    artifact = {'updated': updated, 'use': 'preliminary diagnostic, not final paper evaluation',
                'state': state, 'paired_checkpoint_results': evidence}
    target = ROOT / '.research/pirl-pilot-results.json'
    tmp = target.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(artifact, ensure_ascii=False, indent=2)); tmp.replace(target)
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="30"><title>Post-training 前置实验</title><style>body{max-width:1100px;margin:35px auto;padding:0 24px;background:#f5f7f9;color:#172b3a;font:19px/1.8 system-ui}section{padding:24px;background:white;margin:20px 0;border-radius:12px}table{width:100%;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}a{color:#087e80}</style></head><body><h1>快速验证：先确认能改善，再比较多任务</h1>'''
    page += f'<p>{updated}</p><section><h2>正在做什么</h2><p><strong>{html.escape(state.get("phase", "准备"))}</strong></p><p>{html.escape(state.get("detail", ""))}</p><p>已完成第一个单任务策略的100轮训练。完整队列剩余17个PPO作业与18个成功重放作业暂不启动。全部检查点和已完成的逐场景结果保留。</p></section>'
    page += '<section><h2>第一步：已有权重快速诊断</h2><p>同一个 adjust_bottle 任务、同一组20个场景、同一参考推理入口。并行评估第10、20轮；原始模型与第100轮直接复用已有记录。每个预定检查点都报告，不挑最好结果。</p><table><tr><th>策略</th><th>进度</th><th>成功数</th><th>成功率</th><th>新增成功 / 新增失败</th></tr>' + ''.join(rows) + '</table><p>100轮在这20个场景从9次成功降至1次，是当前配置的退化信号。尚不能据此断言RL不可行，也不能归因于多任务冲突；当前训练只包含一个任务。原100场景评估中的其余已完成记录保留，未完成部分不算失败。</p></section>'
    page += '<section><h2>第二步：条件满足后的小矩阵</h2><p>任务：调整瓶子、叠两块积木、容器放盘。固定1个训练种子、20轮、每任务2个环境。比较原始模型、3个单任务策略、1个三任务共享策略，每任务评估20个配对场景。该三任务子集的原始参照是30/60＝50%。</p><p>保持同一训练配置时，可复用现有瓶子任务第20轮，仅需再训练2个单任务策略和1个共享策略。三个独立策略合计与共享策略各有61440个名义动作点，另记录实际执行步数。若修正训练配置，则受影响策略重新运行，保证比较一致。</p><p>先分析本轮退化，再确定是否启动该矩阵。若需要区分奖励优化和单纯继续监督训练，可增加1个共享成功重放对照，使用同一批经验。</p></section>'
    page += '<section><h2>前置实验只回答三个问题</h2><ol><li>当前后训练配置能否改善已有策略，还是迅速损伤能力？</li><li>相同预算下，共享优化与单独优化是否出现值得复查的差距？</li><li>收益是否覆盖多个任务，同时有哪些任务退化？</li></ol><p>20场景中一次成功相差5个百分点，仅用于发现信号和排查问题。先不扩到三种子、全部22任务、50任务或长程任务。论文结论需在方案确定后，用独立测试场景和多个训练种子确认。</p></section>'
    page += '<p><a href="stability-progress.html">E6正在运行的诊断</a> · <a href="phenomena-paper-audit.html">现象与原论文核对</a> · <a href="pirl-pilot-results.json">当前诊断数据</a> · <a href="../.runtime/experiments/e5-pirl/pilot-plan.json">快速实验计划</a> · <a href="../.runtime/experiments/e5-pirl/plan.json">保留的完整矩阵</a> · <a href="e4-analysis.html">上一轮离线实验</a> · <a href="progress.html">总记录</a></p></body></html>'
    target = ROOT / '.research/pirl-progress.html'
    tmp = target.with_suffix(f'.{os.getpid()}.tmp'); tmp.write_text(page); tmp.replace(target)


if __name__ == '__main__':
    render()
