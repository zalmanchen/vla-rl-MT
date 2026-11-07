"""One concise index of verified results, live runs and organized evidence."""
import csv
from datetime import datetime
import html
import io
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from research_artifacts import ROOT, RESEARCH, artifact_path


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write(path, text):
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(text); tmp.replace(path)


def href(path):
    return os.path.relpath(path, RESEARCH)


def main():
    now = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    e1 = read(artifact_path('e1-results.json'))
    e3 = read(artifact_path('motus-window-diagnostic.json'))
    e4 = read(artifact_path('e4-analysis.json'))
    e5 = read(artifact_path('pirl-pilot-results.json'))
    e6 = read(artifact_path('e6-progress-analysis.json'))
    e7 = read(RESEARCH / 'five-task-curves.json', {})
    pace = read(RESEARCH / 'pace-comparison.json', {})
    e7_analysis = read(RESEARCH / 'results/e7/e7-analysis.json', {})
    organization = read(RESEARCH / 'organization-latest.json', {})
    registry = [
        {'id':'E0', 'name':'推理接入诊断', 'status':'诊断完成；数值对齐未通过', 'finding':'合成观测的全模型数值对照未达原阈值；后续132个真实快照的输入／输出变换核对通过。两者验证范围不同。', 'run':'.runtime/experiments/e0-parity', 'report':'diagnostics/model/motus-transform-audit.json'},
        {'id':'E1', 'name':'原始Motus多任务SFT基线', 'status':'完成：22任务×100场景', 'finding':'566/2200，25.73%。固定Clean本地协议；不是论文官方成绩。', 'run':'.runtime/experiments/e1-reference-clean', 'report':'results/e1/e1-results.csv'},
        {'id':'E2', 'name':'PACE单任务SFT权重对照', 'status':pace.get('phase','尚无结果'), 'finding':'尚无有效评测成绩；保留为待完成外部checkpoint对照，不填为0%。', 'run':'.runtime/experiments/e2-pace-single-task', 'report':'pace-comparison.json'},
        {'id':'E3', 'name':'动作执行窗口诊断', 'status':'完成：3任务×3窗口×20场景', 'finding':'H=8/16/32的排序随任务变化，较短窗口并非总是更好。', 'run':'.runtime/experiments/e3-motus-window-diagnostic', 'report':'results/e3/motus-window-diagnostic.json'},
        {'id':'E4', 'name':'离线RECAP短预算适配', 'status':'完成：5个ST＋1个MT＋1个重放', 'finding':'同100场景：原始47、ST29、MT29、成功重放25；没有总体MT低于ST的证据。不是原论文完整复现。', 'run':'.runtime/experiments/e4-offline-rl', 'report':'results/e4/e4-analysis.html'},
        {'id':'E5', 'name':'旧配置单任务Flow-SDE＋PPO', 'status':'已完成瓶子任务；其余旧矩阵关闭', 'finding':'同20场景：原始9；10/20/100轮为7/2/1。该退化发生在单任务训练。', 'run':'.runtime/experiments/e5-pirl', 'report':'results/e5/pirl-progress.html'},
        {'id':'E6', 'name':'精度、显存与短程PPO诊断', 'status':'完成：D0、D1、D2', 'finding':'原始9/20，第5轮11/20，第10轮9/20；工程审计通过，收益没有维持。精度和更新次数同时变化，不能单独归因。', 'run':'.runtime/experiments/e6-posttraining-diagnostic', 'report':'results/e6/stability-progress.html'},
        {'id':'E7', 'name':'五任务单任务／共享RL曲线', 'status':e7.get('state',{}).get('phase','准备'), 'finding':'当前主实验：5个ST＋1个MT，各100轮、1个训练种子，每任务交互预算一致。未完成对照不能解释为多任务优劣。', 'run':'.runtime/experiments/e7-five-task-curves', 'report':'five-task-curves.html'},
    ]
    if e7.get('state', {}).get('phase') == 'E7需检查' and e7_analysis:
        registry[-1].update(status='中断：共享59/100轮，5个单任务未启动',
                            finding='共享原始47%；第5/10/20/50轮为48/47/41/32%。主机内存压力导致训练中断。已观察到技能收益不均与保持问题，尚无ST/MT对照结论。',
                            report='results/e7/e7-analysis.html')
    rows = []
    def row(experiment, variant, task, step, seed, k, n, target, evidence):
        complete = n == target
        rows.append({'experiment':experiment,'variant':variant,'task':task,'step':step,'training_seed':seed,
                     'successes':k if n else None,'completed':n,'target':target,
                     'success_rate':k/n if complete and n else None,'complete':complete,'evidence':evidence})
    for task in e1['tasks']:
        row('E1','Motus SFT',task['task'],0,'',task['successes'],task['completed'],100,
            f'.runtime/experiments/e1-reference-clean/clean/{task["task"]}/summary.json')
    for task in pace.get('tasks', []):
        row('E2','PACE single-task SFT',task['task'],0,'',task['successes'],task['completed'],task['target'],
            '.research/pace-comparison.json')
    for task in e3['tasks']:
        row('E3',f'execution_horizon_{task["execution_horizon"]}',task['task'],0,'',
            task['successes'],task['completed'],task['target'],'.research/results/e3/motus-window-diagnostic.json')
    for task in e4['rows']:
        for variant in ['original','success_replay','single_task_rl','multi_task_rl']:
            row('E4',variant,task['task'],'',0,task[variant],20,20,'.research/results/e4/e4-analysis.json')
    for data, step in zip(e5['paired_checkpoint_results'], [0, 10, 20, 100], strict=True):
        row('E5',data['label'],'adjust_bottle',step,0,data['successes'],data['completed'],20,data['source'])
    for variant, data in e6['evaluations'].items():
        step = int(variant.rsplit('step',1)[1]) if 'step' in variant else 0
        row('E6',variant,'adjust_bottle',step,0,data['successes'],data['completed'],20,data['source'])
    for job in e7.get('jobs',[]):
        for data in job['evaluations']:
            row('E7',job['name'],data['task'],data['step'],job['seed'],data['successes'],data['completed'],20,
                f'.runtime/experiments/e7-five-task-curves/jobs/{job["name"]}/curve-{data["step"]}/clean/{data["task"]}')
    index = {'updated':now,'experiments':registry,'results':rows,
             'live_e7_snapshot_time':e7.get('updated'), 'organization':organization,
             'recovery_costs':[{k:j[k] for k in ['name','resume_attempts','discarded_attempt_interactions']}
                               for j in e7.get('jobs',[]) if j.get('resume_attempts')],
             'rules':['Do not compare unmatched protocols as controlled algorithm rankings.',
                      'Pending evaluations have no success-rate estimate in this summary.',
                      '20-scene development comparisons and one training seed do not establish final paper performance.']}
    write(RESEARCH/'experiment-index.json',json.dumps(index,ensure_ascii=False,indent=2))
    buf = io.StringIO(); writer = csv.DictWriter(buf,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write(RESEARCH/'results-summary.csv',buf.getvalue())
    md = ['# 实验结果总览',f'\n整理时间：{now}（Asia/Shanghai）。本表为生成时快照，E7最新曲线见[实时页面](five-task-curves.html)。',
          '\n| 实验 | 状态 | 结果与解释 | 详细记录 |','| --- | --- | --- | --- |']
    for exp in registry:
        md.append(f'| {exp["id"]} {exp["name"]} | {exp["status"]} | {exp["finding"]} | [报告]({exp["report"]}) · [原始记录](../{exp["run"]}/) |')
    md += ['\n不同实验的任务集合、预算和适配程度不同，不能将此表直接当作算法排名。E7只在同一任务集合全部评完后计算宏平均；未完成评估不填失败或0%。',
           '\n[可导出CSV](results-summary.csv) · [结构化索引](experiment-index.json) · [训练经验](takeaway.md) · [目录说明](README.md)']
    write(RESEARCH/'results-summary.md','\n'.join(md)+'\n')
    esc = html.escape
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>实验结果总览</title><style>body{max-width:1260px;margin:32px auto;padding:0 24px;font:19px/1.8 system-ui;color:#193245;background:#f3f6f8}section{background:white;padding:24px;margin:22px 0;border-radius:12px}h1{font-size:34px}h2{font-size:25px}table{width:100%;border-collapse:collapse;font-size:18px}th,td{border-bottom:1px solid #d8e1e6;padding:12px;text-align:left;vertical-align:top}a{color:#126b98}.scroll{overflow:auto}nav a{margin-right:18px}code{font-size:17px;overflow-wrap:anywhere}</style></head><body><h1>实验结果总览</h1>'''
    page += f'<p>更新：{esc(now)}（Asia/Shanghai）。原始模型为已完成多任务SFT的Motus π0.5。</p><nav><a href="five-task-curves.html">当前五任务曲线</a><a href="takeaway.md">训练经验</a><a href="results-summary.csv">导出结果CSV</a><a href="README.md">目录与原始证据</a></nav>'
    state=e7.get('state',{}); jobs=e7.get('jobs',[])
    page += '<section><h2>当前主实验：E7</h2>'
    page += f'<p>曲线数据快照：{esc(e7.get("updated","尚无"))}。阶段：{esc(state.get("phase","准备"))}；当前作业：{esc(state.get("job") or "—")}。</p>'
    if e7.get('state', {}).get('phase') == 'E7需检查':
        page += '<p><strong>训练已中断，并非全部完成。</strong>停止详情与已完成结果见 <a href="results/e7/e7-analysis.html">E7结果分析</a>。尚未执行的单任务与第100轮不记为失败。</p>'
    if jobs:
        multi=next(j for j in jobs if j['kind']=='multi')
        trained=sum(j['finished'] for j in jobs if j['kind']=='single')
        started=sum(j['completed_iterations']>0 for j in jobs if j['kind']=='single')
        page += f'<p>共享策略训练 {multi["completed_iterations"]}/100 轮；5个单任务策略中，已开始{started}个、已完成{trained}个。当前1个训练种子，每任务20个冻结开发场景。</p>'
        if multi.get('resume_attempts'):
            lost = multi['discarded_attempt_interactions']
            page += f'<p>该共享策略曾中断并从完整检查点恢复。已回退尝试额外消耗{lost["nominal_actions"]:,}个名义动作点、{lost["executed_actions"]:,}个实际动作点，单列成本；环境与随机数连续性不保证。<a href="five-task-curves.html">恢复记录及当前曲线</a>。</p>'
        progress_steps=[step for step in [5,10,20,50,100] if any(x['step']==step and x['completed'] for x in multi['evaluations'])]
        if progress_steps:
            page += '<div class="scroll"><table><tr><th>任务</th><th>原始模型</th>' + ''.join(f'<th>共享策略第{s}轮</th>' for s in progress_steps) + '</tr>'
            for task,base in e7['reference'].items():
                page += f'<tr><td>{esc(task)}</td><td>{base["successes"]}/20</td>'
                for step in progress_steps:
                    result=next(x for x in multi['evaluations'] if x['task']==task and x['step']==step)
                    score=f'{result["successes"]}/20' if result['completed']==20 else f'评估中：已完成{result["completed"]}/20'
                    page += f'<td>{score}</td>'
                page += '</tr>'
            page += '<tr><td>五任务合计</td><td>47/100</td>'
            for step in progress_steps:
                cells=[x for x in multi['evaluations'] if x['step']==step]
                score=f'{sum(x["successes"] for x in cells)}/100' if len(cells)==5 and all(x['completed']==20 for x in cells) else '等待五任务全部评完'
                page+=f'<td>{score}</td>'
            page+='</tr></table></div>'
    page += '<p>评估未完成的单元不展示成功率；当前不能提前比较共享策略与尚未训练的单任务策略。<a href="five-task-curves.html">完整曲线与运行状态</a>，本总览为整理时快照。</p></section>'
    page += '<section><h2>已完成结果与未完成工作</h2><div class="scroll"><table><tr><th>实验</th><th>完成范围</th><th>结果及可支持的结论</th><th>证据</th></tr>'
    for exp in registry:
        page += f'<tr><td><strong>{exp["id"]}</strong><br>{esc(exp["name"])}</td><td>{esc(exp["status"])}</td><td>{esc(exp["finding"])}</td><td><a href="{exp["report"]}">详细报告</a><br><a href="../{exp["run"]}/">原始产物</a></td></tr>'
    page += '</table></div></section><section><h2>哪些结论已经成立</h2><p>已建立原始模型的本地基线，完成离线适配、旧PPO与稳定性诊断。E4的单任务和多任务结果相同；E5的退化发生在单任务；E6的早期收益未维持。当前还没有受控证据证明多任务梯度冲突，也没有证明critic波动是成功率波动的原因。</p><p>E7计划使用相同初始化、交互预算与冻结场景比较五个独立策略和一个共享策略。目前共享策略已出现能力保持问题；完成单任务对照后才能判断共享训练的额外影响。</p></section>'
    page += '<section><h2>整理后的入口</h2><ul><li><a href="results-summary.md">文字版总览</a>、<a href="results-summary.csv">逐任务结果CSV</a>、<a href="experiment-index.json">机器可读索引</a>。</li><li><a href="results/">已完成实验报告</a>；<a href="diagnostics/">模型接入与RECAP核对</a>。</li><li><a href="literature/phenomena-paper-audit.html">实验现象与论文核对</a>；<a href="planning/">研究计划与讨论</a>；<a href="汇报.html">历史汇报稿</a>。</li><li><a href="takeaway.md">训练经验</a>；<a href="archive/">历史版本与整理清单</a>；<a href="../outputs/README.md">Hydra启动日志归档</a>。</li></ul><p>训练权重、优化器、轨迹、运行配置和失败记录仍保存在各实验原目录，链接可追溯；本次整理未删除实验数据。</p></section></body></html>'
    write(RESEARCH/'progress.html',page)
    print(RESEARCH/'progress.html')


if __name__ == '__main__':
    main()
