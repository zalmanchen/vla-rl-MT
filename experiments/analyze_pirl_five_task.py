"""Audit raw paired E7 episodes and render a reproducible preliminary analysis."""
import csv
from datetime import datetime
import hashlib
import html
import io
import json
import os
from pathlib import Path
import re
import statistics
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / '.runtime/experiments/e7-five-task-curves'
BASE = ROOT / '.runtime/experiments/e1-reference-clean/clean'
OUT = ROOT / '.research/results/e7'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.runtime/cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(text); tmp.replace(path)


def main():
    evidence = {}
    def read(path):
        data = path.read_bytes()
        evidence[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
        return json.loads(data)
    def lines(path):
        if not path.exists(): return []
        data = path.read_bytes()
        evidence[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
        return [json.loads(s) for s in data.splitlines()]
    plan = read(RUN / 'plan.json')
    state = read(RUN / 'status.json')
    failure = read(RUN / 'failure.json') if (RUN / 'failure.json').exists() else None
    live = json.loads((ROOT / '.research/five-task-curves.json').read_text())
    job = next(x for x in plan['jobs'] if x['kind'] == 'multi')
    folder = RUN / 'jobs' / job['name']
    tasks = plan['tasks']; steps = []
    base = {}
    for task in tasks:
        seeds = plan['reference'][task]['seeds']
        assert len(seeds) == len(set(seeds)) == 20
        base[task] = {seed: read(BASE / task / f'episode_{seed}.json') for seed in seeds}
        assert all(x['status'] == 'completed' for x in base[task].values())
        assert sum(x['success'] for x in base[task].values()) == plan['reference'][task]['successes']
    paired, rows, totals, matched_fields = [], [], [], 0
    for step in plan['evaluation_steps']:
        checkpoint = folder / job['name'] / f'checkpoints/global_step_{step}/actor'
        receipt_path = checkpoint / 'diagnostic-state.json'
        if not receipt_path.exists(): continue
        receipt = read(receipt_path)
        assert receipt['global_step'] == step and receipt['optimizer_steps'] == step * len(tasks)
        assert (checkpoint / 'model_state_dict/full_weights.pt').is_file()
        if not all((folder / f'curve-{step}/clean/{t}/episode_{s}.json').exists()
                   for t in tasks for s in base[t]): continue
        step_rows = []
        for task in tasks:
            cell = folder / f'curve-{step}/clean/{task}'
            contract = read(cell / 'sampling-contract.json')
            assert contract['backend'] == 'reference' and contract['policy_mode'] == 'eval'
            assert contract['seed_protocol'] == 'fixed_eval'
            assert Path(contract['checkpoint']).resolve() == checkpoint.resolve()
            assert Path(contract['config']).resolve() == Path(job['config']).resolve()
            k = wins = losses = 0
            for seed, before in base[task].items():
                after = read(cell / f'episode_{seed}.json')
                assert after['status'] == 'completed'
                for field in ['task', 'regime', 'seed', 'instruction', 'limit']:
                    assert before.get(field) == after.get(field), (step, task, seed, field)
                    matched_fields += 1
                b, a = int(before['success']), int(after['success'])
                k += a; wins += int(a > b); losses += int(a < b)
                paired.append({'step':step, 'task':task, 'seed':seed, 'original_success':b,
                               'shared_success':a, 'difference':a-b})
            original = plan['reference'][task]['successes']
            row = {'step':step, 'task':task, 'episodes':20, 'original_successes':original,
                   'shared_successes':k, 'success_rate_percent':k*5, 'delta_pp':(k-original)*5,
                   'new_wins':wins, 'new_losses':losses, 'retained_original_successes':original-losses,
                   'nominal_actions_per_task':step*1024}
            saved = read(folder / f'curve-{step}/results-{task}.json')
            assert (k,wins,losses) == (saved['successes'],saved['new_wins'],saved['new_losses'])
            assert k-original == wins-losses
            step_rows.append(row)
        steps.append(step); rows.extend(step_rows)
        totals.append({'step':step, 'episodes':100, 'successes':sum(x['shared_successes'] for x in step_rows),
                       'new_wins':sum(x['new_wins'] for x in step_rows), 'new_losses':sum(x['new_losses'] for x in step_rows)})
    assert steps == [5, 10, 20, 50], 'This report interprets the interrupted E7 run; review narrative after new checkpoints.'
    audits = lines(folder / 'batch-audit.jsonl'); phases = lines(folder / 'memory-phases.jsonl')
    interactions = lines(folder / 'online-chunks/interactions.jsonl')
    updated = [x['global_step'] + 1 for x in phases if x['phase'] == 'after_training']
    assert updated == list(range(1, 60))
    assert len(audits) == 59 and [x['global_step'] for x in audits] == list(range(59))
    max_gap = max(x['logprob_gap_max_abs'] for x in audits)
    assert max_gap < .01
    source_checks = {}
    for name, expected in plan['source_sha256'].items():
        source_checks[name] = hashlib.sha256((RUN / 'source' / name).read_bytes()).hexdigest() == expected
    assert all(source_checks.values())
    metrics_text = (folder / 'metrics.log').read_text()
    evidence[str((folder / 'metrics.log').relative_to(ROOT))] = hashlib.sha256(metrics_text.encode()).hexdigest()
    parts = re.split(r'Global Step:\s*(\d+)\s*/\s*\d+', metrics_text)
    metrics = []
    for i in range(1, len(parts), 2):
        values = {'step':int(parts[i])}
        for key in ['success_once','num_trajectories','critic/explained_variance','critic/value_loss',
                    'actor/grad_norm','actor/clip_fraction','actor/approx_kl']:
            match = re.search(r'(?<![\w/])'+re.escape(key)+r'=([-+\d.eE]+)',parts[i+1])
            if match: values[key] = float(match[1])
        metrics.append(values)
    windows = []
    for lo,hi in [(1,10),(11,20),(21,50),(51,59)]:
        selected = [x for x in metrics if lo <= x['step'] <= hi]
        windows.append({'from':lo,'to':hi,'rounds':len(selected),
                        'negative_ev_rounds':sum(x['critic/explained_variance']<0 for x in selected),
                        **{key:statistics.mean(x[key] for x in selected) for key in metrics[0] if key != 'step'}})
    active = read(folder / 'active-training.json')
    logpath = Path(active['log']); log = re.sub(r'\x1b\[[0-9;]*m','',logpath.read_text(errors='replace'))
    evidence[str(logpath.relative_to(ROOT))] = hashlib.sha256(logpath.read_bytes()).hexdigest()
    oom_lines = [x for x in log.splitlines() if any(k in x for k in [
        'ray.exceptions.OutOfMemoryError:', 'OOM kill reason:', 'Total non-selected idle workers:',
        'Total non-selected idle workers USS bytes:', '3068665\t14.24', 'Workers (tasks / actors) killed'])]
    assert any('OutOfMemoryError' in x for x in oom_lines)
    events = lines(RUN / 'events.jsonl')
    watchdog = next(x for x in events if x['kind']=='training_watchdog')
    now = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    training = {'completed_rounds':max(updated),'planned_rounds':100,'optimizer_steps':295,
                'complete_evaluated_steps':steps,'single_task_jobs_started':sum(x['completed_iterations']>0 for x in live['jobs'] if x['kind']=='single'),
                'nominal_actions':sum(x['requested_actions'] for x in interactions),
                'executed_actions':sum(x['executed_actions'] for x in interactions),
                'per_task_interactions':{t:{'nominal':sum(x['requested_actions'] for x in interactions if x['task']==t),
                                          'executed':sum(x['executed_actions'] for x in interactions if x['task']==t)} for t in tasks},
                'wall_hours_to_watchdog':(failure['time_unix']-active['started_unix'])/3600,
                'watchdog_idle_minutes':(watchdog['time_unix']-watchdog['last_activity'])/60,
                'last_checkpoint':50,'updated_but_not_checkpointed_rounds':9}
    findings = [
        '共享策略在第5轮的整体收益仅为+1个百分点，按铃的提升抵消了其他技能的下降。',
        '第50轮为32/100，较原始47/100少15次成功；新增成功13次，同时丢失原有成功28次。',
        '第21—50轮critic解释方差全部为正，但固定开发场景表现继续下降；不能以critic不稳定作为已证实根因。',
        '当前有共享后训练的技能收益不均与能力保持问题；单任务对照尚未执行，不能宣称多任务特有冲突。',
        '日志证实Ray主机内存压力杀掉3个工作进程，随后通信断开；看门狗在无进展约30分钟后停止队列。']
    result = {'updated':now,'status':'interrupted_not_complete','state':state,'failure':failure,
              'protocol':plan['evaluation'],'training':training,'baseline_successes':47,
              'evaluated_rl_episodes':len(paired),'unique_development_scenes':100,
              'paired_fields_verified':matched_fields,'source_snapshot_checks':source_checks,
              'rows':rows,'totals':totals,'training_windows':windows,'training_metrics':metrics,
              'max_preupdate_logprob_gap':max_gap,'audit_threshold':.01,'oom_evidence':oom_lines,
              'findings':findings,'limits':['One training seed; 20 reused development scenes per task.',
                'The 400 checkpoint evaluations reuse 100 scenes; they are not 400 independent scenes.',
                'No matched single-task curve exists. E5/E6 differ in configuration or budget.',
                'Training critic metrics use changing rollout distributions, not fixed evaluation scenes.',
                'Mean and paired transitions are descriptive; no causal or statistical significance claim.',
                'Training interruption and policy performance regression are different observations.'],
              'evidence_sha256':evidence}
    write(OUT/'e7-analysis.json',json.dumps(result,ensure_ascii=False,indent=2))
    for name,records in [('e7-analysis.csv',rows),('e7-paired-scenes.csv',paired)]:
        stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        write(OUT/name,stream.getvalue())
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(14,10))
    ax=axes[0,0];xs=[0,*steps];ys=[47,*[x['successes'] for x in totals]]
    ax.plot(xs,ys,'o-',color='#d16b26',linewidth=2,label='Shared policy');ax.axhline(47,color='#777',ls=':',label='Original Motus')
    for x,y in zip(xs,ys):ax.annotate(str(y),(x,y),xytext=(0,8),textcoords='offset points',ha='center')
    ax.set(xlabel='Training rounds',ylabel='Five-task mean success (%)',title='A. Shared policy: early gain is not sustained',ylim=(0,65));ax.legend();ax.grid(alpha=.2)
    ax=axes[0,1];heat=np.array([[next(r['delta_pp'] for r in rows if r['task']==t and r['step']==s) for s in steps] for t in tasks])
    im=ax.imshow(heat,cmap='RdBu',vmin=-50,vmax=50,aspect='auto');ax.set_xticks(range(4),steps);ax.set_yticks(range(5),['Bottle','Bell','Stack blocks','Empty cup','Container / plate'])
    for i in range(5):
        for j in range(4):ax.text(j,i,f'{heat[i,j]:+d}',ha='center',va='center',color='white' if abs(heat[i,j])>=35 else '#162b3b')
    ax.set(xlabel='Training rounds',title='B. Per-task change from original (percentage points)');fig.colorbar(im,ax=ax,fraction=.035,pad=.025)
    ax=axes[1,0];x=np.arange(4)
    ax.bar(x-.18,[a['new_wins'] for a in totals],.36,color='#2c8c6c',label='New successes')
    ax.bar(x+.18,[a['new_losses'] for a in totals],.36,color='#bb4d4d',label='Lost original successes')
    ax.set_xticks(x,steps);ax.set(xlabel='Training rounds',ylabel='Paired scenes (of 100)',title='C. Acquiring and losing skills occur together');ax.legend();ax.grid(axis='y',alpha=.2)
    ax=axes[1,1];ax.plot([x['step'] for x in metrics],[x['critic/explained_variance'] for x in metrics],color='#2462a3');ax.axhline(0,color='#777',ls=':')
    ax.set(xlabel='Training rounds',ylabel='Critic explained variance',title='D. Better critic diagnostic does not establish policy gain');ax.grid(alpha=.2)
    fig.suptitle('E7 interrupted at round 59 | 1 training seed | 20 fixed scenes/task | single-task runs absent',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.97))
    for ext in ['png','svg']:fig.savefig(OUT/f'e7-analysis.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    names={'adjust_bottle':'调整瓶子','click_bell':'按铃','stack_blocks_two':'叠两块积木','place_empty_cup':'放空杯','place_container_plate':'容器放盘'}
    page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>E7 五任务后训练结果分析</title><style>body{max-width:1200px;margin:32px auto;padding:0 24px;font:19px/1.8 system-ui;background:#f3f6f8;color:#193245}section{background:white;padding:24px;margin:22px 0;border-radius:12px}h1{font-size:34px}h2{font-size:25px}table{border-collapse:collapse;width:100%;font-size:19px}th,td{padding:12px;text-align:left;border-bottom:1px solid #d8e1e6}.scroll{overflow:auto}a{color:#126b98}img{width:100%}.notice{border-left:5px solid #b65139}code{overflow-wrap:anywhere}li{margin:10px 0}</style></head><body><h1>E7：多技能收益与保持的阶段分析</h1>'''
    page+=f'<p>核查时间：{html.escape(now)}（北京时间）。从逐场景原始记录重新统计。</p>'
    page+='<section class="notice"><h2>实验中断，现有结果有效但对照未齐</h2><p>共享策略完成59/100轮，完整保存并评估了第5、10、20、50轮。五个单任务策略均未开始，第100轮没有成绩。训练日志记录Ray主机内存压力杀掉3个工作进程，之后通信断开，约30分钟无进展后看门狗停止队列。</p><p>第50轮是最后一个完整评估检查点，不能称为100轮最终成绩。中断前成功率已随训练下降，不能把这一退化归因于中断。</p></section>'
    page+='<section><h2>1. 整体改善没有保持</h2><p>原始模型为已经多任务SFT的Motus π0.5。当前设置为Flow-SDE＋PPO、严格FP32、每批1遍更新、冻结VLM、无critic预热。所有列均使用同一组每任务20个冻结Clean场景。</p><div class="scroll"><table><tr><th>任务</th><th>原始</th>'+''.join(f'<th>第{s}轮</th>' for s in steps)+'<th>第50轮变化</th></tr>'
    for t in tasks:
        rr=[next(r for r in rows if r['task']==t and r['step']==s) for s in steps]
        page+=f'<tr><td>{names[t]}</td><td>{plan["reference"][t]["successes"]*5}%</td>'+''.join(f'<td>{r["success_rate_percent"]}%</td>' for r in rr)+f'<td>{rr[-1]["delta_pp"]:+d}个百分点</td></tr>'
    page+='<tr><th>五任务均值</th><th>47%</th>'+''.join(f'<th>{x["successes"]}%</th>' for x in totals)+'<th>−15个百分点</th></tr></table></div><p>每个任务一次成功相差5个百分点。五任务各20场景，宏平均与合计成功数/100相同。400条检查点评估复用了100个场景，不是400个独立场景。</p><img src="e7-analysis.svg" alt="共享策略成功率、各任务变化、配对得失与critic趋势"><p><a href="e7-analysis.png">PNG图</a> · <a href="e7-analysis.svg">SVG图</a></p></section>'
    page+='<section><h2>2. 平均值掩盖了同时发生的收益与损失</h2><p>第5轮按铃从40%升至90%，贡献+10次成功；其余四个任务合计少9次成功，因此均值只从47%变为48%。这不能称为五个技能共同改善。</p><div class="scroll"><table><tr><th>检查点</th><th>原来失败、现在成功</th><th>原来成功、现在失败</th><th>净变化</th></tr>'
    for x in totals:page+=f'<tr><td>第{x["step"]}轮</td><td>{x["new_wins"]}</td><td>{x["new_losses"]}</td><td>{x["new_wins"]-x["new_losses"]:+d}</td></tr>'
    page+='</table></div><p>第50轮丢失了原始47次成功中的28次，只保留19次；另外获得13次新成功，合计32次。叠块在第20轮为0/20；容器放盘从70%降到30%。当前值得研究的问题是如何获得新收益，同时保住原来能完成的技能。</p></section>'
    page+='<section><h2>3. critic与数值诊断能说明什么</h2><table><tr><th>训练轮次</th><th>critic解释方差均值</th><th>负解释方差轮数</th><th>价值损失均值</th></tr>'
    for w in windows:page+=f'<tr><td>{w["from"]}—{w["to"]}</td><td>{w["critic/explained_variance"]:.3f}</td><td>{w["negative_ev_rounds"]}/{w["rounds"]}</td><td>{w["critic/value_loss"]:.4f}</td></tr>'
    page+=f'</table><p>59个训练批次的更新前概率一致性审计全部通过，最大误差{max_gap:.3g}，低于0.01。该检查排除了已检测到的概率重算偏差，不能证明整个算法实现或更新方向均正确。</p><p>第21—50轮critic解释方差全部为正，均值0.327；这一时期开发场景均值仍从41%降到32%。解释方差改善与任务表现下降同时存在。因此，“再训练更久让critic稳定就会涨分”没有得到本轮支持。价值统计基于持续变化的训练分布，也不能直接等同于独立评估上的价值质量。</p><p>下一步应区分通用的策略漂移、训练／评估分布差异、奖励与任务成功的关系，以及多任务共享更新的影响。梯度冲突和critic因果作用都还没有直接证据。</p></section>'
    page+=f'<section><h2>4. 完成范围与工程中断</h2><p>已完成59轮、295次优化器更新；采集{training["nominal_actions"]:,}个名义动作点，实际执行{training["executed_actions"]:,}个。每任务名义交互量相同，均为60,416；第50轮用于对比的预算为每任务51,200。</p><p>日志中的内存终止阈值为125.41 GiB限额的95%；实际用量128,091,848,704字节越过127,928,451,072字节阈值。并发评估进程约14.24 GiB，383个空闲Ray工作进程的USS约6.82 GiB；这些是占用来源线索，尚不能独立确定哪项贡献了峰值。</p><p>原调度只对GPU显存设置余量，没有同步约束主机／容器内存。恢复前需要补齐内存监控、限制空闲工作进程，并在Ray致命错误后及时终止等待。不要通过关闭OOM保护掩盖问题。</p><p>最新完整检查点为第50轮；第51—59轮更新没有完整保存。如果后续从50轮恢复，环境与随机数连续性不能默认成立，重复采样和恢复成本要另记。</p></section>'
    page+='<section><h2>5. 下一步按什么顺序做</h2><ol><li><strong>先恢复可运行性。</strong>修复主机内存并发门槛和错误处理，再继续原冻结实验；保留本次中断记录。</li><li><strong>先补齐同预算单任务参照。</strong>重点先看按铃、叠块、容器放盘，随后补齐另外两项；比较相同5/10/20/50轮，先回答共享是否比独立训练更容易丢失能力。</li><li><strong>根据对照再选择方法。</strong>若单任务也下降，先查通用后训练稳定性；若单任务改善而共享下降，再做逐任务梯度与共享更新诊断。</li><li><strong>再检验联合SFT＋RL。</strong>以当前RL为参照，加入原始示范重放或策略保持约束，记录收益与保留率；控制交互、数据来源和更新预算。它目前是待验证假设，不能用现有曲线直接证明有效。</li></ol><p>暂不把任务数扩到10—50，也不只挑第5轮作为最终成绩。1个训练种子、20个重复使用的开发场景用于定位问题；论文阶段需要独立场景与多个种子确认。</p></section>'
    page+='<section><h2>6. 与此前实验的关系</h2><p>E4短预算RECAP：ST和MT均29/100，没有总体MT低于ST的证据。E5旧PPO：瓶子单任务也出现明显退化。E6稳定配置：瓶子在第5轮11/20，第10轮回到9/20。本轮E7补充的是共享策略的长期技能保持诊断，尚未完成ST／MT比较。不同配置和预算的实验不能拼成受控排名。</p></section>'
    page+='<p><a href="e7-analysis.json">完整分析与证据哈希</a> · <a href="e7-analysis.csv">逐任务结果CSV</a> · <a href="e7-paired-scenes.csv">400条配对场景CSV</a> · <a href="../../five-task-curves.html">完整曲线</a> · <a href="../../progress.html">实验总览</a> · <a href="../../../.runtime/experiments/e7-five-task-curves/failure.json">中断记录</a> · <a href="../../../.runtime/experiments/e7-five-task-curves/plan.json">冻结计划</a></p></body></html>'
    write(OUT/'e7-analysis.html',page)
    print(json.dumps({'analysis':str(OUT/'e7-analysis.html'),'totals':totals,'training':training,'max_logprob_gap':max_gap,'evidence_files':len(evidence)},ensure_ascii=False))


if __name__ == '__main__':
    main()
