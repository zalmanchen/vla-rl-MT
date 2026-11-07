"""Render E5 evidence and queue state without treating planned jobs as results."""
from research_artifacts import artifact_path, relocate_html
import html,json,re
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'.runtime/experiments/e5-pirl';OUT=ROOT/'.research'
def read(p,default):return json.loads(p.read_text()) if p.exists() else default
def main():
 if read(RUN/'pilot-plan.json',{}).get('active'):
  from render_pirl_pilot import render
  render();return
 plan=read(RUN/'plan.json',{});state=read(RUN/'status.json',{'phase':'验收与准备'})
 rows=[]
 for j in plan.get('jobs',[]):
  p=RUN/'jobs'/j['name'];s=read(p/'status.json',{})
  steps=[]
  for log in p.glob('training-micro*.log'):
   steps+=re.findall(r'Global Step:\s*(\d+)\s*/\s*(\d+)',log.read_text(errors='replace'))
  actual=max((int(a) for a,b in steps),default=0)
  rows.append('<tr><td>'+html.escape(j['name'])+'</td><td>'+html.escape(s.get('phase','planned'))+'</td><td>'+str(actual)+' / '+str(j['rollout_iterations'])+'</td></tr>')
 zero_rows=[]
 for task in plan.get('tasks',[]):
  records=[read(p,{}) for p in (RUN/'zero-update-evaluation/clean'/task).glob('episode_*.json')]
  records=[x for x in records if x.get('status')=='completed']
  zero_rows.append(f'<tr><td>{html.escape(task)}</td><td>{len(records)}/20</td><td>{sum(bool(x.get("success")) for x in records)}</td></tr>')
 result_rows=[]
 for j in plan.get('jobs',[]):
  for mode,sub in [('PPO','evaluation'),('成功重放','success-replay/evaluation')]:
   results=read(RUN/'jobs'/j['name']/sub/'results.json',[])
   if results:
    result_rows.append(f'<tr><td>{html.escape(j["name"])} {mode}</td><td>{len(results)}</td><td>{sum(x["successes"] for x in results)}/{sum(x["episodes"] for x in results)}</td></tr>')
 audit=read(RUN/'zero-update-audit.json',{});env=read(RUN/'multitask-env-smoke.json',{})
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>πRL 全面实验进展</title><style>body{max-width:1100px;margin:35px auto;padding:0 24px;background:#f5f7f9;color:#172b3a;font:18px/1.8 system-ui}section{padding:24px;background:white;margin:20px 0;border-radius:12px}table{width:100%;border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}a{color:#087e80}</style><h1>πRL Flow-SDE＋PPO 实验</h1>'''
 page+=f'<p>更新：{datetime.now(timezone.utc).isoformat()}（UTC）</p><section><h2>当前状态</h2><p>{html.escape(state.get("phase","准备"))}</p><p>{html.escape(state.get("detail",""))}</p><p>零更新数值验收：{audit.get("status","进行中／待完成")}；五任务环境验收：{env.get("status","进行中／待完成")}。终止奖励／路点掩码已通过独立提前成功集成检查。</p></section>'
 page+='<section><h2>新入口零更新闭环诊断（进行中不作最终比较）</h2><table><tr><th>任务</th><th>完成场景</th><th>成功数</th></tr>'+''.join(zero_rows)+'</table></section><section><h2>正式训练后结果</h2><table><tr><th>策略</th><th>已评估任务</th><th>成功／场景</th></tr>'+(''.join(result_rows) or '<tr><td colspan="3">暂无正式策略评估结果</td></tr>')+'</table></section><section><h2>实测基准</h2><p>原始多任务 SFT：22任务，566/2200，25.73%。E4 五任务相同100场景：原始47%、单任务RECAP 29%、共享RECAP 29%、成功重放25%。正式训练后权重转回原始OpenPI布局，用E1同一参考采样器评估；811个参与动作生成的张量已通过逐张量无损往返验收，未使用的动作专家LM输出头保留原始值。新入口零更新闭环结果单独记为接入诊断。新算法的训练结果尚需闭环评估。</p></section><section><h2>冻结实验矩阵</h2><p>5 个独立单任务策略与 1 个五任务共享策略，各3个训练随机种子。优先调整瓶子、叠块、容器放盘，再补齐按铃和空杯。每任务2个并行环境，每轮每环境512动作步，100轮；5任务共享每轮各任务相同预算。共享100轮与五个独立策略各100轮的合计交互量匹配。</p><p>固定32步预测／执行块、10步去噪，FP32初始权重；冻结VLM，更新动作专家与价值头。正式结果使用冻结的每任务100个测试场景；最终权重扩评至22任务，记录训练任务收益与未训练任务保持。中间检查点只作预先规定的曲线报告，不根据测试场景选择超参数。</p><p>已接入同经验成功重放对照，使用各PPO作业记录的成功轨迹，匹配动作专家更新步数；其训练尚未启动。长程连续子任务仍需单独接通。独立重置任务不记为长程整链实验。已核查双瓶拾取与投放场景不兼容；三块堆叠可作为连续子目标候选，但其元信息函数会执行专家动作，接入前必须替换为纯读取实现。<a href="../.runtime/experiments/e5-pirl/long-horizon-feasibility.json">长程场景核查</a>。</p><table><tr><th>作业</th><th>状态</th><th>已完成／计划采集轮数</th></tr>'+''.join(rows)+'</table></section><p><a href="../.runtime/experiments/e5-pirl/status.json">队列状态</a> · <a href="../.runtime/experiments/e5-pirl/plan.json">实验计划</a> · <a href="e4-analysis.html">上一轮实测分析</a> · <a href="progress.html">总记录</a></p></html>'
 (artifact_path('pirl-progress.html')).write_text(relocate_html(page, 'pirl-progress.html'))
if __name__=='__main__':main()
