"""Render the local research record from audited evaluation summaries."""

import html
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".research"


def e(value):
    return html.escape(str(value))


def pct(value):
    return "—" if value is None else f"{value * 100:.1f}%"


def main():
    d = json.loads((OUT / "e1-results.json").read_text())
    c = d["contract"]
    rows = d["tasks"]
    complete = d["evaluation_complete"]
    resuming = d.get("resume_running", False)
    status = (
        "已完成：22 个任务各 100 次"
        if complete
        else ("正在补齐未完成任务" if resuming else "队列已结束，评测部分完成")
    )
    reasons = {
        "Missing ASSETS_PATH environment variable": "资产路径环境变量缺失",
        "Unstable object during scene initialization": "场景初始化时物体不稳定",
        "Watchdog timeout; root cause not established": "工作进程心跳超时，根因待查",
        "See worker log": "查看运行日志",
    }
    table = []
    for r in sorted(
        rows,
        key=lambda x: (
            x["status"] != "complete",
            -(x["success_rate"] if x["success_rate"] is not None else -1),
        ),
    ):
        task = r["task"]
        tag = (
            "完整"
            if r["status"] == "complete"
            else ("未完成" if r["completed"] else "尚无有效 episode")
        )
        ci = (
            "—"
            if not r["completed"]
            else f"[{r['wilson95_low'] * 100:.1f}, {r['wilson95_high'] * 100:.1f}]%"
        )
        note = reasons.get(r["infrastructure_error"], r["infrastructure_error"])
        replacements = r.get("replacement_seeds", [])
        if replacements:
            note += ("；" if note else "") + f"补充官方种子 {len(replacements)} 个"
        table.append(
            f'<tr data-status="{e(r["status"])}" data-task="{e(task)}" data-rate="{r["success_rate"] if r["success_rate"] is not None else -1}"><td><a href="../.runtime/experiments/e1-reference-clean/clean/{e(task)}/summary.json">{e(task)}</a></td><td>{r["completed"]}/100</td><td>{r["successes"]}/{r["completed"]}</td><td><strong>{pct(r["success_rate"])}</strong></td><td>{ci}</td><td><span class="pill {"ok" if tag == "完整" else "warn"}">{tag}</span></td><td>{e(note)} <a href="../.runtime/experiments/e1-reference-clean/logs/clean-{e(task)}.log">日志</a></td></tr>'
        )
    issues = []
    for r in rows:
        if r["status"] == "complete":
            continue
        issues.append(
            f"<li><strong>{e(r['task'])}</strong>：{r['completed']}/100；{e(reasons.get(r['infrastructure_error'], r['infrastructure_error']))}。</li>"
        )
    p = c["parity"]
    updated = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S %Z")
    if complete:
        insight = f"22 个任务已各完成 100 次有效评测，宏平均成功率为 {d['complete_subset_macro_success'] * 100:.2f}%。发生初始化失败的种子及替补关系单独保存；后续 RL 对比需使用最终冻结种子清单。"
    else:
        insight = f"目前只有 {d['complete_tasks']} 个任务跑满 100 次；这些任务的宏平均成功率为 {d['complete_subset_macro_success'] * 100:.2f}%（{d['complete_subset_successes']}/{d['complete_subset_episodes']}）。该数值不是完整 22 任务平均，也不是 50 任务成绩。"
    e7_latest = ""
    if (OUT / "five-task-curves.json").exists():
        e7 = json.loads((OUT / "five-task-curves.json").read_text())
        e7_latest = ('<section><h2>正在运行：五任务单任务／共享RL曲线</h2>'
                     '<p>五个单任务策略与一个五任务共享策略，从相同Motus权重独立开始；统一严格FP32与每批1遍PPO。每个策略100轮，在0／5／10／20／50／100轮用相同开发场景评估。五个单任务合计与共享策略的名义交互预算均为512000。</p>'
                     f'<p>最新快照：{e(e7["updated"])}；{e(e7["state"].get("phase", "准备"))}。'
                     '<a href="five-task-curves.html">查看逐任务曲线、宏平均、critic与GPU进度</a>。曲线页自动刷新，本总览为生成时快照。</p></section>')
    e6_path = OUT / "stability-progress.json"
    e6_latest = ""
    if e6_path.exists():
        e6 = json.loads(e6_path.read_text())
        e6_rows = []
        for result in e6.get("paired_results", []):
            n = result["completed"]
            score = f'{result["successes"]}/{n}' if n else "待评估"
            e6_rows.append(f'<tr><td>{e(result["name"])}</td><td>{n}/20</td><td>{score}</td></tr>')
        e6_latest = ('<section><h2>最新：E6短训练与配对评估</h2>'
                     f'<p>结果快照：{e(e6["updated"])}。{e(e6["state"].get("phase", ""))}。</p>'
                     '<table><tr><th>策略</th><th>评估进度</th><th>成功数</th></tr>'
                     + "".join(e6_rows)
                     + '</table><p>当前严格FP32＋CPU分桶配置已完成10轮训练，10个真实批次均通过概率审计。成功率仍按单任务、单训练种子的小样本诊断解释；更新次数与计算精度同时变化，尚不能单独归因。'
                     '第10轮评估曾出现心跳停滞，已保留原结果并按相同种子恢复。'
                     '<a href="stability-progress.html">实时状态与逐场景配对变化</a> · '
                     '<a href="takeaway.md">训练经验记录</a>。</p>'
                     '<p>下方原始模型卡片与E1表格属于历史基线，不表示当前尚未训练。</p></section>')
    template = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VLA 多任务 RL · 实验进展</title>
<style>
:root{--ink:#172331;--muted:#596b7d;--line:#dce5ed;--blue:#245c9b;--amber:#9b6412;--bg:#f3f6fa}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.75 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}main{max-width:1240px;margin:auto;padding:38px 28px 70px}a{color:var(--blue);text-underline-offset:3px}header .eyebrow{letter-spacing:.1em;color:var(--muted);font-size:12px}h1{font-size:30px;line-height:1.3;margin:8px 0 12px}h2{font-size:21px;margin:0 0 12px}h3{font-size:16px;margin:18px 0 8px}p{margin:8px 0}code{background:#edf2f7;border-radius:4px;padding:2px 5px;font-size:12px;overflow-wrap:anywhere}.meta{color:var(--muted);font-size:13px}.pill{display:inline-block;padding:2px 9px;border-radius:5px;font-size:12px;white-space:nowrap}.warn{background:#fff0d7;color:#865608}.ok{background:#e3f1e9;color:#285a3a}.blue{background:#e5eef9;color:#245c9b}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}.card,section{background:white;border:1px solid var(--line);border-radius:10px;padding:22px}.card small{color:var(--muted)}.value{font-size:29px;font-weight:650;line-height:1.5}section{margin:18px 0}.callout{border-left:4px solid #c18a35;background:#fff9ef;padding:13px 17px;margin:14px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}.steps{padding-left:20px}.steps li{margin:9px 0}.figure img{display:block;max-width:100%;height:auto;margin:auto}.figure{margin:14px 0}.caption{font-size:13px;color:var(--muted);margin-top:10px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;border-bottom:1px solid var(--line);padding:11px 10px;vertical-align:top}th{background:#f4f7fa;white-space:nowrap}tbody tr:hover{background:#f8fafc}td:first-child{white-space:nowrap}button,input{border:1px solid #bbcbd9;background:white;border-radius:5px;padding:8px 11px;color:var(--ink);font:inherit;font-size:13px}button{cursor:pointer}button.active{background:#e5eef9;border-color:#7fa4cc}.tools{display:flex;gap:8px;flex-wrap:wrap;margin:15px 0}input{flex:1;min-width:200px}details{margin-top:12px}summary{cursor:pointer;color:var(--blue)}ul{padding-left:21px}.nowrap{white-space:nowrap}.footer{font-size:12px;color:var(--muted);margin-top:24px}@media(max-width:800px){main{padding:22px 14px}.cards{grid-template-columns:1fr 1fr}.grid{grid-template-columns:1fr}section{padding:17px}}@media print{body{background:white}main{max-width:none;padding:0}.tools{display:none}section,.card{break-inside:avoid}.scroll{overflow:visible}th,td{padding:6px;font-size:10px}}
</style></head><body><main>
<header><div class="eyebrow">VLA-RL-MT / RESEARCH LOG</div><h1>多任务 π0.5：前置实验与 baseline</h1><p><span class="pill warn">@@STATUS@@</span> <span class="pill blue">只评测 · 权重未更新</span></p><p class="meta">更新：@@UPDATED@@（Asia/Shanghai） · 首次启动至最终补齐：@@START@@ 至 @@END@@ · 首轮队列耗时 @@HOURS@@ 小时；首尾跨度含停顿，不代表连续运行耗时</p></header>
@@E7LATEST@@
@@E6LATEST@@
<div class="cards"><div class="card"><small>有效 episode 完成度</small><div class="value">@@N@@ / 2,200</div><small>@@COVERAGE@@ · 不含初始化错误与超时中断</small></div><div class="card"><small>跑满 100 次的任务</small><div class="value">@@COMPLETE@@ / 22</div><small>当前仅 Clean；尚未覆盖全部 50 任务</small></div><div class="card"><small>完整任务子集宏平均成功率</small><div class="value">@@MACRO@@</div><small>@@K@@ / @@FULLN@@；分母只包含完整任务</small></div><div class="card"><small>训练与策略更新</small><div class="value">0 次</div><small>未新增 SFT；尚未启动 RL 或方法复现训练</small></div></div>
<section><h2>新增：E6稳定性诊断与联合后训练方向</h2><p>无更新SDE已完成20场景，成功9次，与原始参考ODE的9/20相同。critic-only已完成5轮及另1轮批次核验，补跑批次确认策略冻结，概率对数最大重算差0.00475，小于预设0.01。10轮单次更新PPO及第5／10轮配对评估的状态见实时页面。PPO曾完成4轮后被概率一致性保护停止；固定权重数值对照确认默认TF32带来的敏感性，禁用后批大小变化导致的比率差由2.63%降至0.0122%。已据此修复采样和训练精度，保留原停止门槛。<a href="stability-progress.html">查看实时进度</a>。已结合灵巧实机资源，整理<a href="unified-posttraining-assessment.html">联合SFT＋RL的研究价值、先例与Science Robotics路线</a>。</p></section><section><h2>新增：实验现象与原论文核对</h2><p>已核对在线退化、离线RECAP、成功重放、执行窗口、跨论文差距等9类现象。πRL报告过早期下降，但本地尚无恢复证据；先检查采样规模、critic与更新稳定性。<a href="phenomena-paper-audit.html">查看原文位置、证据边界与下一步验证</a>。</p></section><section><h2>新增：πRL 在线 RL 实验</h2><p>已完成快速前置诊断：同20场景，原始9/20，单任务PPO第10轮7/20、第20轮2/20、第100轮1/20。退化在前20轮已出现，先排查单任务训练稳定性；完整三种子矩阵暂不扩跑。稳定后再做3任务、1种子的小矩阵。<a href="pirl-progress.html">查看实时进展</a>。</p></section><section><h2>新增：Offline RL 批量实验</h2><p>已完成五任务固定经验采集：250 条示范 + 200 条自主成功／失败轨迹，共 450 条、119294 个观测动作记录。采用 RLinf RECAP 的官方 Gemma3-270M + SigLIP2 价值模型；首轮比较五个单任务策略、五任务共享策略与成功经验重放。</p><p><a href="offline-rl.html">查看离线 RL 实时进展与结果</a> · <a href="offline-rl-pipeline-status.json">队列状态 JSON</a>。下方 E1 是历史 SFT checkpoint 评估，不代表新增 RL 作业的训练状态。</p></section><section><h2>本次验证什么</h2><p>直接使用已经在 RoboTwin 50 任务上联合微调的 <a href="https://huggingface.co/motus-robotics/pi0.5_robotwin2">motus-robotics/pi0.5_robotwin2</a>。本次是 RL 前的能力参照与推理接入检查，不是重新做 SFT，也不是多任务 RL 算法的性能比较。</p><div class="grid"><div><h3>E0 · 推理实现对齐</h3><p><span class="pill warn">尚未通过预设数值阈值</span></p><p>固定合成三相机输入、相同权重与初始噪声下，原始 OpenPI 与 RLinf 原生采样器的最大绝对差异为 <strong>@@MAXABS@@</strong>，相对 L2 差异为 <strong>@@RELL2@@</strong>。预设阈值均为 0.01。</p><p class="meta">GELU 实现差异已做受控检查，但没有解释全部误差。该结果只涉及一次合成输入数值检查，不能直接解释闭环成功率。</p></div><div><h3>E1 · 原始 checkpoint baseline</h3><p><span class="pill blue">原始 OpenPI PyTorch 采样器</span></p><p>为避免将未对齐的原生实现混入参照，E1 使用独立参考推理实现；图像、状态与动作处理复用 RLinf ALOHA 变换，指定 checkpoint 自带归一化统计。</p><p>@@INSIGHT@@</p></div></div></section>
<section><h2>结果与解释</h2><p>最终成功 566/2200（25.73%）。各任务样本量相同，任务宏平均与回合成功比例数值相同。任务成功率中位数为 19.5%；7 个任务低于 10%，9 个在 10% 至不足 40%，6 个达到 40%。</p><p>完整任务中，shake_bottle（75%）、<strong>stack_bowls_two（64%）</strong>、<strong>place_container_plate（61%）</strong>相对较高；adjust_bottle 为 47%，click_bell 为 41%，stack_blocks_two 为 40%。handover_block 为 0/100，lift_pot 为 2/100，pick_dual_bottles 为 4/100，显示任务间能力差异明显。</p><p>这些差异说明存在不同程度的性能提升空间，但<strong>不能据此证明多任务负迁移、梯度冲突、价值估计错误或 RL 必然有效</strong>：本次没有任何 RL 更新，也没有单任务／联合 RL 对照。全部 1634 次有效失败均达到任务步数上限；这是终止方式，不能据此判断根因。零成功任务需先检查失败阶段与接入一致性。</p><div class="callout">@@CAUTION@@</div><figure class="figure"><img src="figures/e1-success-rates.svg" alt="全部有有效结果任务的成功率及95% Wilson置信区间"><figcaption class="caption">蓝色：跑满 100 次；橙色 *：尚未跑满，只能描述当前样本。误差条为 95% Wilson 区间。没有有效 episode 的任务不绘制成功率。<a href="figures/e1-success-rates.png">PNG</a> · <a href="figures/e1-success-rates.svg">SVG</a></figcaption></figure></section>
<section><h2>逐任务记录</h2><div class="tools"><button class="active" data-filter="all">全部任务</button><button data-filter="complete">完整任务</button><button data-filter="incomplete">未完成任务</button><button id="sort-rate">按成功率排序</button><input id="search" type="search" placeholder="搜索任务名称" aria-label="搜索任务名称"></div><div class="scroll"><table id="results"><thead><tr><th>任务</th><th>进度</th><th>成功 / 有效次数</th><th>成功率</th><th>95% Wilson 区间</th><th>状态</th><th>说明与证据</th></tr></thead><tbody>@@ROWS@@</tbody></table></div><p class="caption">运行错误不记为策略失败；没有完成 100 次的任务不进入“完整任务子集宏平均”。每个有效失败 episode 均达到相应任务步数上限；未发现重复种子或越出官方候选种子的记录。</p></section>
<section><h2>中断原因与补跑记录</h2><ul>@@ISSUES@@</ul><p>补跑保留原始结果、源代码快照和原始协议。资产根目录需明确设置；不稳定种子先按相同种子重试 3 次，始终无法初始化时单独登记无效场景，并从同一官方候选列表顺序补足。只替补<strong>策略执行前的无效初始化</strong>，不替换策略失败。</p><p>最终每任务种子清单已冻结，供 RL 前后配对评测复用。策略执行中的超时另行诊断与重试，不通过丢弃困难 episode 提高成功率。</p><p class="caption">早期日志将初始化阶段标为 expert_validation；当前 fixed_eval 协议实际未重新执行专家轨迹，相关异常发生在 setup_demo。</p><details><summary>原始与补跑协议证据</summary><ul><li><a href="../.runtime/experiments/e1-reference-clean/contract.json">首轮冻结协议</a></li><li><a href="../.runtime/experiments/e1-reference-clean/jobs.jsonl">首轮任务退出状态</a></li><li><a href="../.runtime/experiments/e1-reference-clean/resume-amendment.json">补跑协议修订记录</a></li><li><a href="../.runtime/experiments/e1-reference-clean/seed_manifest.json">最终有效种子清单</a></li></ul></details></section>
<section><h2>代表性轨迹</h2><figure class="figure"><img src="figures/e1-rollout-contact-sheet.jpg" alt="handover_block、lift_pot、adjust_bottle和stack_bowls_two的起始、中间、终止帧"><figcaption class="caption">按固定规则选择各任务首个满足相应成功／失败标签的已保存视频，依次展示起始、中间、结束帧。仅作为定性样例，不代表整体失败原因分布。</figcaption></figure><p>handover_block 样例中可以看到双臂与红块发生交互，但结束时未完成目标判定；lift_pot 样例未完成抬锅；adjust_bottle 和 stack_bowls_two 样例分别完成扶正与叠放。进一步区分抓取、交接、放置等失败阶段，需要额外的轨迹标注。</p></section>
<section><h2>论文报告值与新增单任务对照</h2><p>Motus 论文 Table 14 的 π0.5 Clean 全 50 任务均值为 42.98%；取与本轮相同的 22 任务后，论文报告值均值为 46.36%，本地实测为 25.73%。这是外部报告值与本地结果的描述性对照；场景、执行窗口等协议未对齐，差距不能归因于多任务训练。</p><p>典型任务（论文报告／本地）：adjust_bottle 79%／47%，click_bell 75%／41%，stack_blocks_two 48%／40%，handover_block 18%／0%。全部逐任务数据见 <a href="motus-paper-reference.json">论文参考对照 JSON</a>；来源为 <a href="https://arxiv.org/html/2512.13030v2">Motus 论文 Table 14</a>。</p><p>新增 PACE 单任务 checkpoint 对照已授权并启动准备：adjust_bottle、click_bell、stack_blocks_two、handover_block；各模型保留训练时 50 步预测窗口与各自归一化，统一执行前 32 步，使用本轮冻结场景。四任务权重约 49.77 GB，下载并逐文件校验后先做单回合验收，再每任务评估 100 回合。当前结果与阶段见 <a href="pace-comparison.json">实时队列状态 JSON</a>；尚未产生结果的任务不填成功率。该比较是不同已发布 SFT checkpoint 的外部对照，不是单任务 RL，也不是受控训练消融。</p></section>
<section><h2>后续候选工作 · 待确定任务</h2><p>E1 已结束；新训练与论文机制实验仍待确定。用户随后授权了 PACE 四任务 checkpoint 下载及配对评估，队列见上方记录。研究范围倾向 offline RL。</p><ol class="steps"><li><strong>确认评估可信度。</strong> 将 E0 扩展到已保存的真实观测，检查权重、归一化、图像与动作处理，并核对当前 32 步全执行、无 OIDN 等设置；shake_bottle 的高初始化排除率需单独检查。</li><li><strong>标注已有视频的失败阶段。</strong> 从中等成功率与低成功率任务分层抽样，区分抓取、交接、放置、恢复等失败。132 段视频是抽样记录，不能直接代表全部回合的失败比例。</li><li><strong>检查离线数据是否支持问题。</strong> 现有输出是 2200 份回合统计、132 段头相机视频与 132 份初始观测快照，没有完整逐步观测、动作、奖励和下一观测，不能直接作为 offline RL 经验池。未来采集须使用独立训练种子，不复用本批测试种子。</li><li><strong>再确定机制实验。</strong> 中等成功率任务可用于策略改善诊断，高成功率任务可作为能力保留参照；长程验证需要同场景、状态连续的任务链。当前独立重置评估不构成长程结果；方法、任务与数据预算尚未确定。</li></ol></section>
<section><h2>可复现范围与限制</h2><div class="grid"><div><p><strong>模型：</strong><code>@@MODELREV@@</code></p><p><strong>设置：</strong>Clean · ALOHA-AgileX 双臂 · FP32 · 32 步动作窗口全部执行 · 10 次去噪 · 官方 seen 语言模板 · 2 个并行工作进程。</p><p><strong>渲染：</strong>ray tracing，OIDN 关闭；规划器构建为 mplib；固定种子评测不重新执行专家轨迹。</p></div><div><p>该协议不直接等同于官方 leaderboard。当前结果不能用于声称模型在全部 50 任务上的平均能力，也不能作为灵巧手、多任务 RL、LoRA、PCGrad 或奖励模型方法的复现结果。</p><p>原始工作代码与首轮协议哈希匹配：<strong>@@AUDIT@@</strong>。模型权重未更新。首轮记录与补跑修订分别保存。</p></div></div><p><a href="e1-results.csv">逐任务 CSV</a> · <a href="e1-results.json">完整汇总 JSON</a> · <a href="e1-results-initial.json">补跑前快照</a> · <a href="../experiments/README.md">实验入口说明</a></p></section>
<p class="footer">数据来源：本机逐 episode JSON、events.jsonl 与冻结运行协议。HTML 由 experiments/render_progress.py 生成，统计由 experiments/analyze_results.py 从原始记录重算。未将基础设施错误填为策略失败。</p>
</main><script>
let filter='all';const body=document.querySelector('#results tbody');const search=document.querySelector('#search');function apply(){for(const r of body.rows){const match=filter==='all'||(filter==='complete'?r.dataset.status==='complete':r.dataset.status!=='complete');r.hidden=!match||!r.dataset.task.includes(search.value.trim().toLowerCase());}}document.querySelectorAll('[data-filter]').forEach(b=>b.addEventListener('click',()=>{filter=b.dataset.filter;document.querySelectorAll('[data-filter]').forEach(x=>x.classList.toggle('active',x===b));apply();}));search.addEventListener('input',apply);document.querySelector('#sort-rate').addEventListener('click',()=>{[...body.rows].sort((a,b)=>Number(b.dataset.rate)-Number(a.dataset.rate)).forEach(r=>body.appendChild(r));apply();});
</script></body></html>"""
    values = {
        "E7LATEST": e7_latest,
        "E6LATEST": e6_latest,
        "STATUS": status,
        "UPDATED": updated,
        "START": d["started_shanghai"][:19].replace("T", " "),
        "END": d["finished_shanghai"][:19].replace("T", " "),
        "HOURS": f"{d['wall_hours']:.2f}",
        "N": d["completed_episodes"],
        "COVERAGE": f"{d['coverage'] * 100:.1f}%",
        "COMPLETE": d["complete_tasks"],
        "MACRO": f"{d['complete_subset_macro_success'] * 100:.2f}%",
        "K": d["complete_subset_successes"],
        "FULLN": d["complete_subset_episodes"],
        "MAXABS": f"{p['max_abs']:.5f}",
        "RELL2": f"{p['relative_l2'] * 100:.2f}%",
        "INSIGHT": insight,
        "CAUTION": "shake_bottle 首轮仅完成 4 次，4/4 不能支持高成功率结论；其余未完成任务也不与 n=100 的完整任务直接排名。"
        if not complete
        else "shake_bottle 的 75% 是最终有效场景上的条件成功率：首轮前 100 候选中有 40 个因初始化不稳定被替换，补足过程中共排除 54 个。handover_mic 与 move_can_pot 各排除 1 个。该结果不等同于官方成绩；所有有效种子与排除记录已冻结。去掉 shake_bottle 后，另外 21 任务宏平均为 23.38%，仅作敏感性描述，不替代主结果。",
        "ROWS": "".join(table),
        "ISSUES": "".join(issues)
        if issues
        else "<li>22 个任务已补齐。历史初始化错误与超时仍保留在原始日志和补跑记录中。</li>",
        "MODELREV": c["model_revision"],
        "AUDIT": "通过" if d["worker_matches_contract"] else "需核对",
    }
    for key, value in values.items():
        template = template.replace("@@" + key + "@@", str(value))
    if "@@" in template:
        raise ValueError("Unexpanded template marker")
    (OUT / "progress.html").write_text(template)
    print(OUT / "progress.html")


if __name__ == "__main__":
    main()
