"""Record a conditional research direction, separate from measured results."""
import html
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '.research'


def main():
    refs = [
        {'name': 'RLinf-Co', 'url': 'https://arxiv.org/html/2602.12628v1', 'location': '§IV-B Eq.8；§V；附录VII-D',
         'known': '已在π0.5等VLA上把仿真RL与实机监督损失联合优化。固定权重SFT＋RL必须作为强基线；只加一个SFT项不足以构成新方法。'},
        {'name': 'SOP', 'url': 'https://arxiv.org/html/2601.03044v1', 'location': '§IV-C、IV-D、V-C',
         'known': '已研究共享策略的多任务后训练；任务间均衡、任务内按近期loss调整在线／离线采样。因此，简单按任务调整数据比例也不是空白。'},
        {'name': 'LWD', 'url': 'https://arxiv.org/html/2605.00416v1', 'location': '§IV、V',
         'known': '已在16台双臂机器人、8个实机任务上研究同一generalist策略的offline-to-online改进。不能将“单策略、多任务、部署后改善”本身称作新问题。'},
        {'name': 'RECAP / π*0.6', 'url': 'https://www.pi.website/download/pistar06.pdf', 'location': '§IV–VI',
         'known': '从示范、成败经验与纠正中提取学习信号；优势条件化提供与直接策略梯度不同的后训练路径。'},
        {'name': 'CEL / FERT', 'url': 'https://arxiv.org/html/2610.25736v1', 'location': '§3.1、实验与附录B',
         'known': '成功片段监督训练也可改善执行；用失败上下文选择经验是重要对照。其任务分别训练，不能直接视作一个多任务策略的实验。'},
        {'name': 'ReinFlow', 'url': 'https://proceedings.neurips.cc/paper_files/paper/2025/file/98d6f928497d9eaad9f320fb2db3040d-Paper-Conference.pdf', 'location': '§4.4、§6',
         'known': '已讨论保持初始策略的正则与探索之间的权衡。新增保持项本身也不能替代机制创新。'},
    ]
    question = '有限示范与交互预算下，一个已有通用策略如何共同改善多项灵巧操作技能，同时控制其他技能的退化？'
    hypothesis = '不同任务的示范覆盖、奖励／价值估计可靠性和更新冲突不同；统一更新强度或仅按loss调比例可能错配学习信号。能否用可验证的收益与保留证据决定各任务采用监督更新、奖励更新或暂缓更新？'
    stages = [
        ['当前E6：稳定性诊断', '无更新SDE；5轮critic-only，另续跑1轮核验批次审计；10轮update_epoch=1的PPO。', '先确认单任务退化是否来自接入、价值估计或更新强度。队列已启动，具体状态见实时进度；不算SFT＋RL新方法结果。'],
        ['下一步：受控小矩阵', '3任务、1种子：继续监督训练、稳定后的RL、固定权重SFT＋RL。补充单任务对应参照。', '判断固定混合能否改善，以及是否仍存在任务间收益不均／退化。未启动。'],
        ['机制验证', '在相同数据与预算下，检验信号可靠性、任务冲突与收益的关系；再比较候选方法。', '固定混合、SOP式数据比例调节、PCGrad／参数隔离是竞争解释或对照；先选一个核心机制。'],
        ['仿真扩展', '逐步增加任务数和训练种子，报告收益、退化、最差任务与成本。', '先3／5／10任务，稳定后再20及以上；无需先完成50任务大矩阵。'],
        ['灵巧实机核心验证', '同一策略完成多种接触操作；加入对象／布局变化、扰动和新任务组合。', '用户已确认可使用灵巧手或双臂灵巧手实机；具体任务由手型、传感和重置条件确定。'],
    ]
    baselines = [
        ['原始策略', '统一的已有多任务SFT起点；不重新做RoboTwin初始SFT。'],
        ['继续SFT／成功重放', '区分示范监督与加入成功自主经验的收益；后者记录额外数据成本。'],
        ['稳定后的RL', '先获得有效单任务配置，再研究共享训练；当前退化PPO不能作为唯一RL基线。'],
        ['固定权重SFT＋RL', '最直接且必须较强的竞争基线；可利用RLinf已有co-training实现。'],
        ['按任务调节数据比例', '检验改进是否仅由重采样带来，对照SOP相关思路。'],
        ['候选机制', '只有在收益、退化和预算上超越上述对照，才讨论新增方法价值。'],
    ]
    assessment = {
        'updated_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'status': 'research hypothesis and conditional plan, not a validated contribution',
        'user_confirmed_hardware': 'dexterous hand or dual-arm dexterous robot; exact model not yet specified',
        'research_question': question, 'hypothesis': hypothesis,
        'evidence_correction': ['SFT ST/MT difference currently relies largely on cross-paper results, not controlled causal evidence.',
                                'Local E4 ST and MT RECAP both29/100; no overall multitask penalty demonstrated.',
                                'Completed E5 RL failure is single-task; it does not identify multitask interference.'],
        'prior_art': refs, 'stages': stages, 'baselines': baselines,
        'journal_assessment': 'Potential Science Robotics direction if it delivers a distinct mechanism and substantial, reproducible dexterous real-robot capability; simple loss addition or a RoboTwin-only gain is currently insufficient in our assessment. This is a research judgment, not a journal requirement or acceptance prediction.',
        'journal_source_access': 'Author/mission pages returned403 during review. No detailed editorial criteria are attributed to unread pages.',
        'method_status': 'Task-dependent allocation across supervised and reward-driven updates is a candidate; novelty versus SOP, RLinf-Co, LWD and related methods remains to be established.',
        'fairness': ['Same initialization and embodiment mapping for each method.',
                     'Account separately for demonstrations, autonomous episodes, corrections, resets and compute.',
                     'Use fixed-data experiments to isolate learning-rule effects; online experiments separately measure collection-policy effects.',
                     'Tasks used in retention replay are not unseen post-training tasks.',
                     'Use development scenes for decisions and independent tests for final claims; retain all scheduled checkpoints.'],
        'hardware_constraint': 'Motus RoboTwin checkpoint uses the ALOHA14-dimensional robot action interface. Dexterous hand control requires a compatible embodiment/action interface and a common adapted initialization; current RoboTwin results are not dexterous-hand results.',
    }
    (OUT / 'unified-posttraining-assessment.json').write_text(json.dumps(assessment, ensure_ascii=False, indent=2))
    esc = html.escape
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>多技能联合后训练：研究价值与实机路线</title><style>body{max-width:1160px;margin:36px auto;padding:0 24px;font:20px/1.8 system-ui;color:#1c3241;background:#f3f6f9}section{padding:28px;background:white;margin:24px 0;border-radius:12px}h1{font-size:34px}h2{font-size:26px}table{width:100%;border-collapse:collapse}td,th{padding:12px;text-align:left;border-bottom:1px solid #d9e2e8;vertical-align:top}a{color:#007b85}code{white-space:normal}@media print{section{break-inside:avoid}body{font-size:12pt;background:white}}</style><h1>从SFT＋RL组合，走向多技能可靠共同改善</h1>'''
    page += f'<p>{esc(assessment["updated_at"])}。用户已确认有灵巧手或双臂灵巧手实机；本页为候选论文路线。</p><section><h2>核心研究问题</h2><p><strong>{question}</strong></p><p>长程任务体现一个策略掌握多个技能的实际需要。核心评估是多个技能能否共同改善，而不是把“前一技能帮助后一技能”作为主要机制。</p></section>'
    page += '<section><h2>先明确已有证据</h2><p>SFT的ST／MT差距主要来自不同论文，尚非受控消融；E4单任务与多任务RECAP均29/100；E5完成的退化训练是单任务。现阶段可以提出联合后训练假设，不能宣称已经证明SFT和RL都存在共同的多任务缺陷。</p><p>正在运行的E6用于排查基础稳定性。<a href="stability-progress.html">查看实验进度</a> · <a href="phenomena-paper-audit.html">查看原论文核对</a>。</p></section>'
    page += '<section><h2>已有工作覆盖了哪些部分</h2><table><tr><th>工作</th><th>对本研究的约束</th></tr>'
    for r in refs:
        page += f'<tr><td><a href="{r["url"]}">{r["name"]}</a><br>{esc(r["location"])}</td><td>{esc(r["known"])}</td></tr>'
    page += '</table><p>RLinf本地已有 enable_sft_co_train、sft_loss_weight 和π0.5联合训练示例。RoboTwin多任务的数据入口仍需适配和验收，不能视作开关打开即复现成功。</p></section>'
    page += f'<section><h2>值得检验的候选机制</h2><p>{hypothesis}</p><p>可把统一目标写成按任务组合监督FM损失与RL损失，并加入能力保持约束。但这个表达式只是组织问题的框架；新意必须来自学习信号的选择依据和可验证机制，而不是loss相加。</p><p>建议先研究一个中心机制：<strong>依据可靠性与保留风险分配更新</strong>。梯度投影、LoRA隔离和数据调度作为可比对照，不同时堆成多个模块。权重应在训练／开发数据上决定，注意不同loss与梯度的尺度，不能访问最终测试成绩。</p><p>如果固定混合已经解决退化，或候选可靠性指标不能预测收益，就应调整假设，不预设复杂方法必胜。</p></section>'
    page += '<section><h2>实验分阶段推进</h2><table><tr><th>阶段</th><th>范围</th><th>作用与状态</th></tr>'
    page += ''.join('<tr>' + ''.join(f'<td>{esc(v)}</td>' for v in row) + '</tr>' for row in stages) + '</table></section>'
    page += '<section><h2>必要对照与公平性</h2><table><tr><th>对照</th><th>用途</th></tr>'
    page += ''.join('<tr>' + ''.join(f'<td>{esc(v)}</td>' for v in row) + '</tr>' for row in baselines)
    page += '</table><p>固定数据实验隔离学习规则；在线实验另计采集策略带来的数据变化。报告平均成功率、改善技能数、退化技能数、最差技能和实际交互／人力／计算成本。用于保持重放的任务与完全未用于后训练的任务分开报告。</p></section>'
    page += '<section><h2>灵巧实机应成为主要证据</h2><p>建议先确定一个真实工作场景，选择若干接触需求不同的技能，例如旋转／拧动、插接、双手协作和柔性物体处理；具体选择需匹配手型与传感能力。所有技能使用同一策略权重，用语言切换任务。</p><p>先做约4–6个技能的可行性试验，再依据结果扩展；测量重复后训练后的能力保持、陌生物体／布局和扰动下的恢复。最后用新组合的长程流程验证同一技能库的实用性，保持上层调度器一致。</p><p>当前Motus权重服务于ALOHA的14维夹爪动作接口，不能直接视为灵巧手策略。实机需要匹配动作表示与数据，并为所有方法提供同一个适配后的初始策略；这与重复RoboTwin初始SFT是不同工作。</p></section>'
    page += '<section><h2>Science Robotics：可以作为目标，但证据尚不充分</h2><p><strong>我的判断：</strong>具有研究契合度，尤其已有灵巧实机资源；目前的“unified SFT＋RL”概念本身尚不足以支撑投稿定位。</p><ol><li><strong>机制：</strong>解释何种条件下两类学习信号有益或有害，并用受控干预验证。</li><li><strong>能力：</strong>同一实机策略稳定掌握多项有难度的灵巧技能，在分布变化和持续改善中仍可靠。</li><li><strong>证据：</strong>强基线、独立测试、重复实验、能力保持和成本收益共同支持结论。</li></ol><p>以上是研究建议，不是期刊规定或录用预测。不预设必须多少台机器人或多少任务；一个清晰机制与扎实的实机结果，比单纯扩大矩阵更能决定工作价值。</p></section>'
    page += '<p><a href="unified-posttraining-assessment.json">结构化研究判断</a> · <a href="stability-progress.html">正在运行的诊断实验</a> · <a href="progress.html">总记录</a></p></html>'
    (OUT / 'unified-posttraining-assessment.html').write_text(page)
    print('Wrote unified post-training assessment with six primary-source references')


if __name__ == '__main__':
    main()
