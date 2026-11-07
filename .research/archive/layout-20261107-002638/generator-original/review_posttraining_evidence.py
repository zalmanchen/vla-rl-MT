"""Rebuild the paper/evidence audit and plots without changing training jobs."""
import html
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '.research'
RUN = ROOT / '.runtime/experiments/e5-pirl'


def read_metrics():
    path = RUN / 'jobs/single-adjust_bottle-seed0/training-micro4.log'
    text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', path.read_text(errors='replace'))
    blocks = re.split(r'Global Step:\s*(\d+)\s*/\s*100', text)
    keys = ['critic/explained_variance', 'critic/value_loss', 'actor/approx_kl',
            'actor/ratio', 'actor/grad_norm', 'actor/clip_fraction', 'success_once',
            'return', 'num_trajectories', 'actor/lr', 'critic/lr']
    rows = []
    for i in range(1, len(blocks), 2):
        row = {'step': int(blocks[i])}
        for key in keys:
            match = re.search(r'(?<![\w/])' + re.escape(key) + r'=\s*([-+\deE.]+)', blocks[i + 1])
            if match:
                row[key] = float(match.group(1))
        rows.append(row)
    assert [r['step'] for r in rows] == list(range(1, 101))
    assert {r['actor/lr'] for r in rows} == {5e-6}
    artifact = {'source': str(path.relative_to(ROOT)), 'rows': rows,
                'note': 'Rounded console metrics. Sampled approx_kl is signed, not exact KL. Negative explained variance needs target-variance context. Train success is not paired test success.'}
    (OUT / 'pirl-training-metrics.json').write_text(json.dumps(artifact, indent=2))
    return rows


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({'font.size': 11, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    steps = [r['step'] for r in rows]
    ax = axes[0, 0]
    ax.plot([0, 10, 20, 100], [45, 35, 10, 5], 'o-', color='#bd4747')
    for x, y, k in zip([0, 10, 20, 100], [45, 35, 10, 5], [9, 7, 2, 1]):
        ax.annotate(f'{k}/20', (x, y), xytext=(5, 8), textcoords='offset points')
    ax.set(title='Paired evaluation: adjust_bottle, one training seed', xlabel='Training iteration', ylabel='Success (%)', ylim=(0, 60))
    ax = axes[0, 1]
    train = np.array([r['success_once'] for r in rows]) * 100
    ax.plot(steps, train, color='#aebdc7', alpha=.6, label='Logged per-iteration rate')
    ax.plot(steps[9:], np.convolve(train, np.ones(10) / 10, mode='valid'), color='#087f8c', label='10-iteration mean')
    ax.set(title='Stochastic training rollouts (different scenes)', xlabel='Training iteration', ylabel='Logged success (%)')
    ax.legend(fontsize=9)
    ax = axes[1, 0]
    ax.plot(steps, [r['critic/explained_variance'] for r in rows], color='#8367ac')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set(title='Critic explained variance: 71/100 values negative', xlabel='Training iteration', ylabel='Explained variance')
    ax = axes[1, 1]
    ax.plot(steps, [r['actor/ratio'] for r in rows], color='#c47c20')
    ax.axhline(1, color='gray', linestyle='--')
    ax.set(yscale='log', title='Logged mean probability ratio (log scale)', xlabel='Training iteration', ylabel='Mean ratio')
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.savefig(OUT / 'pirl-training-diagnostics.svg')
    fig.savefig(OUT / 'pirl-training-diagnostics.png', dpi=160)
    plt.close(fig)


def main():
    rows = read_metrics()
    plot(rows)
    commit = '5c3562be65daeea2951ccc8cbc261384b7460ec8'
    official = f'https://github.com/RLinf/RLinf/blob/{commit}/examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi_pi05.yaml'
    sources = [
        {'id': 'pirl', 'name': 'πRL v3', 'url': 'https://arxiv.org/html/2510.25889v3', 'location': '§5.4；附录G，图14–15（PDF第19–20页）；附录B/J',
         'finding': '报告critic早期不准伴随先降后升；讨论噪声、去噪步数、学习率与稳定性的耦合。部分设置用余弦退火抑制KL增长。实现采用同GPU串行执行环境、采样与更新。'},
        {'id': 'recap', 'name': 'π*0.6 / RECAP', 'url': 'https://www.pi.website/download/pistar06.pdf', 'location': '§IV-B、VI；附录D–F（PDF第17–18页）',
         'finding': '策略提取依赖优势条件；后训练通常按任务设约40% rollout正标签，条件dropout为30%。咖啡实验只有一轮。其离线PPO变体遇到信赖域约束困难，另作稳定化，不能等同在线Flow-SDE。'},
        {'id': 'reinflow', 'name': 'ReinFlow · NeurIPS 2025', 'url': 'https://proceedings.neurips.cc/paper_files/paper/2025/file/98d6f928497d9eaad9f320fb2db3040d-Paper-Conference.pdf', 'location': '§4.3–4.4、§6；附录G',
         'finding': '噪声幅度影响探索与精细控制；提供W2和熵正则，但正则并非越强越好。其视觉任务有较小PPO裁剪参数，不能直接搬到π0.5。'},
        {'id': 'pace', 'name': 'PACE v1', 'url': 'https://arxiv.org/html/2606.00537v1', 'location': '§4、§4.4；附录B/D',
         'finding': '执行窗口对成功率的影响依任务变化且不单调。每任务独立训练π0.5、预测50步；正式每任务每方法评估900次。固定窗口H=25与PACE共用checkpoint。'},
        {'id': 'motus', 'name': 'Motus v2', 'url': 'https://arxiv.org/html/2512.13030v2', 'location': '§5.2；表14；PDF第7、18页',
         'finding': 'RoboTwin比较使用联合Clean/Randomized数据、4万步微调、每任务100次评估。π0.5是论文基线；表14可作外部参照，不能替代本机实测。'},
        {'id': 'motus_card', 'name': 'Motus π0.5模型卡', 'url': 'https://huggingface.co/motus-robotics/pi0.5_robotwin2', 'location': 'Training Data & Setting / Training Configuration',
         'finding': '明确是50任务联合微调，动作预测长度32；num_train_steps=1000000为配置字段，不能据此认定发布权重实际训练了100万步。'},
        {'id': 'cel', 'name': 'CEL / FERT v1', 'url': 'https://arxiv.org/html/2610.25736v1', 'location': '§3.1.5；附录B（成功重放的局部改进条件）',
         'finding': '成功重放的理论改进依赖当前策略近似、足够小的期望梯度更新等条件；FERT用失败上下文加权成功片段。该论证不保证任意固定混合数据上多轮FM更新均改善。'},
        {'id': 'pcgrad', 'name': 'PCGrad · NeurIPS 2020', 'url': 'https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf', 'location': '§1–3、§5.3',
         'finding': '研究任务梯度冲突、梯度幅度差和曲率共同造成的优化困难；负余弦是诊断量，不能仅由任务成功率下降推断冲突。'},
        {'id': 'rlinf', 'name': 'RLinf 固定版本 RoboTwin 示例', 'url': official, 'location': '本地固定commit的YAML；不将当前仓库示例等同论文实验',
         'finding': '下表参数来自本地固定版本源码。πRL论文实验与此RoboTwin示例、我们的Motus适配是三个不同配置层次。'},
    ]
    observations = [
        {'id': 'early_drop', 'title': '在线RL：训练越久，评估越差',
         'local': '同20场景：原始9/20，10轮7/20，20轮2/20，100轮1/20。第10轮新增成功2个、丢失4个；第20轮新增1个、丢失8个。',
         'sources': ['pirl'], 'verdict': '论文有相似的早期下降现象，但我们的已测检查点没有恢复。71/100轮critic解释方差为负，第100轮−20.803；这不支持把当前退化直接视为正常预热。',
         'next': '检查critic目标分布、优势方向与更新幅度；用冻结actor的critic诊断区别“critic未学好”与其他问题。负解释方差也可能受目标方差很小影响。'},
        {'id': 'rollout_scale', 'title': '快速缩小规模后，优化条件也发生了变化',
         'local': '本地2个环境、1轮rollout、每环境512名义动作点、优化batch32；官方示例256环境、4轮rollout、200点、batch2048。双方update_epoch=5、actor lr=5e-6。',
         'sources': ['rlinf'], 'verdict': '本地每次采集名义动作点只有示例的1/200。这是配置差异，不是已证明的退化原因；小规模验证不能默认保留相同优化强度就能复现。',
         'next': '优先保持足够的独立轨迹覆盖，可通过累积采样和梯度累积控制显存；对同批数据更新1遍与5遍做单因素检查。'},
        {'id': 'noise', 'title': '探索噪声、ODE评估与概率比率',
         'local': 'noise_level=0.3、10步去噪。训练日志第3轮平均ratio=1935.7、第20轮=21.056；零更新logprob重算已通过，但只覆盖初始审计样本。',
         'sources': ['pirl', 'reinflow'], 'verdict': '概率比率长尾和训练／评估分布差异值得检查。ratio不等于精确KL；日志approx_kl也可能为负，不能据此声称真实KL单调增长或已证实爆炸。',
         'next': '先做同权重、同场景的无更新ODE/SDE闭环对照；记录实际训练batch的log-ratio分位数、KL估计和有效mask。减噪声必须同时关注梯度变化，不能认定越小越稳。'},
        {'id': 'recap_gap', 'title': '离线RECAP与成功重放均未提升',
         'local': 'E4五任务同100场景：原始47，单任务RECAP29，多任务RECAP29，成功重放25。',
         'sources': ['recap', 'cel'], 'verdict': '单轮本身不是违背RECAP。我们的普通π0.5初始化、全局top30%标签、仅正条件和10% dropout等与完整论文流程有区别。只能称这次适配未改善。成功重放退化也不反证CEL的有条件局部结论。',
         'next': '离线先检查逐任务优势标签分布、条件分支和数据覆盖；若做对照，固定数据、初始化与更新预算，仅改变一个因素。'},
        {'id': 'value_loss', 'title': '价值损失降低，策略仍未改善',
         'local': 'E4日志可核对的value验证MSE约0.04567，但该数值本身不证明优势标签可靠。E5 value_loss从首轮约0.0019降至第100轮约0.000338，同时成功率下降、解释方差为负。',
         'sources': ['recap', 'rlinf'], 'verdict': '两套value模型和指标不能混为一谈。总体回归误差不保证局部优势差值、阈值附近正负标签或策略提取正确；这是待检验的解释，不是论文给我们的因果结论。',
         'next': '查看分任务、成功／失败、轨迹阶段的校准与排序；检查阈值附近标签稳定性，报告目标方差，不能只看loss。'},
        {'id': 'multitask', 'title': '目前是否已有多任务冲突证据',
         'local': 'E4单任务和共享策略总体均29/100；逐任务有升有降。E5完成的训练只有瓶子一个任务。',
         'sources': ['pcgrad'], 'verdict': '尚无“多任务总体更差”的本地证据。单任务已退化，不能归因于跨任务梯度冲突，也不能将不同任务的涨跌本身作为冲突证明。',
         'next': '先取得稳定单任务对照，再做等预算ST／MT；同时记录逐任务收益与梯度余弦／幅度。总均值、覆盖率和退化任务都需报告。'},
        {'id': 'horizon', 'title': '执行窗口改变成绩，且更短不一定更好',
         'local': 'E3同20场景：瓶子H8/H16/H32为0/6/9；按铃4/10/8；叠块9/12/7。',
         'sources': ['pace'], 'verdict': '与PACE报告的任务依赖、非单调现象一致。这支持固定比较协议，尚不能解释本地与论文成绩差距的全部大小。',
         'next': '所有方法使用同一执行协议。若调窗口，用开发集选择，避免给每个模型按测试最优窗口报成绩。'},
        {'id': 'external_gap', 'title': '本地低于Motus论文；PACE单任务高于Motus多任务',
         'local': 'Motus相同22任务论文均值46.36%，本地25.73%。PACE与Motus的50任务固定窗口成绩57.8%和42.98%来自不同训练／评估协议。',
         'sources': ['motus', 'motus_card', 'pace'], 'verdict': '跨论文差值不能识别多任务训练的因果损失。已排除“发布权重训了100万步”的推断：安全读取的本地元数据global_step=40000与Motus论文一致。其他推理、版本、种子与图像处理差异仍待核查。',
         'next': '本机配对比较以本机原始模型为参照；外部论文只作背景。继续核对相机、指令、归一化、执行窗口、仿真版本与种子清单，不重做初始SFT。'},
        {'id': 'runtime', 'title': 'GPU利用率不持续满载，训练耗时较长',
         'local': '本地单卡、2个训练环境、FP32、CPU offload，并保存多个完整检查点。',
         'sources': ['pirl', 'rlinf'], 'verdict': '串行环境／采样／更新及数据搬运可解释部分空闲；瞬时0%利用率不能证明训练挂起，也不能认定所有时间都由模型计算占用。',
         'next': '分开统计环境推进、推理、参数同步和checkpoint I/O；前置实验先减少重复作业与评估范围，优化系统吞吐另作计时验证。'},
    ]
    comparisons = [
        ['初始化', 'RLinf单任务adjust_bottle SFT', 'Motus 50任务SFT'],
        ['训练环境数', '256', '2'], ['每次采集rollout轮数', '4', '1'],
        ['每环境每轮名义动作点', '200', '512'], ['每次采集名义动作点合计', '204800', '1024'],
        ['优化global_batch_size', '2048', '32'], ['update_epoch', '5', '5'],
        ['Actor / critic学习率', '5e-6 / 1e-4', '5e-6 / 1e-4'],
        ['critic_warmup_steps', '0', '0'], ['执行动作块 / 去噪步数', '50 / 5', '32 / 10'],
        ['训练max_episode_steps', '200', '400'],
    ]
    local_sources = {
        'early_drop': ['pirl-pilot-results.json', 'pirl-training-metrics.json'],
        'rollout_scale': ['../.runtime/experiments/e5-pirl/configs/single-adjust_bottle-seed0.yaml', '../third_party/RLinf/examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi_pi05.yaml'],
        'noise': ['pirl-training-metrics.json', '../.runtime/experiments/e5-pirl/zero-update-audit.json'],
        'recap_gap': ['e4-analysis.json', 'e4-recap-paper-audit.json'],
        'value_loss': ['../.runtime/experiments/e4-offline-rl/value-workflow.log', 'pirl-training-metrics.json'],
        'multitask': ['e4-analysis.json', 'pirl-pilot-results.json'],
        'horizon': ['../.runtime/experiments/e3-motus-window-diagnostic', 'e4-analysis.json'],
        'external_gap': ['motus-paper-reference.json', 'motus-checkpoint-metadata-audit.json'],
        'runtime': ['../.runtime/experiments/e5-pirl/configs/single-adjust_bottle-seed0.yaml', '../.runtime/experiments/e5-pirl/jobs/single-adjust_bottle-seed0/training-micro4.log'],
    }
    for item in observations:
        item['local_evidence'] = local_sources[item['id']]
        assert all((OUT / name).exists() for name in item['local_evidence'])
    checks = [
        ['优先1 · 无更新对照', '同权重同场景比较参考ODE、原生ODE、训练SDE；核对真实训练batch的old/new logprob重算。', '判断探索转换或推理接入是否在更新前已损伤行为。'],
        ['优先2 · critic与更新审计', '记录目标方差、优势分布、概率比率分位数；冻结actor检查critic能否稳定拟合。确认warmup计数单位是optimizer step。', '区分价值估计问题、mask/终止问题与更新过强。'],
        ['优先3 · 小预算单因素', '稳定性审计后，在同采样预算下比较每批1次与5次更新；学习率／预热另做独立因素，固定短检查点。', '找到可重复的单任务改善配置，不按这20个诊断场景挑论文最好结果。'],
        ['随后 · ST／MT小矩阵', '三个任务、一个训练种子、统一协议和实际预算计账；有稳定信号后再扩种子与任务。', '检验共同后训练能否同时改善多个技能，及额外干扰。'],
    ]
    evidence = {'reviewed_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                'scope': 'Literature and existing evidence audit only; no training launched or hyperparameters changed',
                'rlinf_commit': commit, 'sources': sources, 'observations': observations,
                'config_comparison': comparisons, 'next_checks': checks,
                'e5_negative_explained_variance_iterations': sum(r['critic/explained_variance'] < 0 for r in rows),
                'limits': ['One training seed; 20 paired diagnostic scenes; not final paper evidence',
                           'Literature analogy is not identification of the cause of this run',
                           'Official RoboTwin example is not the same as the experiments in piRL v3']}
    (OUT / 'phenomena-paper-audit.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
    esc = html.escape
    by_id = {s['id']: s for s in sources}
    def citations(ids):
        return ' · '.join(f'<a href="{by_id[i]["url"]}">{esc(by_id[i]["name"])}</a>（{esc(by_id[i]["location"])}）' for i in ids)
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>实验现象与原论文核对</title><style>body{max-width:1180px;margin:36px auto;padding:0 24px;color:#173142;background:#f3f6f9;font:19px/1.8 system-ui}section{background:white;padding:26px;margin:22px 0;border-radius:12px}h1{font-size:34px}h2{font-size:25px}table{width:100%;border-collapse:collapse}td,th{padding:12px;text-align:left;border-bottom:1px solid #d6e0e6;vertical-align:top}a{color:#007b83}img{width:100%;height:auto}.tag{color:#895108}p{margin:12px 0}@media print{body{background:white;font-size:12pt}section{break-inside:avoid}}</style></head><body><h1>实验现象与原论文核对</h1>'''
    page += f'<p>核对时间：{esc(evidence["reviewed_at"])}。读取原论文、固定版本代码与本地逐场景／训练日志；本次没有启动训练或修改训练参数。</p>'
    page += '<section><h2>当前判断</h2><p><strong>论文有相似现象和排查线索，但尚未证明本地退化的原因。</strong>重点已从扩大实验改为核对采样规模、critic和策略更新稳定性。当前结果不支持“多任务梯度冲突已得到验证”，也不支持“RL无效”。</p><p>不应把之前的“可能只是训练太久”当结论：20轮已下降到2/20；也不能把“缩短到5–10轮”视为已经验证的修复方案。</p></section>'
    page += '<section><h2>本地证据：早期与后期都需要解释</h2><img src="pirl-training-diagnostics.svg" alt="本地配对评估、训练成功率、critic解释方差和策略概率比率曲线"><p>左上是同20场景配对评估，其余来自训练日志，训练场景不同。解释方差在71/100轮为负；概率比率存在长尾。它们是诊断信号，不能单独确定因果。第10轮只少成功2个，单种子小样本应谨慎解释。</p><p><a href="pirl-training-metrics.json">100轮指标</a> · <a href="pirl-pilot-results.json">配对场景统计</a> · <a href="pirl-training-diagnostics.png">导出PNG</a></p></section>'
    page += '<section><h2>原论文实际说明了什么</h2>'
    for s in sources:
        page += f'<p><strong><a href="{s["url"]}">{esc(s["name"])}</a></strong>（{esc(s["location"])}）：{esc(s["finding"])}</p>'
    page += '</section>'
    for item in observations:
        page += f'<section id="{item["id"]}"><h2>{esc(item["title"])}</h2><p><strong>本地现象：</strong>{esc(item["local"])}</p><p><strong>证据判断：</strong>{esc(item["verdict"])}</p><p><strong>下一项验证：</strong>{esc(item["next"])}</p><p>{citations(item["sources"])}</p></section>'
        page += '<p>本地记录：' + ' · '.join(f'<a href="{esc(name)}">{esc(Path(name).name)}</a>' for name in item['local_evidence']) + '</p>'
    page += '<section><h2>与固定版本官方示例的差异</h2><p>这是配置审计，不是严格论文复现判定。名义动作点按环境数×rollout轮数×每轮点数计算，实际执行量还受提前结束等因素影响；优化batch是训练样本单位，不是完整轨迹数。</p><table><tr><th>项目</th><th>官方RoboTwin示例</th><th>本地E5</th></tr>'
    page += ''.join('<tr>' + ''.join(f'<td>{esc(v)}</td>' for v in row) + '</tr>' for row in comparisons)
    page += '</table><p>critic_warmup_steps=0与官方示例相同，不能说是漏抄必选参数。冻结VLM也不是对πRL方法的偏离。预热或降低更新强度应作为待验证的适配策略。</p></section>'
    page += '<section><h2>按证据安排下一轮快速验证</h2><table><tr><th>顺序</th><th>操作</th><th>回答的问题</th></tr>'
    page += ''.join('<tr>' + ''.join(f'<td>{esc(v)}</td>' for v in row) + '</tr>' for row in checks)
    page += '</table><p>已有20场景用于开发诊断；正式论文采用独立测试场景与多个训练种子。所有预定检查点都报告，保留正向和负向结果。</p></section>'
    page += '<p><a href="phenomena-paper-audit.json">结构化核对记录</a> · <a href="literature-review-20261106/sources.json">下载原文版本与SHA256</a> · <a href="pirl-progress.html">当前实验进度</a> · <a href="progress.html">总记录</a></p></body></html>'
    (OUT / 'phenomena-paper-audit.html').write_text(page)
    print(json.dumps({'observations': len(observations), 'sources': len(sources), 'training_metric_rows': len(rows), 'output': str(OUT / 'phenomena-paper-audit.html')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
