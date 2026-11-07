# 实验结果总览

整理时间：2026-11-07T14:48:24+08:00（Asia/Shanghai）。本表为生成时快照，E7最新曲线见[实时页面](five-task-curves.html)。

| 实验 | 状态 | 结果与解释 | 详细记录 |
| --- | --- | --- | --- |
| E0 推理接入诊断 | 诊断完成；数值对齐未通过 | 合成观测的全模型数值对照未达原阈值；后续132个真实快照的输入／输出变换核对通过。两者验证范围不同。 | [报告](diagnostics/model/motus-transform-audit.json) · [原始记录](../.runtime/experiments/e0-parity/) |
| E1 原始Motus多任务SFT基线 | 完成：22任务×100场景 | 566/2200，25.73%。固定Clean本地协议；不是论文官方成绩。 | [报告](results/e1/e1-results.csv) · [原始记录](../.runtime/experiments/e1-reference-clean/) |
| E2 PACE单任务SFT权重对照 | downloading: adjust_bottle | 尚无有效评测成绩；保留为待完成外部checkpoint对照，不填为0%。 | [报告](pace-comparison.json) · [原始记录](../.runtime/experiments/e2-pace-single-task/) |
| E3 动作执行窗口诊断 | 完成：3任务×3窗口×20场景 | H=8/16/32的排序随任务变化，较短窗口并非总是更好。 | [报告](results/e3/motus-window-diagnostic.json) · [原始记录](../.runtime/experiments/e3-motus-window-diagnostic/) |
| E4 离线RECAP短预算适配 | 完成：5个ST＋1个MT＋1个重放 | 同100场景：原始47、ST29、MT29、成功重放25；没有总体MT低于ST的证据。不是原论文完整复现。 | [报告](results/e4/e4-analysis.html) · [原始记录](../.runtime/experiments/e4-offline-rl/) |
| E5 旧配置单任务Flow-SDE＋PPO | 已完成瓶子任务；其余旧矩阵关闭 | 同20场景：原始9；10/20/100轮为7/2/1。该退化发生在单任务训练。 | [报告](results/e5/pirl-progress.html) · [原始记录](../.runtime/experiments/e5-pirl/) |
| E6 精度、显存与短程PPO诊断 | 完成：D0、D1、D2 | 原始9/20，第5轮11/20，第10轮9/20；工程审计通过，收益没有维持。精度和更新次数同时变化，不能单独归因。 | [报告](results/e6/stability-progress.html) · [原始记录](../.runtime/experiments/e6-posttraining-diagnostic/) |
| E7 五任务单任务／共享RL曲线 | 在线训练 | 当前主实验：5个ST＋1个MT，各100轮、1个训练种子，每任务交互预算一致。未完成对照不能解释为多任务优劣。 | [报告](five-task-curves.html) · [原始记录](../.runtime/experiments/e7-five-task-curves/) |

不同实验的任务集合、预算和适配程度不同，不能将此表直接当作算法排名。E7只在同一任务集合全部评完后计算宏平均；未完成评估不填失败或0%。

[可导出CSV](results-summary.csv) · [结构化索引](experiment-index.json) · [训练经验](takeaway.md) · [目录说明](README.md)
