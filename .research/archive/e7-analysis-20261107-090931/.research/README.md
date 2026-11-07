# 实验记录入口

优先查看 [实验总览](progress.html)、[当前五任务曲线](five-task-curves.html) 和 [训练经验](takeaway.md)。总览与汇总表为生成时快照；E7曲线页由正在运行的队列自动刷新。

| 位置 | 内容 |
| --- | --- |
| `results/e1/` | 原始Motus模型22任务基线，566/2200 |
| `results/e3/` | 动作执行窗口8／16／32的诊断 |
| `results/e4/` | RECAP短预算适配与成功重放，含逐任务结果 |
| `results/e5/` | 旧单任务PPO曲线与训练指标 |
| `results/e6/` | 精度、显存、critic与短程PPO诊断 |
| `diagnostics/model/` | 权重身份、输入／输出变换核对 |
| `diagnostics/recap/` | 初始化差异、原论文适配偏差 |
| `literature/` | 论文来源、原文及现象核对 |
| `planning/` | 研究计划与方法讨论；不作为已完成结果 |
| `figures/` | E1与当前E7的导出图 |
| `archive/` | 旧汇报、补跑前结果、整理前原文件及迁移清单 |

[文字版总览](results-summary.md)、[CSV汇总](results-summary.csv)、[结构化实验索引](experiment-index.json) 均由 `experiments/render_progress.py` 生成。CSV中未完成评估的成功率留空，不能将待运行单任务策略当作0%。不同任务集合／预算／协议不能直接排成算法排行榜。

## 原始数据与报告的分工

原始模型、数据集、检查点、优化器、日志、配置、种子和逐场景记录不在本目录搬动，仍位于 `../.runtime/experiments/e*/`、`../models/` 与 `../datasets/`。原始产物索引见 [运行目录说明](../.runtime/experiments/README.md)。

E6的中断尝试在 [aborted-attempts](../.runtime/experiments/e6-posttraining-diagnostic/aborted-attempts/)，评估卡住的恢复记录在相应evaluation目录。它们属于成本与可复现证据，不并入有效策略失败。

`five-task-curves.html/json` 和 `pace-comparison.json` 保留原位供运行中的进程使用。`e1-results.json` 是指向 `results/e1/e1-results.json` 的兼容符号链接，供已启动的PACE进程读取，不是另一份数据。

## 本次整理

[迁移清单](archive/layout-20261107-002638/manifest.json) 记录旧路径、新路径、文件大小与前后SHA-256。HTML／Markdown更新本地链接，历史HTML增加阶段提示；原版另存，JSON／CSV／PDF及Hydra配置内容保持原样。旧总进度页、旧生成器和完整实验变更日志也已备份。归档不等于删除，未删除任何训练权重或轨迹。

今后生成历史报告会通过 `experiments/research_artifacts.py` 写到对应分组；新增报告也应先登记分组。当前 `汇报.html` 保留为历史汇报稿，最新实验事实以总览和E7曲线为准。

在项目根目录刷新总览：

```bash
.runtime/venv-tools/bin/python experiments/render_progress.py
```

查看后续可归档的启动日志（只列计划）：

```bash
.runtime/venv-tools/bin/python experiments/organize_outputs.py
```
