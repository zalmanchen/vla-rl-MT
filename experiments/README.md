# 实验入口与结果索引

结果从 [总览](../.research/progress.html) 查看；当前主实验为 [E7五任务曲线](../.research/five-task-curves.html)，训练与排障经验固定记录在 [takeaway.md](../.research/takeaway.md)。

## 当前运行：E7

五个单任务策略与一个五任务共享策略，从已完成多任务SFT的Motus π0.5独立初始化。任务为adjust_bottle、click_bell、stack_blocks_two、place_empty_cup、place_container_plate。先使用1个训练种子，各100轮，在0／5／10／20／50／100轮评估相同20个开发场景。

采用Flow-SDE＋PPO、严格FP32、1遍更新、microbatch4、global batch32、冻结VLM、无critic预热。共享策略每任务2个环境，共10个。每任务最终102400个名义动作点；五个单任务合计与共享策略总预算各512000，实际执行量另记。

| 用途 | 入口／记录 |
| --- | --- |
| 冻结计划与配置 | [plan.json](../.runtime/experiments/e7-five-task-curves/plan.json)、`configs/` |
| 控制器 | `pirl_five_task_controller.py` |
| 实际状态与PID | [status.json](../.runtime/experiments/e7-five-task-curves/status.json)、[controller.json](../.runtime/experiments/e7-five-task-curves/controller.json) |
| 训练扩展 | `pirl_diagnostic_actor.py`、`pirl_diagnostic_extension.py`；实际运行使用该轮`source/`快照 |
| 配对评估 | `eval_pirl.py`，原始参考采样器与冻结种子 |
| 曲线生成 | `render_pirl_five_task.py`，CPU绘图；`--watch`自动刷新 |

队列保持1个训练进程；测得至少2轮训练需求，并通过主机内存与显存双重门槛后，允许重叠1个评估进程，纯评估最多2路。评估停滞按相同种子最多自动重试一次。恢复前先确认原控制器已退出；完整权重与优化步数元数据同时存在才用于恢复，环境／RNG逐步连续性不作保证。

旧E5完整三种子矩阵仍保持关闭；E7不会自动扩展任务数、额外训练种子或成功重放。更多轮次不保证提升；当前任务是比较学习曲线，不能先假定多任务冲突或critic波动的因果关系。

## 已完成工作

| 实验 | 结果入口 | 复算／查看脚本 |
| --- | --- | --- |
| E0—E1：接入与原始基线 | [E1 CSV](../.research/results/e1/e1-results.csv)、[接入诊断](../.research/diagnostics/model/) | `analyze_results.py` |
| E3：执行窗口 | [窗口诊断](../.research/results/e3/motus-window-diagnostic.json) | `window_controller.py`的既有产物；无需重跑 |
| E4：离线适配 | [结果分析](../.research/results/e4/e4-analysis.html) | `render_offline_progress.py` |
| E5：旧PPO | [配对曲线](../.research/results/e5/pirl-progress.html) | `render_pirl_pilot.py` |
| E6：稳定性短诊断 | [结果与审计](../.research/results/e6/stability-progress.html) | `render_pirl_stability.py` |
| 论文证据 | [现象核对](../.research/literature/phenomena-paper-audit.html) | `review_posttraining_evidence.py` |

历史报告路径由 `research_artifacts.py` 集中管理；`render_progress.py`生成当前总览、文字版和CSV。整理工具 `organize_outputs.py` 默认只列迁移计划，`--apply`归档已结束的输出并记录哈希，跳过仍打开的Hydra日志。

原始完整变更日志保存在 [整理前README](../.research/archive/layout-20261107-002638/original/experiments/README.md)，属于历史记录。环境与固定版本见 [bootstrap说明](../bootstrap/README.md)。

## E7恢复前的分析快照

2026-11-07恢复前核对：共享策略完成59轮，5／10／20／50轮评估齐全；五个单任务未启动。详见[结果分析](../.research/results/e7/e7-analysis.html)。`analyze_pirl_five_task.py`从原始配对记录生成图表、CSV与证据哈希，包含本次中断快照的范围校验；新增实验结果后需更新解释再生成。
