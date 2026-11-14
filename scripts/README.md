# 脚本入口

先读[两卡任务单](../docs/method-2gpu.md)或[八卡任务单](../docs/prerequisite-8gpu.md)。新主线是SFT→RL协调。普通基线已有[RoboTwin 2.0总配置与启动器](../docs/sft-to-recap.md)；正式数据、候选A—D和跨主机任务分工仍需按任务书准备。

## 通用入口

在项目根目录准备环境并预览官方命令：

```bash
source bootstrap/env.sh
python scripts/run.py --help
python scripts/run.py recap-returns --config configs/robotwin_pilot/returns.yaml --dry-run
```

`scripts/run.py`要求显式Hydra配置，提供SFT、RECAP回报／critic／优势／CFG policy及在线train／eval入口。pilot配置只是历史参考；正式配置必须对应新的数据和预算。多卡放置由RLinf配置控制。

## 可复用研究工具

| 作用 | 工具 |
| --- | --- |
| RoboTwin 2.0 SFT→RECAP | `run_sft_recap.sh`／`sft_recap.py`，见[跨服务器说明](../docs/sft-to-recap.md) |
| 本机逐控制步数据小测→完整采集 | `run_sft_recap_data.py`持有GPU0／1作业，复用官方policy／仿真；只采数据，不启动actor训练 |
| P0模型能力与状态检查 | `evaluate_multitask.py` |
| 注册方法模型／worker，复用官方训练循环 | `train_method_value.py`、`train_method_joint.py` |
| 显式GPU0／1作业表与Ray生命周期 | `run_method_jobs.py`、`run_method_joint_pair.py` |
| 前向／梯度／复载／后端核对 | `check_method_m0.py`、`check_method_m2.py`、`check_method_m4.py`、`check_method_expert.py`、`check_expert_export.py`、`check_policy_paths.py`、`check_backend_paths.py`、`check_training_autograd.py` |
| 数据来源／动作重放与固定噪声拟合 | `audit_bc_data.py`、`evaluate_expert_fit.py` |
| head／层／actor评测与归因 | `evaluate_method_m1.py`、`evaluate_method_layers.py`、`evaluate_method_m2.py`、`evaluate_method_updates.py`、`compare_method_rewards.py` |
| 单任务／联合BC限定协议 | `prepare_task_mixing.py`、`run_task_mixing.py` |
| 标签／曝光与目标响应限定协议 | `prepare_label_exposure.py`、`run_label_exposure.py`、`evaluate_instruction_response.py` |
| BC种子与critic信号限定协议 | `prepare_critic_signal.py`、`plan_critic_signal.py`、`run_critic_signal.py` |
| 固定critic的梯度限定协议 | `prepare_policy_gradients.py`、`run_policy_gradients.py` |
| 历史范围／曲线／覆盖汇总 | `summarize_method_m4.py`、`summarize_method_expert.py`、`summarize_bc_stability.py`、`summarize_data_coverage.py` |
| 独立LIBERO／旧RoboTwin参考 | `recap_official.py`、`recap.py` |

采集、特征缓存及数据准备在[bootstrap](../bootstrap/README.md)，方法定义在`src/vla_rl_mt/`。`configs/multitask/*.json`含已封存的开发协议；这些专用controller常依赖忽略目录中的manifest、父权重及已生成作业表，并非新机器一条命令即能运行。准备新协议／输出路径，旧结果不能因大checkpoint已删除就继续从原断点启动。

## 运行与保留

当前native方法worker只支持独立单卡NO_SHARD；八卡正式训练使用已验收的官方分布式路径，不能直接给本机队列套torchrun。所有新产物进入`.runtime/`，数据视图引用共同母池；默认留一个可评测末点，确需交接／恢复才留optimizer，smoke大文件核对后清理。日志和成绩进入[集中汇总](../docs/previous-experiments.md)，完整运行规则见[开发说明](../docs/development.md)。
