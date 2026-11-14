# 配置入口

| 目录 | 用途 | 当前边界 |
| --- | --- | --- |
| [pipelines/robotwin2_sft_recap.yaml](pipelines/robotwin2_sft_recap.yaml) | RoboTwin 2.0普通SFT→公开RECAP开发链 | 配合`scripts/run_sft_recap.sh`；生成阶段YAML、交接权重及配对评测，合格逐步数据仍需准备 |
| [multitask/](multitask/two_track.json) | 双线规划、P0、[M0](multitask/method_m0.json)、[M1](multitask/method_m1.json)、[读出](multitask/method_m1_readout.json)／[二值回报](multitask/method_m1_binary.json)、[M2](multitask/method_m2.json)／[联合训练](multitask/joint_value.yaml)、[M4](multitask/method_m4.json) | `two_track.json` 不执行训练；M1 冻结特征，M2 联合 warm-up，M4 有限选层／W0／固定标签 native policy 开发对照 |
| `robotwin_pilot/` | 旧两任务 RECAP 四阶段 YAML | 工程参考数据／预算，不能直接作为八卡正式对照 |
| `recap_official/` | 独立 LIBERO 公开流程参考 | 使用独立源码／环境，保留旧实验预算 |

两种服务器的任务见[两卡方法线](../docs/method-2gpu.md)、[八卡前置线](../docs/prerequisite-8gpu.md)。跨服务器启动见[RoboTwin 2.0 SFT→RECAP](../docs/sft-to-recap.md)。总YAML是可组合的开发入口，不等于正式数据／预算已经锁定；候选方法另立协议，不复制训练器。

M4 用 `bootstrap/prepare_method_m4.py` 生成作业表，分别复用 `value_head.yaml` 和 `joint_value.yaml`，由扩展模块选择模型／worker。候选层最多一个；所有策略从同一 W1 初始化，只更新 131,072 个 adapter 参数。它借用官方条件路由与训练循环，属于 RECAP-style 适配；不能替代官方 openpi_cfg／完整动作专家基线。

[动作专家 follow-up](multitask/method_m4_expert.json) 用同一入口的 `policy|eval --expert`：BC／C1 weighted／C2 weighted × 两个更新范围，共6组，仍为128步。保留原 native 计算精度，同一 FP32MasterAdamW；adapter-only 参考计算并丢弃专家梯度，排除专家参数更新，以匹配前向路径。所有范围、实际优化参数、采样、梯度及保存恢复需真实审计，不能只切一个冻结开关。

[路径检查](multitask/native_path_check.json)固定真实观测、噪声与去噪时间，比较native joint／cache；官方CFG独立诊断不改变生产后端。[BC曲线](multitask/bc_stability.json)仅比较两个预定expert学习率与32／64／128步，复用原数据和worker。`prepare_bc_stability.py`生成计划，`bc_stability.py`仅增加中间评测权重导出，当时保存末点DCP，历史大检查点现已按空间规则清理；中间点不含优化器，不能用于恢复训练。

[数据覆盖](multitask/data_coverage.json)固定原Motus采集策略、192个新训练初态和W1更新起点，比较每任务4／12条成功轨迹的嵌套集合。`prepare_data_coverage.py`先审计并冻结数据，再复用原worker；原验证场景及最终测试隔离保持。它同时改变可用轨迹量、场景覆盖和重复率，不能据此唯一归因于某种数据机制。

[目标响应](multitask/instruction_response.json)先验证同场景目标、checker和参考动作，再做原模型／W1／W2的指令干预；未通过不训练目标辅助组件。[标签曝光](multitask/label_exposure.json)冻结已有192条成功／失败训练轨迹和原24条验证轨迹，固定C1／W1，比较真实、严格打乱、等量成功片段与成功BC。4组各128步、统一评末点；`scripts/run_label_exposure.py`管理该限定队列，运行状态只保存在忽略目录，不能重复启动同一输出路径。

## 既有诊断配置

[policy梯度诊断](multitask/policy_gradients.json)固定共享critic标签，比较普通求和／任务范数均衡／PCGrad各3个actor种子、128步及216场开发评测。已选draw按任务循环4次，三组匹配512次有效监督，普通求和参考重新训练。`prepare_policy_gradients.py`封存采样与作业，`run_policy_gradients.py`先验收四组4步检查和原生累积精确重放，再排训练／评测队列；所有输出位于忽略的runtime目录，已有controller不可重复启动。原始梯度范数对齐不等于AdamW实际步长对齐，两者分开记录；任务梯度分组不等于instruction语义监督。

[BC种子与critic信号](multitask/critic_signal.json)固定两个新增BC种子、12个共享／独立／容量控制head，以及5组匹配每任务有效监督量的actor。`prepare_critic_signal.py`封存复测起点及原Motus特征，`plan_critic_signal.py`生成head和按draw权重计划，`run_critic_signal.py`管理各阶段及审计。此处按draw排名保留64次监督／任务，普通FM且无guidance dropout，是归因诊断，与原RECAP全局阈值／CFG流程不同；失败保留，不调到通过。

[单任务／联合BC诊断](multitask/task_mixing.json)从原Motus和共同零输出adapter出发，训练A-only／B-only／mixed三组，复用原官方worker训练／优化器及检查点循环。每任务样本和FM噪声流精确配对；`scripts/prepare_task_mixing.py`验收初始化并封存计划，`scripts/run_task_mixing.py run`管理三组4步检查、128步训练及168场开发评测。同总样本预算与同每任务曝光分别比较；该配置不训练critic或语义reward，也不是官方完整RECAP复现。已有controller目录禁止重复运行。

## 单阶段官方入口

`scripts/run.py` 要求显式 Hydra YAML，`--` 后参数原样传入。先预览命令：

```bash
python scripts/run.py recap-returns --config configs/robotwin_pilot/returns.yaml --dry-run
```

| 子命令 | 官方程序／用途 |
| --- | --- |
| `sft` | `examples/sft/train_vla_sft.py` |
| `recap-returns` | 官方轨迹回报计算 |
| `recap-value` | 官方 RECAP critic 训练 |
| `recap-advantages` | 官方优势标注；可用 `--nproc` 做本机多 GPU 标注 |
| `recap-policy` | 官方 CFG policy 更新 |
| `train` / `eval` | 官方在线训练／独立评估 |

训练的多卡放置由 RLinf 配置控制，不额外套 torchrun。`PROJECT_ROOT`、`REPO_PATH` 等由环境／入口设置；新数据 tag、输出目录与预算需显式记录。

## pilot 配置迁移

旧 `configs/recap_{returns,value,advantages,policy}.yaml` 已原样移到 `configs/robotwin_pilot/{returns,value,advantages,policy}.yaml`。四份内容 SHA256 保持不变，调用方与 Hydra 搜索目录同步更新；`scripts/recap.py` 仍是历史 pilot 入口。旧冻结运行快照保持原样。

收益比较仍须遵守[共同协议](../docs/multitask-critic.md)：critic 实际输入不由 `include_state` 决定，回报与标签标尺一致，任务 raw weight 不等于实际任务曝光，零更新 CFG 条件与训练后部署路径分别记录。
