# 两卡 4090：方法探索任务书

本机使用 2×约 48GB RTX 4090，在 2—3 个任务的固定开发子集上判断候选机制是否值得扩大。研究对象是已有 Motus SFT 模型的继续训练；小数据趋势不作为最终性能或创新结论。

共同模型、RLinf `e1a82eb`、RoboTwin `dca9ec6`、数据与评测约定见[共同协议](multitask-critic.md)，阶段清单见 [two_track.json](../configs/multitask/two_track.json)。不重新训练基础模型，也不改用历史 pilot 的策略作为初始化。

## 当前可以做什么

本仓库已有原模型加载、数据准备／审计和官方训练入口；共享 adapter／value 方法尚未实现，当前也没有可直接启动 M0—M4 的配置。先确定真实特征与梯度接口，再实现最小扩展。

已有两任务 pilot 缺场景追溯、最终 next observation 及部分结果覆盖，只能作为明确标记的工程开发数据。正式 M1／M2 的训练和诊断需使用经过审计并冻结的开发池；提前用于挑方法的数据不能再充当最终测试。

在已准备环境的项目根目录可先做这些核对：

```bash
source bootstrap/env.sh
python bootstrap/prepare_sources.py --verify-only
python scripts/evaluate_multitask.py status --output .runtime/runs/multitask-p0-20261111
```

第二条状态命令只适用于本机已有运行目录。完整参数与测试约定见[开发说明](development.md)。这些命令不运行方法训练。

## M0：显存与梯度路径

先列出 policy 与 critic 实际使用的特征、instruction／state 编码、共享模块位置及可训练参数。候选方案可冻结大部分 VLM，使用轻量 adapter，但位置和冻结范围需通过真实模型短测确定。

验收要求：

- 真实三相机、14 维机器人输入与 32 步动作窗口前向、反向、保存及严格复载正常。
- 分别反传 SFT 与 value loss，记录 VLM、action expert、共享 adapter、value head 的梯度和实际参数变化；检查 critic 未读取未来动作 suffix。
- stop-grad 分支中 value 梯度不进入共享部分；共享分支中存在可测梯度。单纯训练冻结特征上的独立 head 只证明表征复用。
- 记录参数量、单卡／双卡峰值显存、micro／global batch、梯度累积、预热后耗时与失败配置。

合成标签只用于检查工程。单步 loss 下降或一次成功保存不能证明 value 有用。M0 通过后才锁定接口和轻量训练范围。

## M1：最小任务条件 critic

在固定开发池上比较有／无 instruction，保持其余输入、骨干和 head 容量一致。若 q 通过离散 state token 进入网络，去 instruction 时保留相同 q；与原始独立 RECAP critic 的差异另记。

按任务、来源和 episode 报告 value 误差、偏差、校准、回报排序、advantage 分布及标签比例；结果缺类时记 NA。叠块自主数据全失败等缺口不能由专家成功数据掩盖。目标是先确认信号可学、语言是否提供额外信息；此阶段不要求完整 RL 或对比学习。

对同态不同指令，只使用已核验的有效目标和标签。不同指令的正确 V 可以相近，目标完成分类也不等价于 policy value。

## M2：继续 SFT＋critic warm-up

执行共同协议中的 W0 顺序训练、W1 同步 stop-grad、W2 同步共享梯度三个控制。各组使用相同 adapter／head 初值、容量、SFT 样本顺序与噪声、value 数据和更新预算；λ 只在开发集选择，保留 λ=0 控制。

SFT 动作损失只使用可靠成功样本，value 可用成功／失败混合。stop-grad 组隔离优化器与梯度裁剪，避免 head 梯度通过全局裁剪间接改变 actor 更新。W0/W1 在匹配随机数和 SFT 更新日程时，actor 应保持一致；两组 head 的区别仍包含最终／变化中特征的差异。

在 RL 前就评估动作能力、逐任务 value 和标签、训练成本。W2 相比 W1 的结果才主要对应 value 梯度进入共享表征的干预。value 变好而动作能力下降是需要保留的结果，不能只展示 critic loss。

## M3：有证据后再做 grounding

仅当 M1 或前置 P2 提示 goal 使用有问题，并且有独立核验的目标—状态标签时加入辅助目标。未知配对不当负例，不同任务名不当自动负例，不沿用替换指令前的轨迹回报。

所有候选与同表示的 λ=0 基线比较；辅助 loss 必须更新 value／policy 共用的可训练路径。若只改善 goal 分类，不能声称改善 RL 信号。

## M4：小规模离线 RL

先固定 W1 actor，比较普通／候选 critic 生成的标签，沿用同一 RECAP-style 更新。候选值得继续时，再按共同协议做 `actor={W1,W2} × labels={C_W1,C_W2}`，同时报告两个 actor 的零 RL 更新结果。

四组使用相同 D、advantage／阈值规则、policy 更新范围、预算和部署条件。标签生成时冻结完整 critic 快照；评估只用开发场景，不回流数据。短程无提升不单独作为否定方法的证据，也不因出现一次提升就直接扩大任务数。

## 排卡与交付

显存允许时 GPU 0 跑候选、GPU 1 跑对应基线或诊断；需要双卡时顺序运行。首先缩小开发任务／数据／预算，不在组间悄悄更换骨干、输入或冻结范围。具体卡数由 M0 测量决定。

新产物放 `.runtime/runs/method/<run-id>/`。交付代码 revision／diff、配置、数据与拆分哈希、checkpoint、固定标签、逐任务诊断与动作评估、显存和成本。交接字段见[共同协议](multitask-critic.md#两线交接)。

先推进 M0 的接口设计与 M1 的数据审计；待实现和验收后再启动训练。候选必须排除来源、容量、曝光或简单 BC 的解释，再交给[5880 前置线](prerequisite-8gpu.md)做同预算、多种子确认。
