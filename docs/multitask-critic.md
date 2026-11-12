# 两线共同研究协议

两卡方法探索与八卡前置验证从相同 Motus 多任务 SFT 起点出发。两线任务分别见[方法任务书](method-2gpu.md)、[前置任务书](prerequisite-8gpu.md)，阶段清单见 [two_track.json](../configs/multitask/two_track.json)。该 JSON 是规划，不是训练入口；正式数据、预算、adapter 接口和参数范围尚未锁定。

当前研究可检验，但共享／独立 critic、增加辅助目标或 SFT／value warm-up 本身尚未建立创新。目标是先区分原因，不预设语言失效、共享冲突或 RL 优于 BC。

## 来源与模型

| 项目 | 固定来源 |
| --- | --- |
| RLinf | `e1a82ebf0c1af69da6ab9b1519f5f3692fe33566`，另加记录在案的 RoboTwin 兼容补丁 |
| RoboTwin | `dca9ec682688821c42944eacd7f5dd9bfb15396f` |
| Motus π0.5 | `motus-robotics/pi0.5_robotwin2`，revision `effc6e1a99bd4d0edf9447e8ba313def67649cfe` |
| 模型权重 SHA256 | `de8e03a445f3e0af417a9d61a903d342fca5b6a5b4fdc6f20445e14ad25692eb` |

[源码锁](../bootstrap/sources.lock.json)、[补丁清单](../bootstrap/patches/recap-robotwin.json)和 [phase0.json](../configs/multitask/phase0.json)保存完整来源、统计哈希与动作设置。源码是归档快照，没有独立 `.git`；目录内的 Git HEAD 可能属于项目父仓库。用 `python bootstrap/prepare_sources.py --verify-only` 核验归档及成员，允许的 RLinf 补丁单独报告。

[固定版本模型卡](https://huggingface.co/motus-robotics/pi0.5_robotwin2/blob/effc6e1a99bd4d0edf9447e8ba313def67649cfe/README.md)描述了 50 任务 Clean／Randomized SFT；这不是本机能力验收，预训练场景重合未知。当前不重建基础模型；方法线研究的是已有 SFT 模型的继续训练。

保留三相机、14 维环境动作、32 步／32 维模型动作、delta joint／absolute gripper、Motus 统计和 `adapt_to_pi=True`。P0 使用 10 个采样步、执行前缀 32、float32 环境动作。缩短执行前缀、改变 state 或参数范围均须另立因素。

## 数据与评测

- 采集／整理在训练前完成，冻结主 D 与方法开发子集；训练期间不加入新 rollout。验证、测试和诊断续跑不回流 D。
- 按完整 episode 和可追溯场景／初态划分；记录任务、来源、采集策略、指令、成功标签依据、reward、终止／截断、最终 next observation、chunk 阶段及可知时钟。缺失保持未知，不编造。
- SFT 动作损失只用可靠成功数据，value 使用成功／失败混合；失败 BC 另立组。各组报告任务×来源×分支的实际曝光，不能假设 raw weight=1 就是任务均衡。
- 自主／专家指令风格、行为策略和数据量可能相关；按来源诊断，避免用专家成功／自主失败的分类准确率替代 value 质量。恢复行为未知时不推定不存在。
- 固定奖励、回报标尺、真实终止、固有 deadline 与外部截断。只能使用部署可知的时钟，不能用距离记录结束的时间泄漏未来；各独立 critic 不各自归一化后直接混用全局阈值。
- 方法选择只用开发场景，反复使用过的旧场景属于开发集。校准固定五个候选、官方 planner-solvable 池 `[300,320)`；`[500,550)` 保留最终确认，不用于选任务、阈值或停止点。
- 原生／CFG 差异先排查，后训练前后采用同一后端；新增 advantage 提示需零更新控制。setup error 与策略失败分开，不补抽场景，报告分母及不确定性。
- 相同种子或可见关节／物体位姿不等于完整模拟器状态一致。有效目标切换先核验可执行性及目标专用 checker；无 instruction 可能是分布外扰动。

当前 `recap-pilot-v2` 只有两任务：训练示范 16 条／3128 帧、自主 8 条／4230 帧；自主训练 2 成功／6 失败，叠块四条全失败，验证自主均失败。示范 scene seed 未记录、成功标签是转换假设；自主缺最后动作后的观测及完整可恢复状态，终止／chunk 字段也不完整。原始 50 任务 Clean 示范不等于已有 50 任务自主成功／失败池。

上述池只供明确标记的工程开发，不是正式多任务研究数据。M0 可提前做工程检查，不因此豁免训练与最终测试门槛。审计入口为 `bootstrap/audit_multitask.py`；P0 结果和当前缺口见[前置任务书](prerequisite-8gpu.md#本机已有证据与待解决问题)。

## critic 估计对象与控制

离线目标先写为 `E_D[G_t | available context]`，描述数据行为的后续结果，不自动等于新 policy 的真实 value。中间帧可能仍在执行旧 chunk，须区分续完 chunk 与立即重规划。MC 预测、终局完成分类和同态行动排序不是同一诊断。

| 对照 | 必须控制的因素 |
| --- | --- |
| C0 原始共享 critic | 独立 SigLIP2＋Gemma3＋value expert，实际输入 `V(o,l)`；不含 q，也不复用 policy 特征 |
| Ctask 独立 critic | 同输入／初始化／回报标尺；报告每任务曝光、总参数和总成本 |
| Ccapacity 共享容量控制 | 用于检查独立 critic 优势是否由容量／预算解释，不只凭差距断言任务冲突 |
| Cno-text | 保留其余输入；policy 特征版本去语言时须保留相同 state 编码 |
| Caux | 与对应 λ=0 基线同表示、输入、容量和数据；有可靠目标标签才运行 |

q、elapsed time、冻结 policy 特征与共享 adapter 各为独立因素。方法线必须有同特征的纯 value／stop-grad 基线，不能把换骨干、加 state 的收益归给联合训练。critic 只读决策时可用信息，不能使用真实未来动作 suffix。

辅助目标必须更新 value／policy 共用的可训练路径；冻结特征上互不相干的两个 head 不支持“改善共享表征”。多目标可同时为正，未知标签忽略，不同任务名不是负例。替换指令后不能沿用原回报，也不要求不同目标的正确 V 必然不同。

逐任务／来源报告期望 value MAE、偏差、校准、episode 排序、缺类情况，以及 advantage 分布、positive 比例、标签分歧和终止边界。官方 modal-bin MAE／批次 std 不代替这些指标；成功轨迹不必全 positive，失败轨迹不必全 negative。

同态参考策略续跑需要完整恢复状态、明确 chunk 处理及参考策略、多次采样和不确定性；交互只用于评价并单列成本。当前数据不具备该恢复保证。

## warm-up 与 policy 的最小归因比较

| 组 | actor | critic |
| --- | --- | --- |
| W0 | 继续 SFT，保留相同 adapter 容量 | SFT 后冻结表征，训练同一 value head |
| W1 | 相同 SFT 顺序、噪声与日程 | 同步训练 head，value 对共享表征 stop-grad |
| W2 | 相同 SFT，另接收 value 梯度 | 同输入／容量／数据的共享梯度训练 |

三组从相同 actor 和 adapter／head 初值出发，匹配 SFT、value 各自的更新数，报告总计算并保留 λ=0。W0/W1 的 actor 在随机数和日程匹配时应一致；stop-grad 组隔离优化器与梯度裁剪，避免 value head 改变 actor 的裁剪系数。W0/W1 的 head 还受最终／变化中特征影响，差距不能只解释为“提前训练”。W2/W1 才主要干预 value 梯度进入共享表征。

分别测两种 loss 对 VLM、action expert、adapter、head 的梯度和参数变化。已有 RLinf `use_rlt` 分支是 detach prefix 后的重建辅助，并非有回报监督的共享 critic，不能当作本方案已实现。

先报告 warm-up 后、RL 前的 actor 能力。M4 先固定 W1 actor 比较两种标签；关键候选再做 `actor={W1,W2} × labels={C_W1,C_W2}`，冻结完整 critic 快照，区分 actor 起点、标签和交互。

规模 policy 对照保留原始 SFT、同源成功 BC、C0／Ctask／有条件资格的 Caux 标签下的共享 actor。统一数据、初始化、动作、参数范围、采样、更新、阈值和选择规则；global／per-task quantile 是另一因素。BC 与条件 FM 还需同正例／同提示控制，区分 token、筛选与负例分支；总计算与正例曝光不一定可同时匹配，必要时分别做预算对照。

单任务／多任务比较先声明总预算或每任务预算，报告另一项成本。训练任务×评估任务矩阵包含未更新任务；梯度负余弦只作诊断。报告宏平均、最差任务、逐任务新增／丢失／保留和多个关键训练种子；五任务结果不能证明其余 45 任务能力保持。

## 两线交接

机器各自运行单机作业，默认不组成混合十卡训练。共同模型、数据与静态资产存本地盘，以相对数据根路径＋哈希定位，避免跨机逐步读取大文件。新增数据、目标标注或模拟器查询另立版本与预算。

| 节点 | 交接内容 |
| --- | --- |
| 共同起点 | 模型／源码／补丁哈希，D 与开发子集清单，episode／场景拆分，baseline，动作／回报／后端设置 |
| 问题反馈 | instruction 配对覆盖、逐任务 value／advantage 问题、actor 收益与退化、来源／容量／曝光解释及不确定性 |
| 候选确认 | 代码 revision／diff、最终配置与可训练参数、checkpoint／标签哈希、开发消融和显存／耗时，5880 按预定预算重复 |

新运行目录分别为 `.runtime/runs/method/<run-id>/`、`.runtime/runs/prerequisite/<run-id>/`。交付最少包含：

- 来源和最终配置：代码、patch、依赖／驱动、模型／统计、动作、参数清单、随机种子及停止／选择规则。
- 数据：主池／子集／拆分清单和哈希，任务×来源×结果数量，额外标注／查询成本。
- 训练：checkpoint、优化器／调度器恢复信息、标签及阈值版本、实际样本曝光、曲线和成本。
- 评估：逐回合记录、valid／attempted 分母、setup error、成功／失败、场景与噪声种子及不确定性。

这些是交接规范，尚无自动生成全部材料的统一队列。远端接收候选后先核对协议与小样本一致性，再扩预算；最终测试从不用于方法选择。

## 否定条件与历史边界

- 接口、数据来源／拆分或结果覆盖未通过：先补基础，不用长预算包装结论。
- 独立 critic 优势被容量、尺度、曝光解释：不支持“共享冲突”主张。
- 辅助目标只改善分类，value／advantage／policy 未改善：不支持解决 RL 信号问题。
- critic 指标改善而 policy 不改善：检查标签阈值、提取器、正例曝光与数据覆盖。
- 所有 critic 方法不如同源 BC，或任务内打乱标签效果相近：先检查信号／实现，不堆叠模块。

[RECAP](https://arxiv.org/abs/2511.14759)已有多任务优势条件训练；邻近工作与阅读边界见[研究账本](../.research/literature/multitask-flow-rl-evidence.json)。既有实验合并见[实验摘要](previous-experiments.md)；旧逐次记录不再随仓库维护。在线 Flow-SDE＋PPO 不充当冻结离线基线，LIBERO 长预算不自动恢复。
