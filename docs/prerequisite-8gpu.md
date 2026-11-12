# 八卡 5880：前置验证任务书

使用 8×48GB RTX 5880 建立可信事实：原始多任务 SFT 能做什么、共享 critic 是否构成瓶颈、RL 是否改善或损害不同任务。当前未连接或验收该机器；本文是交接任务书，不代表远端实验已启动。

两线从相同 Motus checkpoint 出发，固定 RLinf `e1a82eb` 和 RoboTwin `dca9ec6`。版本、动作、数据、critic 对照及终止规则见[共同协议](multitask-critic.md)，阶段清单见 [two_track.json](../configs/multitask/two_track.json)。

## 机器准备与共同起点

按[安装说明](setup.md)准备依赖、固定源码、模型、统计和 RoboTwin 资产。若从本机转移，两份固定源码已在 `third_party/`；代码之外还需准备 `models/`、所需 `datasets/`、仿真资产与冻结数据清单。源码 ZIP 缓存可选，逐文件清单支持离线源码核验。不要直接复制有绝对路径的 venv 当作环境已验收。

```bash
source bootstrap/env.sh
python bootstrap/prepare_sources.py --verify-only
nvidia-smi --query-gpu=index,name,memory.total --format=csv
nvidia-smi topo -m
python bootstrap/smoke_render.py
```

上述步骤分别验证源码、设备和实际渲染；模型加载、真实输入反向及分布式短测还要单独完成。现有 Docker 镜像的验收范围是 LIBERO，不能代替此处 RoboTwin／SAPIEN GPU 验收。记录 CPU、内存、本地盘、驱动、依赖快照及原始配置。

## P0：统一协议并冻结数据

先完成原生／CFG 输入、动作、数值路径核对；后训练前后使用同一推理后端。审计 action chunk、state／next observation、reward、成功标签、真实终止／固有 deadline／外部截断，补齐最终观测和行为策略来源。

固定任务、采集行为策略、训练／验证／测试场景与随机种子；保留完整 episode 和场景标识，不能按帧随机划分。记录每任务×来源×结果数量、指令风格、恢复行为与 chunk 阶段。收集在数据冻结前完成；评估永不进入 D。

提供主 D 和 2—3 任务开发子集的版本、内容哈希及拆分清单。跨机接口使用相对数据根路径与哈希；旧 pilot conversion 中的绝对路径不是可移植的数据契约。当前 pilot 仅用于工程参考，不能通过复制目录就升级成正式池。

P0 的协议整理可与本机 M0 并行。正式采集、转换及拆分工具需先补齐；当前 `collect_recap_rollouts.py` 仅支持两个已审核任务且字段不完整，不是五任务正式采集器。

## P1：原始 SFT 与同源成功 BC

先评测原始 checkpoint，再用同一冻结 D 中可靠成功子集继续 BC；额外专家示范需单列信息与成本。主要训练对照预先固定多个种子、停止／checkpoint 选择规则和任务采样。

逐任务报告成功率、配对场景结果与不确定性，区分能力饱和、可改善任务和基础接口问题。BC 自身造成的任务退化也要报告。训练尚未开始前，原始 checkpoint 的能力评估不依赖 BC 全部完成。

## P2：instruction 是否被使用

先建立可执行的同场景目标集合及对应 checker，再比较正确指令、另一有效目标和无 instruction。policy 的对象选择／动作／成功，与 critic 的目标判断分别报告。

不能用原任务 checker 判断替换目标的完成情况；不可执行的跨任务替换只是输入扰动。无 instruction 可能是分布外输入，结果不能单独证明语言失效。没有可靠配对时提交覆盖缺口，先继续 P1／P3，不强行产出 grounding 指标。该结果决定方法线是否需要 M3。

## P3：critic 质量与原因控制

先建立原始 RECAP 图像＋instruction 的共享 critic，再比较无 instruction 和每任务独立 critic。保持输入、初始化、回报尺度一致，报告容量、逐任务曝光、优化与总 GPU 成本；有独立 critic 优势时排查共享容量和采样因素。

报告逐任务／来源 value 误差、校准、episode 排序、advantage 分布、positive 比例及 critic 间标签分歧。成功／失败缺类记 NA，不用来源分类代替价值可靠性。关键比较补训练种子，不能只看一次 critic loss。

## P4：RECAP-style 多任务基线

比较原始 SFT、同源成功 BC、共享 critic 标签和独立 critic 标签下的同一个共享 actor。固定初始化、数据、参数范围、部署条件、advantage 和阈值规则；加入同后端零更新及新增条件提示控制。

逐任务报告新增、丢失、保留的成功，宏平均、最差任务与未后训练任务表现。总计算与正例曝光不一定能同时匹配，必要时分别做预算对照。历史两任务 pilot 和 E4／E7 不作为本阶段结果；E7 的在线 Flow-SDE＋PPO 也不属于固定离线基线。

当前 `configs/robotwin_pilot/` 只覆盖旧两任务工程数据和预算，不能直接当成八卡正式配置。正式 Hydra 配置必须在数据和预算锁定后通过 `scripts/run.py` 调用官方入口。

## P5：任务间相互影响

从同一 checkpoint 分别只训练 A、B、C，并对全部声明的任务评估，生成训练任务×评估任务矩阵，再与共享多任务训练比较。先声明总预算或每任务预算，报告另一项曝光和成本。

固定 critic 标签后，梯度方向和范数可作辅助诊断；负余弦不单独证明退化原因。五任务实验只能说明五任务结果，若要宣称保留 50 任务能力，需另列其余任务的评测成本与结果。

## 排卡与执行顺序

先测真实 Motus 配置的 1／2／4／8 卡吞吐、峰值显存、数据等待和 GPU-hours；保持可比较的全局 batch。不要默认一个作业独占八卡，也不要将总显存视为单卡显存。

- 评估／数据准备按预定分片并行，分片独立写入，先测 CPU／渲染／磁盘瓶颈。
- critic 单卡能训练时，可用 0—1 跑共享、2—6 分别跑五个独立 critic、7 做验证；随后分批补关键种子和容量控制。
- policy 阶段按实测选 4＋4 两组、四组双卡或单组八卡；不因改卡数而改变学习率、任务曝光或实验定义。

优先推进 P0 协议／数据、P1 原始能力评估及 P2 配对可行性审核；BC、critic、RL 分别通过其数据和诊断门槛后再开始。候选方法的规模确认与前置基线保持相同预算和选择规则。

## 本机已有证据与待解决问题

原始 P0 calibration、旧两任务 pilot、LIBERO 参考及早期实验统一见[既有实验摘要](previous-experiments.md)。本机 P0 已完成，但 container 的原生／CFG 差异、正式数据覆盖和远端环境验收仍待处理。现有结果不能直接作为 5880 已验收或新方法已有效的证据。

新产物放 `.runtime/runs/prerequisite/<run-id>/`。向[方法线](method-2gpu.md)交付锁定模型／数据／场景、baseline、critic 与标签、逐任务问题证据及不确定性和成本；接收候选时按[共同交接规范](multitask-critic.md#两线交接)核对代码、参数和标签版本。
