# 两卡4090：SFT→RL方法验证任务单

更新：2026-11-14。可直接转发给本机执行端；本轮启动了数据准备，尚未开始actor训练。主线是协调多任务SFT与RL，使SFT提供更适合RL继续学习的初始化，并减少各阶段的有害任务干扰。critic是可选组件。

共同起点为已有Motus多任务π0.5，固定RLinf `e1a82eb`、RoboTwin `dca9ec6`，完整版本与动作接口见[共同协议](multitask-critic.md)。不重新构建基础模型，不把PCGrad或GRPO算法名预设为方法贡献。

普通基线的[总YAML与shell入口](sft-to-recap.md)已提供，固定RoboTwin 2.0。它要求另行准备合格逐控制步数据；本机chunk边界母池用于native开发，不能直接混用。

## 第一批执行顺序

当前实际队列先完成下表，既有PCGrad、critic选层和目标配对不重复启动。采集协议为[sft_recap_data.json](../configs/multitask/sft_recap_data.json)，原始数据输出`.runtime/datasets/sft-recap-control-v1/`；`native-index.json`只引用旧池，不复制图像。它已包含64条成功SFT索引、192条固定成败经验和原24条开发轨迹，不能当作官方逐控制步数据。

| 本机工作 | 预算与要回答的问题 | 执行条件 |
| --- | --- | --- |
| 真实逐控制步数据补齐 | GPU0瓶子／GPU1按铃；先4条工程采集检查，再按固定清单采192条训练＋24条critic验证 | 训练seed列表索引[200,296)，新critic验证[120,132)；开发[300,320)、最终[500,550)隔离。全失败／setup error不混淆，不按成功数补采 |
| 官方入口工程检查 | 合格LeRobot数据后，完整SFT→格式交接→value→优势→CFG policy各2步，并做配对开发rollout | 检查完整模型实际梯度、显存、动作接口、末点加载和阶段续接；四条原始采集检查本身不等于此项通过 |
| 普通SFT→RECAP开发基线 | seed0、两个任务；先SFT128、value128、policy128，固定末点 | 普通基线先跑通，短预算只作方法开发，不声称正式复现或解决多任务冲突 |
| SFT准备与RL学习效率 | 原Motus直接RL、普通SFT→RL及等总成本参照；预定RL0／32／64／128评测 | 先补齐固定点评测与中间权重清理，匹配critic准备、数据与部署；更强handoff不等于更快RL |
| 一个协调机制与两阶段归因 | 候选定义后才排A—D，必要时补actor种子 | 优先有具体预测、可扩展的最小因素；不把“协调”当尚未实现的配置名 |

原始HDF5保存每次环境动作前的三相机和状态、真实最终观测、请求chunk、实际执行命令和mask。`fps=15`是兼容LeRobot索引的名义频率，不声称物理仿真／墙钟频率恒定。后续导出需采用能共享图像／视频的视图，不能复制三套图像或将旧chunk图像插值。采集器不运行optimizer；训练仍由[RoboTwin 2.0官方阶段入口](sft-to-recap.md)负责。

标准RECAP的critic是独立骨干，普通policy SFT不会自动预热它。要检验“SFT特征帮助critic”，需另立固定输入／容量／预算的特征迁移比较；若引入共享value梯度，再与stop-grad配对，不能把简单的并行运行时间差当作表征收益。

开发短程的128步不是正式RECAP复现预算，也不用于宣称性能结论。预先固定少量RL评测点，例如0／32／64／128，报告完整曲线，不选最好的点冒充末点；中间权重评测后清理。各组可达到相同末点与预算才进入比较。

## 候选确定后的共同四组

| 组 | SFT | RL | 目的 |
| --- | --- | --- | --- |
| A | 普通多任务SFT | 标准多任务RECAP | 完整基线 |
| B | 协调SFT | 与A相同的标准RL | 隔离SFT准备作用 |
| C | 与A完全相同的SFT actor | 协调RL | 隔离RL更新作用 |
| D | 与B完全相同的SFT actor | 与C相同的协调RL | 检验两阶段组合 |

若机制包含critic warm-up，再做同actor的提前准备／事后标准训练配对，匹配总value预算；标准准备包括必要的critic训练，不用随机critic立即RL构造弱参考。共享value梯度另配stop-grad，历史W2没有建立actor收益。若保留BC重放，增加固定混合／固定衰减控制，排除“只是多用了成功数据”。

## GPU安排和可选项

当前数据准备阶段GPU0／1分别负责一个任务；之后官方完整SFT／CFG配置可按RLinf placement共同占用两卡，先做真实显存验收。native方法探索则GPU0优先普通参考、GPU1匹配候选，仍为两个独立单卡NO_SHARD作业，不默认支持分布式。任务配额、loss分母和scope变化均记入协议。

GRPO是备选的第二RL算法检验，先不与主机制同时切换。需要时先验收Flow-SDE／logprob／同初态reset，再做2任务×8训练初态×4噪声=64回合的冻结actor信号检查，统计全成功／全失败组及成本；当前192条单次轨迹不能直接凑on-policy组。144次任务loss移除也是按需诊断，旧optimizer已清理，运行前需在新目录生成状态。

## 给八卡线的交付

- 模型／源码／数据／norm哈希，完整指令与场景划分，以及相对数据根路径。
- 普通参考和唯一候选的代码、输入、loss、实际梯度范围、采样、预算、部署设置。
- SFT末点逐任务能力，RL固定点曲线、gain/loss场景、保持和训练成本；critic指标仅对应有critic分支。
- 明确何种中间变化得到支持、哪些未成立；两任务开发不能推广为50任务结论。

## 已完成的四项工作

历史有限选层、完整W0、native桥接和固定标签对照，以及后续BC／critic信号／梯度诊断均已完成；具体成绩只维护在[结果汇总](previous-experiments.md)。它们未证明SFT改善VLM本体、共享critic瓶颈或可靠的PCGrad收益。无需为了启动本批恢复全部历史大实验。

## 存储与执行边界

历史大DCP／optimizer已按用户要求删除，保留内容见`.runtime/storage/cleanup-report-v3.json`；保留的普通BC／加权FM末点只可评测或初始化，不能假定能精确恢复optimizer。新运行写全新目录，不能重启旧complete controller。

新数据只建索引视图，默认只保留可评测末点，交接研究确需的optimizer预先列明。启动前估算峰值后仍至少100GiB可用；smoke大产物完成核对即清理，不长期压缩归档。现有VLM／数据／环境和原始模型不删。

相关入口见[开发说明](development.md)、[整体交接实验](experiment-guide.md#sft到rl的交接与最小实验)。
