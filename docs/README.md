# 文档入口

主线是多任务SFT→RL协调：让SFT提供更适合RL的起点，并改善多任务学习效率与能力保持。`two_track.json`仅作规划，不启动训练；两卡候选与八卡正式基线并行。

| 需要了解什么 | 入口 |
| --- | --- |
| 其他服务器从SFT执行到RL | [RoboTwin 2.0总YAML与shell入口](sft-to-recap.md) |
| 当前目标、A—D实验与M0／W0等编号 | [实验分类与下一步](experiment-guide.md) |
| 交给本机执行端的任务 | [两卡方法任务单](method-2gpu.md) |
| 交给八卡执行端的任务 | [八卡基线与确认任务单](prerequisite-8gpu.md) |
| 版本、数据构建、奖励、评测与共同控制 | [研究协议](multitask-critic.md) |
| 已做实验与实际成绩 | [唯一结果汇总](previous-experiments.md) |
| 新机器安装、root仿真与资源 | [安装](setup.md)、[仿真](simulation.md)、[LIBERO镜像](docker.md) |
| 代码入口、验证与存储规则 | [开发说明](development.md)、[脚本索引](../scripts/README.md)、[配置索引](../configs/README.md) |
| 一手文献与研究假设 | [研究资料](../.research/README.md) |

已有工程、warm-up、标签、BC／critic及梯度诊断不等于已验证新方法；八卡尚未连接，候选协调机制和正式交接预算仍需定义。旧实验大检查点已清理，结果记录保留，不能假定旧目录可恢复训练。新比较写新协议／目录，不重复启动旧complete队列。
