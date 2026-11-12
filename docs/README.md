# 文档入口

当前研究从同一个 Motus 多任务 SFT π0.5 出发，分两条线推进。阶段清单以 [two_track.json](../configs/multitask/two_track.json) 为准；它是规划，不会启动实验。

| 阅读顺序 | 文档 | 用途 |
| --- | --- | --- |
| 1 | [两卡方法探索](method-2gpu.md) | 本机 M0—M4、最小控制、交付与扩大条件 |
| 1 | [八卡前置验证](prerequisite-8gpu.md) | 5880 的 P0—P5、资源安排、基线与问题证据 |
| 2 | [共同研究协议](multitask-critic.md) | 模型／源码／数据／评测、warm-up 与 actor×标签控制、两线交接 |
| 按需 | [安装与资源](setup.md)、[仿真验收](simulation.md) | 新机器准备、root／无显示器环境 |
| 按需 | [开发约定](development.md)、[配置索引](../configs/README.md) | 官方入口与项目扩展、配置边界 |

## 已有实现与待实现阶段

- 已有：源码核验、环境与资源准备、数据审计、五任务 P0 校准工具、通用 RLinf 入口。
- 待实现／锁定：方法线共享 adapter／value 接口，正式数据清单与训练预算，八卡任务配置与 instruction 配对检查。
- 当前没有 M0—M4／前置 P0—P5 的自动队列；任务文档中的阶段不是已经完成的实验。

## 既有实验与参考工具

既有实验统一见[一份摘要](previous-experiments.md)，旧逐次记录、图表和归档代码不再随仓库维护。RoboTwin pilot／LIBERO 的可复用配置与工具仍在原分类目录中，范围由摘要说明，不作为两条线的默认队列。

[Docker 使用](docker.md) · [文献与研究想法](../.research/README.md)
