# VLA 多任务 RL 实验

在固定版本的 RLinf 上，从已有 Motus 多任务 SFT π0.5 出发研究 critic 与共享 policy。两条线并行，共用模型、数据与评测协议。

| 服务器 | 职责 | 任务文档 |
| --- | --- | --- |
| 本机 2×约 48GB RTX 4090 | 2—3 任务方法探索：M0—M4，检查 critic、继续 SFT＋value warm-up 与小规模 RL | [两卡方法探索](docs/method-2gpu.md) |
| 8×48GB RTX 5880 | 前置 P0—P5：初始能力、冻结数据、critic／BC／RECAP-style 基线及任务相互影响 | [八卡前置验证](docs/prerequisite-8gpu.md) |

[文档索引](docs/README.md) · [共同研究协议与交接](docs/multitask-critic.md) · [双线规划](configs/multitask/two_track.json)

## 共同起点

- RLinf `e1a82eb`、RoboTwin `dca9ec6`，完整 revision 与归档 SHA256 见[源码锁](bootstrap/sources.lock.json)。RLinf 的兼容补丁单列在 [patches](bootstrap/patches/recap-robotwin.json)。
- 模型 `motus-robotics/pi0.5_robotwin2`，revision `effc6e1a99bd4d0edf9447e8ba313def67649cfe`；权重、统计和动作设置见 [phase0.json](configs/multitask/phase0.json)。
- SFT 使用可靠成功数据，value 可用成功／失败；数据在训练前冻结，评估不回流训练，最终测试不参与方法选择。
- 原始 RECAP critic 用独立骨干与图像＋instruction；policy 特征、state、共享 adapter 属于另行控制的方法变化。

## 当前状态与入口

源码、基础环境、模型和 RoboTwin 资产已在本机准备；五任务 P0 校准已完成。原生／CFG 推理差异和正式数据覆盖仍需处理。现有两任务 pilot 只提供工程参考；联合 SFT／value 方法与八卡正式配置尚未实现／验收，当前没有新方法训练或远端队列。

本机只读核对：

```bash
source bootstrap/env.sh
python bootstrap/prepare_sources.py --verify-only
python scripts/evaluate_multitask.py status --output .runtime/runs/multitask-p0-20261111
```

新机器从[安装说明](docs/setup.md)开始，再按所属服务器任务书验收。已有状态目录不随 Git 分发。通用官方入口是 `scripts/run.py`，须显式提供 Hydra YAML；新方法按[开发约定](docs/development.md)接入，不复制训练器。

## 目录

```text
configs/multitask/       当前协议与双线规划
configs/robotwin_pilot/  旧两任务四阶段配置
configs/recap_official/  独立 LIBERO 参考配置
scripts/                通用官方入口、P0 校准和独立参考流程
src/vla_rl_mt/          研究方法扩展，按需实现
bootstrap/              环境、资源、数据与验收工具
docs/                  任务书、共同协议与使用说明
tests/                 入口、数据和项目扩展验证
```

固定版本源码随仓库保存在 [third_party/](third_party/README.md)，成员哈希与放行规则排除额外资产／缓存；独立 LIBERO 源码副本由脚本恢复。模型、数据及运行产物仍在 Git 忽略的 `models/`、`datasets/`、`.runtime/`。新实验输出分别使用 `.runtime/runs/method/<run-id>/` 和 `.runtime/runs/prerequisite/<run-id>/`。

## 独立参考与历史

[既有实验摘要](docs/previous-experiments.md) · [文献与研究想法](.research/README.md) · [第三方来源](THIRD_PARTY.md)

旧实验只在摘要中保留结论，逐次记录和图表不再随仓库维护。模型、数据、现有 checkpoint 与可复用代码保留。现有 [Docker 镜像](docs/docker.md)验收范围是 LIBERO，尚不能代表 RoboTwin GPU 环境已封装通过。配置分类与官方入口见 [configs/README.md](configs/README.md)。
