# VLA 多任务后训练实验

在固定版本的RLinf上，从已有Motus多任务SFT π0.5出发研究共享policy的多任务后训练。两条线并行，共用模型、数据与评测协议。

研究目标是协调多任务SFT与RL后训练，缓解各阶段的有害任务干扰，让SFT提供更适合RL继续学习的初始化，在可比预算下加快学习并保留已有能力。普通SFT→RECAP是完整基线，分开验证SFT准备、RL协调及其组合；critic warm-up是可选组件。整体设计见[研究目标与交接实验](docs/experiment-guide.md#研究目标与后续整体设计)。

| 服务器 | 职责 | 任务文档 |
| --- | --- | --- |
| 本机 2×约 48GB RTX 4090 | 2—3任务开发：数据／信号／policy因果诊断，筛选一个候选机制 | [两卡方法探索](docs/method-2gpu.md) |
| 8×48GB RTX 5880 | 正式多任务基线、多种子、任务规模与未更新能力保持；当前仅规划 | [八卡前置验证](docs/prerequisite-8gpu.md) |

[实验分类与下一步任务](docs/experiment-guide.md) · [文档索引](docs/README.md) · [共同研究协议与交接](docs/multitask-critic.md) · [双线规划](configs/multitask/two_track.json)

其他服务器从SFT执行到RL，使用[RoboTwin 2.0启动说明](docs/sft-to-recap.md)、[总YAML](configs/pipelines/robotwin2_sft_recap.yaml)和[shell入口](scripts/run_sft_recap.sh)。流水线连接官方SFT→格式交接→returns→value→advantages→CFG policy及配对开发评测；默认短预算，须先准备合格的逐控制步数据。

历史可重跑的大检查点／优化器已按用户要求清理，保留原始模型、数据、环境及少量基线权重；保存规则见[空间管理](docs/development.md#空间管理)。旧实验需要恢复训练时重新运行。

## 共同起点

- RLinf `e1a82eb`、RoboTwin `dca9ec6`，完整 revision 与归档 SHA256 见[源码锁](bootstrap/sources.lock.json)。RLinf 的兼容补丁单列在 [patches](bootstrap/patches/recap-robotwin.json)。
- 模型 `motus-robotics/pi0.5_robotwin2`，revision `effc6e1a99bd4d0edf9447e8ba313def67649cfe`；权重、统计和动作设置见 [phase0.json](configs/multitask/phase0.json)。
- SFT 使用可靠成功数据，value 可用成功／失败；数据在训练前冻结，评估不回流训练，最终测试不参与方法选择。
- 原始 RECAP critic 用独立骨干与图像＋instruction；policy 特征、state、共享 adapter 属于另行控制的方法变化。

## 当前状态与入口

本机工程、数据与数值审计、warm-up、critic及标签对照、单任务／联合BC、BC三种子和9组policy梯度对照均已完成。现有两任务开发证据尚未建立共享critic瓶颈、联合额外退化或稳定的梯度干预收益，成绩只维护在[实验汇总](docs/previous-experiments.md#当前实验总览与待办)。

当前90项测试及真实采集小测通过；GPU0／1已启动新逐控制步数据准备队列，完整SFT／RL训练待数据导出验收。主线已明确为SFT→RL协调，下一步封存完整交接基线与学习效率指标，再分别验证两阶段作用，见[交接设计](docs/experiment-guide.md#sft到rl的交接与最小实验)。本机native研究适配与官方完整RECAP基线分别报告；正式多任务数据、能力保持和八卡确认仍待完成。

本机只读核对：

```bash
source bootstrap/env.sh
python bootstrap/prepare_sources.py --verify-only
python scripts/evaluate_multitask.py status --output .runtime/runs/multitask-p0-20261111
```

新机器从[安装说明](docs/setup.md)开始，再按所属服务器任务书验收。已有状态目录不随 Git 分发。通用官方入口是 `scripts/run.py`，须显式提供 Hydra YAML；新方法按[开发约定](docs/development.md)接入，不复制训练器。

## 目录

```text
configs/pipelines/      RoboTwin 2.0 SFT→RECAP总配置
configs/multitask/       当前协议与双线规划
configs/robotwin_pilot/  旧两任务四阶段配置
configs/recap_official/  独立 LIBERO 参考配置
scripts/                官方入口、研究准备／诊断／队列与汇总工具
src/vla_rl_mt/          研究方法扩展，按需实现
bootstrap/              环境、资源、数据与验收工具
docs/                  任务书、共同协议与使用说明
tests/                 入口、数据和项目扩展验证
```

固定版本源码随仓库保存在 [third_party/](third_party/README.md)，成员哈希与放行规则排除额外资产／缓存；独立 LIBERO 源码副本由脚本恢复。模型、数据及运行产物仍在 Git 忽略的 `models/`、`datasets/`、`.runtime/`。新实验输出分别使用 `.runtime/runs/method/<run-id>/` 和 `.runtime/runs/prerequisite/<run-id>/`。

## 独立参考与历史

[既有实验摘要](docs/previous-experiments.md) · [文献与研究想法](.research/README.md) · [第三方来源](THIRD_PARTY.md)

旧实验只在摘要中保留结论，逐次记录和图表不再随仓库维护。原始模型、数据和可复用代码保留，历史大检查点已按空间规则清理。现有 [Docker 镜像](docs/docker.md)验收范围是 LIBERO，尚不能代表 RoboTwin GPU 环境已封装通过。配置分类与官方入口见[配置索引](configs/README.md)和[脚本索引](scripts/README.md)。
