# VLA 多任务后训练实验

基于 **RLinf + RoboTwin 2.0 + 已完成多任务 SFT 的 π0.5**，研究一个策略如何同时提升多个操作技能，并保持已有能力。当前主要比较原始模型、单任务后训练和共享多任务后训练；后续考虑示范重放、策略保持约束以及 SFT＋RL 联合后训练。

当前环境为 **ALOHA-AgileX 双臂、14维关节／夹爪动作**。灵巧手实机与长程连续任务属于后续验证，尚未纳入已完成实验。

## 先看哪里

- [实验结果总览（Markdown）](.research/results-summary.md)：GitHub可直接阅读。
- [逐任务结果CSV](.research/results-summary.csv)：用于统计和画图；未完成评估留空。
- [五任务曲线图](.research/figures/e7-learning-curves.png)、[E7阶段分析图](.research/results/e7/e7-analysis.png)。
- [训练与排障经验](.research/takeaway.md)：环境兼容、概率审计、内存管理和恢复经验。
- [实验入口](experiments/README.md)、[环境说明](bootstrap/README.md)、[研究规划](.research/planning/)。

交互查看时下载或在本地打开 [progress.html](.research/progress.html) 和 [five-task-curves.html](.research/five-task-curves.html)。GitHub仓库中的报告是提交时快照；实验服务器上的报告由控制器持续更新。部分历史报告指向未上传的 `.runtime/` 原始证据，只能在原实验机查看。

## 模型、框架与数据

| 项目 | 固定来源 |
| --- | --- |
| 初始策略 | [motus-robotics/pi0.5_robotwin2](https://huggingface.co/motus-robotics/pi0.5_robotwin2)，revision `effc6e1a99bd4d0edf9447e8ba313def67649cfe` |
| RL框架 | [RLinf](https://github.com/RLinf/RLinf)，commit `e1a82ebf0c1af69da6ab9b1519f5f3692fe33566` |
| 仿真环境 | [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin)，commit `dca9ec682688821c42944eacd7f5dd9bfb15396f` |
| 数据集 | [TianxingChen/RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0)，revision `3dc3b798668feb99ac61cc9086d84cbcc3d79186` |
| 主环境 | Python 3.11.15、PyTorch 2.7.1＋CUDA 12.8；完整版本见 [环境锁](bootstrap/environment.freeze.txt) |

初始模型已经做过多任务SFT。本项目的“原始模型”指这一发布权重，不是未微调的通用π0.5，也不是Motus论文中其他模型的成绩。

上游采用固定源码快照，避免嵌套Git仓库。来源与SHA-256见 [sources.lock.json](bootstrap/sources.lock.json)；[源码核对](bootstrap/source-audit.json)确认当前RLinf与RoboTwin原文件未修改。适配与实验扩展保存在 `experiments/`。

## 已得到的结果

不同实验的任务集合、预算和适配程度不同，下表不能直接用作算法排名。

| 实验 | 范围 | 已验证结果 |
| --- | --- | --- |
| E1 原始策略 | 22任务，各100场景 | **566/2200＝25.73%**，本地Clean协议 |
| E3 执行窗口 | 3任务，H＝8／16／32，各20场景 | 窗口效果依赖任务，短窗口并非总是更好 |
| E4 离线RECAP短预算适配 | 5任务，相同100场景 | 原始47%；单任务29%；共享29%；成功重放25% |
| E5 旧单任务PPO | 调整瓶子，相同20场景 | 原始9次成功；第10／20／100轮为7／2／1次 |
| E6 稳定性诊断 | 调整瓶子，相同20场景 | 原始9次；第5轮11次；第10轮9次 |
| E7 五任务曲线 | 5个独立策略与1个共享策略 | 共享第5／10／20／50轮为48%／47%／41%／32%；单任务对照仍在推进 |

E4使用普通Motus初始化与短预算设置，属于RECAP适配诊断，不是原论文完整复现。E7在第59轮因主机内存压力中断，修复后从完整第50轮检查点恢复；共享策略现已完成100轮训练，**第100轮成功率以实际评估完成为准**。恢复前丢弃的9轮更新单列为额外成本，不混入当前曲线。

目前观察到技能收益不均与已有能力丢失，尚不能归因为多任务特有的梯度冲突，也没有证明critic波动是成功率下降的原因。20个重复开发场景、1个训练种子用于前置诊断，论文阶段需独立评估场景和多个种子。

## 当前E7实验协议

任务：`adjust_bottle`、`click_bell`、`stack_blocks_two`、`place_empty_cup`、`place_container_plate`。

| 设置 | 取值 |
| --- | --- |
| 算法 | Flow-SDE＋PPO，严格FP32，每批1遍更新 |
| 可训练部分 | 动作专家和价值头；冻结VLM；无critic预热 |
| batch | microbatch 4，global batch 32 |
| 环境与预算 | 每任务2个环境，每轮每环境512个名义动作点，各100轮 |
| 策略预算 | 每任务102400；5个单任务合计与共享策略各512000个名义动作点 |
| 更新次数 | 共享每轮5次；单任务每轮1次；5个单任务合计匹配 |
| 评估 | 第0／5／10／20／50／100轮，每任务相同20个冻结Clean场景 |
| 推理 | 原始参考ODE采样器，32步预测／执行，10步去噪 |

交互预算指用于有效训练路径的名义动作点，另报实际执行量及中断成本。恢复模型和优化器不意味着环境／随机数逐步连续。

[2026-11-07快照](experiments/snapshots/2026-11-07/README.md)保存冻结计划、配置、源码、种子、恢复验收与状态；[清单](experiments/snapshots/2026-11-07/manifest.json)包含逐文件SHA-256。训练权重、优化器和轨迹另存本地。

## 环境准备与使用

已有实验服务器只需激活环境；**训练期间不要重新同步环境或覆盖运行计划**：

```bash
cd /storage/p1/cx/vla-rl-MT
source bootstrap/env.sh
python experiments/render_pirl_five_task.py --force
python experiments/render_progress.py
```

在新机器上准备基础依赖，需要先安装Git、uv、NVIDIA驱动、CUDA 12.8工具链、Vulkan、ffmpeg及基础图形库：

```bash
git clone https://github.com/zalmanchen/vla-rl-MT.git
cd vla-rl-MT
bash bootstrap/install.sh
uv venv --python 3.11.15 .runtime/venv-tools
uv pip sync --python .runtime/venv-tools/bin/python bootstrap/tools.freeze.txt

# 下载固定版本的模型、Clean示范和仿真资产，再解压
.runtime/venv-tools/bin/python bootstrap/download.py
.runtime/venv-tools/bin/python bootstrap/extract_assets.py
.runtime/venv-tools/bin/python bootstrap/verify_downloads.py

# 原始OpenPI参考评估使用独立的Transformers包目录
uv pip install --python .runtime/venv-rlinf/bin/python \
  --target .runtime/reference-transformers --no-deps \
  rlinf-transformer-openpi==4.53.2

source bootstrap/env.sh
python bootstrap/smoke_model.py
python bootstrap/smoke_episode.py
```

以上是基础环境与模型闭环验收，不会自动开始E7。训练入口依赖 `.runtime/experiments/` 中对应的冻结计划、配置和已有产物；新机器需按快照重定位绝对路径并准备所需阶段数据，当前不提供一条命令重跑全部历史实验的承诺。数据下载需要较大磁盘空间；可选RECAP价值模型的下载与授权说明见 `bootstrap/`。

在已完成准备且旧控制器退出的实验机，队列入口为：

```bash
.runtime/venv-tools/bin/python experiments/pirl_five_task_controller.py
```

控制器使用文件锁防止重复运行；Ray配置16个CPU资源与2 GiB对象存储。并行评估同时检查GPU显存及主机／容器内存，遇到致命worker错误及时停止。具体恢复协议见 [takeaway.md](.research/takeaway.md)。

## 目录与保存范围

```text
bootstrap/                环境安装、下载与来源锁
experiments/              训练、评估、调度、分析和检查脚本
experiments/snapshots/     轻量配置与结果快照
.research/                结果、图表、论文调研、规划和经验
third_party/              固定上游源码（本地，不上传）
models/                   模型权重（本地，不上传）
datasets/                 示范与资产（本地，不上传）
.runtime/                 环境、检查点、轨迹、日志和凭据（本地，不上传）
outputs/                  Hydra启动目录（仅保留目录说明）
```

本次提交保存当前工程结构，替代早期根目录BC脚本与子模块入口；旧版本仍保留在Git历史中。GitHub保存代码、轻量元数据和结果摘要，**不等于模型和数据的完整备份**。下载的论文PDF及全文副本不提交，保留来源链接和研究笔记。

## 检查与来源

```bash
.runtime/venv-tools/bin/python -m unittest discover \
  -s experiments -p test_pirl_runtime_guard.py -v
uv pip check --python .runtime/venv-rlinf/bin/python
```

论文与资源引用集中在 [文献来源](.research/literature/briefing-sources.json)和[现象核对](.research/literature/phenomena-paper-audit.html)。第三方代码、模型和数据分别遵循上游许可与访问条款，见 [THIRD_PARTY.md](THIRD_PARTY.md)。
