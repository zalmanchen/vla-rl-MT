# RoboTwin VLA Multi-Task Research

基于固定版本 **RLinf + RoboTwin 2.0** 的可复现实验入口。当前支持原生 π₀ 的多任务/单任务成功示范 BC、
任务均衡采样、checkpoint 和逐 episode 闭环评价。多任务离线 RL / RECAP 尚未实现，当前 BC 用作后续研究参照。

## 新服务器快速开始

支持 Linux x86_64、NVIDIA GPU。需要管理员预先安装支持 CUDA 12.6 的驱动和 Vulkan/EGL 系统库。
建议 80GB GPU 用于训练；小显存配置需在目标服务器验证。源码可放在任意目录。

本分支的 GitHub 源码 ZIP 已打包固定版本的 RLinf 和 RoboTwin 源码压缩包。解压后执行
`bash ./setup.sh --verify-sources`，脚本会在两个子模块目录均为空时自动准备并校验源码。
详见 [ZIP 安装步骤](docs/INSTALL.md)。Python、依赖包、模型和数据仍需可用的下载源或预先准备。

```bash
git clone --recurse-submodules https://github.com/zalmanchen/vla-rl-MT.git
cd vla-rl-MT

# 1. 自动创建 Python 3.11.18 环境，安装固定版本 OpenPI / PyTorch 和 SAPIEN/mplib
./setup.sh

# 2. 下载权重、tokenizer、两任务原始示范、仿真资产；解压、校验、冻结数据划分与统计量
./download.sh

# 3. CPU 检查数据完整性和原生动作变换，不加载 VLA 权重
./check.sh --deep

# 4. 快速验收：20 步、多任务共享模型
./train.sh --smoke --plan
./train.sh --smoke --gpu 0 --detach
./run.sh status --run latest

# 训练完成后，评价初始模型和已训练模型，每任务 2 episodes
./evaluate.sh --smoke --gpu 0 --detach
```

`--detach` 返回控制器 PID 和日志文件；控制器日志在 `.runtime/controllers/`。
训练和评价是两个阶段：确认训练完成后再运行评价。快速验收用于检查链路，不能作为正式性能结论。
安装和 CPU 检查不会启动训练，`--plan` 不查询或分配 GPU。

首次下载约包括 **13GB 权重、0.84GB 两任务压缩示范、15GB 仿真资产**，解压和 checkpoint 需要额外空间。
建议预留至少 150GB 用于快速验收，正式多组实验需要更多空间。资产下载与后续缓存会保留以便恢复。

## 正式 MT / ST 配对实验

```bash
# 默认 adjust_bottle + handover_block；每个模型 1000 步
# 配对顺序：一个 MT 模型，然后每个任务一个 ST 模型；每任务每步均 8 个窗口
./train.sh --mode paired --gpu 0 --seed 0 --detach
./run.sh status --run latest
# 完成后，每任务 20 episodes
./evaluate.sh --gpu 0 --episodes 20 --detach

# 指定输出位置，便于三种子复验
./train.sh --mode paired --gpu 0 --seed 1 --run .runtime/experiments/seed1 --detach
./evaluate.sh --run .runtime/experiments/seed1 --gpu 0 --episodes 20 --detach
```

当前每个模型使用一张 GPU，配对任务串行执行。同一物理卡有项目锁，避免本仓库重复派发；
它不会停止其他用户的进程。多个 GPU 可以分别启动不同种子的实验，使用不同 `--run` 目录。

每个 run 保存 `contract.json`、共同归一化文件、逐组训练配置、原始日志、指标和最终 checkpoint。
`status.json` 记录完成/失败；组内 `train_metrics.jsonl` 记录实际 loss、梯度、LR 和任务曝光。
评价输出在 `eval/<policy>/<task>/seed_*.json` 与 `eval/summary.json`；运行错误单列，不算作策略失败。

## 数据和方法约定

- 默认是 ALOHA-AgileX 的两任务成功示范；可下载的冻结任务清单在 `resources/releases.json`。
- 读取原始 HDF5 的三相机 JPEG 和 `joint_action/vector[T,14]`。
- BC 目标是从当前观测预测**后续 50 个示范关节位置**，不是将状态字段误称为额外记录的控制命令。
  不跨 episode 构造窗口。原生 OpenPI 将 12 个手臂关节转换为相对动作，两个夹爪保持绝对值。
- 按 episode 内容哈希分组，固定 80/20 训练/验证划分；仅用训练部分拟合共同归一化。
  MT/ST 采用相同统计量和相同每任务曝光。当前训练不做闭环验证择优，评价使用固定预算最后 checkpoint。
- 使用 `RLinf-Pi0-NEW-RoboTwin-SFT-adjust_bottle`，匹配当前 RLinf 的原生模型格式。
  这是已在 adjust_bottle 上 SFT 的起点，不能宣称任务未见或无预训练覆盖；B0 保留其原始归一化，
  训练后的 MT/ST 使用共同的冻结训练统计量。
- 冻结 VLM、更新原生动作专家。当前不是混合质量离线 RL；失败/恢复数据和 RECAP 将作为后续独立协议。
- 评价使用 RLinf 公开种子列表并记录请求 seed；不宣称与训练物理场景完全隔离。
  当前短任务名同时作为训练/评价 prompt，不代表官方完整语言泛化协议或排行榜成绩。

## 常用入口

```bash
./setup.sh --dry-run
./setup.sh --train-only                      # 只训练依赖，暂不安装仿真包
./download.sh --component model
./download.sh --component data --tasks adjust_bottle handover_block
./download.sh --component assets
./download.sh --component tokenizer

# 扩展任务：首次准备时选择；改变冻结任务集需使用新的 manifest 文件
./download.sh --component data --tasks adjust_bottle handover_block lift_pot --no-prepare
.runtime/envs/robotwin/bin/python scripts/prepare_data.py \
  --tasks adjust_bottle handover_block lift_pot --output .data/robotwin/three_tasks.json
./train.sh --manifest .data/robotwin/three_tasks.json --tasks adjust_bottle handover_block lift_pot --plan

# 可选灵巧手数据，仅下载；当前未接入训练
./download.sh --component dexmimicgen

# 恢复同一 run 的单个训练组，预算/源码/数据需匹配
./train.sh --run /absolute/path/to/run --resume-checkpoint /absolute/path/to/global_step_N
```

完整 [安装与系统依赖](docs/INSTALL.md) · [模型/数据来源](docs/ASSETS.md) · [第三方来源](THIRD_PARTY.md)。
当前已执行的检查与范围见 [发布验证记录](docs/VALIDATION.md)。
全部生成内容在仓库内的 `.runtime/`、`.model/`、`.data/` 和 `RoboTwin/assets/`，不上传权重或轨迹。
可用 `VLA_ENV=/absolute/path/to/venv` 指定已有环境。

## 验证

```bash
.runtime/envs/robotwin/bin/python -m unittest discover -s tests -v
bash -n setup.sh download.sh train.sh evaluate.sh check.sh run.sh scripts/env.sh scripts/setup_env.sh
```

CPU 检查验证依赖、数据读取、预算和动作变换；实际 GPU 显存、SAPIEN Vulkan 渲染和模型闭环必须在目标服务器验收。
RoboTwin 官方记录了部分 A/H/V 系列卡的采集/渲染兼容性问题，见
[官方说明](https://robotwin-platform.github.io/doc/common-issue/index.html)。可将训练和仿真分别安排到合适的服务器。
