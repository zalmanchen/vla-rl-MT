# RLinf / RoboTwin / π0.5 环境

项目目录：`/storage/p1/cx/vla-rl-MT`。安装与下载入口集中在本目录。

## 使用

在项目根目录执行：

```bash
source bootstrap/env.sh
python bootstrap/smoke_model.py
python bootstrap/smoke_episode.py
```

`smoke_episode.py` 使用 RLinf 的原生 π0.5 加载器和动作变换，在 RoboTwin `handover_block` 上运行单个 ALOHA-AgileX 双臂 episode，种子 100000，任务默认步数上限。结果和视频位于 `.runtime/validation/`。该脚本只做集成验收，不是正式成功率评测；RL训练与Ray/FSDP链路由 `experiments/` 中的入口运行。

该机器上 SAPIEN 的 OIDN 报 invalid handle，验收入口将去噪器设为 none，保留 ray-tracing 渲染。此设置只用于该入口；调用其他入口时需明确应用同样的兼容处理。规划器使用 mplib；未安装可选 cuRobo / PyTorch3D，不启用点云观测。

## 目录与固定版本

- `third_party/RLinf/`：从官方 GitHub 下载并核对过的完整源码。
- `third_party/RoboTwin/`：固定提交的完整源码；`assets/` 已解压。
- `bootstrap/sources.lock.json`：源码提交、来源与压缩包 SHA-256。
- `.runtime/venv-rlinf/`：Python 3.11.15、PyTorch 2.7.1+cu128、OpenPI、RoboTwin 依赖。
- `bootstrap/environment.freeze.txt`：实际环境固定版本。选择匹配 OpenPI/LeRobot 的依赖组合；CUDA GPU 运算需通过验收。
- `models/pi0.5_robotwin2/`：`motus-robotics/pi0.5_robotwin2`，修订 `effc6e1a99bd4d0edf9447e8ba313def67649cfe`。包括权重及训练归一化统计，不下载 optimizer.pt。
- `datasets/RoboTwin2.0/`：官方数据修订 `3dc3b798668feb99ac61cc9086d84cbcc3d79186`，50 任务 Clean 压缩包与三份仿真资产。
- `datasets/RoboTwin2.0-clean/`：解压的 Clean 示范，保留压缩包内部目录。
- `datasets/RoboTwin2.0-randomized/`：50 任务 `aloha-agilex_randomized_500.zip` 压缩包；尚未批量解压。

每个下载目录包含 `download-manifest.json` 和 `DOWNLOAD_COMPLETE`。`bootstrap/verify_downloads.py` 检查全部选中文件的内容哈希是否匹配 Hugging Face ETag，并输出 `.runtime/validation/downloads.json`。

当前下载是原始 RoboTwin HDF5 数据，原始文件仍为HDF5；E4等阶段另外生成自己的数据清单与划分；灵巧手 embodiment 尚未配置。基础模型为 14 维 ALOHA 双臂关节/夹爪策略。

## Git 管理

仅项目根目录有 Git 元数据；上游采用固定版本源码快照，未创建嵌套仓库。`third_party/`、`.runtime/`、`models/`、`datasets/` 通过根目录 `.gitignore` 排除，避免将资产和环境加入 Git。源码来源及验收入口由 `bootstrap/` 记录。后续上游修改应导出为单独补丁并记录到本项目，或明确迁移到单一 RLinf fork；当前上游原文件未修改，核对记录见 `bootstrap/source-audit.json`；适配通过 `experiments/` 中的运行时扩展实现。

## 复核

```bash
uv pip check --python .runtime/venv-rlinf/bin/python
.runtime/venv-tools/bin/python bootstrap/verify_downloads.py
```

日志均在 `.runtime/logs/`。下载工具依赖独立的 `.runtime/venv-tools`。下载入口使用 Hugging Face 镜像和固定 revision，并保留缓存以便续传。`bootstrap/downloads.lock.json` 保存文件清单；`download.py` 准备模型、Clean 数据与资产，`download_randomized.py` 准备 Randomized 数据。

`bash bootstrap/install.sh` 可按源码锁和环境锁重新准备 Python 依赖；系统需预装 CUDA 12.8 工具链、NVIDIA 驱动、Vulkan loader、ffmpeg 和基础图形库。本机已安装并验证这些系统依赖。该脚本不会自动下载或解压数据。

## 本机验收结果

已完成模型加载和 handover_block 单次 800 步闭环，动作有效；该次任务 `success=false`。关闭 OIDN 后运行结束，输出视频 25 帧。依赖检查 323 个包全部兼容。Clean 为 2,500 条、Randomized 为 25,000 条 HDF5 示范。完整性校验状态查看 `.runtime/validation/downloads.json`（完成后生成）或 `.runtime/logs/verify-downloads.log`；本机安装状态记录在 `.runtime/validation/setup-status.json`。
