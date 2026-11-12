# 环境、模型与数据准备

在项目根目录执行。本机安装、下载、内容校验与解压已完成，自动验收状态为 `base_validated`。以下安装命令供新机器使用；恢复中断任务前先确认没有进程写入同一资源目录。

```bash
cat .runtime/bootstrap/recap-preparation.json
cat .runtime/validation/handover_block.json
```

## 新机器安装

系统要求：NVIDIA 驱动、CUDA toolkit、Vulkan loader／NVIDIA ICD、ffmpeg 和基础图形库。主环境锁定 Python 3.11.15、PyTorch 2.7.1+cu128；本机 toolkit 为 13.2，两张 GPU 的 PyTorch CUDA 运算已通过，新增需要编译的扩展仍需单独验收。不要根据 `nvidia-smi` 的 CUDA 字段推断已安装的 toolkit。

```bash
# uv 已在本项目 bootstrap 环境中安装
export PATH="$PWD/.runtime/venv-bootstrap/bin:$PATH"
bash bootstrap/install.sh
uv venv --python 3.11.15 .runtime/venv-tools
uv pip sync --python .runtime/venv-tools/bin/python bootstrap/tools.freeze.txt
```

新机器若没有 uv，可先用 `python -m venv .runtime/venv-bootstrap`，再用该环境的 pip 安装 uv。`install.sh` 负责固定源码及 Python 依赖，不安装系统包。

源码必须使用 RLinf `e1a82eb`、RoboTwin `dca9ec6`，完整提交与压缩包 SHA256 见 `bootstrap/sources.lock.json`。两份源码已随仓库提供，成员清单在 `third_party/sources.manifest.json`；新 clone 可用 `python bootstrap/prepare_sources.py --verify-only` 离线核验。存在缓存 ZIP 时还会核验归档，RLinf 仅接受记录在案的兼容补丁。源码目录没有独立 Git 元数据，目录内的 Git HEAD 可能属于项目父仓库。

主环境与下载工具分别位于 `.runtime/venv-rlinf`、`.runtime/venv-tools`。`env.sh` 支持从外部设置 `CUDA_HOME`、`HF_ENDPOINT`、`VK_ICD_FILENAMES`，并设置项目、RLinf、RoboTwin 与 `src/` 的导入路径。

## 模型与数据

| 资源 | 本地目录 | 版本锁 |
| --- | --- | --- |
| Motus 多任务 SFT π0.5 | `models/pi0.5_robotwin2/` | `bootstrap/downloads.lock.json` |
| Gemma3-270M、SigLIP2 | `models/gemma-3-270m/`、`models/siglip2-so400m-patch14-224/` | `bootstrap/recap-models.lock.json` |
| Clean 示范及仿真资产 | `datasets/RoboTwin2.0/` | `bootstrap/downloads.lock.json` |
| RLinf、RoboTwin 源码 | `third_party/` | `bootstrap/sources.lock.json` |

策略约 6.96 GiB，critic 骨干合计约 4.80 GiB，Clean 压缩包和资产约 35.67 GiB；还需预留解压、训练及检查点空间。critic 骨干不是训练好的价值模型。

新机器下载：

```bash
# 仅下载策略，不同时启动基础数据下载
.runtime/venv-tools/bin/python bootstrap/download.py --only model

# Gemma 需要已授权的账户；token 只在终端输入
.runtime/venv-tools/bin/hf auth login
.runtime/venv-tools/bin/python bootstrap/download_recap_models.py

# 8 MiB 分块、8 路并行；复用已有 Hub 缓存，逐文件校验内容哈希
.runtime/venv-tools/bin/python bootstrap/download_dataset_ranges.py
.runtime/venv-tools/bin/python bootstrap/verify_downloads.py
.runtime/venv-tools/bin/python bootstrap/extract_assets.py
```

`download.py` 默认同时下载策略和基础数据，是普通 Hub 下载备选；不要与 `download_dataset_ranges.py` 同时写同一数据目录。Randomized 数据是可选扩展，单独用 `download_randomized.py` 下载，校验时加 `--include-randomized`。

本机单次测速官方站经代理约 1.37 MiB/s，镜像经代理约 0.71 MiB/s；不能假设国内镜像在本机更快。分块下载器支持 `--endpoint`，公开数据请求不发送账户 token。大、小写代理变量当前指向不同入口，具体选择取决于工具。

## 运行验收

```bash
source bootstrap/env.sh
.runtime/venv-bootstrap/bin/uv pip check --python .runtime/venv-rlinf/bin/python
python bootstrap/smoke_render.py
python bootstrap/smoke_model.py
python bootstrap/smoke_episode.py
```

主环境 323 个包通过依赖检查；root 下 SAPIEN 物理与真实 RGB 出图、官方 RLinf 策略加载、RoboTwin `handover_block` 800 步闭环均已通过运行检查，见 [仿真说明](simulation.md)。该次任务未成功，集成检查不能替代成功率评估或 critic 梯度验收。

`finish_recap_setup.py` 等待已有主环境和下载完成后自动进行基础校验、解压及 smoke 检查，不启动训练。其 `base_validated` 状态也不代表 critic 或官方 RECAP 数据适配完成。

主环境已执行官方 critic 的加载、反向传播及保存，RoboTwin → LeRobot 与 ALOHA 数据路径已接通；CFG 检查点可复载并闭环运行。既有实验范围见 [实验摘要](previous-experiments.md)，不沿用旧 `.runtime/venv-pace` 的安装结论。接口可运行不代表 critic 标签已可靠或策略成功率提高。

## PyTorch wheel 长连接失败

若安装日志反复停在 `Downloading torch`，先确认旧安装进程退出，再用下载工具环境获取相同锁定版本的本地 wheel：

```bash
.runtime/venv-tools/bin/python bootstrap/download_wheel.py
uv pip install --python .runtime/venv-rlinf/bin/python --no-deps \
  .runtime/wheels/torch-2.7.1+cu128-cp311-cp311-manylinux_2_28_x86_64.whl
bash bootstrap/install.sh
```

下载器从官方 PyTorch 索引取 SHA-256，按 8 MiB 分块续传，校验通过才生成最终 `.whl`。安装阶段使用本地同版本 wheel，不改环境锁。本次修复已完成，记录在 `.runtime/logs/torch-wheel-download.log` 与 `.runtime/logs/recap-install.log`。

其他大 wheel 可使用同一工具，例如 `--package sapien`。工具只接受环境锁中已有的包，按当前 Python 平台选择官方 PyPI wheel 并核对 SHA-256；逐个本地安装完成后再恢复 `install.sh`。本机剩余大包修复日志为 `.runtime/logs/dependency-wheel-download.log`。国内镜像对某些固定 wheel 返回 403 时，应保持官方版本并换下载方式。

若包实际上已在本机 uv 缓存中，可先以 `uv pip install --offline --no-deps` 和原镜像索引尝试安装环境锁中的精确版本；命中后不需要重新下载。本次剩余 11 个大包最终通过缓存复用完成，随后恢复固定依赖同步。不要在同一虚拟环境上并发运行多个安装器。
