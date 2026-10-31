# 安装与运行

## 系统前提

Linux x86_64；NVIDIA 驱动需支持 CUDA 12.6；仿真需要 Vulkan、EGL/OpenGL 和可用的 NVIDIA 图形驱动。
训练推荐 80GB GPU。脚本不更改驱动，不使用 sudo。Ubuntu/Debian 可由管理员安装：

```bash
sudo apt-get update
sudo apt-get install -y git curl python3-venv build-essential cmake pkg-config \
  ffmpeg libegl1 libgl1 libglfw3 libglib2.0-0 libvulkan1 vulkan-tools
```

运行 `nvidia-smi`、`vulkaninfo --summary` 检查 GPU 驱动与 Vulkan 是否就绪；容器需传入 GPU 及图形驱动能力。
CPU 依赖检查通过不代表 Vulkan 渲染通过。

## 一键安装

```bash
git clone --recurse-submodules https://github.com/zalmanchen/vla-rl-MT.git
cd vla-rl-MT
./setup.sh
```

如果通过 GitHub ZIP 获取代码，主仓库 ZIP 不包含两个子模块的内容。请分别下载固定版本
[RLinf ZIP](https://github.com/RLinf/RLinf/archive/e1a82ebf0c1af69da6ab9b1519f5f3692fe33566.zip)
和 [RoboTwin ZIP](https://github.com/RoboTwin-Platform/RoboTwin/archive/dca9ec682688821c42944eacd7f5dd9bfb15396f.zip)，
上传到目标服务器。下面假设两个文件分别保存为 `~/RLinf_e1a82ebf.zip` 和 `~/RoboTwin_dca9ec68.zip`：

```bash
# 在含有 setup.sh 的项目目录中运行
python3 scripts/prepare_sources.py --rlinf ~/RLinf_e1a82ebf.zip --robotwin ~/RoboTwin_dca9ec68.zip
bash ./setup.sh --verify-sources
bash ./setup.sh
```

准备脚本先校验两个 ZIP 的文件清单和内容，再放入空的 `RLinf/` 和 `RoboTwin/` 目录；
已有非空目录会保留，不自动覆盖。源码若已解压到对应目录，可直接运行 `--verify-sources`。

ZIP 模式记录为 `archive_content_verified`：这表示文件内容与固定版本的受跟踪文件一致，
并不表示已经验证 Git 提交对象或其签名。额外文件不包含在校验中。训练的 `contract.json`
同时记录源码清单哈希和验证方式。ZIP 解决源码 Git 元数据问题；Python、uv、依赖、模型和数据
仍需通过可用软件源、镜像或预先准备的文件获取。

缺少 uv 时在 `.runtime/tools/bootstrap/` 内安装 uv 0.13.6；随后创建 Python 3.11.18 环境。
默认安装目录 `.runtime/envs/robotwin`，安装器不会删除重建已有环境；已有环境必须是相同 Python 版本。
安装清单固定 PyTorch 2.7.1/CUDA 12.6、OpenPI 0.1.1、Transformers fork 4.53.2、SAPIEN 3.0.1、mplib 0.2.1 等。
完整基础包版本在 `requirements/runtime-linux-py311.lock`，仿真附加包（含传递依赖）在 `requirements/simulator-linux-py311.lock`。
`requirements/simulator.txt` 另列直接依赖，便于审阅。
安装后执行依赖一致性和 CPU 导入检查。

OpenPI fork 与 stock Transformers 共用文件目录，安装器最后重新安装 fork；请勿随后单独升级 Transformers。
仿真包按固定 RLinf 安装器应用 URDF UTF-8 / mplib screw-planner 兼容修正，只替换该虚拟环境中的文件，
不修改第三方源码子模块或共享缓存。默认选择 mplib，不构建 curobo/PyTorch3D。

```bash
./setup.sh --train-only
./setup.sh --venv /absolute/path/to/robotwin-venv
export VLA_ENV=/absolute/path/to/robotwin-venv
```

可设置 `UV_DEFAULT_INDEX`、`UV_CACHE_DIR`、`HF_ENDPOINT` 与 HTTP 代理。
不要升级正在训练的环境。新服务器应新建 run；禁止将移动后或不同环境的旧目录当作同一协议直接恢复。

## 快速实验

```bash
./download.sh
./check.sh --deep
./train.sh --smoke --gpu 0 --detach
./run.sh status
# 状态显示 training_completed 后再运行
./evaluate.sh --smoke --gpu 0 --detach
```

`--smoke` 使用 20 步和每任务 2 episodes，仍加载完整 π0 模型，不保证小显存卡能运行。
正式实验去掉 `--smoke`，用 `--mode paired` 获取一个 MT 与各任务 ST。
训练、评估错误的原始 traceback 保存在控制器/子任务日志中，不应将错误作为策略失败计入成功率。

## 恢复

原生 RLinf checkpoint 包含模型、优化器、调度器、数据和 RNG 状态。恢复命令需要使用同一 run、相同任务/预算：

```bash
./train.sh --run /path/to/run --mode mt --gpu 0 \
  --resume-checkpoint /path/to/run/mt/bc/checkpoints/global_step_250
```

实际路径以组内输出为准。`contract.json` 不一致会拒绝恢复；配对组建议分别管理恢复，避免向另一组加载 checkpoint。

## GPU 验收范围

仓库的计划命令与单元测试均可在不访问 GPU、不下载权重的条件下运行。
实际训练、显存峰值、SAPIEN 渲染和闭环成功率需要目标服务器运行后确认；不能把 CPU 通过写成这些项目已通过。
