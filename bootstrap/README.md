# 环境与资源准备

本目录负责安装、固定版本下载、数据准备和验收。当前普通基线见[RoboTwin 2.0 SFT→RECAP](../docs/sft-to-recap.md)；P0使用`audit_multitask.py`和`../scripts/evaluate_multitask.py`，共同控制见[多任务协议](../docs/multitask-critic.md)。保留的 LIBERO RECAP 参考使用 `prepare_recap_official.py`、`recap-official.lock.json` 和独立的 `env_recap_official.sh`，范围见[既有实验摘要](../docs/previous-experiments.md)。`../scripts/recap.py` 保留两任务 pilot；`env.sh` 同时用于当前 Motus P0 与 RoboTwin 工程路径。

| 用途 | 入口 |
| --- | --- |
| 新服务器的RoboTwin 2.0普通SFT→RECAP | [总YAML与shell说明](../docs/sft-to-recap.md)，使用主`env.sh`，与独立LIBERO环境分开 |
| 已验证环境快照与 Docker 镜像打包 | `build_recap_image.py`；使用见 [Docker 说明](../docs/docker.md) |
| 官方 LIBERO 数据／原始源码／配置准备 | `prepare_recap_official.py`、`recap-official*.lock.json` |
| 原生 SFT → CFG 的官方键转换／数值核对 | `export_recap_cfg.py`、`check_recap_formats.py` |
| 工程 smoke 的独立完整 episode 子集 | `prepare_recap_smoke.py`，不得用作正式数据 |
| 官方 LIBERO 环境变量／真实仿真测试 | `env_recap_official.sh`、`check_libero.py` |
| 安装主环境与上游源码 | `install.sh`、`prepare_sources.py` |
| 只读核验随仓库源码及可选 ZIP 缓存 | `prepare_sources.py --verify-only`；RLinf `e1a82eb`＋记录补丁、RoboTwin `dca9ec6`，不下载、不应用补丁；可用 `--source` 选择单个来源 |
| PyTorch wheel 分块续传及官方 SHA-256 校验 | `download_wheel.py` |
| 环境变量 | `source bootstrap/env.sh`（在项目根目录） |
| 模型与基础数据下载 | `download.py`、`download_recap_models.py` |
| Clean 数据与资产分块续传 | `download_dataset_ranges.py` |
| SigLIP 分块续传 | `resume_siglip_ranges.py` |
| 可选 Randomized 数据 | `download_randomized.py` |
| 解压与内容校验 | `extract_assets.py`、`verify_downloads.py` |
| 工具／文件清单检查 | `check_recap_prerequisites.py` |
| 实际验收 | `smoke_render.py` → `smoke_model.py` → `smoke_episode.py` |
| 等待已有安装下载后完成基础验收 | `finish_recap_setup.py` |
| 记录并应用 RoboTwin 兼容补丁 | `patches/`、`apply_recap_patch.py` |
| 原生 HDF5／自主轨迹转 LeRobot | `prepare_recap_data.py` |
| M1 固定开发场景采集／冻结 | `prepare_method_m1.py`，包含真正的最后动作后观测与实际 chunk 边界 |
| M1 数据独立核验／冻结 prefix 特征 | `cache_method_m1.py`；支持全局／图像与 prompt 分组读出、同前向形状控制和旧特征精确核对；不训练 actor，不直接生成逐帧 SFT 数据 |
| M1 二值回报派生缓存 | `prepare_method_binary.py`；核验父池／缓存，只派生成功标签与终局 chunk 奖励，输入特征和旧数据保持不变 |
| 冻结 W1 的有界层特征 | `cache_method_layers.py`；最多四层 prefix、同一 final norm／池化，最后层需精确重现原读出 |
| 方法线作业／评测计划 | `prepare_method_m4.py`；从锁定协议生成 head、固定标签 policy 和零更新／训练后评测队列，不直接启动作业 |
| 固定策略的有界自主采集 | `collect_recap_rollouts.py` |
| 零更新 CFG 输入与动作核验 | `check_recap_policy.py` |
| 配对开发场景评估与视频 | `evaluate_recap.py` |

同一资源目录只运行一个下载器。分块下载日志在 `.runtime/logs/recap-data-ranges.log`，验收状态在 `.runtime/bootstrap/recap-preparation.json`。

版本锁：`sources.lock.json` 为上游源码，`downloads.lock.json` 为策略／数据，`recap-models.lock.json` 为 critic 骨干，`environment.freeze.txt` 为主环境，`tools.freeze.txt` 为下载工具，`simulation-check.freeze.txt` 为已通过 root 渲染测试的独立诊断环境。`source-audit.json` 是历史审计记录，不能替代本次运行验收。

详细步骤见 [安装与下载](../docs/setup.md)、[仿真验收](../docs/simulation.md)。旧 PACE／E5 等实验记录与归档脚本已移出仓库正文；现有实验只保留一份摘要。
