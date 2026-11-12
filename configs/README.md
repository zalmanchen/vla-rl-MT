# 配置入口

| 目录 | 用途 | 当前边界 |
| --- | --- | --- |
| [multitask/](multitask/two_track.json) | 当前双线规划、共同约定与既有 P0 协议 | `two_track.json` 不执行训练；`phase0.json` 用于 P0 校准 |
| `robotwin_pilot/` | 旧两任务 RECAP 四阶段 YAML | 工程参考数据／预算，不能直接作为八卡正式对照 |
| `recap_official/` | 独立 LIBERO 公开流程参考 | 使用独立源码／环境，保留旧实验预算 |

两种服务器的任务见[两卡方法线](../docs/method-2gpu.md)、[八卡前置线](../docs/prerequisite-8gpu.md)。新训练 YAML 在任务、数据、预算与参数范围锁定后再加入；当前不创建空配置或复制训练器。

## 官方入口

`scripts/run.py` 要求显式 Hydra YAML，`--` 后参数原样传入。先预览命令：

```bash
python scripts/run.py recap-returns --config configs/robotwin_pilot/returns.yaml --dry-run
```

| 子命令 | 官方程序／用途 |
| --- | --- |
| `sft` | `examples/sft/train_vla_sft.py` |
| `recap-returns` | 官方轨迹回报计算 |
| `recap-value` | 官方 RECAP critic 训练 |
| `recap-advantages` | 官方优势标注；可用 `--nproc` 做本机多 GPU 标注 |
| `recap-policy` | 官方 CFG policy 更新 |
| `train` / `eval` | 官方在线训练／独立评估 |

训练的多卡放置由 RLinf 配置控制，不额外套 torchrun。`PROJECT_ROOT`、`REPO_PATH` 等由环境／入口设置；新数据 tag、输出目录与预算需显式记录。

## pilot 配置迁移

旧 `configs/recap_{returns,value,advantages,policy}.yaml` 已原样移到 `configs/robotwin_pilot/{returns,value,advantages,policy}.yaml`。四份内容 SHA256 保持不变，调用方与 Hydra 搜索目录同步更新；`scripts/recap.py` 仍是历史 pilot 入口。旧冻结运行快照保持原样。

收益比较仍须遵守[共同协议](../docs/multitask-critic.md)：critic 实际输入不由 `include_state` 决定，回报与标签标尺一致，任务 raw weight 不等于实际任务曝光，零更新 CFG 条件与训练后部署路径分别记录。
