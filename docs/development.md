# 开发与运行边界

先读对应的[两卡方法任务书](method-2gpu.md)或[八卡前置任务书](prerequisite-8gpu.md)，共同控制和交接以[研究协议](multitask-critic.md)为准。

## 代码与配置

- `third_party/`：RLinf `e1a82eb`、RoboTwin `dca9ec6`；模型、runner、分布式优化及 checkpoint 沿用官方实现，兼容修改单独记录在 `bootstrap/patches/`。
- `src/vla_rl_mt/`：研究扩展。目前尚无联合 SFT／value 方法实现；确定接口后再建实际模块，不增加空类或独立训练器。
- `configs/multitask/`：当前协议与规划；`configs/robotwin_pilot/`、`configs/recap_official/` 分别保留旧 pilot 和 LIBERO 参考。
- `bootstrap/`：环境、资源、数据与验收；`scripts/`：显式启动官方程序和既有评估／参考流程。

优先使用官方注册接口。官方 worker 会导入 `RLINF_EXT_MODULE`，模块提供 `register()` 时调用它；`bootstrap/env.sh` 与启动器已把 `src/` 加入 `PYTHONPATH`。确定具体模块后再设置扩展名，模块导入时不得启动进程或依赖某台机器的运行目录。

若官方没有必要扩展点，记录最小适配或独立补丁及对应测试，不将本地修改称为原始官方算法。RLinf 的 `use_rlt` 是 prefix 重建路径，不直接实现回报监督的共享 critic。

## 入口与产物

通用入口 `scripts/run.py` 接受显式 Hydra YAML，具体阶段见[配置索引](../configs/README.md)。先检查命令，不启动训练：

```bash
python scripts/run.py recap-returns --config configs/robotwin_pilot/returns.yaml --dry-run
```

当前五任务 P0 用 `scripts/evaluate_multitask.py`；`scripts/recap.py` 是旧两任务 pilot，`scripts/recap_official.py` 是独立 LIBERO 参考。后二者不作为方法线或八卡正式实验的默认队列。

新产物写 `.runtime/runs/method/<run-id>/` 或 `.runtime/runs/prerequisite/<run-id>/`。保存来源、最终配置、训练范围、数据／拆分／标签哈希、checkpoint、逐任务评估和成本，字段见共同协议。不要覆盖旧运行目录或冻结快照。

## 验证

```bash
source bootstrap/env.sh
python bootstrap/prepare_sources.py --verify-only
python -m unittest discover -s tests -v
```

入口改动检查参数、进程退出和路径；数据改动验证真实样本与边界；方法改动验证其声称改变的梯度或行为；环境变更需要实际 smoke。项目测试通过不代表正式训练、策略收益或远端环境通过。

既有实验集中见[一份摘要](previous-experiments.md)。旧实验记录、图表与归档代码已移出仓库正文；文献与研究分析单独保留。新实验只维护实际需要的配置、结果和交接信息，不恢复旧可视化队列。
