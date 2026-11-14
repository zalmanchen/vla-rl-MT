# 开发与运行边界

先读对应的[两卡方法任务书](method-2gpu.md)或[八卡前置任务书](prerequisite-8gpu.md)，共同控制和交接以[研究协议](multitask-critic.md)为准。

## 代码与配置

- `third_party/`：RLinf `e1a82eb`、RoboTwin `dca9ec6`；模型、runner、分布式优化及 checkpoint 沿用官方实现，兼容修改单独记录在 `bootstrap/patches/`。
- `src/vla_rl_mt/`：研究扩展。`shared_value.py` 提供共享 adapter、prefix 读出与动作 mask；`value_head.py`／`layer_value.py` 适配冻结特征 head；`joint_value.py` 提供 M2 实时 chunk 数据与联合模型；`policy_update.py` 接入固定标签的 native FM 更新及官方 guidance tokenizer／routing。沿用官方训练／AdamW／调度／checkpoint 循环，仅适配数据、梯度裁剪、诊断及 addon 导出；M2及原 adapter-only M4 将原 policy 冻结并置于 FSDP 之外；动作专家 follow-up 见下文。方法 worker 目前只支持独立单卡 NO_SHARD，不宣称多卡分布式支持。
- `configs/multitask/`：当前协议与规划；`configs/robotwin_pilot/`、`configs/recap_official/` 分别保留旧 pilot 和 LIBERO 参考。
- `bootstrap/`：环境、资源、数据与验收；`scripts/`：显式启动官方程序和既有评估／参考流程。

优先使用官方注册接口。官方 worker 会导入 `RLINF_EXT_MODULE`，模块提供 `register()` 时调用它；`bootstrap/env.sh` 与启动器已把 `src/` 加入 `PYTHONPATH`。确定具体模块后再设置扩展名，模块导入时不得启动进程或依赖某台机器的运行目录。

若官方没有必要扩展点，记录最小适配或独立补丁及对应测试，不将本地修改称为原始官方算法。RLinf 的 `use_rlt` 是 prefix 重建路径，不直接实现回报监督的共享 critic。

## 入口与产物

跨服务器普通SFT→RECAP基线见[总YAML／shell说明](sft-to-recap.md)。`sft_recap.py`仅编排官方入口、私有metadata、格式转换及配对评测；`recap_workflow.py`只注册RoboTwin 2.0的32步TrainConfig克隆。此入口不使用旧native方法worker，也不自动启动A—D候选实验。

通用入口 `scripts/run.py` 接受显式 Hydra YAML，具体阶段见[配置索引](../configs/README.md)。先检查命令，不启动训练：

```bash
python scripts/run.py recap-returns --config configs/robotwin_pilot/returns.yaml --dry-run
```

当前五任务 P0 用 `scripts/evaluate_multitask.py`；`scripts/recap.py` 是旧两任务 pilot，`scripts/recap_official.py` 是独立 LIBERO 参考。后二者不作为方法线或八卡正式实验的默认队列。

方法线的 `train_method_value.py`／`train_method_joint.py` 通过 `RLINF_EXT_MODULE` 选择注册与 worker 扩展；在 `validate_cfg` 初始化 Cluster 前设置独立 namespace。M4 沿用 value SFT worker／runner 的优化循环，项目模型返回 native FM loss，并非直接运行未改动的 CFG policy worker。`prepare_method_m4.py` 生成显式作业表，`run_method_jobs.py` 维护本次持有的 Ray head、认证与 GPU0／1 队列；不停止其他集群。`evaluate_method_updates.py` 执行固定场景闭环评测，`summarize_method_m4.py` 在 CPU 上审计 checkpoint、梯度、曝光及配对结果。native 轻量更新与未改动的官方 openpi_cfg RECAP 基线分别报告。

动作专家 follow-up 使用 `expert_policy.py`：将 native expert 模块真正纳入 FSDP，并把 native 调用位置绑定到包装模块；冻结 VLM 在 FSDP／checkpoint 外。保留原参数精度，使用官方 FP32MasterAdamW、`lr_multipliers` 和 DCP。两个更新范围使用相同 expert 梯度路径，adapter-only 仅从优化器排除 expert 并清梯度，权重哈希必须不变；不能由 `requires_grad` 数量判断实际优化范围。`prepare_method_m4.py --expert`、`evaluate_method_updates.py --expert` 和 `summarize_method_expert.py` 分别准备计划、评测与审计；不新增训练循环或修改 vendored 源码。

路径诊断用`check_policy_paths.py`和`check_backend_paths.py`，输入与判据分别锁定，后者的临时数学替换只存在于诊断模型；`check_training_autograd.py`补充开启autograd／激活检查点的真实前向核对，三组与no_grad完全相同。BC曲线由`prepare_bc_stability.py`准备，`bc_stability.py`继承原expert更新，只在预定步导出评测权重，并在文件完整写入后发布SHA就绪标记。`evaluate_method_updates.py`可等待该标记；中间权重明确禁止断点恢复，最终128步仍保存官方DCP／优化器。4步导出对照与旧连续训练逐张量相同，完整128步参考还需精确重放才可复用旧闭环结果。`evaluate_expert_fit.py --plan`检查全部点，`summarize_bc_stability.py`核验训练、权重、场景、噪声及复用来源。

新产物写 `.runtime/runs/method/<run-id>/` 或 `.runtime/runs/prerequisite/<run-id>/`。保存来源、最终配置、训练范围、数据／拆分／标签哈希、checkpoint、逐任务评估和成本，字段见共同协议。不要覆盖旧运行目录或冻结快照。

## 空间管理

用户已明确选择删除可重跑的历史大产物，不做长期大归档。原始模型、数据母池、环境、固定源码和锁保留；历史结果只保留集中摘要及必要的小配置／诊断。清理清单和实际释放量见`.runtime/storage/cleanup-report-v3.json`，保留的大权重在该报告的`kept_large_files`中。

本轮删除旧实验优化器／DCP、非保留权重及重复导出，并取消刚开始的大归档。旧pilot此前待确认的删除请求已被本次明确授权取代。保留3个共同初始化、3个成功BC末点和3个普通加权更新末点的大型addon，另保留小于64MiB的小文件；这些末点用于评测／初始化，不再具备原DCP的精确断点恢复能力。旧LIBERO实验权重同样按历史产物清理，原始模型、训练数据和既有结果仍保留；需要时重新运行。

新实验在启动前固定保存策略：默认只存一个可评测末点，恢复／交接研究确需时才为主基线和候选保留优化器状态；工程smoke检查完即清理大权重。方法对照保留相同预算的轻量结果，不因成绩好坏决定记录是否存在。新数据子集只保存索引／清单，共用不可变轨迹，不为每种方法复制图像或视频。

采集／训练前估算新增原始数据、所有检查点／优化器、导出和暂存的峰值占用，默认峰值之后仍留至少100GiB余量。不足时先缩队列或清理已确认可重跑的历史产物，不自动删除原始模型／数据。HF Arrow等缓存只在确认可从保留的源数据重建且无活动作业时清理。

完全相同的已完成文件可在完整SHA256核对后硬链接共用存储；不得原地修改共享文件，重跑写新目录。实际释放量按文件系统可用量变化报告，不重复计数硬链接的逻辑大小。旧目录中的`storage-retention.json`会说明哪些大文件已主动删除；历史`task-audit`表示当时训练验收，不代表被清理后仍可恢复。

## 验证

```bash
source bootstrap/env.sh
python bootstrap/prepare_sources.py --verify-only
python -m unittest discover -s tests -v
```

入口改动检查参数、进程退出和路径；数据改动验证真实样本与边界；方法改动验证其声称改变的梯度或行为；环境变更需要实际 smoke。项目测试通过不代表正式训练、策略收益或远端环境通过。

既有实验集中见[一份摘要](previous-experiments.md)。旧实验记录、图表与归档代码已移出仓库正文；文献与研究分析单独保留。新实验只维护实际需要的配置、结果和交接信息，不恢复旧可视化队列。
