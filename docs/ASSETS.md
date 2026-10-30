# 固定资源与下载

所有下载来自公开源，适用上游各自的许可。`resources/releases.json` 记录主线资源 commit、文件大小和源哈希。
权重、数据和仿真资产不作为 Git 文件上传。

| 资源 | 来源 | 固定版本 |
| --- | --- | --- |
| 原生 π0 RoboTwin SFT | [RLinf/RLinf-Pi0-NEW-RoboTwin-SFT-adjust_bottle](https://huggingface.co/RLinf/RLinf-Pi0-NEW-RoboTwin-SFT-adjust_bottle) | `8975e3349d55a3bbe8c0d39f84642c27c023ecf1` |
| RoboTwin 轨迹与资产 | [TianxingChen/RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0) | `3dc3b798668feb99ac61cc9086d84cbcc3d79186` |
| RLinf 源码 | [RLinf/RLinf](https://github.com/RLinf/RLinf) | `e1a82ebf0c1af69da6ab9b1519f5f3692fe33566` |
| RoboTwin 仿真源码 | [RoboTwin RLinf_support](https://github.com/RoboTwin-Platform/RoboTwin/tree/RLinf_support) | `dca9ec682688821c42944eacd7f5dd9bfb15396f` |

默认两任务：`adjust_bottle`、`handover_block`。另外冻结了 `lift_pot`、`stack_blocks_two`、`beat_block_hammer` 的下载元信息。
只使用 ALOHA-AgileX clean 示范；不混用其他具身或 randomized 500 条版本。

```bash
./download.sh --dry-run
./download.sh --component model
./download.sh --component data
./download.sh --component assets
./download.sh --component tokenizer
```

下载器使用固定 commit；检查大小、LFS SHA256 或 Git blob 哈希。ZIP 解压拒绝目录穿越和符号链接。
模型归一化文件随原始 checkpoint 保存；训练的共同统计量由 `prepare_data.py` 仅用训练 episodes 拟合。
模型初始权重来源已在 adjust_bottle 训练，不得将该任务视为模型未见。

PaliGemma tokenizer 来源：`https://storage.googleapis.com/big_vision/paligemma_tokenizer.model`，
大小 4,264,023 字节，SHA256 `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`。
默认缓存 `.runtime/cache/openpi/`，下载和训练共用 `OPENPI_DATA_HOME`，避免首次训练隐式访问网络。

仿真需要 `objects.zip`、`embodiments.zip`、`background_texture.zip`；合计约 15GB 压缩文件。
解压到 `RoboTwin/assets/` 并运行上游具身路径配置脚本；迁移源代码/资产后需在目标位置重新配置这些绝对路径。

## 灵巧手扩展

```bash
./download.sh --component dexmimicgen
```

下载 [MimicGen/dexmimicgen_datasets](https://huggingface.co/datasets/MimicGen/dexmimicgen_datasets/tree/main/generated)
的箱体整理、抽屉整理和托盘抬升三个 HDF5，约 20GB。
首次把 main 解析成 commit，再固定该 commit 下载并记录校验凭证；恢复时复用已保存版本。
这些数据仅提供下载入口，灵巧手高维动作、模型、奖励和评价尚未接入，不能用主线训练命令直接运行。
