# 第三方来源

本仓库保存实验扩展与来源锁；上游完整源码、模型和数据下载到本地忽略目录。

| 来源 | 记录与使用 |
| --- | --- |
| [RLinf](https://github.com/RLinf/RLinf/tree/e1a82ebf0c1af69da6ab9b1519f5f3692fe33566) | 固定 `e1a82eb`，完整提交与压缩包 SHA256 见 `bootstrap/sources.lock.json`；上游Apache-2.0许可，完整文本位于下载源码的 `LICENSE` |
| [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin/tree/dca9ec682688821c42944eacd7f5dd9bfb15396f) | 固定 `dca9ec6`，完整提交与压缩包 SHA256 见同一来源锁；源码与仿真资产分别遵循上游仓库和数据发布页的条款 |
| [π0.5 RoboTwin2权重](https://huggingface.co/motus-robotics/pi0.5_robotwin2) | 固定revision、下载文件与哈希见 `bootstrap/downloads.lock.json` |
| [RoboTwin2.0数据](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0) | 示范和仿真资产来源；数据不随本仓库重新发布 |
| OpenPI及Transformers适配包 | 实际包版本见 `bootstrap/environment.freeze.txt`；参考推理使用单独的 `rlinf-transformer-openpi` 目录 |
| RECAP critic 资源 | 下载清单见 `bootstrap/recap-models*.lock.json` |

论文出处保存在 `.research/literature/`。下载的PDF和全文副本不提交到Git；实验结果应区分本地复现、短预算适配与论文原始报告。

本机 RECAP 的 RoboTwin 兼容修改记录在 `bootstrap/patches/recap-robotwin.patch` 与对应 JSON 文件哈希清单。固定源码加此补丁是当前运行来源，不能继续描述为完全未修改的上游快照；原始源码锁不变。补丁只处理数据／模型接口与成功标签标量化，训练算法及 runner 沿用官方实现。

官方 LIBERO RECAP 的未修改源码与数据另锁在 `bootstrap/recap-official.lock.json`，源码位于 `third_party/RLinf-recap-official/`。该副本不应用 RoboTwin 适配补丁；环境和实验复现边界见 `docs/previous-experiments.md`。
