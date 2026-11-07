# 第三方来源

本仓库保存实验扩展与来源锁；上游完整源码、模型和数据下载到本地忽略目录。

| 来源 | 记录与使用 |
| --- | --- |
| [RLinf](https://github.com/RLinf/RLinf) | 固定提交见 `bootstrap/sources.lock.json`；上游Apache-2.0许可，完整文本位于下载源码的 `LICENSE` |
| [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) | 固定提交见同一来源锁；源码与仿真资产分别遵循上游仓库和数据发布页的条款 |
| [π0.5 RoboTwin2权重](https://huggingface.co/motus-robotics/pi0.5_robotwin2) | 固定revision、下载文件与哈希见 `bootstrap/downloads.lock.json` |
| [RoboTwin2.0数据](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0) | 示范和仿真资产来源；数据不随本仓库重新发布 |
| OpenPI及Transformers适配包 | 实际包版本见 `bootstrap/environment.freeze.txt`；参考推理使用单独的 `rlinf-transformer-openpi` 目录 |
| RECAP与PACE相关资源 | 下载清单分别见 `bootstrap/recap-models*.lock.json`、`bootstrap/pace-downloads.lock.json` |

论文出处保存在 `.research/literature/`。下载的PDF和全文副本不提交到Git；实验结果应区分本地复现、短预算适配与论文原始报告。
