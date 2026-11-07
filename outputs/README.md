# 输出目录

真正的实验结果入口是 [实验总览](../.research/progress.html)；权重、轨迹和评估记录位于 [运行目录](../.runtime/experiments/README.md)。本目录中的时间戳文件夹主要是Hydra启动配置与启动日志，不是完整实验结果。

- `archive/hydra/e5-pirl/`：已结束的E5启动记录。
- `archive/hydra/e6-posttraining-diagnostic/`：已结束的E6启动与恢复记录。
- `2026-11-06/23-00-41/`：整理时仍被E7训练进程打开，保留原路径；后续新启动目录也应等进程结束后再归档。

每个归档保留`.hydra/config.yaml`、`hydra.yaml`、`overrides.yaml`和原日志，内容未修改。归属来自配置中的`runner.logger.log_path`，不只按日期猜测。

整理清单见 [最新迁移索引](../.research/organization-latest.json)。本次未删除任何文件，也未移动运行中的训练目录。
