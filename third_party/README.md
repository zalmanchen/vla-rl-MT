# 固定版本源码

两个源码快照随本仓库提交，无需 Git submodule 或单独下载源码即可使用：

| 目录 | 上游版本 | 文件数 |
| --- | --- | --- |
| `RLinf/` | `e1a82ebf0c1af69da6ab9b1519f5f3692fe33566` | 2228 |
| `RoboTwin/` | `dca9ec682688821c42944eacd7f5dd9bfb15396f` | 735 |

上游仓库和压缩包 SHA256 见 [sources.lock.json](../bootstrap/sources.lock.json)。[sources.manifest.json](sources.manifest.json)保存锁定归档内每个文件的原始 SHA256；项目 `.gitignore` 按该成员清单仅放行源码，额外仿真资产、缓存和本地文件不提交。保留上游提供的许可证、版权及来源文件。

`RLinf/` 包含 [recap-robotwin.patch](../bootstrap/patches/recap-robotwin.patch) 中已记录的 7 个兼容文件修改；核验时只允许原始内容或清单中的补丁结果。目录内没有独立 `.git`，父项目的 Git HEAD 不能用作上游版本。

```bash
# 在项目根目录执行；没有下载缓存的新 clone 也可离线核验源码
python bootstrap/prepare_sources.py --verify-only
```

该命令优先核验随仓库的清单；本机若有源码 ZIP 缓存，也核验其归档哈希。源码被改动或缺失时会报错，不会覆盖本地修改。正常准备命令也会核验已有目录；源码目录完全不存在时才下载固定 ZIP。

`RLinf-recap-official/` 是同一上游版本的独立原始参考副本，继续忽略，不重复提交。需要时用 `python bootstrap/prepare_recap_official.py sources` 从锁定归档恢复。实际使用还需按[安装说明](../docs/setup.md)准备 Python／系统依赖、模型、数据和 RoboTwin 仿真资产。

更新上游版本时，同步更新来源锁、成员哈希清单、放行规则和兼容补丁，再核验；不直接跟随上游 main。
