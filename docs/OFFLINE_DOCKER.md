# 本地离线安装与 Docker

本目录对应 RoboTwin 发布版，不是 LIBERO 实验分支。当前配套离线包为
`robotwin-offline-linux-x86_64-py31118.tar.gz`，包含 uv 0.13.6、Python 3.11.18 和固定版本依赖缓存，
不含模型、数据或仿真资产。SHA-256：
`f0d3989656be8c507096c04a30b8b5ce4ffad2f9371787edac24fe76a9453054`。
需要 Linux x86_64；离线包约 8.1 GiB，解压和安装环境建议至少预留 40 GiB，实验资产需要额外空间。

## 在可联网的主机下载依赖并生成离线包

将本项目 ZIP 解压到可联网的 Linux x86_64 主机。默认 Python 包源为阿里云，Python 解释器
使用中科大 python-build-standalone 镜像；具体版本是否在镜像上可用需以实际下载为准。
也可使用 `--index URL`、`--python-mirror URL` 指定可用的镜像；`--python-mirror official` 使用上游地址。
联网主机需要 `python3`、`python3-venv`、`curl`、`tar` 和足够磁盘空间。

```bash
mkdir -p .runtime/exports
bash scripts/download_dependencies.sh --output "$PWD/.runtime/exports/robotwin-deps.tar.gz"
```

脚本会安装准备环境、下载依赖、用全新的虚拟环境离线安装并检查，然后生成压缩包和
`robotwin-deps.tar.gz.sha256`。下载中断可用相同参数重试；缓存保存在 `.runtime/offline-build`。
如更换镜像或 lock 文件，使用不同的 `--work-dir`。不要把普通虚拟环境目录直接拷贝到另一台机器；
应复制这两个文件并在目标机器离线重建，或者在联网机器构建 Docker 镜像后传输镜像。

## 在 Linux 直接安装

将项目 ZIP 和离线包通过内网、移动介质或其他方式传到目标服务器。解压项目 ZIP，
进入包含 `setup.sh` 的目录（若更早下载的项目 ZIP 没有这些新脚本，需下载包含它们的修复分支版本）：

```bash
uname -sm
df -h .
bash scripts/install_local.sh --bundle ~/robotwin-offline-linux-x86_64-py31118.tar.gz
```

若使用自己生成的包，需要传入对应的 SHA-256，例如把压缩包和校验文件都复制到家目录后：

```bash
cd ~
sha256sum -c robotwin-deps.tar.gz.sha256
cd /path/to/project
bash scripts/install_local.sh --bundle ~/robotwin-deps.tar.gz \
  --sha256 "$(cut -d ' ' -f1 ~/robotwin-deps.tar.gz.sha256)"
```

校验文件用于检查传输完整性，必须通过可信渠道与压缩包一起传输；它本身不是签名。

脚本检查离线包 SHA-256，首次解压后创建标记；已存在的其他目录不会被覆盖。
安装中断后可重复运行。也可以指定 `--venv /absolute/path/to/venv` 或 `--train-only`。
安装使用离线缓存；包内软件源地址仅用于匹配缓存索引，不要求镜像联网。
无离线包但可以访问国内源时，也可以选择：

```bash
bash scripts/install_local.sh --online --index https://mirrors.aliyun.com/pypi/simple/
```

在线模式设置 pip 和 uv 的包源；uv 下载 Python 解释器可能仍需访问其他服务器，
不能把可访问 PyPI 镜像视为所有下载都可用。

## 在有 Docker 权限的机器制作镜像

准备约 8.1 GiB 的离线包，构建机器还需有足够的 Docker 存储空间。
默认基础镜像为 `ubuntu:22.04`，构建时 apt 使用阿里云 Ubuntu 镜像；离线 Python 依赖不访问网络。
因此构建机必须能取得基础镜像和 apt 包，或提前载入已安装所需系统库的 Ubuntu 22.04 基础镜像。
Docker Hub 无法访问时，可通过 `--base` 指定已经拉取或载入的可信镜像（建议固定 digest）；
apt 无法访问时可先准备包含 Dockerfile 所列系统包的基础镜像，再使用 `--skip-apt`。
不要假定第三方镜像站可用。

```bash
bash docker/build_image.sh \
  --bundle ~/robotwin-offline-linux-x86_64-py31118.tar.gz \
  --tag vla-robotwin:offline \
  --output "$PWD/.runtime/exports/vla-robotwin-offline-image.tar"
```

如果使用 `download_dependencies.sh` 生成的包，额外指定
`--sha256 "$(cut -d ' ' -f1 /path/to/robotwin-deps.tar.gz.sha256)"`。

可选 `--base IMAGE`、`--apt-mirror URL` 和 `--skip-apt`。镜像导出路径已存在时脚本会拒绝覆盖。
成功后通过 `scp` 或移动介质复制导出的 tar，在目标服务器加载：

```bash
docker load -i vla-robotwin-offline-image.tar
docker image inspect vla-robotwin:offline
```

目标服务器仍需正确安装 Docker、NVIDIA 驱动与 NVIDIA Container Toolkit。图形仿真还需要
可用的 Vulkan/EGL 驱动；模型、数据和仿真资产必须单独准备。镜像只提供环境与代码。

下面例子把宿主机 GPU 1 映射到容器，把数据和运行输出保存在宿主机。
请先进入目标机器存放 `.model`、`.data`、`.runtime` 的实验目录；镜像中的环境在 `/opt/vla-env`，
所以挂载 `.runtime` 不会遮住 Python 环境：

```bash
mkdir -p .model .data .runtime RoboTwin/assets
docker run --rm -it --user "$(id -u):$(id -g)" --gpus '"device=1"' --shm-size=16g \
  -e HOME=/tmp -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -v "$PWD/.model:/opt/vla-code/.model" \
  -v "$PWD/.data:/opt/vla-code/.data" \
  -v "$PWD/RoboTwin/assets:/opt/vla-code/RoboTwin/assets" \
  -v "$PWD/.runtime:/opt/vla-code/.runtime" vla-robotwin:offline bash
```

在容器中，映射进来的 GPU 通常以 `0` 编号。先检查 `nvidia-smi` 与
`bash ./train.sh --smoke --plan`，再根据实际资源运行训练。迁移历史 run、checkpoint 或数据协议前
还需核对路径和契约；不要直接假设它们可以跨环境恢复。

## 模型与数据下载

使用默认 `--work-dir` 生成依赖包后，可在联网主机执行
`export VLA_ENV="$PWD/.runtime/offline-build/online-venv"`，使用准备环境运行资源下载。
如果自定义过 `--work-dir`，请相应调整路径。使用 `bash ./download.sh --component model`、
`--component data`、`--component assets` 和 `--component tokenizer` 分别下载。
`HF_ENDPOINT` 可指向可信且可访问的 Hugging Face 兼容镜像；tokenizer 默认来自 Google，
可通过 `VLA_TOKENIZER_URL` 指定可信的同文件镜像，或者单独准备后放到
`.runtime/cache/openpi/big_vision/paligemma_tokenizer.model`，脚本会校验固定 SHA-256。
下载所得的 `.model/`、`.data/`、`RoboTwin/assets/` 与 tokenizer 缓存不包含在环境包或镜像中，
需分别传输。仿真资产路径可能包含绝对路径，应在目标路径重新执行
`RoboTwin/script/update_embodiment_config_path.py`；容器中需在容器内执行并使用容器路径。
