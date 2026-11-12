# RECAP 环境镜像

镜像封装当前已验证的 LIBERO／RECAP 环境：Python 3.11.15、PyTorch 2.7.1+cu128、RLinf 固定源码及完整安装依赖、LIBERO 仿真资产、tokenizer 和 SFT 归一化统计。保留现有 OpenPI／Transformers 实际安装文件，不重新解析 Python 依赖。系统层还包括 FFmpeg、EGL／OSMesa、gcc/g++ 和 Git。

模型参数、训练数据、检查点、Hugging Face 凭据和宿主机 GPU 驱动通过运行时挂载／NVIDIA runtime 提供。镜像中的 LIBERO 资产放在 `/opt/libero-assets`，挂载数据目录不会隐藏它们。封装环境不代表 RECAP 训练或性能复现已经完成。

来源与基础镜像 digest 见 [image-sources.lock.json](../docker/image-sources.lock.json)，Dockerfile 见 [Dockerfile.recap](../docker/Dockerfile.recap)。环境文件沿用当前 `/cloud/vla-rl-MT` 前缀；宿主机的数据可以放在其他目录，只需挂载到该容器路径。

## 导入与检查

在有 Docker Engine、NVIDIA Container Toolkit 和兼容 CUDA 12.8 驱动的 GPU 宿主机运行：

```bash
docker load -i /cloud/vla-rl-MT/.runtime/docker/recap-official/vla-rl-mt-recap-official.tar

# 不依赖外部模型或训练数据：依赖、CUDA、视频解码、LIBERO EGL 物理／RGB
docker run --rm --gpus all --shm-size=16g \
  vla-rl-mt:recap-official-20261111 python docker/check_image.py --gpu
```

没有 GPU 时可检查用户态、CPU 视频解码与 LIBERO OSMesa 离屏物理：

```bash
docker run --rm --shm-size=2g \
  vla-rl-mt:recap-official-20261111 python docker/check_image.py
```

归档 SHA-256、image ID、大小和实际验收结果位于同目录 `image.json`、`SHA256SUMS`。下载／复制到其他机器后可用 `sha256sum -c SHA256SUMS` 校验。

## 使用已有实验目录

```bash
cd /cloud/vla-rl-MT
mkdir -p .runtime/runs .runtime/recap-official

docker run --rm -it --gpus all --shm-size=16g \
  -v "$PWD/bootstrap:/cloud/vla-rl-MT/bootstrap:ro" \
  -v "$PWD/scripts:/cloud/vla-rl-MT/scripts:ro" \
  -v "$PWD/configs:/cloud/vla-rl-MT/configs:ro" \
  -v "$PWD/models:/cloud/vla-rl-MT/models:ro" \
  -v "$PWD/datasets:/cloud/vla-rl-MT/datasets:ro" \
  -v "$PWD/.runtime/runs:/cloud/vla-rl-MT/.runtime/runs" \
  -v "$PWD/.runtime/recap-official:/cloud/vla-rl-MT/.runtime/recap-official" \
  vla-rl-mt:recap-official-20261111 bash

# 容器内：查看解析后的官方配置，不启动训练
python scripts/run.py recap-value --profile recap-official \
  --config .runtime/recap-official/configs/value.yaml -- --cfg job --resolve
```

当前镜像先于完整流程入口生成；上述 `bootstrap/`、`scripts/`、`configs/` 挂载用于带入最新项目入口，不覆盖镜像内固定的 RLinf 和环境。完整流程的源码审计还需固定的上游 archive，可将 `.runtime/bootstrap/RLinf-upstream.zip` 单独只读挂载到相同路径；避免整个 `.runtime/bootstrap` 只读挂载，因为审计会写记录。

挂载 `models/`、`datasets/`、运行目录等具体子目录；直接挂载整个项目到 `/cloud/vla-rl-MT` 会隐藏镜像内的 Python 环境和源码。需要改实验配置时，可以另外挂载 `configs/`；修改后用 `python bootstrap/prepare_recap_official.py configs` 重新生成完整配置。LIBERO 参考的范围见[既有实验摘要](previous-experiments.md)；重新用于训练前仍须验收模型转换、数据与预算。

## 重新打包

在当前已安装环境的机器准备不含凭据／模型／训练数据的构建上下文。`context` 默认复用已冻结快照，不自动覆盖后续的包或代码修改；配置开发优先通过挂载目录进行。

```bash
.runtime/venv-recap-official/bin/python bootstrap/build_recap_image.py context

docker build -t vla-rl-mt:recap-official-20261111 \
  .runtime/docker/recap-official/context
```

当前执行容器没有 Docker socket。此次打包使用固定 NVIDIA CUDA rootfs、chroot 安装系统库、环境快照和 Docker archive 格式，生成可导入的镜像，未启动或修改宿主机 Docker daemon。准备过程和日志位于 `.runtime/docker/recap-official/`；本机无 daemon 的打包命令为 `bootstrap/build_recap_image.py build`，依赖已缓存的 CUDA rootfs、crane 及当前环境快照。

无 daemon 的验收包含归档完整性及 chroot 中的真实用户态／视频／软件渲染测试。Docker GPU 容器的完整验收仍需在上述宿主机执行 `--gpu` 检查，不能用 chroot 测试代替。

GPU 的 `compute,utility,graphics` 能力配置依据 [NVIDIA Container Toolkit 官方说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html)。归档采用 [go-containerregistry 所描述的 Docker archive 格式](https://github.com/google/go-containerregistry/blob/v0.22.1/pkg/v1/tarball/write.go)，并由 crane 实际校验。
