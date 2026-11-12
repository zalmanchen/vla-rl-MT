# Root 与无显示器仿真

**Root 本身不阻止 RoboTwin / SAPIEN 运行。** 本机已在 UID 0 下验证：Vulkan 能识别两张 GPU，SAPIEN 重力步进正常，相机生成有效 128×128×3 RGB 图像。证据位于 `.runtime/validation/headless-render.json` 与 `headless-render.png`。

主环境还通过官方 RLinf 加载 Motus π0.5，在 RoboTwin `handover_block` 上完成了 800 步闭环交互（seed 100000，约 102 秒）。记录与视频为 `.runtime/validation/handover_block.json`、`handover_block.mp4`。该次任务 `success=false`；这验证了运行链路，任务成功率和 RECAP 训练仍需后续验收。此次运行关闭了渲染降噪器。

## 本机检查

```bash
source bootstrap/env.sh
vulkaninfo --summary
python bootstrap/smoke_render.py
python bootstrap/smoke_episode.py
```

已安装 Vulkan loader、vulkan-tools、ffmpeg 和基础图形库；ICD 为 `/etc/vulkan/icd.d/nvidia_icd.json`。`env.sh` 设置私有 `XDG_RUNTIME_DIR`，支持覆盖 ICD 路径，无需桌面 X server。

主环境 `.runtime/venv-rlinf` 与独立诊断环境 `.runtime/venv-sim-check` 均已通过基础渲染验收，机器人策略闭环在主环境完成。诊断环境版本见 `bootstrap/simulation-check.freeze.txt`。SAPIEN 3.0.1 使用 `pkg_resources`，需保留该接口的 setuptools；诊断环境为 80.9.0，主环境锁为 75.8.2。

## 平台无法开放 graphics 能力时

在宿主机通过 NVIDIA Container Toolkit 启动官方 RoboTwin 镜像：

```bash
docker run --rm -it --gpus all --shm-size=32g \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -v /cloud/vla-rl-MT:/workspace/vla-rl-MT \
  rlinf/rlinf:agentic-rlinf0.4-robotwin
# 镜像内
source switch_env openpi
```

这是固定 RLinf 文档中的镜像标签；使用前仍须核对镜像内代码与项目源码锁的 commit。驱动由宿主机提供，不在受限容器内再套 Docker。

若平台始终不提供 graphics 能力，可在本机做 RECAP 离线训练，在通过渲染验收的机器采集 rollout 与评估，保持模型、归一化、动作和评估协议一致。

来源：[固定 RLinf RoboTwin 文档](https://github.com/RLinf/RLinf/blob/e1a82ebf0c1af69da6ab9b1519f5f3692fe33566/docs/source-en/rst_source/examples/embodied/robotwin.rst)、[SAPIEN 离屏渲染](https://sapien.ucsd.edu/docs/2.2/tutorial/basic/installation.html)。
