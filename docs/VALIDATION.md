# 发布验证记录 · 2026-10-30

已执行：

- 在独立的 Python 3.11.18 环境实际运行安装入口，安装 289 个基础包及 37 个仿真附加包。
- `uv pip check`：326 个包全部兼容。
- 确认环境没有安装 `libero`；默认安装和资源下载均只针对 RoboTwin。
- 原生 OpenPI、Torch/CUDA runtime 导入和 CPU 检查通过，未初始化 CUDA context。
- 9 项 CPU 回归通过：未来动作窗口不跨 episode、14 维状态检查、重复内容分组隔离、
  每 microbatch 任务均衡与 sampler 恢复、ZIP 路径安全、损坏文件校验、任意工作目录的冷启动计划、
  MT/ST 配置预算、原生动作变换回环。动作回环测试隔离 tokenizer 网络下载，不加载模型权重。
- Shell 语法与 Python 静态检查通过。

主线模型/数据的公开版本和文件哈希已根据上游元数据固定在 `resources/releases.json`。
此轮没有下载 13GB 权重或完整 RoboTwin 示范/仿真资产，也没有启动 GPU 训练或闭环。
SAPIEN Vulkan 渲染、真实训练 loss/显存、checkpoint 恢复及成功率需要目标服务器实测；
这里的 CPU 测试结果不能替代这些验收。
