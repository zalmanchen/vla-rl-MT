# π0.5 × RoboTwin × RLinf：训练与排障经验

记录日期：2026-11-06，Asia/Shanghai。覆盖 E0—E6 的接入、评估、RECAP 适配与 Flow-SDE＋PPO 训练。本文保存可复用经验；正在运行的任务以[实时进度](results/e6/stability-progress.html)为准。

**工程修复、诊断配置和成功率收益分别验收。** 本文中的“已验证”指写明的实验范围，不代表普适最优配置，也不代表已经提高任务成功率。

## 1. 下次实验先看这张表

| 经验 | 当前证据 | 下次默认做法 |
| --- | --- | --- |
| FP32 权重仍可能启用 TF32 计算 | 固定权重、固定输入，关闭 TF32 后批大小引起的最大 PPO 比率差从 2.6342% 降到 0.0122% | 先用严格 FP32 完成概率一致性验收；精度设置放在模型构造之后，覆盖 actor 和 rollout |
| OOM 不只发生在反向传播 | 实际发生于差分权重编码、rollout 模型重新上卡 | 按采样、训练、同步、保存分别记录显存；本机使用 CPU 分桶暂存与同步后清理 |
| 日志中的 dtype、lr 或类名不能独立证明配置生效 | 动态方法替换没有生成远程审计记录；后来显式子类生成真实记录 | 检查 worker 实际输出、可训练参数、审计文件和完成标记 |
| PPO 重算审计要保留行为策略分母 | 当前回调只测量误差，继续使用采集时的 prev_logprobs | 保存触发批次；不通过替换分母或直接提高阈值让检查通过 |
| 首个检查点应早于容易失败的阶段 | 曾完成 4 轮，但在第 5 轮保存前停止，权重无法恢复 | 短实验保存第 1／5／10 轮；审计中断前另存检查点 |
| 一个动作块不一定执行满 32 步 | 强制提前成功测试只执行 1 步，奖励必须落在第 0 槽 | 记录实际执行量，验证终止奖励落在有效 mask 内 |
| 更短执行窗口不一定更好 | H=8／16／32 在不同任务上的排序不同 | 冻结评估窗口；改窗口作为单独实验 |
| 先做短实验，再扩任务和种子 | 旧 PPO 第 10／20／100 轮配对结果为 7／2／1 次成功，原始为 9 次 | 先检查早期曲线，工程验收和收益验证通过后再扩大矩阵 |

## 2. 适用环境与复现边界

当前在线诊断实现为 E6 `source-v7-cpu-staging`；后续修改须记录适用版本，不能将当前配置自动视为上游默认行为。

- 起点为已经做过 RoboTwin 多任务 SFT 的 `motus-robotics/pi0.5_robotwin2`；不重复进行初始 SFT。
- 当前 embodiment 为 ALOHA 双臂关节／夹爪接口，环境动作 14 维，模型内部动作维度 32。当前仿真结果不能直接当作灵巧手结果。
- 已使用的环境为 Python 3.11.15、PyTorch 2.7.1＋CUDA 12.8，以及固定源码快照。具体依赖以[环境锁](../bootstrap/environment.freeze.txt)为准，源码提交以[源码锁](../bootstrap/sources.lock.json)为准。
- `bootstrap/` 管理环境与资产，`experiments/` 放项目扩展，`.runtime/` 保存运行产物，`.research/` 保存可读报告。修改入口和扩展，不在上游快照中散落临时修补。
- 原始参考评估使用独立的 `.runtime/reference-transformers/`；不要为了某个入口能导入，就直接覆盖整个训练环境的 Transformers。

入口和目录说明见[环境 README](../bootstrap/README.md)、[实验 README](../experiments/README.md)。

## 3. 精度：FP32 存储与 FP32 计算要分别检查

### 现象与验证

E6 的一个真实批次在第 5 轮更新前停止：14,336 个概率分量中有 1 个误差超过 0.01，最大值为 0.0119457。按实际 chunk-level PPO 聚合，最大比率偏差为 1.4570%，没有越过该配置的 20% clipping 边界。

检查源码发现 `Pi0.__init__` 会设置 `torch.set_float32_matmul_precision("high")`。所以 `precision: fp32` 和 AMP 关闭，并不保证矩阵计算禁用 TF32。

随后使用**原始 Motus 固定权重、存留的真实输入、无优化更新**做数值对照：

| 对照 | 默认允许 TF32 | 禁用 TF32 |
| --- | --- | --- |
| 同一批大小重复前向，逐元素最大差 | 0 | 0 |
| batch 2 对 batch 4，逐元素最大差 | 0.00945306 | 0.00003934 |
| batch 2 对 batch 4，PPO 比率最大偏差 | 2.6342% | 0.0122% |
| eval 对 train，同 batch 4，PPO 比率最大偏差 | 2.2569% | 0.0122% |

### 已采用的修复

模型构造完成后，在 actor 和 rollout 各自进程执行：

```python
torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.allow_tf32 = False
```

必须放在模型构造之后，否则构造函数可能覆盖设置。只修改主进程不够。实际 worker 日志须出现 `[E6 precision] matmul=highest; cudnn_tf32=False`，批次审计也记录这两个设置。

原来的 0.01 停止阈值保持不变。这是本项目的诊断门槛，不是 PPO 或论文规定的通用常数。严格 FP32 可能降低吞吐，应单独测量其速度成本。

**结论边界：** 数值对照验证了计算精度对概率一致性的影响；使用的是原始权重，不是已丢失的第 4 轮权重。不能由此认定 TF32 解释了此前所有成功率下降。

证据：[默认精度对照](../.runtime/experiments/e6-posttraining-diagnostic/numerical-diagnostic/results.json)、[严格 FP32 对照](../.runtime/experiments/e6-posttraining-diagnostic/numerical-diagnostic-highest/results.json)。实现：[精度扩展](../experiments/pirl_diagnostic_extension.py)、[可重跑的诊断脚本](../experiments/diagnose_pirl_logprob_numerics.py)。

## 4. 显存：检查阶段切换和张量生命周期

### 这次遇到的两类峰值

1. Patch 同步在 GPU 上构造差分坐标、值和编码缓冲，曾在 `delta_encode` 阶段 OOM。`snapshot_device: cpu` 不表示整个差分计算都在 CPU。
2. 改用 GPU 分桶后，仍在下一轮 rollout 的 `model.to(cuda)` 阶段 OOM；错误中 actor 占用约 26.48 GiB。

### 当前本机采用的组合

- 使用 RLinf 官方 bucket syncer，`bucket_device: cpu`、`bucket_dtype: null`，保持 FP32 数据。
- 目标桶大小为 128 MiB。单个参数张量不会被切开，因此这不是每个桶或总暂存内存的严格上限。
- 当前诊断同步全部 675 个状态张量，包括冻结权重；原默认选择为 198 个可训练参数／持久缓冲区。这样保持首次全量同步的覆盖范围。
- 在基类同步协程返回后，执行 `gc.collect()`、CUDA 同步与 `empty_cache()`。此时局部 GPU `state_dict` 已解除引用；在仍有活引用时调用 `empty_cache()` 无法释放相应存储。
- 用 `memory-phases.jsonl` 记录 allocated、reserved 和设备空闲量，分别观察同步后与训练后。

**不要记成“每次同步后显存都会归零”。** 本次首次同步后 allocated／reserved 为 0；后续同步后的 allocated 仍约 12.51 GiB，reserved 也会变化。初次同步后整卡可用约 88.6 GiB，是一次观测，不是硬性保证。

**结论边界：** CPU 分桶与同步后清理是一起采用的组合修复，未分别消融；不能宣称其中一项单独解释全部 OOM。全量同步增加传输成本，CPU 暂存消耗主机内存，不能无条件照搬到多机训练。

证据：[实际阶段显存](../.runtime/experiments/e6-posttraining-diagnostic/jobs/ppo-update1-adjust_bottle-seed0/memory-phases.jsonl)、[同步范围](../.runtime/experiments/e6-posttraining-diagnostic/jobs/ppo-update1-adjust_bottle-seed0/synchronization-contract.json)、[中断尝试归档](../.runtime/experiments/e6-posttraining-diagnostic/aborted-attempts/)。实现：[诊断 worker](../experiments/pirl_diagnostic_actor.py)。

## 5. Ray 扩展：用远程执行证据验收

固定 RLinf 版本的公开 `algorithm.recompute_logprobs` 开关不接受当前 OpenPI 配置；直接打开会在训练前被校验拒绝。

最初的动态方法替换没有在真实远程 worker 生成审计文件。之后改用模块中显式定义的 `DiagnosticEmbodiedFSDPActor` 子类，通过项目训练入口选择它，并让子进程加载相同扩展。

下次验收顺序：

1. 验证入口选择的是正确的子类。
2. 看远程 worker 的类名和配置日志。
3. 至少跑一个真实 batch，检查 `batch-audit.jsonl` 内容。
4. 检查数值、参数冻结等条件，再写完成标记。

仅有主进程单元检查或配置打印不构成远程生效证据。最初 5 轮 critic-only 没有审计收据，因此单独续跑 1 轮核验，没有把缺失记录补写为通过。

入口：[run_pirl_diagnostic_job.py](../experiments/run_pirl_diagnostic_job.py)。修订轨迹：[E6 plan](../.runtime/experiments/e6-posttraining-diagnostic/plan.json)。

## 6. PPO 审计：固定输入、检查聚合、保留原分母

- 在优化器更新前，用记录的 observation、denoising chain 和 denoise indices 重算概率。
- 当前审计只测量，不替换 rollout 的 `prev_logprobs`。否则可能把采样／训练不一致隐藏在新分母中。
- 同时看逐元素 mean／p99／max，以及真正进入 PPO 的概率比率。当前配置是 chunk-level：先分别累加新旧 logprob 的动作和时间维，再相减并取指数。
- 发生不一致时先保存 batch，再做同权重重复前向、不同批大小、train／eval 模式对照，区分随机性与数值敏感性。
- 检查 NaN／Inf、形状、有效 mask，以及回报方差。不要只看一个平均 ratio。
- 本地审计阈值应随协议记录；未来要改变门槛，必须保存依据和新版本，不能静默改成“能跑”。

配置中的 `algorithm.recompute_logprobs: false` 与实际有审计并不矛盾：项目子类在已有训练 hook 中临时执行 OpenPI 专用测量，随后恢复开关，避免走不适配的官方重算逻辑。

证据：[实际批次记录](../.runtime/experiments/e6-posttraining-diagnostic/jobs/ppo-update1-adjust_bottle-seed0/batch-audit.jsonl)。

## 7. 检查点：保存模型，也保存恢复语义

- 本次曾完成 4 轮，但第 5 轮更新前停止，尚未达到原保存间隔，无法恢复那 4 轮权重。
- 当前短实验额外保存第 1 轮，正常保存第 5／10 轮；审计触发停止时，先保存触发 batch 和更新前的检查点。
- 使用框架保存模型、优化器和学习率调度器；额外的 `diagnostic-state.json` 保存 `global_step`、`optimizer_steps`，后者在加载时恢复。
- 完整保存后才原子写入元数据。恢复选择要求权重文件与完成元数据齐全，目录存在或某个文件出现都不足以判断保存完成。
- 恢复计数元数据的 CPU 往返测试已通过，真实第 1／5 轮检查点已保存；新的异常保存与完整训练恢复链路仍需专门演练，不能据此宣称可应对任意崩溃。
- 仿真状态与随机数状态是否完整恢复需要另行核对。当前重新启动环境的续跑不能宣称与不中断执行逐步相同。

完整检查点写盘可能持续数分钟。GPU 利用率下降时，先检查日志、待完成的检查点和进程状态。进度页已区分“已保存”和“尚未写完”。

**已知限制：** 当前最终完成门控会扫描该作业的全部审计记录。若在同一目录恢复一个包含历史失败审计的作业，必须显式处理 attempt 边界与审计范围；不要删除失败证据，也不要假定恢复后会自动标记完成。OOM 等非审计异常也不保证能即时保存，只能从最近完整检查点恢复。

实现：[诊断 worker](../experiments/pirl_diagnostic_actor.py)、[控制器](../experiments/pirl_stability_controller.py)。

## 8. critic：验证冻结对象，而不是只看日志名称

只训练 critic 时，检查所有 `requires_grad=True` 的参数都属于 `value_head`，以及优化器实际包含哪些参数。学习率为 0、`policy_loss=0` 或日志字段叫 `actor/lr`，都不能独立替代这一检查。

本次补跑批次验证有 8 个可训练参数张量，全部属于 value head。前 5 轮没有远程审计收据，仍保留这一限制。

critic 的训练 loss 降低、单批 explained variance 变正，或训练采样成功率变高，都不足以证明策略提升。应同时报告 return variance、跨批次指标和独立闭环评估；很小的回报方差会使 explained variance 对误差更敏感。

当前 D1 的 critic-only 与 D2 的 PPO **分别从原始 Motus 初始化**，D1 权重没有传给 D2。因此这批实验不能用于声称“critic 预热提高了 PPO 成功率”。

## 9. RoboTwin 动作块：实际执行量、奖励和终止保持一致

策略提交 32 个动作，并不意味着环境执行了全部 32 个。成功、终止或截断可能发生在块内。

本地适配记录实际完成的控制点，将终止成功奖励放到最后一个有效动作位置。验收覆盖 0、部分、完整执行；强制第 1 步成功的测试得到 executed_count=1、reward_slot=0、reward_sum=1，并检查奖励在有效 mask 内。

强制成功场景仅用于集成测试，不进入训练池或性能评估。为了避免逐控制点计数引入额外 RGB 渲染，使用计数路径但暂时关闭中间图像采集，并在 finally 中恢复观察函数。

名义动作预算与实际执行量分别记录。当前每轮 `2 env × 512 = 1024` 是名义动作点，不是 1024 个独立 episode，也不是 1024 个优化器样本；32 步动作块会改变 batch 的计量单位。

证据：[终止集成验收](../.runtime/experiments/e5-pirl/terminal-integration-audit.json)、[奖励 mask 验收](../.runtime/experiments/e5-pirl/reward-mask-audit.json)。实现：[仿真适配](../experiments/pirl_extension.py)、[奖励计算](../experiments/pirl_rewards.py)。

## 10. 模型接入：零更新也可能改变策略

- 使用 checkpoint 自带归一化统计，固定三相机次序、关节／夹爪映射、delta transform、语言模板、去噪步数和执行窗口。
- 本地在 22 任务、132 个真实观测快照上核对了变换输入／输出，达到 1e-6 一致。这只验证变换，不代表整个模型前向完全一致。
- 网络数值接入曾检查 GELU 的精确实现与 tanh 近似；该因素没有解释全部误差，不能把改激活函数记作已经解释成功率差距的技巧。
- `bf16 → fp32` 只能改变存储类型，不能恢复之前舍入丢失的精度。E4 的零更新舍入与提示词变化均产生可测动作差异，需要单独对照。
- safetensors 可能省略 tied embedding 别名。转换到参考模型时显式恢复同一张量的别名，并列出允许缺失项，避免用宽松加载掩盖 action 权重遗漏。
- 当前权重往返验收中，动作生成相关权重完全一致；仅保留未使用的 action-expert token LM head。大权重使用 mmap 加载可以减少主机内存峰值，但仍需检查缺失／多余键。

证据：[变换审计](diagnostics/model/motus-transform-audit.json)、[初始化舍入／提示词审计](diagnostics/recap/e4-initialization-audit.json)、[权重往返审计](../.runtime/experiments/e5-pirl/weight-roundtrip-audit.json)。

## 11. 评估协议也是实验配置

### 执行窗口不能凭经验越改越短

相同 20 个冻结场景的前置对照：

| 任务 | 执行 8 步 | 执行 16 步 | 执行 32 步 |
| --- | --- | --- | --- |
| adjust_bottle | 0/20 | 6/20 | 9/20 |
| click_bell | 4/20 | 10/20 | 8/20 |
| stack_blocks_two | 9/20 | 12/20 | 7/20 |

这些小样本说明窗口有任务依赖性，不构成统一最优窗口结论。不得按最终测试成绩逐任务挑选窗口，再与固定窗口的 baseline 比较。

### 对照时固定与记录

- 同一初始化、冻结的场景种子、相机与语言、Clean／Randomized 设置、推理精度、采样模式和动作窗口。
- 对训练后模型继续使用已固定的参考评估后端；原生 ODE／SDE 零更新检查另列。
- 训练采集种子与冻结评估种子分开。前置开发场景已多次使用，最终论文结论需要方案固定后的独立测试与多个训练种子。
- 初始化错误、下载失败、进程 OOM、尚未完成的 episode 不计为策略失败。若历史协议使用替补种子，保存替补关系并冻结最终清单。
- 不同论文中的单任务／多任务 checkpoint 结果可作为外部描述性对照，不能直接当作受控训练消融。

证据：[窗口诊断](results/e3/motus-window-diagnostic.json)、[训练／评估种子分离](../.runtime/experiments/e5-pirl/seed-separation-audit.json)。

## 12. RECAP 适配：把偏离原论文的地方写进实验名称

本地 E4 是普通多任务 SFT π0.5 上的短预算适配，并非原论文的优势条件化预训练起点。还存在全局 top30% 与按任务阈值、positive-only 条件分支与完整二值条件、10% 与 30% 条件 dropout、32 与 50 步前瞻、冻结主干与完整训练等差异。

因此应先逐项核对、记录有效数据量和各任务正样本比例，再判断算法。单轮本身不是 RECAP 必然错误；增加迭代次数也不能替代单轮正确性和收益验收。

本地同一组 100 场景汇总为：原始 47、单任务 RECAP 29、多任务 RECAP 29、成功重放 25。该结果不能证明多任务一定比单任务差，也不能推出 RECAP 普遍无效。

原始 E1 的成功率 JSON 和抽样视频不能直接当作离线 RL 数据集。需要完整的观测、动作、奖励／终止与轨迹关系；本项目后来独立采集、划分并冻结了 E4 经验池。

证据：[RECAP 对照审计](diagnostics/recap/e4-recap-paper-audit.json)、[经验池清单](../.runtime/experiments/e4-offline-rl/pool-manifest.json)。

## 13. 仍待验证的算法调整

| 调整 | 当前可以说什么 | 还需要什么 |
| --- | --- | --- |
| PPO 每批更新 5 遍改为 1 遍 | 严格FP32组合已跑完：第5轮11/20、第10轮9/20，原始9/20 | 尚无持续收益；固定精度等其余条件再比较，不能单独归因于更新次数 |
| critic warmup | 独立 critic-only 的冻结验收通过 | 真正把预热 critic 接到 PPO 的受控对照；当前 D1/D2 不串接 |
| 冻结 VLM，只训练 action expert | 本地采用的资源约束配置 | 与其他更新范围的收益、保持和成本对照 |
| 增大 rollout／batch | 可能改变估计噪声与计算开销 | 预算与实际数据量匹配的实验；尚不能写成已验证技巧 |
| 继续 SFT、成功重放、SFT＋RL | 是应比较的后训练路径 | 同初始化、数据来源和成本核算下的受控结果 |
| LoRA、梯度投影、按任务调权重 | 研究候选 | 当前没有本项目的充分验证结果 |

旧单任务 PPO 的 20 场景配对结果为：原始 9 次成功，第 10／20／100 轮分别 7／2／1 次。看到早期持续退化，应先排查单任务稳定性，再扩多任务矩阵；单任务退化不能作为多任务冲突证据。

## 14. 下载、仿真与运行管理

- 固定 revision，保留 manifest、文件大小与哈希；完整校验后再写 `DOWNLOAD_COMPLETE`。文件存在或达到预分配大小不表示下载完成。
- Hugging Face 已登录与获得 gated 模型授权是两件事。不要反复重新登录解决尚未获得访问权限的问题，也不要把 token 写入日志、文档或代码。
- SigLIP2 的本地恢复工具使用有上限的并发 Range 下载：检查 206、Content-Range 和每段长度，最终验证全文件 SHA-256；保留已完成分段，避免从零下载。
- 本机 OIDN 曾报 invalid handle，使用关闭 OIDN 的渲染适配；mplib 场景不需要为了消除可选 import 提示而安装 cuRobo／PyTorch3D。是否失败以实际使用路径和进程退出状态判断。
- 跨训练、采集、评估控制器使用同一 `gpu-evaluation.lock`。先查现有 PID 与日志，再决定恢复，不盲目启动第二个同名队列。
- 每次变更保存配置、源码快照／哈希和原因。失败尝试独立归档，记录其采样与计算成本；E6 已归档四次中断、共 8,192 个名义动作点和 6 轮优化，不能把这些成本隐去。
- 监控日志更新时间、阶段、完整检查点和 episode 文件数。完整模型加载、写盘和评估切换都可能让 GPU 利用率暂时下降。

参考：[SigLIP2 续传实现](../bootstrap/resume_siglip_ranges.py)、[环境说明](../bootstrap/README.md)、[E6 修订与失败清单](../.runtime/experiments/e6-posttraining-diagnostic/plan.json)。

## 15. 当前诊断配置参考

这是当前单机短实验的配置摘录，不是推荐给所有任务的最优超参数。`pilot_*` 为项目自定义字段，必须使用项目扩展入口才会生效。

```yaml
actor:
  model:
    precision: fp32
    pilot_strict_fp32: true
    openpi:
      train_expert_only: true
  micro_batch_size: 4
  global_batch_size: 32
  enable_offload: true
  pilot_full_bucket_sync: true
  pilot_save_first_update: true
  optim:
    lr: 5.0e-6
    value_lr: 1.0e-4
    clip_grad: 1.0
    critic_warmup_steps: 0
algorithm:
  update_epoch: 1
  recompute_logprobs: false
weight_syncer:
  type: bucket
  bucket:
    bucket_size: 134217728
    bucket_dtype: null
    bucket_device: cpu
    is_agent: false
    load_instant: true
```

完整配置以[当前 D2 YAML](../.runtime/experiments/e6-posttraining-diagnostic/configs/ppo-update1-adjust_bottle-seed0.yaml)为准；不要仅复制摘录启动一个缺少其余配置的作业。

## 16. 下次开跑与异常排查顺序

1. 固定源码、环境、模型与数据版本，核对归一化和动作接口。
2. 做零更新推理／变换核对，再做真实 episode；一次成功只证明链路可运行。
3. 运行一个真实训练 batch，核对梯度范围、奖励／终止 mask、概率分母与精度。
4. 跨过“更新 → 权重同步 → 再次采样”，观察完整一轮切换的显存峰值。
5. 保存早期检查点，验证参数和优化步数恢复；标明是否恢复了环境与 RNG。
6. 跑短曲线与冻结开发场景评估，查看改善和退化任务，再扩大任务数／种子。

出错时先定位阶段：初始化／采样／反向／同步／写盘／评估。保留触发输入和日志；概率异常先做无更新数值对照，OOM 先查阶段内存和活引用，恢复问题先查完整检查点与步数元数据。

常用命令（在项目根目录执行；诊断脚本占用 GPU，需与训练错开）：

```bash
# 阅读当前状态；不启动训练
cat .runtime/experiments/e6-posttraining-diagnostic/status.json

# 原始与严格 FP32 的同权重数值对照
source bootstrap/env.sh
export PYTHONPATH="$PROJECT_ROOT/experiments:$PYTHONPATH"
python experiments/diagnose_pirl_logprob_numerics.py
python experiments/diagnose_pirl_logprob_numerics.py --highest

# 确认旧控制器已经退出、完成故障诊断后，才恢复现有 E6 队列
.runtime/venv-tools/bin/python experiments/pirl_stability_controller.py

# 只刷新实验页面
.runtime/venv-tools/bin/python experiments/render_pirl_stability.py
```

## 17. 后续新增经验的记录格式

每项写清：**日期／适用版本 → 现象 → 已确认原因与仍未知部分 → 修复 → 验收输入与数值 → 成本／局限 → 日志、配置和代码位置**。

将证据标为“受控验证”“组合修复后观察”“仅配置／代码核对”或“待验证假设”。不要把候选想法直接升级为训练技巧；更新算法结论时，补充独立闭环成绩和公平对照。

## 18. 评估停滞：进程存活不代表场景仍在推进

2026-11-06，E6 第10轮权重评估完成6个场景后，在冻结种子 `100100007` 的第64步失去心跳。正常已完成场景耗时约8—20秒，此处超过30分钟仍无进度；进程处于等待，现有 `SIGALRM(900)` 没有让评估退出。底层阻塞原因尚未确定，不将它记为策略失败。

处理时归档日志、心跳和6个已完成结果的SHA-256，终止该评估进程组，再从同一检查点、同一冻结种子清单续跑。已完成场景不重跑，不替换卡住的种子，不重启训练。恢复后原卡住种子已完成，心跳继续推进。

控制器现在从进程外检查心跳：连续900秒无进度则保存超时记录，先SIGTERM，10秒仍不退出再SIGKILL。模型重新加载期间从新进程启动时间计时，避免旧心跳误触发；每次控制器调用最多自动重试一次，持续失败则明确停止。普通非超时错误仍按原流程退出，不能无限重试直到得到成功结果。

CPU验收覆盖了旧心跳／新进程宽限、正常心跳、缺失心跳、重试使用相同参数、已有结果逐字节保留、两次超时后停止，以及忽略SIGTERM时的SIGKILL回收。后续仍需留意底层仿真／渲染卡住的原因。

证据：[本次恢复归档](../.runtime/experiments/e6-posttraining-diagnostic/jobs/ppo-update1-adjust_bottle-seed0/evaluation-10/recovery/20261106-223204/recovery.json)、[控制器](../experiments/pirl_stability_controller.py)。

## 19. 报完整短曲线，区分数值稳定与任务收益

2026-11-06，E6严格FP32＋CPU分桶＋每批1遍PPO完成10轮训练，10个批次全部通过概率一致性审计，最大逐元素误差为 `4.9114e-5`，保留原来的 `0.01` 门槛。当前成功尝试共320个动作块、10240个名义动作点，实际执行9791个动作点；此前中断尝试成本另外记录。

相同20个冻结开发场景，原始Motus为9/20，第5轮为11/20，第10轮回到9/20。第5轮新增3次成功、丢失1次；第10轮新增1次、丢失1次。原先5遍更新配置的第10轮为7/20，但这次同时改变了计算精度，不能把差异单独归因于减少更新。

**可复用经验：必须报告两个预定检查点，不能只挑第5轮的55%声称训练有效。** 工程链路已在这次10轮训练中通过，稳定提升还没有建立。20场景、1个训练种子的诊断也不能证明等效、退化原因或多任务冲突。

critic explained variance在10轮中有6轮为负，最后一轮为-0.537；这是继续检查价值估计的信号，不是已证明的性能退化根因。每轮只有一次优化器更新时，当前日志中的ratio/KL在更新前计算，接近1／0不代表更新后策略没有变化。

证据：[逐场景分析快照](results/e6/e6-progress-analysis.json)、[实时结果页](results/e6/stability-progress.html)、[第5轮结果](../.runtime/experiments/e6-posttraining-diagnostic/jobs/ppo-update1-adjust_bottle-seed0/evaluation-5/results.json)、[第10轮结果](../.runtime/experiments/e6-posttraining-diagnostic/jobs/ppo-update1-adjust_bottle-seed0/evaluation-10/results.json)。

## 20. GPU利用率与公平预算分开管理

E7五任务曲线实验固定1个训练进程。共享策略通过每任务2个环境、共10个环境并行采样；首批真实记录验证五个任务各32个动作块、1024个名义动作点。首批概率审计最大误差为 `1.5497e-5`，保留原0.01门槛。

截至启动验收，计算阶段已采样观察到99% GPU利用率、显存峰值约46 GiB；这是阶段观测，不是整段运行平均，也不能证明所有时刻都已高效。初始化、仿真、同步及检查点写盘单独影响利用率。

调度先测至少两轮训练的显存需求，再决定是否重叠1个评估进程。配置预留28 GiB给评估和12 GiB余量；纯评估阶段最多2路。若可用显存不足8 GiB，停止重叠评估并保留已完成场景，训练优先。该并发策略已经过CPU条件检查，实际并发吞吐收益仍需运行数据确认；采样得到的显存峰值不等于严格上界。

只在1／5／10／20／50／100轮保存完整训练检查点，减少不用于曲线分析的写盘。完整权重＋恢复状态约28.2 GiB／检查点，因此写盘可能明显降低短时GPU利用率；页面区分已完成更新、已保存检查点和仍在写入的检查点。

所有单任务与共享策略仍使用相同microbatch4、global batch32和1遍PPO。共享策略每轮5次优化器更新；五个单任务合计也是5次。资源调度不静默改变学习率、更新次数或每任务名义预算。实际动作点另行记录，不能用不同数量的任务平均来填补尚未评完的曲线。

证据：[冻结计划](../.runtime/experiments/e7-five-task-curves/plan.json)、[共享策略批次审计](../.runtime/experiments/e7-five-task-curves/jobs/multi-mt5-seed0/batch-audit.jsonl)、[GPU时间序列](../.runtime/experiments/e7-five-task-curves/gpu-utilization.jsonl)、[曲线页](five-task-curves.html)。

## 21. 并行评估必须同时约束主机内存和GPU显存

2026-11-07，E7共享策略完成59轮后在下一轮同步／采样衔接阶段中断。日志明确记录Ray因主机内存压力杀掉3个工作进程，之后出现Gloo连接关闭；控制器在约30分钟没有新进展后结束队列。原并发策略只为GPU显存留余量，未约束主机／容器内存，不能据此保证整体资源安全。

Ray记录的限额为134,661,529,600字节（约125.41 GiB），95%门槛为127,928,451,072字节；触发时用量128,091,848,704字节。并发评估约14.24 GiB，383个未选中空闲worker的USS合计约6.82 GiB。这些是占用线索，不能仅凭进程表断定唯一峰值来源。

恢复前应同时检查主机／容器可用内存与GPU显存，记录初始化、权重同步、保存和并行评估的内存峰值；限制无用worker数量。在日志出现Ray OOM／工作进程退出时及时失败，避免继续空等完整看门狗时限。不要关闭OOM保护来强行继续。这些是待实施的修复要求，本轮分析没有修改调度或重启训练。

本次最后完整保存检查点为第50轮，第51—59轮已更新但未完整保存。后续从50轮恢复时，重复交互、额外成本及环境／随机数不连续要另记，不默认等同于无中断100轮。

证据：[E7分析及日志证据哈希](results/e7/e7-analysis.json)、[原始中断记录](../.runtime/experiments/e7-five-task-curves/failure.json)、[训练日志](../.runtime/experiments/e7-five-task-curves/jobs/multi-mt5-seed0/training-1793977216689382582.log)。

## 22. 总平均、能力保持与critic诊断分开报告

E7共享策略在相同100个开发场景的成绩为原始47、第5轮48、第10轮47、第20轮41、第50轮32。第5轮按铃增加10次成功，其余任务合计减少9次，平均值掩盖了技能间不同方向的变化。第50轮新增13次成功、丢失28次原有成功，只保留原始47次成功中的19次。

59个批次的更新前概率一致性审计均通过，最大误差0.000102。第21—50轮critic解释方差全部为正、均值0.327，但开发场景表现继续下降。审计通过和critic指标改善都不等于策略收益得到保证；它们各自只验证特定范围。训练分布持续变化，也不能直接把解释方差变化解释为固定测试分布上的价值质量变化。

应同时报告每任务成功率、宏平均、新增成功、丢失成功与全部预定检查点。未运行的单任务策略不画基线点冒充学习曲线；缺失第100轮不填0。400条检查点评估重复使用100个场景，不能算成400个独立样本。当前五个单任务尚未启动，因此只能报告共享后训练的技能收益不均与保持问题，不能宣称已证明多任务梯度冲突。

证据：[逐任务结果与分析](results/e7/e7-analysis.html)、[逐场景配对CSV](results/e7/e7-paired-scenes.csv)。

## 23. E7恢复修复：双重资源门槛与训练状态回退

2026-11-07，已将第21节的资源修复落地并启动恢复队列。通过Ray已有的 `RAY_OVERRIDE_RESOURCES` 将CPU资源设为16、对象存储设为2 GiB；CPU远程任务实测配置生效。原来按384个逻辑CPU预启动空闲worker的开销可因此避免，训练每任务环境数、batch、算法、学习率与评估场景均未调整。保留Ray 95%的OOM保护。

评估启动须同时满足GPU预留28 GiB＋12 GiB余量、主机／容器内存预留20 GiB＋24 GiB余量；正在加载的评估也计入预留。训练重叠评估仍须先观察至少2个训练轮次。可用RAM低于18 GiB、内存压力达到85%或GPU剩余不足8 GiB时，撤下并行评估，保留已完成场景；训练独占时达到92%内存压力则明确失败。致命Ray OOM、worker退出或通信断开由增量日志检查及时捕获，不再只等待30分钟无进展。

清理仅针对所启动训练进程的后代，记录PID与启动标识防止误杀复用PID；不执行全局 `ray stop`。纯评估队列因资源门槛暂未启动时继续等待，不能将“没有运行中的评估进程”误判为全部完成。

第50轮检查点包含模型与Adam优化器状态，更新计数为250。恢复后首批应对应 `global_step=50, optimizer_steps=250`，完成后为第51轮。旧第51—59轮记录完整保留；这些回退尝试共46080个名义动作点、44794个实际动作点，计入额外成本。当前曲线仅使用有效训练路径，避免恢复时仍显示旧第59轮或把旧更新当作新更新。环境与随机数连续性不保证。

验收：8项CPU行为检查通过；Ray实际资源及远程任务检查通过；用真实历史日志模拟回退，正确得到50轮、256000个保留名义动作点和上述额外成本；400个已完成评估文件、6份训练配置和9份算法源码快照哈希均保持不变。训练稳定性还需观察后续真实运行，不能仅凭这些检查声称OOM永久解决。

证据：[运行资源设置](../.runtime/experiments/e7-five-task-curves/runtime-policy.json)、[验收记录](../.runtime/experiments/e7-five-task-curves/recovery/20261107-091719/validation.json)、[恢复前备份清单](../.runtime/experiments/e7-five-task-curves/recovery/20261107-091719/manifest.json)、[恢复启动记录](../.runtime/experiments/e7-five-task-curves/recovery/launch-latest.json)。

恢复后的实测验收：已连续完成新的第51、52轮，首批更新前优化器计数确为250；两批概率审计最大误差为 `3.1471252e-05`，均低于0.01。验收时可用主机内存约43.2 GiB，控制器与训练进程均存活。详见[真实训练验收](../.runtime/experiments/e7-five-task-curves/recovery/20261107-091719/live-validation.json)。这验证了本次恢复链路，长时间稳定性由后续队列继续监测。
