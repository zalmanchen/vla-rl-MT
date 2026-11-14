# RoboTwin 2.0：从SFT到RECAP

新服务器使用一份[总YAML](../configs/pipelines/robotwin2_sft_recap.yaml)和一个[shell入口](../scripts/run_sft_recap.sh)。它编排锁定RLinf的原生SFT、回报计算、value训练、advantage标注和CFG policy更新，自动连接阶段权重。版本是RLinf `e1a82eb`＋记录的兼容补丁、**RoboTwin 2.0 `dca9ec6`**，起点为锁定的Motus多任务SFT π0.5。

这是公开RECAP算法在RoboTwin 2.0上的开发基线。它不是LIBERO实验，不承诺论文／博客的48.8%→66.5%，也尚未实现我们的SFT／RL协调候选。默认128步仅验收接口；八卡正式数据、预算和多种子协议仍按[八卡任务书](prerequisite-8gpu.md)另行锁定。训练循环、优化器和检查点继续使用官方实现。

## 新服务器先准备什么

按[安装说明](setup.md)准备主Python环境、Motus、SigLIP2、已授权的Gemma3和RoboTwin 2.0资产；按[仿真验收](simulation.md)检查实际GPU渲染。root并非此环境的禁止条件。已有Docker验收只覆盖LIBERO，不能直接据此宣称RoboTwin 2.0容器已通过。

```bash
bash bootstrap/install.sh
source bootstrap/env.sh
python bootstrap/prepare_sources.py --verify-only
```

总YAML中模型路径、数据路径、任务、训练种子、每阶段步数、batch、GPU和存储预算均集中在`workflow`。通过`--gpus`选择同一机器上的RLinf设备rank，训练阶段不额外套torchrun；advantage标注、格式检查和评测使用其中第一张卡。GPU rank按RLinf宿主设备编号解释，保持`CUDA_VISIBLE_DEVICES`未设置，避免二次编号。新服务器从本地路径重新生成计划，不复制旧机器含绝对路径的运行计划。

当前只有数据准备充分后才能运行训练。模型、数据、环境、运行计划和检查点均不随Git推送。

## 三个数据输入

| YAML字段 | 内容 | 使用范围 |
| --- | --- | --- |
| `datasets.sft` | 一个已合并的多任务、可靠成功LeRobot v2.1数据集；保留每条完整instruction | SFT、value准备、离线policy数据 |
| `datasets.rollout` | 固定行为策略的预定训练尝试，保留自然成败比例 | value与离线policy，不回流新rollout |
| `datasets.validation` | 独立episode与场景族的成败轨迹 | 仅critic验证，不进入policy训练或阈值估计 |

SFT入口要求单个数据集路径，不能直接传多个任务目录列表。每个数据集包含所有配置的训练任务、三相机、14维绝对关节动作／状态、正确FPS、`task_index`及可核验的`is_success`。输入按**逐控制步**保存，归一化、关节delta与gripper转换由官方OpenPI数据管线完成。

现有192条本机母池只记录chunk决策边界图像，**不满足这个逐控制步入口**；不能把它改名或补帧后直接使用。旧pilot虽然可读，也缺充分场景／终止审计，不能冒充新协议数据。需要补合格转换或采集；原始资产、数据和镜像下载并不等于训练数据已经合格。通用数据合同见[研究协议](multitask-critic.md#sft到rl的数据构建)。

每个数据集另需`meta/post_training_provenance.json`。由采集／转换审计生成；以下仅展示字段，不能用手写声明替代缺失证据：

```json
{
  "schema_version": 1,
  "observation_sampling": "control_step",
  "episodes": [{
    "episode_index": 0,
    "episode_id": "stable-original-episode-id",
    "task_id": "click_bell",
    "instruction": "the exact instruction in meta/tasks.jsonl",
    "scene_family_id": "stable-scene-family-id",
    "simulator_seed": 12345,
    "seed_index": 200,
    "split": "train",
    "source_type": "policy",
    "behavior_checkpoint_sha256": "the actual behavior weight SHA256",
    "checker_version": "the checker source revision or SHA256",
    "success": true,
    "termination_reason": "success",
    "terminated": true,
    "truncated": false,
    "alignment_verified": true,
    "final_observation_recorded": true
  }]
}
```

验证集的`split`为`validation`。`seed_index`仅在源头有记录时填写，不能从模拟器seed推造；最终保留索引`[500,550)`不得用于这些输入。专家来源使用`source_type: expert`和真实`source_sha256`，不编造行为模型hash。失败终止为`failure`或预定benchmark边界`benchmark_limit`；外部中断和未知结果不能直接标失败。场景族标识必须可跨数据源追溯，相同场景的变体不能分别放进训练和验证。

检查器会核验真实parquet中的episode／帧索引、标签、instruction和总帧数，并检查训练／验证及开发评测的初态重合。原始数据文件做SHA256绑定。它不能自动证明人为填写的场景族、动作对齐或checker结果真实，仍需源数据审计。运行目录复制metadata，原始图像／视频／parquet通过链接读取，回报与优势只写私有metadata；不复制整套图像。

## 启动与阶段衔接

先生成两步工程检查计划，不训练：

```bash
bash scripts/run_sft_recap.sh plan \
  --config configs/pipelines/robotwin2_sft_recap.yaml \
  --run-dir "$PWD/.runtime/runs/sft-recap-smoke-seed0" \
  --gpus 0 1 --seed 0 --smoke

bash scripts/run_sft_recap.sh check \
  --run-dir "$PWD/.runtime/runs/sft-recap-smoke-seed0"
```

`check`检查数据与模型；缺项会明确失败。`run`还核验锁定源码与空间，并实际执行GPU程序。默认预留100GiB净空及250GiB新增峰值；正式大任务先估算并更新峰值。两卡与八卡的NO_SHARD、micro batch和显存仍需分别实测，CPU配置组合通过不代表GPU训练通过。

数据、模型、渲染及空间合格后执行：

```bash
bash scripts/run_sft_recap.sh run \
  --run-dir "$PWD/.runtime/runs/sft-recap-smoke-seed0"
```

| 顺序 | 阶段 | 交接 |
| --- | --- | --- |
| 1 | 官方原生SFT | 从Motus开始，默认训练完整policy，保存固定末点 |
| 2 | `export-cfg`、`check-cfg` | 官方键转换、所有native张量精确往返；实际观测／固定噪声的输入和动作检查 |
| 3 | 三组SFT末点开发评测 | native、CFG无guidance、CFG正条件零更新；分开解释后端与条件扰动 |
| 4 | 官方returns | 训练及critic验证各写独立带tag sidecar |
| 5 | 官方value训练 | 独立SigLIP2＋Gemma3骨干，不是原生policy特征head |
| 6 | 官方advantage标注 | 只使用训练混合，连接固定value末点 |
| 7 | 官方CFG policy训练 | 必须从步骤2导出的同一个SFT末点初始化 |
| 8 | 训练后CFG正条件开发评测 | 与前面的同场景／噪声设置配对；不回流数据 |

Motus使用32步模型窗口，而锁定上游RoboTwin TrainConfig默认50步。`vla_rl_mt.recap_workflow.register()`只注册一个32步克隆`pi05_aloha_robotwin_h32`，供driver和worker使用；没有改写模型／loss／优化器／训练循环。14维环境动作、32维模型动作、三相机、discrete state、Motus统计保持一致。兼容补丁与此配置适配均需在复现说明中披露。

**奖励沿用锁定RLinf的DynamicReturn**：每步−1、成功终局0、失败终局−300、γ=1；共同return范围默认[−1000,0]并检查是否会截断。它不同于方法线的二值终局奖励，不能混在同一算法对照中。advantage使用32步lookahead、训练混合全局quantile 0.3；SFT数据在官方标注器中强制positive，policy采用官方CFG正条件与0.1无条件dropout。raw weight `[1,1]`在上游balance逻辑下不等于任务或数据集50/50，需从真实曝光继续审计。

## 八卡与断点使用

为八卡复制并编辑总YAML中的**正式数据、任务、预算与batch**，再生成新计划；仅改变`--gpus`不会把开发预算变成正式实验。例如：

```bash
bash scripts/run_sft_recap.sh plan \
  --config /path/to/your-reviewed-robotwin2.yaml \
  --run-dir "$PWD/.runtime/runs/prerequisite/sft-recap-seed0" \
  --gpus 0 1 2 3 4 5 6 7 --seed 0
```

seed1、seed2分别创建独立目录；共享不可变原始数据，保持相同协议。扩大任务时，为每个任务提供已核验的instruction／checker及评测协议，目前默认协议覆盖五个P0候选任务。SFT／RL协调候选的A—D比较尚需另立配置，不能把普通基线文件改名当候选。

`run --until sft`或`run --until value`在对应阶段完成后停下；再用同一目录执行`run`会校验并跳过已完成阶段。配置、源码或已完成产物变化会拒绝续跑。不覆盖未登记／部分写出的检查点；失败阶段保留日志，修复后按报错审查并使用新计划。这里的自动续接是**阶段间续接**，不自动推断中断训练的optimizer恢复点。

```bash
bash scripts/run_sft_recap.sh status --run-dir "$PWD/.runtime/runs/sft-recap-smoke-seed0"
tail -f .runtime/runs/sft-recap-smoke-seed0/sft.log
```

每个训练阶段只保存一次末点，仍包含官方DCP／optimizer。`sft/checkpoints/global_step_<N>/actor`为原生末点，`cfg-initial/`为交接导出，`policy/checkpoints/global_step_<N>/actor`为RL末点；使用同一`assets/norm_stats.json`。运行计划、配置快照、日志、输入hash与评测结果保留在该运行目录。工程smoke的大权重验收后按[空间规则](development.md#空间管理)清理；不归档整套重产物，不删除原始模型／数据。阶段结束只代表该开发链执行完成，不能替代正式学习曲线、多种子及未更新任务保持实验。
