# 原始多任务 checkpoint 的评测实验

训练与排障经验见 [takeaway.md](../.research/takeaway.md)：记录已验证修复、诊断配置、待验证调整及对应证据，供后续实验复用。

直接使用 `motus-robotics/pi0.5_robotwin2`。该模型已在 RoboTwin 50 个任务上联合微调；本阶段不进行 SFT，不创建优化器，不更新权重。

当前运行：`.runtime/experiments/e1-reference-clean/`，22 个有 RLinf 官方评测种子的任务，每任务前 100 个种子，Clean 设置，共 2200 episodes，两个并行工作进程。使用原始 OpenPI PyTorch sampler、FP32、32 步动作窗口、10 次去噪；任务语言使用官方 seen 模板并按 seed 固定。关闭 OIDN，规划器构建使用 mplib，但不重新执行专家轨迹。此协议不直接等同于官方 leaderboard。

原始推理实现位于独立 `.runtime/reference-transformers/`，启动工作进程时优先使用；图像、状态与动作归一化沿用 RLinf 的 ALOHA 变换并指定 checkpoint 自带统计。

每个 episode 保存种子、指令、成功状态、动作步数和时间。基础设施错误单独记录并立即停止该任务，不计入策略失败，也不替换测试种子。每任务保存前 3 个和每 25 个 episode 的视频及初始三相机观测。已完成 episode 支持续跑。

入口：`source bootstrap/env.sh && python experiments/launch_reference.py`。controller.json 保存进程 PID；contract.json 固定协议；logs/ 保存逐任务日志；clean/<task>/summary.json 保存累计结果。

E0 结果：`.runtime/experiments/e0-parity/`。原始参考与 RLinf 原生推理在固定合成观测下尚未通过预设数值阈值，不能视为已对齐。精确 GELU 与 tanh 近似的差异已做受控试验，但未解释全部输出差异；运行适配层保存在 compat.py，当前参考评测不使用它。原生后端的正式评测由门控暂停。

另一次专家规划试运行在 `.runtime/experiments/e1-pipeline-check/`，MPLib 初始碰撞导致专家验证不可靠，已停止。其数据不进入基线。其余 28 个任务及 Randomized 评测需先完成相应种子与规划器验证。

原始参考推理试运行：adjust_bottle 两个固定种子均成功，仅作为链路验证，不能作为正式性能结论。RL 训练尚未启动。


## E4：Offline RL 首轮批量实验

目录：`.runtime/experiments/e4-offline-rl`；计划见 `plan.json`，实时记录见 `.research/offline-rl.html`。

- 固定起点为已发布的 Motus 多任务 π0.5，五个锚定任务：adjust_bottle、click_bell、stack_blocks_two、place_empty_cup、place_container_plate。
- 冻结池已审计：250 条专家示范 + 200 条自主轨迹；共 119294 帧。每任务自主轨迹 32/8 划分训练与价值验证，示范 45/5；与测试种子无交集。
- 算法复用 RLinf RECAP：官方 SigLIP2 与 Gemma3-270M 价值模型、32 步优势、正优势条件化 flow matching。快速验证冻结视觉语言主干，但保留跨模态投影、critic expert 与 action expert 更新；不声称完整复现论文训练预算。
- `launch_recap_workflow.py` 等待固定池和两套官方权重完整校验，随后运行价值验收、1000 次价值更新、共享优势标注，以及 5 个单任务策略、五任务联合策略和成功重放对照。
- 单任务 50 次优化更新，五任务 250 次；有效 batch 32，记录实际每任务样本暴露量；每任务先用 20 个冻结测试场景配对评估。成功重放可以提前运行，不属于 RL。
- `gpu-evaluation.lock` 协调模型评估、采集与训练；每个策略还有独立训练锁，已完成的作业会跳过。当前首轮不自动扩大到 10—50 任务。
- 官方模型下载锁在 `bootstrap/recap-models*.lock.json`；Gemma3 需要用户自行取得 HF 授权。认证信息不写入项目协议或日志。

### E5: piRL Flow-SDE + PPO

Run directory: `.runtime/experiments/e5-pirl`; live report: `.research/pirl-progress.html`.
The persistent queue trains five specialists and one shared five-task policy at seeds
0/1/2, starting from the released multi-task Motus SFT checkpoint. Each task receives
two parallel environments and 100 collection iterations of 512 nominal action points
per environment. Actual reached points are recorded separately in
`online-chunks/interactions.jsonl`. The shared policy's total budget matches the sum
of the five specialists. Checkpoint 20 of the shared policy is an additional low-budget
point, not the equal-total-budget comparison.

Each PPO run has a success-only FM replay control, initialized independently from
Motus and trained on that PPO run's recorded experience with matched optimizer-step
and batch counts. This is a same-experience offline control, not a separate online
success-replay collector. Artificial forced-success fixtures are outside all job
recording paths and never enter either training pool.

Final policies use 100 frozen E1 scenes per task; intermediate checkpoints 10/20/50
use a predetermined 20-scene curve and are not used to tune hyperparameters. All
final PPO and replay policies additionally evaluate all 22 original tasks. Checkpoints
are converted to original OpenPI weights and evaluated using the same FP32 reference
sampler as E1 (32 predicted/executed actions; 10 denoising steps). Native zero-update
inference is a separate 100-scene integration diagnostic (48/100 versus reference
47/100), not a learning result. Original full local baseline remains 566/2200 (25.73%).

Preflight artifacts include likelihood recomputation, loss backward, strict checkpoint
loading, exact weight-layout roundtrip, full training/test seed separation, mixed-task
reset/prompt assignment, a real one-step PPO update and a forced early-success
reward/mask integration check. `pirl_extension.py` applies project-local activation,
renderer, prompt, balanced-task, waypoint-count and terminal-reward adaptations.
Pinned third-party source snapshots are unchanged. Source v1 and amendments v2-v4
are retained; initialization attempts aborted before formal updates are preserved.

The long-horizon scene audit is recorded in `long-horizon-feasibility.json`. No
continuous long-horizon policy result exists yet: independent-reset tasks are not
counted as continuous subtask composition.


### 2026-11-06：收缩为快速前置验证

首个 `single-adjust_bottle-seed0` 已训练100轮；E1冻结前20场景的原始成功数9，100轮权重成功数1。完整队列剩余17个PPO作业和18个成功重放作业暂不启动，原计划、状态和控制器信息归档在 `e5-pirl/pilot-handoff/`，所有检查点及已完成场景记录保留。

`pirl_pilot_controller.py` 只并行评估已有第10、20轮权重，各20个相同场景，不启动训练。首次交接使用 `--take-over`，故障恢复直接运行，不重复交接。实时页面仍为 `.research/pirl-progress.html`，机器可读汇总为 `.research/pirl-pilot-results.json`。`pilot-plan.json` 禁止完整队列自动续跑。

拟议下一步为调整瓶子、叠两块积木、容器放盘，1个训练种子、20轮，3个单任务策略对1个共享策略。训练配置一致时可复用瓶子第20轮，还需3个训练作业；三个独立策略合计与共享策略名义动作预算各61440，另记录实际执行量。三任务原始配对参照30/60，不混用五任务47/100。若修正训练配置，则受影响策略须重新运行。先分析诊断结果再启动该矩阵，可选增加1个共享同经验成功重放对照。

这批场景用于开发和前置诊断，所有预定检查点均报告；论文最终结论需在方案固定后使用独立测试集和多个训练种子。当前单任务退化不能归因于多任务冲突。

快速诊断结果已完成：同20场景，原始9/20，第10轮7/20，第20轮2/20，第100轮1/20。退化在前20轮已出现，原配置不直接扩展；先排查单任务训练稳定性。该结果不是多任务冲突证据。


### E6：后训练稳定性短诊断

入口 `pirl_stability_controller.py`，计划 `.runtime/experiments/e6-posttraining-diagnostic/plan.json`，实时页面 `.research/stability-progress.html`。D0复用参考／原生ODE，新增原始权重SDE20场景；D1从原始权重冻结actor训练critic5轮；D2另从原始权重开始，每批1遍PPO、10轮，评估5／10轮。D1与D2不串接权重。按名义采样预算对比旧5遍PPO的前10轮，策略变化后的轨迹不同。

项目内 `pirl_diagnostic_extension.py` 仅测量真实训练batch的概率重算差异，保留原rollout logprob作为PPO分母；超过0.01时停止。公开recompute选项在固定RLinf版本只支持OpenVLA，首次启动被校验拒绝，未进行优化；source-v2使用OpenPI专用测量回调，原始日志与源码快照保留。第三方仓库未修改。

用户已确认可使用灵巧手或双臂灵巧手实机；研究判断见 `.research/unified-posttraining-assessment.html`。联合损失、按任务调数据比例、共享多任务后训练均有先例，新机制仍待验证。

E6 source-v3补充：首5轮critic-only已完成，但远程动态方法替换未生成批次审计文件，未将其标记为通过。改用项目内显式 DiagnosticEmbodiedFSDPActor 子类，从第5轮检查点续跑1轮仅用于审计验收；不会重跑原5轮，也不把缺失审计补记为成功。PPO对照仍从原始Motus开始训练10轮。

E6 source-v3实批次验收已通过：第6轮续跑生成1条审计记录，32个动作块，概率对数最大重算差0.00475395，8个可训练参数张量均属于value_head；保留原rollout概率分母。首5轮仍不追记审计。critic阶段完成标记已生成，独立的10轮PPO更新一次对照已启动，将自动评估第5／10轮各20场景。该核验不证明critic已学好，也不证明存在多任务冲突。

E6 config-v4补充：PPO首轮已通过批次审计（最大概率对数差0.00482655），但在第2轮前的patch权重差分同步时CUDA OOM。未保存checkpoint，整次尝试归档到 `e6-posttraining-diagnostic/aborted-attempts/ppo-update1-patch-sync-oom/`，其中1轮／1024名义动作点作为额外实际消耗记录。替换为RLinf官方FP32分桶同步（128MiB目标、dtype不转换），重新从Motus初始化；没有修改学习目标、学习率或更新次数。实时页以新尝试为准。

E6 source-v5补充：默认分桶尝试在首轮优化前被预设概率差异门槛拦截（max0.01088595>0.01），未修改阈值。该尝试的1轮采集／0轮更新已归档到 `aborted-attempts/ppo-update1-selective-bucket-audit`。分桶默认只同步可训练参数和持久buffer，而原patch首次同步全部权重；现用项目worker选择全部状态并继续采用官方分桶传输，记录同步范围。该覆盖差异是待验证因素，不宣称已经解释异常。若再次失败，将保存触发批次供离线复查。

E6 source-v6恢复记录：全量分桶尝试完成4/10轮后，在第5轮更新前因max logprob gap0.0119457>0.01停止；14336项仅1项超阈值。此前保存间隔5轮，未留下第4轮权重。已保存批次与4轮日志并归档。固定原始Motus权重和真实输入，无优化更新，仅将batch4改为batch2：默认TF32的最大逐元素差0.009453、PPO比率最大差2.6342%；禁用TF32后分别为0.00003934和0.012207%。所以同时在actor/rollout构造模型后设置matmul highest与cudnn.allow_tf32=False，保留0.01门槛。新的审计中断先保存当前模型、优化器和scheduler；diagnostic-state.json记录并恢复optimizer_steps，控制器仅恢复有完整收据的检查点。旧4轮不能恢复，当前从原始Motus重新开始。该精度变化意味着与历史update5的差异不能单独归因于更新次数。

E6 source-v7：strict FP32在真实训练首批验收通过（max2.48e-5），之后在下一轮rollout重新装载模型时OOM，失败时actor占用26.48GiB。改用官方CPU分桶暂存（保持原FP32数据），并在基类同步方法返回、局部GPU state_dict解除引用后清理缓存；memory-phases.jsonl记录每次同步和训练后的实际显存。增加第1轮检查点，正常第5／10轮保存保持；本次额外失败1轮／1024名义动作点已独立归档。


### E7：收敛为五任务单任务／共享RL曲线

用户于2026-11-06要求将前置实验收敛为五个单任务策略与一个五任务共享策略，并充分利用现有GPU。当前队列位于 `.runtime/experiments/e7-five-task-curves/`，入口 `pirl_five_task_controller.py`；可视化见 [五任务曲线](../.research/five-task-curves.html)，协议见 [冻结计划](../.runtime/experiments/e7-five-task-curves/plan.json)。旧E5全矩阵仍保持关闭。

- 任务：adjust_bottle、click_bell、stack_blocks_two、place_empty_cup、place_container_plate；六个策略均从原始Motus独立开始，1个训练种子，各100轮。
- 统一Flow-SDE＋PPO、严格FP32、1遍更新、microbatch4、global batch32、冻结VLM，无critic预热，不添加SFT重放或其他算法对照。
- 共享策略10个环境，每任务2个；每任务每轮1024个名义动作点。单任务策略各102400；五个单任务合计与共享策略总计各512000。共享策略每轮5次优化器更新，五个单任务合计也是5次。
- 第1轮保存恢复检查点；第5／10／20／50／100轮保存并评估，每任务20个相同E1开发场景。全部预定节点均报告，不因早期成绩挑选最好节点。
- 主图为五张逐任务曲线和一张五任务宏平均曲线，横轴按每任务名义交互量对齐。未完成评估不画成功率点，宏平均须等全部五任务完成；另绘制critic和GPU使用曲线，实际动作点单独统计。
- 同一GPU锁下保持1个训练进程。完成至少2轮并测得足够显存余量后，允许重叠1个评估；纯评估阶段最多2个。评估无心跳900秒后终止并仅重试一次，已有结果和原种子保留；显存紧张时优先让训练继续。
- 训练源码单独快照、配置锁定。当前修改只增加稀疏检查点节点和日志中的global_step；各任务的奖励、动作mask与推理协议沿用已验收路径。
- 这是探索性对照，更多训练不保证收益；已有critic波动不是已确定的因果解释。多任务曲线下降也需进一步排查混合采样、价值估计与优化差异，不能直接当作梯度冲突证据。

恢复前先查控制器PID，避免重复队列。恢复会跳过完整作业／已完成场景，按完整权重及optimizer_steps元数据恢复训练；环境和RNG的逐步连续性不作保证。运行页面由 `render_pirl_five_task.py` 在CPU生成，`--watch`会在队列最终结束或报告错误后补齐最后一次刷新。
