# Project conventions

## Research scope and current work

- Study coordinated multi-task SFT→RL of an existing shared Motus pi0.5 VLA: reduce harmful task interference, improve the RL starting point and learning efficiency, and retain prior capabilities. SFT-only and Motus-direct-RL are controls, not competing project objectives.
- Ordinary SFT→standard RECAP is the complete baseline. Main stage controls are A ordinary SFT/ordinary RL; B coordinated SFT/ordinary RL; C the exact A actor/coordinated RL; D the exact B actor/coordinated RL. Define one concrete candidate before launching these names as experiments. Initially hold critic preparation fixed.
- Critic warm-up is optional. Compare early stop-grad versus standard post-SFT critic training with the same actor, data and total value budget. A random critic immediately used for RL is not the standard baseline. Shared value gradients need a matched stop-grad control.
- SFT pre-RL competence, critic readiness, RL learning curves and total SFT+critic+RL cost are distinct outcomes. Higher starting success alone does not prove better intrinsic RL learnability. Fewer offline updates do not establish fewer environment interactions.
- Current plans and roles: `docs/experiment-guide.md`, `docs/method-2gpu.md`, `docs/prerequisite-8gpu.md`. `configs/multitask/two_track.json` is planning only, not a launcher. Current two-host task sheets are ready. Local real control-step collection has been launched; inspect its runtime status before further work. No new actor training has been launched by this data controller, and the 8-GPU host has not been connected.
- Local 2x48GB GPUs0/1 develop the two-task interface and one candidate; 8x48GB prepares formal multi-task baselines, independent actor seeds, scale and retention. Start remote work only when actually requested with an available connection; preparing task sheets does not send messages or start jobs.

## Sources, implementation and validation

- Use pinned official `third_party/RLinf` and `third_party/RoboTwin`. Locks: RLinf `e1a82ebf0c1af69da6ab9b1519f5f3692fe33566`, RoboTwin `dca9ec682688821c42944eacd7f5dd9bfb15396f`; the recorded RLinf compatibility patch is separate. `third_party/sources.manifest.json` and exact-file ignore allowlists describe the vendored snapshots. They are not nested Git repositories.
- Model initialization is Motus `effc6e1a99bd4d0edf9447e8ba313def67649cfe`; full model/norm/action definitions are in `configs/multitask/phase0.json`. Do not rebuild another foundation model or substitute a historical W1 actor without explicitly changing the experiment.
- Extensions belong in `src/vla_rl_mt/`, configuration in `configs/`, execution in `scripts/`, preparation in `bootstrap/`. Reuse official model/worker/runner/optimizer/checkpoint interfaces; do not copy the training loop.
- Portable ordinary baseline: `configs/pipelines/robotwin2_sft_recap.yaml`, `scripts/run_sft_recap.sh`, and `docs/sft-to-recap.md`. `recap_workflow.register()` only clones the upstream 50-step RoboTwin TrainConfig to the Motus 32-step window. Require audited control-step LeRobot data; the old chunk-image mother pool is ineligible. Normalize reference checkpoint layout and verify every native-to-CFG policy tensor round-trips.
- Existing native method workers support independent single-GPU NO_SHARD jobs. Do not assume they support multi-GPU FSDP. The frozen base is deliberately outside those FSDP state dicts; `expert_policy.py` registers and rebinds the action expert when it is trainable.
- Preserve sealed past protocols and source/model locks. New comparisons get new plans/output directories. Historical code/data/source snapshots under ignored runtime are evidence, not places to implement changes.
- Native and official CFG backends have diagnosed numerical differences; zero-update input/action checks and matching inference settings are required. Native weighted-FM/adapter studies are RECAP-style adaptations, not unmodified official RECAP reproduction.
- Set Ray namespaces before `validate_cfg` initializes Cluster. Use owned job queues and do not terminate unrelated Ray clusters. SAPIEN task setup can change cwd; resolve experiment paths absolutely before setup.
- Follow `docs/development.md`. For launcher/runtime edits run `python -m unittest discover -s tests -v` in the sourced environment and relevant real smoke checks. For docs-only changes validate links/JSON/diff; do not launch training as a publication check.

```bash
source bootstrap/env.sh
python bootstrap/prepare_sources.py --verify-only
python -m unittest discover -s tests -v
```

## Data and scientific controls

- Canonical data contract: `docs/multitask-critic.md`. Freeze offline data before comparisons; evaluation never returns to training. An online branch must declare training resets, behavior versions and total interaction cost separately.
- Keep full-episode/scene-family splits, source and behavior-policy hashes, complete instructions, actions/chunk boundaries, actual lengths and final observations, reward/checker and true termination/truncation reasons. A seed-list index is not the simulator seed itself.
- Local development mother pool is `.runtime/datasets/label-exposure-v1/manifest.json`, hash `feb3f15a72b21a5c9370a1fce9f177ec0b5fb76f500ba4ce7e7d459bfda48ad0`:192 original-Motus training attempts at seed-list indices[200,296), plus24 old development episodes. `sft-recap-control-v1/native-index.json` now seals32-success/task SFT indices, the192-attempt experience and old24 development episodes without image copies. These remain native chunk-data views, not official step datasets.
- New real control-step preparation uses `configs/multitask/sft_recap_data.json`, `bootstrap/prepare_sft_recap_data.py`, and the owned `scripts/run_sft_recap_data.py` queue. Fixed training[200,296), new critic-validation[120,132), four-episode smoke before all216 attempts; no top-up to success quotas. Raw HDF5 has T pre-action observations plus a separate actual final observation. Collection is not policy training; LeRobot export/loader acceptance remains a separate gate.
- Old24 scenes remain development. Final seed-list indices[500,550) remain held out. Motus pretraining overlap is unknown; post-training split isolation is not proof of unseen pretraining scenes.
- Existing local images are decision-boundary plus terminal observations, not every control-step images. Full simulator restore state and verified goal pairs are absent. Do not blindly feed this pool to a step-level loader or fabricate missing frames.
- Reliable success data supplies SFT actions; audited success/failure data can prepare value. Current mechanism studies use binary terminal outcomes, gamma1, no clock input; retain legacy raw rewards as provenance. All-success trajectories alone have constant binary MC targets and cannot teach outcome ranking.
- MC targets describe behavior-data continuations, not arbitrary new-policy values. Current scalar prefix value does not read candidate actions and is not an instruction-action reward. Original RECAP uses its own backbone with image/instruction inputs; adding state, time or policy features is a separate factor.
- Fixed actor labels and policy initialization, actual task supervision/action-time exposure, capacities, update scopes and total costs need matched controls. Equal sample counts are not equal gradient magnitudes. Separate actor seeds from head seeds and scenario uncertainty.
- Old M2 froze the native VLM and action expert and trained adapter/head; later expert studies still froze the VLM. No evidence yet establishes improved native-VLM features or faster RL. To claim native-VLM adaptation introduce explicit LoRA/unfreeze controls and verify gradients/memory; do not silently change freezing.
- Negative gradient cosine or lower critic MSE alone is not negative transfer or policy improvement. Current two-task evidence establishes neither a shared-critic bottleneck nor reliable PCGrad gains. Do not force a conflict/novelty story.
- Verified same-state goal pairs are required for instruction-semantic interventions; different task names are not automatic negatives. I0/I1, semantic reward and broad layer/loss sweeps remain optional, not prerequisites for ordinary task baselines.
- GRPO can test critic-independent transfer, but requires qualified stochastic-flow likelihoods and same-task/instruction/controlled-initial-condition groups. Old single offline episodes are not on-policy groups. Track equal-outcome groups and filtering cost; FM loss is not logprob.
- Strict unupdated-task anchors provide no SFT/critic/replay/regularizer data. If replay is used call those rehearsal tasks and keep separate strict anchors. Test scene/model choice never uses final evaluation results.

## Storage, artifacts and Git

- User explicitly authorized deletion of reproducible historical large experiment payloads and rerunning when needed, instead of archives. This supersedes earlier pending pilot-checkpoint deletion proposals. Do not ask again about that old cleanup.
- `.runtime/storage/cleanup-report-v3.json` records141 removed files and9 retained large initializer/BC/ordinary-weighted addons; small W1/W2/head artifacts remain. Historical large DCP/optimizer and old LIBERO/pilot final/resume payloads are gone. Old complete audits describe past validation, not current checkpoint availability. Consult per-run `storage-retention.json` before reusing a path.
- Retained eval/init weights do not contain the deleted optimizer state. If an optional diagnostic needs that state, generate a new short run; do not auto-restore or rerun all historical experiments.
- Before collection/training estimate data, all checkpoint/export and temporary peaks, leaving at least100GiB free. Default to one evaluable final; retain optimizer only when required by the declared resume/handoff experiment. Clear smoke payloads after checks. Do not automatically delete original models/data/environments.
- Results live only in `docs/previous-experiments.md`; plans in the guide and host sheets. Runtime logs/manifests/source snapshots stay under ignored `.runtime/`. Do not restore old dashboards or duplicate experiment records into the repository.
- Keep models, datasets, credentials, environments and generated checkpoints outside Git. Preserve locks and reusable code. Container packaging uses `bootstrap/build_recap_image.py`/`docker/`; its CPU/chroot evidence is not Docker GPU validation, and the current image covers LIBERO rather than RoboTwin GPU acceptance.
- The historical LIBERO public-workflow reference is separate; original reported48.8%/66.5% start weights were not released. Do not claim exact reproduction or automatically restart its superseded long budgets.
