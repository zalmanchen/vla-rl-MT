"""Freeze the E5 online-RL matrix and compose isolated single-GPU RLinf configs."""
import json
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'.runtime/experiments/e5-pirl'
TASKS=['adjust_bottle','click_bell','stack_blocks_two','place_empty_cup','place_container_plate']

def main():
 RUN.mkdir(parents=True,exist_ok=True); (RUN/'configs').mkdir(exist_ok=True)
 with initialize_config_dir(version_base='1.1',config_dir=str(ROOT/'third_party/RLinf/examples/embodiment/config')):
  base=compose(config_name='robotwin_adjust_bottle_ppo_openpi_pi05')
 jobs=[]
 for stage,tasks,seeds,epochs in [('smoke',TASKS[:1],[0],1),('single',TASKS,[0,1,2],100),('multi',TASKS,[0,1,2],100)]:
  groups=[tasks] if stage=='multi' else [[t] for t in tasks]
  for ts in groups:
   for seed in seeds:
    name=f'{stage}-'+('mt5' if stage=='multi' else ts[0])+f'-seed{seed}'
    c=OmegaConf.create(OmegaConf.to_container(base,resolve=False))
    with open_dict(c):
     c.cluster.component_placement={'actor,env,rollout':'0'}
     c.runner.logger.log_path=str(RUN/'jobs'/name)
     c.runner.logger.experiment_name=name
     c.runner.max_epochs=epochs; c.runner.max_steps=epochs
     c.runner.val_check_interval=-1; c.runner.save_interval=1 if stage=='smoke' else 10
     c.actor.micro_batch_size=1 if stage=='smoke' else 4; c.actor.global_batch_size=4 if stage=='smoke' else 32
     c.actor.seed=seed; c.actor.enable_offload=True
     c.actor.model.model_path=str(ROOT/'models/pi0.5_robotwin2')
     c.actor.model.precision='fp32'
     c.actor.model.num_action_chunks=32; c.actor.model.num_steps=10
     c.actor.model.openpi.num_steps=10; c.actor.model.openpi.action_horizon=32
     c.actor.model.openpi.max_token_len=200; c.actor.model.openpi.discrete_state_input=True
     c.actor.model.openpi.torch_compile=False
     c.actor.model.openpi.noise_method='flow_sde'; c.actor.model.openpi.train_expert_only=True
     c.actor.model.openpi_data={'norm_stats_path':str(ROOT/'models/pi0.5_robotwin2/assets/pi0.5_clean_randomize_joint_training/norm_stats.json'),'adapt_to_pi':True,'extra_delta_transform':True}
     c.actor.fsdp_config.gradient_checkpointing=True
     c.actor.fsdp_config.mixed_precision={'param_dtype':'fp32','reduce_dtype':'fp32','buffer_dtype':'fp32'}
     c.actor.fsdp_config.strategy='fsdp'; c.actor.fsdp_config.sharding_strategy='no_shard'
     c.algorithm.update_epoch=1 if stage=='smoke' else 5
     for split in ['train','eval']:
      env=c.env[split]
      env.total_num_envs=2*len(ts); env.rollout_epoch=1
      env.seed=seed; env.auto_reset=True; env.ignore_terminations=False
      env.max_steps_per_rollout_epoch=64 if stage=='smoke' else 512
      records=sorted((ROOT/'.runtime/experiments/e1-reference-clean/clean'/ts[0]).glob('episode_*.json'))
      limit=json.loads(records[0].read_text())['limit']
      env.max_episode_steps=limit; env.assets_path=str(ROOT/'third_party/RoboTwin')
      env.video_cfg.save_video=False; env.task_config.task_name=ts[0]; env.task_config.step_lim=limit
      env.task_config.clear_cache_freq=8
      env.task_config.save_path=str(RUN/'jobs'/name/'simulator')
      env.task_config.collect_data=False
      env.task_config.pilot_tasks=ts
      env.task_config.pilot_task_limits={t:json.loads(sorted((ROOT/'.runtime/experiments/e1-reference-clean/clean'/t).glob('episode_*.json'))[0].read_text())['limit'] for t in ts}
      if len(ts)>1:env.max_episode_steps=max(env.task_config.pilot_task_limits.values())
     c.env.train.use_fixed_reset_state_ids=False
     if stage!='smoke':c.env.train.task_config.pilot_record_dir=str(RUN/'jobs'/name/'online-chunks')
     c.rollout.model.precision='fp32'
    p=RUN/'configs'/f'{name}.yaml';OmegaConf.save(c,p)
    jobs.append({'name':name,'stage':stage,'tasks':ts,'seed':seed,'rollout_iterations':epochs,'config':str(p),'status':'planned'})
 plan={'algorithm':'piRL Flow-SDE + PPO','setting':'online RL','starting_checkpoint':'models/pi0.5_robotwin2','baseline_full':{'successes':566,'episodes':2200,'tasks':22},'baseline_paired':{'successes':47,'episodes':100,'tasks':5},'training_seeds':[0,1,2],'tasks':TASKS,'single_envs_per_task':2,'rollout_frames_per_env_per_iteration':512,'single_iterations':100,'mt_equal_per_task_iterations':100,'mt_equal_total_budget_checkpoint':100,'mt_low_budget_checkpoint':20,'budget_note':'均衡采样时，相同每任务交互量也意味着五个独立策略合计与共享策略总交互量相同；MT20为低预算点，不称等预算比较','evaluation':'frozen E1 100 scenes/task; final checkpoints evaluated on all 22 tasks; tuning must use separate validation scenes','stages':['zero-update numerical/closed-loop audit','one-iteration official trainer smoke','single-task PPO','five-task PPO with equal-per-task and equal-total-budget comparison','22-task retention','continuous-state long-horizon feasibility audit'],'long_horizon_status':'not yet configured; requires valid continuous scene and identical scheduler for shared vs specialists','jobs':jobs,'status':'preparing; no PPO updates completed yet'}
 (RUN/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
 print(json.dumps({'run':str(RUN),'jobs':len(jobs),'smoke_config':jobs[0]['config']},indent=2))

if __name__=='__main__':main()
