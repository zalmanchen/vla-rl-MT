"""Project-local Motus activation and renderer compatibility for RLinf workers."""
_registered=False

def register():
 global _registered
 if _registered:return
 _registered=True
 import sapien.render
 original=sapien.render.set_ray_tracing_denoiser
 sapien.render.set_ray_tracing_denoiser=lambda name:original('none' if name=='oidn' else name)
 import rlinf.models.embodiment.openpi as factory
 from compat import match_reference_activations
 build=factory.get_model
 def build_motus(cfg,torch_dtype=None):
  model=build(cfg,torch_dtype)
  match_reference_activations(model)
  return model
 factory.get_model=build_motus
 install_task_adapter()
 install_chunk_recorder()
 install_waypoint_tracking()


def install_task_adapter():
 """Assign tasks and train seeds per environment; regenerate prompts after reset."""
 import copy,json,os
 import numpy as np
 import torch
 from pathlib import Path
 from robotwin.envs import vector_env as vector
 from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv
 from description.utils.generate_episode_instructions import generate_episode_descriptions
 root=Path(os.environ['PROJECT_ROOT'])
 base_init=RoboTwinEnv._init_reset_state_ids
 base_update=RoboTwinEnv.update_reset_state_ids
 def init_seeds(self):
  tasks=list(self.cfg.task_config.get('pilot_tasks',[self.task_name]))
  if len(tasks)==1:return base_init(self)
  if self.num_envs%len(tasks):raise ValueError('Environment count must divide evenly across tasks')
  source=json.loads(Path(self.cfg.seeds_path).read_text())
  self._pilot_seed_lists=[];self._pilot_positions=[0]*len(tasks)
  for task in tasks:
   seeds=source[task]['success_seeds']
   order=torch.randperm(len(seeds),generator=torch.Generator().manual_seed(self.base_seed)).tolist()
   self._pilot_seed_lists.append([seeds[i] for i in order])
  self._pilot_tasks=tasks;self._pilot_slots=self.num_envs//len(tasks)
  self.success_seeds=None
  self.reset_state_ids=torch.zeros(self.num_envs,dtype=torch.long)
  update_seeds(self)
 def update_seeds(self,env_idx=None):
  if not hasattr(self,'_pilot_seed_lists'):return base_update(self,env_idx)
  if self.use_fixed_reset_state_ids and getattr(self,'_pilot_seed_initialized',False):return
  indices=range(self.num_envs) if env_idx is None else env_idx
  for raw in indices:
   i=int(raw);task=i//self._pilot_slots
   pos=self._pilot_positions[task];seeds=self._pilot_seed_lists[task]
   self.reset_state_ids[i]=seeds[pos%len(seeds)]
   self._pilot_positions[task]+=1
  self._pilot_seed_initialized=True
 RoboTwinEnv._init_reset_state_ids=init_seeds
 RoboTwinEnv.update_reset_state_ids=update_seeds
 def init_envs(self):
  tasks=list(self.args.get('pilot_tasks',[self.task_name]));slots=self.n_envs//len(tasks)
  for i in range(self.n_envs):
   task=tasks[i//slots];args=copy.deepcopy(self.args);args['task_name']=task
   args['step_lim']=int(args.get('pilot_task_limits',{}).get(task,args['step_lim']))
   sub=vector.SubEnv(env_id=i,task_name=task,args=args,env_seed=self.env_seeds[i] if self.env_seeds else None,instruction_type=self.instruction_type,global_lock=self.global_lock)
   sub.setup_task();self.envs.append(sub)
 vector.VectorEnv._init_envs=init_envs
 reset=vector.SubEnv.reset
 def reset_with_prompt(self,env_seed=None):
  reset(self,env_seed)
  descriptions=generate_episode_descriptions(self.task_name,[self.task.get_info()],1,seed=self.env_seed)
  self.instruction=descriptions[0]['seen'][0]
  self.args['instruction']=self.instruction;self.task.set_instruction(self.instruction)
 vector.SubEnv.reset=reset_with_prompt


def install_chunk_recorder():
 """Persist pre-action RGB/state and executed chunks for matched replay controls."""
 import json,time,uuid
 from pathlib import Path
 import cv2,h5py,numpy as np
 from robotwin.envs import vector_env as vector
 reset=vector.SubEnv.reset;step=vector.SubEnv.step
 def save_episode(self,success,truncated):
  rows=getattr(self,'_pilot_chunks',[])
  if not rows:return
  root=Path(self.args['pilot_record_dir']);root.mkdir(parents=True,exist_ok=True)
  path=root/f'{self.task_name}_{self.env_seed}_{self.env_id}_{uuid.uuid4().hex[:12]}.hdf5'
  with h5py.File(path.with_suffix('.tmp'),'w') as h:
   h.attrs.update(task=self.task_name,seed=int(self.env_seed),instruction=self.instruction,success=bool(success),truncated=bool(truncated),source='online_ppo',record_type='pre_action_chunk',created_unix=time.time())
   h.create_dataset('state',data=np.stack([x['state'] for x in rows]))
   h.create_dataset('actions',data=np.stack([x['actions'] for x in rows]))
   h.create_dataset('executed_action_count',data=[x['executed'] for x in rows])
   for camera in ['head','left_wrist','right_wrist']:
    ds=h.create_dataset('images/'+camera,(len(rows),),dtype=h5py.vlen_dtype(np.dtype('uint8')))
    for i,x in enumerate(rows):ds[i]=x[camera]
  path.with_suffix('.tmp').replace(path);self._pilot_chunks=[]
 def recorded_reset(self,env_seed=None):
  if self.args.get('pilot_record_dir') and getattr(self,'_pilot_chunks',[]):save_episode(self,False,True)
  reset(self,env_seed);self._pilot_chunks=[]
 def recorded_step(self,actions):
  enabled=bool(self.args.get('pilot_record_dir'))
  before_count=int(self.task.take_action_cnt)
  if enabled:
   obs=self.task.get_obs();entry={'state':np.asarray(obs['joint_action']['vector']).copy(),'actions':np.asarray(actions,dtype=np.float32).copy()}
   for dst,src in [('head','head_camera'),('left_wrist','left_camera'),('right_wrist','right_camera')]:
    ok,encoded=cv2.imencode('.jpg',cv2.cvtColor(obs['observation'][src]['rgb'],cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,90])
    if not ok:raise ValueError('Failed to encode replay RGB')
    entry[dst]=encoded.reshape(-1)
  result=step(self,actions)
  count=int(result['info'].get('executed_action_count',int(self.task.take_action_cnt)-before_count))
  result['info']['executed_action_count']=count
  self._pilot_last_executed_count=count
  if enabled:
   log_path=Path(self.args['pilot_record_dir'])/'interactions.jsonl'
   log_path.parent.mkdir(parents=True,exist_ok=True)
   with log_path.open('a') as log:
    log.write(json.dumps({'time_unix':time.time(),'task':self.task_name,'env_id':self.env_id,'seed':int(self.env_seed),'requested_actions':len(actions),'executed_actions':count,'submitted_counter_delta':int(self.task.take_action_cnt)-before_count,'success':bool(self.task.eval_success),'terminated':bool(result['terminated']),'truncated':bool(result['truncated'])})+'\n')
   entry['executed']=count
   if count>0:
    if count<len(entry['actions']):entry['actions'][count:]=entry['actions'][count-1]
    self._pilot_chunks.append(entry)
   if bool(result['terminated']) or bool(result['truncated']):save_episode(self,bool(self.task.eval_success),bool(result['truncated']))
  return result
 vector.SubEnv.reset=recorded_reset;vector.SubEnv.step=recorded_step
 from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv
 from pirl_rewards import terminal_chunk_rewards
 def chunk_rewards(self,step_rewards,chunk_length,terminations,infos):
  counts=[getattr(sub,'_pilot_last_executed_count',chunk_length) for sub in self.venv.envs]
  return terminal_chunk_rewards(step_rewards,terminations,counts,chunk_length)
 RoboTwinEnv._cal_chunk_rewards=chunk_rewards


def install_waypoint_tracking():
 """Count reached control points without rendering intermediate camera frames."""
 from envs._base_task import Base_Task
 original=Base_Task.gen_sparse_reward_data
 def tracked(self,chunk_actions,action_type="qpos",collect_obs=False):
  if collect_obs:return original(self,chunk_actions,action_type=action_type,collect_obs=True)
  get_obs=self.get_obs
  self.get_obs=lambda:None
  try:
   reward,termination,truncation,info,_=original(self,chunk_actions,action_type=action_type,collect_obs=True)
  finally:self.get_obs=get_obs
  return reward,termination,truncation,info
 Base_Task.gen_sparse_reward_data=tracked
