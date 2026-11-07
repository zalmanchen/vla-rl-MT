"""Force one success in a separate test scene to verify terminal reward/mask alignment."""
import json,os
from pathlib import Path
import torch
from omegaconf import OmegaConf
from pirl_extension import register
register()
from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv
ROOT=Path(os.environ['PROJECT_ROOT']);RUN=ROOT/'.runtime/experiments/e5-pirl'
def main():
 cfg=OmegaConf.load(RUN/'configs/smoke-adjust_bottle-seed0.yaml').env.train
 cfg.task_config.pilot_record_dir=str(RUN/'terminal-smoke-not-training-data')
 env=RoboTwinEnv(cfg=cfg,num_envs=2,seed_offset=0,total_num_processes=1,worker_info=None)
 try:
  obs,_=env.reset();task=env.venv.envs[0].task;check=task.check_success
  task.check_success=lambda:True
  _,reward,_,_,_=env.chunk_step(obs['states'].float()[:,None,:].repeat(1,32,1))
  task.check_success=check
  count=env.venv.envs[0]._pilot_last_executed_count
  assert count<32 and count>0,count
  assert reward[0,count-1]==1 and reward[0].sum()==1,reward
  assert torch.all(reward[0,count:]==0)
  receipt={'status':'pass','scope':'artificial forced-success integration test only; scene and data excluded from training/evaluation','terminal_executed_count':count,'reward_slot':count-1,'reward_sum':reward[0].sum().item()}
  (RUN/'terminal-integration-audit.json').write_text(json.dumps(receipt,indent=2));print(receipt,flush=True)
 finally:env.offload()
if __name__=='__main__':main()
