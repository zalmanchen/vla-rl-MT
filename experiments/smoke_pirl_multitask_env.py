"""Validate balanced task assignment and disjoint train seeds without policy updates."""
import json,os
from pathlib import Path
import torch
from omegaconf import OmegaConf
from pirl_extension import register
register()
from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv
ROOT=Path(os.environ['PROJECT_ROOT']);RUN=ROOT/'.runtime/experiments/e5-pirl'

def main():
 cfg=OmegaConf.load(RUN/'configs/multi-mt5-seed0.yaml').env.train
 cfg.task_config.pilot_record_dir=str(RUN/'recorder-smoke-v4')
 env=RoboTwinEnv(cfg=cfg,num_envs=10,seed_offset=0,total_num_processes=1,worker_info=None)
 try:
  obs,_=env.reset(); tasks=[s.task_name for s in env.venv.envs]
  manifest=json.loads((ROOT/'.runtime/experiments/e1-reference-clean/seed_manifest.json').read_text())
  seeds=env.reset_state_ids.tolist()
  assert len(obs['task_descriptions'])==10
  assert all(tasks.count(t)==2 for t in cfg.task_config.pilot_tasks)
  assert all(seed not in manifest['tasks'][task]['valid_seeds'] for task,seed in zip(tasks,seeds))
  assert all(p.strip() for p in obs['task_descriptions'])
  actions=obs['states'].float()[:,None,:].repeat(1,32,1)
  env.chunk_step(actions)
  env.reset()
  files=list((RUN/'recorder-smoke-v4').glob('*.hdf5'))
  assert len(files)==10, f'Recorder wrote {len(files)}/10 traces'
  result={'recorded_traces':len(files),'status':'pass','tasks':tasks,'seeds':seeds,'instructions':obs['task_descriptions'],'evaluation_seed_overlap':False,'scope':'reset/task/seed/prompt plumbing only; no policy training or performance claim'}
  (RUN/'multitask-env-smoke.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
 finally:env.offload()
if __name__=='__main__':main()
