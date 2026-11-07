"""Check zero-update inference drift and stored-chain likelihood consistency."""
import json,os,random
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from pirl_extension import register
register()
from reference_policy import ReferencePolicy
from rlinf.models.embodiment.openpi import get_model
ROOT=Path(os.environ['PROJECT_ROOT']);RUN=ROOT/'.runtime/experiments/e5-pirl'

def main():
 torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
 plan=json.loads((RUN/'plan.json').read_text()); samples=[]
 for task in plan['tasks']:
  p=sorted((ROOT/'.runtime/experiments/e1-reference-clean/clean'/task).glob('input_*.npz'))[0]
  with np.load(p) as d:
   obs={'main_images':torch.from_numpy(d['head'][None].copy()),'wrist_images':torch.from_numpy(d['wrist'][None].copy()),'states':torch.from_numpy(d['state'][None].copy()),'task_descriptions':[str(d['instruction'].item())]}
  samples.append((task,obs))
 reference=ReferencePolicy();expected=[]
 for task,obs in samples:
  a,_=reference.predict_action_batch(obs,rng=torch.Generator(device='cuda').manual_seed(20261106));expected.append(a.cpu())
 del reference;torch.cuda.empty_cache()
 cfg=OmegaConf.load(RUN/'configs/smoke-adjust_bottle-seed0.yaml').actor.model
 model=get_model(cfg,torch_dtype=torch.float32).cuda().eval();rows=[]
 for (task,obs),target in zip(samples,expected):
  a,_=model.predict_action_batch(obs,mode='eval',rng=torch.Generator(device='cuda').manual_seed(20261106))
  delta=a.cpu()-target
  random.seed(20261106)
  with torch.no_grad():
   _,info=model.predict_action_batch(obs,mode='train',rng=torch.Generator(device='cuda').manual_seed(20261106))
   result=model(forward_inputs=info['forward_inputs'])
  lp=(result['logprobs']-info['prev_logprobs']).abs()
  row={'task':task,'action_max_abs_vs_reference':float(delta.abs().max()),'action_relative_l2_vs_reference':float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(target)),'logprob_recompute_max_abs':float(lp.max()),'logprob_recompute_mean_abs':float(lp.mean()),'finite':bool(torch.isfinite(result['logprobs']).all() and torch.isfinite(a).all())}
  rows.append(row);print(row,flush=True)
 output={'status':'pass' if all(x['finite'] and x['logprob_recompute_max_abs']<.01 for x in rows) else 'fail','scope':'5 real initial observations; no optimizer updates; numerical differences are not success-rate evidence','rows':rows,'native_drift_requires_closed_loop_baseline':True}
 (RUN/'zero-update-audit.json').write_text(json.dumps(output,indent=2))
 if output['status']!='pass':raise RuntimeError('Zero-update likelihood audit failed')
if __name__=='__main__':main()
