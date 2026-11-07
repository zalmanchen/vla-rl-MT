"""Audit native zero-update inference; evaluate trained policies with E1's sampler."""
import json,os
from pathlib import Path
import torch
from omegaconf import OmegaConf
from pirl_extension import register
register()
from rlinf.models.embodiment.openpi import get_model

class PirlPolicy:
 def __init__(self,config,checkpoint='base'):
  self.native=checkpoint=='base'
  if self.native:
   cfg=OmegaConf.load(config).actor.model
   self.model=get_model(cfg,torch_dtype=torch.float32).cuda().eval()
   return
  root=Path(os.environ['PROJECT_ROOT'])
  audit=json.loads((root/'.runtime/experiments/e5-pirl/weight-roundtrip-audit.json').read_text())
  if audit['status']!='pass':raise RuntimeError('Reference weight-layout audit has not passed')
  from reference_policy import ReferencePolicy
  from rlinf.models.embodiment.openpi.checkpoint import resolve_full_weights,_normalize_state_dict
  from rlinf.utils.ckpt_convertor.openpi.openpi_to_openpi_pytorch import new_to_old_state_dict,ACTION_EXPERT_LM_HEAD
  self.reference=ReferencePolicy()
  path=resolve_full_weights(Path(checkpoint))
  if path is None:raise FileNotFoundError(checkpoint)
  state=_normalize_state_dict(torch.load(path,map_location='cpu',weights_only=True,mmap=True))
  converted=new_to_old_state_dict(state)
  # Safetensors removes this tied alias; state_dict loading expects both names.
  converted["paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"]=converted["paligemma_with_expert.paligemma.lm_head.weight"]
  missing,unexpected=self.reference.model.load_state_dict(converted,strict=False)
  if sorted(missing)!=[ACTION_EXPERT_LM_HEAD] or unexpected:raise RuntimeError(f'Incomplete action policy conversion: {missing}, {unexpected}')
  self.model=self.reference.model
 def predict_action_batch(self,env_obs,mode='eval',rng=None):
  if mode not in ['eval','train']:raise ValueError('Unsupported sampling mode')
  if self.native:return self.model.predict_action_batch(env_obs,mode=mode,rng=rng)
  if mode!='eval':raise ValueError('Reference checkpoint supports ODE evaluation only')
  return self.reference.predict_action_batch(env_obs,mode='eval',rng=rng)
