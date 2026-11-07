"""Verify every policy-used tensor survives native/reference layout conversion."""
import json,os
from pathlib import Path
import torch
from safetensors.torch import load_file
from rlinf.utils.ckpt_convertor.openpi.openpi_pytorch_to_openpi import old_to_new_state_dict
from rlinf.utils.ckpt_convertor.openpi.openpi_to_openpi_pytorch import new_to_old_state_dict,ACTION_EXPERT_LM_HEAD
root=Path(os.environ['PROJECT_ROOT']);torch.set_num_threads(4)
old=load_file(str(root/'models/pi0.5_robotwin2/model.safetensors'))
back=new_to_old_state_dict(old_to_new_state_dict(old))
missing=sorted(old.keys()-back.keys());unexpected=sorted(back.keys()-old.keys())
changed=[k for k in back if k in old and not torch.equal(old[k],back[k])]
result={'status':'pass' if missing==[ACTION_EXPERT_LM_HEAD] and not unexpected and not changed else 'fail','source_tensors':len(old),'mapped_tensors':len(back),'missing':missing,'unexpected':unexpected,'changed_tensors':changed,'note':'Only the unused action-expert token LM head is retained from the original reference checkpoint. All action-producing weights roundtrip exactly.'}
(root/'.runtime/experiments/e5-pirl/weight-roundtrip-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if result['status']!='pass':raise RuntimeError('Weight conversion roundtrip failed')
