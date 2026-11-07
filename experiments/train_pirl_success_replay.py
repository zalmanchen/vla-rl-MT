"""Success-only flow-matching control on exactly the PPO job's recorded experience."""
import argparse,json,os,time
from pathlib import Path
import cv2,h5py,numpy as np,torch
from torch.utils.data import DataLoader,Dataset,WeightedRandomSampler
from omegaconf import OmegaConf
from pirl_extension import register
register()
from rlinf.models.embodiment.openpi import get_model
from rlinf.models.embodiment.openpi.transforms.pipeline import build_openpi_transforms
from rlinf.models.embodiment.openpi.modules.model import Observation
from openpi.transforms import compose
ROOT=Path(os.environ['PROJECT_ROOT'])
class Chunks(Dataset):
 def __init__(self,entries,transform):self.entries,self.transform=entries,transform
 def __len__(self):return len(self.entries)
 def __getitem__(self,i):
  path,idx,task=self.entries[i]
  with h5py.File(path,'r') as h:
   images={k:cv2.cvtColor(cv2.imdecode(np.asarray(h['images/'+k][idx]),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB) for k in ['head','left_wrist','right_wrist']}
   sample={'observation/image':images['head'],'observation/wrist_image':np.stack([images['left_wrist'],images['right_wrist']]),'observation/state':h['state'][idx],'actions':h['actions'][idx],'prompt':str(h.attrs['instruction'])}
  return self.transform(sample)
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,required=True);p.add_argument('--micro-batch',type=int,default=4);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=True)
 if (a.output/'FINISHED').exists():return
 cfg=OmegaConf.load(a.config).actor.model;seed=int(OmegaConf.load(a.config).actor.seed);torch.manual_seed(seed);np.random.seed(seed)
 entries=[];counts={};files=[]
 for path in sorted(a.source.glob('*.hdf5')):
  with h5py.File(path,'r') as h:
   if not h.attrs['success'] and not a.smoke:continue
   task=str(h.attrs['task']);n=len(h['state'])
   entries.extend((str(path),i,task) for i in range(n));counts[task]=counts.get(task,0)+n;files.append(str(path))
 if not entries:raise ValueError('No successful recorded experience for this control')
 transforms,_=build_openpi_transforms(str(ROOT/'models/pi0.5_robotwin2'),'pi05_aloha_robotwin',dict(cfg.openpi_data))
 model=get_model(cfg,torch_dtype=torch.float32).cuda();model.gradient_checkpointing_enable();model.train();model.freeze_vlm()
 for param in model.value_head.parameters():param.requires_grad=False
 params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=5e-6,betas=(.9,.95),eps=1e-8,weight_decay=.01)
 steps=1 if a.smoke else a.steps;batch=1 if a.smoke else 32;micro=a.micro_batch;assert batch%micro==0
 weights=[1/counts[t] for _,_,t in entries]
 loader=DataLoader(Chunks(entries,compose(transforms)),batch_size=micro,sampler=WeightedRandomSampler(weights,steps*batch,replacement=True),num_workers=4,pin_memory=True)
 receipt={'mode':'success_replay','source_files':files,'samples':len(entries),'tasks':list(counts),'steps':steps,'global_batch':batch,'seed':seed,'smoke':a.smoke,'smoke_note':'smoke may use non-success chunks for finite backward only; never optimizer updates' if a.smoke else '', 'algorithm':'success-only supervised FM on same PPO experience; initialized from original Motus'}
 (a.output/'contract.json').write_text(json.dumps(receipt,indent=2));opt.zero_grad();started=time.time()
 for n,b in enumerate(loader,1):
  obs=Observation.from_dict(b)
  loss=model.sft_forward({'observation':obs,'actions':b['actions']})
  if isinstance(loss,dict):loss=loss['loss']
  if not torch.isfinite(loss):raise ValueError('Non-finite replay loss')
  (loss/(batch//micro)).backward()
  if n%(batch//micro):continue
  step=n//(batch//micro);norm=torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True)
  if not a.smoke:opt.step()
  opt.zero_grad();progress={'step':step,'loss':loss.item(),'grad_norm':norm.item(),'seconds':time.time()-started}
  (a.output/'progress.json').write_text(json.dumps(progress,indent=2))
  if step==1 or step%25==0:print(progress,flush=True)
 if a.smoke:
  (a.output/'SMOKE_PASSED').write_text('finite forward/backward; no optimizer update');return
 (a.output/'model_state_dict').mkdir(exist_ok=True)
 torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},a.output/'model_state_dict/full_weights.pt')
 (a.output/'FINISHED').write_text(str(time.time()))
if __name__=='__main__':main()
