"""Place terminal rewards inside the actually executed part of an action chunk."""
import torch

def terminal_chunk_rewards(step_rewards,terminations,counts,chunk_length):
 counts=torch.as_tensor(counts,device=step_rewards.device,dtype=torch.long)
 if ((counts<0)|(counts>chunk_length)).any():raise ValueError('Invalid executed action count')
 output=torch.zeros(len(counts),chunk_length,device=step_rewards.device,dtype=step_rewards.dtype)
 for i in range(len(counts)):
  if bool(terminations[i]) and counts[i]>0:output[i,counts[i]-1]=step_rewards[i]
 return output
