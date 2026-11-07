"""OpenPI measurement worker; PPO keeps the original behavior denominator."""
import json
from pathlib import Path
import time
import torch
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.utils.nested_dict_process import split_dict_to_chunk, put_tensor_device


class DiagnosticEmbodiedFSDPActor(EmbodiedFSDPActor):
    def get_rollout_state_dict(self):
        state = super().get_rollout_state_dict()
        if (self.cfg.weight_syncer.type == 'bucket'
                and self.cfg.actor.get('pilot_full_bucket_sync', False)
                and not self.weight_syncer.sender_initialized()):
            selected = len(self.param_names_need_sync)
            self.param_names_need_sync = list(state)
            dest = Path(self.cfg.runner.logger.log_path) / 'synchronization-contract.json'
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps({'transport': 'official FP32 buckets',
                                       'scope': 'all state tensors, including frozen weights',
                                       'previous_selected_tensors': selected,
                                       'synchronized_tensors': len(state)}, indent=2))
        return state

    def run_training(self):
        old_flag = self.cfg.algorithm.recompute_logprobs
        self.cfg.algorithm.recompute_logprobs = True
        try:
            return super().run_training()
        finally:
            self.cfg.algorithm.recompute_logprobs = old_flag

    def recompute_prev_logprobs(self, batch_size_per_rank):
        previous = self.rollout_batch['prev_logprobs']
        batches = []
        with torch.no_grad():
            for batch in split_dict_to_chunk(self.rollout_batch, previous.shape[0] // batch_size_per_rank):
                for micro in split_dict_to_chunk(batch, batch_size_per_rank // self.cfg.actor.micro_batch_size):
                    micro = put_tensor_device(micro, self.device)
                    with self.amp_context:
                        result = self.model(forward_inputs=micro['forward_inputs'], compute_logprobs=True,
                                            compute_entropy=False, compute_values=False, use_cache=False)
                    batches.append(result['logprobs'].detach().cpu())
        recomputed = torch.cat(batches).to(previous.dtype)
        if recomputed.shape != previous.shape:
            raise RuntimeError('OpenPI recomputation shape mismatch')
        delta = (recomputed - previous).float()
        delta = delta.reshape(*delta.shape[:-1], -1, self.cfg.actor.model.action_dim)
        mask = self.rollout_batch.get('loss_mask')
        if mask is not None:
            delta = delta[mask.bool().unsqueeze(-1).expand_as(delta)]
        if delta.numel() == 0 or not torch.isfinite(delta).all():
            raise RuntimeError('Invalid pre-update logprob audit')
        receipt = {'time_unix': time.time(), 'optimizer_steps': self.optimizer_steps,
                   'rollout_batch_size': int(previous.shape[0]),
                   'logprob_gap_mean_abs': delta.abs().mean().item(),
                   'logprob_gap_max_abs': delta.abs().max().item(),
                   'logprob_gap_p99_abs': torch.quantile(delta.abs(), .99).item(),
                   'denominator': 'original rollout logprobs retained'}
        for name in ['returns', 'advantages']:
            values = self.rollout_batch[name].detach().float()
            if mask is not None:
                statistic_mask = mask.bool()
                if values.shape[-1] == 1 and statistic_mask.shape[-1] != 1:
                    statistic_mask = statistic_mask.any(dim=-1, keepdim=True)
                values = values[torch.broadcast_to(statistic_mask, values.shape)]
            receipt[name + '_variance'] = values.var(unbiased=False).item()
            receipt[name + '_mean'] = values.mean().item()
        trainable = [name for name, p in self.model.named_parameters() if p.requires_grad]
        receipt['trainable_parameter_tensors'] = len(trainable)
        receipt['only_value_head_trainable'] = bool(trainable) and all('value_head' in name for name in trainable)
        receipt['critic_warmup_active'] = self.optimizer_steps < self.critic_warmup_steps
        if receipt['critic_warmup_active'] and not receipt['only_value_head_trainable']:
            raise RuntimeError('Critic-only diagnostic would update policy parameters')
        dest = Path(self.cfg.runner.logger.log_path) / 'batch-audit.jsonl'
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('a') as file:
            file.write(json.dumps(receipt) + '\n')
        if receipt['logprob_gap_max_abs'] > .01:
            torch.save({'rollout_batch': self.rollout_batch, 'recomputed_logprobs': recomputed,
                        'receipt': receipt}, dest.parent / 'failed-batch-audit.pt')
            raise RuntimeError('Pre-update logprob mismatch exceeds0.01; inspect batch-audit.jsonl')
        assert self.rollout_batch['prev_logprobs'] is previous
        return {'actor/rollout_train_logprob_gap': receipt['logprob_gap_mean_abs'],
                'actor/audit_logprob_gap_max': receipt['logprob_gap_max_abs'],
                'critic/audit_return_variance': receipt['returns_variance']}
