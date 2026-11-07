"""Opt-in measurements for E6; retain rollout logprobs as the PPO denominator."""
_registered = False


def register():
    global _registered
    if _registered:
        return
    _registered = True
    import pirl_extension
    pirl_extension.register()
    import json
    import functools
    import time
    from pathlib import Path
    import torch
    from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
    from rlinf.utils.nested_dict_process import split_dict_to_chunk, put_tensor_device

    original = EmbodiedFSDPActor.recompute_prev_logprobs
    original_training = EmbodiedFSDPActor.run_training

    @functools.wraps(original_training)
    def audited_training(self, *args, **kwargs):
        if not self.cfg.actor.get('pilot_diagnostics', False):
            return original_training(self, *args, **kwargs)
        # Public recomputation is validated only for OpenVLA in this revision.
        # Use its existing measurement location with our OpenPI-specific callback;
        # restore the config and retain behavior logprobs, so PPO is unchanged.
        old_flag = self.cfg.algorithm.recompute_logprobs
        self.cfg.algorithm.recompute_logprobs = True
        try:
            return original_training(self, *args, **kwargs)
        finally:
            self.cfg.algorithm.recompute_logprobs = old_flag

    def audited_recompute(self, batch_size_per_rank):
        if not self.cfg.actor.get('pilot_diagnostics', False):
            return original(self, batch_size_per_rank)
        previous = self.rollout_batch['prev_logprobs']
        try:
            batches = []
            with torch.no_grad():
                for batch in split_dict_to_chunk(self.rollout_batch, previous.shape[0] // batch_size_per_rank):
                    for micro in split_dict_to_chunk(batch, batch_size_per_rank // self.cfg.actor.micro_batch_size):
                        micro = put_tensor_device(micro, self.device)
                        with self.amp_context:
                            result = self.model(forward_inputs=micro['forward_inputs'],
                                                compute_logprobs=True, compute_entropy=False,
                                                compute_values=False, use_cache=False)
                        batches.append(result['logprobs'].detach().cpu())
            recomputed = torch.cat(batches).to(previous.dtype)
            if recomputed.shape != previous.shape:
                raise RuntimeError('OpenPI recomputation shape mismatch')
            delta = (recomputed - previous).float()
            action_dim = self.cfg.actor.model.action_dim
            delta = delta.reshape(*delta.shape[:-1], -1, action_dim)
            mask = self.rollout_batch.get('loss_mask')
            if mask is not None:
                valid = mask.bool().unsqueeze(-1).expand_as(delta)
                delta = delta[valid]
            if delta.numel() == 0 or not torch.isfinite(delta).all():
                raise RuntimeError('Invalid pre-update logprob audit')
            receipt = {'time_unix': time.time(), 'optimizer_steps': self.optimizer_steps,
                       'rollout_batch_size': int(previous.shape[0]),
                       'logprob_gap_mean_abs': delta.abs().mean().item(),
                       'logprob_gap_max_abs': delta.abs().max().item(),
                       'logprob_gap_p99_abs': torch.quantile(delta.abs(), .99).item(),
                       'denominator': 'original rollout logprobs retained'}
            metrics = {'actor/rollout_train_logprob_gap': receipt['logprob_gap_mean_abs']}
            for name in ['returns', 'advantages']:
                values = self.rollout_batch[name].detach().float()
                if mask is not None:
                    values = values[torch.broadcast_to(mask.bool(), values.shape)]
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
            metrics['actor/audit_logprob_gap_max'] = receipt['logprob_gap_max_abs']
            metrics['critic/audit_return_variance'] = receipt['returns_variance']
            # A material pre-update mismatch invalidates the update-count experiment.
            if receipt['logprob_gap_max_abs'] > .01:
                raise RuntimeError('Pre-update logprob mismatch exceeds 0.01; inspect batch-audit.jsonl')
            return metrics
        finally:
            self.rollout_batch['prev_logprobs'] = previous

    EmbodiedFSDPActor.recompute_prev_logprobs = audited_recompute
    EmbodiedFSDPActor.run_training = audited_training
