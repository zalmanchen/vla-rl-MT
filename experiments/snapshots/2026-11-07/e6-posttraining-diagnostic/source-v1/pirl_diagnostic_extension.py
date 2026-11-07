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
    import time
    from pathlib import Path
    import torch
    from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor

    original = EmbodiedFSDPActor.recompute_prev_logprobs

    def audited_recompute(self, batch_size_per_rank):
        if not self.cfg.actor.get('pilot_diagnostics', False):
            return original(self, batch_size_per_rank)
        previous = self.rollout_batch['prev_logprobs']
        try:
            metrics = original(self, batch_size_per_rank)
            recomputed = self.rollout_batch['prev_logprobs']
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
