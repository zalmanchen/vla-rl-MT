"""OpenPI measurement worker; PPO keeps the original behavior denominator."""
import json
import gc
from pathlib import Path
import time
import torch
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.utils.nested_dict_process import split_dict_to_chunk, put_tensor_device


class DiagnosticEmbodiedFSDPActor(EmbodiedFSDPActor):
    async def sync_model_to_rollout(self):
        await super().sync_model_to_rollout()
        # The base method offloads while its local GPU state_dict is still live.
        # Release cached allocations after that local dictionary has gone away.
        if self.enable_offload:
            gc.collect()
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        self._record_memory('after_weight_sync')

    def _record_memory(self, phase):
        free, total = torch.cuda.mem_get_info(self.device)
        dest = Path(self.cfg.runner.logger.log_path) / 'memory-phases.jsonl'
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('a') as file:
            file.write(json.dumps({'phase': phase, 'global_step': int(self.version),
                                   'allocated': torch.cuda.memory_allocated(self.device),
                                   'reserved': torch.cuda.memory_reserved(self.device),
                                   'device_free': free, 'device_total': total,
                                   'time_unix': time.time()}) + '\n')

    def save_checkpoint(self, save_path, step=0):
        super().save_checkpoint(save_path, step)
        path = Path(save_path) / 'diagnostic-state.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps({'global_step': int(step),
                                         'optimizer_steps': int(self.optimizer_steps),
                                         'saved_unix': time.time()}, indent=2))
        temporary.replace(path)

    def load_checkpoint(self, load_path):
        super().load_checkpoint(load_path)
        path = Path(load_path) / 'diagnostic-state.json'
        if path.exists():
            self.optimizer_steps = int(json.loads(path.read_text())['optimizer_steps'])

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
            result = super().run_training()
            self._record_memory('after_training')
            step = int(self.version) + 1
            scheduled = step in self.cfg.actor.get('pilot_checkpoint_steps', [])
            if scheduled or (self.cfg.actor.get('pilot_save_first_update', False) and step == 1):
                path = (Path(self.cfg.runner.logger.log_path) / self.cfg.runner.logger.experiment_name
                        / f'checkpoints/global_step_{step}/actor')
                path.mkdir(parents=True, exist_ok=True)
                self.save_checkpoint(str(path), step=step)
            return result
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
        receipt = {'time_unix': time.time(), 'global_step': int(self.version), 'optimizer_steps': self.optimizer_steps,
                   'rollout_batch_size': int(previous.shape[0]),
                   'logprob_gap_mean_abs': delta.abs().mean().item(),
                   'logprob_gap_max_abs': delta.abs().max().item(),
                   'logprob_gap_p99_abs': torch.quantile(delta.abs(), .99).item(),
                   'denominator': 'original rollout logprobs retained'}
        receipt['matmul_precision'] = torch.get_float32_matmul_precision()
        receipt['cudnn_allow_tf32'] = torch.backends.cudnn.allow_tf32
        if self.cfg.actor.model.get('pilot_strict_fp32', False):
            if receipt['matmul_precision'] != 'highest' or receipt['cudnn_allow_tf32']:
                raise RuntimeError('E6 strict FP32 math policy was not applied')
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
            checkpoint = (dest.parent / self.cfg.runner.logger.experiment_name / 'checkpoints'
                          / f'global_step_{int(self.version)}' / 'actor')
            checkpoint.mkdir(parents=True, exist_ok=True)
            self.save_checkpoint(str(checkpoint), step=int(self.version))
            raise RuntimeError('Pre-update logprob mismatch exceeds0.01; inspect batch-audit.jsonl')
        assert self.rollout_batch['prev_logprobs'] is previous
        return {'actor/rollout_train_logprob_gap': receipt['logprob_gap_mean_abs'],
                'actor/audit_logprob_gap_max': receipt['logprob_gap_max_abs'],
                'critic/audit_return_variance': receipt['returns_variance']}
