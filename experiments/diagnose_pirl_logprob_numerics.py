"""Measure PPO-relevant audit error and same-weight batch-size sensitivity."""
import fcntl
import json
import os
from pathlib import Path
import random
import argparse
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(os.environ['PROJECT_ROOT'])
RUN = ROOT / '.runtime/experiments/e6-posttraining-diagnostic'


def compare(old, new):
    delta = (new - old).float().flatten()
    # Match RLinf preprocess_loss_inputs: sum each input separately in fp32.
    log_ratio = new.reshape(len(new), -1, 14).sum((1, 2)) - old.reshape(len(old), -1, 14).sum((1, 2))
    ratio = log_ratio.exp()
    return {'components': delta.numel(), 'component_max_abs': delta.abs().max().item(),
            'component_mean_abs': delta.abs().mean().item(),
            'component_p99_abs': torch.quantile(delta.abs(), .99).item(),
            'components_over_001': int((delta.abs() > .01).sum()),
            'chunk_ratio_min': ratio.min().item(), 'chunk_ratio_max': ratio.max().item(),
            'chunk_ratio_error_max': (ratio - 1).abs().max().item(),
            'chunk_ratio_error_mean': (ratio - 1).abs().mean().item(),
            'chunks_outside_ppo_clip': int(((ratio < .8) | (ratio > 1.2)).sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--highest', action='store_true')
    parser.add_argument('--source', type=Path, default=RUN / 'aborted-attempts/ppo-update1-full-bucket-tf32-audit/failed-batch-audit.pt')
    args = parser.parse_args()
    lock = (ROOT / '.runtime/experiments/gpu-evaluation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = RUN / ('numerical-diagnostic-highest' if args.highest else 'numerical-diagnostic')
    out.mkdir(exist_ok=True)
    source = args.source
    saved = torch.load(source, map_location='cpu', weights_only=False)
    result = {'source': str(source), 'failed_batch': compare(saved['rollout_batch']['prev_logprobs'], saved['recomputed_logprobs']),
              'control': 'Original Motus weights, stored real inputs, no optimizer updates. This control tests numerical sensitivity; it is not a reload of the lost four-update checkpoint.'}
    (out / 'results.json').write_text(json.dumps(result, indent=2))
    from pirl_extension import register
    register()
    from rlinf.models.embodiment.openpi import get_model
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    cfg = OmegaConf.load(RUN / 'configs/ppo-update1-adjust_bottle-seed0.yaml').actor.model
    model = get_model(cfg, torch_dtype=torch.float32).cuda().train()
    if args.highest:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cudnn.allow_tf32 = False
    result['matmul_precision'] = torch.get_float32_matmul_precision()
    result['cudnn_allow_tf32'] = torch.backends.cudnn.allow_tf32
    inputs = saved['rollout_batch']['forward_inputs']
    def recompute(size, training):
        model.train(training)
        parts = []
        with torch.no_grad():
            for start in range(0, len(saved['recomputed_logprobs']), size):
                micro = {k: v[start:start+size].cuda() for k, v in inputs.items()}
                parts.append(model(forward_inputs=micro, compute_logprobs=True, compute_entropy=False,
                                   compute_values=False, use_cache=False)['logprobs'].cpu())
        return torch.cat(parts)
    outputs = {}
    for name, size, training in [('train4', 4, True), ('repeat4', 4, True), ('train2', 2, True), ('eval4', 4, False)]:
        outputs[name] = recompute(size, training)
        if name != 'train4':
            result[name + '_vs_train4'] = compare(outputs['train4'], outputs[name])
            print(name, result[name + '_vs_train4'], flush=True)
        (out / 'results.json').write_text(json.dumps(result, indent=2))
    torch.save(outputs, out / 'same-weight-logprobs.pt')
    (out / 'FINISHED').write_text('No optimizer update; same-weight controls complete.')


if __name__ == '__main__':
    main()
