"""Minimal shared-prefix value probe for M0, outside the pinned RLinf tree.

The zero-initialized residual adapter is applied before the prefix enters the
native Pi0 transformer. Both action prediction and prefix-only value encoding
use it. All original policy parameters stay frozen. This is an engineering
candidate, not an implementation of the original RECAP critic or a trainer.
"""

from contextlib import nullcontext

import torch
from torch import nn


def pool_prefix(hidden, mask, prompt_mask=None, *, readout="global_mean"):
    """Read fused prefix states; prompt positions include instruction AND q.

    Token groups identify positions, not disentangled modalities after attention.
    The duplicate control matches the concatenated readout's input/head shape.
    """
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("Expected [batch, tokens, width] states and a boolean prefix mask")

    def mean(tokens, valid):
        weights = valid.unsqueeze(-1).float()
        return (tokens.float()*weights).sum(1)/weights.sum(1).clamp_min(1)

    global_mean = mean(hidden, mask)
    if readout == "global_mean":
        return global_mean
    if readout == "global_duplicate":
        return torch.cat((global_mean, global_mean), dim=-1)
    if readout != "vision_prompt_concat":
        raise ValueError(f"Unknown prefix readout: {readout}")
    if prompt_mask is None or prompt_mask.ndim != 2 or prompt_mask.dtype != torch.bool:
        raise ValueError("Grouped readout requires the actual boolean prompt mask")
    prompt_length = prompt_mask.shape[1]
    if (prompt_mask.shape[0] != hidden.shape[0] or not 0 < prompt_length < hidden.shape[1]
            or not torch.equal(mask[:, -prompt_length:], prompt_mask)):
        raise ValueError("Prompt must be the verified suffix of the image+prompt prefix")
    visual_mask = mask[:, :-prompt_length]
    if not bool(visual_mask.any(1).all()) or not bool(prompt_mask.any(1).all()):
        raise ValueError("Each grouped sample requires both visual and prompt tokens")
    return torch.cat((mean(hidden[:, :-prompt_length], visual_mask),
                      mean(hidden[:, -prompt_length:], prompt_mask)), dim=-1)


class PrefixAdapter(nn.Module):
    def __init__(self, width, rank):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, tokens):
        residual = self.up(torch.nn.functional.silu(self.down(tokens.float())))
        return tokens + residual.to(tokens.dtype)


class SharedPrefixValue(nn.Module):
    """Attach a shared adapter and scalar head without editing native Pi0.

    The caller must close the hook before attaching another probe to this
    policy. Checkpoints contain only the addon; frozen base weights are pinned
    separately. Value encoding deliberately never accepts action suffix tokens.
    """

    def __init__(self, policy, width=2048, rank=32, head_width=128,
                 readout="global_mean", register_policy=True):
        super().__init__()
        if getattr(policy, "_shared_value_probe_attached", False):
            raise ValueError("A shared value probe is already attached to this policy")
        policy.requires_grad_(False)
        if register_policy:
            self.policy = policy
        else:
            # Single-GPU addon training: the pinned, frozen policy remains on
            # device, outside FSDP and addon checkpoints. Autograd still crosses
            # its operations. Only the adapter/head are wrapped and optimized.
            object.__setattr__(self, "policy", policy)
        if readout not in ("global_mean", "global_duplicate", "vision_prompt_concat"):
            raise ValueError(f"Unknown prefix readout: {readout}")
        self.readout = readout
        head_input = width if readout == "global_mean" else 2*width
        self.adapter = PrefixAdapter(width, rank)
        self.value_head = nn.Sequential(
            nn.LayerNorm(head_input), nn.Linear(head_input, head_width), nn.SiLU(), nn.Linear(head_width, 1)
        )
        self.architecture = {"width": width, "rank": rank, "head_width": head_width, "readout": readout}
        self._hook = policy.llm.register_forward_pre_hook(self._adapt_prefix, with_kwargs=True)
        policy._shared_value_probe_attached = True

    def _adapt_prefix(self, module, args, kwargs):
        embedded = args[0] if args else kwargs["embedded"]
        if embedded[0] is None:
            return None
        replacement = [self.adapter(embedded[0]), *embedded[1:]]
        if args:
            return (replacement, *args[1:]), kwargs
        return args, {**kwargs, "embedded": replacement}

    def train(self, mode=True):
        super().train(mode)
        self.policy.train(mode)
        # Keep the transformer training flag for native non-reentrant activation
        # checkpointing. The frozen image encoder does not need augmentation or
        # training-time stochasticity.
        self.policy.img.eval()
        return self

    def prefix_tokens(self, observation, *, detach_shared=False):
        from rlinf.models.embodiment.openpi.modules.model import (
            _observation_to_dtype,
            preprocess_observation,
        )

        with torch.no_grad() if detach_shared else nullcontext():
            observation = preprocess_observation(observation, train=False)
            observation = _observation_to_dtype(observation, self.policy.embed_dtype)
            hidden, mask, cache = self.policy.build_prefix_cache(observation)
            del cache
        return hidden, mask

    def features(self, observation, *, detach_shared=False, readout="global_mean"):
        if readout == "vision_prompt_concat" and getattr(self.policy, "pcd", None):
            raise ValueError("Grouped M1 readout supports image+prompt prefixes without point clouds")
        hidden, mask = self.prefix_tokens(observation, detach_shared=detach_shared)
        return pool_prefix(hidden, mask, observation.tokenized_prompt_mask, readout=readout)

    def value(self, observation, *, detach_shared=False):
        return self.value_head(self.features(observation, detach_shared=detach_shared,
                                             readout=self.readout)).squeeze(-1)

    def sft_loss(self, observation, actions, *, noise, flow_time, action_mask=None):
        # No image augmentation during M0: fixed noise/time must expose only the
        # gradient-routing intervention, and use native Pi0's FM implementation.
        loss = self.policy.compute_loss(
            observation, actions, train=False, noise=noise, time=flow_time
        )
        return loss.mean() if action_mask is None else masked_flow_mse(loss, action_mask)

    def addon_state(self):
        return {name: {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
                for name, module in (("adapter", self.adapter), ("value_head", self.value_head))}

    def load_addon(self, state):
        if set(state) != {"adapter", "value_head"}:
            raise ValueError("Expected exactly adapter and value_head checkpoint states")
        self.adapter.load_state_dict(state["adapter"], strict=True)
        self.value_head.load_state_dict(state["value_head"], strict=True)

    def close(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
            self.policy._shared_value_probe_attached = False


def masked_flow_mse(elementwise_loss, action_mask):
    """Average real executed physical actions within each decision, then batch.

    Padding has no direct loss. Native suffix attention still sees the fixed
    padded context; this helper does not change its attention implementation.
    """
    if action_mask.shape != elementwise_loss.shape or action_mask.dtype != torch.bool:
        raise ValueError("Action mask must be boolean and match [batch, horizon, action_dim]")
    counts = action_mask.sum(dim=(1, 2))
    if not bool((counts > 0).all()):
        raise ValueError("Each SFT decision needs at least one real physical action")
    return (elementwise_loss.masked_fill(~action_mask, 0).sum(dim=(1, 2))/counts).mean()


def isolated_optimizer(probe, learning_rate):
    return torch.optim.AdamW([
        {"params": probe.adapter.parameters()}, {"params": probe.value_head.parameters()}
    ], lr=learning_rate, weight_decay=0.0, foreach=False)


def isolated_clip(probe, max_norm=1.0):
    """A detached critic's gradient norm must not change the actor's clipping."""
    for module in (probe.adapter, probe.value_head):
        nn.utils.clip_grad_norm_(module.parameters(), max_norm, error_if_nonfinite=True)


def gradient_inventory(probe):
    result = {}
    for name, module in (("frozen_policy", probe.policy), ("adapter", probe.adapter),
                         ("value_head", probe.value_head)):
        parameters = list(module.parameters())
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        result[name] = {
            "parameters": sum(parameter.numel() for parameter in parameters),
            "trainable": sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
            "gradient_tensors": len(gradients),
            "gradient_l2": sum(float(grad.float().square().sum()) for grad in gradients) ** 0.5,
            "finite": all(bool(torch.isfinite(grad).all()) for grad in gradients),
        }
    return result
