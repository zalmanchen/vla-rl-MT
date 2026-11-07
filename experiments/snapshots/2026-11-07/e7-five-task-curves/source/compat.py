"""Explicit runtime adaptations for the pinned Motus OpenPI checkpoint."""

from types import MethodType

import torch.nn.functional as F


def _siglip_tanh_forward(self, x):
    x = self.fc1(x)
    x = F.gelu(x, approximate="tanh")
    x = self.dropout(x)
    return self.fc2(x)


def match_reference_siglip(model):
    """Use the GELU approximation configured by the original SigLIP model."""
    count = 0
    for layer in model.img.encoder.layers:
        layer.mlp.forward = MethodType(_siglip_tanh_forward, layer.mlp)
        count += 1
    return count


def _reference_gelu_glu(gate, value):
    return F.gelu(gate, approximate="tanh") * value


def match_reference_activations(model):
    from rlinf.models.embodiment.openpi.modules import gemma

    count = match_reference_siglip(model)
    gemma.gelu_glu = _reference_gelu_glu
    return count
