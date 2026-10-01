# Vendored from lucidrains/x-transformers 2.2.11 (PyPI x_transformers-2.2.11-py3-none-any.whl,
# x_transformers/x_transformers.py: RotaryEmbedding, rotate_half, apply_rotary_pos_emb, exists).
# MIT License, Copyright (c) 2020 Phil Wang. The three definitions below are unchanged; only the
# imports around them are trimmed (the full module also imports loguru, einx and packaging, which
# the MeanVC2 inference path never touches).
#
# MeanVC2 pins x-transformers==2.2.11 and imports exactly these names
# (``from x_transformers.x_transformers import RotaryEmbedding, apply_rotary_pos_emb``);
# ``install()`` registers this file under that module path.

from __future__ import annotations

import sys
import types

import torch
from einops import rearrange
from torch import arange, cat, stack
from torch.amp import autocast
from torch.nn import Module


def exists(val):
    return val is not None


class RotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        use_xpos = False,
        scale_base = 512,
        interpolation_factor = 1.,
        base = 10000,
        base_rescale_factor = 1.
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1. / (base ** (arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        assert interpolation_factor >= 1.
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer('scale', None)
            return

        scale = (arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer('scale', scale)

    def forward_from_seq_len(self, seq_len):
        device = self.inv_freq.device

        t = arange(seq_len, device = device)
        return self.forward(t)

    @autocast('cuda', enabled = False)
    def forward(self, t):
        max_pos = t.max() + 1

        if t.ndim == 1:
            t = rearrange(t, 'n -> 1 n')

        freqs = torch.einsum('b i , j -> b i j', t.type_as(self.inv_freq), self.inv_freq) / self.interpolation_factor
        freqs = stack((freqs, freqs), dim = -1)
        freqs = rearrange(freqs, '... d r -> ... (d r)')

        if not exists(self.scale):
            return freqs, 1.

        power = (t - (max_pos // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, '... n -> ... n 1')
        scale = stack((scale, scale), dim = -1)
        scale = rearrange(scale, '... d r -> ... (d r)')

        return freqs, scale

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

@autocast('cuda', enabled = False)
def apply_rotary_pos_emb(t, freqs, scale = 1):
    rot_dim, seq_len, orig_dtype = freqs.shape[-1], t.shape[-2], t.dtype

    freqs = freqs[:, -seq_len:, :]
    scale = scale[:, -seq_len:, :] if isinstance(scale, torch.Tensor) else scale

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, 'b n d -> b 1 n d')

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    out = cat((t, t_unrotated), dim = -1)

    return out.type(orig_dtype)


def install():
    """Register this module as ``x_transformers.x_transformers`` (and an empty parent package)."""
    if "x_transformers.x_transformers" in sys.modules:
        return sys.modules["x_transformers.x_transformers"]
    pkg = sys.modules.get("x_transformers") or types.ModuleType("x_transformers")
    pkg.__path__ = []
    me = sys.modules[__name__]
    pkg.x_transformers = me
    sys.modules["x_transformers"] = pkg
    sys.modules["x_transformers.x_transformers"] = me
    return me
