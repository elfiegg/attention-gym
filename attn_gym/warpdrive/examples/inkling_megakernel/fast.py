"""A CUDA-graph capturable Inkling 2 Mini decode step: the oracle's dataflow (``ref``) with
device-side positions, static shapes and capturable kernels.

- Attention reads the whole cache (``max_len`` keys) behind its mask, so shapes never change.
- Experts run as two ``F.grouped_mm`` calls over the routed (token, expert) rows sorted by
  expert, reading only the routed experts' weights; no host sync (HF's ``grouped_mm`` path).
- Static short convolutions run on ``attn_gym.linear.short_conv.causal_conv1d`` in fp32 with
  the three earlier inputs from the cache rings as its initial state.

Numerics differ from the oracle only in summation order (grouped GEMMs, the top-k sum, the
convolution kernel); the decode-step test bounds the difference.
"""

import math

import torch
import torch.nn.functional as F

from attn_gym.linear.short_conv import causal_conv1d
from attn_gym.warpdrive.examples.inkling_megakernel.ref import (
    CONV_RING,
    InklingCache,
    InklingConfig,
    InklingWeights,
    Kernels,
    LayerWeights,
    hidden_states,
    logits_of,
)


def short_conv(ring, x: torch.Tensor, weight: torch.Tensor, positions, kernel: int):
    """``ref.short_conv`` on the short-convolution library kernel."""
    earlier = positions[0] - torch.arange(kernel - 1, 0, -1, device=x.device)
    past = ring[earlier % CONV_RING] * (earlier >= 0)[:, None].to(ring.dtype)
    ring[positions[-CONV_RING:] % CONV_RING] = x[-CONV_RING:]
    out = causal_conv1d(x.float()[None], weight, initial_state=past.float()[None])[0]
    return ((out + x.float()) / math.sqrt(kernel + 1)).to(torch.bfloat16)


def grouped_experts(
    config: InklingConfig, w: LayerWeights, x: torch.Tensor, experts, weights
) -> torch.Tensor:
    """``ref.experts_out`` as grouped GEMMs over the routed rows."""
    tokens, k = experts.shape
    flat = experts.reshape(-1)
    order = flat.argsort(stable=True)
    counts = torch.histc(flat.int(), bins=config.num_experts, min=0, max=config.num_experts - 1)
    offsets = counts.cumsum(0).int()
    rows = x[order // k]
    gate, up = F.grouped_mm(rows, w.w13.transpose(1, 2), offs=offsets).float().chunk(2, dim=-1)
    act = torch.tanh(gate / config.gate_softcap) * config.gate_softcap * torch.sigmoid(gate)
    act = act * (torch.tanh(up / config.up_softcap) * config.up_softcap)
    y = F.grouped_mm(act.to(torch.bfloat16), w.w2.transpose(1, 2), offs=offsets)
    y = y.float() * weights.reshape(-1)[order, None]
    unsorted = torch.empty_like(y).index_copy_(0, order, y)
    return unsorted.view(tokens, k, -1).sum(1).to(torch.bfloat16)


FAST = Kernels(short_conv=short_conv, experts=grouped_experts)


def decode_step(
    weights: InklingWeights,
    token: torch.Tensor,
    pos: torch.Tensor,
    cache: InklingCache,
    kernels: Kernels = FAST,
) -> torch.Tensor:
    """One decode step of the one-element ``token`` at the one-element device position ``pos``;
    writes the caches there and returns fp32 logits ``(vocab,)``. No host sync as long as
    ``kernels`` sync nothing (``FAST``'s do not)."""
    positions = pos.reshape(1).long()
    h = hidden_states(weights, token.reshape(1), positions, cache, cache.max_len, kernels)
    return logits_of(weights, h)[0]
