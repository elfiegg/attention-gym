"""A CUDA-graph capturable Inkling 2 Mini decode step: the oracle's dataflow (``ref``) with
device-side positions, static shapes and capturable kernels.

- Attention reads the whole cache (``max_len`` keys) behind its mask, so shapes never change.
- Experts run as two ``F.grouped_mm`` calls over the routed (token, expert) rows sorted by
  expert, reading only the routed experts' weights; no host sync (HF's ``grouped_mm`` path).
- Short convolutions are the oracle's (they read the rings by device position).

Numerics differ from the oracle only in summation order (grouped GEMMs for attention and
experts, the top-k sum); the decode-step test bounds the difference.
"""

import torch
import torch.nn.functional as F

from attn_gym.warpdrive.examples.inkling_megakernel.ref import (
    InklingCache,
    InklingConfig,
    InklingWeights,
    Kernels,
    LayerWeights,
    hidden_states,
    logits_of,
    require_dense_experts,
)


def grouped_experts(
    config: InklingConfig, w: LayerWeights, x: torch.Tensor, experts, weights
) -> torch.Tensor:
    """``ref.experts_out`` as grouped GEMMs over the routed rows."""
    require_dense_experts(w)
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


FAST = Kernels(experts=grouped_experts)


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
    h = hidden_states(weights, token.reshape(1), positions, cache, None, kernels)
    return logits_of(weights, h)[0]
