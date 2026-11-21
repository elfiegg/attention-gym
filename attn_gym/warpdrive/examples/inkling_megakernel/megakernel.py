"""Host side of the Inkling 2 Mini decode megakernel, stage S1: a visit's MoE MLP half as one
persistent warpdrive launch (``mlp_kernel.InklingMlpKernel``), and a hybrid decode step that
runs it inside ``fast.decode_step`` for every plain batch-one visit.

``InklingMlpMegakernel.mlp_block`` has ``ref.mlp_block``'s signature (``Kernels.mlp_block``):
visits with a message (core passes after the first) or more than one token fall back to the
oracle's dataflow on the caller's kernels. The launch reads every input from device memory
(the position too) and syncs nothing, so the hybrid step stays CUDA-graph capturable.

Numerics follow ``ref.mlp_block``; outputs differ only by fp32 summation order (the block
residual's reductions, the GEMVs' accumulation, the routing weights' sum and the
convolution), which can flip a bf16 rounding.

Kernels compile from ``mlp_kernel``'s imports plus that module, so host-side edits here
recompile nothing.
"""

import dataclasses

import torch

from attn_gym.warpdrive.examples.inkling_megakernel import ref
from attn_gym.warpdrive.examples.inkling_megakernel.fast import FAST, decode_step
from attn_gym.warpdrive.examples.inkling_megakernel.mlp_kernel import (
    DOWN,
    GATEUP,
    MAX_BLOCKS,
    RING,
    ROUTER,
    ROWS,
    WORDS,
    InklingMlpKernel,
    compile_mlp_kernel,
)
from attn_gym.warpdrive.examples.inkling_megakernel.ref import (
    InklingCache,
    InklingConfig,
    InklingWeights,
    Kernels,
    LayerWeights,
)

__all__ = ["InklingMlpMegakernel", "build_table", "decode_step_megakernel", "mlp_kernel_for"]


def mlp_kernel_for(config: InklingConfig, stages: int | None = None) -> InklingMlpKernel:
    """The kernel for ``config``'s MLP sizes (built once per distinct call)."""
    if config.conv_kernel > RING or ref.CONV_RING != RING:
        raise ValueError(f"the kernel's convolution ring holds {RING} positions")
    return InklingMlpKernel.get_or_build(
        hidden=config.hidden_size,
        width=config.moe_intermediate_size,
        experts=config.num_experts,
        top_k=config.experts_per_token,
        conv_kernel=config.conv_kernel,
        rms_eps=config.rms_eps,
        gate_softcap=config.gate_softcap,
        up_softcap=config.up_softcap,
        route_scale=config.route_scale,
        stages=stages,
    )


def build_table(op: InklingMlpKernel, ctas: int) -> torch.Tensor:
    """The launch's int32 ``(rows, WORDS)`` table laid out ``[slot][cta]`` for
    ``StaticPersistent``. Rows in dependency order: ROUTER tiles, GATEUP and DOWN rows, both
    slot-major, dealt round-robin over the CTAs. Every CTA's rows stay in that order, so with all
    CTAs resident the earliest unfinished row has its producers finished; in particular a row
    tile's last-slot DOWN row, which reduces the tile, follows the rows it polls.

    Round-robin gate/up rows complete the slots in waves (slots 0 and 1 with the first row of
    every CTA), so a CTA's first down rows, of early slots, wait for nothing; only its last one
    (slot 6 or 7) waits for the last wave. Down rows start where gate/up rows end, so the
    CTAs with one row fewer of each kind are mostly different ones; the router's single-row
    items go to the least loaded CTAs. On GB300 at Inkling 2 Mini's shapes this measured about
    1.5 us per launch faster than contiguous runs of rows per CTA (which reuse a down slot's
    activation fetch but leave every slot's down rows waiting for its last gate/up rows)."""
    gate_rows = [[GATEUP, t, s] for s in range(op.top_k) for t in range(op.gate_tiles)]
    down_rows = [[DOWN, t, s] for s in range(op.top_k) for t in range(op.row_tiles)]
    lists = [[] for _ in range(ctas)]
    for j, row in enumerate(gate_rows):
        lists[j % ctas].append(row)
    shift = len(gate_rows) % ctas
    for j, row in enumerate(down_rows):
        lists[(j + shift) % ctas].append(row)
    cost = {GATEUP: op.hidden_k_tiles, DOWN: op.down_k_tiles}
    load = [sum(cost[row[0]] for row in rows) for rows in lists]
    for tile in range(op.experts // ROWS):
        cta = min(range(ctas), key=lambda c: (load[c], c))
        lists[cta].insert(0, [ROUTER, tile, 0])
        load[cta] += op.hidden_k_tiles
    slots = max(len(rows) for rows in lists)
    padding = [-1] * (WORDS - 1)
    words = [
        lists[c][s] if s < len(lists[c]) else padding for s in range(slots) for c in range(ctas)
    ]
    return torch.tensor([w + [i] for i, w in enumerate(words)], dtype=torch.int32)


def bf16_router(router: torch.Tensor) -> torch.Tensor:
    """The router as bf16; the kernel streams it so, which is exact only when every fp32 weight
    is a bf16 value (Inkling 2 Mini's are)."""
    packed = router.to(torch.bfloat16)
    if not torch.equal(packed.float(), router):
        raise ValueError("the router's fp32 weights are not exactly representable in bf16")
    return packed.contiguous()


class InklingMlpMegakernel:
    """Launch state of the MLP-half kernel for ``config``: the instruction table, the tagged
    scratch, the epoch state and bf16 router copies of ``layers``. Launches of one runner must
    run one at a time in stream order (they share the scratch and the epoch)."""

    def __init__(
        self,
        config: InklingConfig,
        layers: tuple[LayerWeights, ...],
        stages: int | None = None,
        device="cuda",
    ):
        self.config = config
        self.op = op = mlp_kernel_for(config, stages)
        self.workspace, self.max_ctas = op.work.launch_args(torch.device(device))
        self.table = build_table(op, self.max_ctas).to(device)
        self.scratch = torch.zeros(op.scratch_size, device=device, dtype=torch.int64)
        self.state = torch.zeros(2, device=device, dtype=torch.int32)
        self.routers = {id(w): bf16_router(w.router) for w in layers}
        self.layers = layers  # keeps the ids of ``routers`` valid

    def __call__(
        self,
        w: LayerWeights,
        residual: torch.Tensor,
        blocks: list[torch.Tensor],
        ring: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``ref.mlp_block`` of one token without a message: fp32 ``residual`` and bf16
        ``blocks`` (1 to MAX_BLOCKS) of ``hidden`` elements, the visit's ``(RING, hidden)``
        convolution ring (written at the position) and the int64 one-element device
        ``positions``. Returns the bf16 ``(hidden,)`` output (into ``out`` if given)."""
        hidden = self.config.hidden_size
        if not 1 <= len(blocks) <= MAX_BLOCKS:
            raise ValueError(f"need 1 to {MAX_BLOCKS} blocks, got {len(blocks)}")
        router = self.routers.get(id(w))
        if router is None:
            raise ValueError("w must be one of the layers this runner was built with")
        states = [b.reshape(hidden) for b in blocks]
        states += [states[0]] * (MAX_BLOCKS - len(states))  # unread
        if out is None:
            out = torch.empty(hidden, device=residual.device, dtype=torch.bfloat16)
        compile_mlp_kernel(self.op)(
            router,
            w.w13,
            w.w2,
            w.router_bias,
            w.mlp_query,
            w.post_norm,
            w.mlp_conv,
            residual.reshape(hidden),
            *states,
            len(blocks),
            ring,
            positions.reshape(1),
            out,
            self.scratch,
            self.state,
            self.table,
            self.workspace,
            self.max_ctas,
        )
        return out

    def routed_experts(self, w: LayerWeights) -> torch.Tensor:
        """The ``(experts_per_token,)`` experts the last launch routed to, ascending: the top-k of
        its published router logits' scores plus ``w``'s correction bias, recomputed here."""
        offset = self.op.scratch_offsets["logits"]
        words = self.scratch[offset : offset + self.op.experts]
        logits = words.view(torch.int32)[0::2].view(torch.float32)
        keys = torch.sigmoid(logits) + w.router_bias
        return keys.topk(self.op.top_k).indices.sort().values

    def mlp_block(
        self,
        config: InklingConfig,
        w: LayerWeights,
        visit: int,
        residual: torch.Tensor,
        blocks: list[torch.Tensor],
        message: torch.Tensor | None,
        cache: InklingCache,
        positions: torch.Tensor,
        kernels: Kernels,
    ) -> torch.Tensor:
        """``ref.mlp_block``: one launch for a plain one-token visit, else the oracle's dataflow
        on ``kernels``."""
        if message is not None or residual.shape[0] != 1 or len(blocks) > MAX_BLOCKS:
            return ref.mlp_block(
                config, w, visit, residual, blocks, message, cache, positions, kernels
            )
        return self(w, residual, blocks, cache.conv_mlp[visit], positions)[None]

    def kernels(self) -> Kernels:
        """``fast.FAST`` with this runner's MLP half."""
        return dataclasses.replace(FAST, mlp_block=self.mlp_block)


def decode_step_megakernel(
    weights: InklingWeights,
    token: torch.Tensor,
    pos: torch.Tensor,
    cache: InklingCache,
    runner: InklingMlpMegakernel,
) -> torch.Tensor:
    """``fast.decode_step`` with ``runner``'s MLP half on plain visits; fp32 logits."""
    return decode_step(weights, token, pos, cache, runner.kernels())
