"""Host side of the Inkling 2 Mini decode megakernel, stages S1-S2: a visit's attention half
(``attn_kernel.InklingAttnKernel``) and MoE MLP half (``mlp_kernel.InklingMlpKernel``) as one
persistent warpdrive launch each, and a hybrid decode step that runs them inside
``fast.decode_step`` for every plain batch-one visit.

``InklingAttnMegakernel.attention_block`` and ``InklingMlpMegakernel.mlp_block`` have the
signatures of ``ref.attention_block`` and ``ref.mlp_block`` (``Kernels``): visits with messages
(core passes after the first) or more than one token fall back to the oracle's dataflow on the
caller's kernels. A plain visit is two launches back to back: the attention launch writes the
fp32 post-attention residual into a device buffer the MLP launch reads, with no torch op
between them. Launches read every input from device memory (the position too) and sync nothing,
so the hybrid step stays CUDA-graph capturable.

Numerics follow the oracle with ``start=None``; outputs differ by fp32 summation order (block
residuals, GEMV accumulation, convolutions, routing weights) and, in attention, by the flash
merge (probabilities are not rounded to bf16 before the value sum), which can flip a bf16
rounding.

Kernels compile from ``attn_kernel``'s and ``mlp_kernel``'s imports plus those modules, so
host-side edits here recompile nothing.
"""

import dataclasses
from dataclasses import dataclass

import torch

from attn_gym.warpdrive.examples.inkling_megakernel import ref
from attn_gym.warpdrive.examples.inkling_megakernel.attn_kernel import (
    ATTN,
    BLOCK,
    HEAD_DIM,
    O_SPLITS,
    OPROJ,
    QKV,
    InklingAttnKernel,
    compile_attn_kernel,
)
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

__all__ = [
    "InklingAttnMegakernel",
    "InklingMegakernel",
    "InklingMlpMegakernel",
    "attn_kernel_for",
    "build_attn_table",
    "build_table",
    "decode_step_megakernel",
    "mlp_kernel_for",
]

# Default KV splits per q head. A sliding layer reads at most window / 64 + 1 = 17 blocks, so 4
# splits keep an item at 5 blocks or fewer; full layers read every cached block.
SLIDING_SPLITS, FULL_SPLITS = 4, 8


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


def attn_kernel_for(
    config: InklingConfig, full: bool, splits: int | None = None, stages: int | None = None
) -> InklingAttnKernel:
    """The attention kernel for ``config``'s full (``full``) or sliding layers."""
    if config.conv_kernel > RING or ref.CONV_RING != RING:
        raise ValueError(f"the kernel's convolution ring holds {RING} positions")
    if config.head_dim != HEAD_DIM:
        raise ValueError(f"head dim must be {HEAD_DIM}, got {config.head_dim}")
    return InklingAttnKernel.get_or_build(
        hidden=config.hidden_size,
        q_heads=config.num_q_heads,
        kv_heads=config.num_kv_heads,
        d_rel=config.d_rel,
        rel_width=config.rel_extent,
        extent=config.rel_extent if full else config.sliding_window,
        conv_kernel=config.conv_kernel,
        rms_eps=config.rms_eps,
        qk_clip=config.qk_clip,
        rel_clip=config.rel_clip,
        splits=splits or (FULL_SPLITS if full else SLIDING_SPLITS),
        full=full,
        window=0 if full else config.sliding_window,
        log_alpha=config.log_scaling_alpha if full else 0.0,
        log_floor=config.log_scaling_floor if full else 0,
        stages=stages,
    )


def deal(rows: list[list[int]], ctas: int) -> torch.Tensor:
    """``rows`` (in dependency order) dealt round-robin over ``ctas`` as an int32 ``(slots *
    ctas, WORDS)`` table laid out ``[slot][cta]``, padded with negative-op rows."""
    lists = [rows[c::ctas] for c in range(ctas)]
    slots = max(len(r) for r in lists)
    padding = [-1] * (WORDS - 1)
    words = [
        lists[c][s] if s < len(lists[c]) else padding for s in range(slots) for c in range(ctas)
    ]
    return torch.tensor([w + [i] for i, w in enumerate(words)], dtype=torch.int32)


def build_attn_table(op: InklingAttnKernel, ctas: int) -> torch.Tensor:
    """The attention launch's table for ``StaticPersistent``: QKV tiles (attention's
    projections first, the gate and sink last), ATTN items split-major, then OPROJ rows
    split-major, dealt round-robin. Every CTA's rows stay in that order, so with all CTAs
    resident the earliest unfinished row has its producers finished; the last split's OPROJ
    rows, which reduce their tiles, follow the rows they poll."""
    rows = [[QKV, t, m] for m in op.matrices for t in range(op.proj_tiles[m])]
    rows += [[ATTN, hq, s] for s in range(op.splits) for hq in range(op.q_heads)]
    rows += [[OPROJ, t, s] for s in range(O_SPLITS) for t in range(op.row_tiles)]
    return deal(rows, ctas)


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


@dataclass(frozen=True)
class AttnLayer:
    """A layer's attention arguments in the kernel's form: per-stream block-residual queries
    and norm weights (one stream in sliding layers, three in full ones), the dynamic tap
    projections (placeholders in sliding layers, which read none) and the k/v taps."""

    query: torch.Tensor  # (streams, hidden) fp32; zeros (unread) at layer 0
    norms: torch.Tensor  # (streams, hidden) bf16
    k_dynamic: torch.Tensor  # (kv_heads * kernel, hidden) bf16
    v_dynamic: torch.Tensor
    k_taps: torch.Tensor  # sliding: (kv_dim, kernel) fp32; full: (kernel, kv_dim) bf16
    v_taps: torch.Tensor

    @classmethod
    def of(cls, config: InklingConfig, w: LayerWeights) -> "AttnLayer":
        if w.q_norm is not None:  # a full layer
            return cls(
                query=w.attn_query.contiguous(),
                norms=torch.stack((w.q_norm, w.k_norm, w.v_norm)),
                k_dynamic=w.k_dynamic,
                v_dynamic=w.v_dynamic,
                k_taps=w.k_static.contiguous(),
                v_taps=w.v_static.contiguous(),
            )
        query = w.attn_query
        if query is None:
            query = torch.zeros(1, config.hidden_size, device=w.wq.device)
        placeholder = w.wk[: config.num_kv_heads * config.conv_kernel]
        return cls(
            query=query.contiguous(),
            norms=w.input_norm[None].contiguous(),
            k_dynamic=placeholder,
            v_dynamic=placeholder,
            k_taps=w.k_conv.contiguous(),
            v_taps=w.v_conv.contiguous(),
        )


class AttnLaunch:
    """One attention kernel's launch state: instruction table, tagged scratch, epoch and the
    trace buffer (a placeholder unless a ``trace_session`` swaps in its own)."""

    def __init__(self, op: InklingAttnKernel, device, trace: str | None = None):
        self.workspace, self.max_ctas = op.work.launch_args(torch.device(device))
        self.table = build_attn_table(op, self.max_ctas).to(device)
        if trace is not None:  # a work item and each of its regions per row of a CTA
            events = 8 * -(-self.table.shape[0] // self.max_ctas) + 16
            op = op.with_settings(trace_backend=trace, trace_events_per_role=events)
        self.op = op
        self.scratch = torch.zeros(op.scratch_size, device=device, dtype=torch.int64)
        self.state = torch.zeros(2, device=device, dtype=torch.int32)
        self.trace_buffer = torch.zeros(1, device=device, dtype=torch.int64)


class InklingAttnMegakernel:
    """Launch state of the attention-half kernels for ``config`` (one per layer kind) and the
    kernel-form arguments of ``layers``. Launches of one runner must run one at a time in stream
    order (each kind's launches share a scratch and an epoch). ``splits``: KV splits per q head
    of (sliding, full) layers. ``trace``: compile with ranges for the work items and
    ``trace_region``s, recorded by ``"globaltimer"`` (into ``launches[full].trace_buffer``) or
    ``"iket"``."""

    def __init__(
        self,
        config: InklingConfig,
        layers: tuple[LayerWeights, ...],
        splits: tuple[int | None, int | None] = (None, None),
        stages: int | None = None,
        device="cuda",
        trace: str | None = None,
    ):
        self.config = config
        self.launches = {
            full: AttnLaunch(attn_kernel_for(config, full, splits[full], stages), device, trace)
            for full in (False, True)
        }
        self.args = {id(w): AttnLayer.of(config, w) for w in layers}
        self.layers = layers  # keeps the ids of ``args`` valid

    def __call__(
        self,
        w: LayerWeights,
        layer: int,
        visit: int,
        h: torch.Tensor,
        blocks: list[torch.Tensor],
        cache: InklingCache,
        positions: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``ref.attention_block`` of one token without a message (``start=None``): bf16
        ``h`` and ``blocks`` (0 to MAX_BLOCKS) of ``hidden`` elements, ``visit``'s caches
        (written at the position) and the int64 one-element device ``positions``. Returns the
        fp32 ``(hidden,)`` post-attention residual (into ``out`` if given)."""
        config, hidden = self.config, self.config.hidden_size
        if len(blocks) > MAX_BLOCKS:
            raise ValueError(f"need at most {MAX_BLOCKS} blocks, got {len(blocks)}")
        if cache.max_len % BLOCK:
            raise ValueError(f"the cache capacity must be a multiple of {BLOCK}")
        args = self.args.get(id(w))
        if args is None:
            raise ValueError("w must be one of the layers this runner was built with")
        launch = self.launches[config.is_full(layer)]
        h = h.reshape(hidden)
        states = [b.reshape(hidden) for b in blocks]
        states += [h] * (MAX_BLOCKS - len(states))  # unread
        if out is None:
            out = torch.empty(hidden, device=h.device, dtype=torch.float32)
        compile_attn_kernel(launch.op)(
            w.wq,
            w.wk,
            w.wv,
            w.wrel,
            args.k_dynamic,
            args.v_dynamic,
            w.wgate,
            w.wsink,
            w.wo,
            cache.k[visit].view(-1, HEAD_DIM),
            cache.v[visit].view(-1, HEAD_DIM),
            args.query,
            args.norms,
            w.q_scale,
            w.rel_scale,
            w.rel_proj,
            args.k_taps,
            args.v_taps,
            w.attn_conv,
            h,
            *states,
            len(blocks),
            int(w.attn_query is not None),
            int(layer % config.block_layers == 0),
            cache.conv_k[visit],
            cache.conv_v[visit],
            cache.conv_attn[visit],
            positions.reshape(1),
            out,
            launch.trace_buffer,
            launch.scratch,
            launch.state,
            launch.table,
            launch.workspace,
            launch.max_ctas,
        )
        return out

    def attention_block(
        self,
        config: InklingConfig,
        w: LayerWeights,
        layer: int,
        visit: int,
        h: torch.Tensor,
        blocks: list[torch.Tensor],
        messages: torch.Tensor | None,
        cache: InklingCache,
        positions: torch.Tensor,
        start: int | None,
        kernels: Kernels,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """``ref.attention_block``: one launch for a plain one-token visit with device
        positions (``start=None``), else the oracle's dataflow on ``kernels``."""
        if (
            messages is not None
            or start is not None
            or h.shape[0] != 1
            or len(blocks) > MAX_BLOCKS
        ):
            return ref.attention_block(
                config, w, layer, visit, h, blocks, messages, cache, positions, start, kernels
            )
        residual = self(w, layer, visit, h, blocks, cache, positions)[None]
        if layer % config.block_layers == 0:
            return residual, blocks + [h]
        return residual, blocks


class InklingMegakernel:
    """Both halves of a visit: an attention runner and an MLP runner for ``layers``."""

    def __init__(self, config: InklingConfig, layers: tuple[LayerWeights, ...], device="cuda"):
        self.attn = InklingAttnMegakernel(config, layers, device=device)
        self.mlp = InklingMlpMegakernel(config, layers, device=device)

    def kernels(self) -> Kernels:
        """``fast.FAST`` with both halves on plain visits."""
        return dataclasses.replace(
            FAST, attention_block=self.attn.attention_block, mlp_block=self.mlp.mlp_block
        )


def decode_step_megakernel(
    weights: InklingWeights,
    token: torch.Tensor,
    pos: torch.Tensor,
    cache: InklingCache,
    runner: "InklingMegakernel | InklingMlpMegakernel",
) -> torch.Tensor:
    """``fast.decode_step`` with ``runner``'s kernels (both halves, or the MLP half) on plain
    visits; fp32 logits."""
    return decode_step(weights, token, pos, cache, runner.kernels())
