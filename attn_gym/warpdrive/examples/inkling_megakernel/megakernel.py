"""Host side of the Inkling 2 Mini decode megakernel: the whole batch-one decode step as
warpdrive launches (``InklingDecoder``), and runners of each launch kind:

- ``InklingAttnMegakernel``: a visit's attention half (``attn_kernel``), with the token's
  embedding at layer 0 and the q/k/v message gates in core passes after the first;
- ``InklingMlpMegakernel``: a visit's MoE MLP half (``mlp_kernel``), with the MLP message gate;
- ``InklingMsgMegakernel``: a core pass's messages and message halves (``msg_kernel``);
- ``InklingHeadMegakernel``: the final block residual, norm, LM head and softcap
  (``head_kernel``).

A step is 2 launches per visit (96), one per core pass after the first (2) and the head: the
attention launch writes the fp32 post-attention residual into a device buffer the MLP launch
reads, and each visit's output stays in a per-visit buffer that later block residuals read.
Launches read every input from device memory (token and position too) and sync nothing, so a
step is CUDA-graph capturable, and every launch kind compiles once for all positions.

``InklingAttnMegakernel.attention_block`` and ``InklingMlpMegakernel.mlp_block`` have the
signatures of ``ref.attention_block`` and ``ref.mlp_block`` (``Kernels``) for the hybrid step
``decode_step_megakernel``: plain one-token visits run as launches, others fall back to the
oracle's dataflow on the caller's kernels.

Numerics follow the oracle with ``start=None``; outputs differ by fp32 summation order (block
residuals, GEMV accumulation, convolutions, routing weights, message gates) and, in attention,
by the flash merge (probabilities are not rounded to bf16 before the value sum), which can flip
a bf16 rounding.

Kernels compile from the kernel modules and their imports, so host-side edits here recompile
nothing.
"""

import dataclasses
from dataclasses import dataclass
from typing import NamedTuple

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
from attn_gym.warpdrive.examples.inkling_megakernel.common import (
    GATE_RANK,
    GATE_TILES,
    GATES,
    MAX_BLOCKS,
    RING,
    WORDS,
)
from attn_gym.warpdrive.examples.inkling_megakernel.fast import FAST, decode_step
from attn_gym.warpdrive.examples.inkling_megakernel.head_kernel import (
    LMHEAD,
    InklingHeadKernel,
    compile_head_kernel,
)
from attn_gym.warpdrive.examples.inkling_megakernel.mlp_kernel import (
    DOWN,
    GATEUP,
    ROUTER,
    InklingMlpKernel,
    compile_mlp_kernel,
)
from attn_gym.warpdrive.examples.inkling_megakernel.msg_kernel import (
    MSG,
    InklingMsgKernel,
    compile_msg_kernel,
)
from attn_gym.warpdrive.examples.inkling_megakernel.ref import (
    InklingCache,
    InklingConfig,
    InklingWeights,
    Kernels,
    LayerWeights,
)
from attn_gym.warpdrive.examples.qwen_megakernel.core import ROWS

__all__ = [
    "GateArgs",
    "InklingAttnMegakernel",
    "InklingDecoder",
    "InklingHeadMegakernel",
    "InklingMegakernel",
    "InklingMlpMegakernel",
    "InklingMsgMegakernel",
    "MessageGates",
    "attn_kernel_for",
    "build_attn_table",
    "build_table",
    "decode_step_megakernel",
    "mlp_kernel_for",
]

# Default KV splits per q head. A sliding layer reads at most window / 64 + 1 = 17 blocks, so 4
# splits keep an item at 5 blocks or fewer; full layers read every cached block.
SLIDING_SPLITS, FULL_SPLITS = 4, 8


def check_rings(config: InklingConfig):
    if config.conv_kernel > RING or ref.CONV_RING != RING:
        raise ValueError(f"the kernels' convolution rings hold {RING} positions")


def mlp_kernel_for(
    config: InklingConfig, messages: bool = False, stages: int | None = None
) -> InklingMlpKernel:
    """The MLP kernel for ``config``'s sizes, with the message gate (``messages``) or without."""
    check_rings(config)
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
        messages=messages,
        stages=stages,
    )


def attn_kernel_for(
    config: InklingConfig,
    full: bool,
    messages: bool = False,
    splits: int | None = None,
    stages: int | None = None,
) -> InklingAttnKernel:
    """The attention kernel for ``config``'s full (``full``) or sliding layers, with message
    gates (``messages``) or without."""
    check_rings(config)
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
        messages=messages,
        stages=stages,
    )


def table_of(lists: list[list[list[int]]]) -> torch.Tensor:
    """Per-CTA row lists as an int32 ``(slots * ctas, WORDS)`` table laid out ``[slot][cta]``
    for ``StaticPersistent``, padded with negative-op rows."""
    ctas, slots = len(lists), max(len(rows) for rows in lists)
    padding = [-1] * (WORDS - 1)
    words = [
        lists[c][s] if s < len(lists[c]) else padding for s in range(slots) for c in range(ctas)
    ]
    return torch.tensor([w + [i] for i, w in enumerate(words)], dtype=torch.int32)


def deal(rows: list[list[int]], ctas: int) -> list[list[list[int]]]:
    """``rows`` (in dependency order) dealt round-robin over ``ctas`` lists."""
    return [rows[c::ctas] for c in range(ctas)]


def build_attn_table(op: InklingAttnKernel, ctas: int) -> torch.Tensor:
    """The attention launch's table for ``StaticPersistent``: message-gate rows, QKV tiles
    (attention's projections first, the gate and sink last), ATTN items split-major, then OPROJ
    rows split-major, dealt round-robin. Every CTA's rows stay in that order, so with all CTAs
    resident the earliest unfinished row has its producers finished; the last split's OPROJ
    rows, which reduce their tiles, follow the rows they poll."""
    rows = op.gate_rows()
    rows += [[QKV, t, m] for m in op.matrices for t in range(op.proj_tiles[m])]
    rows += [[ATTN, hq, s] for s in range(op.splits) for hq in range(op.q_heads)]
    rows += [[OPROJ, t, s] for s in range(O_SPLITS) for t in range(op.row_tiles)]
    return table_of(deal(rows, ctas))


def build_table(op: InklingMlpKernel, ctas: int) -> torch.Tensor:
    """The MLP launch's table for ``StaticPersistent``. Rows in dependency order: message-gate
    rows (dealt round-robin, first in every CTA), ROUTER tiles, GATEUP and DOWN rows, both
    slot-major, dealt round-robin over the CTAs. Every CTA's rows stay in that order, so with
    all CTAs resident the earliest unfinished row has its producers finished; in particular a
    row tile's last-slot DOWN row, which reduces the tile, follows the rows it polls.

    Round-robin gate/up rows complete the slots in waves (slots 0 and 1 with the first row of
    every CTA), so a CTA's first down rows, of early slots, wait for nothing; only its last one
    (slot 6 or 7) waits for the last wave. Down rows start where gate/up rows end, so the
    CTAs with one row fewer of each kind are mostly different ones; the router's single-row
    items go to the least loaded CTAs. On GB300 at Inkling 2 Mini's shapes this measured about
    1.5 us per launch faster than contiguous runs of rows per CTA (which reuse a down slot's
    activation fetch but leave every slot's down rows waiting for its last gate/up rows)."""
    gate_up = [[GATEUP, t, s] for s in range(op.top_k) for t in range(op.gate_tiles)]
    down_rows = [[DOWN, t, s] for s in range(op.top_k) for t in range(op.row_tiles)]
    lists = [[] for _ in range(ctas)]
    for j, row in enumerate(gate_up):
        lists[j % ctas].append(row)
    shift = len(gate_up) % ctas
    for j, row in enumerate(down_rows):
        lists[(j + shift) % ctas].append(row)
    cost = {GATEUP: op.hidden_k_tiles, DOWN: op.down_k_tiles}
    load = [sum(cost[row[0]] for row in rows) for rows in lists]
    for tile in range(op.experts // ROWS):
        cta = min(range(ctas), key=lambda c: (load[c], c))
        lists[cta].insert(0, [ROUTER, tile, 0])
        load[cta] += op.hidden_k_tiles
    gates = deal(op.gate_rows(), ctas)
    return table_of([g + rows for g, rows in zip(gates, lists)])


def bf16_router(router: torch.Tensor) -> torch.Tensor:
    """The router as bf16; the kernel streams it so, which is exact only when every fp32 weight
    is a bf16 value (Inkling 2 Mini's are)."""
    packed = router.to(torch.bfloat16)
    if not torch.equal(packed.float(), router):
        raise ValueError("the router's fp32 weights are not exactly representable in bf16")
    return packed.contiguous()


class GateArgs(NamedTuple):
    """One core layer's message gates in the kernels' form (``common.fake_gate_args``) and the
    current pass's message halves and messages of that layer."""

    down: torch.Tensor  # (GATES * GATE_RANK, 2 * hidden) bf16
    fg: torch.Tensor  # (GATES * hidden, 4 * GATE_RANK) bf16 view of the fp32 [F | G] rows
    mhalf: torch.Tensor  # (GATES * GATE_RANK,) fp32
    messages: torch.Tensor  # (GATES, hidden) fp32

    @classmethod
    def placeholder(cls, hidden: int, device) -> "GateArgs":
        """Operands of the right shapes for launches without message gates (never read)."""
        rows = GATES * GATE_RANK
        return cls(
            torch.zeros(rows, 2 * hidden, device=device, dtype=torch.bfloat16),
            torch.zeros(GATES * hidden, 4 * GATE_RANK, device=device, dtype=torch.bfloat16),
            torch.zeros(rows, device=device),
            torch.zeros(GATES, hidden, device=device),
        )


class MessageGates:
    """The message gates of the core layers ``core`` (in order) in the kernels' form and one
    pass's message buffers, which ``InklingMsgMegakernel`` fills from ``message_queries``
    (``(layers, GATES, hidden)`` fp32) and the pass's launches read (``layer``)."""

    def __init__(self, core: tuple[LayerWeights, ...], message_queries: torch.Tensor):
        down = torch.stack([torch.stack([g.down for g in w.gates]) for w in core])
        if down.shape[2] != GATE_RANK:
            raise ValueError(f"the kernels' message gates have rank {GATE_RANK}")
        layers, hidden = len(core), down.shape[-1] // 2
        self.down = down.reshape(layers * GATES * GATE_RANK, 2 * hidden).contiguous()
        fg = torch.stack(
            [torch.stack([torch.cat((g.forget, g.gain), 1) for g in w.gates]) for w in core]
        )  # (layers, GATES, hidden, 2 * GATE_RANK) fp32
        self.fg = fg.contiguous().view(torch.bfloat16).reshape(layers, GATES * hidden, -1)
        self.queries = message_queries.reshape(layers * GATES, hidden).contiguous()
        self.messages = torch.zeros_like(self.queries)
        self.mhalf = torch.zeros(layers * GATES * GATE_RANK, device=self.queries.device)

    def layer(self, core: int) -> GateArgs:
        """Core layer ``core``'s gates and the current pass's message state."""
        rows = slice(core * GATES * GATE_RANK, (core + 1) * GATES * GATE_RANK)
        messages = self.messages[core * GATES : (core + 1) * GATES]
        return GateArgs(self.down[rows], self.fg[core], self.mhalf[rows], messages)


class Launch:
    """One kernel's launch state: instruction table, tagged scratch, epoch and the trace
    buffer (a placeholder unless a ``trace_session`` swaps in its own). Launches sharing it
    must run one at a time in stream order."""

    def __init__(self, op, build, device, trace: str | None = None):
        self.workspace, self.max_ctas = op.work.launch_args(torch.device(device))
        self.table = build(op, self.max_ctas).to(device)
        if trace is not None:  # a work item, its regions and ring waits per row of a CTA
            events = 32 * -(-self.table.shape[0] // self.max_ctas) + 64
            op = op.with_settings(trace_backend=trace, trace_events_per_role=events)
        self.op = op
        self.scratch = torch.zeros(op.scratch_size, device=device, dtype=torch.int64)
        self.state = torch.zeros(2, device=device, dtype=torch.int32)
        self.trace_buffer = torch.zeros(1, device=device, dtype=torch.int64)


def device_position(positions: torch.Tensor) -> torch.Tensor:
    """The one-element int32 device position the kernels read (a view when already int32)."""
    return positions.reshape(1).to(torch.int32)


class InklingMlpMegakernel:
    """Launch state of the MLP-half kernels for ``config`` (without and with the message gate)
    and bf16 router copies of ``layers``. Launches of one runner must run one at a time in
    stream order (each kind's launches share a scratch and an epoch). ``trace``: as
    ``InklingAttnMegakernel``'s (into ``launches[messages].trace_buffer``)."""

    def __init__(
        self,
        config: InklingConfig,
        layers: tuple[LayerWeights, ...],
        stages: int | None = None,
        device="cuda",
        trace: str | None = None,
    ):
        self.config = config
        self.launches = {
            messages: Launch(mlp_kernel_for(config, messages, stages), build_table, device, trace)
            for messages in (False, True)
        }
        self.op = self.launches[False].op
        self.routers = {id(w): bf16_router(w.router) for w in layers}
        self.layers = layers  # keeps the ids of ``routers`` valid
        self.no_gates = GateArgs.placeholder(config.hidden_size, device)

    def __call__(
        self,
        w: LayerWeights,
        residual: torch.Tensor,
        blocks: list[torch.Tensor],
        ring: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor | None = None,
        gates: GateArgs | None = None,
    ) -> torch.Tensor:
        """``ref.mlp_block`` of one token: fp32 ``residual`` and bf16 ``blocks`` (1 to
        MAX_BLOCKS) of ``hidden`` elements, the visit's ``(RING, hidden)`` convolution ring
        (written at the position), the one-element device ``positions`` and, in core passes
        after the first, the layer's ``gates`` (the MLP message gate reads message 3). Returns
        the bf16 ``(hidden,)`` output (into ``out`` if given)."""
        hidden = self.config.hidden_size
        if not 1 <= len(blocks) <= MAX_BLOCKS:
            raise ValueError(f"need 1 to {MAX_BLOCKS} blocks, got {len(blocks)}")
        router = self.routers.get(id(w))
        if router is None:
            raise ValueError("w must be one of the layers this runner was built with")
        launch = self.launches[gates is not None]
        states = [b.reshape(hidden) for b in blocks]
        states += [states[0]] * (MAX_BLOCKS - len(states))  # unread
        if out is None:
            out = torch.empty(hidden, device=residual.device, dtype=torch.bfloat16)
        compile_mlp_kernel(launch.op)(
            router,
            w.w13,
            w.w2,
            w.router_bias,
            w.mlp_query[None],
            w.post_norm[None],
            w.mlp_conv,
            *(gates or self.no_gates),
            residual.reshape(hidden),
            *states,
            len(blocks),
            ring,
            device_position(positions),
            out,
            launch.trace_buffer,
            launch.scratch,
            launch.state,
            launch.table,
            launch.workspace,
            launch.max_ctas,
        )
        return out

    def routed_experts(self, w: LayerWeights, messages: bool = False) -> torch.Tensor:
        """The ``(experts_per_token,)`` experts the last launch (with or without the message
        gate) routed to, ascending: the top-k of its published router logits' scores plus
        ``w``'s correction bias, recomputed here."""
        launch = self.launches[messages]
        offset = launch.op.scratch_offsets["logits"]
        words = launch.scratch[offset : offset + launch.op.experts]
        logits = words.view(torch.int32)[0::2].view(torch.float32)
        keys = torch.sigmoid(logits) + w.router_bias
        return keys.topk(launch.op.top_k).indices.sort().values

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
    """A layer's attention arguments in the kernel's form: per-query block-residual queries,
    per-stream norm weights (one query in sliding layers, three in full ones; three streams in
    full layers and in launches with message gates, where sliding layers repeat their input
    norm), the dynamic tap projections (placeholders in sliding layers, which read none), the
    k/v taps and the relative-bias projection transposed (a row per distance)."""

    query: torch.Tensor  # (queries, hidden) fp32; zeros (unread) at layer 0
    norms: torch.Tensor  # (queries, hidden) bf16
    gated_norms: torch.Tensor  # (3, hidden) bf16
    k_dynamic: torch.Tensor  # (kv_heads * kernel, hidden) bf16
    v_dynamic: torch.Tensor
    k_taps: torch.Tensor  # sliding: (kv_dim, kernel) fp32; full: (kernel, kv_dim) bf16
    v_taps: torch.Tensor
    rel_proj: torch.Tensor  # (rel_extent, d_rel) bf16

    @classmethod
    def of(cls, config: InklingConfig, w: LayerWeights) -> "AttnLayer":
        if w.q_norm is not None:  # a full layer
            norms = torch.stack((w.q_norm, w.k_norm, w.v_norm))
            return cls(
                query=w.attn_query.contiguous(),
                norms=norms,
                gated_norms=norms,
                k_dynamic=w.k_dynamic,
                v_dynamic=w.v_dynamic,
                k_taps=w.k_static.contiguous(),
                v_taps=w.v_static.contiguous(),
                rel_proj=w.rel_proj.T.contiguous(),
            )
        query = w.attn_query
        if query is None:
            query = torch.zeros(1, config.hidden_size, device=w.wq.device)
        placeholder = w.wk[: config.num_kv_heads * config.conv_kernel]
        return cls(
            query=query.contiguous(),
            norms=w.input_norm[None].contiguous(),
            gated_norms=w.input_norm.expand(3, -1).contiguous(),
            k_dynamic=placeholder,
            v_dynamic=placeholder,
            k_taps=w.k_conv.contiguous(),
            v_taps=w.v_conv.contiguous(),
            rel_proj=w.rel_proj.T.contiguous(),
        )


class EmbedArgs(NamedTuple):
    """The token's embedding, read by layer 0's attention launch."""

    table: torch.Tensor  # (vocab rows, hidden) bf16
    norm: torch.Tensor  # (hidden,) bf16
    token: torch.Tensor  # (1,) int64


class InklingAttnMegakernel:
    """Launch state of the attention-half kernels for ``config`` (per layer kind: sliding or
    full, without or with message gates) and the kernel-form arguments of ``layers``. Launches
    of one runner must run one at a time in stream order (each kind's launches share a scratch
    and an epoch). ``splits``: KV splits per q head of (sliding, full) layers. ``trace``:
    compile with ranges for the work items and ``trace_region``s, recorded by ``"globaltimer"``
    (into ``launches[full, messages].trace_buffer``) or ``"iket"``."""

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
            (full, messages): Launch(
                attn_kernel_for(config, full, messages, splits[full], stages),
                build_attn_table,
                device,
                trace,
            )
            for full in (False, True)
            for messages in (False, True)
        }
        self.args = {id(w): AttnLayer.of(config, w) for w in layers}
        self.layers = layers  # keeps the ids of ``args`` valid
        hidden = config.hidden_size
        self.no_gates = GateArgs.placeholder(hidden, device)
        self.no_embed = EmbedArgs(
            torch.zeros(1, hidden, device=device, dtype=torch.bfloat16),
            torch.zeros(hidden, device=device, dtype=torch.bfloat16),
            torch.zeros(1, device=device, dtype=torch.int64),
        )

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
        gates: GateArgs | None = None,
        embed: EmbedArgs | None = None,
    ) -> torch.Tensor:
        """``ref.attention_block`` of one token (``start=None``): bf16 ``h`` and ``blocks`` (0
        to MAX_BLOCKS) of ``hidden`` elements, ``visit``'s caches (written at the position) and
        the one-element device ``positions``. In core passes after the first, ``gates`` holds
        the layer's message gates (messages 0-2 gate q, k, v). With ``embed`` (layer 0 only),
        the launch computes ``h`` itself, ``ref.hidden_states``' normalized embedding of the
        token, and stores it into ``h``. Returns the fp32 ``(hidden,)`` post-attention residual
        (into ``out`` if given)."""
        config, hidden = self.config, self.config.hidden_size
        if len(blocks) > MAX_BLOCKS:
            raise ValueError(f"need at most {MAX_BLOCKS} blocks, got {len(blocks)}")
        if cache.max_len % BLOCK:
            raise ValueError(f"the cache capacity must be a multiple of {BLOCK}")
        args = self.args.get(id(w))
        if args is None:
            raise ValueError("w must be one of the layers this runner was built with")
        if embed is not None and (w.attn_query is not None or blocks):
            raise ValueError("only layer 0 (no query, no blocks) reads the embedding")
        if gates is not None and w.attn_query is None:
            raise ValueError("message gates need a layer with a block-residual query")
        launch = self.launches[config.is_full(layer), gates is not None]
        h = h.reshape(hidden)
        states = [b.reshape(hidden) for b in blocks]
        states += [h] * (MAX_BLOCKS - len(states))  # unread
        if out is None:
            out = torch.empty(hidden, device=h.device, dtype=torch.float32)
        table, norm, token = embed or self.no_embed
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
            args.norms if gates is None else args.gated_norms,
            w.q_scale,
            w.rel_scale,
            args.rel_proj,
            args.k_taps,
            args.v_taps,
            w.attn_conv,
            *(gates or self.no_gates),
            table,
            norm,
            h,
            *states,
            len(blocks),
            int(w.attn_query is not None),
            int(layer % config.block_layers == 0),
            int(embed is not None),
            cache.conv_k[visit],
            cache.conv_v[visit],
            cache.conv_attn[visit],
            token.reshape(1),
            device_position(positions),
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


class InklingMsgMegakernel:
    """Launch state of the message kernel over ``gates``' core layers: each call fills
    ``gates.messages`` and ``gates.mhalf`` for the next pass."""

    def __init__(self, config: InklingConfig, gates: MessageGates, device="cuda"):
        op = InklingMsgKernel.get_or_build(hidden=config.hidden_size, rms_eps=config.rms_eps)
        tiles = gates.queries.shape[0] * GATE_TILES

        def build(op, ctas):
            return table_of(deal([[MSG, t, 0] for t in range(tiles)], ctas))

        self.launch = Launch(op, build, device)
        self.gates = gates

    def __call__(self, block: torch.Tensor, partial: torch.Tensor):
        """The messages of the previous pass's bf16 ``block`` (its in12) and ``partial``
        (its out17)."""
        g, launch = self.gates, self.launch
        compile_msg_kernel(launch.op)(
            g.down,
            g.queries,
            block.reshape(-1),
            partial.reshape(-1),
            g.messages,
            g.mhalf,
            launch.table,
            launch.workspace,
            launch.max_ctas,
        )


class InklingHeadMegakernel:
    """Launch state of the head kernel (final block residual, norm, LM head, softcap) for
    ``weights``."""

    def __init__(self, weights: InklingWeights, stages: int | None = None, device="cuda"):
        config = weights.config
        op = InklingHeadKernel.get_or_build(
            hidden=config.hidden_size,
            vocab=config.vocab_size,
            rms_eps=config.rms_eps,
            softcap=config.logit_softcap,
            stages=stages,
        )

        def build(op, ctas):
            return table_of(deal([[LMHEAD, t, 0] for t in range(op.row_tiles)], ctas))

        self.launch = Launch(op, build, device)
        self.weights = weights
        self.query = weights.final_query[None].contiguous()
        self.norm = weights.final_norm[None].contiguous()
        self.logits = torch.empty(config.vocab_size, device=device)

    def __call__(self, blocks: list[torch.Tensor], h: torch.Tensor) -> torch.Tensor:
        """``ref.logits_of`` of ``ref.hidden_states``' last block residual and norm over bf16
        ``blocks`` (1 to MAX_BLOCKS) and the last visit's output ``h``: fp32 ``(vocab,)``
        logits (a buffer the next call overwrites)."""
        hidden = self.weights.config.hidden_size
        if not 1 <= len(blocks) <= MAX_BLOCKS:
            raise ValueError(f"need 1 to {MAX_BLOCKS} blocks, got {len(blocks)}")
        states = [b.reshape(hidden) for b in blocks]
        states += [states[0]] * (MAX_BLOCKS - len(states))  # unread
        launch = self.launch
        compile_head_kernel(launch.op)(
            self.weights.lm_head,
            self.query,
            self.norm,
            h.reshape(hidden),
            *states,
            len(blocks),
            self.logits,
            launch.table,
            launch.workspace,
            launch.max_ctas,
        )
        return self.logits


class InklingDecoder:
    """The batch-one decode step of ``weights`` as megakernel launches only: ``ref.hidden_states``
    and ``ref.logits_of`` of one token at a device position, ``start=None``. Calls must run
    one at a time in stream order (they share the runners' state and the step buffers)."""

    def __init__(self, weights: InklingWeights, device="cuda"):
        config = weights.config
        blocks = config.block_layers
        # A pass's messages read two states (the core's second block input and its output), and
        # the final block residual reads every block: the kernels' layout.
        if config.loop_start % blocks or config.loop_layers != 2 * blocks:
            raise ValueError("the core must start a block and span exactly two blocks")
        if config.num_layers > blocks * MAX_BLOCKS:
            raise ValueError(f"the kernels' block residuals read at most {MAX_BLOCKS} blocks")
        self.weights, self.config = weights, config
        self.gates = MessageGates(
            weights.layers[config.loop_start : config.loop_end], weights.message_queries
        )
        self.attn = InklingAttnMegakernel(config, weights.layers, device=device)
        self.mlp = InklingMlpMegakernel(config, weights.layers, device=device)
        self.msg = InklingMsgMegakernel(config, self.gates, device)
        self.head = InklingHeadMegakernel(weights, device=device)
        hidden = config.hidden_size
        # [0]: the token's normalized embedding; [v + 1]: visit v's output.
        self.states = torch.zeros(
            config.num_visits + 1, hidden, device=device, dtype=torch.bfloat16
        )
        self.residual = torch.zeros(hidden, device=device)

    def __call__(
        self, token: torch.Tensor, pos: torch.Tensor, cache: InklingCache
    ) -> torch.Tensor:
        """One decode step of the one-element int64 ``token`` at the one-element device
        position ``pos`` (int32 avoids a cast); writes the caches there and returns fp32 logits
        ``(vocab,)`` (a buffer the next step overwrites). Syncs nothing."""
        config, weights = self.config, self.weights
        positions = device_position(pos)
        embed = EmbedArgs(weights.embed, weights.embed_norm, token.reshape(1))
        h = self.states[0]
        blocks: list[torch.Tensor] = []
        pre_loop = None  # the prelude's output and blocks, where every core pass restarts
        for visit, layer, p in config.visits():
            gates = None
            if config.is_loop(layer):
                core = layer - config.loop_start
                if core == 0 and p == 0:
                    pre_loop = (h, blocks)
                elif core == 0:
                    # The finished pass's messages: its core's second block input and output.
                    (block,) = blocks[len(pre_loop[1]) + 1 :]
                    self.msg(block, h)
                    h, blocks = pre_loop
                if p > 0:
                    gates = self.gates.layer(core)
            w = weights.layers[layer]
            self.attn(
                w, layer, visit, h, blocks, cache, positions, self.residual, gates,
                embed if visit == 0 else None,
            )  # fmt: skip
            if layer % config.block_layers == 0:  # h completes a block
                blocks = blocks + [h]
            out = self.states[visit + 1]
            self.mlp(w, self.residual, blocks, cache.conv_mlp[visit], positions, out, gates)
            h = out
        return self.head(blocks, h)


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
