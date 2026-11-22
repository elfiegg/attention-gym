"""Device code shared by the Inkling 2 Mini megakernel launches (``visit_kernel`` with its
halves ``attn_half`` and ``mlp_half``, ``msg_kernel``, ``head_kernel``): sizes, tagged-word
polling, CTA reductions, TMA plumbing, the block-residual prologue and the message-gate work
items, and the launches whose items are row tiles of one matrix (``MatrixRowsKernel``).

Message gates (core passes after the first, ``ref.message_gate``) run inside the visit launch
as two kinds of work items before the rows of the half whose input they gate (gates 0-2: the
attention's q, k and v streams; gate 3, ``MLP_GATE``: the MLP's input)::

    MSGDOWN  6 per gate  32 rows of the gate's down projection x a third of the x half of its
                         input (bf16 rms(x), from the half's prologue) -> fp32 partials (tagged)
    GATEMIX  36 per gate 64 elements: h = bf16(partials + message half), then
                         (1 - tanh(F h)) x + tanh(G h) m            -> fp32 stream (tagged)

The message half of each down projection (its columns on bf16 rms(m)) and the messages ``m``
themselves depend only on the previous pass, so ``msg_kernel`` computes them once per pass for
every core layer. The CTAs then gather the gated streams and RMS-normalize them into the GEMV
inputs (``gather``) before their first own row of the half.

Sized so that a half's message items (126 in attention, 42 in the MLP) fit one per CTA:
every GATEMIX item then waits for the last MSGDOWN item, not for another GATEMIX item of its CTA.
On GB300 at Inkling 2 Mini's shapes the gather of a sliding attention half ends ~6.5 us after
the prologue (whole-K MSGDOWN tiles and 32-row GATEMIX items, two on some CTAs: ~9 us).
"""

import math
import struct
from dataclasses import dataclass

import cutlass
from cutlass import BFloat16, Float32, Int32, Int64, cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import dsl_user_op

from attn_gym.warpdrive import (
    NamedBarrier,
    Ring,
    SmemPlan,
    StageView,
    StaticPersistent,
    Tma,
    Warps,
    WarpSpecialized,
    WorkStream,
)
from attn_gym.warpdrive.cache import make_fake_compact_tensor
from attn_gym.warpdrive.examples.qwen_megakernel.core import (
    COMPUTE_THREADS,
    COMPUTE_WARPS,
    EVICT_FIRST,
    MAX_STAGES,
    ROWS,
    ROWS_PER_WARP,
    SMEM_BYTES,
    SMEM_HEADROOM,
    STAGE_BYTES,
    TILE_K,
    CudaCoreGemv,
    InstructionTable,
    bf16_round,
    group_sum,
    load_words,
    publish,
    unpack_tagged,
)

WORDS = 4  # table words: op, a, b, table row
MAX_BLOCKS = 4  # completed blocks of a block residual; the partial block is the fifth state
STATES = MAX_BLOCKS + 1
RING = 16  # ref.CONV_RING: positions of the convolution input rings
REDUCTIONS = 3  # CTA reduction slots: state norms, query dots, the inputs' norms
RED_VALUES = 16  # values per CTA reduction
MSGDOWN, GATEMIX = 6, 7  # message-gate ops, after the visit's own ops 0-5
GATES = 4  # message gates per core layer: query, key, value, MLP
MLP_GATE = 3
GATE_RANK = 64  # rows of a gate's down projection: 8 lanes x 8 values in GATEMIX
GATE_TILES = GATE_RANK // ROWS  # MSGDOWN row tiles per gate
GATE_SPLITS = 3  # MSGDOWN split-K: k tiles of the x half per item = hidden / 256 / GATE_SPLITS
MIX_STAGES = 2  # GATEMIX: 32-row stages of [F | G] per item
# MIX: the prologue's state scale factors (STATES), then each query's state weights (STATES).
MIX_FLOATS = STATES * 4
# Sleep between failed polls of a wait that spans other CTAs' work (fetch_uniform's backoff).
BACKOFF_NS = 100


def fp32(x: float) -> float:
    """``x`` rounded to fp32, as a Python float."""
    return struct.unpack("f", struct.pack("f", x))[0]


def fp32_reciprocal(x: float) -> float:
    """``1 / x`` in fp32: PyTorch divides a CUDA tensor by a Python scalar as a multiplication by
    this reciprocal."""
    return fp32(1.0 / fp32(x))


def ring_stages(fixed: int, stages: int | None) -> int:
    """Depth of a weight ring of ``STAGE_BYTES`` stages beside ``fixed`` bytes of other shared
    memory: ``stages``, by default as many as fit up to ``MAX_STAGES``."""
    fit = (SMEM_BYTES - fixed - SMEM_HEADROOM) // STAGE_BYTES
    stages = min(MAX_STAGES, fit) if stages is None else stages
    if not 2 <= stages <= fit:
        raise ValueError(f"stages must be in [2, {fit}], got {stages}")
    return stages


def scratch_offsets(shapes: dict[str, int]) -> tuple[dict[str, int], int]:
    """``(offsets, size)`` of tagged int64 buffers of ``shapes`` words, each 128-byte aligned."""
    offsets, offset = {}, 0
    for name, size in shapes.items():
        offsets[name] = offset
        offset += (size + 15) // 16 * 16
    return offsets, offset


@dsl_user_op
def nanosleep(ns: int, *, loc=None, ip=None):
    """Suspend the thread for about ``ns`` nanoseconds (PTX ``nanosleep``; the hardware picks a
    duration in [0, 2 * ns])."""
    llvm.inline_asm(
        None,
        [],
        f"nanosleep.u32 {ns};",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def fetch_uniform(segments, out, epoch, backoff: cutlass.Constexpr = 0):
    """``fetch`` polled until every lane of the warp has all its words, so the lanes leave the
    loop together: lanes leaving a divergent poll loop one by one and then waiting at a shuffle
    for the rest measured a fixed ~17k-cycle stall. Every lane of the warp must call it. With
    ``backoff``, a failed poll sleeps about that many nanoseconds: long waits then issue few
    L2 requests while other CTAs' work runs."""
    ready = Int32(0)
    while ready == 0:
        mine = Int32(1)
        start = 0
        for s in cutlass.range_constexpr(len(segments)):
            buf, index, count = segments[s]
            words = load_words(buf, index, count)
            for e in cutlass.range_constexpr(count):
                value, tag = unpack_tagged(words[e])
                out[start + e] = value
                mine = mine & Int32(tag == epoch)
            start += count
        ready = Int32(cute.arch.vote_all_sync(mine != 0))
        if cutlass.const_expr(backoff > 0):
            if ready == 0:
                nanosleep(backoff)


@cute.jit
def load_tiles(w, atom, tiles, row, k0, count: cutlass.Constexpr, buffer: cutlass.Constexpr):
    """``count`` stages of tiles ``(row, k0 + k)`` into the ring's ``buffer`` view; weights and
    KV rows are read once per launch, hence evict-first in L2."""
    for k in cutlass.range_constexpr(count):
        with w.produce() as h:
            cute.copy(
                atom,
                tiles[None, row, k0 + k],
                h.view(buffer),
                tma_bar_ptr=h.barrier,
                cache_policy=Int64(EVICT_FIRST),
            )
    return w


def tma_atom(src, box: tuple[int, ...]) -> tuple:
    """TMA ``(atom, tensor)`` loading ``box`` tiles of ``src`` into row-major SMEM."""
    stride = tuple(math.prod(box[i + 1 :]) for i in range(len(box)))
    smem = cute.make_layout(box, stride=stride)
    return cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), src, smem, box)


def tma_partition(atom, smem, gtensor, box: tuple[int, ...]) -> tuple:
    """TMA ``atom``'s ``(SMEM, GMEM)`` partitions: the stages of ``smem`` (box modes first),
    and ``gtensor``'s ``box`` tiles, ``(TMA, tile coordinates...)``."""
    rank = len(box)
    tiles = cute.local_tile(gtensor, box, (None,) * rank)
    return cpasync.tma_partition(
        atom,
        0,
        cute.make_layout(1),
        cute.group_modes(smem, 0, rank),
        cute.group_modes(tiles, 0, rank),
    )


def fake_gate_args(hidden: int) -> tuple:
    """Fake compile arguments of a core layer's message gates (``gate_atoms``, ``GateParams``):
    the down projections ``(GATES * GATE_RANK, 2 * hidden)`` bf16, the fp32 ``(GATES, hidden,
    2 * GATE_RANK)`` ``[F | G]`` rows viewed as bf16 ``(GATES * hidden, 4 * GATE_RANK)``, the
    message halves and the messages."""
    return (
        make_fake_compact_tensor(BFloat16, (GATES * GATE_RANK, 2 * hidden)),
        make_fake_compact_tensor(BFloat16, (GATES * hidden, 4 * GATE_RANK)),
        make_fake_compact_tensor(Float32, (GATES * GATE_RANK,), assumed_align=4),
        make_fake_compact_tensor(Float32, (GATES, hidden), assumed_align=4),
    )


def gate_atoms(down, fg, hidden: int) -> list:
    """TMA ``(atom, tensor)`` pairs of 32 x 256 bf16 tiles of the down projections' x halves
    (their first ``hidden`` columns) and of the ``[F | G]`` rows (one tile: 32 rows)."""
    x_half = cute.make_tensor(
        down.iterator, cute.make_layout((GATES * GATE_RANK, hidden), stride=(2 * hidden, 1))
    )
    return [tma_atom(src, (ROWS, TILE_K)) for src in (x_half, fg)]


@dataclass(frozen=True)
class Inputs:
    """The params ``CudaCoreGemv.tile`` reads: its input rows ``xs``."""

    xs: cute.Tensor


@dataclass(frozen=True)
class GateParams:
    """A launch's message-gate operands for one core layer (``InklingKernel.messages``)."""

    tma: tuple  # TMA atoms: the down projections' x halves, the [F | G] rows
    gmem: tuple  # their tile partitions: (TMA, row tiles, k tiles)
    h_part: cute.Tensor  # tagged int64 (GATE_SPLITS, GATES, GATE_RANK) x-half partial dots
    gated: cute.Tensor  # tagged int64 (GATES * hidden,) gated streams
    mhalf: cute.Tensor  # (GATES * GATE_RANK,) fp32 message-half dots (msg_kernel)
    messages: cute.Tensor  # (GATES, hidden) fp32 messages (msg_kernel)


class InklingKernel(WarpSpecialized):
    """Device helpers of the Inkling launches. Subclasses set ``hidden``, ``rms_eps``,
    ``stages`` (``ring_stages``), ``csync`` (the compute warps' barrier) and ``engine``, and
    define ``scratch_shapes`` (tagged words by buffer) when they publish any; launches with
    message gates call ``init_messages`` and implement ``stream_value``. Their params carry
    ``xs`` (GEMV inputs), ``red`` (``(REDUCTIONS, COMPUTE_WARPS, RED_VALUES)`` fp32), ``mix``
    (``(MIX_FLOATS,)`` fp32), ``blocks`` and ``gate`` (``GateParams`` or None)."""

    messages = False

    def init_messages(self, messages: bool, gate_query: tuple[int, ...]):
        """All GATES message gates when ``messages``; ``gate_query[s]`` is the prologue query
        whose block residual gate s's x is (the xs row and MIX weights its prologue wrote)."""
        if messages and (
            (self.hidden // TILE_K) % GATE_SPLITS or self.hidden % (ROWS * MIX_STAGES)
        ):
            raise ValueError(
                f"message gates need hidden / {TILE_K} divisible by {GATE_SPLITS} and hidden "
                f"divisible by {ROWS * MIX_STAGES}"
            )
        if len(gate_query) != GATES:
            raise ValueError(f"need a prologue query for each of the {GATES} gates")
        self.messages, self.gate_query = messages, gate_query

    def core_layouts(self, x_rows: int) -> dict:
        """SMEM layouts of the weight stages ``W``, the GEMV inputs ``X`` (``x_rows`` rows) and
        the reductions ``RED``."""
        return {
            "W": cute.make_layout((ROWS, TILE_K, self.stages), stride=(TILE_K, 1, ROWS * TILE_K)),
            "X": cute.make_layout((x_rows, self.hidden), stride=(self.hidden, 1)),
            "RED": cute.make_layout(
                (REDUCTIONS, COMPUTE_WARPS, RED_VALUES),
                stride=(COMPUTE_WARPS * RED_VALUES, RED_VALUES, 1),
            ),
        }

    @property
    def scratch_shapes(self) -> dict[str, int]:
        """Tagged int64 words of each per-launch intermediate."""
        return {}

    @property
    def scratch_offsets(self) -> dict[str, int]:
        return scratch_offsets(self.scratch_shapes)[0]

    @property
    def scratch_size(self) -> int:
        return scratch_offsets(self.scratch_shapes)[1]

    def gate_scratch_shapes(self) -> dict[str, int]:
        """Tagged words of the message items: MSGDOWN partials (split, gate, row) and the gated
        streams (gate, element); none without messages."""
        if not self.messages:
            return {}
        return {"h_part": GATE_SPLITS * GATES * GATE_RANK, "gated": GATES * self.hidden}

    def gate_rows(self, gates: range) -> list[list[int]]:
        """Table rows ``[op, a, b]`` of the message items of ``gates`` in dependency order:
        MSGDOWN (tile, split; tile ``a`` of gate ``a // GATE_TILES``), then GATEMIX (tile,
        gate); none without messages."""
        if not self.messages:
            return []
        tiles = range(gates.start * GATE_TILES, gates.stop * GATE_TILES)
        rows = [[MSGDOWN, t, k] for t in tiles for k in range(GATE_SPLITS)]
        mix_tiles = self.hidden // (ROWS * MIX_STAGES)
        return rows + [[GATEMIX, t, s] for s in gates for t in range(mix_tiles)]

    def stream_value(self, p, s, i):
        """Element ``i`` of gate ``s``'s input stream: its prologue's block residual of query
        ``gate_input(s)``, recomputed in that prologue's operation order (``mixed_state``)."""
        raise NotImplementedError

    @cute.jit
    def cta_sum(self, p, vals, slot: cutlass.Constexpr, warp, lane):
        """Sums of each of ``vals`` (at most RED_VALUES per-thread values) over the compute
        threads, in every thread, through RED region ``slot``. Before a slot is written again,
        every thread must pass a barrier after its last read: the prologue's reductions use
        distinct slots, and later reuses follow the prologue's closing barrier or another
        ``cta_sum`` of the same item."""
        n = len(vals)
        for i in cutlass.range_constexpr(n):
            v = group_sum(vals[i], 32)
            if lane == 0:
                p.red[slot, warp, i] = v
        self.csync.sync()
        out = []
        for i in cutlass.range_constexpr(n):
            total = Float32(0.0)
            for wi in cutlass.range_constexpr(COMPUTE_WARPS):
                total = total + p.red[slot, wi, i]
            out.append(total)
        return out

    @cute.jit
    def load_states(self, p, num_blocks, partial, tid, load_partial=True):
        """``(STATES, hidden // COMPUTE_THREADS)`` fp32 registers: the first ``num_blocks``
        completed blocks (zeros past them), then ``partial`` (a ``(hidden,)`` tensor; zeros if
        None or ``load_partial`` is false: the caller fills it). Thread t holds elements
        ``r * 256 + t``."""
        per = self.hidden // COMPUTE_THREADS
        x = cute.make_rmem_tensor((STATES, per), Float32)
        x.fill(0.0)
        for n in cutlass.range_constexpr(MAX_BLOCKS):
            if n < num_blocks:
                for r in cutlass.range_constexpr(per):
                    x[n, r] = Float32(p.blocks[n][r * COMPUTE_THREADS + tid])
        if cutlass.const_expr(partial is not None):
            if load_partial:
                for r in cutlass.range_constexpr(per):
                    x[MAX_BLOCKS, r] = Float32(partial[r * COMPUTE_THREADS + tid])
        return x

    @cute.jit
    def normalize_states(
        self, p, x, num_blocks, query, norms, nq: cutlass.Constexpr, queries, tid, warp, lane
    ):
        """``xs[q]`` for each of the ``nq`` queries: the block residual ``y_q`` of fp32 ``query[q]``
        over ``load_states``'s ``x`` (``ref.block_residual`` over ``num_blocks`` blocks;
        ``queries == 0``: the partial block itself), then ``bf16(norms[q] * rms_norm(y_q))``
        (the half's input norm) or, in a launch with message gates, ``bf16(rms_norm(y_q))``
        (the x half of the gates' down inputs); the state scales and weights go to MIX for
        ``mixed_state``."""
        h = self.hidden
        per = h // COMPUTE_THREADS
        weighted = not self.messages
        # Every load first, so their latencies overlap: the queries and norm weights.
        qv = cute.make_rmem_tensor((nq, per), Float32)
        weights = cute.make_rmem_tensor((nq, per), BFloat16)
        for q in cutlass.range_constexpr(nq):
            for r in cutlass.range_constexpr(per):
                qv[q, r] = query[q, r * COMPUTE_THREADS + tid]
                if cutlass.const_expr(weighted):
                    weights[q, r] = norms[q, r * COMPUTE_THREADS + tid]
        inv_h = fp32_reciprocal(h)
        y = cute.make_rmem_tensor((nq, per), Float32)
        if queries == 0:
            for r in cutlass.range_constexpr(per):
                y[0, r] = x[MAX_BLOCKS, r]
        else:
            squares = []
            for n in cutlass.range_constexpr(STATES):
                ss = Float32(0.0)
                for r in cutlass.range_constexpr(per):
                    ss = ss + x[n, r] * x[n, r]
                squares.append(ss)
            squares = self.cta_sum(p, squares, 0, warp, lane)
            # Unweighted RMSNorm of every state (eps 1e-6, as ref.block_residual), then each
            # query's scores against them; empty block slots are masked out of the softmax.
            rinvs = []
            for n in cutlass.range_constexpr(STATES):
                rinvs.append(cute.math.rsqrt(squares[n] * inv_h + Float32(1e-6)))
                for r in cutlass.range_constexpr(per):
                    x[n, r] = x[n, r] * rinvs[n]
            dots = []
            for q in cutlass.range_constexpr(nq):
                for n in cutlass.range_constexpr(STATES):
                    dot = Float32(0.0)
                    for r in cutlass.range_constexpr(per):
                        dot = dot + qv[q, r] * x[n, r]
                    dots.append(dot)
            dots = self.cta_sum(p, dots, 1, warp, lane)
            scale = fp32(h**-0.5)
            for q in cutlass.range_constexpr(nq):
                scores = []
                top = Float32(-Float32.inf)
                for n in cutlass.range_constexpr(STATES):
                    s = dots[q * STATES + n] * scale
                    if cutlass.const_expr(n < MAX_BLOCKS):
                        s = cutlass.select_(n < num_blocks, s, -Float32.inf)
                    scores.append(s)
                    top = cutlass.max(top, s)
                total = Float32(0.0)
                exps = []
                for n in cutlass.range_constexpr(STATES):
                    exps.append(cute.math.exp(scores[n] - top))
                    total = total + exps[n]
                probs = [exps[n] / total for n in range(STATES)]
                for r in cutlass.range_constexpr(per):
                    acc = Float32(0.0)
                    for n in cutlass.range_constexpr(STATES):
                        acc = acc + probs[n] * x[n, r]
                    y[q, r] = acc
                if cutlass.const_expr(self.messages):
                    if tid == 0:
                        for n in cutlass.range_constexpr(STATES):
                            p.mix[STATES * (q + 1) + n] = probs[n]
            if cutlass.const_expr(self.messages):
                if tid == 0:
                    for n in cutlass.range_constexpr(STATES):
                        p.mix[n] = rinvs[n]
        squares = []
        for q in cutlass.range_constexpr(nq):
            ss = Float32(0.0)
            for r in cutlass.range_constexpr(per):
                ss = ss + y[q, r] * y[q, r]
            squares.append(ss)
        squares = self.cta_sum(p, squares, 2, warp, lane)
        for q in cutlass.range_constexpr(nq):
            rinv = cute.math.rsqrt(squares[q] * inv_h + Float32(self.rms_eps))
            for r in cutlass.range_constexpr(per):
                k = r * COMPUTE_THREADS + tid
                if cutlass.const_expr(weighted):
                    p.xs[q, k] = BFloat16(Float32(weights[q, r]) * (y[q, r] * rinv))
                else:
                    p.xs[q, k] = BFloat16(y[q, r] * rinv)
        self.csync.sync()

    @cute.jit
    def mixed_state(self, p, query, num_blocks, partial, i):
        """Element ``i`` of ``normalize_states``' block residual of ``query`` over ``num_blocks``
        blocks and the partial block's element ``partial``, recomputed from the states and MIX
        in its operation order (the same fp32 value)."""
        weights = STATES * (query + 1)
        acc = Float32(0.0)
        for n in cutlass.range_constexpr(MAX_BLOCKS):
            if n < num_blocks:
                acc = acc + p.mix[weights + n] * (Float32(p.blocks[n][i]) * p.mix[n])
        return acc + p.mix[weights + MAX_BLOCKS] * (partial * p.mix[MAX_BLOCKS])

    @cute.jit
    def advance_epoch(self, p, tid, epoch):
        """Count this CTA finished; the last one advances the launch epoch for the runner's next
        launch, which starts after this one completes. Every CTA read the epoch before."""
        if tid == 0:
            ctas = cute.arch.grid_dim()[0]
            finished = cute.arch.atomic_add(
                p.state.iterator + 1, Int32(1), sem="acq_rel", scope="gpu"
            )
            if finished == ctas - 1:
                cute.arch.atomic_exch(p.state.iterator + 1, Int32(0), scope="gpu")
                p.state[0] = epoch

    # ------------------------------------------------------------ message gates

    @cute.jit
    def gate_input(self, s):
        """The prologue query (xs row and MIX weights) of gate ``s``."""
        query = Int32(self.gate_query[0])
        for g in cutlass.range_constexpr(1, GATES):
            query = cutlass.select_(s == g, Int32(self.gate_query[g]), query)
        return query

    @cute.jit
    def gate_params(self, sW, tma, gtensors, tagged: dict, tensors: dict):
        """The kernel's ``GateParams`` (None without messages): the gates' TMA partitions over
        the weight stages ``sW``, their tagged buffers (popped from ``tagged``) and the
        ``mhalf``/``messages`` argument tensors."""
        if cutlass.const_expr(not self.messages):
            return None
        parts = [tma_partition(atom, sW, g, (ROWS, TILE_K)) for atom, g in zip(tma, gtensors)]
        return GateParams(
            tma=tuple(tma),
            gmem=tuple(part[1] for part in parts),
            h_part=tagged.pop("h_part"),
            gated=tagged.pop("gated"),
            mhalf=tensors["mhalf"],
            messages=tensors["messages"],
        )

    @cute.jit
    def load_gate_item(self, p, w, op, a, b):
        """The load warp's stages of a MSGDOWN (tile ``a``, split ``b``) or GATEMIX (tile ``a``
        of gate ``b``) item."""
        g = p.gate
        if op == MSGDOWN:
            per = self.hidden // TILE_K // GATE_SPLITS
            w = load_tiles(w, g.tma[0], g.gmem[0], a, b * per, per, "W")
        else:
            for stage in cutlass.range_constexpr(MIX_STAGES):
                row = b * (self.hidden // ROWS) + a * MIX_STAGES + stage
                w = load_tiles(w, g.tma[1], g.gmem[1], row, 0, 1, "W")
        return w

    @cute.jit
    def msgdown(self, p, w, a, split, epoch, warp, lane):
        """MSGDOWN tile ``a``, split ``split``: 32 rows of gate ``a // GATE_TILES``'s down
        projection against a third of the x half of its input (``xs`` row of its query),
        published unrounded."""
        per = self.hidden // TILE_K // GATE_SPLITS
        xbuf = self.gate_input(a // GATE_TILES)
        value = Float32(0.0)
        row = Int32(0)
        # One branch per split: the GEMV input starts at a constant (aligned) column.
        for k in cutlass.range_constexpr(GATE_SPLITS):
            if split == k:
                xs = cute.make_tensor(p.xs.iterator + k * per * TILE_K, p.xs.layout)
                vals, tile_row, chans = self.engine.tile(
                    Inputs(xs), (w,), (), xbuf, per, warp, lane
                )
                value, row, w = vals[0], tile_row, chans[0]
        if self.engine.owns_row(warp, lane):
            index = (split * GATES * GATE_TILES + a) * ROWS + row
            publish(p.gate.h_part, index, value, epoch)
        return w

    @cute.jit
    def gatemix(self, p, w, tile, s, epoch, warp, lane):
        """GATEMIX tile ``tile`` of gate ``s``: elements i of its MIX_STAGES * 32 rows,
        ``ref.message_gate``'s ``(1 - tanh(F_i . h)) x_i + tanh(G_i . h) m_i`` with ``h =
        bf16(x-half partials in split order + message half)`` of the gate's down projection.
        In each stage (32 rows of the fp32 ``[F | G]``) warp w takes rows 4w .. 4w + 3, lane
        group lane // 8 a row, lane j = lane % 8 its columns 8j .. 8j + 7 of F and of G. The
        elements' x and m load before the poll for h, so their latency overlaps it."""
        g = p.gate
        j = lane % 8
        row = ROWS_PER_WARP * warp + lane // 8
        xm = cute.make_rmem_tensor((MIX_STAGES, 2), Float32)
        for stage in cutlass.range_constexpr(MIX_STAGES):
            i = (tile * MIX_STAGES + stage) * ROWS + row
            xm[stage, 0] = self.stream_value(p, s, i)
            xm[stage, 1] = g.messages[s, i]
        hv = cute.make_rmem_tensor((GATE_SPLITS * 8,), Float32)
        stride = GATES * GATE_RANK
        segments = [(g.h_part, k * stride + s * GATE_RANK + 8 * j, 8) for k in range(GATE_SPLITS)]
        fetch_uniform(segments, hv, epoch)
        for e in cutlass.range_constexpr(8):
            acc = hv[e]
            for k in cutlass.range_constexpr(1, GATE_SPLITS):
                acc = acc + hv[8 * k + e]
            hv[e] = bf16_round(acc + g.mhalf[s * GATE_RANK + 8 * j + e])
        for stage in cutlass.range_constexpr(MIX_STAGES):
            i = (tile * MIX_STAGES + stage) * ROWS + row
            forget = Float32(0.0)
            gain = Float32(0.0)
            with w.consume() as h:
                rows = cute.recast_tensor(h.view("W"), Float32)  # (ROWS, 2 * GATE_RANK)
                for e in cutlass.range_constexpr(8):
                    forget = forget + rows[row, 8 * j + e] * hv[e]
                    gain = gain + rows[row, GATE_RANK + 8 * j + e] * hv[e]
            forget = group_sum(forget, 8)
            gain = group_sum(gain, 8)
            if j == 0:
                x, m = xm[stage, 0], xm[stage, 1]
                out = (Float32(1.0) - cute.math.tanh(forget)) * x + cute.math.tanh(gain) * m
                publish(g.gated, s * self.hidden + i, out, epoch)
        return w

    @cute.jit
    def gather(self, p, first: cutlass.Constexpr, n: cutlass.Constexpr, norms, epoch, tid, warp,
               lane):  # fmt: skip
        """``xs[s] = bf16(norms[s] * rms_norm(gated stream first + s))`` for each of the ``n``
        gates from ``first``: a half's GEMV inputs, once its GATEMIX items have published."""
        h = self.hidden
        per = h // COMPUTE_THREADS
        vals = cute.make_rmem_tensor((n * per,), Float32)
        segments = [
            (p.gate.gated, (first + s) * h + r * COMPUTE_THREADS + tid, 1)
            for s in range(n)
            for r in range(per)
        ]
        weights = cute.make_rmem_tensor((n, per), BFloat16)
        for s in cutlass.range_constexpr(n):
            for r in cutlass.range_constexpr(per):
                weights[s, r] = norms[s, r * COMPUTE_THREADS + tid]
        fetch_uniform(segments, vals, epoch)
        squares = []
        for s in cutlass.range_constexpr(n):
            ss = Float32(0.0)
            for r in cutlass.range_constexpr(per):
                ss = ss + vals[s * per + r] * vals[s * per + r]
            squares.append(ss)
        squares = self.cta_sum(p, squares, 2, warp, lane)
        for s in cutlass.range_constexpr(n):
            rinv = cute.math.rsqrt(squares[s] * fp32_reciprocal(h) + Float32(self.rms_eps))
            for r in cutlass.range_constexpr(per):
                k = r * COMPUTE_THREADS + tid
                p.xs[s, k] = BFloat16(Float32(weights[s, r]) * (vals[s * per + r] * rinv))
        self.csync.sync()


class MatrixRowsKernel(InklingKernel):
    """A launch whose items are 32-row tiles (table word ``a``) of one bf16 ``(rows, hidden)``
    matrix against the GEMV input ``xs[0]``: ``msg_kernel`` and ``head_kernel``. Items depend
    on nothing outside the CTA, so the load warp streams every item's stages at once
    (``load_matrix_rows``). Their params carry ``tma``/``gmem`` (the matrix's atom and tile
    partition), ``sW``, ``xs`` (``(1, hidden)``) and ``red`` (``matrix_params``)."""

    def init_matrix_rows(self, hidden: int, rms_eps: float, stages: int | None):
        """``stages``: weight ring depth in 16 KB stages."""
        if hidden % COMPUTE_THREADS or hidden % TILE_K:
            raise ValueError(f"hidden size must be a multiple of {COMPUTE_THREADS} and {TILE_K}")
        self.hidden, self.rms_eps = hidden, rms_eps
        self.hidden_k_tiles = hidden // TILE_K
        self.stages = ring_stages(2 * hidden + 4 * REDUCTIONS * COMPUTE_WARPS * RED_VALUES, stages)
        self.smem_plan = SmemPlan()
        w = self.smem_plan.buffer("W", BFloat16, align=1024)
        self.smem_plan.buffer("X", BFloat16, align=16)
        self.smem_plan.buffer("RED", Float32, align=16)
        # Compute reads stages with generic loads before TMA refills them; the layer emits the
        # fence.proxy.async on a Warps release of a Tma-filled ring.
        self.w = Ring(self.stages, Tma(STAGE_BYTES), Warps, buffer={"W": w})
        self.csync = NamedBarrier(threads=COMPUTE_THREADS)
        self.engine = CudaCoreGemv()
        self.work = WorkStream(mapping=InstructionTable(words=WORDS), schedule=StaticPersistent())

    @cute.jit
    def load_matrix_rows(self, p, w, work):
        """The load role: every item's weight stages."""
        w = w.views(W=p.sW)
        while work.valid:
            _op, tile, _b, _row = work.item
            w = load_tiles(w, p.tma[0], p.gmem[0], tile, 0, self.hidden_k_tiles, "W")
            work.advance()
        return w, work

    def matrix_params(self, atom, gtensor) -> dict:
        """The params fields of the matrix (TMA ``atom`` over ``gtensor``) and the SMEM
        buffers, after ``begin_roles(layouts=self.core_layouts(1))``."""
        smem, gmem = tma_partition(atom, self.buffer("W"), gtensor, (ROWS, TILE_K))
        return {
            "tma": (atom,),
            "gmem": (gmem,),
            "sW": StageView(smem, axis=1),
            "xs": self.buffer("X"),
            "red": self.buffer("RED"),
        }
