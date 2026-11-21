"""Device code shared by the Inkling 2 Mini megakernel launches (``attn_kernel``,
``mlp_kernel``, ``msg_kernel``, ``head_kernel``): sizes, tagged-word polling, CTA reductions,
the block-residual prologue and the message-gate work items.

Message gates (core passes after the first, ``ref.message_gate``) run inside the attention
and MLP launches as two kinds of work items before the launch's own::

    MSGDOWN  6 per gate  32 rows of the gate's down projection x a third of the x half of its
                         input (bf16 rms(x), from the prologue)     -> fp32 partials (tagged)
    GATEMIX  36 per gate 64 elements: h = bf16(partials + message half), then
                         (1 - tanh(F h)) x + tanh(G h) m            -> fp32 stream (tagged)

The message half of each down projection (its columns on bf16 rms(m)) and the messages ``m``
themselves depend only on the previous pass, so ``msg_kernel`` computes them once per pass for
every core layer. The CTAs then gather the gated streams and RMS-normalize them into the GEMV
inputs (``gather``) before their first item of the launch's own.

Sized so that a launch's message items (126 in attention, 42 in the MLP) fit one per CTA:
every GATEMIX item then waits for the last MSGDOWN item, not for another GATEMIX item of its CTA.
On GB300 at Inkling 2 Mini's shapes, 6 whole-K MSGDOWN tiles and 32-row GATEMIX items (two on
some CTAs) put the gather ~9 us after the prologue.
"""

import struct
from dataclasses import dataclass

import cutlass
from cutlass import BFloat16, Float32, Int32, Int64, cute
from cutlass.cute.nvgpu import cpasync

from attn_gym.warpdrive import WarpSpecialized
from attn_gym.warpdrive.cache import make_fake_compact_tensor
from attn_gym.warpdrive.examples.qwen_megakernel.core import (
    COMPUTE_THREADS,
    COMPUTE_WARPS,
    EVICT_FIRST,
    ROWS,
    ROWS_PER_WARP,
    TILE_K,
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
SMEM_BYTES = 232448  # SM100 per-block shared memory (227 KB)
SMEM_HEADROOM = 1024
REDUCTIONS = 3  # CTA reduction slots: state norms, query dots, the inputs' norms
RED_VALUES = 16  # values per CTA reduction
MSGDOWN, GATEMIX = 3, 4  # message-gate ops, after each launch's own ops 0-2
GATES = 4  # message gates per core layer: query, key, value, MLP
GATE_RANK = 64  # rows of a gate's down projection: 8 lanes x 8 values in GATEMIX
GATE_TILES = GATE_RANK // ROWS  # MSGDOWN row tiles per gate
GATE_SPLITS = 3  # MSGDOWN split-K: k tiles of the x half per item = hidden / 256 / GATE_SPLITS
MIX_STAGES = 2  # GATEMIX: 32-row stages of [F | G] per item
# MIX: the prologue's state scale factors (STATES), then each query's state weights (STATES).
MIX_FLOATS = STATES * 4


def fp32(x: float) -> float:
    """``x`` rounded to fp32, as a Python float."""
    return struct.unpack("f", struct.pack("f", x))[0]


def fp32_reciprocal(x: float) -> float:
    """``1 / x`` in fp32: PyTorch divides a CUDA tensor by a Python scalar as a multiplication by
    this reciprocal."""
    return fp32(1.0 / fp32(x))


def scratch_offsets(shapes: dict[str, int]) -> tuple[dict[str, int], int]:
    """``(offsets, size)`` of tagged int64 buffers of ``shapes`` words, each 128-byte aligned."""
    offsets, offset = {}, 0
    for name, size in shapes.items():
        offsets[name] = offset
        offset += (size + 15) // 16 * 16
    return offsets, offset


@cute.jit
def fetch_uniform(segments, out, epoch):
    """``fetch`` polled until every lane of the warp has all its words, so the lanes leave the
    loop together: lanes leaving a divergent poll loop one by one and then waiting at a shuffle
    for the rest measured a fixed ~17k-cycle stall. Every lane of the warp must call it."""
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
    op = cpasync.CopyBulkTensorTileG2SOp()
    smem = cute.make_layout((ROWS, TILE_K), stride=(TILE_K, 1))
    x_half = cute.make_tensor(
        down.iterator, cute.make_layout((GATES * GATE_RANK, hidden), stride=(2 * hidden, 1))
    )
    return [cpasync.make_tiled_tma_atom(op, src, smem, (ROWS, TILE_K)) for src in (x_half, fg)]


@dataclass(frozen=True)
class Inputs:
    """The params ``CudaCoreGemv.tile`` reads: its input rows ``xs``."""

    xs: cute.Tensor


@dataclass(frozen=True)
class GateParams:
    """A launch's message-gate operands for one core layer (``InklingKernel.messages``)."""

    tma: tuple  # TMA atoms: the down projections' x halves, the [F | G] rows
    gmem: tuple  # their tile partitions: (TMA, row tiles, k tiles)
    h_part: cute.Tensor  # tagged int64 (GATE_SPLITS, gates, GATE_RANK) x-half partial dots
    gated: cute.Tensor  # tagged int64 (gates * hidden,) gated streams
    mhalf: cute.Tensor  # (GATES * GATE_RANK,) fp32 message-half dots (msg_kernel)
    messages: cute.Tensor  # (GATES, hidden) fp32 messages (msg_kernel)


class InklingKernel(WarpSpecialized):
    """Device helpers of the Inkling launches. Subclasses set ``hidden``, ``rms_eps``,
    ``csync`` (the compute warps' barrier) and ``engine``; launches with message gates also set
    ``messages`` (true), ``first_gate`` and ``gate_query`` (each gate's prologue query) and
    implement ``partial``. Their params carry ``xs`` (GEMV inputs), ``red`` (``(REDUCTIONS,
    COMPUTE_WARPS, RED_VALUES)`` fp32), ``mix`` (``(MIX_FLOATS,)`` fp32), ``blocks``,
    ``num_blocks``, ``norms`` and ``gate`` (``GateParams`` or None)."""

    messages = False

    @property
    def gates(self) -> int:
        return len(self.gate_query) if self.messages else 0

    def gate_scratch_shapes(self) -> dict[str, int]:
        """Tagged words of the message items: MSGDOWN partials (split, gate, row) and the gated
        streams (gate, element); none without messages."""
        if not self.messages:
            return {}
        return {
            "h_part": GATE_SPLITS * self.gates * GATE_RANK,
            "gated": self.gates * self.hidden,
        }

    def gate_rows(self) -> list[list[int]]:
        """Table rows ``[op, a, b]`` of the message items in dependency order: MSGDOWN (tile,
        split), then GATEMIX (tile, gate); none without messages."""
        rows = [
            [MSGDOWN, t, k] for t in range(self.gates * GATE_TILES) for k in range(GATE_SPLITS)
        ]
        tiles = self.hidden // (ROWS * MIX_STAGES)
        return rows + [[GATEMIX, t, s] for s in range(self.gates) for t in range(tiles)]

    def partial(self, p):
        """The partial block of the launch's block residual (a ``(hidden,)`` tensor)."""
        raise NotImplementedError

    @cute.jit
    def cta_sum(self, p, vals, slot: cutlass.Constexpr, warp, lane):
        """Sums of each of ``vals`` (at most RED_VALUES per-thread values) over the compute
        threads, in every thread; ``slot`` picks a RED region of its own, so no barrier guards
        its reuse."""
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
    def load_states(self, p, partial, tid):
        """``(STATES, hidden // COMPUTE_THREADS)`` fp32 registers: the completed blocks (zeros
        past ``num_blocks``), then ``partial``. Thread t holds elements ``r * 256 + t``."""
        per = self.hidden // COMPUTE_THREADS
        x = cute.make_rmem_tensor((STATES, per), Float32)
        x.fill(0.0)
        for n in cutlass.range_constexpr(MAX_BLOCKS):
            if n < p.num_blocks:
                for r in cutlass.range_constexpr(per):
                    x[n, r] = Float32(p.blocks[n][r * COMPUTE_THREADS + tid])
        for r in cutlass.range_constexpr(per):
            x[MAX_BLOCKS, r] = Float32(partial[r * COMPUTE_THREADS + tid])
        return x

    @cute.jit
    def normalize_states(self, p, x, query, nq: cutlass.Constexpr, queries, tid, warp, lane):
        """``xs[q]`` for each of the ``nq`` queries: the block residual ``y_q`` of fp32 ``query[q]``
        over ``load_states``'s ``x`` (``ref.block_residual``; ``queries == 0``: the partial
        block itself), then ``bf16(norms[q] * rms_norm(y_q))`` (the launch's input norm) or, in
        a launch with message gates, ``bf16(rms_norm(y_q))`` (the x half of the gates' down
        inputs); the state scales and weights go to MIX for ``stream_value``."""
        h = self.hidden
        per = h // COMPUTE_THREADS
        nb = p.num_blocks
        weighted = not self.messages
        # Every load first, so their latencies overlap: the queries and norm weights.
        qv = cute.make_rmem_tensor((nq, per), Float32)
        norms = cute.make_rmem_tensor((nq, per), BFloat16)
        for q in cutlass.range_constexpr(nq):
            for r in cutlass.range_constexpr(per):
                qv[q, r] = query[q, r * COMPUTE_THREADS + tid]
                if cutlass.const_expr(weighted):
                    norms[q, r] = p.norms[q, r * COMPUTE_THREADS + tid]
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
                        s = cutlass.select_(n < nb, s, -Float32.inf)
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
                    p.xs[q, k] = BFloat16(Float32(norms[q, r]) * (y[q, r] * rinv))
                else:
                    p.xs[q, k] = BFloat16(y[q, r] * rinv)
        self.csync.sync()

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
        for g in cutlass.range_constexpr(1, self.gates):
            query = cutlass.select_(s == g, Int32(self.gate_query[g]), query)
        return query

    @cute.jit
    def load_gate_item(self, p, w, op, a, b):
        """The load warp's stages of a MSGDOWN (tile ``a``, split ``b``) or GATEMIX (tile ``a``
        of gate ``b``) item."""
        g = p.gate
        if op == MSGDOWN:
            row = self.first_gate * GATE_TILES + a
            per = self.hidden // TILE_K // GATE_SPLITS
            w = load_tiles(w, g.tma[0], g.gmem[0], row, b * per, per, "W")
        else:
            for stage in cutlass.range_constexpr(MIX_STAGES):
                row = (self.first_gate + b) * (self.hidden // ROWS) + a * MIX_STAGES + stage
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
            index = (split * self.gates * GATE_TILES + a) * ROWS + row
            publish(p.gate.h_part, index, value, epoch)
        return w

    @cute.jit
    def stream_value(self, p, query, i):
        """Element ``i`` of the prologue's block residual of ``query``, recomputed from the
        states and MIX in the prologue's operation order (the same fp32 value)."""
        weights = STATES * (query + 1)
        acc = Float32(0.0)
        for n in cutlass.range_constexpr(MAX_BLOCKS):
            if n < p.num_blocks:
                acc = acc + p.mix[weights + n] * (Float32(p.blocks[n][i]) * p.mix[n])
        partial = Float32(self.partial(p)[i]) * p.mix[MAX_BLOCKS]
        return acc + p.mix[weights + MAX_BLOCKS] * partial

    @cute.jit
    def gatemix(self, p, w, tile, s, epoch, warp, lane):
        """GATEMIX tile ``tile`` of gate ``s``: elements i of its MIX_STAGES * 32 rows,
        ``ref.message_gate``'s ``(1 - tanh(F_i . h)) x_i + tanh(G_i . h) m_i`` with ``h =
        bf16(x-half partials in split order + message half)`` of the gate's down projection.
        In each stage (32 rows of the fp32 ``[F | G]``) warp w takes rows 4w .. 4w + 3, lane
        group lane // 8 a row, lane j = lane % 8 its columns 8j .. 8j + 7 of F and of G. The
        elements' x and m load before the poll for h, so their latency overlaps it."""
        g = p.gate
        gate = self.first_gate + s
        j = lane % 8
        row = ROWS_PER_WARP * warp + lane // 8
        query = self.gate_input(s)
        xm = cute.make_rmem_tensor((MIX_STAGES, 2), Float32)
        for stage in cutlass.range_constexpr(MIX_STAGES):
            i = (tile * MIX_STAGES + stage) * ROWS + row
            xm[stage, 0] = self.stream_value(p, query, i)
            xm[stage, 1] = g.messages[gate, i]
        hv = cute.make_rmem_tensor((GATE_SPLITS * 8,), Float32)
        stride = self.gates * GATE_RANK
        segments = [(g.h_part, k * stride + s * GATE_RANK + 8 * j, 8) for k in range(GATE_SPLITS)]
        fetch_uniform(segments, hv, epoch)
        for e in cutlass.range_constexpr(8):
            acc = hv[e]
            for k in cutlass.range_constexpr(1, GATE_SPLITS):
                acc = acc + hv[8 * k + e]
            hv[e] = bf16_round(acc + g.mhalf[gate * GATE_RANK + 8 * j + e])
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
    def gather(self, p, epoch, tid, warp, lane):
        """``xs[s] = bf16(norms[s] * rms_norm(gated stream s))`` for every gate s: the launch's
        GEMV inputs, once every GATEMIX item has published."""
        h = self.hidden
        per = h // COMPUTE_THREADS
        n = self.gates
        vals = cute.make_rmem_tensor((n * per,), Float32)
        segments = [
            (p.gate.gated, s * h + r * COMPUTE_THREADS + tid, 1)
            for s in range(n)
            for r in range(per)
        ]
        norms = cute.make_rmem_tensor((n, per), BFloat16)
        for s in cutlass.range_constexpr(n):
            for r in cutlass.range_constexpr(per):
                norms[s, r] = p.norms[s, r * COMPUTE_THREADS + tid]
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
                p.xs[s, k] = BFloat16(Float32(norms[s, r]) * (vals[s * per + r] * rinv))
        self.csync.sync()
