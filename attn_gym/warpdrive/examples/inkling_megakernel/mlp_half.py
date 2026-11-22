"""The MoE MLP half of an Inkling 2 Mini decode visit: the device code of the visit launch's
MLP rows (``MlpHalf``, a mixin of ``visit_kernel.InklingVisitKernel``).

Batch one, one token. The rows implement ``ref.mlp_block`` on the attention half's residual::

    every CTA   prologue (``visit_kernel``): block residual over [blocks..., residual],
                RMSNorm -> bf16 x (SMEM)
    [messages]  MSGDOWN, GATEMIX of the MLP gate (``common``), then each CTA gathers the gated
                stream and RMS-normalizes it into x
    ROUTER  4   32 router rows x hidden                       -> fp32 logits (tagged)
    GATEUP  576 slot s x tile t: 16 gate + 16 up rows of the  -> bf16 softcapped SwiGLU (tagged)
                slot's expert x hidden
    DOWN    576 row tile t x slot s: 32 down rows x the       -> bf16 expert outputs (tagged)
                slot's 1152 activations; slot 7's item also
                reduces its 32 columns: fp32 sum over slots,
                post-MLP conv (ring write), residual add      -> bf16 output

Slots are the top-k experts in ascending expert order, so summing slot by slot is the
oracle's ascending-expert ``index_add_`` order. Routing (sigmoid scores plus the correction
bias, top-k with ties to the lower index, renormalized and scaled weights) is recomputed from
the 128 tagged logits by every consumer that needs it: the load warp for the experts' TMA
coordinates, compute warp 0 for the reducer's weights.

The expert GEMVs (``load_expert`` streams an expert row's stages, ``expert_tile`` consumes
them into the GATEUP/DOWN epilogues) read the experts in the format chosen at construction
(``experts_format``), with the same dot products bit for bit (``quant_gemv``):

- ``"bf16"``: gate and up rows come interleaved from a strided TMA view of the layer's
  ``(experts, 2 * width, hidden)`` weights (no repacked copy); the down projection's 1152
  columns are five 256-column stages: TMA zero-fills the last stage's upper half, and the
  activation buffer's padding is zeroed.
- ``"nvfp4"``/``"mxfp8"``: the checkpoint's experts as ``quant.QuantExperts`` (checkpoint row
  order: a 32-row gate/up tile is 16 (gate, up) pairs, as the bf16 view's). A weight stage is
  the ring's 16 KB slot holding 32 rows x 128 words: 4 subtiles of NVFP4, 2 of MXFP8 (a row's
  words past its end are TMA zero fill). Each expert row first takes a stage of the ``meta``
  ring: the row tile's scales of every subtile (the down rows' 5 subtiles in a 9-subtile box,
  zero filled) and, NVFP4, the expert's magnitude planes ``lut[expert]`` (4 KB). Every
  ``meta`` stage thus has the same byte count, and with two stages the next row's planes and
  scales arrive while the compute warps, holding the current row's stage, consume its weight
  stages.
"""

import math

import cutlass
from cutlass import BFloat16, Float32, Int32, Int64, cute

from attn_gym.warpdrive.cache import make_fake_compact_tensor
from attn_gym.warpdrive.examples.inkling_megakernel.common import (
    BACKOFF_NS,
    conv_inputs,
    fetch_uniform,
    fp32_reciprocal,
    load_tiles,
    post_conv,
    tma_atom,
)
from attn_gym.warpdrive.examples.inkling_megakernel.formats import (
    BF16,
    BLOCKS,
    FORMATS,
    LANE_WORDS,
    NVFP4,
    SUBTILE_WORDS,
)
from attn_gym.warpdrive.examples.inkling_megakernel.quant_gemv import accumulate_subtile
from attn_gym.warpdrive.examples.qwen_megakernel.core import (
    COMPUTE_THREADS,
    EVICT_FIRST,
    ROWS,
    ROWS_PER_WARP,
    STAGE_BYTES,
    TILE_K,
    bf16_round,
    fetch,
    group_sum,
    publish,
    warp_transpose_reduce,
)

ROUTER, GATEUP, DOWN = 3, 4, 5  # after the attention half's ops 0-2
STAGE_WORDS = STAGE_BYTES // (4 * ROWS)  # Int32 words per row of a quantized weight stage
META_STAGES = 2
PLANE_ROWS = 16  # an expert's planes (1024 words) as a (16, 64) TMA box


class MlpHalf:
    """The MLP rows of a visit launch (see the module docstring). Its params carry, besides
    ``InklingKernel``'s: ``tma_mlp``/``gmem_mlp`` (router, gate/up, down), ``sW``, ``route``
    (``(2, 2 * top_k)`` fp32: slot s's expert at s, its weight at top_k + s; row 0 for the load
    warp, row 1 for compute warp 0), the tagged ``logits``, ``act``, ``down`` and ``resid``
    (the attention half's residual) and the argument tensors of ``visit_kernel.VisitParams``;
    quantized, the stage views ``sQW`` (weights), ``sS`` (scales) and ``sP`` (NVFP4 planes)."""

    def init_mlp(
        self,
        width: int,
        experts: int,
        top_k: int,
        gate_softcap: float,
        up_softcap: float,
        route_scale: float,
        experts_format: str,
    ):
        """Model sizes and constants of the MLP rows (``self.hidden`` set before) and the
        experts' format (``formats.FORMATS``)."""
        hidden = self.hidden
        if experts_format not in FORMATS:
            raise ValueError(f"experts format must be one of {FORMATS}, got {experts_format!r}")
        if width % (ROWS // 2) or width % 8 or math.ceil(width / TILE_K) * TILE_K > hidden:
            raise ValueError(f"expert width {width}: need a multiple of 16 that fits {hidden}")
        per_lane = experts // 32  # routing: lane l's logits are one aligned run of words
        if experts % 32 or experts % ROWS or not (per_lane in (1, 2) or per_lane % 4 == 0):
            raise ValueError(
                f"{experts} experts: routing reads experts / 32 logits per lane, which must be "
                "1, 2 or a multiple of 4 (64, 128, 256, ... experts)"
            )
        if per_lane > 32 or not 1 <= top_k <= 32:
            raise ValueError("need at most 1024 experts and 1 to 32 of them per token")
        if (width + 7) // 8 + (TILE_K - width % TILE_K) % TILE_K // 8 > COMPUTE_THREADS:
            raise ValueError("the activation fetch covers at most 8 words per compute thread")
        self.width, self.experts, self.top_k = width, experts, top_k
        self.gate_softcap, self.up_softcap = gate_softcap, up_softcap
        self.route_scale = route_scale
        self.down_k_tiles = -(-width // TILE_K)  # the last stage's columns past width are zero
        self.gate_tiles = width // (ROWS // 2)  # gate/up tiles per slot: 16 (gate, up) pairs
        self.row_tiles = hidden // ROWS  # down tiles per slot
        self.experts_format = experts_format
        self.quantized = experts_format != BF16
        # Each kind's (GATEUP, DOWN) 256-column subtiles, and weight stages of stage_subtiles.
        self.expert_subtiles = (self.hidden_k_tiles, self.down_k_tiles)
        self.stage_subtiles = STAGE_WORDS // SUBTILE_WORDS[experts_format] if self.quantized else 1
        self.expert_stages = tuple(-(-n // self.stage_subtiles) for n in self.expert_subtiles)
        # A ``meta`` stage: tile-major scale words of meta_subtiles subtiles (as many as the
        # gate/up rows have) and the expert's plane words (NVFP4).
        self.meta_subtiles = self.hidden_k_tiles
        self.scale_words = 8 * BLOCKS[experts_format] if self.quantized else 0
        self.plane_words = 1024 if experts_format == NVFP4 else 0
        self.meta_stage_bytes = 4 * (self.meta_subtiles * self.scale_words + self.plane_words)

    def mlp_scratch_shapes(self) -> dict[str, int]:
        """Tagged int64 words of each MLP intermediate."""
        return {
            "logits": self.experts,
            "act": self.top_k * self.width,
            "down": self.top_k * self.hidden,
        }

    # ------------------------------------------------------------------ rows

    @cute.jit
    def load_mlp(self, p, w, meta, op, a, b, routed, lane, epoch):
        """The load warp's stages of MLP row ``(op, a, b)``. The expert rows follow the
        routing: the first one waits for the router's logits (``routed`` then 1). Returns the
        weight and ``meta`` rings' producers and ``routed``."""
        if op == ROUTER:
            w = load_tiles(w, p.tma_mlp[0], p.gmem_mlp[0], a, 0, self.hidden_k_tiles, "W")
        else:
            if routed == 0:
                with self.trace_region("route"):
                    self.route(p, lane, epoch, 0)
                routed = Int32(1)
            w, meta = self.load_expert(p, w, meta, op, a, Int32(p.route[0, b]))
        return w, meta, routed

    @cute.jit
    def load_expert(self, p, w, meta, op, a, expert):
        """The stages of GATEUP tile ``a`` or DOWN row tile ``a`` of ``expert``: quantized, its
        ``meta`` stage, then the weight stages. ``tma_mlp``: router, gate/up and down weights,
        then (quantized) their scales and (NVFP4) planes."""
        for kind in cutlass.range_constexpr(2):
            if op == (GATEUP, DOWN)[kind]:
                tile = expert * (self.gate_tiles, self.row_tiles)[kind] + a
                buffer = "W"
                if cutlass.const_expr(self.quantized):
                    buffer = "QW"
                    with meta.produce() as h:
                        cute.copy(
                            p.tma_mlp[3 + kind],
                            p.gmem_mlp[3 + kind][None, tile, 0, 0],
                            h.view("S"),
                            tma_bar_ptr=h.barrier,
                            cache_policy=Int64(EVICT_FIRST),
                        )
                        if cutlass.const_expr(self.plane_words):
                            # Other CTAs read the expert's planes too: default L2 policy.
                            cute.copy(
                                p.tma_mlp[5 + kind],
                                p.gmem_mlp[5 + kind][None, expert, 0],
                                h.view("P"),
                                tma_bar_ptr=h.barrier,
                            )
                w = load_tiles(
                    w, p.tma_mlp[1 + kind], p.gmem_mlp[1 + kind], tile, 0,
                    self.expert_stages[kind], buffer,
                )  # fmt: skip
        return w, meta

    @cute.jit
    def mlp_row(self, p, w, meta, op, a, b, held, abuf, routed, epoch, tid, warp, lane):
        """The compute warps' part of MLP row ``(op, a, b)``. DOWN rows read their slot's
        activations from xs row ``abuf`` (1 or 2), fetched when the slot changes from
        ``held``; the reducer's routing is computed once (``routed``). Returns the weight and
        ``meta`` rings' consumers, ``held``, ``abuf`` and ``routed``."""
        if op == ROUTER:
            vals, row, chans = self.engine.tile(p, (w,), (), 0, self.hidden_k_tiles, warp, lane)
            w = chans[0]
            if self.engine.owns_row(warp, lane):
                publish(p.logits, a * ROWS + row, vals[0], epoch)
        else:
            if (op == DOWN) & (b != held):
                # Alternate buffers: every warp passed the last fetch's barrier after
                # finishing the row before it, the last reader of this buffer.
                abuf = 3 - abuf
                held = b
                self.fetch_act(p, b, abuf, epoch, tid)
                self.csync.sync()
            w, meta = self.expert_tile(p, w, meta, op, a, b, abuf, epoch, warp, lane)
            if (op == DOWN) & (b == self.top_k - 1) & (warp == 0):
                if routed == 0:
                    self.route(p, lane, epoch, 1)
                    routed = Int32(1)
                self.reduce_experts(p, a * ROWS + lane, epoch)
        return w, meta, held, abuf, routed

    @cute.jit
    def expert_tile(self, p, w, meta, op, a, b, abuf, epoch, warp, lane):
        """GATEUP tile ``a`` of slot ``b`` against the MLP input ``xs[0]``, or DOWN row tile
        ``a`` of slot ``b`` against its activations ``xs[abuf]``, then the epilogue."""
        # Constant stage counts, so the k loops unroll fully.
        for kind in cutlass.range_constexpr(2):
            if op == (GATEUP, DOWN)[kind]:
                xbuf = (0, abuf)[kind]
                if cutlass.const_expr(self.quantized):
                    with meta.consume() as h:
                        v, row, w = self.quant_tile(p, w, h, kind, xbuf, warp, lane)
                    self.epilogue(p, op, a, b, v, row, epoch, warp, lane)
                else:
                    k_tiles = self.expert_subtiles[kind]
                    vals, row, chans = self.engine.tile(p, (w,), (), xbuf, k_tiles, warp, lane)
                    w = chans[0]
                    self.epilogue(p, op, a, b, vals[0], row, epoch, warp, lane)
        return w, meta

    @cute.jit
    def quant_tile(self, p, w, meta_stage, kind: cutlass.Constexpr, xbuf, warp, lane):
        """``CudaCoreGemv.tile`` of a quantized expert row tile of ``kind`` (GATEUP 0, DOWN 1)
        against ``xs[xbuf]``, given the row's ``meta`` stage: the same fp32 operations in the
        same order on the decoded weights, so the same ``(value, row, w)`` bit for bit as the
        bf16 engine on the dequantized weights."""
        fmt = self.experts_format
        blocks = BLOCKS[fmt]
        scales = meta_stage.view("S")
        planes = None
        if cutlass.const_expr(self.plane_words):
            plane_stage = meta_stage.view("P")
            planes = cute.make_tensor(plane_stage.iterator, cute.make_layout(self.plane_words))
        xwords = cute.recast_tensor(p.xs[xbuf, None], Int32)
        acc = cute.make_rmem_tensor((ROWS_PER_WARP, 2), Float32)
        acc.fill(0.0)
        for st in cutlass.range_constexpr(self.expert_stages[kind]):
            with w.consume() as h:
                stage = h.view("QW")
                for j in cutlass.range_constexpr(self.stage_subtiles):
                    kt = st * self.stage_subtiles + j
                    if cutlass.const_expr(kt < self.expert_subtiles[kind]):
                        xw = cute.make_rmem_tensor((4,), Int32)
                        cute.autovec_copy(cute.local_tile(xwords, (4,), (kt * 32 + lane,)), xw)
                        word = scales[0, kt, warp * blocks + lane // (32 // blocks)]
                        acc = accumulate_subtile(
                            fmt, acc, stage, warp * ROWS_PER_WARP, j * SUBTILE_WORDS[fmt], xw,
                            word, planes, lane,
                        )  # fmt: skip
        for r in cutlass.range_constexpr(ROWS_PER_WARP):
            acc[r, 0] = acc[r, 0] + acc[r, 1]
        v = warp_transpose_reduce(acc[None, 0], lane, ROWS_PER_WARP)
        return v, warp * ROWS_PER_WARP + lane // (32 // ROWS_PER_WARP), w

    # ------------------------------------------------------------ compute helpers

    @cute.jit
    def fetch_act(self, p, slot, abuf, epoch, tid):
        """Slot ``slot``'s activations into ``xs[abuf]``, 8 tagged words per thread, and zeros
        in the padding the last down stage reads (its weights there are TMA's zero fill)."""
        chunks = self.width // 8
        pad = (self.down_k_tiles * TILE_K - self.width) // 8
        if tid < chunks:
            vals = cute.make_rmem_tensor((8,), Float32)
            fetch([(p.act, slot * self.width + tid * 8, 8)], vals, epoch)
            for e in cutlass.range_constexpr(8):
                p.xs[abuf, tid * 8 + e] = BFloat16(vals[e])
        elif tid < chunks + pad:
            for e in cutlass.range_constexpr(8):
                p.xs[abuf, tid * 8 + e] = BFloat16(0.0)

    @cute.jit
    def epilogue(self, p, op, a, b, v, row, epoch, warp, lane):
        """Publish one expert tile's results; ``v`` is tile row ``row``'s dot product on the
        lanes the engine says own it."""
        if op == GATEUP:
            # Tile rows alternate gate, up: the owner of gate row 2j gets up row 2j + 1's.
            up = self.engine.pair_value(v)
            if self.engine.owns_pair(warp, lane):
                act = self.swiglu(bf16_round(v), bf16_round(up))
                col = b * self.width + a * (ROWS // 2) + row // 2
                publish(p.act, col, bf16_round(act), epoch)
        elif self.engine.owns_row(warp, lane):
            publish(p.down, b * self.hidden + a * ROWS + row, bf16_round(v), epoch)

    @cute.jit
    def swiglu(self, g, u):
        """The softcapped SwiGLU in fp32, in ``ref.experts_out``'s operation order (PyTorch
        divides by a scalar as a multiplication by its fp32 reciprocal)."""
        gate = cute.math.tanh(g * fp32_reciprocal(self.gate_softcap)) * self.gate_softcap
        gate = gate * (Float32(1.0) / (Float32(1.0) + cute.math.exp(-g)))
        cap = cute.math.tanh(u * fp32_reciprocal(self.up_softcap)) * self.up_softcap
        return gate * cap

    @cute.jit
    def reduce_experts(self, p, col, epoch):
        """Output column ``col``, by one lane of warp 0: the fp32 sum of the slots' bf16 down
        outputs times their routing weights in slot (ascending expert) order, rounded to bf16
        and written to the convolution ring at the position, then the post-MLP convolution
        ``(conv + y) / sqrt(kernel + 1)`` (bf16) plus the attention half's residual."""
        k = self.top_k
        pos = Int32(p.pos[0])
        # The weights and ring rows load before the poll (their latency overlaps it).
        weights, past = conv_inputs(p.ring_mlp, p.mlp_conv, col, pos, self.conv_kernel)
        vals = cute.make_rmem_tensor((k + 1,), Float32)
        segments = [(p.down, s * self.hidden + col, 1) for s in range(k)]
        fetch(segments + [(p.resid, col, 1)], vals, epoch)
        acc = Float32(0.0)
        for s in cutlass.range_constexpr(k):
            acc = acc + vals[s] * p.route[1, k + s]
        z = post_conv(p.ring_mlp, weights, past, col, pos, bf16_round(acc), self.conv_kernel)
        p.out[col] = BFloat16(vals[k] + z)

    @cute.jit
    def route(self, p, lane, epoch, which: cutlass.Constexpr):
        """One warp's routing into ``route[which]``, as ``ref.route``: the top-k experts of
        ``sigmoid(logit) + bias`` (ties to the lower index) in ascending expert order, then
        their scores renormalized over the top k and scaled by ``route_scale``. Each round
        takes the warp's largest remaining key; the slots come from a scan of the chosen
        counts, so no sort is needed. The logits are polled warp-uniformly
        (``fetch_uniform``): the shuffles that follow need every lane. The load warp polls them
        from the end of its CTA's attention stages, so it backs off between polls."""
        n = self.experts // 32  # lane l holds experts n*l .. n*l + n - 1
        k = self.top_k
        logits = cute.make_rmem_tensor((n,), Float32)
        fetch_uniform([(p.logits, n * lane, n)], logits, epoch, BACKOFF_NS)
        keys = cute.make_rmem_tensor((n,), Float32)
        scores = cute.make_rmem_tensor((n,), Float32)
        for i in cutlass.range_constexpr(n):
            scores[i] = Float32(1.0) / (Float32(1.0) + cute.math.exp(-logits[i]))
            keys[i] = scores[i] + p.router_bias[n * lane + i]
        taken = Int32(0)  # bit i: expert n*lane + i is in the top k
        for _ in cutlass.range(k, unroll=1):
            best, index = keys[0], n * lane
            for i in cutlass.range_constexpr(1, n):
                better = keys[i] > best
                best = cutlass.select_(better, keys[i], best)
                index = cutlass.select_(better, n * lane + i, index)
            for step in cutlass.range_constexpr(5):
                other = cute.arch.shuffle_sync_bfly(best, offset=16 >> step)
                other_index = cute.arch.shuffle_sync_bfly(index, offset=16 >> step)
                better = (other > best) | ((other == best) & (other_index < index))
                best = cutlass.select_(better, other, best)
                index = cutlass.select_(better, other_index, index)
            for i in cutlass.range_constexpr(n):
                hit = n * lane + i == index
                taken = taken | (Int32(hit) << i)
                keys[i] = cutlass.select_(hit, -Float32.inf, keys[i])
        # A chosen expert's slot counts the chosen experts below it: an exclusive scan over
        # lanes of their counts, then the lane's own lower bits.
        count = cute.arch.popc(taken)
        below = count
        for step in cutlass.range_constexpr(5):
            up = cute.arch.shuffle_sync_up(below, offset=1 << step)
            below = below + cutlass.select_(lane >= (1 << step), up, Int32(0))
        slot = below - count
        chosen = Float32(0.0)
        for i in cutlass.range_constexpr(n):
            if (taken >> i) & 1:
                p.route[which, slot] = Float32(n * lane + i)
                p.route[which, k + slot] = scores[i]
                chosen = chosen + scores[i]
                slot = slot + 1
        total = group_sum(chosen, 32)
        cute.arch.sync_warp()
        if lane < k:
            weight = p.route[which, k + lane] / (total + Float32(1e-20)) * self.route_scale
            p.route[which, k + lane] = weight
        cute.arch.sync_warp()

    # ------------------------------------------------------------- TMA atoms

    def mlp_targets(self) -> list[tuple[str, tuple[int, ...]]]:
        """The SMEM buffer and box of each of ``mlp_atoms``' atoms, in order."""
        if not self.quantized:
            return [("W", (ROWS, TILE_K))] * 3
        targets = [("W", (ROWS, TILE_K))] + [("QW", (ROWS, STAGE_WORDS))] * 2
        targets += [("S", (1, self.meta_subtiles, self.scale_words))] * 2
        if self.plane_words:
            targets += [("P", (PLANE_ROWS, self.plane_words // PLANE_ROWS))] * 2
        return targets

    @cute.jit
    def mlp_atoms(self, router, w13, w2, s13, s2, lut13, lut2):
        """TMA ``(atom, tensor)`` pairs of the router's 32 x 256 bf16 tiles, then of the
        experts (``fake_expert_args``), boxed as ``mlp_targets``. bf16: the gate/up rows as
        ``((gate | up), row, expert)`` (a tile holds 16 (gate, up) pairs, interleaved like
        ``qwen_megakernel``'s packed weights) and the down rows. Quantized: weight stages of the
        gate/up and down rows, each row tile's scales and (NVFP4) each expert's planes."""
        h, width, e = self.hidden, self.width, self.experts
        sources = [router, w13, w2, s13, s2, lut13, lut2]
        if cutlass.const_expr(not self.quantized):
            layout = cute.make_layout(
                ((2, width, e), h), stride=((width * h, h, 2 * width * h), 1)
            )
            sources[1] = cute.make_tensor(w13.iterator, layout)
            layout = cute.make_layout((e * h, width), stride=(width, 1))
            sources[2] = cute.make_tensor(w2.iterator, layout)
        return [tma_atom(src, box) for src, (_, box) in zip(sources, self.mlp_targets())]


def fake_expert_args(op: MlpHalf) -> list:
    """Fake compile arguments of a layer's experts in ``op``'s format: ``w13``, ``w2``,
    their scales and their planes. bf16: ``(experts, 2 * width, hidden)`` gate rows then up
    rows and ``(experts, hidden, width)``, then placeholders. Quantized
    (``quant.QuantExperts``, checkpoint row order): the weights as Int32 words ``(experts *
    rows, columns / 8 | 4)``, the tile-major scales as Int32 ``(experts * rows / 32,
    subtiles, scale_words)`` and (NVFP4; else placeholders) the planes ``(experts * 16,
    64)``."""
    i32, e, h, width = Int32, op.experts, op.hidden, op.width
    if not op.quantized:
        return [
            make_fake_compact_tensor(BFloat16, (e, 2 * width, h)),
            make_fake_compact_tensor(BFloat16, (e, h, width)),
            *(make_fake_compact_tensor(i32, (1, 1, 4)) for _ in range(2)),
            *(make_fake_compact_tensor(i32, (1, 4)) for _ in range(2)),
        ]
    per_word = 8 // LANE_WORDS[op.experts_format]  # columns per word
    planes = (e * PLANE_ROWS, op.plane_words // PLANE_ROWS) if op.plane_words else (1, 4)
    return [
        make_fake_compact_tensor(i32, (e * 2 * width, h // per_word)),
        make_fake_compact_tensor(i32, (e * h, width // per_word)),
        make_fake_compact_tensor(i32, (e * op.gate_tiles, op.hidden_k_tiles, op.scale_words)),
        make_fake_compact_tensor(i32, (e * op.row_tiles, op.down_k_tiles, op.scale_words)),
        *(make_fake_compact_tensor(i32, planes) for _ in range(2)),
    ]
