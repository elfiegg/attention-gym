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

The expert GEMVs are one swappable piece: ``load_expert`` streams an expert row's weight
stages and ``expert_tile`` consumes them into the GATEUP/DOWN epilogues. Here they are bf16:
gate and up rows come interleaved from a strided TMA view of the layer's ``(experts, 2 *
width, hidden)`` weights (no repacked copy); the down projection's 1152 columns are five
256-column stages: TMA zero-fills the last stage's upper half, and the activation buffer's
padding is zeroed.
"""

import math

import cutlass
from cutlass import BFloat16, Float32, Int32, cute
from cutlass.cute.nvgpu import cpasync

from attn_gym.warpdrive.examples.inkling_megakernel.common import (
    BACKOFF_NS,
    RING,
    fetch_uniform,
    fp32_reciprocal,
    load_tiles,
)
from attn_gym.warpdrive.examples.qwen_megakernel.core import (
    COMPUTE_THREADS,
    ROWS,
    TILE_K,
    bf16_round,
    fetch,
    group_sum,
    publish,
)

ROUTER, GATEUP, DOWN = 3, 4, 5  # after the attention half's ops 0-2


class MlpHalf:
    """The MLP rows of a visit launch (see the module docstring). Its params carry, besides
    ``InklingKernel``'s: ``tma_mlp``/``gmem_mlp`` (router, gate/up, down), ``sW``, ``route``
    (``(2, 2 * top_k)`` fp32: slot s's expert at s, its weight at top_k + s; row 0 for the load
    warp, row 1 for compute warp 0), the tagged ``logits``, ``act``, ``down`` and ``resid``
    (the attention half's residual) and the argument tensors of ``visit_kernel.VisitParams``."""

    def init_mlp(
        self,
        width: int,
        experts: int,
        top_k: int,
        gate_softcap: float,
        up_softcap: float,
        route_scale: float,
    ):
        """Model sizes and constants of the MLP rows (``self.hidden`` set before)."""
        hidden = self.hidden
        if width % (ROWS // 2) or width % 8 or math.ceil(width / TILE_K) * TILE_K > hidden:
            raise ValueError(f"expert width {width}: need a multiple of 16 that fits {hidden}")
        if experts % 32 or experts % ROWS or not 1 <= top_k <= 32:
            raise ValueError("need a multiple of 32 experts and at most 32 per token")
        if (width + 7) // 8 + (TILE_K - width % TILE_K) % TILE_K // 8 > COMPUTE_THREADS:
            raise ValueError("the activation fetch covers at most 8 words per compute thread")
        self.width, self.experts, self.top_k = width, experts, top_k
        self.gate_softcap, self.up_softcap = gate_softcap, up_softcap
        self.route_scale = route_scale
        self.down_k_tiles = -(-width // TILE_K)  # the last stage's columns past width are zero
        self.gate_tiles = width // (ROWS // 2)  # gate/up tiles per slot: 16 (gate, up) pairs
        self.row_tiles = hidden // ROWS  # down tiles per slot

    def mlp_scratch_shapes(self) -> dict[str, int]:
        """Tagged int64 words of each MLP intermediate."""
        return {
            "logits": self.experts,
            "act": self.top_k * self.width,
            "down": self.top_k * self.hidden,
        }

    # ------------------------------------------------------------------ rows

    @cute.jit
    def load_mlp(self, p, w, op, a, b, routed, lane, epoch):
        """The load warp's stages of MLP row ``(op, a, b)``. The expert rows follow the
        routing: the first one waits for the router's logits (``routed`` then 1). Returns the
        weight ring's producer and ``routed``."""
        if op == ROUTER:
            w = load_tiles(w, p.tma_mlp[0], p.gmem_mlp[0], a, 0, self.hidden_k_tiles, "W")
        else:
            if routed == 0:
                with self.trace_region("route"):
                    self.route(p, lane, epoch, 0)
                routed = Int32(1)
            w = self.load_expert(p, w, op, a, Int32(p.route[0, b]))
        return w, routed

    @cute.jit
    def load_expert(self, p, w, op, a, expert):
        """The weight stages of GATEUP tile ``a`` or DOWN row tile ``a`` of ``expert``."""
        if op == GATEUP:
            tile = expert * self.gate_tiles + a
            w = load_tiles(w, p.tma_mlp[1], p.gmem_mlp[1], tile, 0, self.hidden_k_tiles, "W")
        else:
            tile = expert * self.row_tiles + a
            w = load_tiles(w, p.tma_mlp[2], p.gmem_mlp[2], tile, 0, self.down_k_tiles, "W")
        return w

    @cute.jit
    def mlp_row(self, p, w, op, a, b, held, abuf, routed, epoch, tid, warp, lane):
        """The compute warps' part of MLP row ``(op, a, b)``. DOWN rows read their slot's
        activations from xs row ``abuf`` (1 or 2), fetched when the slot changes from
        ``held``; the reducer's routing is computed once (``routed``). Returns the weight
        ring's consumer, ``held``, ``abuf`` and ``routed``."""
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
            w = self.expert_tile(p, w, op, a, b, abuf, epoch, warp, lane)
            if (op == DOWN) & (b == self.top_k - 1) & (warp == 0):
                if routed == 0:
                    self.route(p, lane, epoch, 1)
                    routed = Int32(1)
                self.reduce_experts(p, a * ROWS + lane, epoch)
        return w, held, abuf, routed

    @cute.jit
    def expert_tile(self, p, w, op, a, b, abuf, epoch, warp, lane):
        """GATEUP tile ``a`` of slot ``b`` against the MLP input ``xs[0]``, or DOWN row tile
        ``a`` of slot ``b`` against its activations ``xs[abuf]``, then the epilogue."""
        # Constant stage counts, so the engine's k loop unrolls fully.
        if op == DOWN:
            vals, row, chans = self.engine.tile(p, (w,), (), abuf, self.down_k_tiles, warp, lane)
            w = chans[0]
            self.epilogue(p, op, a, b, vals[0], row, epoch, warp, lane)
        else:
            vals, row, chans = self.engine.tile(p, (w,), (), 0, self.hidden_k_tiles, warp, lane)
            w = chans[0]
            self.epilogue(p, op, a, b, vals[0], row, epoch, warp, lane)
        return w

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
        vals = cute.make_rmem_tensor((k + 1,), Float32)
        segments = [(p.down, s * self.hidden + col, 1) for s in range(k)]
        fetch(segments + [(p.resid, col, 1)], vals, epoch)
        acc = Float32(0.0)
        for s in cutlass.range_constexpr(k):
            acc = acc + vals[s] * p.route[1, k + s]
        y = bf16_round(acc)
        pos = Int32(p.pos[0])
        p.ring_mlp[pos % RING, col] = BFloat16(y)
        taps = self.conv_kernel
        conv = Float32(0.0)
        for j in cutlass.range_constexpr(taps - 1):
            src = pos - (taps - 1) + j  # >= -(taps - 1): positions before 0 read zero
            past = Float32(p.ring_mlp[(src + RING) % RING, col])
            conv = conv + p.mlp_conv[col, j] * cutlass.select_(src >= 0, past, Float32(0.0))
        conv = conv + p.mlp_conv[col, taps - 1] * y
        z = bf16_round((conv + y) * fp32_reciprocal(math.sqrt(taps + 1)))
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

    @cute.jit
    def mlp_atoms(self, router, w13, w2):
        """TMA ``(atom, tensor)`` pairs of 32 x 256 bf16 tiles of the router, the gate/up rows
        as ``((gate | up), row, expert)`` (a tile holds 16 (gate, up) pairs, interleaved like
        ``qwen_megakernel``'s packed weights) and the down rows."""
        h, width, e = self.hidden, self.width, self.experts
        gate_up = cute.make_tensor(
            w13.iterator,
            cute.make_layout(((2, width, e), h), stride=((width * h, h, 2 * width * h), 1)),
        )
        down = cute.make_tensor(w2.iterator, cute.make_layout((e * h, width), stride=(width, 1)))
        op = cpasync.CopyBulkTensorTileG2SOp()
        smem = cute.make_layout((ROWS, TILE_K), stride=(TILE_K, 1))
        return [
            cpasync.make_tiled_tma_atom(op, src, smem, (ROWS, TILE_K))
            for src in (router, gate_up, down)
        ]
