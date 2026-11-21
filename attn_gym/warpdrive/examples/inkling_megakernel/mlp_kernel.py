"""The MoE MLP half of one Inkling 2 Mini decode visit as one persistent warpdrive launch
(``InklingMlpKernel``) and its compile function; ``megakernel`` holds the host side.

Batch one, one token. The launch implements ``ref.mlp_block``::

    every CTA   prologue: block residual over [blocks..., residual], RMSNorm -> bf16 x (SMEM)
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

Roles of one CTA (9 warps), as ``qwen_megakernel``: warp 8 ``load`` streams 32 x 256 weight
stages through the ``w`` ring (it waits on activations only for the routing), warps 0-7
``compute`` run ``CudaCoreGemv`` over the stages and publish tagged words
``(fp32 value, launch epoch)``; consumers spin on relaxed loads until every word carries the
epoch. Gate and up rows come interleaved from a strided TMA view of the layer's
``(experts, 2 * width, hidden)`` weights (no repacked copy). The down projection's 1152
columns are five 256-column stages: TMA zero-fills the last stage's upper half, and the
activation buffer's padding is zeroed.

The host assigns rows to CTAs in dependency order (``megakernel.build_table``) for
``StaticPersistent``; with every CTA resident, the earliest unfinished row always has its
producers finished. The last CTA to finish advances the epoch, so one runner's launches may
follow each other in a stream or a CUDA graph.
"""

import math
from dataclasses import dataclass

import cutlass
from cuda.bindings import driver as cuda
from cutlass import BFloat16, Float32, Int32, Int64, cute
from cutlass.cute.nvgpu import cpasync

from attn_gym.warpdrive import (
    Consumer,
    Local,
    NamedBarrier,
    Producer,
    Ring,
    SmemPlan,
    StageView,
    StaticPersistent,
    Tma,
    Warps,
    WorkCursor,
    WorkStream,
    role,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor
from attn_gym.warpdrive.examples.inkling_megakernel.common import (
    MAX_BLOCKS,
    MIX_FLOATS,
    MSGDOWN,
    RED_VALUES,
    REDUCTIONS,
    RING,
    SMEM_BYTES,
    SMEM_HEADROOM,
    WORDS,
    GateParams,
    InklingKernel,
    fake_gate_args,
    fetch_uniform,
    fp32_reciprocal,
    gate_atoms,
    load_tiles,
)
from attn_gym.warpdrive.examples.qwen_megakernel.core import (
    COMPUTE_THREADS,
    COMPUTE_WARPS,
    ROWS,
    STAGE_BYTES,
    TILE_K,
    CudaCoreGemv,
    InstructionTable,
    bf16_round,
    fetch,
    group_sum,
    publish,
)

ROUTER, GATEUP, DOWN = range(3)
MAX_STAGES = 12
# MlpParams fields passed to the kernel as one tuple of argument tensors, in this order.
TENSOR_ARGS = (
    "trace_buffer", "state", "pos", "residual", "query", "norms", "conv", "router_bias", "ring", "mhalf",
    "messages", "out",
)  # fmt: skip


@dataclass(frozen=True)
class MlpParams:
    tma: tuple  # TMA atoms: router, gate/up, down
    gmem: tuple  # their tile partitions: (TMA, row tiles, k tiles)
    sW: StageView
    xs: cute.Tensor  # (3, hidden) bf16: the normalized input, then two activation buffers
    red: cute.Tensor  # (REDUCTIONS, COMPUTE_WARPS, RED_VALUES) fp32 prologue reductions
    mix: cute.Tensor  # (MIX_FLOATS,) fp32 the prologue's block-residual weights (messages)
    route: cute.Tensor  # (2, 2 * top_k) fp32: slot s's expert at s, its weight at top_k + s
    logits: cute.Tensor  # tagged int64 (experts,)
    act: cute.Tensor  # tagged int64 (top_k * width,)
    down: cute.Tensor  # tagged int64 (top_k * hidden,)
    gate: GateParams | None  # the MLP message gate (``messages`` kernels)
    state: cute.Tensor  # int32 (2,): launch epoch, finished CTAs
    pos: cute.Tensor  # int32 (1,) position of the token
    residual: cute.Tensor  # (hidden,) fp32
    blocks: tuple  # MAX_BLOCKS (hidden,) bf16 completed blocks; the first num_blocks are read
    num_blocks: Int32
    query: cute.Tensor  # (1, hidden) fp32 block-residual query
    norms: cute.Tensor  # (1, hidden) bf16 post-attention RMSNorm weight
    conv: cute.Tensor  # (hidden, kernel) fp32 post-MLP taps, the last on the current token
    router_bias: cute.Tensor  # (experts,) fp32
    ring: cute.Tensor  # (RING, hidden) bf16 the visit's post-MLP convolution inputs
    mhalf: cute.Tensor  # (GATES * GATE_RANK,) fp32 message halves (read when ``messages``)
    messages: cute.Tensor  # (GATES, hidden) fp32
    out: cute.Tensor  # (hidden,) bf16


class InklingMlpKernel(InklingKernel):
    """One visit's MoE MLP half (see the module docstring). Constructor arguments are the model
    sizes and constants the kernel reads; ``messages``: a core visit after the first pass, whose
    input passes through the MLP message gate; ``stages``: weight ring depth in 16 KB stages."""

    def __init__(
        self,
        hidden: int,
        width: int,
        experts: int,
        top_k: int,
        conv_kernel: int,
        rms_eps: float,
        gate_softcap: float,
        up_softcap: float,
        route_scale: float,
        messages: bool = False,
        stages: int | None = None,
    ):
        if hidden % COMPUTE_THREADS or hidden % TILE_K:
            raise ValueError(f"hidden size must be a multiple of {COMPUTE_THREADS} and {TILE_K}")
        if width % (ROWS // 2) or width % 8 or math.ceil(width / TILE_K) * TILE_K > hidden:
            raise ValueError(f"expert width {width}: need a multiple of 16 that fits {hidden}")
        if experts % 32 or experts % ROWS or not 1 <= top_k <= 32:
            raise ValueError("need a multiple of 32 experts and at most 32 per token")
        if (width + 7) // 8 + (TILE_K - width % TILE_K) % TILE_K // 8 > COMPUTE_THREADS:
            raise ValueError("the activation fetch covers at most 8 words per compute thread")
        self.hidden, self.width, self.experts, self.top_k = hidden, width, experts, top_k
        self.conv_kernel = conv_kernel
        self.rms_eps = rms_eps
        self.gate_softcap, self.up_softcap = gate_softcap, up_softcap
        self.route_scale = route_scale
        self.init_messages(messages, 3, (0,))  # the MLP gate
        self.hidden_k_tiles = hidden // TILE_K
        self.down_k_tiles = -(-width // TILE_K)  # the last stage's columns past width are zero
        self.gate_tiles = width // (ROWS // 2)  # gate/up tiles per slot: 16 (gate, up) pairs
        self.row_tiles = hidden // ROWS  # down tiles per slot
        fixed = 2 * 3 * hidden + SMEM_HEADROOM
        fixed += 4 * (REDUCTIONS * COMPUTE_WARPS * RED_VALUES + MIX_FLOATS + 4 * top_k)
        fit = (SMEM_BYTES - fixed) // STAGE_BYTES
        stages = min(MAX_STAGES, fit) if stages is None else stages
        if not 2 <= stages <= fit:
            raise ValueError(f"stages must be in [2, {fit}], got {stages}")
        self.stages = stages
        self.smem_plan = SmemPlan()
        w = self.smem_plan.buffer("W", BFloat16, align=1024)
        self.smem_plan.buffer("X", BFloat16, align=16)
        self.smem_plan.buffer("RED", Float32, align=16)
        self.smem_plan.buffer("MIX", Float32, align=16)
        # Routing by slot (expert, weight), one row for the load warp, one for compute warp 0.
        self.smem_plan.buffer("ROUTE", Float32, align=16)
        # Compute reads stages with generic loads before TMA refills them; the layer emits the
        # fence.proxy.async on a Warps release of a Tma-filled ring.
        self.w = Ring(stages, Tma(STAGE_BYTES), Warps, buffer={"W": w})
        self.csync = NamedBarrier(threads=COMPUTE_THREADS)
        self.engine = CudaCoreGemv()
        self.work = WorkStream(mapping=InstructionTable(words=WORDS), schedule=StaticPersistent())

    @property
    def scratch_shapes(self) -> dict[str, int]:
        """Tagged int64 words of each per-launch intermediate."""
        shapes = {
            "logits": self.experts,
            "act": self.top_k * self.width,
            "down": self.top_k * self.hidden,
        }
        return shapes | self.gate_scratch_shapes()

    def partial(self, p: MlpParams):
        return p.residual

    def get_name(self) -> str:
        return (
            f"warpdrive_inkling_mlp{'_msg' if self.messages else ''}_h{self.hidden}"
            f"_i{self.width}_e{self.experts}k{self.top_k}_s{self.stages}{self.trace_key()}"
        )

    # ------------------------------------------------------------------ roles

    @role(warps=COMPUTE_WARPS)
    @cute.jit
    def load(self, p: MlpParams, *, w: Producer, work: WorkCursor):
        w = w.views(W=p.sW)
        tma_router, tma_gu, tma_down = p.tma
        g_router, g_gu, g_down = p.gmem
        epoch = p.state[0] + 1
        lane = cute.arch.lane_idx()
        routed = Int32(0)
        while work.valid:
            op, a, b, _row = work.item
            if op == ROUTER:
                w = load_tiles(w, tma_router, g_router, a, 0, self.hidden_k_tiles, "W")
            elif op <= DOWN:
                # The expert rows follow the routing: wait for the router's logits once.
                if routed == 0:
                    self.route(p, lane, epoch, 0)
                    routed = Int32(1)
                expert = Int32(p.route[0, b])
                if op == GATEUP:
                    tile = expert * self.gate_tiles + a
                    w = load_tiles(w, tma_gu, g_gu, tile, 0, self.hidden_k_tiles, "W")
                else:
                    tile = expert * self.row_tiles + a
                    w = load_tiles(w, tma_down, g_down, tile, 0, self.down_k_tiles, "W")
            if cutlass.const_expr(self.messages):
                if op >= MSGDOWN:
                    w = self.load_gate_item(p, w, op, a, b)
            work.advance()
        return w, work

    @role(warps=range(COMPUTE_WARPS))
    @cute.jit
    def compute(self, p: MlpParams, *, w: Consumer, work: WorkCursor, me: Local):
        tid, warp, lane = me.tid, me.warp, me.lane
        epoch = p.state[0] + 1
        x = self.load_states(p, p.residual, tid)
        self.normalize_states(p, x, p.query, 1, Int32(1), tid, warp, lane)
        held = Int32(-1)  # the slot whose activations the current buffer holds
        abuf = Int32(1)
        routed = Int32(0)
        gathered = Int32(0)
        while work.valid:
            op, a, b, item = work.item
            if cutlass.const_expr(self.messages):
                w, gathered = self.message_item(
                    p, w, op, a, b, gathered, epoch, tid, warp, lane, item
                )
            xbuf = Int32(0)
            if op == DOWN:
                if b != held:
                    # Alternate buffers: every warp passed the last fetch's barrier after
                    # finishing the row before it, the last reader of this buffer.
                    abuf = 3 - abuf
                    held = b
                    self.fetch_act(p, b, abuf, epoch, tid)
                    self.csync.sync()
                xbuf = abuf
            # Constant stage counts, so the engine's k loop unrolls fully.
            if op == DOWN:
                vals, row, chans = self.engine.tile(
                    p, (w,), (), xbuf, self.down_k_tiles, warp, lane
                )
                w = chans[0]
                self.epilogue(p, op, a, b, vals[0], row, epoch, warp, lane)
            elif op < DOWN:
                vals, row, chans = self.engine.tile(
                    p, (w,), (), xbuf, self.hidden_k_tiles, warp, lane
                )
                w = chans[0]
                self.epilogue(p, op, a, b, vals[0], row, epoch, warp, lane)
            if (op == DOWN) & (b == self.top_k - 1) & (warp == 0):
                if routed == 0:
                    self.route(p, lane, epoch, 1)
                    routed = Int32(1)
                self.reduce(p, a * ROWS + lane, epoch)
            work.advance()
        self.advance_epoch(p, tid, epoch)
        return w, work

    # ------------------------------------------------------------ compute helpers

    @cute.jit
    def fetch_act(self, p: MlpParams, slot, abuf, epoch, tid):
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
    def epilogue(self, p: MlpParams, op, a, b, v, row, epoch, warp, lane):
        """Publish one tile's results; ``v`` is tile row ``row``'s dot product on the lanes the
        engine says own it."""
        if op == GATEUP:
            # Tile rows alternate gate, up: the owner of gate row 2j gets up row 2j + 1's.
            up = self.engine.pair_value(v)
            if self.engine.owns_pair(warp, lane):
                act = self.swiglu(bf16_round(v), bf16_round(up))
                col = b * self.width + a * (ROWS // 2) + row // 2
                publish(p.act, col, bf16_round(act), epoch)
        elif self.engine.owns_row(warp, lane):
            if op == ROUTER:
                publish(p.logits, a * ROWS + row, v, epoch)
            else:
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
    def reduce(self, p: MlpParams, col, epoch):
        """Output column ``col``, by one lane of warp 0: the fp32 sum of the slots' bf16 down
        outputs times their routing weights in slot (ascending expert) order, rounded to bf16
        and written to the convolution ring at the position, then the post-MLP convolution
        ``(conv + y) / sqrt(kernel + 1)`` (bf16) plus the residual."""
        k = self.top_k
        vals = cute.make_rmem_tensor((k,), Float32)
        fetch([(p.down, s * self.hidden + col, 1) for s in range(k)], vals, epoch)
        acc = Float32(0.0)
        for s in cutlass.range_constexpr(k):
            acc = acc + vals[s] * p.route[1, k + s]
        y = bf16_round(acc)
        pos = Int32(p.pos[0])
        p.ring[pos % RING, col] = BFloat16(y)
        taps = self.conv_kernel
        conv = Float32(0.0)
        for j in cutlass.range_constexpr(taps - 1):
            src = pos - (taps - 1) + j  # >= -(taps - 1): positions before 0 read zero
            past = Float32(p.ring[(src + RING) % RING, col])
            conv = conv + p.conv[col, j] * cutlass.select_(src >= 0, past, Float32(0.0))
        conv = conv + p.conv[col, taps - 1] * y
        z = bf16_round((conv + y) * fp32_reciprocal(math.sqrt(taps + 1)))
        p.out[col] = BFloat16(p.residual[col] + z)

    @cute.jit
    def route(self, p: MlpParams, lane, epoch, which: cutlass.Constexpr):
        """One warp's routing into ``route[which]``, as ``ref.route``: the top-k experts of
        ``sigmoid(logit) + bias`` (ties to the lower index) in ascending expert order, then
        their scores renormalized over the top k and scaled by ``route_scale``. Each round
        takes the warp's largest remaining key; the slots come from a scan of the chosen
        counts, so no sort is needed. The logits are polled warp-uniformly
        (``fetch_uniform``): the shuffles that follow need every lane."""
        n = self.experts // 32  # lane l holds experts n*l .. n*l + n - 1
        k = self.top_k
        logits = cute.make_rmem_tensor((n,), Float32)
        fetch_uniform([(p.logits, n * lane, n)], logits, epoch)
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

    # ----------------------------------------------------------------- kernel

    @cute.kernel
    def kernel(self, tma, gtensors, gate_tma, gate_g, scratch, tensors, blocks, bound, num_blocks):
        layouts = {
            "W": cute.make_layout((ROWS, TILE_K, self.stages), stride=(TILE_K, 1, ROWS * TILE_K)),
            "X": cute.make_layout((3, self.hidden), stride=(self.hidden, 1)),
            "RED": cute.make_layout(
                (REDUCTIONS, COMPUTE_WARPS, RED_VALUES),
                stride=(COMPUTE_WARPS * RED_VALUES, RED_VALUES, 1),
            ),
            "MIX": cute.make_layout((MIX_FLOATS,)),
            "ROUTE": cute.make_layout((2, 2 * self.top_k), stride=(2 * self.top_k, 1)),
        }
        self.begin_roles(work=bound, layouts=layouts)
        # Materialize the argument tensors in converged code: a first use inside one role's
        # branch would not dominate the other roles' uses.
        tensors = {k: cute.make_tensor(t.iterator, t.layout) for k, t in zip(TENSOR_ARGS, tensors)}
        trace_buffer = tensors.pop("trace_buffer")
        blocks = tuple(cute.make_tensor(t.iterator, t.layout) for t in blocks)
        offsets = self.scratch_offsets
        tagged = {
            name: cute.make_tensor(scratch.iterator + offsets[name], cute.make_layout((size,)))
            for name, size in self.scratch_shapes.items()
        }
        sW = self.buffer("W")
        parts = []
        for atom, g in zip(tma, gtensors):
            tiles = cute.local_tile(g, (ROWS, TILE_K), (None, None))
            parts.append(
                cpasync.tma_partition(
                    atom,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sW, 0, 2),
                    cute.group_modes(tiles, 0, 2),
                )
            )
        gate = self.gate_params(sW, gate_tma, gate_g, tagged, tensors)
        params = MlpParams(
            tma=tuple(tma),
            gmem=tuple(part[1] for part in parts),
            sW=StageView(parts[0][0], axis=1),
            xs=self.buffer("X"),
            red=self.buffer("RED"),
            mix=self.buffer("MIX"),
            route=self.buffer("ROUTE"),
            gate=gate,
            blocks=blocks,
            num_blocks=num_blocks,
            **tagged,
            **tensors,
        )
        self.run_roles(params, trace_buffer=trace_buffer)

    @cute.jit
    def __call__(
        self,
        router: cute.Tensor,
        w13: cute.Tensor,
        w2: cute.Tensor,
        router_bias: cute.Tensor,
        query: cute.Tensor,
        norms: cute.Tensor,
        conv: cute.Tensor,
        gate_down: cute.Tensor,
        gate_fg: cute.Tensor,
        mhalf: cute.Tensor,
        messages: cute.Tensor,
        residual: cute.Tensor,
        block0: cute.Tensor,
        block1: cute.Tensor,
        block2: cute.Tensor,
        block3: cute.Tensor,
        num_blocks: Int32,
        ring: cute.Tensor,
        pos: cute.Tensor,
        out: cute.Tensor,
        trace_buffer: cute.Tensor,
        scratch: cute.Tensor,
        state: cute.Tensor,
        table: cute.Tensor,
        workspace: cute.Tensor,
        max_ctas: Int32,
        stream: cuda.CUstream,
    ):
        h, width, e = self.hidden, self.width, self.experts
        # Gate/up rows as ((gate | up), row, expert): a 32-row tile holds 16 (gate, up) pairs,
        # interleaved like qwen_megakernel's packed weights.
        gate_up = cute.make_tensor(
            w13.iterator,
            cute.make_layout(((2, width, e), h), stride=((width * h, h, 2 * width * h), 1)),
        )
        down = cute.make_tensor(w2.iterator, cute.make_layout((e * h, width), stride=(width, 1)))
        op = cpasync.CopyBulkTensorTileG2SOp()
        smem = cute.make_layout((ROWS, TILE_K), stride=(TILE_K, 1))
        pairs = [
            cpasync.make_tiled_tma_atom(op, src, smem, (ROWS, TILE_K))
            for src in (router, gate_up, down)
        ]
        gate_pairs = []
        if cutlass.const_expr(self.messages):
            gate_pairs = gate_atoms(gate_down, gate_fg, self.hidden)
        bound = self.work.bind(table, workspace, max_ctas)
        tensors = (trace_buffer, state, pos, residual, query, norms, conv, router_bias, ring, mhalf, messages,
                   out)  # fmt: skip
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            tuple(a for a, _ in pairs),
            tuple(g for _, g in pairs),
            tuple(a for a, _ in gate_pairs),
            tuple(g for _, g in gate_pairs),
            scratch,
            tensors,
            (block0, block1, block2, block3),
            bound,
            num_blocks,
        ).launch(**self.launch_config(bound), stream=stream)


@jit_cache
def compile_mlp_kernel(op: InklingMlpKernel):
    """One compile serves every visit, layer and position: weights and states are arguments,
    the position and block count are read at run time."""
    bf16, f32 = BFloat16, Float32
    h, width, e = op.hidden, op.width, op.experts
    vector = [make_fake_compact_tensor(bf16, (h,), assumed_align=2) for _ in range(MAX_BLOCKS)]
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(bf16, (e, h)),  # router (bf16-exact copy)
        make_fake_compact_tensor(bf16, (e, 2 * width, h)),
        make_fake_compact_tensor(bf16, (e, h, width)),
        make_fake_compact_tensor(f32, (e,), assumed_align=4),
        make_fake_compact_tensor(f32, (1, h), assumed_align=4),  # block-residual query
        make_fake_compact_tensor(bf16, (1, h), assumed_align=2),  # post norm
        make_fake_compact_tensor(f32, (h, op.conv_kernel), assumed_align=4),
        *fake_gate_args(h),
        make_fake_compact_tensor(f32, (h,), assumed_align=4),  # residual
        *vector,
        Int32(1),
        make_fake_compact_tensor(bf16, (RING, h), assumed_align=2),
        make_fake_compact_tensor(Int32, (1,), assumed_align=4),  # pos
        make_fake_compact_tensor(bf16, (h,), assumed_align=2),
        make_fake_compact_tensor(Int64, (cute.sym_int(),)),  # trace buffer (globaltimer only)
        make_fake_compact_tensor(Int64, (op.scratch_size,)),
        make_fake_compact_tensor(Int32, (2,), assumed_align=4),
        make_fake_compact_tensor(Int32, (cute.sym_int(), WORDS)),
        make_fake_compact_tensor(Int32, (cute.sym_int(),)),
        Int32(1),
        name=op.get_name(),
    )
