"""The attention half of one Inkling 2 Mini decode visit as one persistent warpdrive launch
(``InklingAttnKernel``) and its compile function; ``megakernel`` holds the host side.

Batch one, one token. The launch implements ``ref.attention_block`` with ``start=None``
(device position, the whole cache behind its mask)::

    every CTA   prologue: block residual over [blocks..., h] for each of the layer's queries
                (none at layer 0; there ``embed`` makes h the token's normalized embedding
                row), RMSNorm -> bf16 attention inputs (SMEM)
    [messages]  MSGDOWN, GATEMIX of the query, key and value gates (``common``), then each
                CTA gathers the three gated streams and RMS-normalizes them into the inputs
    QKV         32-row tiles of q, k, v, rel, (full layers: dynamic k/v taps), gate, sink
                x hidden                                    -> bf16 projections (tagged)
    ATTN        q head x KV split: q scale/clip, k/v short  -> unnormalized O, (m, l) (tagged)
                conv, k clip, v RMS norm (redundant per
                item), relative-bias scores, flash partial
                over the split's 64-key blocks; the newest
                split adds the new token; one item per kv
                head writes the ring rows, publishes the
                new K/V row
    OPROJ       row tile x quarter of the q heads: merge    -> fp32 partials (tagged)
                the heads' splits with the sink, sigmoid
                gate, then 32 o_proj rows x 768 columns;
                the last quarter's item reduces its 32
                columns: split-order sum, post-attention
                conv (ring write), residual               -> fp32 residual
                (row tile 0's items also store the new
                K/V rows once their heads' items finished)

Sliding layers read the keys in ``(pos - window, pos]`` with static k/v convolutions; full
layers read ``[0, pos]`` with dynamic taps and log scaling of the query and relative state.
Full layers and layers with messages project q, k and v from three input streams. ``full``
and ``messages`` are constructor choices: one compile per layer kind. Scores are rounded to bf16 as
the oracle's (``bf16(bf16(bf16(q.k) * scale) + bf16(rel . rel_proj[:, distance]))``); the
softmax runs in the log2 domain over flash partials, so probabilities are not rounded to bf16
before the value sum as in the oracle: the merge rounds the normalized output once.

Roles, tagged words, the weight ring and the epoch follow ``mlp_kernel`` (and
``qwen_megakernel``): warp 8 ``load`` streams 32 x 256 weight stages and 64 x 128 KV blocks
through one ring, warps 0-7 ``compute`` consume them. The host deals rows round-robin in
dependency order (QKV, ATTN, OPROJ by split) for ``StaticPersistent``; with every CTA resident
the earliest unfinished row has its producers finished. The cache row at the position is
stored only after every item reading the block that holds it has finished; those items mask
the row's stale contents.
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
    GATEMIX,
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
    fp32,
    fp32_reciprocal,
    gate_atoms,
    load_tiles,
    scratch_offsets,
)
from attn_gym.warpdrive.examples.qwen_megakernel.core import (
    BLOCK,
    COMPUTE_THREADS,
    COMPUTE_WARPS,
    HEAD_DIM,
    LOG2E,
    MAX_STAGES,
    ROWS,
    STAGE_BYTES,
    TILE_K,
    CudaCoreGemv,
    InstructionTable,
    bf16_round,
    fetch,
    group_sum,
    load_vec,
    publish,
)

QKV, ATTN, OPROJ = range(3)
# Projection matrices of the QKV rows (their b word), in the order the table deals them: what
# attention reads first, the output gate and sink (read by OPROJ) last.
Q, K, V, REL, KDYN, VDYN, GATE, SINK = range(8)
O_SPLITS = 4  # o_proj split-K: a quarter of the q heads each
QKVS_FLOATS = 512  # an attention item's head vectors (see ``InklingAttnKernel.qkvs_words``)
# AttnParams fields passed to the kernel as one tuple of argument tensors, in this order.
TENSOR_ARGS = (
    "trace_buffer", "state", "pos", "token", "embed", "embed_norm", "h", "query", "norms",
    "q_scale", "rel_scale", "rel_proj", "k_taps", "v_taps", "attn_conv", "ring_k", "ring_v",
    "ring_attn", "k_cache", "v_cache", "mhalf", "messages", "out",
)  # fmt: skip


@dataclass(frozen=True)
class AttnParams:
    tma: tuple  # TMA atoms: projection matrices (Q .. SINK, kernel's order), wo, k and v cache
    gmem: tuple  # their tile partitions: (TMA, row tiles, k tiles)
    sW: StageView  # weight stage partition
    sKV: StageView  # KV stage partition
    xs: cute.Tensor  # (STREAMS + 2, hidden) bf16: attention inputs, then two o_proj inputs
    red: cute.Tensor  # (REDUCTIONS, COMPUTE_WARPS, RED_VALUES) fp32 prologue reductions
    mix: cute.Tensor  # (MIX_FLOATS,) fp32 the prologue's block-residual weights (messages)
    qkvs: cute.Tensor  # (QKVS_FLOATS,) fp32 an attention item's head vectors
    att: cute.Tensor  # (COMPUTE_WARPS + 1, HEAD_DIM + 2) fp32 per-warp flash states
    proj: cute.Tensor  # tagged int64 projections (``proj_offsets``)
    attn_o: cute.Tensor  # tagged int64 (q_heads, splits, HEAD_DIM) unnormalized O
    attn_ml: cute.Tensor  # tagged int64 (q_heads, splits, 2) (m in log2 units, l)
    o_part: cute.Tensor  # tagged int64 (O_SPLITS, hidden) o_proj partials
    kv_new: cute.Tensor  # tagged int64 (2, kv_dim): the new K, then V row of every kv head
    gate: GateParams | None  # message gates (``messages`` kernels)
    state: cute.Tensor  # int32 (2,): launch epoch, finished CTAs
    pos: cute.Tensor  # int32 (1,) position of the token
    token: cute.Tensor  # int64 (1,) the token (read when ``embed``)
    embed: cute.Tensor  # (vocab rows, hidden) bf16 embedding table
    embed_norm: cute.Tensor  # (hidden,) bf16
    h: cute.Tensor  # (hidden,) bf16 visit input; written by CTA 0 when ``embed``
    blocks: tuple  # MAX_BLOCKS (hidden,) bf16 completed blocks; the first num_blocks are read
    num_blocks: Int32
    queries: Int32  # 0: the input is h itself (layer 0), 1: block residuals
    boundary: Int32  # 1: the residual restarts from the attention output
    embed_input: Int32  # 1: h is the token's normalized embedding row (layer 0)
    query: cute.Tensor  # (streams, hidden) fp32 block-residual queries
    norms: cute.Tensor  # (streams, hidden) bf16 input RMSNorm weights
    q_scale: cute.Tensor  # (q_heads, HEAD_DIM) bf16
    rel_scale: cute.Tensor  # (q_heads, d_rel) bf16
    rel_proj: cute.Tensor  # (rel_width, d_rel) bf16: ref's rel_proj transposed, a row per distance
    k_taps: cute.Tensor  # sliding: (kv_dim, kernel) fp32; full: (kernel, kv_dim) bf16 static
    v_taps: cute.Tensor
    attn_conv: cute.Tensor  # (hidden, kernel) fp32 post-attention taps
    ring_k: cute.Tensor  # (RING, kv_dim) bf16 the visit's k/v/attention convolution inputs
    ring_v: cute.Tensor
    ring_attn: cute.Tensor  # (RING, hidden) bf16
    k_cache: cute.Tensor  # (kv_heads * max_len, HEAD_DIM) bf16 the visit's keys
    v_cache: cute.Tensor
    mhalf: cute.Tensor  # (GATES * GATE_RANK,) fp32 message halves (read when ``messages``)
    messages: cute.Tensor  # (GATES, hidden) fp32
    out: cute.Tensor  # (hidden,) fp32 post-attention residual


@cute.jit
def head_vec(src, base, d0):
    """This lane's dims d0..d0+7 and 64+d0..64+d0+7 of a head starting at ``base``."""
    lo = load_vec(src, base + d0, 8)
    hi = load_vec(src, base + 64 + d0, 8)
    out = cute.make_rmem_tensor((16,), Float32)
    for e in cutlass.range_constexpr(8):
        out[e] = lo[e]
        out[8 + e] = hi[e]
    return out


@cute.jit
def stage_row(stage, t, d0):
    """Dims d0..d0+7 and 64+d0..64+d0+7 of position ``t`` of a KV stage, as fp32."""
    out = cute.make_rmem_tensor((16,), Float32)
    for half in cutlass.range_constexpr(2):
        frag = cute.make_rmem_tensor((1, 8), BFloat16)
        cute.autovec_copy(cute.local_tile(stage, (1, 8), (t, d0 // 8 + 8 * half)), frag)
        vals = frag.load().to(Float32)
        for e in cutlass.range_constexpr(8):
            out[8 * half + e] = vals[0, e]
    return out


@cute.jit
def clip_factor(ss, threshold: cutlass.Constexpr):
    """``ref.l2_clip``'s scale for a vector with squared norm ``ss``: PyTorch evaluates
    ``threshold / (norm + 1e-6)`` as ``reciprocal(norm + 1e-6) * threshold``."""
    scale = Float32(1.0) / (cute.math.sqrt(ss) + Float32(1e-6)) * Float32(threshold)
    return cutlass.min(scale, Float32(1.0))


class InklingAttnKernel(InklingKernel):
    """One visit's attention half (see the module docstring). Constructor arguments are the
    model sizes and constants the kernel reads. ``full``: a full-attention layer (three
    block-residual queries, dynamic k/v taps, log scaling ``log_alpha``/``log_floor``), else a
    sliding one (``window`` keys). ``messages``: a core visit after the first pass, whose q, k
    and v inputs pass through message gates. ``extent``: distances with a relative bias;
    ``rel_width``: columns of ``rel_proj``. ``splits``: KV splits per q head (static; pick for
    the window or the contexts decoded). ``stages``: ring depth in 16 KB stages."""

    def __init__(
        self,
        hidden: int,
        q_heads: int,
        kv_heads: int,
        d_rel: int,
        rel_width: int,
        extent: int,
        conv_kernel: int,
        rms_eps: float,
        qk_clip: float,
        rel_clip: float,
        splits: int,
        full: bool,
        window: int = 0,
        log_alpha: float = 0.0,
        log_floor: int = 0,
        messages: bool = False,
        stages: int | None = None,
    ):
        if hidden % COMPUTE_THREADS or hidden % TILE_K or hidden % ROWS:
            raise ValueError(f"hidden size must be a multiple of {COMPUTE_THREADS}")
        if q_heads % kv_heads or q_heads % O_SPLITS:
            raise ValueError(f"q heads must be a multiple of kv heads and of {O_SPLITS}")
        if q_heads // O_SPLITS > COMPUTE_WARPS or (q_heads // O_SPLITS * HEAD_DIM) % TILE_K:
            raise ValueError("an o_proj split merges one head per warp in whole 256-column tiles")
        if (q_heads // O_SPLITS) % (q_heads // kv_heads):
            raise ValueError("an o_proj split must cover whole groups of q heads")
        if d_rel % 8 or d_rel > 32 or not 0 < extent <= min(rel_width, hidden):
            raise ValueError("need d_rel a multiple of 8 up to 32 and extent <= rel_width, hidden")
        if full and conv_kernel % 2:
            raise ValueError("full layers fetch dynamic taps in pairs: need an even kernel")
        if not 2 <= conv_kernel <= RING or not 1 <= splits <= 32:
            raise ValueError("need a kernel-2..RING convolution and 1 to 32 splits")
        if full != (window == 0) or full != (log_floor > 0):
            raise ValueError("full layers take log scaling and no window; sliding the reverse")
        self.hidden, self.q_heads, self.kv_heads = hidden, q_heads, kv_heads
        self.group = q_heads // kv_heads
        self.d_rel, self.rel_width, self.extent = d_rel, rel_width, extent
        self.conv_kernel, self.rms_eps = conv_kernel, rms_eps
        self.qk_clip, self.rel_clip = qk_clip, rel_clip
        self.splits, self.full, self.window = splits, full, window
        self.log_alpha, self.log_floor = log_alpha, log_floor
        self.messages = messages
        self.queries = 3 if full else 1  # block-residual queries of the prologue
        self.streams = 3 if full or messages else 1  # q, k, v input streams
        self.first_gate = 0
        self.gate_query = tuple(range(3)) if full else (0, 0, 0)
        self.kv_dim = kv_heads * HEAD_DIM
        self.heads_per_split = q_heads // O_SPLITS
        self.hidden_k_tiles = hidden // TILE_K
        self.oproj_k_tiles = self.heads_per_split * HEAD_DIM // TILE_K
        self.row_tiles = hidden // ROWS
        if self.qkvs_words > QKVS_FLOATS:
            raise ValueError("an attention item's head vectors exceed QKVS_FLOATS")
        rows = {Q: q_heads * HEAD_DIM, K: self.kv_dim, V: self.kv_dim, REL: q_heads * d_rel,
                GATE: q_heads * HEAD_DIM, SINK: q_heads}  # fmt: skip
        if full:
            rows.update({KDYN: kv_heads * conv_kernel, VDYN: kv_heads * conv_kernel})
        self.matrices = tuple(sorted(rows))
        self.proj_rows = rows
        self.proj_tiles = {m: -(-r // ROWS) for m, r in rows.items()}
        offsets, offset = {}, 0
        for m in self.matrices:
            offsets[m] = offset
            offset += self.proj_tiles[m] * ROWS
        self.proj_offsets, self.proj_size = offsets, offset
        fixed = (
            2 * (self.streams + 2) * hidden
            + 4 * (REDUCTIONS * COMPUTE_WARPS * RED_VALUES + MIX_FLOATS + QKVS_FLOATS)
            + 4 * (COMPUTE_WARPS + 1) * (HEAD_DIM + 2)
            + SMEM_HEADROOM
        )
        fit = (SMEM_BYTES - fixed) // STAGE_BYTES
        stages = min(MAX_STAGES, fit) if stages is None else stages
        if not 2 <= stages <= fit:
            raise ValueError(f"stages must be in [2, {fit}], got {stages}")
        self.stages = stages
        self.smem_plan = SmemPlan()
        w = self.smem_plan.buffer("W", BFloat16, align=1024)
        kv = self.smem_plan.buffer("KV", BFloat16, align=1024, overlaps=w)
        self.smem_plan.buffer("X", BFloat16, align=16)
        self.smem_plan.buffer("RED", Float32, align=16)
        self.smem_plan.buffer("MIX", Float32, align=16)
        self.smem_plan.buffer("QKVS", Float32, align=16)
        self.smem_plan.buffer("ATT", Float32, align=16)
        # Compute reads stages with generic loads before TMA refills them; the layer emits the
        # fence.proxy.async on a Warps release of a Tma-filled ring.
        self.w = Ring(stages, Tma(STAGE_BYTES), Warps, buffer={"W": w, "KV": kv})
        self.csync = NamedBarrier(threads=COMPUTE_THREADS)
        self.engine = CudaCoreGemv()
        self.work = WorkStream(mapping=InstructionTable(words=WORDS), schedule=StaticPersistent())

    @property
    def qkvs_words(self) -> int:
        """An attention item's head vectors in QKVS: q, k, v (HEAD_DIM each), the relative
        state (d_rel), then in full layers the dynamic k and v taps (kernel each)."""
        return 3 * HEAD_DIM + self.d_rel + (2 * self.conv_kernel if self.full else 0)

    @property
    def scratch_shapes(self) -> dict[str, int]:
        """Tagged int64 words of each per-launch intermediate."""
        shapes = {
            "proj": self.proj_size,
            "attn_o": self.q_heads * self.splits * HEAD_DIM,
            "attn_ml": self.q_heads * self.splits * 2,
            "o_part": O_SPLITS * self.hidden,
            "kv_new": 2 * self.kv_dim,
        }
        return shapes | self.gate_scratch_shapes()

    @property
    def scratch_offsets(self) -> dict[str, int]:
        return scratch_offsets(self.scratch_shapes)[0]

    @property
    def scratch_size(self) -> int:
        return scratch_offsets(self.scratch_shapes)[1]

    def partial(self, p: AttnParams):
        return p.h

    def get_name(self) -> str:
        kind = ("full" if self.full else f"swa{self.window}") + ("_msg" if self.messages else "")
        return (
            f"warpdrive_inkling_attn_{kind}_h{self.hidden}_q{self.q_heads}kv{self.kv_heads}"
            f"_kv{self.splits}_s{self.stages}{self.trace_key()}"
        )

    # ------------------------------------------------------------------ roles

    @role(warps=COMPUTE_WARPS)
    @cute.jit
    def load(self, p: AttnParams, *, w: Producer, work: WorkCursor):
        w = w.views(W=p.sW, KV=p.sKV)
        count = len(self.matrices)
        tma_o, tma_k, tma_v = p.tma[count], p.tma[count + 1], p.tma[count + 2]
        g_o, g_k, g_v = p.gmem[count], p.gmem[count + 1], p.gmem[count + 2]
        pos = Int32(p.pos[0])
        max_len = p.k_cache.shape[0] // self.kv_heads
        while work.valid:
            op, a, b, _row = work.item
            if op == QKV:
                for i in cutlass.range_constexpr(count):
                    if b == self.matrices[i]:
                        w = load_tiles(w, p.tma[i], p.gmem[i], a, 0, self.hidden_k_tiles, "W")
            elif op == ATTN:
                blk0, blk1, _newest = self.split_blocks(pos, b)
                row0 = (a // self.group) * (max_len // BLOCK)
                for blk in cutlass.range(blk0, blk1, unroll=1):
                    w = load_tiles(w, tma_k, g_k, row0 + blk, 0, 1, "KV")
                    w = load_tiles(w, tma_v, g_v, row0 + blk, 0, 1, "KV")
            elif op == OPROJ:
                k_tiles = self.oproj_k_tiles
                w = load_tiles(w, tma_o, g_o, a, b * k_tiles, k_tiles, "W")
            if cutlass.const_expr(self.messages):
                if op >= MSGDOWN:
                    w = self.load_gate_item(p, w, op, a, b)
            work.advance()
        return w, work

    @role(warps=range(COMPUTE_WARPS))
    @cute.jit
    def compute(self, p: AttnParams, *, w: Consumer, work: WorkCursor, me: Local):
        tid, warp, lane = me.tid, me.warp, me.lane
        epoch = p.state[0] + 1
        pos = Int32(p.pos[0])
        with self.trace_region("normalize"):
            x = self.load_states(p, p.h, tid)
            if p.embed_input != 0:
                self.embed_token(p, x, tid, warp, lane)
            self.normalize_states(p, x, p.query, self.queries, p.queries, tid, warp, lane)
        obuf = Int32(self.streams)
        gathered = Int32(0)
        while work.valid:
            op, a, b, item = work.item
            if cutlass.const_expr(self.messages):
                if op == MSGDOWN:
                    w = self.msgdown(p, w, a, b, epoch, warp, lane)
                elif op == GATEMIX:
                    w = self.gatemix(p, w, a, b, epoch, warp, lane)
                elif gathered == 0:
                    with self.trace_region("gather", payload=item):
                        self.gather(p, epoch, tid, warp, lane)
                    gathered = Int32(1)
            if op == QKV:
                vals, row, chans = self.engine.tile(
                    p, (w,), (), self.stream_of(b), self.hidden_k_tiles, warp, lane
                )
                w = chans[0]
                if self.engine.owns_row(warp, lane):
                    index = self.proj_offset(b) + a * ROWS + row
                    publish(p.proj, index, bf16_round(vals[0]), epoch)
            elif op == ATTN:
                blk0, blk1, newest = self.split_blocks(pos, b)
                if (blk0 == blk1) & (b != newest):
                    self.publish_empty_partial(p, a, b, epoch, tid)
                else:
                    w = self.attention(p, w, a, b, pos, epoch, tid, warp, lane, item)
            elif op == OPROJ:
                # Alternate o_proj input buffers: every warp passed the last merge's barrier
                # after finishing the row before it, the last reader of this buffer.
                obuf = 2 * self.streams + 1 - obuf
                with self.trace_region("merge", payload=item):
                    if warp < self.heads_per_split:
                        hq = b * self.heads_per_split + warp
                        self.merge_head(p, hq, warp, obuf, epoch, lane)
                    self.csync.sync()
                if a == 0:
                    self.store_kv_rows(p, b, pos, epoch, tid)
                vals, row, chans = self.engine.tile(
                    p, (w,), (), obuf, self.oproj_k_tiles, warp, lane
                )
                w = chans[0]
                if self.engine.owns_row(warp, lane):
                    publish(p.o_part, b * self.hidden + a * ROWS + row, vals[0], epoch)
                if (b == O_SPLITS - 1) & (warp == 0):
                    with self.trace_region("reduce", payload=item):
                        self.reduce(p, a * ROWS + lane, pos, epoch)
            work.advance()
        self.advance_epoch(p, tid, epoch)
        return w, work

    # ------------------------------------------------------------ compute helpers

    @cute.jit
    def embed_token(self, p: AttnParams, x, tid, warp, lane):
        """The partial state ``x[MAX_BLOCKS]`` (``load_states`` read a stale ``h``) becomes the
        token's embedding row through the weighted RMSNorm (``ref.hidden_states``' first ``h``),
        which CTA 0 also stores to ``h`` for the launches after this one."""
        per = self.hidden // COMPUTE_THREADS
        row = Int32(p.token[0])
        ss = Float32(0.0)
        for r in cutlass.range_constexpr(per):
            v = Float32(p.embed[row, r * COMPUTE_THREADS + tid])
            x[MAX_BLOCKS, r] = v
            ss = ss + v * v
        ss = self.cta_sum(p, [ss], 0, warp, lane)[0]
        rinv = cute.math.rsqrt(ss * fp32_reciprocal(self.hidden) + Float32(self.rms_eps))
        first = cute.arch.block_idx()[0] == 0
        for r in cutlass.range_constexpr(per):
            k = r * COMPUTE_THREADS + tid
            v = BFloat16(Float32(p.embed_norm[k]) * (x[MAX_BLOCKS, r] * rinv))
            x[MAX_BLOCKS, r] = Float32(v)
            if first:
                p.h[k] = v
        # The prologue's reductions reuse RED slot 0.
        self.csync.sync()

    @cute.jit
    def stream_of(self, matrix):
        """The input stream (xs row) of a projection: with three streams, k (and its dynamic
        taps) come from stream 1, v from stream 2, the rest from stream 0 (the query input)."""
        stream = Int32(0)
        if cutlass.const_expr(self.streams == 3):
            stream = cutlass.select_((matrix == K) | (matrix == KDYN), Int32(1), stream)
            stream = cutlass.select_((matrix == V) | (matrix == VDYN), Int32(2), stream)
        return stream

    @cute.jit
    def proj_offset(self, matrix):
        offset = Int32(0)
        for m in self.matrices:
            offset = cutlass.select_(matrix == m, Int32(self.proj_offsets[m]), offset)
        return offset

    @cute.jit
    def split_blocks(self, pos, split):
        """``(blk0, blk1, newest)``: the KV blocks ``[blk0, blk1)`` of cached keys that
        ``split`` reads (positions ``[0, pos)``, sliding layers only the last ``window - 1``),
        and the split of the last such block, which also scores the new token at ``pos``."""
        lo = Int32(0)
        if cutlass.const_expr(not self.full):
            lo = cutlass.max(pos - (self.window - 1), Int32(0))
        first = lo // BLOCK
        last = (pos + BLOCK - 1) // BLOCK
        n = last - first
        per = cutlass.max((n + self.splits - 1) // self.splits, Int32(1))
        blk0 = first + split * per
        blk1 = cutlass.max(cutlass.min(blk0 + per, last), blk0)
        newest = cutlass.max(n - 1, Int32(0)) // per
        return blk0, blk1, newest

    @cute.jit
    def log_scale(self, pos):
        """``ref.log_scale`` in fp32 (full layers): ``1 + alpha * log(max((pos + 1) / floor,
        1))``."""
        ratio = Float32(pos + 1) * fp32_reciprocal(self.log_floor)
        return Float32(1.0) + Float32(self.log_alpha) * cute.math.log(
            cutlass.max(ratio, Float32(1.0))
        )

    @cute.jit
    def qkvs_column(self, hq, kvh, word):
        """The proj index of word ``word`` of an item's QKVS (``qkvs_words``)."""
        d, r, c = HEAD_DIM, self.d_rel, self.conv_kernel
        off = self.proj_offsets
        column = off[Q] + hq * d + word
        column = cutlass.select_(word >= d, off[K] + kvh * d + word - d, column)
        column = cutlass.select_(word >= 2 * d, off[V] + kvh * d + word - 2 * d, column)
        column = cutlass.select_(word >= 3 * d, off[REL] + hq * r + word - 3 * d, column)
        if cutlass.const_expr(self.full):
            base = 3 * d + r
            column = cutlass.select_(word >= base, off[KDYN] + kvh * c + word - base, column)
            base = base + c
            column = cutlass.select_(word >= base, off[VDYN] + kvh * c + word - base, column)
        return column

    @cute.jit
    def distances(self, pos, blk0, blk1, newest):
        """``(first, last)``: the distances ``pos - key`` with a relative bias that the split's
        keys (and the new token, at distance 0, in the newest split) can have."""
        first = cutlass.max(pos - blk1 * BLOCK + 1, Int32(0))
        first = cutlass.select_(newest, Int32(0), first)
        last = cutlass.min(pos - blk0 * BLOCK, Int32(self.extent - 1))
        return first, last

    @cute.jit
    def prepare_heads(
        self, p: AttnParams, hq, kvh, pos, writer, first, last, epoch, tid, warp, lane
    ):
        """Fetch the item's projections into QKVS, then in place (one warp each): the q head's
        scale, L2 clip (and log scaling), the kv head's k and v convolutions (writing their
        ring rows when ``writer``), k's L2 clip and v's RMS norm, and the relative state's
        scale and clip (and log scaling). Then every thread builds the relative bias of
        distances ``[first, last]`` in BIAS (``score``). Every value is rounded to bf16 as the
        oracle's.

        The weights, ring rows and ``rel_proj`` rows each thread needs load before the poll and
        are used after it, so their latency overlaps the wait for QKV (this launch writes ring
        row ``pos`` only, after reading the earlier ones)."""
        kernel, r = self.conv_kernel, self.d_rel
        rounds = -(-self.extent // COMPUTE_THREADS)
        d0 = 4 * lane  # warps 0-2: this lane's 4 dims of the head
        scales = cute.make_rmem_tensor((4,), BFloat16)
        taps = cute.make_rmem_tensor((4, kernel), p.k_taps.element_type)
        past = cute.make_rmem_tensor((4, kernel - 1), BFloat16)
        rel_proj = cute.make_rmem_tensor((rounds, r), BFloat16)
        if warp == 0:
            for e in cutlass.range_constexpr(4):
                scales[e] = p.q_scale[hq, d0 + e]
        elif warp == 1:
            self.load_conv(p.ring_k, p.k_taps, kvh, pos, lane, taps, past)
        elif warp == 2:
            self.load_conv(p.ring_v, p.v_taps, kvh, pos, lane, taps, past)
        elif warp == 3:
            if lane < r:
                scales[0] = p.rel_scale[hq, lane]
        for i in cutlass.range_constexpr(rounds):
            d = first + i * COMPUTE_THREADS + tid
            if d <= last:
                cute.autovec_copy(p.rel_proj[d, None], rel_proj[i, None])
        words = self.qkvs_words
        # Thread t fetches words 2t, 2t + 1; threads past them refetch the last pair.
        word = cutlass.min(2 * tid, words - 2)
        raw = cute.make_rmem_tensor((2,), Float32)
        fetch([(p.proj, self.qkvs_column(hq, kvh, word), 2)], raw, epoch)
        if 2 * tid < words:
            p.qkvs[word] = raw[0]
            p.qkvs[word + 1] = raw[1]
        self.csync.sync()
        ls = Float32(1.0)
        if cutlass.const_expr(self.full):
            ls = self.log_scale(pos)
        if warp == 0:
            x = cute.make_rmem_tensor((4,), Float32)
            ss = Float32(0.0)
            for e in cutlass.range_constexpr(4):
                x[e] = p.qkvs[d0 + e] * Float32(scales[e])
                ss = ss + x[e] * x[e]
            factor = clip_factor(group_sum(ss, 32), self.qk_clip)
            for e in cutlass.range_constexpr(4):
                v = bf16_round(x[e] * factor)
                if cutlass.const_expr(self.full):
                    v = bf16_round(v * ls)
                p.qkvs[d0 + e] = v
        elif warp == 1:
            self.conv_head(p, p.ring_k, HEAD_DIM, 3 * HEAD_DIM + r, kvh, pos, writer, lane,
                           taps, past, True)  # fmt: skip
        elif warp == 2:
            self.conv_head(p, p.ring_v, 2 * HEAD_DIM, 3 * HEAD_DIM + r + kernel, kvh, pos,
                           writer, lane, taps, past, False)  # fmt: skip
        elif warp == 3:
            x = Float32(0.0)
            if lane < r:
                x = p.qkvs[3 * HEAD_DIM + lane] * Float32(scales[0])
            factor = clip_factor(group_sum(x * x, 32), self.rel_clip)
            v = bf16_round(x * factor)
            if cutlass.const_expr(self.full):
                v = bf16_round(v * ls)
            if lane < r:
                p.qkvs[3 * HEAD_DIM + lane] = v
        self.csync.sync()
        rel = cute.make_rmem_tensor((r,), Float32)
        for k in cutlass.range_constexpr(r):
            rel[k] = p.qkvs[3 * HEAD_DIM + k]
        bias = self.bias_table(p)
        for i in cutlass.range_constexpr(rounds):
            d = first + i * COMPUTE_THREADS + tid
            if d <= last:
                acc = Float32(0.0)
                for k in cutlass.range_constexpr(r):
                    acc = acc + rel[k] * Float32(rel_proj[i, k])
                bias[d] = BFloat16(acc)
        self.csync.sync()

    def bias_table(self, p: AttnParams):
        """BIAS: the bf16 relative bias by distance (``extent`` entries). It reuses the first
        o_proj input row of X: every CTA runs its ATTN rows before its OPROJ rows."""
        return p.xs[self.streams, None]

    @cute.jit
    def load_conv(self, ring, taps, kvh, pos, lane, tap, past):
        """One lane's 4 channels of convolution taps (``conv_head``'s layout) and ring rows
        ``pos - 1 - t`` (``past[:, t]``; rows before position 0 are read but unused)."""
        kernel = self.conv_kernel
        for e in cutlass.range_constexpr(4):
            c = kvh * HEAD_DIM + 4 * lane + e
            for t in cutlass.range_constexpr(kernel):
                if cutlass.const_expr(self.full):
                    tap[e, t] = taps[t, c]
                else:
                    tap[e, t] = taps[c, t]
            for t in cutlass.range_constexpr(kernel - 1):
                past[e, t] = ring[(pos - 1 - t + RING) % RING, c]

    @cute.jit
    def conv_head(self, p: AttnParams, ring, base, dyn, kvh, pos, writer, lane, tap, past,
                  is_key: cutlass.Constexpr):  # fmt: skip
        """One warp: the k (``is_key``: then L2 clip) or v (then unweighted RMS norm) head at
        QKVS ``base`` through its short convolution, lane l on dims 4l .. 4l + 3, from
        ``load_conv``'s taps and ring rows. Sliding layers: ``ref.short_conv`` with static
        (channel, kernel) taps, the last on the current token. Full layers:
        ``ref.dynamic_conv``, tap t (t back) is the static (kernel, channel) weight plus the
        head's dynamic tap at QKVS ``dyn + t``."""
        taps_n = self.conv_kernel
        z = cute.make_rmem_tensor((4,), Float32)
        ss = Float32(0.0)
        for e in cutlass.range_constexpr(4):
            y = p.qkvs[base + 4 * lane + e]
            hist = [y]  # hist[t]: the input t positions back, zero before position 0
            for t in cutlass.range_constexpr(1, taps_n):
                hist.append(cutlass.select_(pos - t >= 0, Float32(past[e, t - 1]), Float32(0.0)))
            conv = Float32(0.0)
            if cutlass.const_expr(self.full):
                for t in cutlass.range_constexpr(taps_n):
                    conv = conv + hist[t] * (Float32(tap[e, t]) + p.qkvs[dyn + t])
                out = bf16_round((conv + y) * fp32_reciprocal(math.sqrt(2 * taps_n + 1)))
            else:
                for t in cutlass.range_constexpr(taps_n):
                    conv = conv + tap[e, t] * hist[taps_n - 1 - t]
                out = bf16_round((conv + y) * fp32_reciprocal(math.sqrt(taps_n + 1)))
            if writer:
                ring[pos % RING, kvh * HEAD_DIM + 4 * lane + e] = BFloat16(y)
            z[e] = out
            ss = ss + out * out
        ss = group_sum(ss, 32)
        if cutlass.const_expr(is_key):
            factor = clip_factor(ss, self.qk_clip)
        else:
            factor = cute.math.rsqrt(ss * fp32_reciprocal(HEAD_DIM) + Float32(self.rms_eps))
        for e in cutlass.range_constexpr(4):
            p.qkvs[base + 4 * lane + e] = bf16_round(z[e] * factor)

    @cute.jit
    def score(self, p: AttnParams, dot, distance):
        """A key's softmax logit in log2 units from its 8 lanes' partial q.k ``dot``: the bf16
        score plus the relative bias at ``distance`` (BIAS; zero past the extent), rounded to
        bf16 as ``ref.attention``."""
        index = cutlass.min(cutlass.max(distance, Int32(0)), Int32(self.extent - 1))
        bias = Float32(self.bias_table(p)[index])
        bias = cutlass.select_(distance < self.extent, bias, Float32(0.0))
        s = bf16_round(bf16_round(group_sum(dot, 8)) * fp32(HEAD_DIM**-0.5))
        return bf16_round(s + bias) * LOG2E

    @cute.jit
    def attention(self, p: AttnParams, w, hq, split, pos, epoch, tid, warp, lane, item):
        """Flash-decoding partial of q head ``hq`` over KV split ``split``.

        Warp w scores positions 8w + 4i + g (i = 0, 1) of each 64-position block: lane group
        g = lane // 8 holds a position, lane j = lane % 8 its dims 8j..8j+7 and 64+8j..64+8j+7
        and relative-state entries 2j, 2j + 1. Each warp keeps its own online-softmax state; the
        warps, plus the new token in the newest split, merge through ATT at the end."""
        g = lane // 8
        j = lane % 8
        d0 = 8 * j
        kvh = hq // self.group
        blk0, blk1, newest = self.split_blocks(pos, split)
        is_newest = split == newest
        writer = is_newest & (hq % self.group == 0)  # one item per kv head
        first, last = self.distances(pos, blk0, blk1, is_newest)
        with self.trace_region("attn.prepare", payload=item):
            self.prepare_heads(p, hq, kvh, pos, writer, first, last, epoch, tid, warp, lane)
        q = head_vec(p.qkvs, 0, d0)
        lo = Int32(0)
        if cutlass.const_expr(not self.full):
            lo = pos - (self.window - 1)
        m = Float32(-1e30)  # running max (log2 units), warp-uniform
        lsum = Float32(0.0)  # this lane group's sum of p
        o = cute.make_rmem_tensor((16,), Float32)  # this lane group's sum of p * v
        o.fill(0.0)
        with self.trace_region("attn.blocks", payload=item):
            for blk in cutlass.range(blk0, blk1, unroll=1):
                sc = cute.make_rmem_tensor((2,), Float32)
                with w.consume() as h:
                    sK = h.view("KV")
                    for i in cutlass.range_constexpr(2):
                        pt = 8 * warp + 4 * i + g
                        kp = blk * BLOCK + pt
                        kv = stage_row(sK, pt, d0)
                        dot = Float32(0.0)
                        for e in cutlass.range_constexpr(16):
                            dot = dot + q[e] * kv[e]
                        s = self.score(p, dot, pos - kp)
                        sc[i] = cutlass.select_((kp < pos) & (kp >= lo), s, -Float32.inf)
                bm = cutlass.max(sc[0], sc[1])
                bm = cutlass.max(bm, cute.arch.shuffle_sync_bfly(bm, offset=8))
                bm = cutlass.max(bm, cute.arch.shuffle_sync_bfly(bm, offset=16))
                m_new = cutlass.max(m, bm)
                alpha = cute.math.exp2(m - m_new)
                m = m_new
                pr0 = cute.math.exp2(sc[0] - m_new)
                pr1 = cute.math.exp2(sc[1] - m_new)
                lsum = lsum * alpha + pr0 + pr1
                for e in cutlass.range_constexpr(16):
                    o[e] = o[e] * alpha
                with w.consume() as h:
                    sV = h.view("KV")
                    for i in cutlass.range_constexpr(2):
                        pt = 8 * warp + 4 * i + g
                        kp = blk * BLOCK + pt
                        if (kp < pos) & (kp >= lo):  # masked: p is 0, but a stale v may be NaN
                            vv = stage_row(sV, pt, d0)
                            pr = pr0 if i == 0 else pr1
                            for e in cutlass.range_constexpr(16):
                                o[e] = o[e] + pr * vv[e]
        # The new K/V row goes to the cache only after every item reading the block that holds
        # its position finished (``store_kv_rows``): the writer publishes it.
        if writer & (tid < HEAD_DIM):
            publish(p.kv_new, kvh * HEAD_DIM + tid, p.qkvs[HEAD_DIM + tid], epoch)
            publish(
                p.kv_new, self.kv_dim + kvh * HEAD_DIM + tid, p.qkvs[2 * HEAD_DIM + tid], epoch
            )
        # Merge the four lane groups (they share m), then publish this warp's state to ATT.
        lsum = lsum + cute.arch.shuffle_sync_bfly(lsum, offset=8)
        lsum = lsum + cute.arch.shuffle_sync_bfly(lsum, offset=16)
        for e in cutlass.range_constexpr(16):
            o[e] = o[e] + cute.arch.shuffle_sync_bfly(o[e], offset=8)
            o[e] = o[e] + cute.arch.shuffle_sync_bfly(o[e], offset=16)
        att = p.att
        if g == 0:
            for e in cutlass.range_constexpr(8):
                att[warp, d0 + e] = o[e]
                att[warp, d0 + 64 + e] = o[8 + e]
            if j == 0:
                att[warp, HEAD_DIM] = m
                att[warp, HEAD_DIM + 1] = lsum
        if is_newest:
            if warp == 0:  # entry COMPUTE_WARPS is the new token: m = its score, l = 1, O = v
                k_new = head_vec(p.qkvs, HEAD_DIM, d0)
                v_new = head_vec(p.qkvs, 2 * HEAD_DIM, d0)
                dot = Float32(0.0)
                for e in cutlass.range_constexpr(16):
                    dot = dot + q[e] * k_new[e]
                s = self.score(p, dot, Int32(0))
                if g == 0:
                    for e in cutlass.range_constexpr(8):
                        att[COMPUTE_WARPS, d0 + e] = v_new[e]
                        att[COMPUTE_WARPS, d0 + 64 + e] = v_new[8 + e]
                    if j == 0:
                        att[COMPUTE_WARPS, HEAD_DIM] = s
                        att[COMPUTE_WARPS, HEAD_DIM + 1] = Float32(1.0)
        self.csync.sync()
        if tid < HEAD_DIM:
            self.merge_partials(p, hq, split, is_newest, tid, epoch)
        # The next attention row rewrites ATT and QKVS.
        self.csync.sync()
        return w

    @cute.jit
    def merge_partials(self, p: AttnParams, hq, split, newest, d, epoch):
        """Merge the warps' online-softmax states in ATT (plus the new token's in the newest
        split) into this split's partial O (dim ``d``), m and l, and publish them."""
        att = p.att
        mx = Float32(-1e30)
        for e in cutlass.range_constexpr(COMPUTE_WARPS):
            mx = cutlass.max(mx, att[e, HEAD_DIM])
        if newest:
            mx = cutlass.max(mx, att[COMPUTE_WARPS, HEAD_DIM])
        num = Float32(0.0)
        den = Float32(0.0)
        for e in cutlass.range_constexpr(COMPUTE_WARPS):
            scale = cute.math.exp2(att[e, HEAD_DIM] - mx)
            num = num + att[e, d] * scale
            den = den + att[e, HEAD_DIM + 1] * scale
        if newest:
            scale = cute.math.exp2(att[COMPUTE_WARPS, HEAD_DIM] - mx)
            num = num + att[COMPUTE_WARPS, d] * scale
            den = den + att[COMPUTE_WARPS, HEAD_DIM + 1] * scale
        item = hq * self.splits + split
        publish(p.attn_o, item * HEAD_DIM + d, num, epoch)
        if d == 0:
            publish(p.attn_ml, 2 * item, mx, epoch)
            publish(p.attn_ml, 2 * item + 1, den, epoch)

    @cute.jit
    def publish_empty_partial(self, p: AttnParams, hq, split, epoch, tid):
        """A split with no KV blocks at this position: m = -1e30, l = 0 (zero weight in the
        merge) and O = 0, published at once without waiting for QKV."""
        item = hq * self.splits + split
        if tid < HEAD_DIM:
            publish(p.attn_o, item * HEAD_DIM + tid, Float32(0.0), epoch)
        if tid == 0:
            publish(p.attn_ml, 2 * item, Float32(-1e30), epoch)
            publish(p.attn_ml, 2 * item + 1, Float32(0.0), epoch)

    @cute.jit
    def store_kv_rows(self, p: AttnParams, split, pos, epoch, tid):
        """Cache rows ``pos`` of the kv heads whose q heads o_proj split ``split`` covers, from
        their writers' published K/V, by row tile 0's item of the split after its merge: every
        ATTN item of those heads has then published its partial, after its last KV load, so no
        TMA read of the block holding ``pos`` is in flight."""
        heads = self.heads_per_split // self.group
        max_len = p.k_cache.shape[0] // self.kv_heads
        for i in cutlass.range_constexpr(-(-heads * HEAD_DIM // COMPUTE_THREADS)):
            index = i * COMPUTE_THREADS + tid
            if index < heads * HEAD_DIM:
                kvh = split * heads + index // HEAD_DIM
                d = index % HEAD_DIM
                vals = cute.make_rmem_tensor((2,), Float32)
                column = kvh * HEAD_DIM + d
                fetch([(p.kv_new, column, 1), (p.kv_new, self.kv_dim + column, 1)], vals, epoch)
                p.k_cache[kvh * max_len + pos, d] = BFloat16(vals[0])
                p.v_cache[kvh * max_len + pos, d] = BFloat16(vals[1])

    @cute.jit
    def merge_head(self, p: AttnParams, hq, slot, obuf, epoch, lane):
        """One warp: q head ``hq``'s o_proj input at head slot ``slot`` of ``xs[obuf]``, lane l
        on dims 4l .. 4l + 3. The splits' partials merge with the sink logit in the softmax
        denominator; the bf16 output times the sigmoid gate is rounded to bf16, as
        ``ref.attention``. Lane l also fetches split l % splits' (m, l), shared by shuffles,
        so the words are polled warp-uniformly."""
        splits = self.splits
        d = 4 * lane
        segments = [(p.attn_ml, 2 * (hq * splits + lane % splits), 2)]
        for sp in cutlass.range_constexpr(splits):
            segments.append((p.attn_o, (hq * splits + sp) * HEAD_DIM + d, 4))
        segments.append((p.proj, self.proj_offsets[GATE] + hq * HEAD_DIM + d, 4))
        segments.append((p.proj, self.proj_offsets[SINK] + hq, 1))
        vals = cute.make_rmem_tensor((2 + 4 * splits + 5,), Float32)
        fetch_uniform(segments, vals, epoch)
        m = vals[2 + 4 * splits + 4] * LOG2E  # the sink
        ms = cute.make_rmem_tensor((splits,), Float32)
        for sp in cutlass.range_constexpr(splits):
            ms[sp] = cute.arch.shuffle_sync(vals[0], offset=sp)
            m = cutlass.max(m, ms[sp])
        den = cute.math.exp2(vals[2 + 4 * splits + 4] * LOG2E - m)
        for sp in cutlass.range_constexpr(splits):
            ms[sp] = cute.math.exp2(ms[sp] - m)  # split sp's scale
            den = den + cute.arch.shuffle_sync(vals[1], offset=sp) * ms[sp]
        for e in cutlass.range_constexpr(4):
            num = Float32(0.0)
            for sp in cutlass.range_constexpr(splits):
                num = num + vals[2 + 4 * sp + e] * ms[sp]
            gate = vals[2 + 4 * splits + e]
            sig = Float32(1.0) / (Float32(1.0) + cute.math.exp(-gate))
            out = bf16_round(num / den)
            p.xs[obuf, slot * HEAD_DIM + d + e] = BFloat16(out * sig)

    @cute.jit
    def reduce(self, p: AttnParams, col, pos, epoch):
        """Residual column ``col``, by one lane of warp 0: the bf16 o_proj output (the fp32 sum
        of the splits' partials in split order), written to the convolution ring at the
        position, then the post-attention convolution ``(conv + y) / sqrt(kernel + 1)`` (bf16)
        plus ``h`` (unless the layer starts a block)."""
        taps = self.conv_kernel
        # The weights, ring rows and input load before the poll (their latency overlaps it).
        weights = cute.make_rmem_tensor((taps,), Float32)
        past = cute.make_rmem_tensor((taps - 1,), BFloat16)
        for j in cutlass.range_constexpr(taps):
            weights[j] = p.attn_conv[col, j]
        for j in cutlass.range_constexpr(taps - 1):
            past[j] = p.ring_attn[(pos - (taps - 1) + j + RING) % RING, col]
        h = Float32(0.0)
        if p.boundary == 0:  # (layer 0's h may still be in CTA 0's embedding store)
            h = Float32(p.h[col])
        vals = cute.make_rmem_tensor((O_SPLITS,), Float32)
        fetch([(p.o_part, s * self.hidden + col, 1) for s in range(O_SPLITS)], vals, epoch)
        acc = vals[0]
        for s in cutlass.range_constexpr(1, O_SPLITS):
            acc = acc + vals[s]
        y = bf16_round(acc)
        p.ring_attn[pos % RING, col] = BFloat16(y)
        conv = Float32(0.0)
        for j in cutlass.range_constexpr(taps - 1):
            src = pos - (taps - 1) + j  # >= -(taps - 1): positions before 0 read zero
            conv = conv + weights[j] * cutlass.select_(src >= 0, Float32(past[j]), Float32(0.0))
        conv = conv + weights[taps - 1] * y
        z = bf16_round((conv + y) * fp32_reciprocal(math.sqrt(taps + 1)))
        p.out[col] = cutlass.select_(p.boundary != 0, z, h + z)

    # ----------------------------------------------------------------- kernel

    @cute.kernel
    def kernel(self, tma, gtensors, gate_tma, gate_g, scratch, tensors, blocks, bound, flags):
        layouts = {
            "W": cute.make_layout((ROWS, TILE_K, self.stages), stride=(TILE_K, 1, ROWS * TILE_K)),
            "KV": cute.make_layout(
                (BLOCK, HEAD_DIM, self.stages), stride=(HEAD_DIM, 1, BLOCK * HEAD_DIM)
            ),
            "X": cute.make_layout((self.streams + 2, self.hidden), stride=(self.hidden, 1)),
            "RED": cute.make_layout(
                (REDUCTIONS, COMPUTE_WARPS, RED_VALUES),
                stride=(COMPUTE_WARPS * RED_VALUES, RED_VALUES, 1),
            ),
            "MIX": cute.make_layout((MIX_FLOATS,)),
            "QKVS": cute.make_layout((QKVS_FLOATS,)),
            "ATT": cute.make_layout((COMPUTE_WARPS + 1, HEAD_DIM + 2), stride=(HEAD_DIM + 2, 1)),
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
        sW, sKV = self.buffer("W"), self.buffer("KV")
        parts = []
        weights = len(tma) - 2
        for i, (atom, g) in enumerate(zip(tma, gtensors)):
            is_kv = i >= weights
            tile = (BLOCK, HEAD_DIM) if is_kv else (ROWS, TILE_K)
            smem = sKV if is_kv else sW
            tiles = cute.local_tile(g, tile, (None, None))
            parts.append(
                cpasync.tma_partition(
                    atom,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(smem, 0, 2),
                    cute.group_modes(tiles, 0, 2),
                )
            )
        gate = None
        if cutlass.const_expr(self.messages):
            gate_parts = [
                cpasync.tma_partition(
                    atom,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sW, 0, 2),
                    cute.group_modes(cute.local_tile(g, (ROWS, TILE_K), (None, None)), 0, 2),
                )
                for atom, g in zip(gate_tma, gate_g)
            ]
            gate = GateParams(
                tma=tuple(gate_tma),
                gmem=tuple(part[1] for part in gate_parts),
                h_part=tagged.pop("h_part"),
                gated=tagged.pop("gated"),
                mhalf=tensors["mhalf"],
                messages=tensors["messages"],
            )
        num_blocks, queries, boundary, embed_input = flags
        params = AttnParams(
            tma=tuple(tma),
            gmem=tuple(part[1] for part in parts),
            sW=StageView(parts[0][0], axis=1),
            sKV=StageView(parts[weights][0], axis=1),
            xs=self.buffer("X"),
            red=self.buffer("RED"),
            mix=self.buffer("MIX"),
            qkvs=self.buffer("QKVS"),
            att=self.buffer("ATT"),
            gate=gate,
            blocks=blocks,
            num_blocks=num_blocks,
            queries=queries,
            boundary=boundary,
            embed_input=embed_input,
            **tagged,
            **tensors,
        )
        self.run_roles(params, trace_buffer=trace_buffer)

    @cute.jit
    def __call__(
        self,
        wq: cute.Tensor,
        wk: cute.Tensor,
        wv: cute.Tensor,
        wrel: cute.Tensor,
        k_dynamic: cute.Tensor,
        v_dynamic: cute.Tensor,
        wgate: cute.Tensor,
        wsink: cute.Tensor,
        wo: cute.Tensor,
        k_cache: cute.Tensor,
        v_cache: cute.Tensor,
        query: cute.Tensor,
        norms: cute.Tensor,
        q_scale: cute.Tensor,
        rel_scale: cute.Tensor,
        rel_proj: cute.Tensor,
        k_taps: cute.Tensor,
        v_taps: cute.Tensor,
        attn_conv: cute.Tensor,
        gate_down: cute.Tensor,
        gate_fg: cute.Tensor,
        mhalf: cute.Tensor,
        messages: cute.Tensor,
        embed: cute.Tensor,
        embed_norm: cute.Tensor,
        h: cute.Tensor,
        block0: cute.Tensor,
        block1: cute.Tensor,
        block2: cute.Tensor,
        block3: cute.Tensor,
        num_blocks: Int32,
        queries: Int32,
        boundary: Int32,
        embed_input: Int32,
        ring_k: cute.Tensor,
        ring_v: cute.Tensor,
        ring_attn: cute.Tensor,
        token: cute.Tensor,
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
        # In the order of self.matrices (sliding layers have no dynamic taps), then wo and the
        # caches.
        by_matrix = {Q: wq, K: wk, V: wv, REL: wrel, KDYN: k_dynamic, VDYN: v_dynamic,
                     GATE: wgate, SINK: wsink}  # fmt: skip
        op = cpasync.CopyBulkTensorTileG2SOp()
        w_smem = cute.make_layout((ROWS, TILE_K), stride=(TILE_K, 1))
        kv_smem = cute.make_layout((BLOCK, HEAD_DIM), stride=(HEAD_DIM, 1))
        pairs = [
            cpasync.make_tiled_tma_atom(op, src, w_smem, (ROWS, TILE_K))
            for src in [by_matrix[m] for m in self.matrices] + [wo]
        ]
        pairs += [
            cpasync.make_tiled_tma_atom(op, src, kv_smem, (BLOCK, HEAD_DIM))
            for src in (k_cache, v_cache)
        ]
        gate_pairs = []
        if cutlass.const_expr(self.messages):
            gate_pairs = gate_atoms(gate_down, gate_fg, self.hidden)
        bound = self.work.bind(table, workspace, max_ctas)
        tensors = (trace_buffer, state, pos, token, embed, embed_norm, h, query, norms, q_scale,
                   rel_scale, rel_proj, k_taps, v_taps, attn_conv, ring_k, ring_v, ring_attn,
                   k_cache, v_cache, mhalf, messages, out)  # fmt: skip
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
            (num_blocks, queries, boundary, embed_input),
        ).launch(**self.launch_config(bound), stream=stream)


@jit_cache
def compile_attn_kernel(op: InklingAttnKernel):
    """One compile per layer kind serves every visit, layer and position: weights, caches and
    states are arguments; the position, token, block count, query, boundary and embedding
    flags and the cache capacity are read at run time."""
    bf16, f32 = BFloat16, Float32
    h, qh, kv_dim = op.hidden, op.q_heads, op.kv_dim
    cache_rows = cute.sym_int(divisibility=BLOCK)
    taps = op.kv_heads * op.conv_kernel
    if op.full:  # static taps (kernel, channel), tap 0 the current token
        tap_shape, tap_dtype = (op.conv_kernel, kv_dim), bf16
    else:  # (channel, kernel), the last tap the current token
        tap_shape, tap_dtype = (kv_dim, op.conv_kernel), f32
    vector = [make_fake_compact_tensor(bf16, (h,), assumed_align=2) for _ in range(MAX_BLOCKS)]
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(bf16, (qh * HEAD_DIM, h)),  # wq
        make_fake_compact_tensor(bf16, (kv_dim, h)),  # wk
        make_fake_compact_tensor(bf16, (kv_dim, h)),  # wv
        make_fake_compact_tensor(bf16, (qh * op.d_rel, h)),  # wrel
        make_fake_compact_tensor(bf16, (taps, h)),  # k_dynamic (unread in sliding layers)
        make_fake_compact_tensor(bf16, (taps, h)),  # v_dynamic
        make_fake_compact_tensor(bf16, (qh * HEAD_DIM, h)),  # wgate
        make_fake_compact_tensor(bf16, (qh, h)),  # wsink
        make_fake_compact_tensor(bf16, (h, qh * HEAD_DIM)),  # wo
        make_fake_compact_tensor(bf16, (cache_rows, HEAD_DIM)),
        make_fake_compact_tensor(bf16, (cache_rows, HEAD_DIM)),
        make_fake_compact_tensor(f32, (op.queries, h), assumed_align=4),  # queries
        make_fake_compact_tensor(bf16, (op.streams, h), assumed_align=2),  # norms
        make_fake_compact_tensor(bf16, (qh, HEAD_DIM), assumed_align=2),
        make_fake_compact_tensor(bf16, (qh, op.d_rel), assumed_align=2),
        make_fake_compact_tensor(bf16, (op.rel_width, op.d_rel), assumed_align=16),
        make_fake_compact_tensor(tap_dtype, tap_shape, assumed_align=2),
        make_fake_compact_tensor(tap_dtype, tap_shape, assumed_align=2),
        make_fake_compact_tensor(f32, (h, op.conv_kernel), assumed_align=4),
        *fake_gate_args(h),
        make_fake_compact_tensor(bf16, (cute.sym_int(), h)),  # embedding table
        make_fake_compact_tensor(bf16, (h,), assumed_align=2),  # embedding norm
        make_fake_compact_tensor(bf16, (h,), assumed_align=2),  # h
        *vector,
        Int32(1),
        Int32(1),
        Int32(0),
        Int32(0),
        make_fake_compact_tensor(bf16, (RING, kv_dim), assumed_align=2),
        make_fake_compact_tensor(bf16, (RING, kv_dim), assumed_align=2),
        make_fake_compact_tensor(bf16, (RING, h), assumed_align=2),
        make_fake_compact_tensor(Int64, (1,), assumed_align=8),  # token
        make_fake_compact_tensor(Int32, (1,), assumed_align=4),  # pos
        make_fake_compact_tensor(f32, (h,), assumed_align=4),
        make_fake_compact_tensor(Int64, (cute.sym_int(),)),  # trace buffer (globaltimer only)
        make_fake_compact_tensor(Int64, (op.scratch_size,)),
        make_fake_compact_tensor(Int32, (2,), assumed_align=4),
        make_fake_compact_tensor(Int32, (cute.sym_int(), WORDS)),
        make_fake_compact_tensor(Int32, (cute.sym_int(),)),
        Int32(1),
        name=op.get_name(),
    )
