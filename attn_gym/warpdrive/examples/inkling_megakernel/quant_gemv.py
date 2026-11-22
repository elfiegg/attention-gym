"""Expert GEMVs on the checkpoint's quantized experts (``quant.QuantExperts``): CuTeDSL
helpers that decode a lane's weights in registers, and ``QuantGemvKernel``, a standalone
warpdrive launch over one token's routed experts that the fused visit kernel's GATEUP/DOWN
phases can adopt.

The decode reproduces the bf16-dequantized weights exactly and the accumulation is the one of
``qwen_megakernel.core.CudaCoreGemv`` (per lane, even and odd columns in two fp32
accumulators over the subtiles in order, then ``warp_transpose_reduce``), so for the same
input the dot products are bit-identical to that GEMV on ``QuantExperts.dequantize()``
weights; ``format="bf16"`` runs it for comparison.

Per lane, row and 256-column subtile (columns ``8l .. 8l + 7``):

- NVFP4: one re-encoded 32-bit word and the 16-byte ``lut[expert, scale]`` byte planes;
  ``nvfp4_bf16x8`` selects each column's low and high magnitude byte with ``prmt`` (the
  code's magnitude is the selector nibble), ORs the sign into the high byte and interleaves
  the bytes into 4 bf16x2 words: 12 ALU-pipe instructions and 2 FMA-pipe shifts per 8
  weights. The planes of the routed experts are staged in SMEM (global gathers through L1
  cost 4.4 us of 19 per expert set).
- MXFP8: two words of e4m3 bytes; ``mxfp8_bf16x8`` converts each byte pair exactly
  (``cvt.rn.bf16x2.e4m3x2``) and multiplies by the bf16 power of two ``2 ** (s - 127)``
  (``mul.rn.bf16x2``, exact): 4 ALU + 4 FMA-pipe instructions per 8 weights.
- bf16: four bf16x2 words.

bf16x2 weight words accumulate with ``fma_bf16x2`` (``fma.rn.f32.bf16``, SASS
``FHFMA.BF16``): an fp32 FMA of exact bf16 operands at full FMA rate, with the same results
as FFMA on the widened values and no widening instructions for weights or input.

``QuantGemvKernel`` (one CTA per SM, static persistent over ``(row tile, slot, kind)``
items): warp 8 streams each tile's stages by TMA (the weights' ``(32 rows, stage_subtiles x
256 columns)`` box and, quantized, the stage's tile-major scales into the same ring stage;
the down projection's 4.5 subtiles end in TMA zero fill); warps 0-7 compute. Kind 0 is the
slot's gate/up rows (checkpoint order: 16 interleaved (gate, up) pairs per tile) on ``x``,
kind 1 its down rows on ``act[slot]`` (zero padded to whole subtiles). Outputs are the raw
fp32 dot products ``out[kind, slot, row]``.

GB300, one expert set (``main``; graph of launches on distinct routings, us per launch):
bf16 21.9 (127 MB, 5.8 TB/s), MXFP8 13.2 (66 MB, 5.0 TB/s), NVFP4 15.0 (36 MB, 2.4 TB/s).
Without the math, NVFP4 streams in 7.4 us and MXFP8 in 11.4. NVFP4 is compute-bound (the
same time with the weights in L2; ALU pipe about half busy): without the decode's PRMT/LOP3
work it takes 12.9 us, the rest being the per-row loads (weights, planes), address math and
FMAs. Neither 16 compute warps of 2 rows nor 4-byte plane entries (planes derived in
registers) were faster.
"""

from dataclasses import dataclass

import cutlass
import torch
from cuda.bindings import driver as cuda
from cutlass import BFloat16, Float32, Int32, Int64, cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import T, dsl_user_op

from attn_gym.warpdrive import (
    Consumer,
    Local,
    NamedBarrier,
    Producer,
    Raster,
    Ring,
    SmemPlan,
    StageView,
    StaticPersistent,
    Tma,
    Warps,
    WarpSpecialized,
    WorkCursor,
    WorkStream,
    role,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor
from attn_gym.warpdrive.examples.inkling_megakernel import quant
from attn_gym.warpdrive.examples.qwen_megakernel.core import (
    COMPUTE_THREADS,
    COMPUTE_WARPS,
    EVICT_FIRST,
    ROWS,
    ROWS_PER_WARP,
    TILE_K,
    warp_transpose_reduce,
)

BF16 = "bf16"
FORMATS = (BF16, quant.NVFP4, quant.MXFP8)
SMEM_BYTES, SMEM_HEADROOM = 232448, 1024
MAX_STAGES = 16
LANE_WORDS = {BF16: 4, quant.NVFP4: 1, quant.MXFP8: 2}  # weight words per lane, row, subtile
SUBTILE_WORDS = {f: 32 * w for f, w in LANE_WORDS.items()}
# Tile-major scale words per subtile: 8 warps x blocks (each word: a block's 4 warp rows).
BLOCKS = {quant.NVFP4: TILE_K // 16, quant.MXFP8: TILE_K // 32}
GATEUP, DOWN = range(2)
TENSOR_ARGS = ("experts", "x", "act", "lut13", "lut2", "out")

assert (ROWS, TILE_K, COMPUTE_WARPS) == (quant.TILE_ROWS, quant.SUBTILE, quant.WARPS)
assert TILE_K // 32 == quant.LANE_COLUMNS
assert COMPUTE_THREADS == 256  # stage_inputs: one 16-byte plane entry per thread and slot


# ------------------------------------------------------------------ register decode


def inline_asm(result_types: list, operands: list, body: str, constraints: str, loc, ip):
    """Pure inline PTX with one or more results."""
    values = [v.ir_value(loc=loc, ip=ip) for v in operands]
    single = len(result_types) == 1
    result = llvm.inline_asm(
        result_types[0] if single else llvm.StructType.get_literal(result_types),
        values,
        body,
        constraints,
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    if single:
        return [result]
    return [llvm.extractvalue(t, result, [i], loc=loc, ip=ip) for i, t in enumerate(result_types)]


@dsl_user_op
def nvfp4_bf16x8(word, planes, *, loc=None, ip=None) -> list:
    """The 8 bf16 weights of one re-encoded NVFP4 word (``quant``'s layout) as 4 bf16x2 words
    (columns 2j, 2j + 1 in word j, the even column low), given the scale's magnitude byte
    planes ``lut[expert, scale]`` (4 words: low bytes of codes 0-3, 4-7, high bytes of 0-3,
    4-7). PRMT reads selector nibbles from the low 16 bits; a nibble's top bit (a sign here)
    would select sign replication, hence the masks."""
    out = inline_asm(
        [T.i32()] * 4,
        [Int32(word), *(Int32(p) for p in planes)],
        "{\n"
        ".reg .b32 sa, sb, la, ha, lb, hb, t;\n"
        "and.b32 sa, $4, 0x7777;\n"  # magnitudes of columns 0-3
        "mul.hi.u32 t, $4, 65536;\n"  # >> 16 on the FMA pipe (the ALU pipe is the busy one)
        "and.b32 sb, t, 0x7777;\n"  # columns 4-7
        "prmt.b32 la, $5, $6, sa;\n"
        "prmt.b32 ha, $7, $8, sa;\n"
        "prmt.b32 lb, $5, $6, sb;\n"
        "prmt.b32 hb, $7, $8, sb;\n"
        "lop3.b32 ha, ha, $4, 0x80808080, 0xF8;\n"  # | signs of columns 0-3 (bits 8k + 7)
        "shl.b32 t, $4, 4;\n"
        "lop3.b32 hb, hb, t, 0x80808080, 0xF8;\n"  # | columns 4-7 (bits 8(k - 4) + 3)
        "prmt.b32 $0, la, ha, 0x5140;\n"
        "prmt.b32 $1, la, ha, 0x7362;\n"
        "prmt.b32 $2, lb, hb, 0x5140;\n"
        "prmt.b32 $3, lb, hb, 0x7362;\n"
        "}",
        "=r,=r,=r,=r,r,r,r,r,r",
        loc,
        ip,
    )
    return [Int32(v) for v in out]


@dsl_user_op
def mxfp8_bf16x8(word0, word1, scale, *, loc=None, ip=None) -> list:
    """The 8 bf16 weights ``e4m3 * 2 ** (s - 127)`` of two words of e4m3 bytes (byte k of
    ``word0`` is column k, of ``word1`` column 4 + k) as 4 bf16x2 words, given the scale as a
    bf16x2 word ``2 ** (s - 127)`` twice: an exact ``cvt`` to bf16, then an exact bf16
    multiply (the product is a normal number for scale bytes >= ``quant.MIN_MX_SCALE``)."""
    out = inline_asm(
        [T.i32()] * 4,
        [Int32(word0), Int32(word1), Int32(scale)],
        "{\n"
        ".reg .b16 a, b, c, d;\n"
        ".reg .b32 v;\n"
        "mov.b32 {a, b}, $4;\n"
        "mov.b32 {c, d}, $5;\n"
        "cvt.rn.bf16x2.e4m3x2 v, a;\n"
        "mul.rn.bf16x2 $0, v, $6;\n"
        "cvt.rn.bf16x2.e4m3x2 v, b;\n"
        "mul.rn.bf16x2 $1, v, $6;\n"
        "cvt.rn.bf16x2.e4m3x2 v, c;\n"
        "mul.rn.bf16x2 $2, v, $6;\n"
        "cvt.rn.bf16x2.e4m3x2 v, d;\n"
        "mul.rn.bf16x2 $3, v, $6;\n"
        "}",
        "=r,=r,=r,=r,r,r,r",
        loc,
        ip,
    )
    return [Int32(v) for v in out]


@dsl_user_op
def fma_bf16x2(acc0, acc1, w, x, *, loc=None, ip=None):
    """``(acc0 + w.lo * x.lo, acc1 + w.hi * x.hi)`` for bf16x2 words, each one fp32 FMA."""
    out = inline_asm(
        [T.f32(), T.f32()],
        [Float32(acc0), Float32(acc1), Int32(w), Int32(x)],
        "{\n"
        ".reg .b16 w0, w1, x0, x1;\n"
        "mov.b32 {w0, w1}, $4;\n"
        "mov.b32 {x0, x1}, $5;\n"
        "fma.rn.f32.bf16 $0, w0, x0, $2;\n"
        "fma.rn.f32.bf16 $1, w1, x1, $3;\n"
        "}",
        "=f,=f,f,f,r,r",
        loc,
        ip,
    )
    return Float32(out[0]), Float32(out[1])


@dsl_user_op
def scale_byte(scales, r: int, *, loc=None, ip=None) -> Int32:
    """Byte ``r`` of a lane's scale word (row ``r`` of its 4): one ``prmt``."""
    out = inline_asm(
        [T.i32()], [Int32(scales)], f"prmt.b32 $0, $1, 0, {0x4440 + r:#x};", "=r,r", loc, ip
    )
    return Int32(out[0])


@cute.jit
def mx_scale(scales, r: cutlass.Constexpr) -> Int32:
    """bf16x2 ``2 ** (s - 127)`` twice for byte ``r`` of a lane's scale word (s >= 1)."""
    return scale_byte(scales, r) * Int32(0x00800080)


@cute.jit
def accumulate_subtile(
    fmt: cutlass.Constexpr, acc, sW, row0, word0, xw, scales, planes, plane_base, lane
):
    """Add one 256-column subtile to ``acc`` (``(ROWS_PER_WARP, 2)`` fp32: the even and odd
    columns of the lane's rows ``row0 ..``). ``sW``: the stage's ``(32, words)`` Int32
    weights, this subtile from word ``word0``; ``xw``: the lane's 8 input columns as 4 bf16x2
    words; ``scales``: the lane's tile-major scale word (its 4 rows' bytes; quantized
    formats); ``planes``: flat Int32 NVFP4 magnitude planes, the expert's 256 x 4 words from
    ``plane_base``."""
    n = LANE_WORDS[fmt]
    for r in cutlass.range_constexpr(ROWS_PER_WARP):
        wq = cute.make_rmem_tensor((n,), Int32)
        cute.autovec_copy(cute.local_tile(sW[row0 + r, None], (n,), (word0 // n + lane,)), wq)
        if cutlass.const_expr(fmt == quant.MXFP8):
            words = mxfp8_bf16x8(wq[0], wq[1], mx_scale(scales, r))
        elif cutlass.const_expr(fmt == quant.NVFP4):
            lut = cute.make_rmem_tensor((4,), Int32)
            index = plane_base // 4 + scale_byte(scales, r)
            cute.autovec_copy(cute.local_tile(planes, (4,), (index,)), lut)
            words = nvfp4_bf16x8(wq[0], [lut[j] for j in range(4)])
        else:
            words = [wq[j] for j in range(4)]
        a0, a1 = acc[r, 0], acc[r, 1]
        for j in cutlass.range_constexpr(4):
            a0, a1 = fma_bf16x2(a0, a1, words[j], xw[j])
        acc[r, 0], acc[r, 1] = a0, a1
    return acc


# ------------------------------------------------------------------ standalone kernel


@dataclass(frozen=True)
class GemvParams:
    tma: tuple  # TMA atoms: w13, w2 weights, then (quantized) their scales
    gmem: tuple  # their global tile partitions
    tW: StageView  # the TMA partitions of the weight and scale stages (the load warp's views)
    tS: StageView | None
    xs: cute.Tensor  # SMEM bf16 (hidden + top_k * act_columns,): x, then each slot's act
    sl: cute.Tensor | None  # SMEM int32 (2 * top_k * 1024,): planes by (kind, slot) (NVFP4)
    experts: cute.Tensor  # (top_k,) int32 routed experts by slot
    x: cute.Tensor  # (hidden,) bf16
    act: cute.Tensor  # (top_k * act_columns,) bf16, zero past width in each slot
    lut13: cute.Tensor  # flat int32 (experts * 256 * 4,) NVFP4 magnitude planes
    lut2: cute.Tensor
    out: cute.Tensor  # (2, top_k, hidden) fp32


class QuantGemvKernel(WarpSpecialized):
    """One token's routed expert GEMVs (see the module). ``fmt``: "bf16", "nvfp4" or
    "mxfp8"; ``stage_subtiles``: 256-column subtiles per ring stage (bf16: 1, a 16 KB stage);
    ``stages``: ring depth (default: as many as fit)."""

    def __init__(
        self,
        fmt: str,
        hidden: int = 2304,
        width: int = 1152,
        experts: int = 128,
        top_k: int = 8,
        stage_subtiles: int | None = None,
        stages: int | None = None,
    ):
        if fmt not in FORMATS:
            raise ValueError(f"format must be one of {FORMATS}, got {fmt!r}")
        if 2 * width != hidden or hidden % TILE_K or width % 8:
            raise ValueError("need 2 * width == hidden, a multiple of 256, and width % 8 == 0")
        if stage_subtiles is None:
            stage_subtiles = 1 if fmt == BF16 else 4
        if fmt == BF16 and stage_subtiles != 1:
            raise ValueError("bf16 stages are one subtile (a TMA box is at most 256 elements)")
        if not 1 <= stage_subtiles * SUBTILE_WORDS[fmt] <= 256:
            raise ValueError(f"stage_subtiles {stage_subtiles} exceeds a 256-word TMA box")
        self.fmt, self.hidden, self.width = fmt, hidden, width
        self.experts, self.top_k = experts, top_k
        self.stage_subtiles = stage_subtiles
        self.subtiles = (hidden // TILE_K, -(-width // TILE_K))  # gate/up, down (zero filled)
        self.act_columns = self.subtiles[DOWN] * TILE_K
        self.row_tiles = hidden // ROWS
        self.stage_words = stage_subtiles * SUBTILE_WORDS[fmt]
        self.scale_words = 8 * BLOCKS[fmt] if fmt != BF16 else 0
        stage_bytes = 4 * (ROWS * self.stage_words + stage_subtiles * self.scale_words)
        # NVFP4: the routed experts' magnitude planes of both matrices, staged per launch.
        self.lut_words = 2 * top_k * 1024 if fmt == quant.NVFP4 else 0
        fixed = 2 * (hidden + top_k * self.act_columns) + 4 * self.lut_words + SMEM_HEADROOM
        fit = min(MAX_STAGES, (SMEM_BYTES - fixed) // stage_bytes)
        stages = fit if stages is None else stages
        if not 2 <= stages <= fit:
            raise ValueError(f"stages must be in [2, {fit}], got {stages}")
        self.stages = stages
        self.smem_plan = SmemPlan()
        buffers = {"W": self.smem_plan.buffer("W", Int32, align=1024)}
        if fmt != BF16:
            buffers["S"] = self.smem_plan.buffer("S", Int32, align=128)
        self.smem_plan.buffer("X", BFloat16, align=16)
        if fmt == quant.NVFP4:
            self.smem_plan.buffer("L", Int32, align=16)
        self.w = Ring(stages, Tma(stage_bytes), Warps, buffer=buffers)
        self.csync = NamedBarrier(threads=COMPUTE_THREADS)
        self.work = WorkStream(mapping=Raster(), schedule=StaticPersistent())

    def stage_count(self, kind: int) -> int:
        return -(-self.subtiles[kind] // self.stage_subtiles)

    def get_name(self) -> str:
        return (
            f"warpdrive_inkling_quant_gemv_{self.fmt}_h{self.hidden}_i{self.width}"
            f"_e{self.experts}k{self.top_k}_t{self.stage_subtiles}_s{self.stages}"
        )

    @role(warps=COMPUTE_WARPS)
    @cute.jit
    def load(self, p: GemvParams, *, w: Producer, work: WorkCursor):
        quantized = cutlass.const_expr(self.fmt != BF16)
        if cutlass.const_expr(quantized):
            w = w.views(W=p.tW, S=p.tS)
        else:
            w = w.views(W=p.tW)
        while work.valid:
            m, slot, kind = work.item
            tile = p.experts[slot] * self.row_tiles + m
            for k in cutlass.range_constexpr(2):
                if kind == k:
                    for st in cutlass.range_constexpr(self.stage_count(k)):
                        with w.produce() as h:
                            cute.copy(
                                p.tma[k],
                                p.gmem[k][None, tile, st],
                                h.view("W"),
                                tma_bar_ptr=h.barrier,
                                cache_policy=Int64(EVICT_FIRST),
                            )
                            if cutlass.const_expr(quantized):
                                cute.copy(
                                    p.tma[2 + k],
                                    p.gmem[2 + k][None, tile, st, 0],
                                    h.view("S"),
                                    tma_bar_ptr=h.barrier,
                                    cache_policy=Int64(EVICT_FIRST),
                                )
            work.advance()
        return w, work

    @role(warps=range(COMPUTE_WARPS))
    @cute.jit
    def compute(self, p: GemvParams, *, w: Consumer, work: WorkCursor, me: Local):
        tid, warp, lane = me.tid, me.warp, me.lane
        self.stage_inputs(p, tid)
        self.csync.sync()
        while work.valid:
            m, slot, kind = work.item
            for k in cutlass.range_constexpr(2):
                if kind == k:
                    w = self.tile(p, w, k, m, slot, warp, lane)
            work.advance()
        return w, work

    @cute.jit
    def stage_inputs(self, p: GemvParams, tid):
        """``x``, every slot's activations and (NVFP4) the slots' magnitude planes into SMEM.
        Every load is issued before the first store, so their L2 latencies overlap."""
        x_chunks = self.hidden // 8
        chunks = x_chunks + self.top_k * self.act_columns // 8
        rounds = -(-chunks // COMPUTE_THREADS)
        frags = []
        for i in cutlass.range_constexpr(rounds):
            c = i * COMPUTE_THREADS + tid
            frag = cute.make_rmem_tensor((8,), BFloat16)
            if c < x_chunks:
                cute.autovec_copy(cute.local_tile(p.x, (8,), (c,)), frag)
            elif c < chunks:
                cute.autovec_copy(cute.local_tile(p.act, (8,), (c - x_chunks,)), frag)
            frags.append(frag)
        planes = []
        if cutlass.const_expr(self.fmt == quant.NVFP4):
            for kind in cutlass.range_constexpr(2):
                lut = p.lut13 if kind == GATEUP else p.lut2
                for slot in cutlass.range_constexpr(self.top_k):
                    words = cute.make_rmem_tensor((4,), Int32)
                    cute.autovec_copy(
                        cute.local_tile(lut, (4,), (p.experts[slot] * 256 + tid,)), words
                    )
                    planes.append(words)
        for i in cutlass.range_constexpr(rounds):
            c = i * COMPUTE_THREADS + tid
            if c < chunks:
                cute.autovec_copy(frags[i], cute.local_tile(p.xs, (8,), (c,)))
        for i in cutlass.range_constexpr(len(planes)):
            cute.autovec_copy(planes[i], cute.local_tile(p.sl, (4,), (i * 256 + tid,)))

    @cute.jit
    def tile(self, p: GemvParams, w, kind: cutlass.Constexpr, m, slot, warp, lane):
        """One 32-row tile: the stages' dot products, then the rows' fp32 results."""
        fmt = self.fmt
        xoff = 0 if kind == GATEUP else self.hidden + slot * self.act_columns
        plane_base = (kind * self.top_k + slot) * 1024
        xs_words = cute.recast_tensor(p.xs, Int32)
        acc = cute.make_rmem_tensor((ROWS_PER_WARP, 2), Float32)
        acc.fill(0.0)
        for st in cutlass.range_constexpr(self.stage_count(kind)):
            with w.consume() as h:
                sW = h.view("W")
                for j in cutlass.range_constexpr(self.stage_subtiles):
                    kt = st * self.stage_subtiles + j
                    if cutlass.const_expr(kt < self.subtiles[kind]):
                        xw = cute.make_rmem_tensor((4,), Int32)
                        word = (xoff + kt * TILE_K) // 2 + 4 * lane
                        cute.autovec_copy(cute.local_tile(xs_words, (4,), (word // 4,)), xw)
                        scales = Int32(0)
                        if cutlass.const_expr(fmt != BF16):
                            blocks = BLOCKS[fmt]
                            scales = h.view("S")[0, j, warp * blocks + lane // (32 // blocks)]
                        acc = accumulate_subtile(
                            fmt, acc, sW, warp * ROWS_PER_WARP, j * SUBTILE_WORDS[fmt], xw,
                            scales, p.sl, plane_base, lane,
                        )  # fmt: skip
        for r in cutlass.range_constexpr(ROWS_PER_WARP):
            acc[r, 0] = acc[r, 0] + acc[r, 1]
        v = warp_transpose_reduce(acc[None, 0], lane, ROWS_PER_WARP)
        if lane % (32 // ROWS_PER_WARP) == 0:
            row = warp * ROWS_PER_WARP + lane // (32 // ROWS_PER_WARP)
            p.out[kind, slot, m * ROWS + row] = v
        return w

    @cute.kernel
    def kernel(self, tma, gtensors, tensors, bound):
        layouts = {
            "W": cute.make_layout(
                (ROWS, self.stage_words, self.stages),
                stride=(self.stage_words, 1, ROWS * self.stage_words),
            ),
            "X": cute.make_layout((self.hidden + self.top_k * self.act_columns,)),
        }
        if cutlass.const_expr(self.fmt == quant.NVFP4):
            layouts["L"] = cute.make_layout((self.lut_words,))
        if cutlass.const_expr(self.fmt != BF16):
            n, words = self.stage_subtiles, self.scale_words
            layouts["S"] = cute.make_layout(
                (1, n, words, self.stages), stride=(0, words, 1, n * words)
            )
        self.begin_roles(work=bound, layouts=layouts)
        tensors = {k: cute.make_tensor(t.iterator, t.layout) for k, t in zip(TENSOR_ARGS, tensors)}
        sW = self.buffer("W")
        smem = [sW, sW]
        tiles = [cute.local_tile(g, (ROWS, self.stage_words), (None, None)) for g in gtensors[:2]]
        rank = [2, 2]
        if cutlass.const_expr(self.fmt != BF16):
            sS = self.buffer("S")
            smem += [sS, sS]
            box = (1, self.stage_subtiles, self.scale_words)
            tiles += [cute.local_tile(g, box, (None, None, None)) for g in gtensors[2:]]
            rank += [3, 3]
        parts = [
            cpasync.tma_partition(
                atom,
                0,
                cute.make_layout(1),
                cute.group_modes(s, 0, r),
                cute.group_modes(t, 0, r),
            )
            for atom, s, t, r in zip(tma, smem, tiles, rank)
        ]
        params = GemvParams(
            tma=tuple(tma),
            gmem=tuple(part[1] for part in parts),
            tW=StageView(parts[0][0], axis=1),
            tS=StageView(parts[2][0], axis=1) if cutlass.const_expr(self.fmt != BF16) else None,
            xs=self.buffer("X"),
            sl=self.buffer("L") if cutlass.const_expr(self.fmt == quant.NVFP4) else None,
            **tensors,
        )
        self.run_roles(params)

    @cute.jit
    def __call__(
        self,
        w13: cute.Tensor,
        w2: cute.Tensor,
        s13: cute.Tensor,
        s2: cute.Tensor,
        lut13: cute.Tensor,
        lut2: cute.Tensor,
        experts: cute.Tensor,
        x: cute.Tensor,
        act: cute.Tensor,
        out: cute.Tensor,
        workspace: cute.Tensor,
        max_ctas: Int32,
        stream: cuda.CUstream,
    ):
        op = cpasync.CopyBulkTensorTileG2SOp()
        smem = cute.make_layout((ROWS, self.stage_words), stride=(self.stage_words, 1))
        pairs = [
            cpasync.make_tiled_tma_atom(op, g, smem, (ROWS, self.stage_words)) for g in (w13, w2)
        ]
        if cutlass.const_expr(self.fmt != BF16):
            box = (1, self.stage_subtiles, self.scale_words)
            s_layout = cute.make_layout(box, stride=(0, self.scale_words, 1))
            pairs += [cpasync.make_tiled_tma_atom(op, g, s_layout, box) for g in (s13, s2)]
        bound = self.work.bind((self.row_tiles, self.top_k, 2), workspace, max_ctas)
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            tuple(a for a, _ in pairs),
            tuple(g for _, g in pairs),
            (experts, x, act, lut13, lut2, out),
            bound,
        ).launch(**self.launch_config(bound), stream=stream)


@jit_cache
def compile_quant_gemv(op: QuantGemvKernel):
    """One compile per format and stage shape; the experts, inputs and routing are runtime."""
    i32 = Int32
    e, h, width = op.experts, op.hidden, op.width
    if op.fmt == BF16:
        weights = [
            make_fake_compact_tensor(i32, (e * h, h // 2)),
            make_fake_compact_tensor(i32, (e * h, width // 2)),
        ]
        scales = [make_fake_compact_tensor(i32, (1, 1, 4)) for _ in range(2)]
        luts = [make_fake_compact_tensor(i32, (4,)) for _ in range(2)]
    else:
        per_word = 8 if op.fmt == quant.NVFP4 else 4  # columns per Int32 word
        weights = [
            make_fake_compact_tensor(i32, (e * h, h // per_word)),
            make_fake_compact_tensor(i32, (e * h, width // per_word)),
        ]
        scales = [
            make_fake_compact_tensor(i32, (e * op.row_tiles, kt, op.scale_words))
            for kt in op.subtiles
        ]
        lut_words = e * 1024 if op.fmt == quant.NVFP4 else 4
        luts = [make_fake_compact_tensor(i32, (lut_words,)) for _ in range(2)]
    return compile_tvm_ffi(
        op,
        *weights,
        *scales,
        *luts,
        make_fake_compact_tensor(i32, (op.top_k,), assumed_align=4),
        make_fake_compact_tensor(BFloat16, (h,)),
        make_fake_compact_tensor(BFloat16, (op.top_k * op.act_columns,)),
        make_fake_compact_tensor(Float32, (2, op.top_k, h), assumed_align=4),
        make_fake_compact_tensor(i32, (cute.sym_int(),)),
        Int32(1),
        name=op.get_name(),
    )


class QuantGemv:
    """Host runner: ``run(experts, x, act)`` returns ``out`` (``(2, top_k, hidden)`` fp32) for
    the layer's ``w13``/``w2``: ``QuantExperts`` (checkpoint row order) or bf16 tensors in
    checkpoint row order (``format="bf16"``)."""

    def __init__(self, w13, w2, top_k: int = 8, device="cuda", **options):
        dense = isinstance(w13, torch.Tensor)
        fmt = BF16 if dense else w13.format
        experts, h = w13.shape[:2] if dense else (w13.experts, w13.rows)
        width = w2.shape[2] if dense else w2.columns
        self.op = QuantGemvKernel(fmt, h, width, experts, top_k, **options)
        self.workspace, self.max_ctas = self.op.work.launch_args(torch.device(device))
        # Unused arguments of a format (bf16 scales, MXFP8 planes) get 4-word placeholders.
        dummy = torch.zeros(4, dtype=torch.int32, device=device)
        if dense:
            self.weights = tuple(w.view(torch.int32).view(experts * h, -1) for w in (w13, w2))
            self.scales = (dummy.view(1, 1, 4), dummy.view(1, 1, 4))
            self.luts = (dummy, dummy)
        else:
            self.weights = tuple(
                q.weight.view(torch.int32).view(experts * q.rows, -1) for q in (w13, w2)
            )
            self.scales = tuple(
                q.scale.view(torch.int32).view(experts * q.rows // ROWS, q.subtiles, -1)
                for q in (w13, w2)
            )
            self.luts = tuple(dummy if q.lut is None else q.lut.view(-1) for q in (w13, w2))
        self.out = torch.empty(2, top_k, h, dtype=torch.float32, device=device)

    def run(self, experts: torch.Tensor, x: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """``experts``: int32 ``(top_k,)``; ``x``: bf16 ``(hidden,)``; ``act``: bf16
        ``(top_k, act_columns)`` (zero past width)."""
        compile_quant_gemv(self.op)(
            *self.weights, *self.scales, *self.luts, experts, x, act.view(-1), self.out,
            self.workspace, self.max_ctas,
        )  # fmt: skip
        return self.out


def expert_set_bytes(fmt: str, hidden: int, width: int, top_k: int) -> int:
    """Bytes one expert set streams from DRAM: the slots' gate/up and down weights plus,
    quantized, their scales (the down rows' padded last subtile included)."""
    weights = top_k * 3 * width * hidden
    if fmt == BF16:
        return 2 * weights
    subtiles = hidden // TILE_K + -(-width // TILE_K)
    scales = top_k * (hidden // ROWS) * subtiles * 4 * 8 * BLOCKS[fmt]
    return weights // (2 if fmt == quant.NVFP4 else 1) + scales


def main(
    configs: tuple[str, ...] = ("bf16", "nvfp4:3", "nvfp4:4", "mxfp8:1", "mxfp8:2", "mxfp8:4"),
):
    """GB300 timing of one expert set (8 experts' gate/up + down) per format and stage shape
    (``fmt:stage_subtiles``): a CUDA graph of 16 launches with distinct routings (so the
    weights come from DRAM), median of 20 replays, random weights at the checkpoint's
    shapes."""
    hidden, width, experts, top_k, sets = 2304, 1152, 128, 8, 16
    gen = torch.Generator(device="cuda").manual_seed(0)
    routings = [
        torch.randperm(experts, generator=gen, device="cuda")[:top_k].sort().values.int()
        for _ in range(sets)
    ]
    x = torch.randn(hidden, generator=gen, device="cuda").bfloat16()
    act = torch.zeros(top_k, -(-width // TILE_K) * TILE_K, device="cuda", dtype=torch.bfloat16)
    act[:, :width] = torch.randn(top_k, width, generator=gen, device="cuda")
    layers = {}
    for config in configs:
        fmt, _, subtiles = config.partition(":")
        quant_fmt = quant.NVFP4 if fmt == BF16 else fmt
        if quant_fmt not in layers:
            layers[quant_fmt] = tuple(
                quant.QuantExperts.from_checkpoint(
                    *quant.random_checkpoint_experts(quant_fmt, experts, rows, columns, seed)
                )
                for rows, columns, seed in ((2 * width, hidden, 1), (hidden, width, 2))
            )
        w13, w2 = layers[quant_fmt]
        if fmt == BF16:
            runner = QuantGemv(w13.dequantize(), w2.dequantize(), top_k)
        else:
            runner = QuantGemv(w13, w2, top_k, stage_subtiles=int(subtiles))
        for routing in routings:
            runner.run(routing, x, act)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for routing in routings:
                runner.run(routing, x, act)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(20):
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1e3 / sets)
        us = sorted(times)[len(times) // 2]
        nbytes = expert_set_bytes(fmt, hidden, width, top_k)
        print(
            f"{config:10s} stages {runner.op.stages:2d}  {us:6.2f} us  {nbytes / 1e6:6.1f} MB"
            f"  {nbytes / us / 1e6:5.2f} TB/s"
        )


if __name__ == "__main__":
    main()
