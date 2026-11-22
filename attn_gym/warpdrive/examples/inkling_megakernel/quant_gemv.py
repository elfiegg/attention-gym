"""Register decode of the checkpoint's quantized experts (``quant.QuantExperts``) for the
expert GEMVs of the visit kernel (``mlp_half``): CuTeDSL helpers that turn a lane's weight
words into bf16x2 words and accumulate them.

The decode reproduces the bf16-dequantized weights exactly and ``accumulate_subtile`` adds
them in the order of ``qwen_megakernel.core.CudaCoreGemv`` (per lane, even and odd columns in
two fp32 accumulators over the subtiles in order; the caller then sums the two and runs
``warp_transpose_reduce``), so for the same input the dot products are bit-identical to that
GEMV on ``QuantExperts.dequantize()`` weights.

Per lane, row and 256-column subtile (columns ``8l .. 8l + 7``):

- NVFP4: one re-encoded 32-bit word and the 16-byte ``lut[expert, scale]`` byte planes;
  ``nvfp4_bf16x8`` selects each column's low and high magnitude byte with ``prmt`` (the
  code's magnitude is the selector nibble), XORs the sign into the high byte and interleaves
  the bytes into 4 bf16x2 words: 12 ALU-pipe instructions (PRMT, LOP3) per 8 weights, plus 2
  shifts that ptxas emits as IMAD.HI/IMAD.SHL on the FMA pipe. The planes must be in SMEM:
  gathers from global memory through L1 cost a standalone expert GEMV 4.4 us of 19 per expert
  set on GB300.
- MXFP8: two words of e4m3 bytes; ``mxfp8_bf16x8`` converts each byte pair exactly
  (``cvt.rn.bf16x2.e4m3x2``) and multiplies by the bf16 power of two ``2 ** (s - 127)``
  (``mul.rn.bf16x2``, exact): 4 ALU + 4 FMA-pipe instructions per 8 weights.
- bf16: four bf16x2 words.

bf16x2 weight words accumulate with ``fma_bf16x2`` (``fma.rn.f32.bf16``, SASS
``FHFMA.BF16``): an fp32 FMA of exact bf16 operands at full FMA rate, with the same results
as FFMA on the widened values and no widening instructions for weights or input. On GB300,
PRMT, LOP3, F2FP and the packed fp32 FFMA2/FMUL2/HADD2.F32 issue at half rate, FHFMA.BF16 at
full rate. A standalone GEMV of one expert set (8 experts' gate/up and down rows) measured
bf16 21.9 us, MXFP8 13.2, NVFP4 15.0: NVFP4 is bound by the decode's ALU work (12.9 us
without the PRMT/LOP3 decode), not by its 36 MB of DRAM traffic.
"""

import cutlass
from cutlass import Float32, Int32, cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from attn_gym.warpdrive.examples.inkling_megakernel import quant
from attn_gym.warpdrive.examples.qwen_megakernel.core import (
    COMPUTE_WARPS,
    ROWS,
    ROWS_PER_WARP,
    TILE_K,
)

BF16 = "bf16"
FORMATS = (BF16, quant.NVFP4, quant.MXFP8)
LANE_WORDS = {BF16: 4, quant.NVFP4: 1, quant.MXFP8: 2}  # weight words per lane, row, subtile
SUBTILE_WORDS = {f: 32 * w for f, w in LANE_WORDS.items()}
# Tile-major scale words per subtile: 8 warps x blocks (each word: a block's 4 warp rows).
BLOCKS = {fmt: TILE_K // columns for fmt, columns in quant.BLOCK_COLUMNS.items()}

assert (ROWS, TILE_K, COMPUTE_WARPS) == (quant.TILE_ROWS, quant.SUBTILE, quant.WARPS)


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
    4-7); the code's sign flips the value's. PRMT reads selector nibbles from the low 16 bits;
    a nibble's top bit (a sign here) would select sign replication, hence the masks."""
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
        "lop3.b32 ha, ha, $4, 0x80808080, 0x78;\n"  # ^ signs of columns 0-3 (bits 8k + 7)
        "shl.b32 t, $4, 4;\n"
        "lop3.b32 hb, hb, t, 0x80808080, 0x78;\n"  # ^ columns 4-7 (bits 8(k - 4) + 3)
        "prmt.b32 $0, la, ha, 0x5140;\n"
        "prmt.b32 $1, la, ha, 0x7362;\n"
        "prmt.b32 $2, lb, hb, 0x5140;\n"
        "prmt.b32 $3, lb, hb, 0x7362;\n"
        "}",
        "=&r,=&r,=&r,=&r,r,r,r,r,r",  # early clobber: outputs are written in turn
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
        "=&r,=&r,=&r,=&r,r,r,r",  # early clobber: $0 is written before $6 is last read
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
        "=&f,=&f,f,f,r,r",  # early clobber: $0 is written before $3 is read
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
