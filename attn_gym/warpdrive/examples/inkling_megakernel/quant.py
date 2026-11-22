"""The checkpoint's quantized experts in the layout of the expert GEMV (``quant_gemv``), and
their exact bf16 dequantization.

Inkling 2 Mini stores the experts of layers 0-19 as NVFP4 and of layers 20-23 as MXFP8
(``ref.dequantize_experts`` defines the values). ``QuantExperts.from_checkpoint`` keeps them
quantized, in the checkpoint's row order: ``w13`` rows interleave gate and up (row 2j is gate
row j, 2j + 1 up row j), so a 32-row tile holds 16 (gate, up) pairs, as the MLP kernel's
strided view of the bf16 weights does. ``load_quantized_inkling_weights`` loads the model with
such experts: 19.0 GB of device memory, 15.7 GB of it experts (bf16 experts: 48.9 GB). The
oracle's dense code paths reject them; ``forward_tokens`` runs the oracle on such a model by
dequantizing only each MLP's routed experts, and ``dequantized_weights`` returns the bf16 model.

The layout follows the GEMV: 32-row x 256-column subtiles, warp w of 8 owning rows
4w .. 4w + 3 and lane l columns 8l .. 8l + 7 of every subtile.

- NVFP4 weights keep two codes per byte, but each little-endian 32-bit word (a lane's 8
  columns ``k = 0..7``) is re-encoded for the kernel's byte-permute decode: nibble k holds the
  3-bit magnitude of column k in bits 4k .. 4k + 2, and the sign of column k sits at bit
  8k + 7 for k < 4 and 8(k - 4) + 3 for k >= 4 (the nibbles' top bits). ``lut[e, s]`` holds
  the bf16 values ``bf16((e2m1[i] * s) * amax_e / denominator)`` of the 8 non-negative codes
  i for fp8 scale byte s as byte planes: four int32 words, the low bytes of codes 0-3, of
  4-7, then the high bytes of codes 0-3, of 4-7; the code's sign flips the value's sign bit.
  The table evaluates ``ref.dequantize_experts``' fp32 operations and negation is exact, so
  the decode is exact for every non-NaN scale byte and finite factor; ``from_checkpoint``
  rejects the others (a NaN's sign bit would differ from the oracle's). The per-expert factor
  stays fp32 (amax / denominator) as in the oracle; HF's loader rounds amax to bf16 first.
- MXFP8 weights are the checkpoint's e4m3 bytes; the kernel converts them to bf16 and
  multiplies by the bf16 power of two ``2 ** (s - 127)`` of the scale byte s. That is the
  oracle's value whenever the product is a normal number or overflows, which
  ``from_checkpoint`` checks (every scale byte in [10, 254]: 255 is E8M0's NaN; the
  checkpoint's are 113 .. 118).
- Scales of both formats are tile-major bytes ``(experts, rows / 32, subtiles, 8, blocks, 4)``:
  per subtile and warp, block by block, the warp's 4 rows; a lane reads its 4 rows' scales as
  one word. ``blocks`` is 16 (NVFP4: 16 columns each) or 8 (MXFP8: 32 columns). A partial last
  subtile (the down projection's 1152 columns are 4.5 subtiles) has zero scales past the end.
"""

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import torch

from attn_gym.warpdrive.examples.inkling_megakernel import ref
from attn_gym.warpdrive.examples.inkling_megakernel.quant_gemv import (
    BLOCK_COLUMNS,
    MXFP8,
    NVFP4,
    SUBTILE,
    TILE_ROWS,
    WARPS,
)

WARP_ROWS = TILE_ROWS // WARPS
MIN_MX_SCALE = 10  # smallest MXFP8 scale byte whose products with e4m3 values are normal
CHUNK = 8  # experts per conversion step: bounds the int64 temporaries


def e2m1_value_table(factor: torch.Tensor) -> torch.Tensor:
    """bf16 ``(experts, 256, 8)`` values of the 8 non-negative E2M1 codes for every fp8 scale
    byte: ``ref.dequantize_experts``' ``(e2m1 * scale) * factor`` in fp32, rounded once."""
    e2m1 = torch.tensor(ref.E2M1_VALUES[:8], device=factor.device)
    scales = torch.arange(256, device=factor.device, dtype=torch.uint8).view(torch.float8_e4m3fn)
    products = e2m1 * scales.float()[:, None]  # (256, 8), exact
    return (products * factor[:, None, None]).to(torch.bfloat16)


def magnitude_planes(table: torch.Tensor) -> torch.Tensor:
    """int32 ``(..., 4)`` byte planes of bf16 ``(..., 8)`` magnitudes (see the module)."""
    bits = table.view(torch.int16).long() & 0xFFFF
    planes = torch.stack((bits & 0xFF, bits >> 8), dim=-2).unflatten(-1, (2, 4))  # (.., 2, 2, 4)
    words = (planes << (8 * torch.arange(4, device=table.device))).sum(-1)
    return to_int32(words.flatten(-2))


def to_int32(words: torch.Tensor) -> torch.Tensor:
    """Unsigned 32-bit values held in int64 as int32 bit patterns."""
    return torch.where(words >= 2**31, words - 2**32, words).int()


def sign_bit(k: int) -> int:
    """The bit of column ``k``'s sign in a re-encoded NVFP4 word (see the module)."""
    return 8 * k + 7 if k < 4 else 8 * (k - 4) + 3


def encode_nvfp4(codes: torch.Tensor) -> torch.Tensor:
    """Checkpoint NVFP4 bytes ``(..., columns / 2)`` re-encoded for the kernel (see the
    module): per 32-bit word, magnitudes in nibbles and signs at the nibbles' top bits."""
    words = codes.contiguous().view(torch.int32).long() & 0xFFFFFFFF
    out = torch.zeros_like(words)
    for k in range(8):
        code = (words >> (4 * k)) & 0xF
        out |= ((code & 7) << (4 * k)) | ((code >> 3) << sign_bit(k))
    return to_int32(out).view(torch.uint8).view(codes.shape)


def decode_nvfp4(encoded: torch.Tensor) -> torch.Tensor:
    """The checkpoint's 4-bit codes ``(..., columns)`` (sign << 3 | magnitude, column order)
    of re-encoded bytes ``(..., columns / 2)``."""
    words = encoded.contiguous().view(torch.int32).long() & 0xFFFFFFFF
    codes = []
    for k in range(8):
        codes.append(((words >> (4 * k)) & 7) | (((words >> sign_bit(k)) & 1) << 3))
    return torch.stack(codes, dim=-1).flatten(-2)


def tile_scales(scale: torch.Tensor, block_columns: int) -> torch.Tensor:
    """Row-major ``(experts, rows, blocks)`` scale bytes in the tile-major kernel layout."""
    experts, rows, blocks = scale.shape
    per_subtile = SUBTILE // block_columns
    subtiles = -(-blocks // per_subtile)
    padded = torch.zeros(
        experts, rows, subtiles * per_subtile, dtype=torch.uint8, device=scale.device
    )
    padded[..., :blocks] = scale.view(torch.uint8)
    tiles = padded.view(experts, rows // TILE_ROWS, WARPS, WARP_ROWS, subtiles, per_subtile)
    return tiles.permute(0, 1, 4, 2, 5, 3).contiguous()


def untile_scales(tiled: torch.Tensor, blocks: int) -> torch.Tensor:
    """The inverse of ``tile_scales``: row-major ``(experts, rows, blocks)`` bytes."""
    experts, row_tiles = tiled.shape[:2]
    rows = tiled.permute(0, 1, 3, 5, 2, 4).reshape(experts, row_tiles * TILE_ROWS, -1)
    return rows[..., :blocks]


@dataclass(frozen=True)
class QuantExperts:
    """One layer's expert matrix ``(experts, rows, columns)`` in the kernel layout (see the
    module). ``weight``: uint8 ``(experts, rows, columns / 2)`` re-encoded NVFP4 or
    ``(experts, rows, columns)`` e4m3 bytes; ``scale``: tile-major uint8 scales; ``lut``: int32
    ``(experts, 256, 4)`` NVFP4 magnitude planes (None for MXFP8). Not a tensor: dense code
    paths (``ref.experts_out``) reject it."""

    format: str
    columns: int
    weight: torch.Tensor
    scale: torch.Tensor
    lut: torch.Tensor | None = None

    @property
    def experts(self) -> int:
        return self.weight.shape[0]

    @property
    def rows(self) -> int:
        return self.weight.shape[1]

    @property
    def subtiles(self) -> int:
        return self.scale.shape[2]

    @property
    def nbytes(self) -> int:
        tensors = (self.weight, self.scale) + (() if self.lut is None else (self.lut,))
        return sum(t.numel() * t.element_size() for t in tensors)

    @classmethod
    def from_checkpoint(
        cls,
        weight: torch.Tensor,
        scale: torch.Tensor,
        amax: torch.Tensor | None,
        denominator: torch.Tensor | None,
    ) -> "QuantExperts":
        """The kernel layout of checkpoint tensors as ``ref.dequantize_experts`` takes them:
        NVFP4 when ``amax`` is given, else MXFP8."""
        experts, rows = weight.shape[:2]
        if scale is None:
            raise ValueError("unquantized experts: dequantize them with ref.dequantize_experts")
        if rows % TILE_ROWS:
            raise ValueError(f"rows must be a multiple of {TILE_ROWS}, got {rows}")
        if amax is None:
            if int(scale.min()) < MIN_MX_SCALE:
                raise ValueError(
                    f"MXFP8 scale bytes below {MIN_MX_SCALE} give subnormal products, which the "
                    "kernel's bf16 multiply does not reproduce"
                )
            if int(scale.max()) == 255:
                raise ValueError("MXFP8 scale byte 255 is NaN (E8M0)")
            scales = tile_scales(scale, BLOCK_COLUMNS[MXFP8])
            return cls(MXFP8, weight.shape[2], weight.view(torch.uint8), scales)
        factor = amax.float() / denominator.float()
        if not bool(torch.isfinite(factor).all()):
            raise ValueError("NVFP4 amax / denominator must be finite")
        if bool(scale.float().isnan().any()):
            raise ValueError("NVFP4 scales must not be NaN")
        encoded = torch.empty_like(weight, dtype=torch.uint8)
        for start in range(0, experts, CHUNK):
            encoded[start : start + CHUNK] = encode_nvfp4(weight[start : start + CHUNK])
        lut = magnitude_planes(e2m1_value_table(factor))
        scales = tile_scales(scale, BLOCK_COLUMNS[NVFP4])
        return cls(NVFP4, 2 * weight.shape[2], encoded, scales, lut)

    def select(self, experts: torch.Tensor) -> "QuantExperts":
        """The experts ``experts`` (indices), in that order."""
        lut = None if self.lut is None else self.lut[experts]
        return QuantExperts(
            self.format, self.columns, self.weight[experts], self.scale[experts], lut
        )

    def dequantize(self, deinterleave: bool = False) -> torch.Tensor:
        """bf16 ``(experts, rows, columns)`` weights decoded from the kernel layout, equal to
        ``ref.dequantize_experts`` of the checkpoint tensors. Checkpoint row order, or with
        ``deinterleave`` gate rows then up rows (``ref.load_inkling_weights``' ``w13``)."""
        block = BLOCK_COLUMNS[self.format]
        scale = untile_scales(self.scale, self.columns // block)
        out = torch.empty(
            self.experts, self.rows, self.columns, dtype=torch.bfloat16, device=self.weight.device
        )
        for start in range(0, self.experts, CHUNK):
            chunk = slice(start, start + CHUNK)
            s = scale[chunk].long()
            if self.format == MXFP8:
                values = self.weight[chunk].view(torch.float8_e4m3fn).float()
                values = values.unflatten(-1, (-1, block)) * torch.exp2(s.float() - 127)[..., None]
                out[chunk] = values.flatten(-2).to(torch.bfloat16)
                continue
            codes = decode_nvfp4(self.weight[chunk])  # (e, rows, columns)
            magnitude = (codes & 7).unflatten(-1, (-1, block))
            # The kernel's lookup: byte planes of lut[e, s], low byte then high byte ^ sign.
            planes = self.lut[chunk].long() & 0xFFFFFFFF  # (e, 256, 4)
            plane_bytes = (planes[..., None] >> (8 * torch.arange(4, device=planes.device))) & 0xFF
            lo_hi = plane_bytes.view(*planes.shape[:2], 2, 8)  # (e, 256, lo|hi, code)
            e = torch.arange(magnitude.shape[0], device=planes.device)[:, None, None, None]
            lo = lo_hi[e, s[..., None], 0, magnitude]
            hi = lo_hi[e, s[..., None], 1, magnitude] ^ (
                (codes >> 3).unflatten(-1, (-1, block)) << 7
            )
            bits = (hi << 8 | lo).flatten(-2)
            out[chunk] = (
                torch.where(bits >= 2**15, bits - 2**16, bits).short().view(torch.bfloat16)
            )
        if deinterleave:
            out = out.unflatten(1, (-1, 2)).transpose(1, 2).flatten(1, 2).contiguous()
        return out


def random_checkpoint_experts(
    fmt: str,
    experts: int,
    rows: int,
    columns: int,
    seed: int,
    mx_scales: tuple[int, int] = (MIN_MX_SCALE, 254),
    device="cuda",
) -> tuple:
    """Random checkpoint-format expert tensors ``(weight, scale, amax, denominator)`` as
    ``ref.dequantize_experts`` takes them, over every non-NaN code: NVFP4 with every non-NaN
    fp8 scale byte (negative and subnormal ones included) and amax in [0.1, 0.6); MXFP8 with
    scale bytes in ``mx_scales`` (inclusive; large ones overflow to inf; the checkpoint's are
    113 .. 118)."""
    gen = torch.Generator(device=device).manual_seed(seed)

    def random_bytes(*shape, low=0, high=256):
        return torch.randint(low, high, shape, generator=gen, device=device, dtype=torch.uint8)

    if fmt == NVFP4:
        scale = random_bytes(experts, rows, columns // 16)
        scale[(scale & 0x7F) == 0x7F] = 0x38  # e4m3 NaN -> 1.0
        amax = torch.rand(experts, generator=gen, device=device) * 0.5 + 0.1
        weight = random_bytes(experts, rows, columns // 2)
        denominator = torch.tensor([2688.0], device=device)
        return weight, scale.view(torch.float8_e4m3fn), amax, denominator
    weight = random_bytes(experts, rows, columns)
    weight[(weight & 0x7F) == 0x7F] = 0x3C  # e4m3 NaN codes -> 1.5
    scale = random_bytes(experts, rows, columns // 32, low=mx_scales[0], high=mx_scales[1] + 1)
    return weight.view(torch.float8_e4m3fn), scale, None, None


def load_quantized_inkling_weights(
    path_or_repo: str | Path = "thinkingmachines/Inkling-2-Mini", device="cuda"
) -> ref.InklingWeights:
    """``ref.load_inkling_weights`` with every layer's ``w13``/``w2`` a ``QuantExperts`` in
    checkpoint row order (bf16 checkpoint experts, if any, are rejected)."""
    return ref.load_inkling_weights(path_or_repo, device, experts=QuantExperts.from_checkpoint)


def dense_layer(w: ref.LayerWeights, experts: torch.Tensor | None = None) -> ref.LayerWeights:
    """``w`` with bf16 experts in ``ref.load_inkling_weights``' layout: its ``QuantExperts``
    dequantized (only ``experts``, in that order, if given), dense ones as they are."""
    if isinstance(w.w13, torch.Tensor):
        return w
    w13, w2 = (w.w13, w.w2) if experts is None else (w.w13.select(experts), w.w2.select(experts))
    return dataclasses.replace(w, w13=w13.dequantize(deinterleave=True), w2=w2.dequantize())


def dequantized_weights(weights: ref.InklingWeights) -> ref.InklingWeights:
    """``weights`` with every layer's experts in bf16 (``dense_layer``)."""
    return dataclasses.replace(weights, layers=tuple(dense_layer(w) for w in weights.layers))


def experts_out(config: ref.InklingConfig, w: ref.LayerWeights, x, experts, weights):
    """``ref.experts_out`` on a layer whose experts may be ``QuantExperts``: the routed experts
    dequantized (ascending, so the sum keeps the oracle's order) give the same result."""
    if isinstance(w.w13, torch.Tensor):
        return ref.experts_out(config, w, x, experts, weights)
    used = experts.unique()
    local = torch.searchsorted(used, experts)
    return ref.experts_out(config, dense_layer(w, used), x, local, weights)


QUANTIZED = ref.Kernels(experts=experts_out)


def forward_tokens(
    weights: ref.InklingWeights, tokens: torch.Tensor, pos: int, cache: ref.InklingCache
) -> torch.Tensor:
    """``ref.forward_tokens`` (the last token's fp32 logits) on a model whose experts may be
    ``QuantExperts`` (``experts_out``), with the oracle's results."""
    return ref.forward_tokens(weights, tokens, pos, cache, kernels=QUANTIZED)
