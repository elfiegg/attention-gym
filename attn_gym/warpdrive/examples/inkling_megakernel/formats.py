"""The experts formats of the Inkling kernels and the weight-tile geometry that the loader
(``quant``) and the register decode (``quant_gemv``) share. A leaf with no imports: the
PyTorch loader and oracle paths need no CuTeDSL, and kernel modules import no host module."""

BF16, NVFP4, MXFP8 = "bf16", "nvfp4", "mxfp8"
FORMATS = (BF16, NVFP4, MXFP8)
# A weight tile's rows, its subtiles' columns and its warps: ``qwen_megakernel.core``'s ROWS,
# TILE_K and COMPUTE_WARPS (``quant_gemv`` checks they agree).
TILE_ROWS, SUBTILE, WARPS = 32, 256, 8
BLOCK_COLUMNS = {NVFP4: 16, MXFP8: 32}  # columns per scale byte
LANE_WORDS = {NVFP4: 1, MXFP8: 2}  # weight words per lane, row, subtile
SUBTILE_WORDS = {f: 32 * w for f, w in LANE_WORDS.items()}
# Tile-major scale words per subtile: 8 warps x blocks (each word: a block's 4 warp rows).
BLOCKS = {fmt: SUBTILE // columns for fmt, columns in BLOCK_COLUMNS.items()}
