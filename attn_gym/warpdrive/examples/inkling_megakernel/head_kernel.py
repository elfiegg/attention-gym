"""The end of an Inkling 2 Mini decode step as one persistent warpdrive launch
(``InklingHeadKernel``) and its compile function; ``megakernel`` holds the host side.

Batch one, one token. The launch implements ``ref.hidden_states``' last block residual and
``ref.logits_of``::

    every CTA   prologue: block residual of the final query over [blocks..., h], final RMSNorm
                -> bf16 x (SMEM)
    LMHEAD      32 LM-head rows x hidden -> fp32 logits tanh(bf16(row . x) / cap) * cap

The LM head's rows need no padding: TMA zero-fills the last tile's rows past the vocabulary,
whose logits are not stored. Items depend on nothing but the prologue: no tagged words, no
epoch. Roles as ``mlp_kernel``: warp 8 ``load`` streams 32 x 256 weight stages, warps 0-7
``compute`` run ``CudaCoreGemv``.
"""

from dataclasses import dataclass

from cuda.bindings import driver as cuda
from cutlass import BFloat16, Float32, Int32, cute
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
    RED_VALUES,
    REDUCTIONS,
    SMEM_BYTES,
    SMEM_HEADROOM,
    WORDS,
    InklingKernel,
    fp32_reciprocal,
    load_tiles,
)
from attn_gym.warpdrive.examples.qwen_megakernel.core import (
    COMPUTE_THREADS,
    COMPUTE_WARPS,
    MAX_STAGES,
    ROWS,
    STAGE_BYTES,
    TILE_K,
    CudaCoreGemv,
    InstructionTable,
    bf16_round,
)

LMHEAD = 0


@dataclass(frozen=True)
class HeadParams:
    tma: tuple  # TMA atom of the LM head
    gmem: tuple  # its tile partition: (TMA, row tiles, k tiles)
    sW: StageView
    xs: cute.Tensor  # (1, hidden) bf16: the final normalized state
    red: cute.Tensor  # (REDUCTIONS, COMPUTE_WARPS, RED_VALUES) fp32 prologue reductions
    blocks: tuple  # MAX_BLOCKS (hidden,) bf16 completed blocks; the first num_blocks are read
    num_blocks: Int32
    h: cute.Tensor  # (hidden,) bf16 the last visit's output (the partial block)
    query: cute.Tensor  # (1, hidden) fp32 final block-residual query
    norms: cute.Tensor  # (1, hidden) bf16 final RMSNorm weight
    logits: cute.Tensor  # (vocab,) fp32 output


class InklingHeadKernel(InklingKernel):
    """The final block residual, norm, LM head and logit softcap (see the module docstring).
    Constructor arguments are the model sizes and constants the kernel reads; ``stages``:
    weight ring depth in 16 KB stages."""

    def __init__(
        self,
        hidden: int,
        vocab: int,
        rms_eps: float,
        softcap: float,
        stages: int | None = None,
    ):
        if hidden % COMPUTE_THREADS or hidden % TILE_K:
            raise ValueError(f"hidden size must be a multiple of {COMPUTE_THREADS} and {TILE_K}")
        self.hidden, self.vocab = hidden, vocab
        self.rms_eps, self.softcap = rms_eps, softcap
        self.hidden_k_tiles = hidden // TILE_K
        self.row_tiles = -(-vocab // ROWS)
        fixed = 2 * hidden + 4 * REDUCTIONS * COMPUTE_WARPS * RED_VALUES + SMEM_HEADROOM
        fit = (SMEM_BYTES - fixed) // STAGE_BYTES
        stages = min(MAX_STAGES, fit) if stages is None else stages
        if not 2 <= stages <= fit:
            raise ValueError(f"stages must be in [2, {fit}], got {stages}")
        self.stages = stages
        self.smem_plan = SmemPlan()
        w = self.smem_plan.buffer("W", BFloat16, align=1024)
        self.smem_plan.buffer("X", BFloat16, align=16)
        self.smem_plan.buffer("RED", Float32, align=16)
        # Compute reads stages with generic loads before TMA refills them; the layer emits the
        # fence.proxy.async on a Warps release of a Tma-filled ring.
        self.w = Ring(stages, Tma(STAGE_BYTES), Warps, buffer={"W": w})
        self.csync = NamedBarrier(threads=COMPUTE_THREADS)
        self.engine = CudaCoreGemv()
        self.work = WorkStream(mapping=InstructionTable(words=WORDS), schedule=StaticPersistent())

    def get_name(self) -> str:
        return (
            f"warpdrive_inkling_head_h{self.hidden}_v{self.vocab}_s{self.stages}{self.trace_key()}"
        )

    @role(warps=COMPUTE_WARPS)
    @cute.jit
    def load(self, p: HeadParams, *, w: Producer, work: WorkCursor):
        w = w.views(W=p.sW)
        while work.valid:
            _op, tile, _b, _row = work.item
            w = load_tiles(w, p.tma[0], p.gmem[0], tile, 0, self.hidden_k_tiles, "W")
            work.advance()
        return w, work

    @role(warps=range(COMPUTE_WARPS))
    @cute.jit
    def compute(self, p: HeadParams, *, w: Consumer, work: WorkCursor, me: Local):
        tid, warp, lane = me.tid, me.warp, me.lane
        x = self.load_states(p, p.h, tid)
        self.normalize_states(p, x, p.query, 1, Int32(1), tid, warp, lane)
        cap = self.softcap
        while work.valid:
            _op, tile, _b, _row = work.item
            vals, row, chans = self.engine.tile(p, (w,), (), 0, self.hidden_k_tiles, warp, lane)
            w = chans[0]
            index = tile * ROWS + row
            if self.engine.owns_row(warp, lane) & (index < self.vocab):
                # ref.logits_of: the bf16 LM-head output, softcapped in fp32.
                logit = bf16_round(vals[0]) * fp32_reciprocal(cap)
                p.logits[index] = cute.math.tanh(logit) * cap
            work.advance()
        return w, work

    @cute.kernel
    def kernel(self, tma, gtensor, tensors, blocks, bound, num_blocks):
        layouts = {
            "W": cute.make_layout((ROWS, TILE_K, self.stages), stride=(TILE_K, 1, ROWS * TILE_K)),
            "X": cute.make_layout((1, self.hidden), stride=(self.hidden, 1)),
            "RED": cute.make_layout(
                (REDUCTIONS, COMPUTE_WARPS, RED_VALUES),
                stride=(COMPUTE_WARPS * RED_VALUES, RED_VALUES, 1),
            ),
        }
        self.begin_roles(work=bound, layouts=layouts)
        # Materialize the argument tensors in converged code: a first use inside one role's
        # branch would not dominate the other roles' uses.
        h, query, norms, logits = (cute.make_tensor(t.iterator, t.layout) for t in tensors)
        blocks = tuple(cute.make_tensor(t.iterator, t.layout) for t in blocks)
        sW = self.buffer("W")
        tiles = cute.local_tile(gtensor, (ROWS, TILE_K), (None, None))
        part = cpasync.tma_partition(
            tma, 0, cute.make_layout(1), cute.group_modes(sW, 0, 2), cute.group_modes(tiles, 0, 2)
        )
        params = HeadParams(
            tma=(tma,),
            gmem=(part[1],),
            sW=StageView(part[0], axis=1),
            xs=self.buffer("X"),
            red=self.buffer("RED"),
            blocks=blocks,
            num_blocks=num_blocks,
            h=h,
            query=query,
            norms=norms,
            logits=logits,
        )
        self.run_roles(params)

    @cute.jit
    def __call__(
        self,
        lm_head: cute.Tensor,
        query: cute.Tensor,
        norms: cute.Tensor,
        h: cute.Tensor,
        block0: cute.Tensor,
        block1: cute.Tensor,
        block2: cute.Tensor,
        block3: cute.Tensor,
        num_blocks: Int32,
        logits: cute.Tensor,
        table: cute.Tensor,
        workspace: cute.Tensor,
        max_ctas: Int32,
        stream: cuda.CUstream,
    ):
        smem = cute.make_layout((ROWS, TILE_K), stride=(TILE_K, 1))
        atom, gtensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), lm_head, smem, (ROWS, TILE_K)
        )
        bound = self.work.bind(table, workspace, max_ctas)
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            atom,
            gtensor,
            (h, query, norms, logits),
            (block0, block1, block2, block3),
            bound,
            num_blocks,
        ).launch(**self.launch_config(bound), stream=stream)


@jit_cache
def compile_head_kernel(op: InklingHeadKernel):
    """One compile serves every step: the LM head, final query and norm, states and logits are
    arguments; the block count is read at run time."""
    bf16, f32 = BFloat16, Float32
    h = op.hidden
    vector = [make_fake_compact_tensor(bf16, (h,), assumed_align=2) for _ in range(MAX_BLOCKS)]
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(bf16, (op.vocab, h)),
        make_fake_compact_tensor(f32, (1, h), assumed_align=4),
        make_fake_compact_tensor(bf16, (1, h), assumed_align=2),
        make_fake_compact_tensor(bf16, (h,), assumed_align=2),  # h
        *vector,
        Int32(1),
        make_fake_compact_tensor(f32, (op.vocab,), assumed_align=4),
        make_fake_compact_tensor(Int32, (cute.sym_int(), WORDS)),
        make_fake_compact_tensor(Int32, (cute.sym_int(),)),
        Int32(1),
        name=op.get_name(),
    )
