"""The Inkling 2 Mini quantized experts: the loader's kernel layout (``quant``) decodes to
``ref.dequantize_experts`` exactly, and the quantized expert GEMV (``quant_gemv``) returns
bit-identical dot products to the same GEMV on the bf16-dequantized weights and to a torch
emulation of its fp32 accumulation order, for NVFP4 and MXFP8, in a CUDA graph and with one
compile across routings; the oracle's MoE runs on a quantized layer (``quant.experts_out``)
with the dense results."""

import gc

import pytest
import torch

from attn_gym._backends.arch import cuda_runs_sm100_kernels
from attn_gym.warpdrive.examples.inkling_megakernel import quant, ref
from attn_gym.warpdrive.examples.inkling_megakernel.quant import (
    MXFP8,
    NVFP4,
    QuantExperts,
    load_quantized_inkling_weights,
    random_checkpoint_experts,
)

MODEL = "thinkingmachines/Inkling-2-Mini"
HIDDEN, WIDTH, EXPERTS, TOP_K = 2304, 1152, 128, 8


@pytest.fixture(autouse=True)
def release_cuda_memory():
    yield
    gc.collect()
    torch.cuda.empty_cache()


def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")


def require_sm100():
    if not cuda_runs_sm100_kernels():
        pytest.skip("the GEMV requires an SM100-family GPU")


def random_checkpoint(fmt, rows, columns, seed, mx_scales=(10, 254), experts=EXPERTS):
    return random_checkpoint_experts(fmt, experts, rows, columns, seed, mx_scales)


# ------------------------------------------------------------------ loader layout


@pytest.mark.parametrize("fmt", [NVFP4, MXFP8])
@pytest.mark.parametrize("columns", [HIDDEN, WIDTH])
def test_layout_round_trip(fmt, columns):
    """Every non-NaN code and scale byte (NVFP4 negative and subnormal scales, MXFP8 scales up
    to overflow) and the 4.5-subtile row: the kernel layout decodes to the oracle's bf16
    weights bit for bit (signed zeros and infinities included)."""
    require_cuda()
    tensors = random_checkpoint(fmt, 64, columns, seed=columns, experts=4)
    q = QuantExperts.from_checkpoint(*tensors)
    assert q.format == fmt and q.columns == columns
    expected = ref.dequantize_experts(*tensors)
    assert torch.equal(q.dequantize().view(torch.int16), expected.view(torch.int16))


def test_mxfp8_rejects_subnormal_scales():
    require_cuda()
    weight, scale, _, _ = random_checkpoint(MXFP8, 32, 256, seed=0, experts=1)
    scale[0, 0, 0] = 9
    with pytest.raises(ValueError, match="subnormal"):
        QuantExperts.from_checkpoint(weight, scale, None, None)


def test_dense_paths_reject_quantized_experts():
    require_cuda()
    q = QuantExperts.from_checkpoint(*random_checkpoint(NVFP4, 32, 256, seed=1, experts=2))
    w = ref.LayerWeights(
        **dict.fromkeys(ref.LayerWeights.__dataclass_fields__, None) | {"w13": q, "w2": q}
    )
    x = torch.zeros(1, 256, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="dense bf16 experts"):
        ref.experts_out(ref.InklingConfig(), w, x, torch.zeros(1, 1, dtype=torch.long), None)


@pytest.fixture(scope="module")
def checkpoint_layers():
    """Expert tensors of layers 0, 19 (NVFP4) and 20, 23 (MXFP8) as stored, read lazily."""
    require_cuda()
    hub = pytest.importorskip("huggingface_hub")
    if not isinstance(hub.try_to_load_from_cache(MODEL, "config.json"), str):
        pytest.skip(f"{MODEL} is not in the local HF cache")
    from safetensors import safe_open

    handles = {}
    for shard in ref.checkpoint_dir(MODEL).glob("model*.safetensors"):
        handle = safe_open(str(shard), "pt", device="cuda")
        handles.update(dict.fromkeys(handle.keys(), handle))

    def tensors(layer: int, name: str):
        prefix = f"model.llm.layers.{layer}.mlp.experts.{name}"
        parts = ("", "_scale", "_amax", "_outer_scale_denominator")
        return tuple(
            handles[prefix + p].get_tensor(prefix + p) if prefix + p in handles else None
            for p in parts
        )

    return tensors


@pytest.mark.parametrize("layer", [0, 19, 20, 23])
def test_checkpoint_layers(checkpoint_layers, layer):
    """Real layers: equal to ``ref.dequantize_experts``, with w13 in checkpoint row order and,
    deinterleaved, in ``ref.load_inkling_weights``' gate-then-up order."""
    for name in ("w13_weight", "w2_weight"):
        tensors = checkpoint_layers(layer, name)
        q = QuantExperts.from_checkpoint(*tensors)
        assert q.format == (NVFP4 if layer < 20 else MXFP8)
        expected = ref.dequantize_experts(*tensors)
        assert torch.equal(q.dequantize().view(torch.int16), expected.view(torch.int16))
        if name == "w13_weight":
            ref_order = expected.unflatten(1, (-1, 2)).transpose(1, 2).flatten(1, 2)
            assert torch.equal(q.dequantize(deinterleave=True), ref_order)


def test_load_quantized_model_memory(checkpoint_layers):
    """The whole model with experts as stored: ~20 GB of device memory."""
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    weights = load_quantized_inkling_weights(MODEL)
    gigabytes = (torch.cuda.memory_allocated() - before) / 1e9
    formats = [layer.w13.format for layer in weights.layers]
    assert formats == [NVFP4] * 20 + [MXFP8] * 4
    assert all(isinstance(layer.w2, QuantExperts) for layer in weights.layers)
    assert 15 < gigabytes < 24, gigabytes


# ------------------------------------------------------------------ oracle on quantized


def test_experts_out_on_quantized_layer():
    """``quant.experts_out`` (the oracle's MoE on a quantized layer, as the chat's prefill runs
    it) equals ``ref.experts_out`` on the dequantized layer bit for bit."""
    require_cuda()
    hidden, width, experts, tokens = 256, 128, 16, 5
    q13, q2 = (
        QuantExperts.from_checkpoint(
            *random_checkpoint(NVFP4, rows, columns, seed, experts=experts)
        )
        for rows, columns, seed in ((2 * width, hidden, 1), (hidden, width, 2))
    )
    fields = dict.fromkeys(ref.LayerWeights.__dataclass_fields__, None)
    w = ref.LayerWeights(**fields | {"w13": q13, "w2": q2})
    dense = ref.LayerWeights(
        **fields | {"w13": q13.dequantize(deinterleave=True), "w2": q2.dequantize()}
    )
    gen = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(tokens, hidden, device="cuda", generator=gen).bfloat16()
    routed = torch.stack(
        [torch.randperm(experts, device="cuda", generator=gen)[:4] for _ in range(tokens)]
    )
    route_weights = torch.rand(tokens, 4, device="cuda", generator=gen)
    config = ref.InklingConfig()
    expected = ref.experts_out(config, dense, x, routed, route_weights)
    out = quant.experts_out(config, w, x, routed, route_weights)
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


# ------------------------------------------------------------------ GEMV


def warp_transpose_reduce(a: torch.Tensor) -> torch.Tensor:
    """``core.warp_transpose_reduce`` for 4 values: ``a`` (groups, 4, 32 lanes) fp32 ->
    (groups, 4), the sums in the kernel's shuffle order."""
    lane = torch.arange(32, device=a.device)
    hi = (lane & 16) != 0
    halves = []
    for j in range(2):
        send = torch.where(hi, a[:, j], a[:, j + 2])
        keep = torch.where(hi, a[:, j + 2], a[:, j])
        halves.append(keep + send[:, lane ^ 16])
    hi = (lane & 8) != 0
    send = torch.where(hi, halves[0], halves[1])
    v = torch.where(hi, halves[1], halves[0]) + send[:, lane ^ 8]
    for offset in (4, 2, 1):
        v = v + v[:, lane ^ offset]
    return v[:, ::8]


def reference_gemv(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """fp32 dot products of bf16 rows ``w`` (rows, K) with bf16 ``x`` (K padded to 256) in the
    engine's order: lane l, columns 8l .. 8l + 7 of each subtile, even and odd columns in two
    accumulators (bf16 products are exact in fp32, so each FMA is one fp32 add), their sum,
    then the warp transpose-reduce over groups of 4 rows."""
    rows, k = w.shape
    subtiles = -(-k // 256)
    wp = torch.zeros(rows, subtiles * 256, device=w.device)
    wp[:, :k] = w.float()
    products = (wp * x.float()[: subtiles * 256]).view(rows, subtiles, 32, 8)
    acc = torch.zeros(rows, 2, 32, device=w.device)
    for kt in range(subtiles):
        for e in range(0, 8, 2):
            acc = acc + products[:, kt, :, e : e + 2].transpose(1, 2)
    lanes = acc[:, 0] + acc[:, 1]  # (rows, 32)
    return warp_transpose_reduce(lanes.view(rows // 4, 4, 32)).reshape(rows)


@pytest.fixture(scope="module")
def routed_inputs():
    gen = torch.Generator(device="cuda").manual_seed(7)
    experts = torch.randperm(EXPERTS, generator=gen, device="cuda")[:TOP_K].sort().values.int()
    x = torch.randn(HIDDEN, generator=gen, device="cuda").bfloat16()
    act = torch.zeros(TOP_K, 1280, device="cuda", dtype=torch.bfloat16)
    act[:, :WIDTH] = torch.randn(TOP_K, WIDTH, generator=gen, device="cuda") * 3
    return experts, x, act


def quant_gemv_module():
    require_sm100()
    pytest.importorskip("cutlass.cute")
    # Imported here: the layout tests need no CuTeDSL.
    from attn_gym.warpdrive.examples.inkling_megakernel import quant_gemv

    return quant_gemv


def expected_gemv(d13, d2, experts, x, act) -> torch.Tensor:
    """``reference_gemv`` of every slot: (gate/up | down, slot, row)."""
    slots = list(enumerate(experts.tolist()))
    return torch.stack(
        [
            torch.stack([reference_gemv(d13[e], x) for _, e in slots]),
            torch.stack([reference_gemv(d2[e], act[s]) for s, e in slots]),
        ]
    )


@pytest.mark.parametrize("fmt", [NVFP4, MXFP8])
def test_quant_gemv_bitwise(fmt, routed_inputs):
    """8 routed experts' gate/up (K 2304) and down (K 1152, 4.5 subtiles) tiles over every
    code and (NVFP4) every scale byte: the quantized kernel equals the bf16 kernel on the
    dequantized weights and the torch emulation bitwise, for every stage shape; a CUDA graph
    replay and a new routing (and runner) reuse one compile."""
    qg = quant_gemv_module()
    experts, x, act = routed_inputs
    mx = (100, 130)  # finite products: an inf would make the dot products NaN
    w13 = QuantExperts.from_checkpoint(*random_checkpoint(fmt, 2 * WIDTH, HIDDEN, 11, mx))
    w2 = QuantExperts.from_checkpoint(*random_checkpoint(fmt, HIDDEN, WIDTH, 12, mx))
    d13, d2 = w13.dequantize(), w2.dequantize()
    expected = expected_gemv(d13, d2, experts, x, act)
    assert torch.equal(qg.QuantGemv(d13, d2).run(experts, x, act), expected)
    for stage_subtiles in (None, 1, 2, 3):  # None: the default, 4
        runner = qg.QuantGemv(w13, w2, stage_subtiles=stage_subtiles)
        out = runner.run(experts, x, act)
        assert torch.equal(out, expected), (stage_subtiles, (out - expected).abs().max())

    runner.out.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner.run(experts, x, act)
    graph.replay()
    assert torch.equal(runner.out, expected)
    info = qg.compile_quant_gemv.cache_info()
    other = experts.flip(0).contiguous()  # another slot order, through a new runner
    out = qg.QuantGemv(w13, w2, stage_subtiles=3).run(other, x, act)
    assert torch.equal(out, expected_gemv(d13, d2, other, x, act))
    after = qg.compile_quant_gemv.cache_info()
    assert (after.currsize, after.misses) == (info.currsize, info.misses)


@pytest.mark.parametrize("layer", [0, 20])
def test_quant_gemv_checkpoint(checkpoint_layers, routed_inputs, layer):
    """Real NVFP4 (layer 0) and MXFP8 (layer 20) experts: bit-identical to the bf16 kernel on
    the oracle's dequantized weights."""
    qg = quant_gemv_module()
    experts, x, act = routed_inputs
    weights = [checkpoint_layers(layer, name) for name in ("w13_weight", "w2_weight")]
    w13, w2 = (QuantExperts.from_checkpoint(*tensors) for tensors in weights)
    dense = qg.QuantGemv(*(ref.dequantize_experts(*tensors) for tensors in weights))
    expected = dense.run(experts, x, act).clone()
    assert torch.equal(qg.QuantGemv(w13, w2).run(experts, x, act), expected)
