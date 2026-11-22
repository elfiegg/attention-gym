"""The Inkling 2 Mini quantized experts: the loader's kernel layout (``quant``) decodes to
``ref.dequantize_experts`` exactly, and the oracle's MoE runs on a quantized layer
(``quant.experts_out``) with the dense results. The visit launch on them is tested in
``test_warpdrive_inkling_megakernel_quant``."""

from pathlib import Path

import pytest
import torch
from inkling_checkpoint import checkpoint_path

from attn_gym.warpdrive.examples.inkling_megakernel import quant, ref
from attn_gym.warpdrive.examples.inkling_megakernel.quant import (
    MXFP8,
    NVFP4,
    QuantExperts,
    load_quantized_inkling_weights,
    random_checkpoint_experts,
)

pytestmark = pytest.mark.usefixtures("release_cuda_memory")

HIDDEN, WIDTH, EXPERTS = 2304, 1152, 128


def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")


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
    from safetensors import safe_open

    handles = {}
    for shard in Path(checkpoint_path()).glob("model*.safetensors"):
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


def test_load_quantized_model_memory():
    """The whole model with experts as stored: ~20 GB of device memory."""
    require_cuda()
    path = checkpoint_path()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    weights = load_quantized_inkling_weights(path)
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
