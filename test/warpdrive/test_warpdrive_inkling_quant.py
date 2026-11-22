"""The Inkling 2 Mini quantized experts: the loader's kernel layout (``quant``) decodes to
``ref.dequantize_experts`` exactly."""

import gc

import pytest
import torch

from attn_gym._backends.arch import cuda_runs_sm100_kernels
from attn_gym.warpdrive.examples.inkling_megakernel import ref
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


def random_checkpoint(fmt: str, rows: int, columns: int, seed: int, experts: int = EXPERTS):
    return random_checkpoint_experts(fmt, experts, rows, columns, seed)


# ------------------------------------------------------------------ loader layout


@pytest.mark.parametrize("fmt", [NVFP4, MXFP8])
@pytest.mark.parametrize("columns", [HIDDEN, WIDTH])
def test_layout_round_trip(fmt, columns):
    """Every code, scale and the 4.5-subtile row: the kernel layout decodes to the oracle's
    bf16 weights bit for bit (signed zeros included)."""
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
