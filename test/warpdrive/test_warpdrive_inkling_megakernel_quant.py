"""Quantized experts in the Inkling 2 Mini visit launch (``mlp_half``): streaming the
checkpoint's NVFP4 and MXFP8 experts as stored (``quant.QuantExperts``), every launch output
(visit output, post-attention residual, cache and convolution ring rows, routing) equals the
same launch on the bf16-dequantized experts bit for bit: on random weights over every code
(and every NVFP4 scale byte), in plain and message visits of sliding and full layers, in a
CUDA graph replay with one compile per launch kind across positions, and on the checkpoint
over a greedy decode (every visit of every step)."""

import dataclasses
import gc

import pytest
import torch

pytest.importorskip("cutlass.cute")

from test_warpdrive_inkling_megakernel_messages import random_core_layer, random_pass
from test_warpdrive_inkling_megakernel_visit import (
    CONFIG,
    clone,
    random_cache,
    random_inputs,
    random_layer,
    require_sm100,
)

from attn_gym.warpdrive.examples.inkling_megakernel import quant, ref
from attn_gym.warpdrive.examples.inkling_megakernel.head_kernel import compile_head_kernel
from attn_gym.warpdrive.examples.inkling_megakernel.megakernel import (
    InklingDecoder,
    InklingMsgMegakernel,
    InklingVisitMegakernel,
    MessageGates,
    experts_format,
)
from attn_gym.warpdrive.examples.inkling_megakernel.msg_kernel import compile_msg_kernel
from attn_gym.warpdrive.examples.inkling_megakernel.quant import MXFP8, NVFP4
from attn_gym.warpdrive.examples.inkling_megakernel.visit_kernel import compile_visit_kernel

MODEL = "thinkingmachines/Inkling-2-Mini"
# MXFP8 scale bytes with finite products: an inf weight would make dot products NaN.
MX_SCALES = (100, 130)
STEPS = 32
COMPILES = (compile_visit_kernel, compile_msg_kernel, compile_head_kernel)


@pytest.fixture(autouse=True)
def release_cuda_memory():
    yield
    gc.collect()
    torch.cuda.empty_cache()


def with_quantized_experts(w: ref.LayerWeights, fmt: str, seed: int):
    """``(quantized, dense)``: ``w`` with random checkpoint-format experts of ``fmt`` at
    ``CONFIG``'s shapes, and the same layer with them dequantized."""
    c = CONFIG
    shapes = (
        (2 * c.moe_intermediate_size, c.hidden_size),
        (c.hidden_size, c.moe_intermediate_size),
    )
    w13, w2 = (
        quant.QuantExperts.from_checkpoint(
            *quant.random_checkpoint_experts(
                fmt, c.num_experts, rows, columns, seed + i, MX_SCALES
            )
        )
        for i, (rows, columns) in enumerate(shapes)
    )
    quantized = dataclasses.replace(w, w13=w13, w2=w2)
    return quantized, quant.dense_layer(quantized)


def launch(runner, w, layer, h, states, cache, positions, gates=None):
    """One visit launch on a copy of ``cache``: its output, residual, routing and cache."""
    cache = clone(cache)
    out = runner(w, layer, 0, h, states, cache, positions, gates=gates).clone()
    return out, runner.residual().clone(), runner.routed_experts(w), cache


def assert_bitwise(mine, theirs):
    out, residual, experts, cache = mine
    assert torch.equal(out.view(torch.int16), theirs[0].view(torch.int16))
    assert torch.equal(residual.view(torch.int32), theirs[1].view(torch.int32))
    assert torch.equal(experts, theirs[2])
    for name, tensor in vars(cache).items():
        assert torch.equal(tensor, getattr(theirs[3], name)), name


@pytest.mark.parametrize(
    "fmt,layer,blocks",
    # Sliding layer 0 (a block boundary without a query), sliding and full layers.
    [(NVFP4, 1, 1), (NVFP4, 5, 1), (MXFP8, 0, 0), (MXFP8, 23, 4)],
)
@torch.no_grad()
def test_plain_visit_random_weights(fmt, layer, blocks):
    """Positions 0, before and past the sliding window; the down rows' half subtile."""
    require_sm100()
    quantized, dense = with_quantized_experts(random_layer(CONFIG, layer, seed=layer), fmt, 10)
    runners = [InklingVisitMegakernel(CONFIG, (w,)) for w in (quantized, dense)]
    cache = random_cache(CONFIG, visits=1, max_len=1152, seed=layer)
    for pos in (0, 70, 1100):
        h, states = random_inputs(CONFIG, blocks, seed=pos)
        positions = torch.tensor([pos], device="cuda")
        mine, theirs = (
            launch(r, w, layer, h, states, cache, positions)
            for r, w in zip(runners, (quantized, dense))
        )
        assert_bitwise(mine, theirs)


@pytest.mark.parametrize("layer", [7, 11])
@torch.no_grad()
def test_message_visit_random_weights(layer):
    """Core visits after the first pass (sliding layer 7, full layer 11) with NVFP4 experts,
    as the checkpoint's."""
    require_sm100()
    w = random_core_layer(CONFIG, layer, seed=layer)
    quantized, dense = with_quantized_experts(w, NVFP4, 20)
    queries, block, partial = random_pass(CONFIG, 1, seed=layer)
    gates = MessageGates((w,), queries)
    InklingMsgMegakernel(CONFIG, gates)(block, partial)
    runners = [InklingVisitMegakernel(CONFIG, (w,)) for w in (quantized, dense)]
    cache = random_cache(CONFIG, visits=1, max_len=1152, seed=layer)
    h, states = random_inputs(CONFIG, 2, seed=layer)
    positions = torch.tensor([1100], device="cuda")
    mine, theirs = (
        launch(r, w, layer, h, states, cache, positions, gates.layer(0))
        for r, w in zip(runners, (quantized, dense))
    )
    assert_bitwise(mine, theirs)


@pytest.mark.parametrize("fmt", [NVFP4, MXFP8])
@torch.no_grad()
def test_graph_replay_and_compile_reuse(fmt):
    """A quantized launch replays bit-identically to eager launches in a CUDA graph at device
    positions that change the KV splits; positions, block counts and cache capacities compile
    nothing new."""
    require_sm100()
    quantized, _ = with_quantized_experts(random_layer(CONFIG, 1, seed=1), fmt, 30)
    runner = InklingVisitMegakernel(CONFIG, (quantized,))
    cache = random_cache(CONFIG, visits=1, max_len=1152, seed=1)
    initial = clone(cache)
    h, states = random_inputs(CONFIG, 1, seed=2)
    positions = torch.tensor([1030], device="cuda")
    out = torch.empty(CONFIG.hidden_size, device="cuda", dtype=torch.bfloat16)
    runner(quantized, 1, 0, h, states, cache, positions, out)
    entries = compile_visit_kernel.cache_info().currsize
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runner(quantized, 1, 0, h, states, cache, positions, out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            runner(quantized, 1, 0, h, states, cache, positions, out)
    torch.cuda.current_stream().wait_stream(stream)
    for pos in (1030, 70):
        positions.fill_(pos)
        expected_cache = clone(initial)
        expected = runner(quantized, 1, 0, h, states, expected_cache, positions).clone()
        for name, tensor in vars(cache).items():
            tensor.copy_(getattr(initial, name))
        out.zero_()
        graph.replay()
        assert torch.equal(out, expected), pos
        for name, tensor in vars(cache).items():
            assert torch.equal(tensor, getattr(expected_cache, name)), (pos, name)
    other = random_cache(CONFIG, visits=1, max_len=2048, seed=3)
    for pos, blocks, target in [(0, 0, cache), (64, 4, cache), (2000, 2, other)]:
        positions.fill_(pos)
        h, states = random_inputs(CONFIG, blocks, seed=pos)
        runner(quantized, 1, 0, h, states, target, positions)
    assert compile_visit_kernel.cache_info().currsize == entries


class ResidualRecorder:
    """A visit runner that also keeps each launch's post-attention residual."""

    def __init__(self, runner: InklingVisitMegakernel):
        self.runner, self.residuals = runner, []

    def __call__(self, *args, **kwargs):
        out = self.runner(*args, **kwargs)
        self.residuals.append(self.runner.residual().clone())
        return out


@pytest.mark.xdist_group("inkling-checkpoint")
@torch.no_grad()
def test_checkpoint_greedy_decode():
    """``STEPS`` greedy steps after a prefill on Inkling 2 Mini, from one cache: the decoder on
    the stored experts (NVFP4 layers 0-19, MXFP8 20-23) and on their bf16 dequantization give
    the same logits, visit outputs and residuals (all 48 visits, message visits included),
    caches and tokens bit for bit; positions compile nothing after the first step; a CUDA
    graph of the quantized step replays bit-identically to its eager launches. Skips unless
    the checkpoint is cached; never downloads."""
    require_sm100()
    hub = pytest.importorskip("huggingface_hub")
    if not isinstance(hub.try_to_load_from_cache(MODEL, "config.json"), str):
        pytest.skip(f"{MODEL} is not in the local HF cache")
    weights = quant.load_quantized_inkling_weights(MODEL)
    assert [experts_format(w) for w in weights.layers] == [NVFP4] * 20 + [MXFP8] * 4
    dense = quant.dequantized_weights(weights)
    config = weights.config
    prompt = torch.arange(1000, 1040, device="cuda")
    caches = [ref.InklingCache.empty(config, max_len=256)]
    token = ref.forward_tokens(dense, prompt, 0, caches[0]).argmax().reshape(1)
    caches.append(clone(caches[0]))
    decoders = [InklingDecoder(w) for w in (weights, dense)]
    recorders = []
    for decoder in decoders:
        decoder.visit = ResidualRecorder(decoder.visit)
        recorders.append(decoder.visit)
    pos = torch.tensor([prompt.numel()], device="cuda", dtype=torch.int32)
    counts = None
    for step in range(STEPS):
        logits = [d(token, pos, c).clone() for d, c in zip(decoders, caches)]
        assert torch.equal(logits[0], logits[1]), step
        assert torch.equal(decoders[0].states, decoders[1].states), step
        assert all(torch.equal(a, b) for a, b in zip(*(r.residuals for r in recorders))), step
        for name, tensor in vars(caches[0]).items():
            assert torch.equal(tensor, getattr(caches[1], name)), (step, name)
        for r in recorders:
            r.residuals.clear()
        if counts is None:
            counts = [f.cache_info().currsize for f in COMPILES]
        token = logits[0].argmax().reshape(1)
        pos += 1
    assert [f.cache_info().currsize for f in COMPILES] == counts

    decoder, cache = decoders[0], caches[0]
    decoder.visit = recorders[0].runner
    initial = clone(cache)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        decoder(token, pos, cache)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            logits = decoder(token, pos, cache)
    torch.cuda.current_stream().wait_stream(stream)
    for name, tensor in vars(cache).items():
        tensor.copy_(getattr(initial, name))
    eager = decoder(token, pos, cache).clone()
    eager_cache = clone(cache)
    for name, tensor in vars(cache).items():
        tensor.copy_(getattr(initial, name))
    logits.zero_()
    graph.replay()
    assert torch.equal(logits, eager)
    for name, tensor in vars(cache).items():
        assert torch.equal(tensor, getattr(eager_cache, name)), name
