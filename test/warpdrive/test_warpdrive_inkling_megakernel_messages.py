"""The Inkling 2 Mini message launches (``inkling_megakernel.megakernel``) on random weights at
the real shapes: a pass's messages and message halves (``InklingMsgMegakernel``) against
``ref.block_residual`` and the gates' down projections, and the attention and MLP halves of
core visits after the first pass, whose inputs pass through the message gates, against
``ref.attention_block`` and ``ref.mlp_block`` with the oracle's messages; a CUDA graph replay
of a whole message visit and compile reuse across positions."""

import dataclasses
import gc

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cutlass.cute")

from test_warpdrive_inkling_megakernel_attn import (
    CONFIG,
    clone,
    random_cache,
    random_inputs,
    require_sm100,
)
from test_warpdrive_inkling_megakernel_attn import check_visit as check_attention
from test_warpdrive_inkling_megakernel_attn import random_layer as random_attention_layer
from test_warpdrive_inkling_megakernel_mlp import check_visit as check_mlp
from test_warpdrive_inkling_megakernel_mlp import random_layer as random_mlp_layer

from attn_gym.warpdrive.examples.inkling_megakernel import ref
from attn_gym.warpdrive.examples.inkling_megakernel.attn_kernel import compile_attn_kernel
from attn_gym.warpdrive.examples.inkling_megakernel.common import GATE_SPLITS
from attn_gym.warpdrive.examples.inkling_megakernel.megakernel import (
    InklingAttnMegakernel,
    InklingMlpMegakernel,
    InklingMsgMegakernel,
    MessageGates,
)
from attn_gym.warpdrive.examples.inkling_megakernel.mlp_kernel import compile_mlp_kernel
from attn_gym.warpdrive.examples.inkling_megakernel.msg_kernel import compile_msg_kernel


@pytest.fixture(autouse=True)
def release_cuda_memory():
    yield
    gc.collect()
    torch.cuda.empty_cache()


def random_core_layer(config: ref.InklingConfig, layer: int, seed: int) -> ref.LayerWeights:
    """A core layer at ``config``'s shapes: the attention and MLP tests' random weights and four
    message gates whose down projections give unit-scale ``h`` and whose forget and gain
    projections spread ``tanh`` over its range."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    h, rank = config.hidden_size, config.gate_rank

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, device="cuda", generator=gen) * scale

    gates = tuple(
        ref.MessageGate(
            down=randn(rank, 2 * h, scale=(2 * h) ** -0.5).bfloat16(),
            forget=randn(h, rank, scale=rank**-0.5),
            gain=randn(h, rank, scale=rank**-0.5),
        )
        for _ in range(4)
    )
    mlp = random_mlp_layer(config, seed)
    fields = {name: getattr(mlp, name) for name in
              ("mlp_conv", "post_norm", "mlp_query", "router", "router_bias", "w13", "w2")}  # fmt: skip
    return dataclasses.replace(random_attention_layer(config, layer, seed), **fields, gates=gates)


def random_pass(config: ref.InklingConfig, layers: int, seed: int):
    """``(queries, in12, out17)``: message queries of ``layers`` core layers and the previous
    pass's two states."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    h = config.hidden_size
    queries = torch.randn(layers, 4, h, device="cuda", generator=gen)
    states = [(torch.randn(1, h, device="cuda", generator=gen) * 3).bfloat16() for _ in range(2)]
    return queries, *states


def oracle_messages(queries, block, partial):
    """``ref.hidden_states``' messages: (layers, 1, 4, hidden) fp32."""
    return torch.stack([ref.block_residual(q, [block], partial) for q in queries])


def tagged(launch, name: str, count: int) -> torch.Tensor:
    """The fp32 values of a launch's tagged scratch buffer ``name``."""
    offset = launch.op.scratch_offsets[name]
    return launch.scratch[offset : offset + count].view(torch.int32)[0::2].view(torch.float32)


def check_gates(config, launch, w, first: int, streams, messages, gates):
    """The last launch's gates ``first ..`` on their fp32 input ``streams``: ``h`` within one
    bf16 ulp of ``ref.message_gate``'s bf16 GEMM, and the gated streams equal to the oracle's
    gate evaluated on the kernel's own ``h`` up to fp32 rounding. (An ``h`` rounding flip moves
    every gated element by about 1e-3 relative: the launch outputs differ from the oracle's
    through those flips, not through the gate arithmetic.) Returns the count of flipped ``h``."""
    count, h, rank = launch.op.gates, config.hidden_size, config.gate_rank
    h_part = tagged(launch, "h_part", GATE_SPLITS * count * rank).view(GATE_SPLITS, count, rank)
    gated = tagged(launch, "gated", count * h).view(count, h)
    mhalf = gates.mhalf.view(-1, rank)
    flips = 0
    for s in range(count):
        gate, x, m = w.gates[first + s], streams[s], messages[0, first + s]
        inputs = torch.cat([ref.rms_norm(v, None, config.rms_eps) for v in (x, m)])
        expected = F.linear(inputs.bfloat16(), gate.down).float()
        mine = h_part[0, s]
        for k in range(1, GATE_SPLITS):  # the kernel's summation order
            mine = mine + h_part[k, s]
        mine = (mine + mhalf[first + s]).bfloat16().float()
        ulp = 2.0 ** (torch.floor(torch.log2(expected.abs())) - 7)
        assert ((mine - expected).abs() <= ulp).all(), s
        flips += int((mine != expected).sum())
        out = (1 - torch.tanh(gate.forget @ mine)) * x + torch.tanh(gate.gain @ mine) * m
        torch.testing.assert_close(gated[s], out, rtol=1e-5, atol=1e-5)
    return flips


@torch.no_grad()
def test_messages_and_message_halves():
    """Every core layer's four messages match ``ref.block_residual`` to fp32 rounding, and the
    message halves match the gates' down projections on the oracle's bf16 ``rms(m)``; a second
    pass through the same runner rewrites both."""
    require_sm100()
    config, h = CONFIG, CONFIG.hidden_size
    core = tuple(random_core_layer(config, 7, seed) for seed in range(2))
    queries, _, _ = random_pass(config, len(core), seed=0)
    gates = MessageGates(core, queries)
    runner = InklingMsgMegakernel(config, gates)
    for seed in (1, 2):
        _, block, partial = random_pass(config, len(core), seed)
        runner(block, partial)
        expected = oracle_messages(queries, block, partial)[:, 0]  # (layers, 4, hidden)
        torch.testing.assert_close(
            gates.messages.view_as(expected), expected, atol=1e-5, rtol=1e-5
        )
        inputs = ref.rms_norm(expected, None, config.rms_eps).bfloat16().float()
        down = torch.stack([torch.stack([g.down for g in w.gates]) for w in core])
        halves = torch.einsum("lgk,lgrk->lgr", inputs, down[..., h:].float())
        # A flipped bf16 rounding of one rms(m) element moves a half by ~6e-5; a few such flips
        # and the fp32 order bound the difference.
        torch.testing.assert_close(gates.mhalf.view_as(halves), halves, atol=2e-3, rtol=1e-3)


# A flipped bf16 rounding of a gate's h moves its whole gated stream by ~1e-3 relative, which
# flips the bf16 rounding of up to about half the normalized MLP input (one ulp each): far more
# MLP outputs change by an ulp than in a plain visit (``check_gates`` checks the gate itself).
MLP_MAX_CHANGED = 0.6
# (layer, blocks, positions): sliding core layers inside a block and at a block boundary, full
# core layers; positions before the convolution history fills and past the sliding window.
VISITS = [(7, 2, (1, 1100)), (12, 2, (64, 300)), (11, 2, (3, 1100)), (17, 3, (700,))]


@pytest.mark.parametrize("layer,blocks,positions", VISITS)
@torch.no_grad()
def test_message_visit_random_weights(layer, blocks, positions):
    """Both halves of a core visit after the first pass: the message launch, then the attention
    launch (q, k, v gates) against ``ref.attention_block`` and the MLP launch (MLP gate) against
    ``ref.mlp_block``, each on the oracle's inputs and messages, within the plain launches'
    budgets."""
    require_sm100()
    config = CONFIG
    w = random_core_layer(config, layer, seed=layer)
    queries, block, partial = random_pass(config, 1, seed=layer)
    gates = MessageGates((w,), queries)
    InklingMsgMegakernel(config, gates)(block, partial)
    messages = oracle_messages(queries, block, partial)[0]  # (1, 4, hidden)
    attention = InklingAttnMegakernel(config, (w,))
    mlp = InklingMlpMegakernel(config, (w,))
    cache = random_cache(config, visits=2, max_len=1152, seed=layer)
    full = config.is_full(layer)
    for case, pos in enumerate(positions):
        positions_ = torch.tensor([pos], device="cuda")
        h, states = random_inputs(config, blocks, seed=100 * layer + case)
        args = gates.layer(0)
        errors = check_attention(
            attention, config, w, layer, 1, h, states, cache, positions_, messages, args
        )
        streams = ref.block_residual(w.attn_query, states, h)[0]
        streams = streams if full else streams.expand(3, -1)
        flips = check_gates(config, attention.launches[full, True], w, 0, streams, messages, args)
        print(f"layer {layer} pos {pos} attention: max {errors.max():.2f} "
              f"median {errors.median():.2f} ulps, {flips} h flips")  # fmt: skip
        residual, mlp_blocks = ref.attention_block(
            config, w, layer, 1, h, states, messages, clone(cache), positions_, None, ref.REFERENCE
        )
        errors = check_mlp(
            mlp, config, w, 1, residual, mlp_blocks, cache, positions_, messages[:, 3], args,
            max_changed=MLP_MAX_CHANGED,
        )  # fmt: skip
        stream = ref.block_residual(w.mlp_query[None], mlp_blocks, residual)[0]
        flips = check_gates(config, mlp.launches[True], w, 3, stream, messages, args)
        print(f"layer {layer} pos {pos} MLP: max {errors.max():.2f} ulps, "
              f"{(errors > 0).float().mean():.3f} changed, {flips} h flips")  # fmt: skip


@torch.no_grad()
def test_message_visit_graph_replay_and_compile_reuse():
    """A message visit (messages, attention, MLP) replays bit-identically to eager launches in
    a CUDA graph at device positions that change the KV splits, and positions and block counts
    compile nothing new."""
    require_sm100()
    config, layer = CONFIG, 11
    w = random_core_layer(config, layer, seed=3)
    queries, block, partial = random_pass(config, 1, seed=3)
    gates = MessageGates((w,), queries)
    msg = InklingMsgMegakernel(config, gates)
    attention = InklingAttnMegakernel(config, (w,))
    mlp = InklingMlpMegakernel(config, (w,))
    cache = random_cache(config, visits=1, max_len=1152, seed=3)
    initial = clone(cache)
    h, states = random_inputs(config, 2, seed=4)
    positions = torch.tensor([1030], device="cuda", dtype=torch.int32)
    residual = torch.empty(config.hidden_size, device="cuda")
    out = torch.empty(config.hidden_size, device="cuda", dtype=torch.bfloat16)

    def visit(cache=cache):
        msg(block, partial)
        attention(w, layer, 0, h, states, cache, positions, residual, gates.layer(0))
        mlp(w, residual, states, cache.conv_mlp[0], positions, out, gates.layer(0))
        return out

    visit()
    counts = [f.cache_info().currsize for f in (compile_msg_kernel, compile_attn_kernel,
                                                compile_mlp_kernel)]  # fmt: skip
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        visit()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            visit()
    torch.cuda.current_stream().wait_stream(stream)
    for pos in (1030, 70, 1100):
        positions.fill_(pos)
        expected_cache = clone(initial)
        expected = visit(expected_cache).clone()
        for name, tensor in vars(cache).items():
            tensor.copy_(getattr(initial, name))
        out.zero_()
        graph.replay()
        assert torch.equal(out, expected), pos
        for name, tensor in vars(cache).items():
            assert torch.equal(tensor, getattr(expected_cache, name)), (pos, name)
    for blocks in (1, 3):  # block counts are runtime values too
        h, others = random_inputs(config, blocks, seed=blocks)
        attention(w, layer, 0, h, others, cache, positions, residual, gates.layer(0))
        mlp(w, residual, others, cache.conv_mlp[0], positions, out, gates.layer(0))
    assert counts == [f.cache_info().currsize for f in (compile_msg_kernel, compile_attn_kernel,
                                                        compile_mlp_kernel)]  # fmt: skip
