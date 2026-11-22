"""The Inkling 2 Mini visit launch (``inkling_megakernel.megakernel.InklingVisitMegakernel``):
its attention half against ``ref.attention_block`` with ``start=None`` (block residual inputs,
projections, k/v convolutions, clips, relative bias, window, sink, gate, o_proj,
post-attention convolution and residual, for sliding and full layers; the cache and ring rows
it writes), its MLP half against ``ref.mlp_block`` on the launch's own post-attention residual
(routing, block residual, experts, post-MLP convolution and residual), a CUDA graph replay and
compile reuse across positions, on random weights at the real shapes and on the checkpoint's
own visits."""

import dataclasses
import gc

import pytest
import torch

pytest.importorskip("cutlass.cute")

from attn_gym._backends.arch import cuda_runs_sm100_kernels
from attn_gym.warpdrive.examples.inkling_megakernel import ref
from attn_gym.warpdrive.examples.inkling_megakernel.fast import FAST, decode_step
from attn_gym.warpdrive.examples.inkling_megakernel.megakernel import InklingVisitMegakernel
from attn_gym.warpdrive.examples.inkling_megakernel.visit_kernel import compile_visit_kernel

CONFIG = ref.InklingConfig()
MODEL = "thinkingmachines/Inkling-2-Mini"
# The residual differs from the oracle mainly through the flash merge: the oracle rounds the
# softmax probabilities to bf16 before the value sum, the kernel sums unrounded ones. The
# attention output then differs by an ulp in many elements, which o_proj spreads into bf16
# rounding flips of about half the outputs on the checkpoint (max 7 ulps, median <= 0.25, in
# ``ulps`` of the post-attention output; against an oracle with fp32 probabilities: max 2).
MAX_ULPS, MAX_MEDIAN_ULPS = 8.0, 0.5
# The k/v ring rows hold raw projections, whose fp32 order can flip a bf16 rounding; the
# convolution, clip and norm can carry such a flip of a large element into two ulps of a
# smaller cache element.
RING_ULPS, CACHE_ULPS = 1.0, 2.0
# The MLP output differs from the oracle's on the same residual only through fp32 summation
# order, which can flip a bf16 rounding of a gate/up or down output and so move a few output
# elements by an ulp of the summands' scale: bounded in ulps of a bf16 value at the expected
# output's RMS, and in the fraction of changed elements. One flipped gate/up activation moves
# every down output of its expert a little: on the checkpoint's visits 0.0-0.11 of the outputs
# change on the oracle's residuals, 0.0-0.23 on the launch's own (max 4 ulps).
MLP_ULPS, MLP_MAX_CHANGED = 8.0, 0.3


@pytest.fixture(autouse=True)
def release_cuda_memory():
    yield
    gc.collect()
    torch.cuda.empty_cache()


def require_sm100():
    if not cuda_runs_sm100_kernels():
        pytest.skip("the megakernel requires an SM100-family GPU")


def random_attention(config: ref.InklingConfig, layer: int, gen) -> dict:
    """Attention weights of ``layer`` (full or sliding, with or without a block-residual query)
    at ``config``'s shapes, scaled so q clips sometimes, scores and biases spread over a few
    units and outputs have unit scale."""
    h, heads, kv, d = config.hidden_size, config.num_q_heads, config.kv_dim, config.head_dim
    kernel, rel = config.conv_kernel, config.d_rel

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, device="cuda", generator=gen) * scale

    def weight(rows, cols, scale=1.0):
        return randn(rows, cols, scale=scale * cols**-0.5).bfloat16()

    def norm():
        return (1 + randn(h, scale=0.1)).bfloat16()

    fields = {
        "wq": weight(heads * d, h),
        "wk": weight(kv, h, 0.5),
        "wv": weight(kv, h),
        "wo": weight(h, heads * d),
        "wgate": weight(heads * d, h, 2.0),
        "wrel": weight(heads * rel, h),
        "wsink": weight(heads, h, 2.0),
        "q_scale": (1 + 2 * torch.rand(heads, d, device="cuda", generator=gen)).bfloat16(),
        "rel_scale": (1 + randn(heads, rel, scale=0.2)).bfloat16(),
        "rel_proj": randn(rel, config.rel_extent, scale=0.5).bfloat16(),
        "attn_conv": randn(h, kernel, scale=0.3),
    }
    if layer > 0:
        queries = 3 if config.is_full(layer) else 1
        fields["attn_query"] = randn(queries, h)
    if config.is_full(layer):
        fields.update(
            q_norm=norm(),
            k_norm=norm(),
            v_norm=norm(),
            k_static=randn(kernel, kv, scale=0.3).bfloat16(),
            v_static=randn(kernel, kv, scale=0.3).bfloat16(),
            k_dynamic=weight(config.num_kv_heads * kernel, h, 0.3),
            v_dynamic=weight(config.num_kv_heads * kernel, h, 0.3),
        )
    else:
        fields.update(
            input_norm=norm(),
            k_conv=randn(kv, kernel, scale=0.3),
            v_conv=randn(kv, kernel, scale=0.3),
        )
    return fields


def random_mlp(config: ref.InklingConfig, gen) -> dict:
    """MLP weights at ``config``'s shapes, scaled like a trained layer's (unit gate/up and down
    outputs)."""
    h, width, experts = config.hidden_size, config.moe_intermediate_size, config.num_experts

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, device="cuda", generator=gen) * scale

    return {
        "mlp_conv": randn(h, config.conv_kernel, scale=0.3),
        "post_norm": (1 + randn(h, scale=0.1)).bfloat16(),
        "mlp_query": randn(h),
        "router": randn(experts, h, scale=h**-0.5).bfloat16().float(),  # bf16-exact, as Inkling's
        "router_bias": randn(experts, scale=0.05),
        "w13": randn(experts, 2 * width, h, scale=h**-0.5).bfloat16(),
        "w2": randn(experts, h, width, scale=width**-0.5).bfloat16(),
    }


def random_layer(config: ref.InklingConfig, layer: int, seed: int) -> ref.LayerWeights:
    """Both halves' random weights of ``layer`` (no message gates)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return ref.LayerWeights(**random_attention(config, layer, gen), **random_mlp(config, gen))


def random_cache(config: ref.InklingConfig, visits: int, max_len: int, seed: int):
    """A cache of ``visits`` visits with random keys, values and rings everywhere: rows at and
    past the position must be masked, ring rows before position 0 read as zeros."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    kv, d = config.num_kv_heads, config.head_dim

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda", generator=gen) * scale).bfloat16()

    ring = ref.CONV_RING
    return ref.InklingCache(
        k=randn(visits, kv, max_len, d, scale=0.7),
        v=randn(visits, kv, max_len, d),
        conv_k=randn(visits, ring, config.kv_dim, scale=0.5),
        conv_v=randn(visits, ring, config.kv_dim),
        conv_attn=randn(visits, ring, config.hidden_size),
        conv_mlp=randn(visits, ring, config.hidden_size),
    )


def random_inputs(config: ref.InklingConfig, blocks: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    h = config.hidden_size
    states = [
        torch.randn(1, h, device="cuda", generator=gen).bfloat16() for _ in range(blocks + 1)
    ]
    return states[-1], states[:-1]


def ulps(out: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    """|out - expected| in bf16 ulps of each expected element, at least the ulp of a value at
    the expected tensor's RMS (an element that cancelled to near zero keeps its inputs' scale)."""
    expected = expected.float()
    rms = expected.square().mean().sqrt()
    exponent = torch.floor(torch.log2(expected.abs().clamp(min=rms.item())))
    return (out.float() - expected).abs() / 2.0 ** (exponent - 7)


def scale_ulps(out: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    """|out - expected| in ulps of a bf16 value at the expected tensor's RMS."""
    rms = expected.float().square().mean().sqrt()
    return (out.float() - expected.float()).abs() / 2.0 ** (torch.floor(torch.log2(rms)) - 7)


def clone(cache: ref.InklingCache) -> ref.InklingCache:
    return ref.InklingCache(**{k: v.clone() for k, v in vars(cache).items()})


def check_visit(
    runner,
    config,
    w,
    layer,
    visit,
    h,
    blocks,
    cache,
    positions,
    messages=None,
    gates=None,
    mlp_max_changed=MLP_MAX_CHANGED,
):
    """One launch against the oracle on a copy of ``cache``. Attention: ``ref.attention_block``
    (``start=None``) on the launch's inputs: the residual and the attention ring row within
    ``MAX_ULPS``, the k/v cache and ring rows within theirs, every other row untouched. MLP:
    ``ref.mlp_block`` on the launch's own residual: the same experts, the output and the MLP
    ring's new row within ``MLP_ULPS`` (at most ``mlp_max_changed`` of the outputs changed),
    every other ring row untouched. A visit with ``messages`` (the oracle's ``(1, 4, hidden)``)
    launches with the matching ``gates`` (``megakernel.GateArgs``). Returns the residual's and
    the output's errors (``ulps`` of the post-attention output, ``scale_ulps``) and the
    launch's residual."""
    theirs = clone(cache)
    expected, mlp_blocks = ref.attention_block(
        config, w, layer, visit, h, blocks, messages, theirs, positions, None, FAST
    )
    out = runner(w, layer, visit, h, blocks, cache, positions, gates=gates)
    residual = runner.residual()[None]
    boundary = layer % config.block_layers == 0
    attn_out = expected[0] if boundary else expected[0] - h[0].float()
    errors = ulps(residual[0] - expected[0] + attn_out, attn_out)
    pos = int(positions)
    for name in ("k", "v"):
        mine, ref_rows = getattr(cache, name)[visit], getattr(theirs, name)[visit]
        assert ulps(mine[:, pos], ref_rows[:, pos]).max() <= CACHE_ULPS, name
        others = torch.arange(cache.max_len, device="cuda") != pos
        assert torch.equal(mine[:, others], ref_rows[:, others]), name
    slot = pos % ref.CONV_RING
    others = torch.arange(ref.CONV_RING, device="cuda") != slot
    for name in ("conv_k", "conv_v", "conv_attn"):
        mine, ref_ring = getattr(cache, name)[visit], getattr(theirs, name)[visit]
        assert torch.equal(mine[others], ref_ring[others]), name
        budget = MAX_ULPS if name == "conv_attn" else RING_ULPS
        assert ulps(mine[slot], ref_ring[slot]).max() <= budget, name
    assert errors.max() <= MAX_ULPS and errors.median() <= MAX_MEDIAN_ULPS, (
        errors.max(),
        errors.median(),
    )

    message = None if messages is None else messages[:, 3]
    expected = ref.mlp_block(
        config, w, visit, residual, mlp_blocks, message, theirs, positions, ref.REFERENCE
    )
    x = ref.block_residual(w.mlp_query[None], mlp_blocks, residual)[:, 0]
    if message is not None:
        x = ref.message_gate(w.gates[3], x, message, config.rms_eps)
    x = ref.rms_norm(x, w.post_norm, config.rms_eps).to(torch.bfloat16)
    experts = ref.route(config, w, x)[0][0].sort().values
    assert torch.equal(runner.routed_experts(w), experts)
    out_errors = scale_ulps(out, expected[0])
    changed = (out_errors > 0).float().mean()
    assert out_errors.max() <= MLP_ULPS and changed <= mlp_max_changed, out_errors.max()
    ring, ref_ring = cache.conv_mlp[visit], theirs.conv_mlp[visit]
    assert torch.equal(ring[others], ref_ring[others])
    assert scale_ulps(ring[slot], ref_ring[slot]).max() <= MLP_ULPS
    return errors, out_errors, residual


CASES = [
    # (layer, blocks, positions): sliding layer 0 (no query, a block boundary: the MLP reads h
    # as its only block), a sliding layer inside a block, a sliding boundary layer with a
    # query, full layers; the MLP half reads 1 to 4 blocks.
    (0, 0, (0, 1, 63, 64, 1100)),
    (1, 1, (1, 63, 64, 1100)),
    (6, 2, (65, 1100)),
    (5, 1, (0, 1, 63, 64, 1100)),
    (23, 4, (130, 1100)),
]


@pytest.mark.parametrize("layer,blocks,positions", CASES)
@torch.no_grad()
def test_visit_random_weights(layer, blocks, positions):
    """Position 0 (only the new token and the sink), positions before the convolution history
    fills, at and across a KV block edge, and past the sliding window (1100 keys: sliding
    layers must read only the last 1024); several launches through one runner (the epoch
    advances)."""
    require_sm100()
    w = random_layer(CONFIG, layer, seed=layer)
    runner = InklingVisitMegakernel(CONFIG, (w,))
    cache = random_cache(CONFIG, visits=2, max_len=1152, seed=layer)
    for case, pos in enumerate(positions):
        h, states = random_inputs(CONFIG, blocks, seed=100 * layer + case)
        positions_ = torch.tensor([pos], device="cuda")
        errors, out_errors, _ = check_visit(
            runner, CONFIG, w, layer, 1, h, states, cache, positions_
        )
        print(f"layer {layer} pos {pos}: attention max {errors.max():.2f} median "
              f"{errors.median():.2f} ulps, MLP max {out_errors.max():.2f}")  # fmt: skip


@torch.no_grad()
def test_full_layer_log_scaling():
    """A full layer past ``log_scaling_floor``, where queries and relative states scale up."""
    require_sm100()
    layer = 11
    w = random_layer(CONFIG, layer, seed=7)
    runner = InklingVisitMegakernel(CONFIG, (w,))
    cache = random_cache(CONFIG, visits=1, max_len=9024, seed=7)
    h, states = random_inputs(CONFIG, 2, seed=7)
    check_visit(runner, CONFIG, w, layer, 0, h, states, cache, torch.tensor([9000], device="cuda"))


@torch.no_grad()
def test_graph_replay_and_compile_reuse():
    """The launch replays bit-identically to eager launches inside a CUDA graph, at device
    positions that change the KV splits, and positions, block counts and cache capacities are
    runtime values: varying them compiles nothing."""
    require_sm100()
    w = random_layer(CONFIG, 1, seed=1)
    runner = InklingVisitMegakernel(CONFIG, (w,))
    cache = random_cache(CONFIG, visits=1, max_len=1152, seed=1)
    initial = clone(cache)
    h, states = random_inputs(CONFIG, 1, seed=2)
    positions = torch.tensor([1030], device="cuda")
    out = torch.empty(CONFIG.hidden_size, device="cuda", dtype=torch.bfloat16)
    runner(w, 1, 0, h, states, cache, positions, out)
    entries = compile_visit_kernel.cache_info().currsize

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runner(w, 1, 0, h, states, cache, positions, out)  # warm up on the capture stream
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            runner(w, 1, 0, h, states, cache, positions, out)
    torch.cuda.current_stream().wait_stream(stream)
    for pos in (1030, 70, 1030):
        positions.fill_(pos)
        expected_cache = clone(initial)
        expected = runner(w, 1, 0, h, states, expected_cache, positions).clone()
        expected_residual = runner.residual().clone()
        for name, tensor in vars(cache).items():
            tensor.copy_(getattr(initial, name))
        out.zero_()
        graph.replay()
        assert torch.equal(out, expected), pos
        assert torch.equal(runner.residual(), expected_residual), pos
        for name, tensor in vars(cache).items():
            assert torch.equal(tensor, getattr(expected_cache, name)), (pos, name)

    before = compile_visit_kernel.cache_info()
    other = random_cache(CONFIG, visits=1, max_len=2048, seed=3)
    for pos, blocks, target in [(0, 0, cache), (64, 4, cache), (2000, 2, other), (1100, 3, other)]:
        positions.fill_(pos)
        h, states = random_inputs(CONFIG, blocks, seed=pos)
        runner(w, 1, 0, h, states, target, positions)
    info = compile_visit_kernel.cache_info()
    assert info.currsize == entries
    assert info.misses == before.misses
    assert info.hits + info.misses == before.hits + before.misses + 4  # one per eager launch


def checkpoint_weights():
    """The checkpoint's text decoder from a complete local snapshot; skips otherwise (never
    downloads)."""
    hub = pytest.importorskip("huggingface_hub")
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        path = hub.snapshot_download(
            MODEL, allow_patterns=["*.json", "model*.safetensors"], local_files_only=True
        )
    except LocalEntryNotFoundError as error:
        pytest.skip(f"{MODEL} is not in the local HF cache: {error}")
    return ref.load_inkling_weights(path)


@pytest.mark.xdist_group("inkling-checkpoint")
@torch.no_grad()
def test_checkpoint_visits():
    """Every plain visit of a decode step after a prefill on Inkling 2 Mini: the launch on the
    visit's own inputs stays within ``check_visit``'s budgets of ``ref.attention_block`` and of
    ``ref.mlp_block`` on its residual, and routes to the oracle's experts. Skips unless the
    checkpoint is cached; never downloads."""
    require_sm100()
    weights = checkpoint_weights()
    config = weights.config
    runner = InklingVisitMegakernel(config, weights.layers)
    cache = ref.InklingCache.empty(config, max_len=128)
    prompt = torch.arange(1000, 1040, device="cuda")
    token = ref.forward_tokens(weights, prompt, 0, cache).argmax().reshape(1)
    pos = torch.tensor([prompt.numel()], device="cuda")

    visits = []

    def spy(config, w, layer, visit, h, blocks, messages, cache, positions, start, kernels):
        if messages is None:
            errors, out_errors, _ = check_visit(
                runner, config, w, layer, visit, h, blocks, clone(cache), positions
            )
            print(f"visit {visit}: attention max {errors.max():.2f} median "
                  f"{errors.median():.2f} ulps, MLP max {out_errors.max():.2f}")  # fmt: skip
            visits.append(visit)
        return ref.attention_block(
            config, w, layer, visit, h, blocks, messages, cache, positions, start, kernels
        )

    decode_step(weights, token, pos, cache, dataclasses.replace(FAST, attention_block=spy))
    plain = [v for v, layer, p in config.visits() if not (config.is_loop(layer) and p > 0)]
    assert visits == plain
