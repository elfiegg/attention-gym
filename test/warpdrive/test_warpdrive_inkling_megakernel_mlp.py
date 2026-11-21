"""The Inkling 2 Mini MLP-half megakernel launch (``inkling_megakernel.megakernel``) against
``ref.mlp_block``: routing, block residual, experts, post-MLP convolution and residual, a CUDA
graph replay and compile reuse across positions, on random weights at the real shapes and on
the checkpoint's own visits."""

import dataclasses
import gc

import pytest
import torch

pytest.importorskip("cutlass.cute")

from attn_gym._backends.arch import cuda_runs_sm100_kernels
from attn_gym.warpdrive.examples.inkling_megakernel import ref
from attn_gym.warpdrive.examples.inkling_megakernel.fast import FAST, decode_step
from attn_gym.warpdrive.examples.inkling_megakernel.megakernel import (
    InklingMlpMegakernel,
    decode_step_megakernel,
)
from attn_gym.warpdrive.examples.inkling_megakernel.mlp_kernel import compile_mlp_kernel

CONFIG = ref.InklingConfig()
MODEL = "thinkingmachines/Inkling-2-Mini"
# Outputs differ from the oracle only through fp32 summation order, which can flip a bf16
# rounding of a gate/up or down output and so move a few output elements by an ulp of the
# summands' scale: bounded in ulps of a bf16 value at the expected output's RMS.
MAX_ULPS, MAX_CHANGED = 8.0, 0.15


@pytest.fixture(autouse=True)
def release_cuda_memory():
    yield
    gc.collect()
    torch.cuda.empty_cache()


def require_sm100():
    if not cuda_runs_sm100_kernels():
        pytest.skip("the megakernel requires an SM100-family GPU")


def random_layer(config: ref.InklingConfig, seed: int) -> ref.LayerWeights:
    """MLP weights of one layer at ``config``'s shapes, scaled like a trained layer's (unit
    gate/up and down outputs); attention weights are empty placeholders."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    h, width, experts = config.hidden_size, config.moe_intermediate_size, config.num_experts

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, device="cuda", generator=gen) * scale

    empty = torch.empty(0, device="cuda")
    attention = dict.fromkeys(
        ("wq", "wk", "wv", "wo", "wgate", "wrel", "wsink", "q_scale", "rel_scale", "rel_proj",
         "attn_conv"),
        empty,
    )  # fmt: skip
    return ref.LayerWeights(
        **attention,
        mlp_conv=randn(h, config.conv_kernel, scale=0.3),
        post_norm=(1 + randn(h, scale=0.1)).bfloat16(),
        mlp_query=randn(h),
        router=randn(experts, h, scale=h**-0.5).bfloat16().float(),  # bf16-exact, as Inkling's
        router_bias=randn(experts, scale=0.05),
        w13=randn(experts, 2 * width, h, scale=h**-0.5).bfloat16(),
        w2=randn(experts, h, width, scale=width**-0.5).bfloat16(),
    )


def scale_ulps(out: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    """|out - expected| in ulps of a bf16 value at the expected tensor's RMS."""
    rms = expected.float().square().mean().sqrt()
    return (out.float() - expected.float()).abs() / 2.0 ** (torch.floor(torch.log2(rms)) - 7)


def check_visit(runner, config, w, visit, residual, blocks, cache, positions):
    """One launch against ``ref.mlp_block`` on a copy of ``cache``: the same experts, the
    output and the convolution ring's new row within the budget, every other ring row
    untouched. Returns the output's error in ulps."""
    theirs = ref.InklingCache(**{k: v.clone() for k, v in vars(cache).items()})
    expected = ref.mlp_block(
        config, w, visit, residual, blocks, None, theirs, positions, ref.REFERENCE
    )
    ring = cache.conv_mlp[visit]
    out = runner(w, residual, blocks, ring, positions)
    x = ref.block_residual(w.mlp_query[None], blocks, residual)[:, 0]
    x = ref.rms_norm(x, w.post_norm, config.rms_eps).to(torch.bfloat16)
    experts = ref.route(config, w, x)[0][0].sort().values
    assert torch.equal(runner.routed_experts(w), experts)
    errors = scale_ulps(out, expected[0])
    assert errors.max() <= MAX_ULPS and (errors > 0).float().mean() <= MAX_CHANGED, errors.max()
    slot = int(positions) % ref.CONV_RING
    others = torch.arange(ref.CONV_RING, device="cuda") != slot
    assert torch.equal(ring[others], theirs.conv_mlp[visit][others])
    ring_errors = scale_ulps(ring[slot], theirs.conv_mlp[visit][slot])
    assert ring_errors.max() <= MAX_ULPS
    return errors


def random_state(config, blocks: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    h = config.hidden_size
    states = [torch.randn(1, h, device="cuda", generator=gen).bfloat16() for _ in range(blocks)]
    residual = torch.randn(1, h, device="cuda", generator=gen) * 2
    return residual, states


@torch.no_grad()
def test_mlp_block_random_weights():
    """1 to 4 completed blocks, positions before and past the convolution's history and
    past the ring's wrap, several launches through one runner (the epoch advances)."""
    require_sm100()
    w = random_layer(CONFIG, seed=0)
    runner = InklingMlpMegakernel(CONFIG, (w,))
    cache = ref.InklingCache.empty(CONFIG, max_len=64)
    cache.conv_mlp.normal_()  # nonzero history: positions before 0 must read zeros
    for case, (blocks, pos) in enumerate([(1, 0), (2, 1), (3, 2), (4, 5), (3, 37), (1, 63)]):
        residual, states = random_state(CONFIG, blocks, seed=case)
        positions = torch.tensor([pos], device="cuda")
        check_visit(runner, CONFIG, w, 3, residual, states, cache, positions)


@torch.no_grad()
def test_graph_replay_and_compile_reuse():
    """The launch replays bit-identically inside a CUDA graph, and positions and block counts
    are runtime values: varying them compiles nothing."""
    require_sm100()
    w = random_layer(CONFIG, seed=1)
    runner = InklingMlpMegakernel(CONFIG, (w,))
    cache = ref.InklingCache.empty(CONFIG, max_len=64)
    cache.conv_mlp.normal_()
    ring = cache.conv_mlp[0]
    initial = ring.clone()
    residual, states = random_state(CONFIG, 3, seed=2)
    positions = torch.tensor([9], device="cuda")
    out = torch.empty(CONFIG.hidden_size, device="cuda", dtype=torch.bfloat16)
    eager = runner(w, residual, states, ring, positions).clone()
    eager_ring = ring.clone()
    entries = compile_mlp_kernel.cache_info().currsize

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        runner(w, residual, states, ring, positions, out)  # warm up on the capture stream
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            runner(w, residual, states, ring, positions, out)
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(3):
        ring.copy_(initial)
        out.zero_()
        graph.replay()
        assert torch.equal(out, eager)
        assert torch.equal(ring, eager_ring)

    before = compile_mlp_kernel.cache_info()
    for pos, blocks in [(0, 1), (1, 4), (17, 2), (1000, 3)]:
        positions.fill_(pos)
        graph.replay()
        # New inputs for the eager launch; the graph keeps reading the captured ones.
        other_residual, other_states = random_state(CONFIG, blocks, seed=pos)
        runner(w, other_residual, other_states, ring, positions)
    info = compile_mlp_kernel.cache_info()
    assert info.currsize == entries
    assert info.misses == before.misses
    assert info.hits + info.misses == before.hits + before.misses + 4  # one per eager launch


def checkpoint_weights():
    hub = pytest.importorskip("huggingface_hub")
    if not isinstance(hub.try_to_load_from_cache(MODEL, "config.json"), str):
        pytest.skip(f"{MODEL} is not in the local HF cache")
    return ref.load_inkling_weights(MODEL)


@pytest.mark.xdist_group("inkling-checkpoint")
@torch.no_grad()
def test_checkpoint_visits_and_hybrid_step():
    """Every plain visit of a decode step after a prefill on Inkling 2 Mini: the launch routes
    to the oracle's experts and stays within the budget of ``ref.mlp_block`` on the visit's own
    inputs. Then the hybrid step (``decode_step_megakernel``) against ``fast.decode_step``:
    close logits, the same next token. Skips unless the checkpoint is cached; never
    downloads."""
    require_sm100()
    weights = checkpoint_weights()
    config = weights.config
    runner = InklingMlpMegakernel(config, weights.layers)
    cache = ref.InklingCache.empty(config, max_len=128)
    prompt = torch.arange(1000, 1040, device="cuda")
    token = ref.forward_tokens(weights, prompt, 0, cache).argmax().reshape(1)
    pos = torch.tensor([prompt.numel()], device="cuda")
    start = ref.InklingCache(**{k: v.clone() for k, v in vars(cache).items()})

    visits = []

    def spy(config, w, visit, residual, blocks, message, cache, positions, kernels):
        if message is None:
            mine = ref.InklingCache(**{k: v.clone() for k, v in vars(cache).items()})
            check_visit(runner, config, w, visit, residual, blocks, mine, positions)
            visits.append(visit)
        return ref.mlp_block(
            config, w, visit, residual, blocks, message, cache, positions, kernels
        )

    spying = dataclasses.replace(FAST, mlp_block=spy)
    expected = decode_step(weights, token, pos, cache, spying)
    plain = [v for v, layer, p in config.visits() if not (config.is_loop(layer) and p > 0)]
    assert visits == plain

    logits = decode_step_megakernel(weights, token, pos, start, runner)
    assert logits.argmax() == expected.argmax()
    torch.testing.assert_close(logits, expected, atol=0.25, rtol=0)
