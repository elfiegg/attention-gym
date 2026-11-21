"""The Inkling 2 Mini decode step as megakernel launches only (``megakernel.InklingDecoder``):
the head launch (final block residual, norm, LM head, softcap) against the oracle on random
weights at the real shapes, and on the checkpoint a greedy decode against ``fast.decode_step``,
compile reuse across positions and a CUDA graph replay bit-identical to eager launches."""

import gc
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass.cute")

from test_warpdrive_inkling_megakernel_attn import CONFIG, MODEL, clone, require_sm100

from attn_gym.warpdrive.examples.inkling_megakernel import ref
from attn_gym.warpdrive.examples.inkling_megakernel.attn_kernel import compile_attn_kernel
from attn_gym.warpdrive.examples.inkling_megakernel.head_kernel import compile_head_kernel
from attn_gym.warpdrive.examples.inkling_megakernel.megakernel import (
    InklingDecoder,
    InklingHeadMegakernel,
)
from attn_gym.warpdrive.examples.inkling_megakernel.mlp_kernel import compile_mlp_kernel
from attn_gym.warpdrive.examples.inkling_megakernel.msg_kernel import compile_msg_kernel

COMPILES = (compile_attn_kernel, compile_mlp_kernel, compile_msg_kernel, compile_head_kernel)
PROMPT = "What is the capital of France? One sentence."
STEPS = 40
# Logits differ from the oracle's where a bf16 rounding of an LM-head output or of the
# normalized final state flips: one bf16 ulp of a logit (2 ** -3 at |logit| in [16, 32)), at
# most, compressed by the softcap.
HEAD_ATOL = 0.13
# A decode step's logits stay within about a unit of the oracle's: the caches carry a step's
# rounding differences into later steps, and where a router's 8th and 9th keys nearly tie
# (margins of 1e-5 occur within tens of steps) rounding picks another expert at one visit,
# which moves logits by up to a few units without changing the greedy token.
STEP_ATOL = 4.0
# Best-two logit margins under which the oracle's token is a tie: one bf16 ulp of a logit in
# [16, 32) is 0.125 (the oracle itself ties exactly at times, broken by the lower index).
TIE = 0.25


@pytest.fixture(autouse=True)
def release_cuda_memory():
    yield
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def test_head_random_weights():
    """The head launch against ``ref.hidden_states``' last block residual and norm and
    ``ref.logits_of`` on a random LM head at the real vocabulary (its last tile partial), with
    logits spread over the softcap's range; 1 to 4 blocks."""
    require_sm100()
    config, h = CONFIG, CONFIG.hidden_size
    gen = torch.Generator(device="cuda").manual_seed(0)

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, device="cuda", generator=gen) * scale

    weights = SimpleNamespace(
        config=config,
        lm_head=randn(config.vocab_size, h, scale=20 * h**-0.5).bfloat16(),
        final_query=randn(h),
        final_norm=(1 + randn(h, scale=0.1)).bfloat16(),
    )
    runner = InklingHeadMegakernel(weights)
    for count in (4, 1, 3):
        states = [(randn(1, h) * 2).bfloat16() for _ in range(count + 1)]
        blocks, partial = states[:-1], states[-1]
        logits = runner(blocks, partial)
        x = ref.block_residual(weights.final_query[None], blocks, partial)[:, 0]
        x = ref.rms_norm(x, weights.final_norm, config.rms_eps).to(torch.bfloat16)
        expected = ref.logits_of(weights, x)[0]
        errors = (logits - expected).abs()
        print(f"{count} blocks: max {errors.max():.4f}, {(errors > 0).float().mean():.4f} differ")
        assert errors.max() <= HEAD_ATOL
        assert logits.argmax() == expected.argmax()


def checkpoint_path():
    """A complete local snapshot of the checkpoint with its tokenizer; skips otherwise (never
    downloads)."""
    hub = pytest.importorskip("huggingface_hub")
    from huggingface_hub.errors import LocalEntryNotFoundError

    patterns = ["*.json", "model*.safetensors", "*.jinja", "tiktoken/*"]
    try:
        return hub.snapshot_download(MODEL, allow_patterns=patterns, local_files_only=True)
    except LocalEntryNotFoundError as error:
        pytest.skip(f"{MODEL} is not in the local HF cache: {error}")


def prompt_ids(path) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="low",
    )
    return tokenizer(text, add_special_tokens=False).input_ids


@pytest.mark.xdist_group("inkling-checkpoint")
@torch.no_grad()
def test_checkpoint_greedy_decode():
    """``STEPS`` greedy steps after a chat prompt's prefill on Inkling 2 Mini, both decoders fed
    the oracle's token (``ref.forward_tokens``, HF's numerics) into their own caches: the
    decoder picks the oracle's token every step, unless the oracle's best two logits nearly tie
    (``TIE``) and it picks the other, with close logits (``STEP_ATOL``); positions compile
    nothing after the first step; one step captured in a CUDA graph replays bit-identically to
    the same step's eager launches, logits and caches. Skips unless the checkpoint is cached;
    never downloads."""
    require_sm100()
    path = checkpoint_path()
    weights = ref.load_inkling_weights(path)
    ids = torch.tensor(prompt_ids(path), device="cuda")
    config = weights.config
    theirs = ref.InklingCache.empty(config, max_len=256)
    token = ref.forward_tokens(weights, ids, 0, theirs).argmax().reshape(1)
    mine = clone(theirs)
    decoder = InklingDecoder(weights)
    pos = torch.tensor([ids.numel()], device="cuda", dtype=torch.int32)
    counts = None
    for step in range(STEPS):
        expected = ref.forward_tokens(weights, token, ids.numel() + step, theirs)
        logits = decoder(token, pos, mine)
        error = (logits - expected).abs().max().item()
        (best, second), (first, other) = expected.topk(2)
        print(f"step {step}: token {int(first)}, logit max error {error:.4f}")
        assert logits.argmax() == first or (logits.argmax() == other and best - second < TIE)
        assert error <= STEP_ATOL, step
        if counts is None:
            counts = [f.cache_info().currsize for f in COMPILES]
        token = first.reshape(1)
        pos += 1
    assert [f.cache_info().currsize for f in COMPILES] == counts

    initial = clone(mine)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        decoder(token, pos, mine)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            logits = decoder(token, pos, mine)
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(2):
        for name, tensor in vars(mine).items():
            tensor.copy_(getattr(initial, name))
        eager = decoder(token, pos, mine).clone()
        eager_cache = clone(mine)
        for name, tensor in vars(mine).items():
            tensor.copy_(getattr(initial, name))
        logits.zero_()
        graph.replay()
        assert torch.equal(logits, eager)
        for name, tensor in vars(mine).items():
            assert torch.equal(tensor, getattr(eager_cache, name)), name
