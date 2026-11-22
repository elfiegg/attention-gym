"""The Inkling 2 Mini decode step as megakernel launches only (``megakernel.InklingDecoder``):
the head launch (final block residual, norm, LM head, softcap) against the oracle on random
weights at the real shapes, the whole step against the oracle on a random model of the real
widths and a reduced depth, and on the checkpoint a greedy decode against the oracle
(``ref.forward_tokens``), compile reuse across positions and a CUDA graph replay
bit-identical to eager launches."""

import dataclasses
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cutlass.cute")

from inkling_checkpoint import TOKENIZER, WEIGHTS, checkpoint_path
from test_warpdrive_inkling_megakernel_messages import random_core_layer
from test_warpdrive_inkling_megakernel_visit import (
    CONFIG,
    assert_caches_equal,
    capture,
    clone,
    restore,
    scale_ulps,
)

from attn_gym.warpdrive.examples.inkling_megakernel import ref
from attn_gym.warpdrive.examples.inkling_megakernel.head_kernel import compile_head_kernel
from attn_gym.warpdrive.examples.inkling_megakernel.megakernel import (
    InklingDecoder,
    InklingHeadMegakernel,
)
from attn_gym.warpdrive.examples.inkling_megakernel.msg_kernel import compile_msg_kernel
from attn_gym.warpdrive.examples.inkling_megakernel.visit_kernel import compile_visit_kernel

pytestmark = pytest.mark.usefixtures("requires_blackwell", "release_cuda_memory")

COMPILES = (compile_visit_kernel, compile_msg_kernel, compile_head_kernel)
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
# A visit's output against the oracle's on the same inputs, in ulps of a bf16 value at its RMS:
# plain visits stay within the visit tests' 8; a message gate's h rounding flip moves the whole
# MLP input of its visit, measured up to 12. Routers whose 8th and 9th keys are closer than
# NEAR_TIE may pick another expert for inputs that differ by such rounding: those visits are
# exempt (a few percent of visits on the random model).
VISIT_ULPS, NEAR_TIE = 16.0, 1e-3
# Best-two logit margins under which the oracle's token is a tie: one bf16 ulp of a logit in
# [16, 32) is 0.125 (the oracle itself ties exactly at times, broken by the lower index).
TIE = 0.25


@torch.no_grad()
def test_head_random_weights():
    """The head launch against ``ref.hidden_states``' last block residual and norm and
    ``ref.logits_of`` on a random LM head at the real vocabulary (its last tile partial), with
    logits spread over the softcap's range; 1 to 4 blocks."""
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


# The real widths at a quarter of the experts and a third of the depth, keeping the structure:
# blocks of two layers, a full layer ending each, a core of two blocks run three times.
SMALL = dataclasses.replace(
    CONFIG, num_layers=8, full_layers=(1, 3, 5, 7), block_layers=2, loop_start=2, loop_end=6,
    num_experts=32, vocab_size=1000,
)  # fmt: skip


def random_model(config: ref.InklingConfig, seed: int) -> ref.InklingWeights:
    """Random weights of ``config`` from the visit tests' layers (gates only in core layers)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    h = config.hidden_size

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, device="cuda", generator=gen) * scale

    layers = tuple(
        dataclasses.replace(
            random_core_layer(config, layer, seed + layer),
            **({} if config.is_loop(layer) else {"gates": ()}),
        )
        for layer in range(config.num_layers)
    )
    return ref.InklingWeights(
        embed=randn(config.vocab_size + 24, h).bfloat16(),
        embed_norm=(1 + randn(h, scale=0.1)).bfloat16(),
        final_norm=(1 + randn(h, scale=0.1)).bfloat16(),
        final_query=randn(h),
        lm_head=randn(config.vocab_size, h, scale=4 * h**-0.5).bfloat16(),
        message_queries=randn(config.loop_layers, 4, h),
        layers=layers,
        config=config,
    )


def routing_margin(config, w, residual, blocks, message) -> float:
    """The gap between the 8th and 9th router keys of ``ref.mlp_block``'s expert input."""
    x = ref.block_residual(w.mlp_query[None], blocks, residual)[:, 0]
    if message is not None:
        x = ref.message_gate(w.gates[3], x, message, config.rms_eps)
    x = ref.rms_norm(x, w.post_norm, config.rms_eps).to(torch.bfloat16)
    keys = torch.sigmoid(F.linear(x.float(), w.router))[0] + w.router_bias
    top = keys.topk(config.experts_per_token + 1).values
    return (top[-2] - top[-1]).item()


@torch.no_grad()
def test_decoder_random_weights():
    """Two decode steps after a prefill on a random model of ``SMALL``'s shape, each checked
    visit by visit: the oracle (``ref.hidden_states``, ``start=None``) runs the step on the
    decoder's own visit outputs, so each visit's output and the logits compare against the
    oracle on the same inputs (a random model amplifies one ulp of a visit output into
    different logits: whole-step outputs would not compare). This checks the decoder's
    sequencing: the embedding in layer 0's launch, the block lists, the passes' restarts and
    message states, and the head's blocks."""
    weights = random_model(SMALL, seed=0)
    gen = torch.Generator(device="cuda").manual_seed(1)
    prompt = torch.randint(0, SMALL.vocab_size, (20,), device="cuda", generator=gen)
    cache = ref.InklingCache.empty(SMALL, max_len=128)
    token = ref.forward_tokens(weights, prompt, 0, cache).argmax().reshape(1)
    decoder = InklingDecoder(weights)
    pos = torch.tensor([prompt.numel()], device="cuda", dtype=torch.int32)
    for step in range(2):
        theirs = clone(cache)
        logits = decoder(token, pos, cache)
        errors, ties = [], []

        def mlp_block(
            config, w, visit, residual, blocks, message, cache, positions, kernels,
            errors=errors, ties=ties,
        ):  # fmt: skip
            expected = ref.mlp_block(
                config, w, visit, residual, blocks, message, cache, positions, kernels
            )
            mine = decoder.states[visit + 1][None]
            if routing_margin(config, w, residual, blocks, message) < NEAR_TIE:
                ties.append(visit)
            else:
                errors.append(scale_ulps(mine, expected).max().item())
            return mine  # the oracle continues on the decoder's output

        oracle = dataclasses.replace(ref.REFERENCE, mlp_block=mlp_block)
        h = ref.hidden_states(weights, token, pos.long(), theirs, None, oracle)
        embedding = ref.rms_norm(weights.embed[token], weights.embed_norm, SMALL.rms_eps)
        assert scale_ulps(decoder.states[0], embedding).max() <= 1
        expected = ref.logits_of(weights, h)[0]
        print(f"step {step}: visit max ulps {[round(e, 1) for e in errors]}, near ties {ties}, "
              f"logit max error {(logits - expected).abs().max():.4f}")  # fmt: skip
        assert len(errors) + len(ties) == SMALL.num_visits and len(ties) <= 2
        assert max(errors) <= VISIT_ULPS
        assert (logits - expected).abs().max() <= HEAD_ATOL
        token = logits.argmax().reshape(1)
        pos += 1


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
    path = checkpoint_path(WEIGHTS + TOKENIZER)
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
    graph, logits = capture(lambda: decoder(token, pos, mine))
    for _ in range(2):
        restore(mine, initial)
        eager = decoder(token, pos, mine).clone()
        eager_cache = clone(mine)
        restore(mine, initial)
        logits.zero_()
        graph.replay()
        assert torch.equal(logits, eager)
        assert_caches_equal(mine, eager_cache)
