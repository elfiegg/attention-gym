"""The Inkling 2 Mini oracle against HF's ``Inkling2LoopForCausalLM`` on a small random model.

Checkpoint-free: a random tiny model is saved in the original checkpoint naming and loaded by
both implementations. The tiny shape keeps every structural feature: sliding and full layers,
block boundaries inside and outside the loop, three passes with messages, a sliding window and
relative extent the prompt outgrows, and enough tokens to wrap the convolution rings.
"""

import pytest
import torch

from attn_gym.warpdrive.examples.inkling_megakernel.fast import decode_step
from attn_gym.warpdrive.examples.inkling_megakernel.ref import (
    CONV_RING,
    InklingCache,
    forward_tokens,
    load_inkling_weights,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
transformers_loop = pytest.importorskip(
    "transformers.models.inkling2_loop.modeling_inkling2_loop",
    reason="needs transformers with Inkling2Loop (huggingface/transformers add-inkling2-mini)",
)

PROMPT, STEPS = 24, CONV_RING + 4


def tiny_config():
    """Two layers per block, so the four-layer core spans two blocks as in the real model (12
    layers of 6): passes after the first then read messages from a completed block."""
    from transformers import Inkling2LoopTextConfig

    return Inkling2LoopTextConfig(
        hidden_size=256, num_hidden_layers=8, layer_types=["hybrid_sliding", "hybrid"] * 4,
        num_attention_heads=6, num_key_value_heads=2, head_dim=64,
        swa_num_attention_heads=6, swa_num_key_value_heads=2, swa_head_dim=64,
        sliding_window_size=8, rel_extent=16, d_rel=16, log_scaling_n_floor=16,
        vocab_size=512, unpadded_vocab_size=500, eos_token_id=1,
        moe_intermediate_size=128, n_routed_experts=16, num_experts_per_tok=4,
        n_layers_per_attn_res_block=2, loop_layer_index_start=2, loop_layer_index_end=6,
        loop_num_passes=3, loop_gate_rank=16,
    )  # fmt: skip


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    model_class = transformers_loop.Inkling2LoopForCausalLM
    torch.manual_seed(0)
    model = model_class(tiny_config())
    with torch.no_grad():  # zero-initialized or tiny parameters would hide their wiring
        for name, param in model.named_parameters():
            if "message_gate" in name:  # gates near 0 would barely mix in the messages
                param.normal_(0, 2 * param.shape[-1] ** -0.5)
            elif param.abs().max() == 0:
                param.normal_(0, 0.5)
            elif "norm" in name or name.endswith("scale"):
                param.add_(torch.randn_like(param) * 0.2)
    path = tmp_path_factory.mktemp("tiny_inkling")
    model.to(torch.bfloat16).save_pretrained(path)
    hf = model_class.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="eager")
    return hf.cuda().eval(), load_inkling_weights(path)


def prompt(weights):
    generator = torch.Generator().manual_seed(1)
    return torch.randint(weights.config.vocab_size, (PROMPT,), generator=generator).cuda()


@torch.no_grad()
def test_prefill_and_decode_match_hf_bitwise(models):
    """Prefill, then greedy decode past the sliding window, the relative extent, the log-scaling
    floor and a wrap of the convolution rings: every logit equals HF's (eager attention)."""
    hf, weights = models
    tokens = prompt(weights)
    cache = InklingCache.empty(weights.config, max_len=64)
    output = hf(tokens[None], use_cache=True)
    logits = forward_tokens(weights, tokens, 0, cache, all_logits=True)
    torch.testing.assert_close(logits, output.logits[0], atol=0, rtol=0)
    for step in range(STEPS):
        token = output.logits[0, -1].argmax()
        output = hf(token.reshape(1, 1), cache=output.past_key_values, use_cache=True)
        logits = forward_tokens(weights, token, PROMPT + step, cache)
        torch.testing.assert_close(logits, output.logits[0, -1], atol=0, rtol=0, msg=f"{step}")


@torch.no_grad()
def test_split_prefill_matches_hf_bitwise(models):
    """A second prefill (a chat's next turn) reads the first one's caches."""
    hf, weights = models
    tokens = prompt(weights)
    cache = InklingCache.empty(weights.config, max_len=64)
    forward_tokens(weights, tokens[:13], 0, cache)
    logits = forward_tokens(weights, tokens[13:], 13, cache, all_logits=True)
    output = hf(tokens[None, :13], use_cache=True)
    expected = hf(tokens[None, 13:], cache=output.past_key_values, use_cache=True).logits[0]
    torch.testing.assert_close(logits, expected, atol=0, rtol=0)


@torch.no_grad()
def test_fast_decode_step_tracks_oracle_and_captures(models):
    """``fast.decode_step`` differs from the oracle only in summation order: same greedy tokens
    and close logits over the window, extent and ring boundaries. Its CUDA-graph replay equals its
    eager run bit for bit."""
    _, weights = models
    tokens = prompt(weights)
    oracle = InklingCache.empty(weights.config, max_len=64)
    fast = InklingCache.empty(weights.config, max_len=64)
    forward_tokens(weights, tokens, 0, oracle)
    forward_tokens(weights, tokens, 0, fast)
    token, pos = tokens[-1:].clone(), torch.zeros(1, device="cuda", dtype=torch.int32)
    for step in range(STEPS):
        expected = forward_tokens(weights, token, PROMPT + step, oracle)
        pos.fill_(PROMPT + step)
        logits = decode_step(weights, token, pos, fast)
        torch.testing.assert_close(logits, expected, atol=0.1, rtol=0, msg=f"step {step}")
        assert logits.argmax() == expected.argmax()
        token = expected.argmax().reshape(1)

    # Replay the last step: same inputs, same caches (it rewrites the same rows).
    eager = decode_step(weights, token, pos, fast)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = decode_step(weights, token, pos, fast)
    graph.replay()
    torch.testing.assert_close(captured, eager, atol=0, rtol=0)
