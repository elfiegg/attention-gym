"""Inkling 2 Mini in the warpdrive chat (``attn_gym.warpdrive.examples.chat``).

    python -m attn_gym.warpdrive.examples.inkling_megakernel.chat --backend reference --prompt "Hi"

Prefill runs the PyTorch oracle (``ref.forward_tokens``, one forward pass into the caches).
Backends:

- ``reference``: the oracle, one eager step at a time (a host sync per step: no CUDA graph);
- ``torch``: ``fast.decode_step`` (the oracle's dataflow on capturable kernels) in CUDA graphs;
- ``compile``: the same step under torch.compile (minutes to compile cold), in CUDA graphs;
- ``megakernel``: the ``torch`` step with each plain visit's MLP half as one warpdrive
  megakernel launch (``megakernel.InklingMlpMegakernel``; visits with messages stay on torch),
  in CUDA graphs.

Replies are sequences of ``<|message_model|><|content_X|>...<|end_message|>`` parts, ``X`` being
``thinking`` or ``text``, closed by ``<|content_model_end_sampling|>``; the reply shows thinking
parts between ``[thinking]`` markers and keeps both for the next turn's history.
"""

import functools
import re
from pathlib import Path

import torch
import typer

from attn_gym.warpdrive.examples.chat import SYNC_EVERY, ChatOptions, run
from attn_gym.warpdrive.examples.inkling_megakernel.fast import decode_step
from attn_gym.warpdrive.examples.inkling_megakernel.ref import (
    CONV_RING,
    InklingCache,
    checkpoint_dir,
    forward_tokens,
    load_inkling_weights,
)

__all__ = ["DEFAULT_BACKEND", "check_options", "checkpoint_dir", "load_chat_model"]

EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
PART = re.compile(
    r"(?:<\|message_model\|>)?<\|content_(thinking|text)\|>(.*?)(<\|end_message\|>|$)", re.DOTALL
)


class Reference:
    """One eager decode step of the oracle."""

    capturable = False

    def __init__(self, weights, cache: InklingCache):
        self.weights, self.cache = weights, cache

    def step(self, token: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return forward_tokens(self.weights, token, int(pos.item()), self.cache)


class Torch:
    """``fast.decode_step``, captured by the chat's CUDA graphs."""

    capturable = True

    def __init__(self, weights, cache: InklingCache, compile: bool = False):
        self.weights, self.cache = weights, cache
        self.fn = torch.compile(decode_step) if compile else decode_step

    def step(self, token: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return self.fn(self.weights, token, pos, self.cache)


class Megakernel:
    """``fast.decode_step`` with the megakernel MLP half (stage S1 of the Inkling megakernel),
    captured by the chat's CUDA graphs."""

    capturable = True

    def __init__(self, weights, cache: InklingCache):
        from attn_gym.warpdrive.examples.inkling_megakernel.megakernel import (
            InklingMlpMegakernel,
        )

        self.weights, self.cache = weights, cache
        self.kernels = InklingMlpMegakernel(weights.config, weights.layers).kernels()

    def step(self, token: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return decode_step(self.weights, token, pos, self.cache, self.kernels)


BACKENDS = {
    "reference": Reference,
    "torch": Torch,
    "compile": functools.partial(Torch, compile=True),
    "megakernel": Megakernel,
}
DEFAULT_BACKEND = "torch"


def reply_parts(text: str) -> list[tuple[str, str, bool]]:
    """``(kind, text, closed)`` parts of a reply decoded with its special tokens, kind
    ``thinking`` or ``text``; only the last part may be unclosed (still streaming)."""
    return [(match[1], match[2], bool(match[3])) for match in PART.finditer(text)]


def assistant_message(text: str) -> dict:
    """The chat-template message of a reply decoded with its special tokens. The template keeps
    one thinking and one text part per message, so a reply with several parts of a kind
    re-renders them joined (and thinking first); the chat then rebuilds its caches. Otherwise the
    rendering reproduces the reply's tokens and the next turn reuses their cache entries."""
    parts = reply_parts(text)
    message = {"role": "assistant", "content": "".join(t for k, t, _ in parts if k == "text")}
    thinking = "".join(t for k, t, _ in parts if k == "thinking")
    if thinking:
        message["reasoning_content"] = thinking
    return message


class InklingChatModel:
    def __init__(self, checkpoint: Path, options: ChatOptions):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.weights = load_inkling_weights(checkpoint)
        # The chat decodes up to SYNC_EVERY steps past a stop token, then resumes at the stop
        # token's position: its convolutions read the kernel - 1 earlier ring rows, which those
        # over-run steps must not have overwritten.
        if SYNC_EVERY + self.weights.config.conv_kernel - 1 > CONV_RING:
            raise ValueError(f"CONV_RING {CONV_RING} is too short for SYNC_EVERY {SYNC_EVERY}")
        self.max_len = options.max_len
        self.cache = InklingCache.empty(self.weights.config, options.max_len)
        self.backend = BACKENDS[options.backend](self.weights, self.cache)
        self.label = f"the {options.backend} backend"
        effort = options.reasoning_effort
        self.effort = effort if effort in EFFORTS else float(effort)
        stops = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        ]
        self.stops = {s for s in stops if isinstance(s, int) and s >= 0}

    def prompt_ids(self, messages: list[dict]) -> list[int]:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, reasoning_effort=self.effort
        )
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def prefill(self, ids: list[int], start: int) -> torch.Tensor:
        return forward_tokens(self.weights, torch.tensor(ids, device="cuda"), start, self.cache)

    def parts(self, ids: list[int]) -> list[tuple[str, str, bool]]:
        return reply_parts(self.tokenizer.decode(ids, skip_special_tokens=False))

    def reply_text(self, ids: list[int]) -> str:
        """Thinking between markers; the closing marker waits for the part's end, so each
        text extends the previous one (the chat streams the difference)."""
        shown = []
        for kind, text, closed in self.parts(ids):
            if kind == "thinking":
                text = f"[thinking]{text}" + ("[/thinking]\n" if closed else "")
            shown.append(text)
        return "".join(shown)

    def assistant_message(self, ids: list[int]) -> dict:
        return assistant_message(self.tokenizer.decode(ids, skip_special_tokens=False))


def check_options(options: ChatOptions):
    if options.backend not in BACKENDS:
        raise typer.BadParameter(f"Inkling backends: {sorted(BACKENDS)}, got {options.backend!r}")
    effort = options.reasoning_effort
    if effort not in EFFORTS:
        try:
            valid = 0 <= float(effort) <= 0.99
        except ValueError:
            valid = False
        if not valid:
            raise typer.BadParameter(f"--reasoning-effort: one of {EFFORTS} or 0-0.99")


def load_chat_model(checkpoint: Path, options: ChatOptions) -> InklingChatModel:
    return InklingChatModel(checkpoint, options)


if __name__ == "__main__":
    run("thinkingmachines/Inkling-2-Mini")
