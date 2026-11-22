"""Inkling 2 Mini in the warpdrive chat (``attn_gym.warpdrive.examples.chat``).

    python -m attn_gym.warpdrive.examples.inkling_megakernel.chat --backend reference --prompt "Hi"

Prefill runs the PyTorch oracle (``quant.forward_tokens``: ``ref.forward_tokens``, one forward
pass into the caches, dequantizing only the routed experts of a quantized model). Backends:

- ``reference``: the oracle, one eager step at a time (a host sync per step: no CUDA graph);
- ``torch``: ``fast.decode_step`` (the oracle's dataflow on capturable kernels) in CUDA graphs;
- ``compile``: the same step under torch.compile (minutes to compile cold), in CUDA graphs;
- ``megakernel`` (default): the step as warpdrive megakernel launches only
  (``megakernel.InklingDecoder``: one per visit, one per core pass's messages, one for the LM
  head), in CUDA graphs, streaming the checkpoint's experts as stored (NVFP4 and MXFP8,
  ``quant.load_quantized_inkling_weights``: 19 GB of weights). It needs an SM100-family GPU
  and CuTeDSL; elsewhere use ``torch``;
- ``megakernel-bf16``: the same launches on the experts dequantized to bf16 (49 GB), with the
  same tokens bit for bit.

The other backends load bf16 experts.

Replies are sequences of ``<|message_model|><|content_X|>...<|end_message|>`` parts, ``X`` being
``thinking`` or ``text``, closed by ``<|content_model_end_sampling|>``; the reply shows thinking
parts between ``[thinking]`` markers and keeps both for the next turn's history.
"""

import re
from pathlib import Path

import torch
import typer

from attn_gym.warpdrive.examples.chat import SYNC_EVERY, ChatOptions, run
from attn_gym.warpdrive.examples.inkling_megakernel import quant
from attn_gym.warpdrive.examples.inkling_megakernel.fast import decode_step
from attn_gym.warpdrive.examples.inkling_megakernel.ref import (
    CONV_RING,
    InklingCache,
    checkpoint_dir,
    forward_tokens,
    load_inkling_weights,
)

__all__ = ["DEFAULT_BACKEND", "check_options", "checkpoint_dir", "load_chat_model"]

DEFAULT_EFFORT = "low"
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
PART = re.compile(
    r"(?:<\|message_model\|>)?<\|content_(thinking|text)\|>(.*?)(<\|end_message\|>|$)", re.DOTALL
)


class Reference:
    """One eager decode step of the oracle."""

    capturable = False
    quantized = False  # whether the backend takes the checkpoint's experts as stored

    def __init__(self, weights, cache: InklingCache):
        self.weights, self.cache = weights, cache

    def step(self, token: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return forward_tokens(self.weights, token, int(pos.item()), self.cache)


class Torch:
    """``fast.decode_step``, captured by the chat's CUDA graphs."""

    capturable = True
    quantized = False
    compiled = False  # whether the step runs under torch.compile

    def __init__(self, weights, cache: InklingCache):
        self.weights, self.cache = weights, cache
        self.fn = torch.compile(decode_step) if self.compiled else decode_step

    def step(self, token: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return self.fn(self.weights, token, pos, self.cache)


class Megakernel:
    """``megakernel.InklingDecoder``: the step as megakernel launches, captured by the chat's
    CUDA graphs."""

    capturable = True
    quantized = True

    def __init__(self, weights, cache: InklingCache):
        from attn_gym.warpdrive.examples.inkling_megakernel.megakernel import InklingDecoder

        self.decoder, self.cache = InklingDecoder(weights), cache

    def step(self, token: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return self.decoder(token, pos, self.cache)


class MegakernelBf16(Megakernel):
    """``Megakernel`` on the experts dequantized to bf16."""

    quantized = False


class Compile(Torch):
    """``Torch`` under torch.compile."""

    compiled = True


BACKENDS = {
    "reference": Reference,
    "torch": Torch,
    "compile": Compile,
    "megakernel": Megakernel,
    "megakernel-bf16": MegakernelBf16,
}
DEFAULT_BACKEND = "megakernel"


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
        backend = BACKENDS[options.backend]
        load = quant.load_quantized_inkling_weights if backend.quantized else load_inkling_weights
        self.weights = load(checkpoint)
        # The chat decodes up to SYNC_EVERY steps past a stop token, then resumes at the stop
        # token's position: its convolutions read the kernel - 1 earlier ring rows, which those
        # over-run steps must not have overwritten.
        if SYNC_EVERY + self.weights.config.conv_kernel - 1 > CONV_RING:
            raise ValueError(f"CONV_RING {CONV_RING} is too short for SYNC_EVERY {SYNC_EVERY}")
        self.max_len = options.max_len
        self.cache = InklingCache.empty(self.weights.config, options.max_len)
        self.backend = backend(self.weights, self.cache)
        self.label = f"the {options.backend} backend"
        effort = options.reasoning_effort or DEFAULT_EFFORT
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
        tokens = torch.tensor(ids, device="cuda")
        return quant.forward_tokens(self.weights, tokens, start, self.cache)

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
    if options.backend.startswith("megakernel") and options.max_len % 64:
        raise typer.BadParameter("--max-len must be a multiple of 64 (the megakernel's KV block)")
    qwen_only = {"--engine": options.engine, "--thinking": options.thinking}
    qwen_only["--yarn-factor"] = options.yarn_factor
    if given := [name for name, value in qwen_only.items() if value is not None]:
        raise typer.BadParameter(f"Qwen-only options given for Inkling: {', '.join(given)}")
    effort = options.reasoning_effort or DEFAULT_EFFORT
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
