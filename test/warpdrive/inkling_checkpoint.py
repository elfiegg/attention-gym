"""The Inkling 2 Mini checkpoint of the Inkling tests: a local snapshot, never downloaded."""

import pytest

MODEL = "thinkingmachines/Inkling-2-Mini"
WEIGHTS = ("*.json", "model*.safetensors")
TOKENIZER = ("*.jinja", "tiktoken/*")


def checkpoint_path(patterns: tuple[str, ...] = WEIGHTS) -> str:
    """The local snapshot directory of ``MODEL``'s files matching ``patterns``; skips the test
    unless the snapshot is in the local HF cache."""
    hub = pytest.importorskip("huggingface_hub")
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return hub.snapshot_download(MODEL, allow_patterns=list(patterns), local_files_only=True)
    except LocalEntryNotFoundError as error:
        pytest.skip(f"{MODEL} is not in the local HF cache: {error}")
