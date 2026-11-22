"""The Inkling 2 Mini checkpoint of the Inkling tests: a local snapshot, never downloaded."""

import json
from pathlib import Path

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
        path = hub.snapshot_download(MODEL, allow_patterns=list(patterns), local_files_only=True)
    except LocalEntryNotFoundError as error:
        pytest.skip(f"{MODEL} is not in the local HF cache: {error}")
    # A snapshot can exist with only some files (say config.json): check the weight shards the
    # index names, so a partial cache skips instead of failing in the loader.
    index = Path(path) / "model.safetensors.index.json"
    if "model*.safetensors" in patterns:
        shards = (
            set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else ()
        )
        missing = [s for s in shards if not (Path(path) / s).exists()] if shards else ["index"]
        if missing:
            pytest.skip(f"{MODEL}'s local snapshot lacks {missing}")
    return path
