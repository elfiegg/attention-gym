"""The shared chat driver with a fake model: stop tokens, incremental prefill, and the cache
rebuild after a new CUDA graph's warm-up overwrote cached positions."""

import pytest
import torch
import torch.nn.functional as F

from attn_gym.warpdrive.examples.chat import SYNC_EVERY, Chat

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

VOCAB, STOP = 64, 63


class CountingBackend:
    """Capturable: the next token is ``(pos + 1) % STOP`` until position ``stop_at``, then STOP."""

    capturable = True

    def __init__(self, stop_at: int):
        self.stop_at = stop_at

    def step(self, token, pos):
        nxt = torch.where(pos >= self.stop_at, STOP, (pos.long() + 1) % STOP)
        return F.one_hot(nxt.long(), VOCAB)[0].float()


class FakeModel:
    def __init__(self, stop_at: int):
        self.backend = CountingBackend(stop_at)
        self.tokenizer, self.label, self.max_len, self.stops = None, "fake", 1024, {STOP}
        self.prefills: list[tuple[int, int]] = []  # (start, count)

    def prompt_ids(self, messages):
        ids = []
        for message in messages:
            ids += [1, 2, 3] if message["role"] == "user" else message["ids"]
        return ids

    def prefill(self, ids, start):
        self.prefills.append((start, len(ids)))
        last = start + len(ids) - 1
        return F.one_hot(torch.tensor((last + 1) % STOP, device="cuda"), VOCAB).float()

    def reply_text(self, ids):
        return " ".join(map(str, ids))

    def assistant_message(self, ids):
        return {"role": "assistant", "ids": list(ids)}


def test_stop_incremental_prefill_and_graph_rebuild():
    model = FakeModel(stop_at=12)
    chat = Chat(model)
    turn = chat.reply("hi", None, 0.0, lambda *_: None)
    # Prompt positions 0-2; the prefill samples 3, then steps feed 3, 4, ... until the stop.
    assert turn.text.split() == [str(t) for t in range(3, 13)]
    assert model.prefills == [(0, 3)]
    assert len(turn.text.split()) + 3 < 3 + 10 + SYNC_EVERY  # stopped inside a replay

    model.backend.stop_at = 30
    chat.reply("again", None, 0.0, lambda *_: None)
    assert model.prefills[-1] == (13, 3), "same graph: only the new prompt is prefilled"

    chat.reply("hot", 3, 0.5, lambda *_: None)  # a new graph: warm-up ran at positions 1-8
    assert model.prefills[-1][0] == 0, "the cached prefix must be rebuilt after warm-up"
