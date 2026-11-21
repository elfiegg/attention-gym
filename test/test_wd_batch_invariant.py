"""``kernel_options["batch_invariant"]`` on the warpdrive chunk ops: a document's output, final
state and gradients keep their bits whatever else the stream packs."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cutlass.cute", reason="the warpdrive backend requires CuTeDSL")

from attn_gym._backends.arch import cuda_runs_sm100_kernels
from attn_gym.linear import ReplayState, chunk_gdn, chunk_kda, paged_chunk_gdn, paged_chunk_kda
from attn_gym.testing.kda import cumulative_sequence_offsets

pytestmark = pytest.mark.skipif(
    not cuda_runs_sm100_kernels(),
    reason="the warpdrive GDN/KDA kernels require an SM100-family GPU",
)

HEADS, DIM = 4, 128
# Alone, the long document runs the piece chain both ways and the short one the forward d_v
# split (with KDA's shared prep); among the neighbors both run uncut whole-d_v tables. The chain
# and the shared prep change the document's bits; the d_v split alone keeps them, so
# ``batch_invariant`` keeps it (``ForwardPlan.invariant``).
DOCS = {"chain": 8192, "dv": 512}
NEIGHBORS = [512] * 40
INVARIANT = {"backend": "warpdrive", "batch_invariant": True}


def _inputs(family: str, tokens: int, dtype, seed: int, batch: int = 1) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def normal(*shape):
        return torch.randn(*shape, device="cuda", generator=gen)

    shape = (batch, tokens, HEADS, DIM)
    gate_shape = shape if family == "kda" else shape[:3]
    return {
        "q": F.normalize(normal(*shape), dim=-1).to(dtype),
        "k": F.normalize(normal(*shape), dim=-1).to(dtype),
        "v": (normal(*shape) * 0.5).to(dtype),
        # Slow decay, so a chain piece's entry state carries earlier pieces' rounding.
        "gate": -torch.rand(gate_shape, device="cuda", generator=gen) * 0.02,
        "beta": torch.rand(shape[:3], device="cuda", generator=gen),
        "do": normal(*shape).to(dtype),
    }


def _states(count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    shape = (count, HEADS, DIM, DIM)
    return tuple(torch.randn(shape, device="cuda", generator=gen) * 0.05 for _ in range(2))


def _run(family: str, parts, lengths, doc: int, stateful: bool, options) -> dict:
    """Forward and backward of the packed parts; returns the document's results."""
    chunk = chunk_kda if family == "kda" else chunk_gdn
    names = ("q", "k", "v", "gate", "beta")
    packed = {name: torch.cat([part[name] for part in parts], dim=1) for name in (*names, "do")}
    leaves = [packed[name].clone().requires_grad_() for name in names]
    if stateful:
        initial, d_final = (torch.cat(t) for t in zip(*(part["state"] for part in parts)))
        leaves.append(initial.clone().requires_grad_())
    output, final = chunk(
        *leaves[:5],
        leaves[5] if stateful else None,
        cu_seqlens=cumulative_sequence_offsets(lengths),
        output_final_state=stateful,
        kernel_options=options,
    )
    outputs, cotangents = [output], [packed["do"]]
    if stateful:
        outputs.append(final)
        cotangents.append(d_final)
    grads = torch.autograd.grad(outputs, leaves, cotangents)
    start = sum(lengths[:doc])
    rows = slice(start, start + lengths[doc])
    results = {"output": output[:, rows], **{f"d{n}": g[:, rows] for n, g in zip(names, grads)}}
    if stateful:
        results.update(final_state=final[doc], d_initial_state=grads[5][doc])
    return results


def _packs(family: str, doc_tokens: int, dtype, stateful: bool):
    """The document alone, first and last among its neighbors: ``(parts, lengths, index)``."""
    doc = _inputs(family, doc_tokens, dtype, seed=1)
    neighbors = _inputs(family, sum(NEIGHBORS), dtype, seed=2)
    if stateful:
        doc["state"] = _states(1, seed=3)
        neighbors["state"] = _states(len(NEIGHBORS), seed=4)
    return {
        "alone": ((doc,), [doc_tokens], 0),
        "first": ((doc, neighbors), [doc_tokens, *NEIGHBORS], 0),
        "last": ((neighbors, doc), [*NEIGHBORS, doc_tokens], len(NEIGHBORS)),
    }


def _default_plans(family: str, lengths) -> tuple:
    from attn_gym.linear._delta_rule.warpdrive.drivers import gdn, kda

    driver = kda if family == "kda" else gdn
    shape = (sum(lengths), len(lengths), HEADS, DIM, torch.device("cuda"))
    return driver.ForwardPlan.build(*shape), driver.BackwardPlan.build(*shape)


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.int16 if tensor.element_size() == 2 else torch.int32)


CASES = [
    pytest.param(family, doc, dtype, stateful, id=f"{family}-{doc}-{dtype_id}-{state_id}")
    for family in ("kda", "gdn")
    for doc in DOCS
    for dtype, dtype_id in ((torch.bfloat16, "bf16"), (torch.float16, "fp16"))
    for stateful, state_id in ((False, "stateless"), (True, "state"))
    if dtype == torch.bfloat16 or family == "kda"
]


@pytest.mark.parametrize(("family", "doc", "dtype", "stateful"), CASES)
def test_document_bits_ignore_the_pack(family: str, doc: str, dtype, stateful: bool) -> None:
    """The document's default plan changes with the pack (the chain or the d_v split alone,
    uncut packed); with ``batch_invariant`` its output, final state and gradients keep their
    bits. By default the chain and KDA's shared prep change some of them; GDN's d_v split
    alone does not."""
    packs = _packs(family, DOCS[doc], dtype, stateful)
    alone, packed = (_default_plans(family, packs[name][1]) for name in ("alone", "first"))
    flipped = alone[0].pieces if doc == "chain" else alone[0].tiles_per_head > 1
    assert flipped and packed[0] == _default_plans(family, packs["last"][1])[0]
    assert alone[0] != packed[0], (alone, packed)
    results = {
        invariant: {
            name: _run(family, *pack, stateful, {"backend": "warpdrive", **options})
            for name, pack in packs.items()
        }
        for invariant, options in ((True, INVARIANT), (False, {}))
    }
    default_changes = doc == "chain" or family == "kda"
    for name in ("first", "last"):
        for result, tensor in results[True][name].items():
            want = results[True]["alone"][result]
            assert torch.equal(_bits(tensor), _bits(want)), (name, result)
        changed = any(
            not torch.equal(tensor, results[False]["alone"][result])
            for result, tensor in results[False][name].items()
        )
        assert changed == default_changes, name


@pytest.mark.parametrize("family", ["kda", "gdn"])
def test_paged_calls_ignore_the_pack(family: str) -> None:
    """The paged forward's d_v split (and KDA's shared prep) follows the pack too; with
    ``batch_invariant`` the document's output and advanced slot keep their bits. By default
    KDA's shared prep changes them."""
    paged = paged_chunk_kda if family == "kda" else paged_chunk_gdn
    doc_tokens = DOCS["dv"]
    doc = _inputs(family, doc_tokens, torch.bfloat16, seed=1)
    neighbors = _inputs(family, sum(NEIGHBORS), torch.bfloat16, seed=2)
    pool = _states(len(NEIGHBORS) + 2, seed=3)[0]  # slot 0 stays unused
    names = ("q", "k", "v", "gate", "beta")

    def run(parts, lengths, slots, doc_slot, start, options):
        tokens = [torch.cat([part[name] for part in parts], dim=1) for name in names]
        advanced = pool.clone()
        with torch.no_grad():
            output = paged(
                *tokens, advanced, torch.tensor(slots, dtype=torch.int32, device="cuda"),
                cu_seqlens=cumulative_sequence_offsets(lengths), kernel_options=options,
            )  # fmt: skip
        return output[:, start : start + doc_tokens], advanced[doc_slot]

    others = list(range(2, len(NEIGHBORS) + 2))
    packs = {
        "alone": ((doc,), [doc_tokens], [1], 0),
        "first": ((doc, neighbors), [doc_tokens, *NEIGHBORS], [1, *others], 0),
        "last": ((neighbors, doc), [*NEIGHBORS, doc_tokens], [*others, 1], sum(NEIGHBORS)),
    }
    results = {
        invariant: {
            name: run(parts, lengths, slots, 1, start, options)
            for name, (parts, lengths, slots, start) in packs.items()
        }
        for invariant, options in ((True, INVARIANT), (False, {"backend": "warpdrive"}))
    }
    for name in ("first", "last"):
        for got, want in zip(results[True][name], results[True]["alone"], strict=True):
            assert torch.equal(got, want), name
        changed = not torch.equal(results[False][name][0], results[False]["alone"][0])
        assert changed == (family == "kda"), name


def test_replay_calls_ignore_the_batch() -> None:
    """Replay-backed KDA runs its prefill and decode windows on the pinned plans too: a request's
    output keeps its bits whether it is served alone or beside others (by default the decode
    windows' d_v split follows the batch)."""
    tokens, prefill, batch = 100, 70, 40
    inputs = _inputs("kda", tokens, torch.bfloat16, seed=5, batch=batch)

    def serve(requests: int, options) -> torch.Tensor:
        state_cache = torch.zeros(requests + 1, HEADS, DIM, DIM, device="cuda")
        replay = ReplayState(
            *(
                torch.zeros((requests + 1, *shape), device="cuda", dtype=dtype)
                for shape, dtype in ReplayState.slot_layout(HEADS, DIM)
            )
        )
        slots = torch.arange(1, requests + 1, dtype=torch.int32, device="cuda")
        outputs = []
        with torch.no_grad():
            for start, stop in ((0, prefill), *((t, t + 1) for t in range(prefill, tokens))):
                window = (
                    inputs[name][:requests, start:stop] for name in ("q", "k", "v", "gate", "beta")
                )
                outputs.append(
                    paged_chunk_kda(
                        *window, state_cache, slots, kernel_options=options, replay_state=replay
                    )
                )
        return torch.cat(outputs, dim=1)[0]

    assert torch.equal(serve(1, INVARIANT), serve(batch, INVARIANT))
    default = {"backend": "warpdrive"}
    assert not torch.equal(serve(1, default), serve(batch, default))


def test_option_validation() -> None:
    inputs = _inputs("kda", 64, torch.bfloat16, seed=0)
    q, k, v, gate, beta = (inputs[name] for name in ("q", "k", "v", "gate", "beta"))
    with pytest.raises(ValueError, match="requires kernel_options\\['backend'\\]='warpdrive'"):
        chunk_kda(q, k, v, gate, beta, kernel_options={"batch_invariant": True})
    with pytest.raises(ValueError, match="batch_invariant excludes the split schedules"):
        chunk_kda(q, k, v, gate, beta, kernel_options={**INVARIANT, "split_forward": True})


@pytest.mark.parametrize("family", ["kda", "gdn"])
def test_staged_handles_ignore_the_pack(family: str) -> None:
    """The staged handles' ``run`` (forward and backward) runs the same pinned plans: the
    document's results keep their bits alone and among neighbors."""
    from attn_gym.linear.gdn.stages import chunk_gdn_prepare, chunk_gdn_prepare_backward
    from attn_gym.linear.kda.stages import chunk_kda_prepare, chunk_kda_prepare_backward

    prepare, prepare_backward = (
        (chunk_kda_prepare, chunk_kda_prepare_backward)
        if family == "kda"
        else (chunk_gdn_prepare, chunk_gdn_prepare_backward)
    )
    names = ("q", "k", "v", "gate", "beta")
    results = {}
    for name, (parts, lengths, doc) in _packs(family, DOCS["chain"], torch.bfloat16, True).items():
        packed = {n: torch.cat([part[n] for part in parts], dim=1) for n in (*names, "do")}
        initial, d_final = (torch.cat(t) for t in zip(*(part["state"] for part in parts)))
        cu = cumulative_sequence_offsets(lengths)
        prepared = prepare(*(packed[n] for n in names), cu_seqlens=cu, kernel_options=INVARIANT)
        output, final = prepared.run(initial, output_final_state=True)
        backward = prepare_backward(
            prepared.saved, packed["do"], initial, scale=prepared.scale, kernel_options=INVARIANT
        )
        *grads, d_initial = backward.run(d_final)
        start = sum(lengths[:doc])
        rows = slice(start, start + lengths[doc])
        results[name] = (output[:, rows], final[doc], *(g[:, rows] for g in grads), d_initial[doc])
    for name in ("first", "last"):
        for got, want in zip(results[name], results["alone"], strict=True):
            assert torch.equal(got, want), name


def _op_args(family: str, name: str, dtype) -> tuple:
    """The op's positional arguments before ``batch_invariant`` (a serialized graph's call);
    callers append it. Fresh tensors per call: the paged ops advance their pool."""
    x = _inputs(family, 1000, dtype, seed=6)
    operands = tuple(x[n] for n in ("q", "k", "v", "gate", "beta"))
    do, cu, scale = x["do"], cumulative_sequence_offsets([1000]), DIM**-0.5
    state, d_state = _states(1, seed=7)
    pool, slots = _states(2, seed=8)[0], torch.ones(1, dtype=torch.int32, device="cuda")
    if family == "gdn":
        return {
            "fwd": (*operands, cu, False, scale),
            "fwd_with_initial_state": (*operands, state, cu, scale),
            "fwd_with_state": (*operands, state, cu, scale),
            "fwd_paged": (*operands, pool, slots, None, cu, scale),
            "bwd": (*operands, do, cu, False, scale),
            "bwd_with_state": (*operands, do, state, d_state, cu, scale),
        }[name]
    dtypes = "tf32+fp16" if dtype == torch.float16 else "tf32+bf16"
    return {
        "fwd": (*operands, cu, False, scale, True, dtypes),
        "fwd_with_initial_state": (*operands, state, cu, scale, True, dtypes),
        "fwd_with_state": (*operands, state, cu, scale, True, False, dtypes),
        "fwd_paged": (*operands, pool, slots, None, cu, scale, dtypes),
        "local_bwd": (*operands, do, cu, False, scale, True, dtypes),
        "bwd_with_state": (*operands, do, cu, state, d_state, scale, True, dtypes),
        "bwd_with_exit_cotangent": (*operands, do, cu, d_state, scale, True, dtypes),
    }[name]


OPS = [
    *(("kda", name) for name in ("fwd", "fwd_with_initial_state", "fwd_with_state", "fwd_paged")),
    *(("kda", name) for name in ("local_bwd", "bwd_with_state", "bwd_with_exit_cotangent")),
    *(("gdn", name) for name in ("fwd", "fwd_with_initial_state", "fwd_with_state", "fwd_paged")),
    *(("gdn", name) for name in ("bwd", "bwd_with_state")),
]


@pytest.mark.parametrize(("family", "name"), [pytest.param(*op, id="-".join(op)) for op in OPS])
def test_pre_option_op_calls_bind_as_before(family: str, name: str) -> None:
    """``batch_invariant`` is each op's last argument with default ``False``: a call serialized
    before it binds and runs the default plans (bitwise), and the schemas pass ``opcheck``."""
    op = getattr(torch.ops.attn_gym, f"{family}_chunk_cudnn_packed_{name}").default

    def results(*extra):
        args = _op_args(family, name, torch.bfloat16)
        out = op(*args, *extra)
        return (out if isinstance(out, tuple) else (out,)) + (
            (args[5],) if name == "fwd_paged" else ()
        )

    for got, want in zip(results(), results(False), strict=True):
        assert torch.equal(got, want)
    for batch_invariant in (False, True):
        args = (*_op_args(family, name, torch.bfloat16), batch_invariant)
        torch.library.opcheck(op, args, test_utils=("test_schema", "test_faketensor"))


@pytest.mark.parametrize("family", ["kda", "gdn"])
def test_flag_compiles_no_kernels_of_its_own(family: str) -> None:
    """Across token and sequence counts of every specialization class, the flag reuses the
    uncut bundles' compiles, keyed by the value-tile count (the d_v split) and state flags,
    and never builds a chain bundle."""
    from attn_gym.linear._delta_rule.warpdrive.bundles import (
        gdn_chain_backward,
        gdn_chain_forward,
        gdn_warmup_backward,
        gdn_warmup_forward,
        kda_chain_backward,
        kda_chain_forward,
        kda_warmup_backward,
        kda_warmup_forward,
    )

    if family == "kda":
        uncut = (kda_warmup_forward._compile_warmup_forward,)
        uncut_backward = (kda_warmup_backward._compile_warmup_backward,)
        chains = (
            kda_chain_forward._compile_chain_forward,
            kda_chain_backward._compile_chain_backward,
        )
    else:
        uncut = (gdn_warmup_forward._compile_warmup_forward,)
        uncut_backward = (gdn_warmup_backward._compile_warmup_backward,)
        chains = (
            gdn_chain_forward._compile_chain_forward,
            gdn_chain_backward._compile_chain_backward_head,
            gdn_chain_backward._compile_chain_backward_tail,
        )
    for compile_fn in (*uncut, *uncut_backward, *chains):
        compile_fn.cache_clear()
    chunk = chunk_kda if family == "kda" else chunk_gdn
    # Sequence counts on both sides of the d_v split; token counts 1, a multiple of 16 and not.
    for lengths in ([1], [16], [1000], [8192], [64] * 3, [16] * 40, [1000] * 40, [3, 17, 4096]):
        x = _inputs(family, sum(lengths), torch.bfloat16, seed=0)
        leaves = [x[n].requires_grad_() for n in ("q", "k", "v", "gate", "beta")]
        output, _ = chunk(
            *leaves, cu_seqlens=cumulative_sequence_offsets(lengths), kernel_options=INVARIANT
        )
        torch.autograd.grad(output, leaves, x["do"])
    assert all(fn.cache_info().currsize <= 2 for fn in uncut), [f.cache_info() for f in uncut]
    assert all(fn.cache_info().currsize == 1 for fn in uncut_backward)
    assert all(fn.cache_info().misses == 0 for fn in chains)


@pytest.mark.parametrize("family", ["kda", "gdn"])
def test_full_graph_compile_runs_the_invariant_plans(family: str) -> None:
    """Under ``torch.compile(fullgraph=True)`` the flag reaches the ops, forward and backward:
    a lone long document (chained by default) gets the eager invariant bits."""
    chunk = chunk_kda if family == "kda" else chunk_gdn
    x = _inputs(family, DOCS["chain"], torch.bfloat16, seed=1)
    names = ("q", "k", "v", "gate", "beta")

    def run(fn, options):
        leaves = [x[n].clone().requires_grad_() for n in names]
        output, _ = fn(*leaves, kernel_options=options)
        return output, *torch.autograd.grad(output, leaves, x["do"])

    compiled = run(torch.compile(chunk, fullgraph=True), INVARIANT)
    eager = run(chunk, INVARIANT)
    default = run(chunk, {"backend": "warpdrive"})
    for got, want in zip(compiled, eager, strict=True):
        assert torch.equal(got, want)
    assert any(not torch.equal(got, want) for got, want in zip(compiled, default, strict=True))


@pytest.mark.parametrize("family", ["kda", "gdn"])
def test_staged_backward_pins_exactly_when_its_forward_did(family: str, monkeypatch) -> None:
    """With the summaries uncut but the backward chained by default, a default forward's
    backward runs the unstaged op's default (chained) plan and a ``batch_invariant`` forward's
    runs its pinned plan, bit for bit, whatever the backward's own option; asking a default
    forward's backward to pin raises."""
    from attn_gym.linear._delta_rule.warpdrive.drivers import summary
    from attn_gym.linear.gdn.stages import chunk_gdn_prepare, chunk_gdn_prepare_backward
    from attn_gym.linear.kda.stages import chunk_kda_prepare, chunk_kda_prepare_backward

    monkeypatch.setattr(summary, "summary_span", lambda *args, **kwargs: 0)
    prepare, prepare_backward = (
        (chunk_kda_prepare, chunk_kda_prepare_backward)
        if family == "kda"
        else (chunk_gdn_prepare, chunk_gdn_prepare_backward)
    )
    lengths, names = [DOCS["chain"]], ("q", "k", "v", "gate", "beta")
    assert _default_plans(family, lengths)[1].chain
    x = _inputs(family, lengths[0], torch.bfloat16, seed=1)
    x["state"] = _states(1, seed=3)
    initial, d_final = x["state"]
    default = {"backend": "warpdrive"}
    results = {}
    for name, options in (("default", default), ("pinned", INVARIANT)):
        prepared = prepare(*(x[n] for n in names), kernel_options=options)
        assert prepared.pieces is None
        backward = prepare_backward(
            prepared.saved, x["do"], initial, scale=prepared.scale, kernel_options=default
        )
        *grads, d_initial = backward.run(d_final)
        results[name] = (*grads, d_initial[0])
        unstaged = _run(family, (x,), lengths, 0, True, options)
        want = (*(unstaged[f"d{n}"] for n in names), unstaged["d_initial_state"])
        for got, expected in zip(results[name], want, strict=True):
            assert torch.equal(_bits(got), _bits(expected)), name
    pairs = zip(results["default"], results["pinned"], strict=True)
    assert any(not torch.equal(_bits(a), _bits(b)) for a, b in pairs)
    prepared = prepare(*(x[n] for n in names), kernel_options=default)
    with pytest.raises(ValueError, match="batch_invariant"):
        prepare_backward(
            prepared.saved, x["do"], initial, scale=prepared.scale, kernel_options=INVARIANT
        )
