"""Inkling 2 Mini batch-one decode timings in CUDA graphs: each launch kind of the megakernel
decode step (``megakernel.InklingDecoder``) and the whole step, against the torch step
(``fast.decode_step``).

    python -m attn_gym.warpdrive.examples.inkling_megakernel.bench --context 1100

The cache is prefilled with ``context`` tokens by the oracle; every timed call decodes at that
position with the inputs a real megakernel step produced there, rewriting the same cache rows
(a fixed-position, reused-buffer workload: weights larger than L2 stream from DRAM, the caches
and activations may stay in L2). A launch graph replays ``reps`` calls of one launch of the
step back to back and reports the time per call; the breakdown multiplies each kind's time by
its launches per step, and the rest of the measured step is the gaps between launches, plus
whatever a repeated launch gains over its first run inside a step (an attention layer's ~54 MB
of weights may stay partly in L2 across repeats despite evict-first loads). Graphs
are timed in ``rounds`` interleaved rounds (rotating order) and each row reports the median of
the round medians. Default DVFS.
"""

import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Annotated, NamedTuple

import torch
import typer

from attn_gym.warpdrive.examples.bench_support import (
    benchmark_cuda_graph_stats,
    provenance,
    settle,
)
from attn_gym.warpdrive.examples.inkling_megakernel import ref
from attn_gym.warpdrive.examples.inkling_megakernel.fast import decode_step
from attn_gym.warpdrive.examples.inkling_megakernel.megakernel import InklingDecoder


class Captured(NamedTuple):
    """A graph of ``reps`` calls of ``fn``. ``fn`` stays referenced: its closure owns buffers
    the graph's kernels read and write (runners, caches, inputs)."""

    graph: torch.cuda.CUDAGraph
    reps: int
    fn: Callable[[], object]


def capture(fn: Callable[[], object], reps: int, warmup: int = 3) -> Captured:
    """A graph of ``reps`` back-to-back calls of ``fn``, after ``warmup`` eager calls on the
    capture stream (compiles and allocator pools)."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(reps):
                fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    return Captured(graph, reps, fn)


def time_graphs(graphs: dict[str, Captured], rounds: int, iters: int) -> dict[str, float]:
    """Median over ``rounds`` interleaved rounds (each in a rotated order) of each graph's
    median replay time, divided by its calls per replay: microseconds per call."""
    settle([c.graph for c in graphs.values()], 1.0)
    names = list(graphs)
    medians = {name: [] for name in names}
    for r in range(rounds):
        for name in names[r % len(names) :] + names[: r % len(names)]:
            stats = benchmark_cuda_graph_stats(graphs[name].graph, num_iters=iters, warmup_iters=5)
            medians[name].append(stats.median_us / graphs[name].reps)
    return {name: sorted(m)[len(m) // 2] for name, m in medians.items()}


class Recorder:
    """A runner whose calls also append themselves to ``calls``, ready to replay."""

    def __init__(self, runner, calls: list):
        self.runner, self.calls = runner, calls

    def __call__(self, *args, **kwargs):
        self.calls.append(partial(self.runner, *args, **kwargs))
        return self.runner(*args, **kwargs)


def launch_kinds(config: ref.InklingConfig) -> dict[str, tuple[str, int, int]]:
    """``{kind: (runner, call, launches per step)}``: each launch kind of a step, the runner
    and the index of the call that represents it (the first of the kind)."""
    kinds = {}

    def add(kind, runner, call):
        first, count = kinds.get(kind, (runner, call, 0))[1:]
        kinds[kind] = (runner, first, count + 1)

    for visit, layer, p in config.visits():
        layer_kind = "full" if config.is_full(layer) else "sliding"
        messages = " + messages" if config.is_loop(layer) and p > 0 else ""
        add(f"attention{messages}, {layer_kind}", "attn", visit)
        add(f"MLP{messages}", "mlp", visit)
    for p in range(1, config.num_passes):
        add("messages", "msg", p - 1)
    add("head", "head", 0)
    return kinds


@torch.no_grad()
def main(
    context: Annotated[int, typer.Option(help="prefilled tokens; the decoded position")] = 1100,
    max_len: Annotated[int, typer.Option(help="cache capacity (a multiple of 64)")] = 2048,
    reps: Annotated[int, typer.Option(help="calls per launch graph replay")] = 20,
    rounds: Annotated[int, typer.Option(help="interleaved timing rounds (odd)")] = 5,
    iters: Annotated[int, typer.Option(help="replays per graph per round")] = 50,
    checkpoint: str = "thinkingmachines/Inkling-2-Mini",
    json_out: Annotated[Path | None, typer.Option(help="write results, provenance")] = None,
):
    weights = ref.load_inkling_weights(checkpoint)
    config = weights.config
    cache = ref.InklingCache.empty(config, max_len)
    gen = torch.Generator(device="cuda").manual_seed(0)
    prompt = torch.randint(1000, 100000, (context,), device="cuda", generator=gen)
    token = ref.forward_tokens(weights, prompt, 0, cache).argmax().reshape(1)
    pos = torch.tensor([context], device="cuda", dtype=torch.int32)
    decoder = InklingDecoder(weights)
    calls = {name: [] for name in ("attn", "mlp", "msg", "head")}
    for name, recorded in calls.items():
        setattr(decoder, name, Recorder(getattr(decoder, name), recorded))
    decoder(token, pos, cache)
    for name in calls:
        setattr(decoder, name, getattr(decoder, name).runner)

    kinds = launch_kinds(config)
    graphs = {
        kind: capture(calls[runner][call], reps) for kind, (runner, call, _) in kinds.items()
    }
    graphs["decode step: torch"] = capture(partial(decode_step, weights, token, pos, cache), 1)
    graphs["decode step: megakernel"] = capture(partial(decoder, token, pos, cache), 1)
    results = time_graphs(graphs, rounds, iters)
    print(f"{'launch kind':40s} {'us/launch':>10s} {'per step':>9s} {'us/step':>9s}")
    launches = 0.0
    for kind, (_, _, count) in kinds.items():
        launches += count * results[kind]
        print(f"{kind:40s} {results[kind]:10.1f} {count:9d} {count * results[kind]:9.1f}")
    step = results["decode step: megakernel"]
    print(f"{'launches (sum)':40s} {'':10s} {'':9s} {launches:9.1f}")
    print(f"{'gaps (step - sum)':40s} {'':10s} {'':9s} {step - launches:9.1f}")
    for name in ("decode step: torch", "decode step: megakernel"):
        print(f"{name:40s} {results[name]:10.1f} us")
    if json_out is not None:
        here = Path(__file__).parent
        sources = {p.stem: p for p in sorted(here.glob("*.py"))}
        arguments = {"context": context, "max_len": max_len, "reps": reps, "rounds": rounds,
                     "iters": iters}  # fmt: skip
        json_out.write_text(
            json.dumps({"results_us": results, **provenance(sources, arguments)}, indent=2)
        )


if __name__ == "__main__":
    typer.run(main)
