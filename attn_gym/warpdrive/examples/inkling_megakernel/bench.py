"""Inkling 2 Mini batch-one decode timings in CUDA graphs: one plain visit's attention and MLP
halves (torch ``fast.FAST`` kernels against the megakernel launches) and the decode step
(torch, the MLP-half hybrid, both halves).

    python -m attn_gym.warpdrive.examples.inkling_megakernel.bench --context 1100

The cache is prefilled with ``context`` tokens by the oracle; every timed call decodes at that
position with the visit inputs a real decode step produced there, rewriting the same cache
rows (a fixed-position, reused-buffer workload: weights larger than L2 stream from DRAM, the
caches and activations may stay in L2). A visit graph replays ``reps`` calls back to back and
reports the time per call; graphs are timed in ``rounds`` interleaved rounds (rotating order)
and each row reports the median of the round medians. Default DVFS.
"""

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import torch
import typer

from attn_gym.warpdrive.examples.bench_support import (
    benchmark_cuda_graph_stats,
    provenance,
    settle,
)
from attn_gym.warpdrive.examples.inkling_megakernel import ref
from attn_gym.warpdrive.examples.inkling_megakernel.fast import FAST, decode_step
from attn_gym.warpdrive.examples.inkling_megakernel.megakernel import (
    InklingMegakernel,
    InklingMlpMegakernel,
)


def capture(fn: Callable[[], object], reps: int, warmup: int = 3) -> torch.cuda.CUDAGraph:
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
    return graph


def time_graphs(
    graphs: dict[str, tuple[torch.cuda.CUDAGraph, int]], rounds: int, iters: int
) -> dict[str, float]:
    """Median over ``rounds`` interleaved rounds (each in a rotated order) of each graph's
    median replay time, divided by its calls per replay: microseconds per call."""
    settle([g for g, _ in graphs.values()], 1.0)
    names = list(graphs)
    medians = {name: [] for name in names}
    for r in range(rounds):
        for name in names[r % len(names) :] + names[: r % len(names)]:
            graph, reps = graphs[name]
            stats = benchmark_cuda_graph_stats(graph, num_iters=iters, warmup_iters=5)
            medians[name].append(stats.median_us / reps)
    return {name: sorted(m)[len(m) // 2] for name, m in medians.items()}


def visit_inputs(weights, token, pos, cache, visits: set[int]) -> dict[int, tuple]:
    """``(layer, h, blocks, residual, mlp_blocks)`` of each of ``visits`` (plain visits) in a
    ``fast`` decode step at ``pos``; the step runs on a copy of ``cache``."""
    found = {}

    def spy(config, w, layer, visit, h, blocks, messages, cache, positions, start, kernels):
        residual, mlp_blocks = ref.attention_block(
            config, w, layer, visit, h, blocks, messages, cache, positions, start, kernels
        )
        if visit in visits:
            found[visit] = (layer, h.clone(), list(blocks), residual.clone(), list(mlp_blocks))
        return residual, mlp_blocks

    scratch = ref.InklingCache(**{k: v.clone() for k, v in vars(cache).items()})
    decode_step(weights, token, pos, scratch, dataclasses.replace(FAST, attention_block=spy))
    return found


@torch.no_grad()
def main(
    context: Annotated[int, typer.Option(help="prefilled tokens; the decoded position")] = 1100,
    max_len: Annotated[int, typer.Option(help="cache capacity (a multiple of 64)")] = 2048,
    visits: Annotated[str, typer.Option(help="plain visits (1: sliding, 5: full)")] = "1,5",
    reps: Annotated[int, typer.Option(help="calls per visit graph replay")] = 20,
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
    pos = torch.tensor([context], device="cuda")
    runner = InklingMegakernel(config, weights.layers)
    chosen = [int(v) for v in visits.split(",")]
    inputs = visit_inputs(weights, token, pos, cache, set(chosen))
    order = config.visits()

    graphs = {}
    for visit in chosen:
        layer, h, blocks, residual, mlp_blocks = inputs[visit]
        w = weights.layers[layer]
        kind = f"visit {visit} (layer {layer}, {'full' if config.is_full(layer) else 'sliding'})"
        assert order[visit][1] == layer
        res = torch.empty(config.hidden_size, device="cuda")

        def torch_attn(w=w, layer=layer, visit=visit, h=h, blocks=blocks):
            return FAST.attention_block(
                config, w, layer, visit, h, blocks, None, cache, pos, None, FAST
            )

        def torch_mlp(w=w, visit=visit, residual=residual, blocks=mlp_blocks):
            return FAST.mlp_block(config, w, visit, residual, blocks, None, cache, pos, FAST)

        def torch_visit(w=w, layer=layer, visit=visit, h=h, blocks=blocks):
            r, b = FAST.attention_block(
                config, w, layer, visit, h, blocks, None, cache, pos, None, FAST
            )
            return FAST.mlp_block(config, w, visit, r, b, None, cache, pos, FAST)

        def mk_attn(w=w, layer=layer, visit=visit, h=h, blocks=blocks, res=res):
            return runner.attn(w, layer, visit, h, blocks, cache, pos, res)

        def mk_mlp(w=w, visit=visit, residual=residual, blocks=mlp_blocks):
            return runner.mlp(w, residual, blocks, cache.conv_mlp[visit], pos)

        def mk_visit(w=w, layer=layer, visit=visit, h=h, blocks=blocks, mb=mlp_blocks, res=res):
            runner.attn(w, layer, visit, h, blocks, cache, pos, res)
            return runner.mlp(w, res, mb, cache.conv_mlp[visit], pos)

        for name, fn in [
            ("torch attention", torch_attn),
            ("megakernel attention", mk_attn),
            ("torch MLP", torch_mlp),
            ("megakernel MLP", mk_mlp),
            ("torch visit", torch_visit),
            ("megakernel visit", mk_visit),
        ]:
            graphs[f"{kind}: {name}"] = (capture(fn, reps), reps)

    steps = {
        "torch": FAST,
        "MLP-half hybrid": InklingMlpMegakernel(config, weights.layers).kernels(),
        "both-halves hybrid": runner.kernels(),
    }
    for name, kernels in steps.items():
        graphs[f"decode step: {name}"] = (
            capture(lambda k=kernels: decode_step(weights, token, pos, cache, k), 1),
            1,
        )
    results = time_graphs(graphs, rounds, iters)
    for name, us in results.items():
        print(f"{name:60s} {us:10.1f} us")
    if json_out is not None:
        here = Path(__file__).parent
        sources = {p.stem: p for p in sorted(here.glob("*.py"))}
        arguments = {"context": context, "max_len": max_len, "visits": chosen, "reps": reps,
                     "rounds": rounds, "iters": iters}  # fmt: skip
        json_out.write_text(
            json.dumps({"results_us": results, **provenance(sources, arguments)}, indent=2)
        )


if __name__ == "__main__":
    typer.run(main)
