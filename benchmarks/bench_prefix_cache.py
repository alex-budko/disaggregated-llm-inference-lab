"""Demonstrate prefix-cache speedup on repeated long-context workloads.

We construct a long shared "system prompt" + many short user queries, then
run the workload twice through the same `PrefixCache`. On the second pass
every request hits the cached prefix and skips almost all prefill work.

Run:
    python -m benchmarks.bench_prefix_cache --model HuggingFaceTB/SmolLM2-135M-Instruct
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time

from miniserve.engine import generate
from miniserve.model import DEFAULT_MODEL, load
from miniserve.prefix_cache import PrefixCache

from ._plot import save_box_plot

LONG_SYSTEM = (
    "You are a careful technical assistant. " * 80
)  # ~640 tokens of repetitive instructions; cheap stand-in for a long system prompt.

USER_QUERIES = [
    "Briefly: what is a B-tree?",
    "Briefly: what is consistent hashing?",
    "Briefly: what is a Bloom filter?",
    "Briefly: what is a write-ahead log?",
    "Briefly: what is a circuit breaker?",
    "Briefly: what is a vector clock?",
    "Briefly: what is exponential backoff?",
    "Briefly: what is two-phase commit?",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-tokens", type=int, default=24)
    p.add_argument("--out-csv", default="results/bench_prefix_cache.csv")
    p.add_argument("--out-png", default="results/bench_prefix_cache.png")
    args = p.parse_args()

    model, tok = load(args.model, args.device)
    cache = PrefixCache(max_entries=64)

    rows = []
    cold_prefill, warm_prefill = [], []
    cold_ttft, warm_ttft = [], []

    print(f"model={args.model} device={args.device}")
    print(f"{'pass':>6}  {'i':>3}  {'prefill_s':>10}  {'ttft_s':>8}  {'cached':>7}")
    for pass_name in ("cold", "warm"):
        for i, q in enumerate(USER_QUERIES):
            prompt = LONG_SYSTEM + "\n\nQ: " + q + "\nA:"
            r = generate(
                model, tok, prompt, args.max_tokens,
                use_kv_cache=True, prefix_cache=cache, device=args.device,
            )
            print(f"{pass_name:>6}  {i:>3}  {r.prefill_seconds:>10.3f}  {r.ttft_seconds:>8.3f}  {r.cached_prefix_tokens:>7}")
            rows.append({
                "pass": pass_name,
                "i": i,
                "prefill_s": r.prefill_seconds,
                "ttft_s": r.ttft_seconds,
                "total_s": r.total_seconds,
                "prompt_tokens": r.prompt_tokens,
                "cached_prefix_tokens": r.cached_prefix_tokens,
            })
            if pass_name == "cold":
                cold_prefill.append(r.prefill_seconds)
                cold_ttft.append(r.ttft_seconds)
            else:
                warm_prefill.append(r.prefill_seconds)
                warm_ttft.append(r.ttft_seconds)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    save_box_plot(
        {"cold prefill": cold_prefill, "warm prefill (hit)": warm_prefill},
        ylabel="Prefill time (s)",
        title=f"Prefix cache: prefill cost  ({args.model})",
        out_path=args.out_png,
    )

    def mean(xs):
        return statistics.mean(xs) if xs else float("nan")

    print()
    print(f"Mean cold prefill: {mean(cold_prefill):.3f}s")
    print(f"Mean warm prefill: {mean(warm_prefill):.3f}s")
    if mean(cold_prefill) > 0:
        print(f"Speedup:           {mean(cold_prefill) / max(mean(warm_prefill), 1e-9):.1f}x")
    print(f"Cache stats:       {cache.stats()}")
    print(f"Wrote {args.out_csv} and {args.out_png}")


if __name__ == "__main__":
    main()
