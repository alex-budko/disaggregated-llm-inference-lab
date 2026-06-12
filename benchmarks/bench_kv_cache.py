"""Demonstrate the cost of NOT having a KV cache.

Without a KV cache, every new token re-encodes the full growing sequence,
so total decode work grows roughly quadratically in output length. With
a KV cache each step costs roughly constant attention over the (cached)
context. We sweep `max_new_tokens` and plot the two curves.

Run:
    python -m benchmarks.bench_kv_cache --model HuggingFaceTB/SmolLM2-135M-Instruct
"""

from __future__ import annotations

import argparse
import csv
import os

from miniserve.engine import generate
from miniserve.model import DEFAULT_MODEL, load

from ._plot import save_line_plot


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-csv", default="results/bench_kv_cache.csv")
    p.add_argument("--out-png", default="results/bench_kv_cache.png")
    p.add_argument(
        "--token-counts",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128],
        help="Output-token settings to sweep.",
    )
    p.add_argument(
        "--prompt",
        default="In a calm and methodical tone, explain how an internal combustion engine works:",
    )
    args = p.parse_args()

    model, tok = load(args.model, args.device)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    rows = []
    cached_curve, uncached_curve = [], []

    print(f"model={args.model} device={args.device}")
    print(f"{'N_out':>6}  {'cached_s':>10}  {'uncached_s':>12}  {'speedup':>8}")
    for n in args.token_counts:
        c = generate(model, tok, args.prompt, n, use_kv_cache=True, device=args.device)
        u = generate(model, tok, args.prompt, n, use_kv_cache=False, device=args.device)
        speedup = u.total_seconds / c.total_seconds if c.total_seconds > 0 else float("nan")
        print(f"{n:>6}  {c.total_seconds:>10.3f}  {u.total_seconds:>12.3f}  {speedup:>8.2f}x")
        rows.append({
            "max_new_tokens": n,
            "cached_total_s": c.total_seconds,
            "uncached_total_s": u.total_seconds,
            "cached_decode_s": c.decode_seconds,
            "uncached_decode_s": u.decode_seconds,
            "cached_tps": c.tokens_per_sec,
            "uncached_tps": u.tokens_per_sec,
            "speedup": speedup,
        })
        cached_curve.append(c.total_seconds)
        uncached_curve.append(u.total_seconds)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    save_line_plot(
        xs=args.token_counts,
        series={"with KV cache": cached_curve, "without KV cache": uncached_curve},
        xlabel="Output tokens",
        ylabel="Total latency (s)",
        title=f"KV cache vs no cache  ({args.model})",
        out_path=args.out_png,
    )
    print(f"\nWrote {args.out_csv} and {args.out_png}")


if __name__ == "__main__":
    main()
