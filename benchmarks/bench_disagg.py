"""Compare monolithic vs disaggregated serving under concurrent load.

The point of disagg prefill is that a heavy prefill on one request doesn't
block decoding of others, because prefill and decode run on different
workers. We send N concurrent requests to two endpoints and compare the
TTFT distribution.

This benchmark hits live servers — start them in two terminals first.

    # baseline: one monolithic server on :8000
    python -m miniserve.monolithic_server

    # disagg: three processes
    python -m miniserve.prefill_server   # :8001
    python -m miniserve.decode_server    # :8002
    python -m miniserve.gateway          # :8000  (set MONO_URL elsewhere)

Then run:
    python -m benchmarks.bench_disagg \\
        --mono-url http://localhost:8000 \\
        --disagg-url http://localhost:8000 -n 16 -c 8
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import statistics
import time
from dataclasses import dataclass

import httpx

from ._plot import save_box_plot


LONG_PROMPT = (
    "You are a careful technical assistant. " * 80
    + "\n\nQuestion: explain consistent hashing in two sentences.\nAnswer:"
)


@dataclass
class Sample:
    topology: str
    request_id: int
    ttft_s: float
    total_s: float
    prefill_s: float
    decode_s: float
    output_tokens: int


async def _one(client: httpx.AsyncClient, url: str, topology: str, i: int, prompt: str, max_tokens: int) -> Sample:
    t0 = time.perf_counter()
    r = await client.post(
        f"{url}/v1/generate",
        json={"prompt": prompt, "max_tokens": max_tokens, "use_prefix_cache": True},
    )
    r.raise_for_status()
    body = r.json()
    return Sample(
        topology=topology,
        request_id=i,
        ttft_s=body.get("ttft_seconds", time.perf_counter() - t0),
        total_s=body.get("total_seconds", time.perf_counter() - t0),
        prefill_s=body.get("prefill_seconds", 0.0),
        decode_s=body.get("decode_seconds", 0.0),
        output_tokens=body.get("completion_tokens", 0),
    )


async def _run(url: str, topology: str, n: int, concurrency: int, prompt: str, max_tokens: int) -> list[Sample]:
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=300) as client:
        async def task(i):
            async with sem:
                return await _one(client, url, topology, i, prompt, max_tokens)
        return await asyncio.gather(*(task(i) for i in range(n)))


def _summarize(name: str, samples: list[Sample]) -> None:
    ttfts = [s.ttft_s for s in samples]
    totals = [s.total_s for s in samples]
    print(f"\n=== {name} ===")
    print(f"  TTFT   mean={statistics.mean(ttfts):.3f}s  p50={statistics.median(ttfts):.3f}s  "
          f"p95={_p(ttfts, 95):.3f}s  max={max(ttfts):.3f}s")
    print(f"  Total  mean={statistics.mean(totals):.3f}s  p50={statistics.median(totals):.3f}s  "
          f"p95={_p(totals, 95):.3f}s")


def _p(xs: list[float], p: float) -> float:
    s = sorted(xs)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mono-url", required=True, help="Base URL of the monolithic server")
    ap.add_argument("--disagg-url", required=True, help="Base URL of the disagg gateway")
    ap.add_argument("-n", "--num-requests", type=int, default=16)
    ap.add_argument("-c", "--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--out-csv", default="results/bench_disagg.csv")
    ap.add_argument("--out-png", default="results/bench_disagg.png")
    args = ap.parse_args()

    print(f"Sending {args.num_requests} requests at concurrency={args.concurrency} to each topology...")
    mono = asyncio.run(_run(args.mono_url, "monolithic", args.num_requests, args.concurrency, LONG_PROMPT, args.max_tokens))
    disagg = asyncio.run(_run(args.disagg_url, "disagg", args.num_requests, args.concurrency, LONG_PROMPT, args.max_tokens))

    _summarize("monolithic", mono)
    _summarize("disagg", disagg)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["topology", "request_id", "ttft_s", "total_s", "prefill_s", "decode_s", "output_tokens"])
        for s in mono + disagg:
            w.writerow([s.topology, s.request_id, s.ttft_s, s.total_s, s.prefill_s, s.decode_s, s.output_tokens])

    save_box_plot(
        {"monolithic": [s.ttft_s for s in mono], "disagg": [s.ttft_s for s in disagg]},
        ylabel="TTFT (s)",
        title=f"TTFT under concurrency={args.concurrency}",
        out_path=args.out_png,
    )
    print(f"\nWrote {args.out_csv} and {args.out_png}")


if __name__ == "__main__":
    main()
