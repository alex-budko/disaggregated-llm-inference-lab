"""Benchmark a vLLM OpenAI-compatible server.

Sends N chat-completion requests (optionally concurrent), measures TTFT,
total latency, input/output token counts, and tokens-per-second, then
writes a CSV row per request.
"""

import argparse
import asyncio
import csv
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Optional

from openai import AsyncOpenAI


@dataclass
class RequestResult:
    request_id: int
    status: str
    ttft_s: Optional[float]
    total_latency_s: Optional[float]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    tokens_per_sec: Optional[float]
    error: str = ""


DEFAULT_PROMPT = (
    "Write a detailed technical explanation of how transformer attention works, "
    "including the role of queries, keys, and values."
)


async def run_one(
    client: AsyncOpenAI,
    request_id: int,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> RequestResult:
    start = time.perf_counter()
    ttft: Optional[float] = None
    output_tokens = 0
    input_tokens: Optional[int] = None
    reported_output_tokens: Optional[int] = None

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
        )

        async for chunk in stream:
            if chunk.usage is not None:
                input_tokens = chunk.usage.prompt_tokens
                reported_output_tokens = chunk.usage.completion_tokens

            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                if ttft is None:
                    ttft = time.perf_counter() - start
                output_tokens += 1

        total_latency = time.perf_counter() - start
        final_output_tokens = reported_output_tokens if reported_output_tokens is not None else output_tokens
        generation_time = total_latency - (ttft or 0.0)
        tps: Optional[float] = None
        if final_output_tokens and generation_time > 0:
            decode_tokens = max(final_output_tokens - 1, 0)
            tps = decode_tokens / generation_time if decode_tokens > 0 else None

        return RequestResult(
            request_id=request_id,
            status="ok",
            ttft_s=ttft,
            total_latency_s=total_latency,
            input_tokens=input_tokens,
            output_tokens=final_output_tokens,
            tokens_per_sec=tps,
        )
    except Exception as exc:
        return RequestResult(
            request_id=request_id,
            status="error",
            ttft_s=None,
            total_latency_s=time.perf_counter() - start,
            input_tokens=None,
            output_tokens=None,
            tokens_per_sec=None,
            error=f"{type(exc).__name__}: {exc}",
        )


async def run_all(args: argparse.Namespace) -> list[RequestResult]:
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def guarded(i: int) -> RequestResult:
        async with semaphore:
            return await run_one(
                client=client,
                request_id=i,
                model=args.model,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )

    tasks = [asyncio.create_task(guarded(i)) for i in range(args.num_requests)]
    results: list[RequestResult] = []
    for coro in asyncio.as_completed(tasks):
        res = await coro
        results.append(res)
        marker = "ok " if res.status == "ok" else "ERR"
        print(
            f"[{marker}] req={res.request_id:<4} "
            f"ttft={res.ttft_s if res.ttft_s is not None else float('nan'):.3f}s "
            f"latency={res.total_latency_s if res.total_latency_s is not None else float('nan'):.3f}s "
            f"in={res.input_tokens} out={res.output_tokens} "
            f"tps={res.tokens_per_sec if res.tokens_per_sec is not None else float('nan'):.2f}"
            + (f" err={res.error}" if res.error else ""),
            flush=True,
        )

    results.sort(key=lambda r: r.request_id)
    return results


def write_csv(path: str, results: list[RequestResult]) -> None:
    fieldnames = [
        "request_id",
        "status",
        "ttft_s",
        "total_latency_s",
        "input_tokens",
        "output_tokens",
        "tokens_per_sec",
        "error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def print_summary(results: list[RequestResult], wall_time_s: float) -> None:
    ok = [r for r in results if r.status == "ok"]
    failed = len(results) - len(ok)
    print()
    print(f"=== Summary ===")
    print(f"Requests:        {len(results)}  (ok={len(ok)}, failed={failed})")
    print(f"Wall time:       {wall_time_s:.2f}s")
    if not ok:
        return

    def stat(name: str, values: list[float], unit: str = "") -> None:
        if not values:
            return
        print(
            f"{name:<17}"
            f"mean={statistics.mean(values):.3f}{unit}  "
            f"p50={statistics.median(values):.3f}{unit}  "
            f"p95={_percentile(values, 95):.3f}{unit}  "
            f"max={max(values):.3f}{unit}"
        )

    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    latencies = [r.total_latency_s for r in ok if r.total_latency_s is not None]
    tpss = [r.tokens_per_sec for r in ok if r.tokens_per_sec is not None]
    total_out = sum((r.output_tokens or 0) for r in ok)

    stat("TTFT:", ttfts, "s")
    stat("Latency:", latencies, "s")
    stat("Decode TPS:", tpss, " tok/s")
    print(f"Total out tokens: {total_out}")
    print(f"Aggregate TPS:    {total_out / wall_time_s:.2f} tok/s (output tokens / wall time)")


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark a vLLM OpenAI-compatible server.")
    p.add_argument("--base-url", default="http://localhost:8000/v1",
                   help="vLLM server base URL (default: %(default)s)")
    p.add_argument("--api-key", default="EMPTY",
                   help="API key (vLLM ignores by default; default: %(default)s)")
    p.add_argument("--model", required=True, help="Model name as registered on the server")
    p.add_argument("-n", "--num-requests", type=int, default=20, help="Total requests to send")
    p.add_argument("-c", "--concurrency", type=int, default=1, help="Max in-flight requests")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt text for every request")
    p.add_argument("--prompt-file", help="If set, read prompt text from this file")
    p.add_argument("--max-tokens", type=int, default=256, help="Max output tokens per request")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("-o", "--output", default="results.csv", help="CSV output path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            args.prompt = f.read()

    print(
        f"Benchmarking {args.model} at {args.base_url} "
        f"(n={args.num_requests}, concurrency={args.concurrency}, max_tokens={args.max_tokens})"
    )

    start = time.perf_counter()
    results = asyncio.run(run_all(args))
    wall = time.perf_counter() - start

    write_csv(args.output, results)
    print_summary(results, wall)
    print(f"\nWrote {len(results)} rows to {args.output}")
    return 0 if all(r.status == "ok" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
