"""Prometheus metrics shared by all servers."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter(
    "miniserve_requests_total",
    "Total requests handled",
    labelnames=("endpoint", "status"),
)

PREFILL_SECONDS = Histogram(
    "miniserve_prefill_seconds",
    "Time spent on the prompt forward pass",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)

DECODE_SECONDS = Histogram(
    "miniserve_decode_seconds",
    "Time spent generating output tokens",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)

TTFT_SECONDS = Histogram(
    "miniserve_ttft_seconds",
    "Time to first token (from request receipt to first sampled token)",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)

OUTPUT_TOKENS = Histogram(
    "miniserve_output_tokens",
    "Output tokens per request",
    buckets=(1, 8, 32, 64, 128, 256, 512, 1024),
)

DECODE_TPS = Histogram(
    "miniserve_decode_tokens_per_sec",
    "Sustained decode tokens/sec per request",
    buckets=(1, 2, 5, 10, 20, 50, 100, 200),
)

PREFIX_CACHE_HITS = Counter("miniserve_prefix_cache_hits_total", "Prefix cache hits")
PREFIX_CACHE_MISSES = Counter("miniserve_prefix_cache_misses_total", "Prefix cache misses")
PREFIX_CACHE_ENTRIES = Gauge("miniserve_prefix_cache_entries", "Entries currently in prefix cache")
PREFIX_CACHED_TOKENS = Histogram(
    "miniserve_prefix_cached_tokens",
    "Number of prompt tokens served from prefix cache per request",
    buckets=(0, 16, 64, 128, 256, 512, 1024, 2048),
)
