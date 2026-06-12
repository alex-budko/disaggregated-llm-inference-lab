"""Monolithic server: prefill + decode in one process, with prefix cache.

This is the baseline you'd compare against the disaggregated topology.
Run with:

    python -m miniserve.monolithic_server
"""

from __future__ import annotations

import os
import threading
import time

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from . import metrics as M
from .engine import generate
from .model import DEFAULT_MODEL, load
from .prefix_cache import PrefixCache
from .schemas import GenerateRequest, GenerateResponse

DEVICE = os.environ.get("DEVICE", "cpu")
MODEL_NAME = os.environ.get("MODEL_NAME", DEFAULT_MODEL)

app = FastAPI(title="miniserve-monolithic")

# Lock around model use: a single torch model isn't safe to call from
# multiple threads concurrently. The disaggregated topology gets around
# this by running prefill and decode in separate processes.
_model_lock = threading.Lock()
_prefix_cache = PrefixCache(max_entries=int(os.environ.get("PREFIX_CACHE_SIZE", "64")))


@app.on_event("startup")
def _warm() -> None:
    load(MODEL_NAME, DEVICE)


@app.get("/healthz")
def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE}


@app.get("/metrics")
def metrics_endpoint() -> Response:
    M.PREFIX_CACHE_ENTRIES.set(len(_prefix_cache))
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/generate", response_model=GenerateResponse)
def generate_endpoint(req: GenerateRequest) -> GenerateResponse:
    model, tok = load(MODEL_NAME, DEVICE)
    cache = _prefix_cache if req.use_prefix_cache else None

    t0 = time.perf_counter()
    with _model_lock:
        result = generate(
            model, tok, req.prompt, req.max_tokens,
            use_kv_cache=True, prefix_cache=cache, device=DEVICE,
        )
    _ = time.perf_counter() - t0

    M.REQUESTS.labels(endpoint="generate", status="ok").inc()
    M.PREFILL_SECONDS.observe(result.prefill_seconds)
    M.DECODE_SECONDS.observe(result.decode_seconds)
    M.TTFT_SECONDS.observe(result.ttft_seconds)
    M.OUTPUT_TOKENS.observe(result.completion_tokens)
    if result.tokens_per_sec is not None:
        M.DECODE_TPS.observe(result.tokens_per_sec)
    if cache is not None:
        if result.cached_prefix_tokens > 0:
            M.PREFIX_CACHE_HITS.inc()
        else:
            M.PREFIX_CACHE_MISSES.inc()
        M.PREFIX_CACHED_TOKENS.observe(result.cached_prefix_tokens)

    return GenerateResponse(
        text=result.text,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        prefill_seconds=result.prefill_seconds,
        decode_seconds=result.decode_seconds,
        ttft_seconds=result.ttft_seconds,
        total_seconds=result.total_seconds,
        tokens_per_sec=result.tokens_per_sec,
        cached_prefix_tokens=result.cached_prefix_tokens,
        served_by="monolithic",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "miniserve.monolithic_server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        log_level="info",
    )
