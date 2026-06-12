"""Disaggregated-serving gateway.

Exposes the same OpenAI-ish endpoint as the monolithic server but fans out
prefill to one worker and decode to another. The two workers can be on
different hosts / GPUs / pods entirely.

    Client
       v
    Gateway (this file)  --POST /v1/prefill-->  Prefill worker
                                          <--  KV cache + first logits
       |
       +----------POST /v1/decode (KV+logits)--> Decode worker
                                          <--   final text

Run with:
    python -m miniserve.gateway
"""

from __future__ import annotations

import os
import time

import httpx
from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from . import metrics as M
from .schemas import (
    DecodeRequest,
    GenerateRequest,
    GenerateResponse,
    PrefillRequest,
    PrefillResponse,
)

PREFILL_URL = os.environ.get("PREFILL_URL", "http://localhost:8001")
DECODE_URL = os.environ.get("DECODE_URL", "http://localhost:8002")
TIMEOUT = float(os.environ.get("HTTP_TIMEOUT", "120"))

app = FastAPI(title="miniserve-gateway")
_client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def _startup() -> None:
    global _client
    _client = httpx.AsyncClient(timeout=TIMEOUT)


@app.on_event("shutdown")
async def _shutdown() -> None:
    if _client is not None:
        await _client.aclose()


@app.get("/healthz")
async def health() -> dict:
    assert _client is not None
    p, d = await _check(PREFILL_URL), await _check(DECODE_URL)
    return {"status": "ok", "prefill": p, "decode": d}


async def _check(url: str) -> str:
    try:
        assert _client is not None
        r = await _client.get(f"{url}/healthz")
        return "ok" if r.status_code == 200 else f"http {r.status_code}"
    except Exception as exc:
        return f"unreachable: {exc!s}"


@app.get("/metrics")
def metrics_endpoint() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/generate", response_model=GenerateResponse)
async def generate_endpoint(req: GenerateRequest) -> GenerateResponse:
    assert _client is not None

    t0 = time.perf_counter()
    pr = await _client.post(
        f"{PREFILL_URL}/v1/prefill",
        json=PrefillRequest(prompt=req.prompt, use_prefix_cache=req.use_prefix_cache).model_dump(),
    )
    pr.raise_for_status()
    pres = PrefillResponse(**pr.json())
    ttft = time.perf_counter() - t0  # client-perceived TTFT

    dr = await _client.post(
        f"{DECODE_URL}/v1/decode",
        json=DecodeRequest(
            kv_b64=pres.kv_b64,
            first_token_logits_b64=pres.first_token_logits_b64,
            max_tokens=req.max_tokens,
        ).model_dump(),
    )
    dr.raise_for_status()
    dres = dr.json()
    total = time.perf_counter() - t0

    M.REQUESTS.labels(endpoint="gateway_generate", status="ok").inc()
    M.TTFT_SECONDS.observe(ttft)

    return GenerateResponse(
        text=dres["text"],
        prompt_tokens=pres.prompt_tokens,
        completion_tokens=dres["completion_tokens"],
        prefill_seconds=pres.prefill_seconds,
        decode_seconds=dres["decode_seconds"],
        ttft_seconds=ttft,
        total_seconds=total,
        tokens_per_sec=dres.get("tokens_per_sec"),
        cached_prefix_tokens=pres.cached_prefix_tokens,
        served_by=f"disagg (prefill={PREFILL_URL}, decode={DECODE_URL})",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "miniserve.gateway:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        log_level="info",
    )
