"""Prefill-only worker. Owns the prefix cache.

Runs the prompt forward pass, ships the resulting KV state (plus the
first-token logits) to whoever called us. The gateway then hands that
state to a decode worker.

Run with:
    python -m miniserve.prefill_server
"""

from __future__ import annotations

import io
import os
import threading

import torch
from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from . import metrics as M
from .engine import prefill
from .model import DEFAULT_MODEL, load
from .prefix_cache import PrefixCache
from .schemas import PrefillRequest, PrefillResponse
from .serialization import encode_kv, kv_byte_size

DEVICE = os.environ.get("DEVICE", "cpu")
MODEL_NAME = os.environ.get("MODEL_NAME", DEFAULT_MODEL)

app = FastAPI(title="miniserve-prefill")
_lock = threading.Lock()
_prefix_cache = PrefixCache(max_entries=int(os.environ.get("PREFIX_CACHE_SIZE", "64")))


def _encode_logits(t: torch.Tensor) -> str:
    import base64
    buf = io.BytesIO()
    torch.save(t.detach().to("cpu"), buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.on_event("startup")
def _warm() -> None:
    load(MODEL_NAME, DEVICE)


@app.get("/healthz")
def health() -> dict:
    return {"status": "ok", "role": "prefill", "model": MODEL_NAME}


@app.get("/metrics")
def metrics_endpoint() -> Response:
    M.PREFIX_CACHE_ENTRIES.set(len(_prefix_cache))
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/prefill", response_model=PrefillResponse)
def prefill_endpoint(req: PrefillRequest) -> PrefillResponse:
    model, tok = load(MODEL_NAME, DEVICE)
    cache = _prefix_cache if req.use_prefix_cache else None
    with _lock:
        res = prefill(
            model, tok, req.prompt,
            use_kv_cache=True, prefix_cache=cache, device=DEVICE,
        )

    kv_b64 = encode_kv(res.past_key_values)
    logits_b64 = _encode_logits(res.first_token_logits)

    M.REQUESTS.labels(endpoint="prefill", status="ok").inc()
    M.PREFILL_SECONDS.observe(res.prefill_seconds)
    if cache is not None:
        if res.cached_prefix_tokens > 0:
            M.PREFIX_CACHE_HITS.inc()
        else:
            M.PREFIX_CACHE_MISSES.inc()
        M.PREFIX_CACHED_TOKENS.observe(res.cached_prefix_tokens)

    return PrefillResponse(
        kv_b64=kv_b64,
        first_token_logits_b64=logits_b64,
        prompt_tokens=res.prompt_tokens,
        cached_prefix_tokens=res.cached_prefix_tokens,
        prefill_seconds=res.prefill_seconds,
        kv_bytes=kv_byte_size(kv_b64),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "miniserve.prefill_server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8001")),
        log_level="info",
    )
