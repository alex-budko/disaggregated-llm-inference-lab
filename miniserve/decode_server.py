"""Decode-only worker.

Receives a serialized KV cache + first-token logits from the prefill worker,
generates the rest of the sequence.

Run with:
    python -m miniserve.decode_server
"""

from __future__ import annotations

import base64
import io
import os
import threading

import torch
from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from . import metrics as M
from .engine import decode_from
from .model import DEFAULT_MODEL, load
from .schemas import DecodeRequest, DecodeResponse
from .serialization import decode_kv

DEVICE = os.environ.get("DEVICE", "cpu")
MODEL_NAME = os.environ.get("MODEL_NAME", DEFAULT_MODEL)

app = FastAPI(title="miniserve-decode")
_lock = threading.Lock()


def _decode_logits(b64: str) -> torch.Tensor:
    buf = io.BytesIO(base64.b64decode(b64))
    return torch.load(buf, map_location=DEVICE, weights_only=False)


@app.on_event("startup")
def _warm() -> None:
    load(MODEL_NAME, DEVICE)


@app.get("/healthz")
def health() -> dict:
    return {"status": "ok", "role": "decode", "model": MODEL_NAME}


@app.get("/metrics")
def metrics_endpoint() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/decode", response_model=DecodeResponse)
def decode_endpoint(req: DecodeRequest) -> DecodeResponse:
    model, tok = load(MODEL_NAME, DEVICE)
    past = decode_kv(req.kv_b64, device=DEVICE)
    first_logits = _decode_logits(req.first_token_logits_b64)

    with _lock:
        generated, decode_seconds = decode_from(
            model, tok,
            past_key_values=past,
            first_token_logits=first_logits,
            max_new_tokens=req.max_tokens,
            device=DEVICE,
        )
    text = tok.decode(generated, skip_special_tokens=True)
    n_decode_steps = max(len(generated) - 1, 0)
    tps = (n_decode_steps / decode_seconds) if decode_seconds > 0 and n_decode_steps > 0 else None

    M.REQUESTS.labels(endpoint="decode", status="ok").inc()
    M.DECODE_SECONDS.observe(decode_seconds)
    M.OUTPUT_TOKENS.observe(len(generated))
    if tps is not None:
        M.DECODE_TPS.observe(tps)

    return DecodeResponse(
        text=text,
        completion_tokens=len(generated),
        decode_seconds=decode_seconds,
        tokens_per_sec=tps,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "miniserve.decode_server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8002")),
        log_level="info",
    )
