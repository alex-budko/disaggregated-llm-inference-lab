"""KV-cache (de)serialization for the disaggregated prefill path.

Modern transformers returns a ``DynamicCache`` object from forward passes;
older versions return a tuple-of-tuples-of-tensors. We normalize to the
legacy tuple form for transport (pickle + base64) and rehydrate to a
``DynamicCache`` when re-entering the model.

Pickle-over-HTTP is fine for a teaching demo. Production disaggregated
systems (vLLM/MoonCake/NIXL) move KV blocks over NCCL / RDMA / shared
GPU memory because the JSON+pickle round-trip is the dominant cost here.
"""

from __future__ import annotations

import base64
import io
import pickle
from typing import Any

import torch


def _to_legacy(past: Any):
    if past is None:
        return None
    if hasattr(past, "to_legacy_cache"):
        return past.to_legacy_cache()
    return past  # already tuple-of-tuples


def _from_legacy(past_tuple: Any):
    if past_tuple is None:
        return None
    try:
        from transformers import DynamicCache

        return DynamicCache.from_legacy_cache(past_tuple)
    except Exception:
        return past_tuple


def encode_kv(past: Any) -> str:
    """Serialize past_key_values to a base64 string."""
    legacy = _to_legacy(past)
    if legacy is None:
        return ""
    # Move to CPU first so the receiver doesn't get a CUDA tensor it can't load.
    cpu_legacy = tuple(
        tuple(t.detach().to("cpu").contiguous() for t in layer) for layer in legacy
    )
    buf = io.BytesIO()
    torch.save(cpu_legacy, buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_kv(b64: str, device: str = "cpu"):
    """Deserialize a base64 string back into a Cache the model can consume."""
    if not b64:
        return None
    buf = io.BytesIO(base64.b64decode(b64))
    legacy = torch.load(buf, map_location=device, weights_only=False)
    return _from_legacy(legacy)


def kv_byte_size(b64: str) -> int:
    """Approximate serialized size in bytes (for metrics)."""
    return len(b64) * 3 // 4 if b64 else 0
