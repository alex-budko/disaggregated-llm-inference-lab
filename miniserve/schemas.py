"""Pydantic schemas shared by the servers."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """OpenAI-ish chat-completion subset, simplified."""
    prompt: str
    max_tokens: int = 64
    use_prefix_cache: bool = True


class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    prefill_seconds: float
    decode_seconds: float
    ttft_seconds: float
    total_seconds: float
    tokens_per_sec: Optional[float] = None
    cached_prefix_tokens: int = 0
    served_by: str = ""


# --- Disagg-internal protocol ---------------------------------------------

class PrefillRequest(BaseModel):
    prompt: str
    use_prefix_cache: bool = True


class PrefillResponse(BaseModel):
    kv_b64: str = Field(..., description="Serialized past_key_values (legacy tuple, torch.save'd, base64).")
    first_token_logits_b64: str
    prompt_tokens: int
    cached_prefix_tokens: int
    prefill_seconds: float
    kv_bytes: int


class DecodeRequest(BaseModel):
    kv_b64: str
    first_token_logits_b64: str
    max_tokens: int = 64


class DecodeResponse(BaseModel):
    text: str
    completion_tokens: int
    decode_seconds: float
    tokens_per_sec: Optional[float] = None
