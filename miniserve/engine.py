"""The actual generation loop.

This is the file to read if you want to understand the concepts. Three knobs:

* ``use_kv_cache``: when False, each new token re-runs the model on the full
  growing sequence (O(N^2) total work for N output tokens). When True, the
  model keeps ``past_key_values`` and we only feed one new token per step.

* ``prefix_cache``: optional in-memory cache keyed by prompt token IDs. On a
  hit we reuse the cached ``past_key_values`` for the matched prefix and
  only prefill the suffix. On a *full* hit (entire prompt already cached
  along with its final logits) we skip prefill entirely.

* ``prefilled_past`` / ``prefilled_logits``: when set, we skip prefill and
  jump straight to decoding from the supplied state. The disaggregated
  ``decode_server`` uses this to consume KV cache produced by a separate
  ``prefill_server``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from .prefix_cache import PrefixCache


@dataclass
class GenerationResult:
    text: str
    output_token_ids: list[int]
    prompt_tokens: int
    completion_tokens: int
    prefill_seconds: float
    decode_seconds: float
    ttft_seconds: float
    total_seconds: float
    tokens_per_sec: Optional[float]
    cached_prefix_tokens: int
    # Carry the post-prefill state out so the disagg pipeline can hand it
    # to a separate decode worker.
    past_key_values: Any = field(default=None, repr=False)
    first_token_logits: Any = field(default=None, repr=False)


@torch.no_grad()
def prefill(
    model,
    tokenizer,
    prompt: str,
    *,
    use_kv_cache: bool = True,
    prefix_cache: Optional[PrefixCache] = None,
    device: str = "cpu",
) -> GenerationResult:
    """Run only the prompt forward pass. Returns the state needed for decode.

    Used by the disaggregated prefill worker. Sets ``past_key_values`` and
    ``first_token_logits`` on the result; ``output_token_ids`` is empty.
    """
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    seq = ids[0].tolist()
    L = len(seq)

    cached_len, past, cached_logits = 0, None, None
    if prefix_cache is not None and use_kv_cache:
        cached_len, past, cached_logits = prefix_cache.get(seq)

    t0 = time.perf_counter()
    if cached_len < L:
        suffix = ids[:, cached_len:L]
        out = model(suffix, past_key_values=past, use_cache=use_kv_cache)
        past = out.past_key_values if use_kv_cache else None
        first_logits = out.logits[:, -1, :]
        if prefix_cache is not None and use_kv_cache:
            prefix_cache.put(seq, past, first_logits.detach().clone())
    else:
        # Full hit: cached_logits already predicts the next token.
        first_logits = cached_logits
    prefill_time = time.perf_counter() - t0

    return GenerationResult(
        text="",
        output_token_ids=[],
        prompt_tokens=L,
        completion_tokens=0,
        prefill_seconds=prefill_time,
        decode_seconds=0.0,
        ttft_seconds=prefill_time,
        total_seconds=prefill_time,
        tokens_per_sec=None,
        cached_prefix_tokens=cached_len,
        past_key_values=past,
        first_token_logits=first_logits,
    )


@torch.no_grad()
def decode_from(
    model,
    tokenizer,
    *,
    past_key_values: Any,
    first_token_logits: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
    device: str = "cpu",
) -> tuple[list[int], float]:
    """Greedy decode starting from a pre-computed prefill state.

    Returns (generated_token_ids, decode_seconds).
    """
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    t0 = time.perf_counter()
    next_token = first_token_logits.argmax(dim=-1, keepdim=True).to(device)
    generated: list[int] = [int(next_token.item())]

    past = past_key_values
    for _ in range(max_new_tokens - 1):
        if generated[-1] == eos_token_id:
            break
        out = model(next_token, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(int(next_token.item()))

    return generated, time.perf_counter() - t0


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    *,
    use_kv_cache: bool = True,
    prefix_cache: Optional[PrefixCache] = None,
    device: str = "cpu",
) -> GenerationResult:
    """End-to-end generate, with KV-cache and prefix-cache toggles."""
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    seq = ids[0].tolist()
    L = len(seq)

    # ------------------------------------------------------------------- prefill
    cached_len, past, cached_logits = 0, None, None
    if prefix_cache is not None and use_kv_cache:
        cached_len, past, cached_logits = prefix_cache.get(seq)

    t_start = time.perf_counter()
    if cached_len < L:
        suffix = ids[:, cached_len:L]
        out = model(suffix, past_key_values=past, use_cache=use_kv_cache)
        past = out.past_key_values if use_kv_cache else None
        next_logits = out.logits[:, -1, :]
        if prefix_cache is not None and use_kv_cache:
            prefix_cache.put(seq, past, next_logits.detach().clone())
    else:
        next_logits = cached_logits
    prefill_time = time.perf_counter() - t_start

    next_token = next_logits.argmax(dim=-1, keepdim=True).to(device)
    generated: list[int] = [int(next_token.item())]
    ttft = time.perf_counter() - t_start

    # -------------------------------------------------------------------- decode
    eos = tokenizer.eos_token_id
    t_decode_start = time.perf_counter()
    for _ in range(max_new_tokens - 1):
        if generated[-1] == eos:
            break
        if use_kv_cache:
            out = model(next_token, past_key_values=past, use_cache=True)
            past = out.past_key_values
        else:
            # The whole point of disabling KV cache: re-encode the full
            # growing context every step. This is the slow baseline.
            full = torch.tensor([seq + generated], device=device)
            out = model(full, use_cache=False)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(int(next_token.item()))
    decode_time = time.perf_counter() - t_decode_start
    total = time.perf_counter() - t_start

    text = tokenizer.decode(generated, skip_special_tokens=True)
    n_decode_steps = max(len(generated) - 1, 0)
    tps = (n_decode_steps / decode_time) if decode_time > 0 and n_decode_steps > 0 else None

    return GenerationResult(
        text=text,
        output_token_ids=generated,
        prompt_tokens=L,
        completion_tokens=len(generated),
        prefill_seconds=prefill_time,
        decode_seconds=decode_time,
        ttft_seconds=ttft,
        total_seconds=total,
        tokens_per_sec=tps,
        cached_prefix_tokens=cached_len,
        past_key_values=past,
        first_token_logits=None,
    )
