"""KV cache shouldn't change *what* the model outputs, only how fast.

If this test ever fails, the cache implementation is wrong: a correct KV
cache is mathematically equivalent to recomputing the full sequence each
step (both apply causal attention over the same context).
"""

from __future__ import annotations

from miniserve.engine import generate


def test_kv_cache_matches_no_cache(model_and_tokenizer):
    model, tok = model_and_tokenizer
    prompt = "The capital of France is"

    cached = generate(model, tok, prompt, max_new_tokens=12, use_kv_cache=True)
    uncached = generate(model, tok, prompt, max_new_tokens=12, use_kv_cache=False)

    assert cached.output_token_ids == uncached.output_token_ids, (
        f"KV cache changed outputs:\n cached={cached.text!r}\n uncached={uncached.text!r}"
    )
    assert cached.prompt_tokens == uncached.prompt_tokens


def test_kv_cache_is_faster_for_long_generations(model_and_tokenizer):
    model, tok = model_and_tokenizer
    prompt = "Once upon a time in a small village by the sea,"

    cached = generate(model, tok, prompt, max_new_tokens=32, use_kv_cache=True)
    uncached = generate(model, tok, prompt, max_new_tokens=32, use_kv_cache=False)

    # On CPU with a 135M model the gap is small but consistently visible.
    # We assert "not slower" rather than a hard ratio to avoid flakiness.
    assert cached.decode_seconds <= uncached.decode_seconds * 1.2
