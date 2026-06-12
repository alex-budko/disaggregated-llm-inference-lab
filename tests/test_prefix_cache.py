"""Prefix cache: must be correct on hits, and visibly faster on hits."""

from __future__ import annotations

from miniserve.engine import generate
from miniserve.prefix_cache import PrefixCache


def test_full_hit_skips_prefill(model_and_tokenizer):
    model, tok = model_and_tokenizer
    cache = PrefixCache()
    prompt = "Explain why the sky appears blue:"

    cold = generate(model, tok, prompt, 16, use_kv_cache=True, prefix_cache=cache)
    warm = generate(model, tok, prompt, 16, use_kv_cache=True, prefix_cache=cache)

    assert warm.output_token_ids == cold.output_token_ids
    assert warm.cached_prefix_tokens == cold.prompt_tokens
    # Full hit means no model forward in prefill -> should be much faster.
    assert warm.prefill_seconds <= cold.prefill_seconds * 0.5 + 1e-3


def test_partial_hit_is_correct(model_and_tokenizer):
    """A request whose prompt extends a cached prompt must produce the same
    tokens as if there were no cache at all."""
    model, tok = model_and_tokenizer
    cache = PrefixCache()

    base = "The quick brown fox"
    extended = "The quick brown fox jumps over the lazy dog and then"

    # Prime cache with the shorter prompt.
    generate(model, tok, base, 8, use_kv_cache=True, prefix_cache=cache)

    with_cache = generate(model, tok, extended, 12, use_kv_cache=True, prefix_cache=cache)
    no_cache = generate(model, tok, extended, 12, use_kv_cache=True, prefix_cache=None)

    assert with_cache.output_token_ids == no_cache.output_token_ids
    assert with_cache.cached_prefix_tokens > 0


def test_miss_does_not_use_cache(model_and_tokenizer):
    model, tok = model_and_tokenizer
    cache = PrefixCache()
    generate(model, tok, "Apples are red.", 4, use_kv_cache=True, prefix_cache=cache)

    other = generate(model, tok, "Octopuses have eight", 6, use_kv_cache=True, prefix_cache=cache)
    assert other.cached_prefix_tokens == 0


def test_lru_eviction():
    cache = PrefixCache(max_entries=2)
    cache.put([1, 2, 3], past="a", last_logits="A")
    cache.put([4, 5, 6], past="b", last_logits="B")
    cache.put([7, 8, 9], past="c", last_logits="C")
    # [1,2,3] should be gone.
    assert cache.get([1, 2, 3]) == (0, None, None)
    matched_len, past, _ = cache.get([4, 5, 6])
    assert matched_len == 3 and past == "b"
