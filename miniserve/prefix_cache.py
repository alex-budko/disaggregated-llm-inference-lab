"""In-memory prefix cache.

On a hit we skip prefill for the matched prefix. We do exact longest-prefix
match against stored entries (linear scan). vLLM/SGLang use a radix tree
chunked at block granularity; this is the same idea, simpler.

Stored: token_id tuple -> (past_key_values_snapshot, last_logits). Caching
the final logits lets a *full* hit return the first sampled token without
any forward pass at all.

A snapshot is an immutable tuple-of-tuples-of-cloned-tensors. We must
snapshot because ``DynamicCache`` is mutable: the very next decode step
extends ``past`` in place, and without a snapshot it would corrupt the
entry we just stored.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Optional, Tuple

import torch


def _looks_like_legacy_past(x: Any) -> bool:
    return isinstance(x, tuple) and len(x) > 0 and isinstance(x[0], tuple)


def _snapshot_past(past: Any):
    """Convert past_key_values into an immutable, decoupled form.

    For real Caches / legacy tuples we clone every tensor so subsequent
    decode mutations can't reach the cache entry. For anything else
    (e.g. sentinel values in unit tests) we pass through unchanged.
    """
    if past is None:
        return None
    if hasattr(past, "to_legacy_cache"):
        past = past.to_legacy_cache()
    if not _looks_like_legacy_past(past):
        return past
    return tuple(
        tuple(t.detach().clone() if isinstance(t, torch.Tensor) else t for t in layer)
        for layer in past
    )


def _rebuild_past(snapshot: Any):
    """Rebuild a fresh DynamicCache from a snapshot so the caller can mutate
    it freely without touching our stored entry."""
    if snapshot is None or not _looks_like_legacy_past(snapshot):
        return snapshot
    try:
        from transformers import DynamicCache

        return DynamicCache.from_legacy_cache(snapshot)
    except Exception:
        return snapshot


class PrefixCache:
    def __init__(self, max_entries: int = 32):
        self._entries: "OrderedDict[Tuple[int, ...], dict]" = OrderedDict()
        self._max = max_entries
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    def get(
        self, token_ids: list[int]
    ) -> Tuple[int, Optional[Any], Optional[Any]]:
        """Return (matched_len, past_key_values, last_logits) for the longest
        stored prefix of ``token_ids``."""
        target = tuple(token_ids)
        best_key: Optional[Tuple[int, ...]] = None
        best_len = 0
        with self._lock:
            for key in self._entries:
                klen = len(key)
                if klen <= best_len or klen > len(target):
                    continue
                if target[:klen] == key:
                    best_key = key
                    best_len = klen
            if best_key is None:
                self.misses += 1
                return 0, None, None
            entry = self._entries[best_key]
            self._entries.move_to_end(best_key)
            self.hits += 1
            past = _rebuild_past(entry["past_snapshot"])
            logits = entry["logits"]
            if isinstance(logits, torch.Tensor):
                logits = logits.clone()
            return best_len, past, logits

    def put(self, token_ids: list[int], past: Any, last_logits: Any) -> None:
        key = tuple(token_ids)
        snapshot = _snapshot_past(past)
        if isinstance(last_logits, torch.Tensor):
            last_logits = last_logits.detach().clone()
        with self._lock:
            self._entries[key] = {"past_snapshot": snapshot, "logits": last_logits}
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "entries": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total else 0.0,
        }
