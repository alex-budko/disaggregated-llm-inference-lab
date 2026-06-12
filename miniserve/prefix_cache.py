"""In-memory prefix cache.

On a hit we skip prefill for the matched prefix. We do exact longest-prefix
match against stored entries (linear scan). vLLM/SGLang use a radix tree
chunked at block granularity; this is the same idea, simpler.

Stored: token_id tuple -> (past_key_values, last_logits). Caching the
final logits lets a *full* hit return the first sampled token without any
forward pass at all.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Optional, Tuple


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
            return best_len, entry["past"], entry["logits"]

    def put(self, token_ids: list[int], past: Any, last_logits: Any) -> None:
        key = tuple(token_ids)
        with self._lock:
            self._entries[key] = {"past": past, "logits": last_logits}
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
