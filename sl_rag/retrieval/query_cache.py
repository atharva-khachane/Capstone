"""
Query Result Cache for retrieval stability and latency reduction.

Caches (query, top_k) → List[Tuple[Chunk, float]] in RAM with TTL and
LRU eviction so repeated or near-identical queries skip the full FAISS +
hybrid fusion + cross-encoder reranking pipeline.

Benefits:
- Eliminates retrieval non-determinism from IVFFlat floating-point probe ordering
- Cuts repeated-query latency from 80-130s (model generation only) to near-zero
- Configurable TTL keeps stale results from persisting across re-ingests
"""

import hashlib
import time
from collections import OrderedDict
from typing import List, Optional, Tuple

from ..core.schemas import Chunk


class QueryResultCache:
    """LRU in-process cache for retrieval results.

    Key: SHA-256 prefix of ``(query.lower().strip() + ":" + str(top_k))``
    Entry: list of (Chunk, score) tuples plus bookkeeping.

    Parameters
    ----------
    ttl_seconds:
        How long a cached entry stays valid (default 3600 s = 1 hour).
        Set to 0 to disable expiry.
    max_entries:
        Maximum number of distinct (query, top_k) pairs to keep.
        Oldest entries are evicted first (LRU order).
    """

    def __init__(self, ttl_seconds: int = 3600, max_entries: int = 200):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._store: OrderedDict = OrderedDict()   # key → (results, timestamp, hit_count)
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self, query: str, top_k: int
    ) -> Optional[List[Tuple[Chunk, float]]]:
        """Return cached results if still valid, else None."""
        key = self._key(query, top_k)
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None

        results, ts, hits = entry
        if self.ttl_seconds > 0 and (time.time() - ts) > self.ttl_seconds:
            del self._store[key]
            self._misses += 1
            return None

        # Promote to most-recent position (LRU)
        self._store.move_to_end(key)
        self._store[key] = (results, ts, hits + 1)
        self._hits += 1
        return results

    def put(self, query: str, top_k: int, results: List[Tuple[Chunk, float]]) -> None:
        """Store retrieval results for a query."""
        key = self._key(query, top_k)
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (results, time.time(), 0)

        # Evict oldest entry when over capacity
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)

    def invalidate(self) -> None:
        """Clear all entries (call after re-ingest)."""
        self._store.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict:
        """Return cache hit/miss counters and current size."""
        total = self._hits + self._misses
        return {
            "entries": len(self._store),
            "max_entries": self.max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total else 0.0,
            "ttl_seconds": self.ttl_seconds,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(query: str, top_k: int) -> str:
        raw = f"{query.lower().strip()}:{top_k}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
