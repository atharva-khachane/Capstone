"""
Cross-Chunk Resolver (Fix 3: Context Precision).

After reranking, multiple overlapping or adjacent chunks from the same
document section frequently appear in the top-k window, diluting context
precision with near-duplicate content.

CrossChunkResolver de-duplicates results by grouping chunks into "proximity
buckets" (doc_id × approx_section) and keeping only the highest-scoring chunk
from each bucket. The output list is re-sorted by score.
"""

from typing import List, Tuple

from ..core.schemas import Chunk


class CrossChunkResolver:
    """De-duplicate retrieved chunks to improve context precision.

    Grouping key: (doc_id, chunk_index // bucket_size)
    Within each group, the highest-scoring chunk is retained. Chunks without
    a chunk_index fall into bucket 0 for their document (still deduplicated
    against other index-less chunks from the same document).

    Args:
        bucket_size: Number of adjacent chunk indices collapsed into one bucket.
                     Default 3 means chunks 0-2 are one group, 3-5 the next, etc.
        min_score_gap: If the second-best chunk in a bucket scores within this
                       margin of the best, keep both (avoids discarding a
                       slightly-lower-scoring but genuinely different section).
                       Set to 0.0 to always keep exactly one per bucket.
    """

    def __init__(self, bucket_size: int = 3, min_score_gap: float = 0.15):
        self.bucket_size = bucket_size
        self.min_score_gap = min_score_gap

    def resolve(
        self,
        results: List[Tuple[Chunk, float]],
    ) -> List[Tuple[Chunk, float]]:
        """De-duplicate and return a cleaned result list sorted by score.

        Args:
            results: List of (chunk, score) tuples from reranking.

        Returns:
            De-duplicated list sorted by score (descending).
        """
        if not results:
            return []

        # Group by proximity bucket
        buckets: dict = {}  # key → list of (chunk, score)
        for chunk, score in results:
            idx = chunk.chunk_index if chunk.chunk_index is not None else 0
            key = (chunk.doc_id or "", idx // self.bucket_size)
            buckets.setdefault(key, []).append((chunk, score))

        # From each bucket pick the best chunk (and optionally a second if close)
        resolved: List[Tuple[Chunk, float]] = []
        for key, group in buckets.items():
            group.sort(key=lambda x: x[1], reverse=True)
            best_chunk, best_score = group[0]
            resolved.append((best_chunk, best_score))

            # Keep runner-up if within min_score_gap of best (genuinely different)
            if self.min_score_gap > 0.0 and len(group) > 1:
                runner_chunk, runner_score = group[1]
                if (best_score - runner_score) <= self.min_score_gap:
                    # Only keep if it's from a meaningfully different text span
                    if not self._texts_overlap(best_chunk.content, runner_chunk.content):
                        resolved.append((runner_chunk, runner_score))

        # Re-sort by score
        resolved.sort(key=lambda x: x[1], reverse=True)
        return resolved

    @staticmethod
    def _texts_overlap(text_a: str, text_b: str, threshold: float = 0.6) -> bool:
        """Return True if the two texts share more than `threshold` of their tokens."""
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())
        if not tokens_a or not tokens_b:
            return False
        overlap = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        return (overlap / union) >= threshold if union > 0 else False
