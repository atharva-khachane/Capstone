"""
Deterministic retrieval/rerank policy helpers.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class RetrievalPolicy:
    similarity_threshold: float = 0.5
    rerank_candidates: int = 20
    final_top_k: int = 5
    initial_top_k_candidates: int = 20

    def resolve_top_k(self, requested_top_k: int) -> int:
        if requested_top_k and requested_top_k > 0:
            return requested_top_k
        return self.final_top_k

    def candidate_window(self, available: int) -> int:
        return min(max(self.rerank_candidates, self.final_top_k), max(0, available))
