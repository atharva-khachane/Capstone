"""
Multi-Feature Trust Scorer (Trust-RAG Module 4).

Replaces raw retrieval scores with a composite trust score:

    T(q, d) = sigmoid(w1*S + w2*C + w3*F + w4*I)

Where:
    S = Semantic similarity     — cosine(query, doc)      [from retrieval]
    C = Credibility             — rule-based source score [from metadata]
    F = Freshness               — exp(-λ * days_elapsed)  [from metadata]
    I = Inter-doc Consistency   — mean cosine(doc, peers) [computed here]

Weights: (0.4, 0.2, 0.2, 0.2) — heuristically set, semantics-first.

100% OFFLINE - No external APIs.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.schemas import Chunk


# ---------------------------------------------------------------------------
# Source credibility lookup
# ---------------------------------------------------------------------------

# Credibility scores by document source keywords (case-insensitive substring match)
# Ordered from highest to lowest credibility.
_CREDIBILITY_RULES: List[Tuple[str, float]] = [
    ("gfr",            0.95),   # General Financial Rules — government mandated
    ("general financial", 0.95),
    ("procurement",    0.90),   # Procurement manuals — official policy docs
    ("tender",         0.88),
    ("consultancy",    0.87),
    ("technical",      0.80),   # Technical memos / DPRs
    ("memo",           0.80),
    ("telemetry",      0.72),   # Telemetry data — factual but narrow scope
    ("scada",          0.72),
    ("report",         0.75),
    ("manual",         0.85),
]
_DEFAULT_CREDIBILITY = 0.60


def _source_credibility(metadata: Dict) -> float:
    """Score credibility from document metadata (filename / filepath / domain)."""
    search_str = " ".join(str(v) for v in metadata.values()).lower()
    for keyword, score in _CREDIBILITY_RULES:
        if keyword in search_str:
            return score
    return _DEFAULT_CREDIBILITY


# ---------------------------------------------------------------------------
# Dataclass for trust results
# ---------------------------------------------------------------------------

@dataclass
class TrustBreakdown:
    """Per-chunk trust decomposition returned by TrustScorer."""
    semantic: float          # cosine(query, chunk) — from retrieval pipeline
    credibility: float       # source authority [0, 1]
    freshness: float         # recency score [0, 1]
    consistency: float       # agreement with other retrieved chunks [0, 1]
    trust_score: float       # sigmoid(weighted sum) — final trust [0, 1]
    raw_weighted_sum: float  # pre-sigmoid value (for calibration / debugging)

    def to_dict(self) -> Dict:
        return {
            "semantic":       round(self.semantic, 4),
            "credibility":    round(self.credibility, 4),
            "freshness":      round(self.freshness, 4),
            "consistency":    round(self.consistency, 4),
            "trust_score":    round(self.trust_score, 4),
        }


# ---------------------------------------------------------------------------
# TrustScorer
# ---------------------------------------------------------------------------

class TrustScorer:
    """
    Compute composite trust scores for retrieved chunks.

    Usage::

        scorer = TrustScorer()
        results = pipeline.retrieve(query, top_k=10)
        scored = scorer.score_chunks(query_emb, results)
        # scored: List[Tuple[Chunk, float, TrustBreakdown]]
        # sort by scored[i][1] (trust_score) for final ranking

    Parameters
    ----------
    weights : (w_semantic, w_credibility, w_freshness, w_consistency)
        Must sum to 1.0.  Defaults: (0.4, 0.2, 0.2, 0.2)
    lambda_decay : float
        Freshness decay constant (per day). Default 0.001 ≈ half-life ~693 days.
    reference_date : datetime | None
        Date to measure freshness from. Defaults to today (UTC).
    """

    def __init__(
        self,
        weights: Tuple[float, float, float, float] = (0.4, 0.2, 0.2, 0.2),
        lambda_decay: float = 0.001,
        reference_date: Optional[datetime] = None,
    ):
        assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1.0"
        self.w_semantic, self.w_credibility, self.w_freshness, self.w_consistency = weights
        self.lambda_decay = lambda_decay
        self.reference_date = reference_date or datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_chunks(
        self,
        query_emb: np.ndarray,
        retrieved: List[Tuple[Chunk, float]],
    ) -> List[Tuple[Chunk, float, TrustBreakdown]]:
        """
        Compute trust scores for all retrieved chunks.

        Args:
            query_emb:  Normalized (D,) query embedding.
            retrieved:  List of (Chunk, retrieval_score) from the pipeline.

        Returns:
            List of (Chunk, trust_score, TrustBreakdown), sorted by trust_score DESC.
        """
        if not retrieved:
            return []

        # Pre-compute all chunk embeddings for consistency calculation
        chunk_embs = np.array([
            c.embedding if c.embedding is not None else np.zeros(768)
            for c, _ in retrieved
        ], dtype=np.float32)

        results: List[Tuple[Chunk, float, TrustBreakdown]] = []
        for i, (chunk, retrieval_score) in enumerate(retrieved):
            breakdown = self._compute_breakdown(
                query_emb=query_emb,
                chunk=chunk,
                retrieval_score=retrieval_score,
                chunk_embs=chunk_embs,
                own_idx=i,
            )
            results.append((chunk, breakdown.trust_score, breakdown))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Per-chunk feature computation
    # ------------------------------------------------------------------

    def _compute_breakdown(
        self,
        query_emb: np.ndarray,
        chunk: Chunk,
        retrieval_score: float,
        chunk_embs: np.ndarray,
        own_idx: int,
    ) -> TrustBreakdown:
        # Use the pipeline's pre-computed retrieval score as the semantic component.
        # This is either the hybrid (BM25 + dense) score or the cross-encoder score
        # (already sigmoid-transformed to [0, 1]), both of which are more informative
        # than recomputing raw cosine from bi-encoder embeddings here.
        # Fall back to embedding cosine only if the score is out of the expected range.
        if 0.0 <= retrieval_score <= 1.0:
            S = float(retrieval_score)
        else:
            S = self._semantic(query_emb, chunk)
        C = self._credibility(chunk)
        F = self._freshness(chunk)
        I = self._consistency(chunk_embs, own_idx)

        raw = (
            self.w_semantic     * S
            + self.w_credibility * C
            + self.w_freshness   * F
            + self.w_consistency * I
        )
        T = self._sigmoid(raw)

        return TrustBreakdown(
            semantic=float(S),
            credibility=float(C),
            freshness=float(F),
            consistency=float(I),
            trust_score=float(T),
            raw_weighted_sum=float(raw),
        )

    def _semantic(self, query_emb: np.ndarray, chunk: Chunk) -> float:
        """Cosine similarity between query and chunk (already computed by retrieval)."""
        if chunk.embedding is None:
            return 0.0
        q = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        c = chunk.embedding / (np.linalg.norm(chunk.embedding) + 1e-10)
        sim = float(np.dot(q, c))
        # Cosine in [-1,1]; map to [0,1]
        return (sim + 1.0) / 2.0

    @staticmethod
    def _credibility(chunk: Chunk) -> float:
        """Rule-based credibility from metadata."""
        return _source_credibility(chunk.metadata or {})

    def _freshness(self, chunk: Chunk) -> float:
        """Exponential freshness decay based on document creation date."""
        meta = chunk.metadata or {}
        # Try common metadata keys for creation date
        raw_date = (
            meta.get("creation_date")
            or meta.get("created")
            or meta.get("date")
            or meta.get("modification_date")
        )
        if raw_date is None:
            return 0.75  # neutral if unknown

        try:
            if isinstance(raw_date, (int, float)):
                doc_date = datetime.fromtimestamp(raw_date, tz=timezone.utc)
            elif isinstance(raw_date, str):
                # Handle ISO strings and "D:YYYYMMDDHHmmss" PDF format
                raw_date = raw_date.strip()
                if raw_date.startswith("D:"):
                    raw_date = raw_date[2:16]
                    doc_date = datetime.strptime(raw_date, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
                else:
                    doc_date = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
            elif isinstance(raw_date, datetime):
                doc_date = raw_date if raw_date.tzinfo else raw_date.replace(tzinfo=timezone.utc)
            else:
                return 0.75
        except Exception:
            return 0.75

        ref = self.reference_date if self.reference_date.tzinfo else self.reference_date.replace(tzinfo=timezone.utc)
        days_elapsed = max(0.0, (ref - doc_date).total_seconds() / 86400.0)
        return math.exp(-self.lambda_decay * days_elapsed)

    @staticmethod
    def _consistency(chunk_embs: np.ndarray, own_idx: int) -> float:
        """Mean cosine similarity of this chunk with all other retrieved chunks."""
        n = len(chunk_embs)
        if n <= 1:
            return 1.0  # only chunk — trivially consistent

        own = chunk_embs[own_idx]
        own_norm = own / (np.linalg.norm(own) + 1e-10)

        sims = []
        for i, emb in enumerate(chunk_embs):
            if i == own_idx:
                continue
            emb_norm = emb / (np.linalg.norm(emb) + 1e-10)
            sim = float(np.dot(own_norm, emb_norm))
            sims.append((sim + 1.0) / 2.0)  # map to [0,1]

        return float(np.mean(sims)) if sims else 1.0

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Numerically stable sigmoid."""
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)
