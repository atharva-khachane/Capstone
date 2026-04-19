"""
Adversarial Query Detector using Activation Shift Index (ASI).

Trust-RAG Module 1: Supplements regex injection detection with embedding-space
anomaly detection. Queries that deviate significantly from "normal" corpus
queries (measured by L2 distance to anchor embeddings) are flagged.

ASI(q) = mean( || h_q - h_a ||² for a in anchors )

If ASI > threshold → reject / flag as adversarial.

100% OFFLINE - anchors are built from corpus embeddings at ingest time.
"""

import logging
import re
import numpy as np
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Keyword patterns that indicate prompt injection / jailbreak attempts.
# These are checked verbatim (case-insensitive) BEFORE the embedding-space ASI
# check so that text-based attacks are caught even when the query embedding
# lands inside the normal distribution.
_INJECTION_KEYWORDS: List[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"ignore\s+(previous|prior|above|all)\s+(instructions?|rules?|guidelines?|prompts?|directions?)",
    r"disregard\s+(?:all\s+)?(?:previous\s+)?(?:instructions?|rules?|guidelines?|prompts?)",
    r"forget\s+(your\s+)?(previous\s+)?(instructions?|training|rules?)",
    r"you\s+are\s+now\s+(a|an|the)\s+",
    r"act\s+as\s+(a|an|the)\s+(?!government|regulatory|official)",
    r"pretend\s+(you\s+)?(are|have|don'?t)\s+",
    r"roleplay\s+as\s+",
    r"\bDAN\b",
    r"do\s+anything\s+now",
    r"jailbreak",
    r"bypass\s+(safety|filter|restriction|policy|guideline|system)",
    r"override\s+(safety|filter|restriction|policy|guideline|instructions?|system)",
    r"new\s+(system\s+)?instructions?\s*:",
    r"<\s*system\s*>",
    r"\[INST\]|\[/?SYS\]|<<SYS>>",
    r"reveal\s+(your\s+)?(system\s+)?(prompt|instructions?|rules?)",
    r"print\s+(your\s+)?(system\s+)?(prompt|instructions?)",
    r"what\s+(are\s+)?your\s+(system\s+)?(prompt|instructions?|guidelines?|rules?)",
]]


class ASIDetector:
    """
    Activation Shift Index adversarial query detector.

    Build anchors once from corpus chunk embeddings (call build_anchors_from_chunks),
    then call is_adversarial(query_emb) at query time. Complements the existing
    regex-based PromptBuilder.detect_injection() check.
    """

    def __init__(
        self,
        anchor_embeddings: Optional[np.ndarray] = None,
        threshold: float = 2.5,
        max_anchors: int = 300,
    ):
        """
        Args:
            anchor_embeddings: (N, D) float32 array of normal query/chunk embeddings.
            threshold: ASI value above which a query is flagged as adversarial.
                       Tune by observing the distribution on known-good queries.
            max_anchors: Subsample to this many anchors for speed.
        """
        self.threshold = threshold
        self.max_anchors = max_anchors
        self.anchors: Optional[np.ndarray] = None

        if anchor_embeddings is not None:
            self._set_anchors(anchor_embeddings)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _set_anchors(self, embeddings: np.ndarray) -> None:
        """Subsample and store anchor embeddings."""
        if len(embeddings) > self.max_anchors:
            idx = np.random.choice(len(embeddings), self.max_anchors, replace=False)
            embeddings = embeddings[idx]
        self.anchors = embeddings.astype(np.float32)

    def build_anchors_from_chunks(self, chunks) -> None:
        """
        Build anchor set from corpus Chunk objects (post-embedding).

        Call this once after pipeline.ingest() so the detector reflects
        the actual corpus embedding distribution.

        Args:
            chunks: List of Chunk objects with .embedding set.
        """
        valid = [c for c in chunks if c.embedding is not None]
        if not valid:
            print("[ASI] WARNING: No embedded chunks provided. Detector disabled.")
            return

        embeddings = np.stack([c.embedding for c in valid]).astype(np.float32)
        self._set_anchors(embeddings)
        print(f"[ASI] Built anchor set from {len(self.anchors)} chunk embeddings "
              f"(threshold={self.threshold})")

    # ------------------------------------------------------------------
    # Core ASI computation
    # ------------------------------------------------------------------

    def compute_asi(self, query_emb: np.ndarray) -> float:
        """
        Compute the Activation Shift Index for a single query embedding.

        ASI(q) = mean( || h_q - h_a ||² ) over all anchors.

        Args:
            query_emb: (D,) float32 normalized query embedding.

        Returns:
            ASI score (float). Higher = more anomalous.

        Note:
            Returns 0.0 (pass-through) when anchors have not been built, but
            emits a WARNING so a missed build_anchors_from_chunks() call is
            always visible — prevents the silent 0.000 metric from Run 6.
        """
        if self.anchors is None or len(self.anchors) == 0:
            # Bug 1 fix: do NOT silently pass through — warn loudly so an
            # LLM-backend switch that skips ingest() is immediately audible.
            logger.warning(
                "[ASI] Detector not initialised (anchors=None). "
                "Every query will score 0.0 (no adversarial filtering). "
                "Ensure build_anchors_from_chunks() is called after ingest()."
            )
            return 0.0  # pass-through, but now visible in logs

        q = query_emb.astype(np.float32)
        # Vectorized: squared L2 distances to all anchors
        diffs = self.anchors - q[np.newaxis, :]          # (N, D)
        sq_dists = np.sum(diffs ** 2, axis=1)            # (N,)
        return float(np.mean(sq_dists))

    def is_adversarial(
        self,
        query_emb: np.ndarray,
        query_text: Optional[str] = None,
    ) -> Tuple[bool, float]:
        """Determine if a query is adversarial via keyword check then ASI.

        The check runs in two stages:
          1. Keyword injection scan (fast, text-based) — catches explicit
             injection / jailbreak patterns regardless of embedding distance.
          2. ASI embedding-space anomaly detection — catches semantic drift
             that text patterns may miss.

        Args:
            query_emb: (D,) float32 normalized query embedding.
            query_text: Optional raw query string for keyword scanning.

        Returns:
            Tuple of (is_adversarial: bool, asi_score: float).
            When a keyword match triggers, asi_score is returned as 9.99 to
            signal a definitive block without polluting the normal ASI range.
        """
        # Stage 1: keyword / pattern injection check
        if query_text:
            for pattern in _INJECTION_KEYWORDS:
                if pattern.search(query_text):
                    return True, 9.99

        # Stage 2: embedding-space anomaly (ASI)
        score = self.compute_asi(query_emb)
        return score > self.threshold, score

    # ------------------------------------------------------------------
    # Threshold calibration helper
    # ------------------------------------------------------------------

    def calibrate_threshold(
        self,
        normal_query_embs: np.ndarray,
        percentile: float = 95.0,
    ) -> float:
        """
        Suggest a threshold based on the ASI distribution of known-normal queries.

        Set threshold = percentile(ASI scores of normal queries).
        Queries above this are in the top (100-percentile)% of anomaly scores.

        Args:
            normal_query_embs: (M, D) embeddings of known-good queries.
            percentile: Percentile of normal distribution to use as cutoff.

        Returns:
            Suggested threshold value (also updates self.threshold).
        """
        scores = np.array([self.compute_asi(q) for q in normal_query_embs])
        suggested = float(np.percentile(scores, percentile))
        print(f"[ASI] Calibrated threshold at p{percentile:.0f} = {suggested:.4f} "
              f"(mean={scores.mean():.4f}, std={scores.std():.4f})")
        self.threshold = suggested
        return suggested

    def get_info(self) -> dict:
        """Return detector metadata."""
        return {
            "num_anchors": len(self.anchors) if self.anchors is not None else 0,
            "threshold": self.threshold,
            "embedding_dim": self.anchors.shape[1] if self.anchors is not None else None,
            "enabled": self.anchors is not None,
        }

    @property
    def is_ready(self) -> bool:
        """True when anchor embeddings have been built and the detector is active.

        Use this for the pipeline smoke-test:
            assert self.asi_detector.is_ready, "ASI guardrail has no anchors"
        """
        return self.anchors is not None and len(self.anchors) > 0
