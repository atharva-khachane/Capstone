"""
Post-Retrieval Validation Pipeline (Layer 5).

Per ANTIGRAVITY_PROMPT.md Phase 5, validates retrieved results by:
  1. Citation verification   -- every result has a traceable source
  2. Confidence scoring      -- 0.6 * avg(retrieval_scores) + 0.4 * consistency
  3. Hallucination detection -- flag answers with >30% novel words

Thresholds (from methodology + papers/04_Ji2023_Hallucination_Survey.pdf):
  - min_consistency_score : 0.7
  - hallucination_high   : >30% unique words not in context
  - hallucination_medium : >15% unique words not in context

100% OFFLINE - No external APIs.
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter

from ..core.schemas import Chunk


class ValidationPipeline:
    """
    Multi-stage validation of retrieval results (and future LLM answers).

    Validation steps:
        1. Citation verification  -- ensure all chunks have valid source metadata
        2. Confidence scoring     -- weighted blend of retrieval + consistency
        3. Hallucination detection (used when LLM layer is added)
    """

    def __init__(
        self,
        embedding_generator=None,
        min_consistency_score: float = 0.7,
    ):
        self.embedding_generator = embedding_generator
        self.min_consistency_score = min_consistency_score

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def validate_retrieval(
        self,
        query: str,
        results: List[Tuple[Chunk, float]],
    ) -> Dict[str, Any]:
        """Validate a set of retrieval results for a given query."""
        if not results:
            return {
                "is_valid": False,
                "reason": "no_results",
                "confidence": 0.0,
                "citations": [],
                "citation_quality": "none",
            }

        citations = self._verify_citations(results)
        scores = [score for _, score in results]
        confidence = self._compute_confidence(scores)
        quality = "good" if all(c["valid"] for c in citations) else "partial"

        return {
            "is_valid": confidence >= 0.3 and quality != "none",
            "confidence": round(confidence, 4),
            "citations": citations,
            "citation_quality": quality,
            "num_results": len(results),
            "score_stats": {
                "mean": round(float(np.mean(scores)), 4),
                "min": round(float(np.min(scores)), 4),
                "max": round(float(np.max(scores)), 4),
            },
        }

    def validate_answer(
        self,
        answer: str,
        context_chunks: List[Chunk],
        retrieval_scores: List[float],
    ) -> Dict[str, Any]:
        """Validate an LLM-generated answer against its source context.

        This method is designed for use once the LLM layer (Layer 4) is added.
        """
        result: Dict[str, Any] = {}

        # 1. Consistency
        result["consistency_score"] = self._check_consistency(answer, context_chunks)

        # 2. Citation extraction
        result["extracted_citations"] = self._extract_answer_citations(answer)
        result["citations_valid"] = self._verify_answer_citations(
            result["extracted_citations"], context_chunks,
        )

        # 3. Confidence
        result["confidence"] = self._compute_confidence(
            retrieval_scores, result["consistency_score"],
        )

        # 4. Hallucination
        result["hallucination_risk"] = self._detect_hallucination(answer, context_chunks)

        # 5. Overall validity
        result["is_valid"] = (
            result["consistency_score"] >= self.min_consistency_score
            and result["citations_valid"]
            and result["hallucination_risk"] != "high"
        )

        return result

    # ------------------------------------------------------------------
    # Citation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _verify_citations(
        results: List[Tuple[Chunk, float]],
    ) -> List[Dict[str, Any]]:
        citations = []
        for chunk, score in results:
            meta = chunk.metadata or {}
            citations.append({
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "domain": chunk.domain,
                "score": round(float(score), 4),
                "filepath": meta.get("filepath", meta.get("source_document", "")),
                "valid": bool(chunk.doc_id and chunk.chunk_id),
            })
        return citations

    @staticmethod
    def _extract_answer_citations(answer: str) -> List[Tuple[str, str]]:
        """Extract [Source: doc_id, Chunk: N] markers from an LLM answer."""
        return re.findall(r"\[Source:\s*(.*?),\s*Chunk:\s*(\d+)\]", answer)

    @staticmethod
    def _verify_answer_citations(
        citations: List[Tuple[str, str]],
        context_chunks: List[Chunk],
    ) -> bool:
        if not citations:
            return True  # no citations to verify
        for doc_id, chunk_idx in citations:
            found = any(
                c.doc_id == doc_id and c.chunk_index == int(chunk_idx)
                for c in context_chunks
            )
            if not found:
                return False
        return True

    # ------------------------------------------------------------------
    # Confidence & consistency
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_confidence(
        retrieval_scores: List[float],
        consistency_score: float = 1.0,
    ) -> float:
        """confidence = 0.6 * avg(retrieval_scores) + 0.4 * consistency_score"""
        avg_ret = float(np.mean(retrieval_scores)) if retrieval_scores else 0.0
        return 0.6 * avg_ret + 0.4 * consistency_score

    def _check_consistency(
        self,
        answer: str,
        context_chunks: List[Chunk],
    ) -> float:
        """Semantic similarity between answer and best-matching context chunk."""
        if not self.embedding_generator or not context_chunks:
            return 0.0

        answer_emb = self.embedding_generator.generate_query_embedding(answer, normalize=True)
        best = 0.0
        for chunk in context_chunks:
            if not chunk.has_embedding():
                continue
            sim = float(np.dot(answer_emb, chunk.embedding))
            best = max(best, sim)
        return best

    # ------------------------------------------------------------------
    # Hallucination detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_hallucination(
        answer: str,
        context_chunks: List[Chunk],
    ) -> str:
        """Heuristic hallucination check (papers/04_Ji2023_Hallucination_Survey.pdf).

        Returns 'low', 'medium', or 'high'.
        """
        answer_words = set(answer.lower().split())
        context_words: set = set()
        for chunk in context_chunks:
            context_words.update(chunk.content.lower().split())

        if not answer_words:
            return "low"

        unique_to_answer = answer_words - context_words
        ratio = len(unique_to_answer) / len(answer_words)

        if ratio > 0.30:
            return "high"
        if ratio > 0.15:
            return "medium"
        return "low"
