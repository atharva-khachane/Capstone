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
        self._stopwords = {
            "the", "and", "for", "with", "this", "that", "from", "into", "are", "was",
            "were", "have", "has", "had", "will", "would", "should", "could", "can",
            "you", "your", "our", "their", "they", "them", "there", "here", "then",
            "also", "only", "based", "provided", "context", "source", "chunk", "note",
            "please", "know", "like", "more", "about", "according", "document",
            "documents", "information", "response", "query", "answer",
        }

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
        debug: bool = False,
    ) -> Dict[str, Any]:
        """Validate an LLM-generated answer against its source context."""
        result: Dict[str, Any] = {}

        # 1. Consistency (semantic similarity)
        result["consistency_score"] = self._check_consistency(answer, context_chunks)

        # 2. Citation extraction
        result["extracted_citations"] = self._extract_answer_citations(answer)
        result["citations_valid"] = self._verify_answer_citations(
            result["extracted_citations"], context_chunks,
        )

        # 3. Claim-level faithfulness (Trust-RAG Module 1D)
        result["faithfulness_score"] = self._compute_faithfulness(answer, context_chunks)

        # 3.5. Internal consistency — structured vs prose contradiction check
        # Detects inversions like QCBS (table: technical=65-80%, bullets: cost=70-100%)
        # without requiring any external grounding. Pure self-consistency check.
        internal = self._check_internal_consistency(answer)
        result["internal_consistency"] = internal["consistent"]
        result["internal_contradiction"] = internal["contradiction"]
        if not internal["consistent"]:
            # Bump hallucination risk to at least medium when answer contradicts itself
            result["_internal_inconsistency_flag"] = True

        # 4. Confidence — weighted blend with faithfulness + retrieval coverage penalty
        result["confidence"] = self._compute_confidence(
            retrieval_scores,
            result["consistency_score"],
            result["faithfulness_score"],
            internal_consistent=internal["consistent"],
        )

        # 5. Hallucination
        hallucination = self._analyze_hallucination(answer, context_chunks, retrieval_scores)
        # Escalate to at least medium if internal consistency failed
        raw_risk = hallucination["risk"]
        if not internal["consistent"] and raw_risk == "low":
            raw_risk = "medium"
        result["hallucination_risk"] = raw_risk
        if debug:
            result["hallucination_debug"] = hallucination

        # 6. Overall validity
        result["is_valid"] = (
            result["consistency_score"] >= self.min_consistency_score
            and result["citations_valid"]
            and result["hallucination_risk"] != "high"
            and result["internal_consistency"]
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
    def _extract_answer_citations(answer: str) -> List[int]:
        """Extract [N] block-index citations from an LLM answer.

        Matches citations like [1], [2], [3] that correspond to the numbered
        context blocks shown in the prompt. Single-digit and multi-digit block
        numbers are both captured; markdown bold markers (**[1]**) are handled
        by stripping surrounding non-bracket characters before matching.
        """
        return [int(m) for m in re.findall(r'\[(\d+)\]', answer)]

    @staticmethod
    def _verify_answer_citations(
        citations: List[int],
        context_chunks: List[Chunk],
    ) -> bool:
        """Check that every cited block number falls within the context window."""
        if not citations:
            return True  # no citations to verify
        return all(1 <= n <= len(context_chunks) for n in citations)

    # ------------------------------------------------------------------
    # Confidence & consistency
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_confidence(
        retrieval_scores: List[float],
        consistency_score: float = 1.0,
        faithfulness_score: float = -1.0,
        internal_consistent: bool = True,
    ) -> float:
        """Confidence blend: faithfulness + retrieval + consistency + coverage penalty.

        Formula (Trust-RAG v2 — rebalanced weights):
            If faithfulness available: U = 0.4*Faith + 0.4*avg_top3 + 0.2*consistency
            Otherwise:                 U = 0.75*avg_top3 + 0.25*consistency  (legacy)

        Rationale for 0.4/0.4/0.2:
            Equalising faithfulness and retrieval gives avg_top3 more signal so the
            score tracks retrieval instability rather than masking it.

        Retrieval coverage penalty:
            When fewer than 3 distinct chunks are retrieved, the model has shallow
            context and is more prone to confabulation (e.g. QCBS table inversion).
            Penalty: -0.15 for 1 chunk, -0.10 for 2 chunks.

        Internal consistency penalty:
            -0.10 applied when structured content and prose in the same answer
            contain contradictory numeric/percentage claims.
        """
        if not retrieval_scores:
            return 0.0
        top3 = sorted([float(s) for s in retrieval_scores], reverse=True)[:3]
        avg_top3 = float(np.mean(top3))

        # Normalize raw cosine consistency to [0, 1].
        # all-mpnet-base-v2 answer-vs-chunk cosine similarities fall in roughly
        # [0.15, 0.75]; mapping that range to [0, 1] makes the component
        # comparable to faithfulness and retrieval scores in the blend.
        if consistency_score > 0.05:
            normalized_consistency = min(1.0, max(0.0, (consistency_score - 0.15) / 0.60))
        else:
            normalized_consistency = 0.75  # unknown / no embedder — neutral fallback
        effective_consistency = normalized_consistency

        if faithfulness_score >= 0.0:
            # Trust-RAG v2: retrieval and faithfulness carry equal weight
            raw = 0.4 * faithfulness_score + 0.4 * avg_top3 + 0.2 * effective_consistency
        else:
            # legacy formula (when called without faithfulness)
            raw = 0.75 * avg_top3 + 0.25 * effective_consistency

        # Retrieval coverage penalty — shallow context inflates confidence
        n = len(retrieval_scores)
        if n == 1:
            raw -= 0.15
        elif n == 2:
            raw -= 0.10

        # Internal consistency penalty — answer contradicts itself
        if not internal_consistent:
            raw -= 0.10

        return min(1.0, max(0.0, round(raw, 4)))

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
    # Internal consistency (Trust-RAG Module 1E)
    # ------------------------------------------------------------------

    @staticmethod
    def _check_internal_consistency(answer: str) -> Dict[str, Any]:
        """Detect numeric/percentage contradictions between structured and prose content.

        Motivation:
            Cosine-similarity faithfulness cannot catch inversions — "technical score
            is primary" and "cost score is primary" share nearly identical token sets.
            This check requires no external grounding: it compares what the answer's
            own table/list rows say against what its prose summary says.

        Algorithm:
            1. Split answer into structured lines (table rows, bullet points) and
               prose sentences.
            2. Extract (subject, percentage) pairs from each section using regex.
            3. For each subject that appears in both sections, compare the percentage
               ranges. Flag a contradiction if the ranges disagree by > 15pp.

        Returns:
            Dict with keys:
                consistent (bool): False if a numeric contradiction was found.
                contradiction (str): Human-readable description, or empty string.
        """
        # --- Regex helpers ---
        # Extracts patterns like "technical 80%", "cost 25-35%", "quality score 70%"
        _num_pat = re.compile(
            r"(technical|quality|cost|price|lcs|qcbs)[^\n]{0,40}?(\d{1,3})\s*[-–]?\s*(\d{0,3})\s*%",
            re.IGNORECASE,
        )

        def _extract_pairs(text: str) -> Dict[str, Tuple[float, float]]:
            """Return {subject_lower: (low_pct, high_pct)} from text."""
            pairs: Dict[str, Tuple[float, float]] = {}
            for m in _num_pat.finditer(text):
                subj = m.group(1).lower()
                lo = float(m.group(2))
                hi = float(m.group(3)) if m.group(3) else lo
                if subj not in pairs:
                    pairs[subj] = (min(lo, hi), max(lo, hi))
                else:
                    # Expand range if multiple mentions
                    existing_lo, existing_hi = pairs[subj]
                    pairs[subj] = (min(existing_lo, lo, hi), max(existing_hi, lo, hi))
            return pairs

        # --- Split answer into structured vs prose lines ---
        lines = answer.split("\n")
        structured_lines: List[str] = []
        prose_lines: List[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Table rows (contain |) and bullet points (* or -)
            if "|" in stripped or stripped.startswith(("*", "-", "•")):
                structured_lines.append(stripped)
            else:
                prose_lines.append(stripped)

        structured_text = " ".join(structured_lines)
        prose_text = " ".join(prose_lines)

        struct_pairs = _extract_pairs(structured_text)
        prose_pairs = _extract_pairs(prose_text)

        if not struct_pairs or not prose_pairs:
            # Not enough numeric content to compare
            return {"consistent": True, "contradiction": ""}

        # --- Compare shared subjects ---
        TOLERANCE_PP = 15.0  # percentage-point tolerance before flagging
        for subj in struct_pairs:
            if subj not in prose_pairs:
                continue
            s_lo, s_hi = struct_pairs[subj]
            p_lo, p_hi = prose_pairs[subj]
            # Ranges overlap if max(lo) <= min(hi)+tolerance
            overlap = min(s_hi, p_hi) - max(s_lo, p_lo)
            if overlap < -TOLERANCE_PP:
                contradiction = (
                    f"Contradiction on '{subj}': "
                    f"structured content shows {s_lo:.0f}–{s_hi:.0f}%, "
                    f"prose summary shows {p_lo:.0f}–{p_hi:.0f}%."
                )
                return {"consistent": False, "contradiction": contradiction}

        return {"consistent": True, "contradiction": ""}

    # ------------------------------------------------------------------
    # Faithfulness (Trust-RAG Module 1D)
    # ------------------------------------------------------------------

    def _compute_faithfulness(
        self,
        answer: str,
        context_chunks: List[Chunk],
        semantic_threshold: float = 0.40,
        jaccard_threshold: float = 0.15,
    ) -> float:
        """Claim-level faithfulness: fraction of answer sentences supported by context.

        Algorithm (Trust-RAG v2 — semantic-primary):
            1. Split answer into sentence-level claims.
            2. PRIMARY: If embedder available, embed each claim and compute cosine
               similarity against each context chunk embedding. A claim is 'supported'
               if max(cosine_sim) >= semantic_threshold.
            3. FALLBACK: If no embedder, use Jaccard token overlap at jaccard_threshold.
               (Jaccard is kept as fallback only — it cannot handle paraphrase or
               multi-word legal phrases like "valid appropriation".)
            4. faithfulness = supported_claims / total_claims

        Args:
            answer: LLM-generated answer string.
            context_chunks: Retrieved chunks used as context.
            semantic_threshold: Min cosine similarity for semantic support (primary).
                Raised to 0.40 (from 0.25) to avoid near-universal claim support
                with all-mpnet-base-v2, which made faithfulness ≈ 1.0 for virtually
                every answer and masked real answer quality differences.
            jaccard_threshold: Min Jaccard overlap for lexical support (fallback only).

        Returns:
            Faithfulness score in [0.0, 1.0].
        """
        if not answer or not context_chunks:
            return 0.0

        # Split into claims (sentences), filter trivially short ones
        claims = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer)]
        claims = [c for c in claims if len(c.split()) >= 4]
        if not claims:
            return 1.0  # single-token or empty answer — no claims to falsify

        # --- PRIMARY PATH: sentence-level cosine similarity ---------------
        # Handles paraphrase and domain-specific multi-word phrases that Jaccard
        # systematically misses (e.g. "valid appropriation", "fresh charge").
        if self.embedding_generator is not None:
            # Collect pre-computed chunk embeddings (skip chunks without one)
            chunk_embs = [
                (chunk, chunk.embedding)
                for chunk in context_chunks
                if chunk.has_embedding()
            ]
            if chunk_embs:
                supported = 0
                for claim in claims:
                    claim_emb = self.embedding_generator.generate_query_embedding(
                        claim, normalize=True
                    )
                    best_sim = max(
                        float(np.dot(claim_emb, emb))
                        for _, emb in chunk_embs
                    )
                    if best_sim >= semantic_threshold:
                        supported += 1
                return round(supported / len(claims), 4)

        # --- FALLBACK PATH: Jaccard token overlap -------------------------
        # Used only when no embedder is available (offline / unit-test mode).
        def _tokenset(text: str) -> set:
            return {
                w.lower() for w in re.findall(r"[a-zA-Z]{3,}", text)
                if w.lower() not in self._stopwords
            }

        context_token_sets = [_tokenset(c.content) for c in context_chunks]
        supported = 0
        for claim in claims:
            claim_tokens = _tokenset(claim)
            if not claim_tokens:
                supported += 1  # empty claim trivially supported
                continue
            for ctx_tokens in context_token_sets:
                union = claim_tokens | ctx_tokens
                if not union:
                    continue
                jaccard = len(claim_tokens & ctx_tokens) / len(union)
                if jaccard >= jaccard_threshold:
                    supported += 1
                    break

        return round(supported / len(claims), 4)

    # ------------------------------------------------------------------
    # Hallucination detection
    # ------------------------------------------------------------------

    def _detect_hallucination(
        self,
        answer: str,
        context_chunks: List[Chunk],
    ) -> str:
        """Backward-compatible hallucination API (uses neutral retrieval quality)."""
        analysis = self._analyze_hallucination(
            answer,
            context_chunks,
            retrieval_scores=[],
        )
        return analysis["risk"]

    @staticmethod
    def _light_stem(token: str) -> str:
        """Lightweight stemming to reduce inflectional mismatch noise."""
        if len(token) > 5 and token.endswith("ing"):
            return token[:-3]
        if len(token) > 4 and token.endswith("ed"):
            return token[:-2]
        if len(token) > 4 and token.endswith("es"):
            return token[:-2]
        if len(token) > 3 and token.endswith("s"):
            return token[:-1]
        return token

    def _strip_boilerplate(self, text: str) -> str:
        cleaned = text
        # Remove bracket citations and source id markers.
        cleaned = re.sub(r"\[Source:[^\]]+\]", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\[SOURCE:[^\]]+\]", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\[\d+\]\(#reference-\d+\)", " ", cleaned)
        # Remove common assistant boilerplate lines.
        cleaned = re.sub(
            r"(?is)\b(note:|please let me know|if you'd like|i cannot provide general|"
            r"this response only uses|based on the provided context).*",
            " ",
            cleaned,
        )
        return cleaned

    def _normalize_tokens(self, text: str) -> List[str]:
        cleaned = self._strip_boilerplate(text.lower())
        tokens = re.findall(r"[a-z][a-z0-9]{1,}", cleaned)
        normalized = []
        for token in tokens:
            if token in self._stopwords:
                continue
            stemmed = self._light_stem(token)
            if len(stemmed) < 3 or stemmed in self._stopwords:
                continue
            normalized.append(stemmed)
        return normalized

    def _analyze_hallucination(
        self,
        answer: str,
        context_chunks: List[Chunk],
        retrieval_scores: List[float],
    ) -> Dict[str, Any]:
        """
        Retrieval-aware hallucination analysis.

        Combines normalized novelty with retrieval quality rather than relying on
        a single raw token overlap threshold.
        """
        answer_tokens = set(self._normalize_tokens(answer))
        context_tokens: set = set()
        for chunk in context_chunks:
            context_tokens.update(self._normalize_tokens(chunk.content))

        if not answer_tokens:
            novelty_ratio = 0.0
            novel_tokens = set()
        else:
            novel_tokens = answer_tokens - context_tokens
            novelty_ratio = len(novel_tokens) / max(len(answer_tokens), 1)

        # Retrieval quality uses top-3 mean when available.
        if retrieval_scores:
            sorted_scores = sorted([float(s) for s in retrieval_scores], reverse=True)
            top_scores = sorted_scores[: min(3, len(sorted_scores))]
            retrieval_quality = float(np.mean(top_scores))
        else:
            retrieval_quality = 0.5

        # Risk decision: novelty + retrieval quality.
        if retrieval_quality < 0.20:
            risk = "high"
        elif novelty_ratio > 0.45:
            risk = "high"
        elif novelty_ratio > 0.28 or (retrieval_quality < 0.35 and novelty_ratio > 0.20):
            risk = "medium"
        else:
            risk = "low"

        return {
            "risk": risk,
            "novelty_ratio": round(float(novelty_ratio), 4),
            "answer_token_count": len(answer_tokens),
            "context_token_count": len(context_tokens),
            "novel_token_count": len(novel_tokens),
            "retrieval_quality": round(float(retrieval_quality), 4),
            "sample_novel_tokens": sorted(list(novel_tokens))[:20],
        }
