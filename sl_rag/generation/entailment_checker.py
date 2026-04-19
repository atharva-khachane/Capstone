"""
Sentence-Level Entailment Filter (Fix 2: hallucination reduction).

For each sentence in the generated answer, scores it against the best
retrieved context chunk using a cross-encoder model.  Sentences that
score below the threshold AND exceed a minimum word count are stripped
from the answer.

The cross-encoder already loaded by the reranker (ms-marco-MiniLM-L-6-v2)
is reused — no extra model load.
"""

import math
import re
from typing import List, Tuple

try:
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import sent_tokenize
    _nltk_ok = True
except ImportError:
    _nltk_ok = False

from ..core.schemas import Chunk


_REFUSAL_RE = re.compile(
    r"cannot answer|cannot assist|cannot fulfill|cannot process|cannot provide"
    r"|not (?:available )?in the (?:provided )?(?:context|documents?)"
    r"|no (?:relevant )?information (?:available|found)"
    r"|cannot be (?:answered|determined|verified)"
    r"|outside (?:the )?(?:scope|provided context)"
    r"|I don'?t have (?:enough|sufficient|the) (?:information|data|context)",
    re.IGNORECASE,
)

_MD_LIST_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+\.\s+)", re.MULTILINE)


def _looks_like_markdown(text: str) -> bool:
    if not text:
        return False
    # Preserve formatting when the answer likely uses Markdown structure.
    return (
        "\n" in text
        and (
            _MD_LIST_RE.search(text) is not None
            or "```" in text
            or "\n#" in text
        )
    )


class EntailmentChecker:
    """Filter answer sentences not supported by retrieved context."""

    def __init__(self, cross_encoder_model, threshold: float = 0.20, min_words: int = 8):
        """
        Args:
            cross_encoder_model: A sentence_transformers.CrossEncoder instance.
            threshold: Sigmoid score below which a sentence is flagged.
            min_words: Sentences with fewer words skip the check.
        """
        self.model = cross_encoder_model
        self.threshold = threshold
        self.min_words = min_words

    def check_and_filter(
        self,
        answer: str,
        context_chunks: List[Chunk],
    ) -> Tuple[str, int]:
        """Score each sentence against context; strip unsupported ones.

        Returns:
            (filtered_answer, num_flagged_sentences)
            The filtered answer does NOT include the warning prefix —
            the caller should prepend the warning if num_flagged > 1.
        """
        if not answer or not context_chunks:
            return answer, 0

        chunk_texts = [c.content for c in context_chunks]
        if not chunk_texts:
            return answer, 0

        # Markdown answers (lists/code/headers) must keep line breaks;
        # sentence-tokenizing + re-joining with spaces destroys indentation.
        if _looks_like_markdown(answer):
            flagged = 0
            kept_lines: List[str] = []
            in_code_block = False

            for line in answer.splitlines():
                stripped = line.strip()

                # Preserve code fences verbatim and skip filtering inside.
                if stripped.startswith("```"):
                    in_code_block = not in_code_block
                    kept_lines.append(line)
                    continue
                if in_code_block:
                    kept_lines.append(line)
                    continue

                # Keep blank lines to preserve paragraph spacing.
                if not stripped:
                    kept_lines.append(line)
                    continue

                # Strip leading list markers for scoring, but keep original line if kept.
                scorable = re.sub(r"^\s*(?:[-*+]\s+|\d+\.\s+)", "", line).strip()
                if len(scorable.split()) < self.min_words or _REFUSAL_RE.search(scorable):
                    kept_lines.append(line)
                    continue

                pairs = [[scorable, ct] for ct in chunk_texts]
                raw_scores = self.model.predict(pairs, show_progress_bar=False)
                if hasattr(raw_scores, "tolist"):
                    raw_scores = raw_scores.tolist()

                best = max(
                    (1.0 / (1.0 + math.exp(-s)) for s in raw_scores),
                    default=0.0,
                )

                if best < self.threshold:
                    flagged += 1
                    continue

                kept_lines.append(line)

            filtered = "\n".join(kept_lines).strip()
            return filtered, flagged

        sentences = sent_tokenize(answer) if _nltk_ok else re.split(r"(?<=[.!?])\s+", answer)
        if not sentences:
            return answer, 0

        flagged = 0
        kept: List[str] = []

        for sent in sentences:
            if len(sent.split()) < self.min_words or _REFUSAL_RE.search(sent):
                kept.append(sent)
                continue

            pairs = [[sent, ct] for ct in chunk_texts]
            raw_scores = self.model.predict(pairs, show_progress_bar=False)
            if hasattr(raw_scores, 'tolist'):
                raw_scores = raw_scores.tolist()

            best = max(
                (1.0 / (1.0 + math.exp(-s)) for s in raw_scores),
                default=0.0,
            )

            if best < self.threshold:
                flagged += 1
                continue

            kept.append(sent)

        return " ".join(kept), flagged
