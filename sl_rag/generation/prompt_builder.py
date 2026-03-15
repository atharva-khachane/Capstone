"""
Prompt Builder with injection detection (Layer 4, Component 4.1).

Per ANTIGRAVITY_PROMPT.md Phase 4:
  - Strict hierarchy: SYSTEM -> CONTEXT -> QUERY -> ANSWER
  - Prompt injection detection (blocks boundary markers, override attempts)
  - Token limit enforcement for context window
  - Citation formatting: [Source: doc_id, Chunk: index]

100% OFFLINE - No external APIs.
"""

import re
from typing import List, Tuple, Optional

from ..core.schemas import Chunk


INJECTION_PATTERNS = [
    r"ignore\s+(previous|above|prior|all)\s+(instructions|directions|rules|prompts)",
    r"disregard\s+(previous|above|prior|all)\s+(instructions|directions|rules)",
    r"forget\s+(previous|above|your)\s+(instructions|rules|training)",
    r"forget\s+your\s+previous\s+(instructions|rules|training)",
    r"you\s+are\s+now\s+",
    r"new\s+instructions?\s*:",
    r"override\s+(instructions?|rules?|system)",
    r"system\s*:",
    r"SYSTEM\s*:",
    r"###",
    r"---\s*\n",
    r"^---$",
    r"<\|.*?\|>",
    r"\[INST\]",
    r"\[/INST\]",
    r"</?s>",
    r"<<SYS>>",
    r"<</SYS>>",
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


class PromptBuilder:
    """Build prompts with hierarchical structure and injection protection.

    Hierarchy (immutable):
        1. SYSTEM  -- role definition, constraints, behaviour rules
        2. CONTEXT -- retrieved chunks with citation markers
        3. QUERY   -- user question
        4. ANSWER  -- generation target marker
    """

    SYSTEM_PROMPT = (
        "You are a precise question-answering assistant for ISRO government documentation. "
        "Your responses must:\n"
        "1. ONLY use information from the provided context\n"
        "2. CITE sources using [Source: doc_id, Chunk: chunk_index]\n"
        "3. If information is not in context, say "
        '"I cannot answer based on the provided documents"\n'
        "4. Be concise and factual\n"
        "5. Never speculate or add external knowledge\n"
        "6. If multiple sources agree, cite all of them"
    )

    def __init__(
        self,
        max_context_tokens: int = 6000,
        words_per_token: float = 0.75,
    ):
        self.max_context_tokens = max_context_tokens
        self.words_per_token = words_per_token

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        query: str,
        retrieved_chunks: List[Tuple[Chunk, float]],
    ) -> str:
        """Construct the full prompt.

        Returns the assembled prompt string. If the query contains an
        injection attempt, a sanitised refusal prompt is returned instead.
        """
        if self.detect_injection(query):
            return self._build_refusal_prompt(query)

        context_block = self._format_context(retrieved_chunks)

        prompt = (
            f"SYSTEM:\n{self.SYSTEM_PROMPT}\n\n"
            f"CONTEXT:\n{context_block}\n\n"
            f"QUERY: {query}\n\n"
            "ANSWER:\n"
        )
        return prompt

    def detect_injection(self, query: str) -> bool:
        """Return True if query contains prompt injection patterns."""
        for pattern in COMPILED_PATTERNS:
            if pattern.search(query):
                return True

        special_char_ratio = sum(
            1 for c in query if not c.isalnum() and c not in " .,?!'-/()"
        ) / max(len(query), 1)
        if special_char_ratio > 0.25:
            return True

        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_context(
        self, chunks: List[Tuple[Chunk, float]],
    ) -> str:
        max_words = int(self.max_context_tokens * self.words_per_token)
        word_count = 0
        parts: list = []

        for idx, (chunk, score) in enumerate(chunks, start=1):
            header = (
                f"[{idx}] (Score: {score:.2f}, "
                f"Source: {chunk.doc_id[:12]}, "
                f"Chunk: {chunk.chunk_index})"
            )
            content = chunk.content.strip()
            entry_words = len(content.split())

            if word_count + entry_words > max_words:
                remaining = max_words - word_count
                if remaining > 50:
                    truncated = " ".join(content.split()[:remaining]) + " [truncated]"
                    parts.append(f"{header}\n{truncated}")
                break

            parts.append(f"{header}\n{content}")
            word_count += entry_words

        if not parts:
            return "[No relevant context found]"
        return "\n\n".join(parts)

    def _build_refusal_prompt(self, query: str) -> str:
        safe_query = re.sub(r"[^\w\s?.,'\"/-]", "", query)[:200]
        return (
            f"SYSTEM:\n{self.SYSTEM_PROMPT}\n\n"
            "CONTEXT:\n[No context provided - query flagged for injection]\n\n"
            f"QUERY: {safe_query}\n\n"
            "ANSWER:\nI cannot process this query as it contains patterns "
            "that may compromise the system's integrity.\n"
        )
