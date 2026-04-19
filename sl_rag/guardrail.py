import logging
import re


INJECTION_PATTERNS = [
    r"ignore (?:(?:all\s+)?(?:previous|above)|all) instructions",
    r"you are now",
    r"pretend (you are|to be)",
    r"disregard (your|all)",
    r"as an? (AI|LLM|assistant|model) without (restrictions|guidelines|filters)",
    r"jailbreak",
    r"DAN mode",
    r"do anything now",
    r"forget your (instructions|guidelines|rules)",
    r"override (your|the) (instructions|guidelines|system)",
    r"reveal (your|the) (system|instructions|prompt)",
    r"act as if you have no (restrictions|guidelines)",
]


SAFE_FALLBACK = (
    "I can only answer questions based on the retrieved documents. "
    "Please ask a question about the available knowledge base."
)


def is_adversarial(query: str) -> bool:
    return any(re.search(pattern, query, re.IGNORECASE) for pattern in INJECTION_PATTERNS)


def check_and_gate(query: str) -> str | None:
    """
    Returns SAFE_FALLBACK string if adversarial, None if safe.
    Usage: blocked = check_and_gate(query); if blocked: return blocked
    """
    if is_adversarial(query):
        logging.warning(f"[GUARDRAIL] Adversarial query blocked: {query[:80]}")
        return SAFE_FALLBACK
    return None
