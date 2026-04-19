"""Rule-based confidence scoring.

This replaces earlier Platt scaling (which overfit on a tiny sample and
collapsed confidence values). It computes a deterministic confidence score
from signals already produced by the pipeline.

Important: confidence should reflect *semantic* answer quality and grounding
(e.g., BERTScore / entailment / validation signals), not surface-form overlap
metrics like ROUGE-L that can penalize correct paraphrases.

Signals (all expected in [0, 1]):
    retrieval_score   – top-1 reranked similarity
    groundedness      – faithfulness / entailment
    context_precision – answer-context consistency
    context_recall    – coverage of relevant context (optional)
"""

from typing import List


def compute_rule_based_confidence(
    retrieval_score: float,
    groundedness: float,
    context_precision: float,
    context_recall: float = 0.0,
) -> float:
    """Compute confidence from pipeline signals without any fitted model.

    This is intended to behave like a *genuine* confidence score: when
    retrieval is strong and the answer is grounded/consistent, confidence
    should be high (no artificial ceiling at 0.50).

    Args:
        retrieval_score: Top-1 retrieval similarity (0-1).
        groundedness: Faithfulness score from validation (0-1).
        context_precision: Consistency score from validation (0-1).
        context_recall: Coverage score (0-1). Optional; defaults to 0.0.

    Returns:
        Confidence score clamped to [0.05, 0.95].
    """
    if retrieval_score <= 0:
        return 0.05

    base = (
        0.40 * groundedness
        + 0.25 * context_precision
        + 0.20 * retrieval_score
        + 0.15 * context_recall
    )
    return round(min(max(base, 0.05), 0.95), 3)


def compute_ece(confs: List[float], accuracies: List[float], n_bins: int = 5) -> float:
    """Expected Calibration Error with equal-width bins.

    Kept as a standalone helper so eval_extended.py can compute ECE without
    importing any fitting / sklearn machinery.
    """
    if not confs:
        return 0.0
    bin_size = 1.0 / n_bins
    ece_sum = 0.0
    for b in range(n_bins):
        lo, hi = b * bin_size, (b + 1) * bin_size
        idx = [i for i, c in enumerate(confs) if lo <= c < hi]
        if not idx:
            continue
        avg_conf = sum(confs[i] for i in idx) / len(idx)
        avg_acc = sum(accuracies[i] for i in idx) / len(idx)
        ece_sum += len(idx) * abs(avg_conf - avg_acc)
    return round(ece_sum / len(confs), 4)
