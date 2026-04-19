"""
Cross-Encoder Re-ranker for final result refinement.

Cross-encoders process query and document together (unlike bi-encoders)
and provide highly accurate relevance scores. They're slower but more
accurate, so we use them for re-ranking top candidates only.
"""

import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    raise ImportError("sentence-transformers required")

from ..core.schemas import Chunk


class CrossEncoderReranker:
    """
    Re-rank retrieval results using a cross-encoder model.
    
    Cross-encoders are more accurate than bi-encoders but slower,
    so they're best used for re-ranking a small set of candidates.
    
    Features:
    - Cross-encoder model (ms-marco-MiniLM)
    - GPU acceleration
    - Batch processing
    - Score normalization
    
    Args:
        model_name: HuggingFace cross-encoder model
        use_gpu: Whether to use GPU
        batch_size: Batch size for re-ranking
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_gpu: bool = True,
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Determine device
        import torch
        if use_gpu and torch.cuda.is_available():
            self.device = "cuda"
            print(f"[RERANKER] Using GPU for cross-encoder")
        else:
            self.device = "cpu"
            print(f"[RERANKER] Using CPU for cross-encoder")
        
        # Load model
        print(f"[RERANKER] Loading cross-encoder: {model_name}...")
        self.model = CrossEncoder(model_name, device=self.device)
        print(f"[RERANKER] Cross-encoder loaded")
    
    def rerank(
        self,
        query: str,
        results: List[Tuple[Chunk, float]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Re-rank results using cross-encoder.
        
        Args:
            query: Query string
            results: List of (chunk, score) tuples to re-rank
            top_k: Optional number of top results to return (default: all)
            
        Returns:
            Re-ranked list of (chunk, score) tuples
        """
        if not results:
            return []
        
        # Extract chunks
        chunks = [chunk for chunk, _ in results]
        
        # Create query-document pairs
        pairs = [[query, chunk.content] for chunk in chunks]
        
        # Get cross-encoder scores
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        
        # Convert to list if numpy array
        if isinstance(scores, np.ndarray):
            scores = scores.tolist()

        # ms-marco cross-encoders output raw logits (~[-6, +6]).
        # Apply sigmoid to map to a probability-like range.
        import math
        scores = [1.0 / (1.0 + math.exp(-s)) for s in scores]

        # After sigmoid, relevant passages cluster in [0.65, 0.92], making
        # avg_top3 non-discriminating in the confidence formula.  Min-max
        # normalization within this result set stretches the scores to [0, 1]
        # so the spread between the best and worst candidates is preserved.
        #
        # Minimum range guard: only normalize when the actual score spread is
        # at least 0.05.  If all logits are uniformly high (e.g., every chunk
        # is genuinely relevant to a clear query), the sigmoid outputs cluster
        # within <0.05 of each other and normalization would amplify numerical
        # noise into spurious [0, 1] extremes, producing false confidence
        # scores near 1.0.  In that case, keep the raw sigmoid values so the
        # confidence formula reflects the actual (high but not perfect) retrieval.
        _MIN_NORM_RANGE = 0.05
        if len(scores) > 1:
            mn, mx = min(scores), max(scores)
            if mx - mn >= _MIN_NORM_RANGE:
                scores = [(s - mn) / (mx - mn) for s in scores]

        # Create re-ranked results
        reranked = [(chunk, float(score)) for chunk, score in zip(chunks, scores)]
        
        # Sort by cross-encoder score
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k if specified
        if top_k:
            reranked = reranked[:top_k]
        
        return reranked
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'batch_size': self.batch_size,
        }
