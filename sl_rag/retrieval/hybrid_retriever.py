"""
Hybrid Retriever combining BM25 (sparse) and Dense (FAISS) retrieval.

This module implements a hybrid search strategy that combines:
- BM25: Keyword-based sparse retrieval (good for exact matches)
- Dense: Semantic embedding-based retrieval (good for meaning)

The results are fused using weighted combination (Reciprocal Rank Fusion).
"""

import numpy as np
from typing import FrozenSet, List, Optional, Tuple
from collections import defaultdict

from ..core.schemas import Chunk
from ..core.faiss_index import FAISSIndexManager
from ..core.embedding_generator import EmbeddingGenerator
from .bm25_retriever import BM25Retriever


class HybridRetriever:
    """
    Hybrid retrieval combining BM25 and dense search.
    
    Fusion Strategy:
    - Retrieves top-k candidates from both BM25 and Dense
    - Combines scores using weighted average or Reciprocal Rank Fusion (RRF)
    - Re-ranks combined results
    
    Args:
        bm25_retriever: BM25Retriever instance
        faiss_index: FAISSIndexManager instance
        embedding_generator: EmbeddingGenerator instance
        alpha: Weight for dense retrieval (1-alpha for BM25)
               alpha=0.7 means 70% dense, 30% BM25
        fusion_method: 'weighted' or 'rrf' (Reciprocal Rank Fusion)
    """
    
    # Domains with heavy technical vocabulary where BM25 is unreliable.
    # When a query is filtered to one of these domains, alpha is boosted to
    # tech_domain_alpha (default 0.95) for near-pure dense retrieval.
    DEFAULT_TECH_DOMAINS: frozenset = frozenset({"telemetry", "technical_report"})

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        faiss_index: FAISSIndexManager,
        embedding_generator: EmbeddingGenerator,
        alpha: float = 0.7,
        fusion_method: str = 'weighted',
        enable_domain_filtering: bool = True,
        tech_domain_alpha: float = 0.95,
        tech_domains: Optional[List[str]] = None,
    ):
        self.bm25_retriever = bm25_retriever
        self.faiss_index = faiss_index
        self.embedding_generator = embedding_generator
        self.alpha = alpha  # Weight for dense retrieval (Fix 5: configurable via bm25_alpha_param)
        self.fusion_method = fusion_method
        self.enable_domain_filtering = enable_domain_filtering
        self.tech_domain_alpha = tech_domain_alpha
        self.tech_domains: frozenset = (
            frozenset(tech_domains) if tech_domains else self.DEFAULT_TECH_DOMAINS
        )

        # Build domain-specific indices for fast filtering
        self.domain_to_chunk_ids = {}
        if enable_domain_filtering:
            self._build_domain_indices()

        print(f"[HYBRID] Initialized with alpha={alpha} ({int(alpha*100)}% dense, {int((1-alpha)*100)}% BM25)")
        print(f"[HYBRID] Fusion method: {fusion_method}")
        print(f"[HYBRID] Tech domains: {sorted(self.tech_domains)} → alpha={tech_domain_alpha}")
        if enable_domain_filtering:
            print(f"[HYBRID] Domain filtering: ENABLED ({len(self.domain_to_chunk_ids)} domains)")
    
    def _build_domain_indices(self):
        """Create per-domain chunk indices for fast filtering."""
        # Get all chunks from BM25 retriever
        for chunk in self.bm25_retriever.chunks:
            domain = chunk.domain
            if domain:
                if domain not in self.domain_to_chunk_ids:
                    self.domain_to_chunk_ids[domain] = set()
                self.domain_to_chunk_ids[domain].add(chunk.chunk_id)

    
    def search(
        self,
        query: str,
        top_k: int = 10,
        top_k_candidates: int = 20,
        filter_domains: Optional[List[str]] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Hybrid search combining BM25 and dense retrieval.
        
        Args:
            query: Query string
            top_k: Final number of results to return
            top_k_candidates: Number of candidates to retrieve from each method
            filter_domains: Optional list of domains to search (if None, search all)
            
        Returns:
            List of (chunk, score) tuples, sorted by fused score
        """
        # Determine which chunk IDs to search
        valid_chunk_ids = None
        if filter_domains and self.enable_domain_filtering:
            valid_chunk_ids = set()
            for domain in filter_domains:
                valid_chunk_ids.update(self.domain_to_chunk_ids.get(domain, set()))

            if not valid_chunk_ids:
                print(f"[HYBRID] WARNING: No chunks found in domains: {filter_domains}")
                return []

            print(f"[HYBRID] Filtering to {len(valid_chunk_ids)} chunks in domains: {filter_domains}")

        # Fix 2: Use dense-heavy alpha for technical vocabulary domains to avoid
        # BM25 vocabulary mismatch on telemetry / technical-report queries.
        effective_alpha = self.alpha
        if filter_domains and self.tech_domains.intersection(filter_domains):
            effective_alpha = self.tech_domain_alpha
            print(
                f"[HYBRID] Tech domain detected {filter_domains} — "
                f"boosting alpha {self.alpha} → {effective_alpha}"
            )

        # 1. BM25 retrieval
        bm25_results = self.bm25_retriever.search(query, top_k=top_k_candidates)

        # 2. Dense retrieval
        query_embedding = self.embedding_generator.generate_query_embedding(query, normalize=True)
        dense_results = self.faiss_index.search(query_embedding, top_k=top_k_candidates)
        
        # 3. Filter by domain if specified
        if valid_chunk_ids:
            bm25_results = [(c, s) for c, s in bm25_results if c.chunk_id in valid_chunk_ids]
            dense_results = [(c, s) for c, s in dense_results if c.chunk_id in valid_chunk_ids]
        
        # 4. Fuse results (using effective_alpha which may be boosted for tech domains)
        if self.fusion_method == 'rrf':
            fused_results = self._reciprocal_rank_fusion(
                bm25_results, dense_results, effective_alpha=effective_alpha
            )
        else:  # weighted
            fused_results = self._weighted_fusion(
                bm25_results, dense_results, effective_alpha=effective_alpha
            )
        
        # 5. Return top-k
        return fused_results[:top_k]
    
    def _weighted_fusion(
        self,
        bm25_results: List[Tuple[Chunk, float]],
        dense_results: List[Tuple[Chunk, float]],
        effective_alpha: Optional[float] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Fuse results using weighted score combination.
        
        Normalizes scores from each method to [0, 1] range,
        then combines using alpha weighting.
        """
        # Normalize BM25 scores
        bm25_scores = {}
        if bm25_results:
            bm25_score_values = [score for _, score in bm25_results]
            max_bm25 = max(bm25_score_values) if bm25_score_values else 1.0
            min_bm25 = min(bm25_score_values) if bm25_score_values else 0.0
            
            for chunk, score in bm25_results:
                # Min-max normalization
                if max_bm25 > min_bm25:
                    normalized_score = (score - min_bm25) / (max_bm25 - min_bm25)
                else:
                    normalized_score = 1.0 if score > 0 else 0.0
                bm25_scores[chunk.chunk_id] = normalized_score
        
        # Normalize Dense scores (already in [0, 1] range for cosine similarity)
        dense_scores = {}
        for chunk, score in dense_results:
            # Cosine similarity is already in [-1, 1], typically [0, 1] for our use case
            dense_scores[chunk.chunk_id] = max(0.0, min(1.0, score))
        
        # Combine scores
        _alpha = effective_alpha if effective_alpha is not None else self.alpha
        all_chunk_ids = set(bm25_scores.keys()) | set(dense_scores.keys())
        combined_scores = {}
        chunk_map = {}
        
        for chunk, _ in bm25_results:
            chunk_map[chunk.chunk_id] = chunk
        for chunk, _ in dense_results:
            chunk_map[chunk.chunk_id] = chunk

        for chunk_id in all_chunk_ids:
            bm25_score = bm25_scores.get(chunk_id, 0.0)
            dense_score = dense_scores.get(chunk_id, 0.0)

            # Weighted combination (alpha = dense weight, 1-alpha = BM25 weight)
            combined_score = _alpha * dense_score + (1 - _alpha) * bm25_score
            combined_scores[chunk_id] = combined_score
        
        # Sort by combined score
        sorted_chunks = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Create result list
        results = [(chunk_map[chunk_id], score) for chunk_id, score in sorted_chunks]
        
        return results
    
    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Tuple[Chunk, float]],
        dense_results: List[Tuple[Chunk, float]],
        k: int = 60,
        effective_alpha: Optional[float] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Fuse results using Reciprocal Rank Fusion (RRF).
        
        RRF formula: score(d) = SUM[1 / (k + rank(d))]
        where k is a constant (usually 60) and rank is 1-indexed.
        """
        _alpha = effective_alpha if effective_alpha is not None else self.alpha
        rrf_scores = defaultdict(float)
        chunk_map = {}

        # Add BM25 results
        for rank, (chunk, _) in enumerate(bm25_results, start=1):
            rrf_scores[chunk.chunk_id] += (1 - _alpha) / (k + rank)
            chunk_map[chunk.chunk_id] = chunk

        # Add Dense results
        for rank, (chunk, _) in enumerate(dense_results, start=1):
            rrf_scores[chunk.chunk_id] += _alpha / (k + rank)
            chunk_map[chunk.chunk_id] = chunk
        
        # Sort by RRF score
        sorted_chunks = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Create result list
        results = [(chunk_map[chunk_id], score) for chunk_id, score in sorted_chunks]
        
        return results
    
    def get_statistics(self) -> dict:
        """Get statistics from both retrievers."""
        return {
            'bm25': self.bm25_retriever.get_statistics(),
            'dense': self.faiss_index.get_statistics(),
            'alpha': self.alpha,
            'fusion_method': self.fusion_method,
        }
