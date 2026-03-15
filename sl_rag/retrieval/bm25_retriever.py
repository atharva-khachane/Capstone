"""
BM25 Retriever for sparse keyword-based retrieval.

BM25 (Best Matching 25) is a probabilistic ranking function used for
keyword-based search. It complements dense retrieval by finding lexical matches.
"""

import numpy as np
from typing import List, Tuple
import re

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("rank-bm25 required. Install with: pip install rank-bm25")

from ..core.schemas import Chunk


class BM25Retriever:
    """
    BM25 sparse retrieval for keyword-based search.
    
    Features:
    - BM25Okapi algorithm (improved BM25)
    - Simple tokenization
    - Efficient keyword matching
    - Complements dense retrieval
    
    Args:
        chunks: List of chunks to index
        k1: BM25 parameter controlling term frequency saturation (default: 1.5)
        b: BM25 parameter controlling length normalization (default: 0.75)
    """
    
    def __init__(
        self,
        chunks: List[Chunk] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.k1 = k1
        self.b = b
        self.chunks = []
        self.bm25 = None
        self.tokenized_corpus = []
        
        if chunks:
            self.index_chunks(chunks)
    
    def index_chunks(self, chunks: List[Chunk]):
        """
        Index chunks for BM25 retrieval.
        
        Args:
            chunks: List of chunks to index
        """
        self.chunks = chunks
        
        # Tokenize all chunks
        self.tokenized_corpus = [self._tokenize(chunk.content) for chunk in chunks]
        
        # Create BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        
        print(f"[BM25] Indexed {len(chunks)} chunks")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[Chunk, float]]:
        """
        Search for chunks using BM25.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of (chunk, score) tuples, sorted by BM25 score (highest first)
        """
        if not self.bm25:
            return []
        
        # Tokenize query
        tokenized_query = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_k = min(top_k, len(self.chunks))
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Create results
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            score = float(scores[idx])
            results.append((chunk, score))
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for BM25.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens (lowercase words)
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters, keep only alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Split on whitespace
        tokens = text.split()
        
        # Remove very short tokens (< 2 chars)
        tokens = [t for t in tokens if len(t) >= 2]
        
        return tokens
    
    def get_statistics(self) -> dict:
        """Get BM25 index statistics."""
        if not self.bm25:
            return {
                'total_chunks': 0,
                'avg_doc_length': 0,
                'k1': self.k1,
                'b': self.b,
            }
        
        doc_lengths = [len(doc) for doc in self.tokenized_corpus]
        
        return {
            'total_chunks': len(self.chunks),
            'avg_doc_length': np.mean(doc_lengths) if doc_lengths else 0,
            'min_doc_length': min(doc_lengths) if doc_lengths else 0,
            'max_doc_length': max(doc_lengths) if doc_lengths else 0,
            'k1': self.k1,
            'b': self.b,
        }
