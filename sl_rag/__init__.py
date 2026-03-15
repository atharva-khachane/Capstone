"""
SL-RAG: Secure and Accurate Retrieval-Augmented Generation Pipeline
=====================================================================

A production-ready RAG system for ISRO's sensitive government documentation
with maximum security, 100% offline operation, and comprehensive audit trails.

Features:
- AES-256 encryption for all stored data
- PII anonymization (India-specific patterns)
- GPU-accelerated embeddings (all-mpnet-base-v2)
- Hybrid retrieval (BM25 + Dense) with cross-encoder re-ranking
- LLM answer generation (Llama 3.2 3B, 4-bit quantized)
- Tamper-evident audit logging
- Content-based domain clustering

Author: Capstone Project
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Capstone Project"

# Import main schemas for easy access
from .core.schemas import Document, Chunk

__all__ = [
    "Document",
    "Chunk",
    "__version__",
]
