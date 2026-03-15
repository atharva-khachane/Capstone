"""
Embedding Generator using sentence-transformers.

This module provides GPU-accelerated embedding generation using
the all-mpnet-base-v2 model (768-dimensional embeddings).
"""

import numpy as np
from typing import List, Union
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError:
    raise ImportError(
        "sentence-transformers and torch required. Install with: pip install sentence-transformers torch"
    )

from .schemas import Chunk


class EmbeddingGenerator:
    """
    Generate embeddings using sentence-transformers.
    
    Features:
    - GPU-accelerated inference
    - Batch processing for efficiency
    - L2 normalization for cosine similarity
    - Progress tracking
    - Model caching
    
    Args:
        model_name: HuggingFace model name (default: all-mpnet-base-v2)
        use_gpu: Whether to use GPU if available
        batch_size: Batch size for encoding
        show_progress: Show progress bar
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        use_gpu: bool = True,
        batch_size: int = 32,
        show_progress: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.show_progress = show_progress
        
        # Determine device
        if use_gpu and torch.cuda.is_available():
            self.device = "cuda"
            print(f"[EMBEDDING] Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"[EMBEDDING] VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            self.device = "cpu"
            print(f"[EMBEDDING] Using CPU (GPU not available or disabled)")
        
        # Load model
        print(f"[EMBEDDING] Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"[EMBEDDING] Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        
        # Cache for statistics
        self.total_embeddings_generated = 0
    
    def generate_embeddings(
        self, 
        chunks: List[Chunk],
        normalize: bool = True
    ) -> List[Chunk]:
        """
        Generate embeddings for chunks.
        
        Args:
            chunks: List of chunks to embed
            normalize: Whether to L2-normalize embeddings (for cosine similarity)
            
        Returns:
            Chunks with embeddings added
        """
        if not chunks:
            return []
        
        # Extract texts
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        print(f"[EMBEDDING] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        
        # Ensure float32
        embeddings = embeddings.astype(np.float32)
        
        # Assign embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        self.total_embeddings_generated += len(chunks)
        
        print(f"[EMBEDDING] [OK] Generated {len(chunks)} embeddings (total: {self.total_embeddings_generated})")
        
        return chunks
    
    def generate_query_embedding(
        self,
        query: str,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text
            normalize: Whether to L2-normalize
            
        Returns:
            Query embedding (768-dim numpy array)
        """
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        
        return embedding.astype(np.float32)
    
    def generate_batch_embeddings(
        self,
        texts: List[str],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of text strings
            normalize: Whether to L2-normalize
            
        Returns:
            Array of embeddings (N x 768)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        
        return embeddings.astype(np.float32)
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
    
    def get_device_info(self) -> dict:
        """Get information about the device being used."""
        info = {
            'device': self.device,
            'model': self.model_name,
            'embedding_dim': self.get_embedding_dimension(),
            'total_generated': self.total_embeddings_generated,
        }
        
        if self.device == "cuda":
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['vram_total_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            info['vram_allocated_gb'] = torch.cuda.memory_allocated(0) / 1024**3
        
        return info
