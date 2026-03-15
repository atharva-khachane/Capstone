"""
FAISS Index Manager with encryption support.

This module provides encrypted vector storage using FAISS for
efficient similarity search with AES-256 encryption.
"""

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

from .schemas import Chunk
from .encryption_manager import EncryptionManager


class FAISSIndexManager:
    """
    Manage FAISS index with encryption.
    
    Features:
    - Flat L2/Inner Product index (exact search)
    - AES-256 encrypted storage
    - Add/search/delete operations
    - Index persistence
    - Metadata tracking
    
    Args:
        index_path: Path to store encrypted index
        embedding_dim: Dimension of embeddings (default: 768)
        metric: Distance metric ('L2' or 'IP' for inner product/cosine)
        encryption_manager: EncryptionManager instance
    """
    
    def __init__(
        self,
        index_path: str,
        embedding_dim: int = 768,
        metric: str = 'IP',  # Inner Product for cosine similarity
        encryption_manager: Optional[EncryptionManager] = None,
    ):
        self.index_path = Path(index_path)
        self.embedding_dim = embedding_dim
        self.metric = metric
        self.encryption_manager = encryption_manager
        
        # Create FAISS index
        if metric == 'IP':
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product (for normalized vectors = cosine sim)
        elif metric == 'L2':
            self.index = faiss.IndexFlatL2(embedding_dim)  # L2 distance
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'IP' or 'L2'")
        
        # Metadata storage: chunk_id -> chunk mapping
        self.chunk_metadata = {}
        
        # FAISS index -> chunk_id mapping
        self.index_to_chunk_id = []
        
        print(f"[FAISS] Created {metric} index with dimension {embedding_dim}")
    
    def _maybe_upgrade_to_ivf(self, n_vectors: int) -> None:
        """Switch to IndexIVFFlat when the corpus exceeds 1000 vectors.

        Per ANTIGRAVITY_PROMPT.md (Layer 2):
          - < 1000 docs: IndexFlatIP (exact search)
          - 1000+ docs: IndexIVFFlat with nlist = sqrt(N)
        """
        if n_vectors < 1000:
            return
        if not isinstance(self.index, faiss.IndexFlat):
            return  # already upgraded

        nlist = max(4, int(np.sqrt(n_vectors)))
        print(f"[FAISS] Upgrading to IndexIVFFlat (nlist={nlist}) for {n_vectors} vectors")

        if self.metric == "IP":
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            ivf = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            ivf = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)

        # Reconstruct existing vectors to train the new index
        if self.index.ntotal > 0:
            existing = np.zeros((self.index.ntotal, self.embedding_dim), dtype=np.float32)
            for i in range(self.index.ntotal):
                existing[i] = self.index.reconstruct(i)
            ivf.train(existing)
            ivf.add(existing)

        self.index = ivf
        print(f"[FAISS] Upgraded to IVFFlat index (nlist={nlist})")

    def add_chunks(self, chunks: List[Chunk]):
        """
        Add chunks to the index.
        
        Args:
            chunks: List of chunks with embeddings
        """
        if not chunks:
            return
        
        # Filter chunks with embeddings
        chunks_with_emb = [c for c in chunks if c.has_embedding()]
        
        if not chunks_with_emb:
            print("[FAISS] No chunks with embeddings to add")
            return
        
        # Extract embeddings
        embeddings = np.array([c.embedding for c in chunks_with_emb], dtype=np.float32)
        
        # Normalize if using IP metric (for cosine similarity)
        if self.metric == 'IP':
            faiss.normalize_L2(embeddings)

        # Auto-upgrade to IVFFlat for large corpora (methodology spec)
        future_total = self.index.ntotal + len(embeddings)
        self._maybe_upgrade_to_ivf(future_total)

        # If IVF index and not yet trained, train it first
        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            self.index.train(embeddings)

        # Add to FAISS index
        self.index.add(embeddings)
        
        # Update metadata
        for chunk in chunks_with_emb:
            self.chunk_metadata[chunk.chunk_id] = chunk
            self.index_to_chunk_id.append(chunk.chunk_id)
        
        print(f"[FAISS] Added {len(chunks_with_emb)} chunks. Total vectors: {self.index.ntotal}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[Chunk, float]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query vector (768-dim)
            top_k: Number of results to return
            
        Returns:
            List of (chunk, score) tuples, sorted by similarity (highest first)
        """
        if self.index.ntotal == 0:
            return []
        
        # Ensure query is 2D array
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = query_embedding.astype(np.float32)
        
        # Normalize if using IP metric
        if self.metric == 'IP':
            faiss.normalize_L2(query_embedding)
        
        # Search
        top_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Convert to results
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for not found
                continue
            
            chunk_id = self.index_to_chunk_id[idx]
            chunk = self.chunk_metadata[chunk_id]
            
            # Convert distance to similarity score
            # For IP: higher is better (cosine similarity)
            # For L2: lower is better (convert to similarity)
            if self.metric == 'IP':
                score = float(distance)  # Already similarity
            else:
                score = 1.0 / (1.0 + float(distance))  # Convert L2 distance to similarity
            
            results.append((chunk, score))
        
        return results
    
    def save(self, encrypt: bool = True):
        """
        Save index to disk (encrypted if encryption manager provided).
        
        Args:
            encrypt: Whether to encrypt the index
        """
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Serialize index
        index_bytes = faiss.serialize_index(self.index)
        
        # Serialize metadata
        metadata = {
            'chunk_metadata': self.chunk_metadata,
            'index_to_chunk_id': self.index_to_chunk_id,
            'embedding_dim': self.embedding_dim,
            'metric': self.metric,
        }
        metadata_bytes = pickle.dumps(metadata)
        
        # Combine
        data = {
            'index': index_bytes,
            'metadata': metadata_bytes,
        }
        data_bytes = pickle.dumps(data)
        
        # Encrypt if requested
        if encrypt and self.encryption_manager:
            print(f"[FAISS] Encrypting index...")
            encrypted_bytes = self.encryption_manager.encrypt_text(data_bytes.decode('latin1'))
            save_path = str(self.index_path) + ".encrypted"
        else:
            encrypted_bytes = data_bytes
            save_path = str(self.index_path)
        
        # Save
        with open(save_path, 'wb') as f:
            if encrypt and self.encryption_manager:
                f.write(encrypted_bytes)  # Already bytes from encryption
            else:
                f.write(encrypted_bytes)
        
        print(f"[FAISS] Saved {'encrypted ' if encrypt else ''}index to {save_path}")
        print(f"[FAISS] Total vectors: {self.index.ntotal}, Size: {len(data_bytes) / 1024:.1f}KB")
    
    def load(self, encrypted: bool = True):
        """
        Load index from disk.
        
        Args:
            encrypted: Whether the index is encrypted
        """
        load_path = str(self.index_path) + (".encrypted" if encrypted else "")
        
        if not Path(load_path).exists():
            raise FileNotFoundError(f"Index not found: {load_path}")
        
        # Load
        with open(load_path, 'rb') as f:
            data_bytes = f.read()
        
        # Decrypt if necessary
        if encrypted and self.encryption_manager:
            print(f"[FAISS] Decrypting index...")
            data_bytes = self.encryption_manager.decrypt_text(data_bytes).encode('latin1')
        
        # Deserialize
        data = pickle.loads(data_bytes)
        index_bytes = data['index']
        metadata_bytes = data['metadata']
        
        # Restore index
        self.index = faiss.deserialize_index(index_bytes)
        
        # Restore metadata
        metadata = pickle.loads(metadata_bytes)
        self.chunk_metadata = metadata['chunk_metadata']
        self.index_to_chunk_id = metadata['index_to_chunk_id']
        self.embedding_dim = metadata['embedding_dim']
        self.metric = metadata['metric']
        
        print(f"[FAISS] Loaded {'encrypted ' if encrypted else ''}index from {load_path}")
        print(f"[FAISS] Total vectors: {self.index.ntotal}")
    
    def get_statistics(self) -> dict:
        """Get index statistics."""
        return {
            'total_vectors': self.index.ntotal,
            'embedding_dim': self.embedding_dim,
            'metric': self.metric,
            'total_chunks': len(self.chunk_metadata),
            'index_path': str(self.index_path),
        }
    
    def clear(self):
        """Clear the index."""
        if self.metric == 'IP':
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        self.chunk_metadata.clear()
        self.index_to_chunk_id.clear()
        
        print(f"[FAISS] Index cleared")
