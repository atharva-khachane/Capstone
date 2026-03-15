"""
Domain Manager for intelligent document domain assignment and routing.

Implements content-based K-means clustering for automatic domain detection,
domain centroids computation, and query-to-domain routing.

100% OFFLINE - No external models or APIs.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ..core.schemas import Chunk


class DomainManager:
    """
    Content-based domain manager using K-means clustering.
    
    Strategy:
    1. Cluster document chunks using K-means on embeddings
    2. Auto-tune K using silhouette score
    3. Assign domain labels to each cluster
    4. Enable multi-domain query routing
    """
    
    def __init__(self, 
                 min_clusters: int = 3,
                 max_clusters: int = 8,
                 auto_tune_clusters: bool = True,
                 enable_multi_domain_routing: bool = True):
        """
        Initialize domain manager.
        
        Args:
            min_clusters: Min clusters for K-means
            max_clusters: Max clusters for K-means
            auto_tune_clusters: Automatically find optimal K using silhouette score
            enable_multi_domain_routing: Enable multi-domain query routing
        """
        self.domains: Dict[str, np.ndarray] = {}
        self.domain_chunks: Dict[str, List[str]] = {}
        self.cluster_labels: Dict[str, int] = {}
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.auto_tune_clusters = auto_tune_clusters
        self.enable_multi_domain_routing = enable_multi_domain_routing
    
    def detect_domains(self, chunks: List[Chunk]) -> Dict:
        """
        Detect domains using content-based K-means clustering.
        
        Algorithm:
        1. Extract embeddings from all chunks
        2. Auto-tune K (number of clusters) using silhouette score
        3. Run K-means clustering
        4. Assign domain labels (domain_0, domain_1, ...)
        5. Compute domain centroids
        
        Args:
            chunks: List of chunks with embeddings
            
        Returns:
            Dict with clustering results and statistics
        """
        print("[DOMAIN] Using content-based K-means clustering...")
        
        # Validate embeddings exist
        chunks_with_embeddings = [c for c in chunks if c.has_embedding()]
        if not chunks_with_embeddings:
            raise ValueError("No chunks with embeddings found")
        
        if len(chunks_with_embeddings) < self.min_clusters:
            print(f"[DOMAIN] WARNING: Only {len(chunks_with_embeddings)} chunks, using 1 cluster")
            # Assign all to single domain
            for chunk in chunks:
                chunk.domain = "domain_0"
                self.cluster_labels[chunk.chunk_id] = 0
            
            self.compute_centroids(chunks)
            
            return {
                'num_domains': 1,
                'optimal_k': 1,
                'silhouette_score': 0.0,
                'method': 'content_based_kmeans',
                'domain_distribution': {'domain_0': len(chunks)}
            }
        
        # Extract embeddings
        embeddings = np.array([c.embedding for c in chunks_with_embeddings])
        
        # Auto-tune K if enabled
        if self.auto_tune_clusters:
            optimal_k, silhouette = self._find_optimal_k(embeddings)
        else:
            optimal_k = min(self.max_clusters, len(chunks_with_embeddings))
            silhouette = 0.0
        
        print(f"[DOMAIN] Using K={optimal_k} clusters (silhouette={silhouette:.3f})")
        
        # Cluster embeddings
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # Assign domain labels to chunks
        domain_counts = {}
        chunk_idx = 0
        for chunk in chunks:
            if not chunk.has_embedding():
                # Assign to nearest cluster based on content similarity
                # For now, assign to domain_0
                chunk.domain = "domain_0"
                self.cluster_labels[chunk.chunk_id] = 0
            else:
                cluster_id = int(labels[chunk_idx])
                domain_name = f"domain_{cluster_id}"
                chunk.domain = domain_name
                self.cluster_labels[chunk.chunk_id] = cluster_id
                
                if domain_name not in domain_counts:
                    domain_counts[domain_name] = 0
                domain_counts[domain_name] += 1
                
                chunk_idx += 1
        
        # Compute centroids
        self.compute_centroids(chunks)
        
        print(f"\n[DOMAIN] Created {optimal_k} domains:")
        for domain, count in sorted(domain_counts.items()):
            pct = count / len(chunks) * 100
            print(f"  - {domain:20s}: {count:4d} chunks ({pct:5.1f}%)")
        
        return {
            'num_domains': optimal_k,
            'optimal_k': optimal_k,
            'silhouette_score': float(silhouette),
            'method': 'content_based_kmeans',
            'domain_distribution': domain_counts
        }
    
    def _find_optimal_k(self, embeddings: np.ndarray) -> Tuple[int, float]:
        """
        Find optimal number of clusters using silhouette score.
        
        Args:
            embeddings: Array of embeddings (N x D)
            
        Returns:
            Tuple of (optimal_k, silhouette_score)
        """
        print(f"[DOMAIN] Auto-tuning K in range [{self.min_clusters}, {self.max_clusters}]...")
        
        # Limit max_clusters to number of samples
        max_k = min(self.max_clusters, len(embeddings) - 1)
        min_k = min(self.min_clusters, max_k)
        
        if max_k < 2:
            print("[DOMAIN] Not enough samples for clustering, using K=1")
            return 1, 0.0
        
        best_k = min_k
        best_score = -1.0
        
        # Try different K values
        for k in range(min_k, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            # Calculate silhouette score
            score = silhouette_score(embeddings, labels)
            
            print(f"  K={k}: silhouette={score:.3f}")
            
            if score > best_score:
                best_score = score
                best_k = k
        
        print(f"[DOMAIN] Optimal K={best_k} with silhouette={best_score:.3f}")
        
        return best_k, best_score
    
    def compute_centroids(self, chunks: List[Chunk]) -> None:
        """
        Compute centroid embedding for each domain.
        """
        domain_embeddings = {}
        
        for chunk in chunks:
            if not chunk.has_embedding():
                continue
                
            if chunk.domain not in domain_embeddings:
                domain_embeddings[chunk.domain] = []
                self.domain_chunks[chunk.domain] = []
            
            domain_embeddings[chunk.domain].append(chunk.embedding)
            self.domain_chunks[chunk.domain].append(chunk.chunk_id)
        
        # Compute centroids
        for domain, embeddings in domain_embeddings.items():
            centroid = np.mean(embeddings, axis=0)
            # L2 normalize
            centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
            self.domains[domain] = centroid
        
        print(f"[DOMAIN] Computed centroids for {len(self.domains)} domains")
    
    def route_query(self, 
                    query_embedding: np.ndarray, 
                    top_k_domains: int = 3,
                    similarity_threshold: float = 0.3) -> List[Tuple[str, float]]:
        """
        Route query to most relevant domains.
        
        Args:
            query_embedding: Query vector
            top_k_domains: Return top-K most similar domains
            similarity_threshold: Minimum similarity to include domain
        
        Returns:
            List of (domain_name, similarity_score) tuples
        """
        if not self.domains:
            return []
        
        # Normalize query embedding
        query_emb = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        
        # Compute similarity to each domain centroid
        similarities = []
        for domain, centroid in self.domains.items():
            sim = np.dot(query_emb, centroid)
            similarities.append((domain, float(sim)))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by threshold and top-k
        filtered = [
            (domain, score) 
            for domain, score in similarities 
            if score >= similarity_threshold
        ][:top_k_domains]
        
        if not filtered and similarities:
            # If nothing passed threshold, return top-1 anyway
            filtered = [similarities[0]]
        
        return filtered
    
    def get_domain_stats(self) -> Dict[str, any]:
        """Return statistics for each domain."""
        stats = {}
        
        for domain in self.domains.keys():
            stats[domain] = {
                'num_chunks': len(self.domain_chunks.get(domain, [])),
                'centroid_norm': float(np.linalg.norm(self.domains[domain])),
            }
        
        return stats
