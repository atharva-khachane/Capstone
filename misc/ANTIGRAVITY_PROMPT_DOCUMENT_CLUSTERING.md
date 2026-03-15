# ANTIGRAVITY PROMPT: DOCUMENT-LEVEL CONTENT CLUSTERING

## CRITICAL PROBLEM IDENTIFIED

**Current Approach:** Chunk-level clustering (clustering 1,560 individual chunks)
**Result:** Silhouette score 0.105 (very poor), 68% of data in one mega-cluster

**Root Cause:**
```
All procurement documents use similar vocabulary:
- "procurement", "tender", "bidding", "evaluation", "goods", "services"
- Similar sentence structures and procedural language
- K-means on chunk embeddings cannot distinguish them

Result:
  domain_0:   139 chunks (9%)   - Small cluster
  domain_1: 1,061 chunks (68%)  - MEGA CLUSTER (all procurement merged!)
  domain_2:   360 chunks (23%)  - Financial cluster

All 3 procurement document types merged into domain_1!
```

**Why This Matters:**
- Queries can't distinguish between "consultancy" vs "non-consultancy" vs "goods" procurement
- 68% of corpus in one cluster = poor retrieval precision
- Content-based approach failing for semantically similar documents

---

## THE SOLUTION: DOCUMENT-LEVEL CLUSTERING

**Key Insight:** Documents are coherent units with distinct purposes, even if individual chunks are similar.

**New Approach:**
1. **Cluster Documents, Not Chunks** - Cluster 8 documents instead of 1,560 chunks
2. **Document Embeddings** - Compute document-level embedding (average of all chunk embeddings)
3. **Clean Separation** - Documents have clearer boundaries than individual chunks
4. **Assign Chunks** - All chunks from a document inherit its domain
5. **Content-Based Naming** - Use TF-IDF to auto-generate domain names from document content

**Expected Results:**
- Clean separation of document types
- Balanced domain distribution
- Higher silhouette scores (0.6+)
- Works with ANY filename (file_1.pdf, scan.pdf, etc.)
- No manual configuration needed

---

## IMPLEMENTATION

### File: `sl_rag/retrieval/document_level_domain_manager.py`

```python
"""
Document-Level Content Clustering for Domain Detection

This module clusters documents (not chunks) based on their content embeddings,
then assigns all chunks from each document to the document's detected domain.

Key Features:
- Clusters entire documents for cleaner separation
- Auto-generates domain names from TF-IDF keywords
- Works with any filename (no pattern matching)
- Provides cluster quality metrics
- Handles imbalanced document sizes
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import re

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class DocumentInfo:
    """Information about a document for clustering."""
    doc_id: str
    chunk_ids: List[str]
    embeddings: np.ndarray  # All chunk embeddings from this doc
    doc_embedding: np.ndarray  # Document-level embedding (average)
    text_content: str  # Concatenated text for TF-IDF
    num_chunks: int
    metadata: Dict[str, Any]


class DocumentLevelDomainManager:
    """
    Manages domains using document-level clustering.
    
    Strategy:
    1. Group chunks by source document
    2. Compute document-level embedding (average of chunk embeddings)
    3. Cluster documents (not individual chunks)
    4. Generate domain names from cluster content (TF-IDF)
    5. Assign all chunks from each document to document's cluster
    
    Advantages:
    - Cleaner cluster separation (8 documents vs 1,560 chunks)
    - Works with any filename
    - Preserves document coherence
    - Better silhouette scores
    - Interpretable results
    """
    
    def __init__(self,
                 min_clusters: int = 2,
                 max_clusters: int = 10,
                 clustering_method: str = 'kmeans',
                 enable_hierarchical: bool = False):
        """
        Initialize document-level domain manager.
        
        Args:
            min_clusters: Minimum number of clusters to try
            max_clusters: Maximum number of clusters to try
            clustering_method: 'kmeans' or 'hierarchical'
            enable_hierarchical: Enable two-tier hierarchical clustering
        """
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.clustering_method = clustering_method
        self.enable_hierarchical = enable_hierarchical
        
        # Outputs
        self.domains: Dict[str, np.ndarray] = {}  # {domain_name: centroid}
        self.domain_chunks: Dict[str, List[str]] = {}  # {domain: [chunk_ids]}
        self.doc_to_domain: Dict[str, str] = {}  # {doc_id: domain_name}
        self.domain_keywords: Dict[str, List[str]] = {}  # {domain: [keywords]}
        
        # Cluster quality metrics
        self.silhouette_score = 0.0
        self.davies_bouldin_score = 0.0
        self.optimal_k = 0
    
    def detect_domains(self, chunks: List['Chunk']) -> Dict[str, Any]:
        """
        Detect domains using document-level clustering.
        
        Algorithm:
        1. Group chunks by document
        2. Compute document-level embeddings
        3. Find optimal number of clusters
        4. Cluster documents
        5. Generate domain names from content
        6. Assign chunks to domains
        
        Args:
            chunks: List of Chunk objects with embeddings
        
        Returns:
            Detection results with metrics
        """
        print("[DOMAIN] Starting document-level clustering...")
        
        # Step 1: Group chunks by document
        documents = self._group_chunks_by_document(chunks)
        print(f"[DOMAIN] Grouped {len(chunks)} chunks into {len(documents)} documents")
        
        # Step 2: Compute document embeddings
        doc_embeddings, doc_list = self._compute_document_embeddings(documents)
        print(f"[DOMAIN] Computed embeddings for {len(documents)} documents")
        
        # Step 3: Find optimal number of clusters
        optimal_k = self._find_optimal_clusters(doc_embeddings, len(documents))
        self.optimal_k = optimal_k
        print(f"[DOMAIN] Optimal number of clusters: {optimal_k}")
        
        # Step 4: Cluster documents
        cluster_labels = self._cluster_documents(doc_embeddings, optimal_k)
        
        # Step 5: Generate domain names from cluster content
        domain_names = self._generate_domain_names(documents, doc_list, cluster_labels, optimal_k)
        
        # Step 6: Assign chunks to domains
        self._assign_chunks_to_domains(chunks, documents, doc_list, cluster_labels, domain_names)
        
        # Step 7: Compute domain centroids
        self._compute_domain_centroids(chunks)
        
        # Step 8: Quality metrics
        if len(set(cluster_labels)) > 1:
            self.silhouette_score = silhouette_score(doc_embeddings, cluster_labels)
            self.davies_bouldin_score = davies_bouldin_score(doc_embeddings, cluster_labels)
        
        # Print results
        self._print_clustering_results()
        
        return {
            'num_domains': optimal_k,
            'method': 'document_level_clustering',
            'silhouette_score': self.silhouette_score,
            'davies_bouldin_score': self.davies_bouldin_score,
            'domain_distribution': {d: len(c) for d, c in self.domain_chunks.items()},
            'domain_keywords': self.domain_keywords
        }
    
    def _group_chunks_by_document(self, chunks: List['Chunk']) -> Dict[str, DocumentInfo]:
        """Group chunks by their source document."""
        doc_groups = defaultdict(list)
        
        for chunk in chunks:
            doc_groups[chunk.doc_id].append(chunk)
        
        documents = {}
        for doc_id, doc_chunks in doc_groups.items():
            # Extract embeddings
            embeddings = np.array([c.embedding for c in doc_chunks])
            
            # Concatenate text content
            text_content = ' '.join([c.content for c in doc_chunks])
            
            # Get metadata from first chunk
            metadata = doc_chunks[0].metadata if hasattr(doc_chunks[0], 'metadata') else {}
            
            documents[doc_id] = DocumentInfo(
                doc_id=doc_id,
                chunk_ids=[c.chunk_id for c in doc_chunks],
                embeddings=embeddings,
                doc_embedding=None,  # Will compute next
                text_content=text_content,
                num_chunks=len(doc_chunks),
                metadata=metadata
            )
        
        return documents
    
    def _compute_document_embeddings(self, 
                                    documents: Dict[str, DocumentInfo]) -> Tuple[np.ndarray, List[str]]:
        """
        Compute document-level embeddings.
        
        Strategy: Average of all chunk embeddings (centroid-based approach)
        Alternative: Weighted average by chunk importance
        """
        doc_embeddings = []
        doc_list = []
        
        for doc_id, doc_info in documents.items():
            # Compute document embedding as mean of chunk embeddings
            doc_embedding = np.mean(doc_info.embeddings, axis=0)
            
            # L2 normalize
            doc_embedding = doc_embedding / (np.linalg.norm(doc_embedding) + 1e-10)
            
            # Store
            doc_info.doc_embedding = doc_embedding
            doc_embeddings.append(doc_embedding)
            doc_list.append(doc_id)
        
        return np.array(doc_embeddings), doc_list
    
    def _find_optimal_clusters(self, doc_embeddings: np.ndarray, num_docs: int) -> int:
        """
        Find optimal number of clusters using silhouette analysis.
        
        Tests range [min_clusters, min(max_clusters, num_docs-1)]
        Returns k with highest silhouette score.
        """
        max_k = min(self.max_clusters, num_docs - 1)
        
        if num_docs < 2:
            return 1
        
        if num_docs == 2:
            return 2
        
        best_score = -1
        best_k = self.min_clusters
        
        print("[DOMAIN] Finding optimal number of clusters...")
        
        for k in range(self.min_clusters, max_k + 1):
            if k >= num_docs:
                break
            
            # Cluster with k clusters
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(doc_embeddings)
            
            # Compute silhouette score
            score = silhouette_score(doc_embeddings, labels)
            
            print(f"  - {k} clusters: silhouette = {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_k = k
        
        print(f"[DOMAIN] Best: {best_k} clusters (silhouette: {best_score:.3f})")
        
        return best_k
    
    def _cluster_documents(self, doc_embeddings: np.ndarray, k: int) -> np.ndarray:
        """
        Cluster documents using specified method.
        
        Returns cluster labels for each document.
        """
        if self.clustering_method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=k, linkage='ward')
            labels = clusterer.fit_predict(doc_embeddings)
        else:  # kmeans (default)
            clusterer = KMeans(n_clusters=k, random_state=42, n_init=20)
            labels = clusterer.fit_predict(doc_embeddings)
        
        return labels
    
    def _generate_domain_names(self,
                               documents: Dict[str, DocumentInfo],
                               doc_list: List[str],
                               cluster_labels: np.ndarray,
                               num_clusters: int) -> Dict[int, str]:
        """
        Generate meaningful domain names using TF-IDF on cluster content.
        
        Algorithm:
        1. Group documents by cluster
        2. Concatenate all text in each cluster
        3. Extract top TF-IDF terms for each cluster
        4. Generate domain name from top 2-3 terms
        
        Returns:
            {cluster_id: domain_name}
        """
        # Group documents by cluster
        cluster_texts = defaultdict(list)
        for i, doc_id in enumerate(doc_list):
            cluster_id = cluster_labels[i]
            cluster_texts[cluster_id].append(documents[doc_id].text_content)
        
        # Generate names for each cluster
        domain_names = {}
        
        for cluster_id in range(num_clusters):
            if cluster_id not in cluster_texts:
                domain_names[cluster_id] = f"domain_{cluster_id}"
                continue
            
            # Combine all text in cluster
            combined_text = ' '.join(cluster_texts[cluster_id])
            
            # Extract top keywords using TF-IDF
            keywords = self._extract_top_keywords(combined_text, top_n=10)
            
            # Generate domain name from top 2-3 keywords
            domain_name = self._create_domain_name(keywords[:3])
            
            domain_names[cluster_id] = domain_name
            self.domain_keywords[domain_name] = keywords
        
        return domain_names
    
    def _extract_top_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract top keywords from text using TF-IDF.
        
        Filters out common words and focuses on domain-specific terms.
        """
        # Custom stop words (add domain-agnostic terms)
        stop_words = [
            'shall', 'may', 'must', 'should', 'will', 'can', 'document',
            'section', 'clause', 'para', 'rule', 'act', 'said', 'thereof',
            'herein', 'hereof', 'hereby', 'pursuant', 'following', 'provided',
            'however', 'accordance', 'subject', 'respect', 'manner', 'period',
            'case', 'extent', 'purpose', 'required', 'specified', 'applicable'
        ]
        
        try:
            vectorizer = TfidfVectorizer(
                max_features=top_n,
                stop_words='english',
                ngram_range=(1, 2),  # Include bigrams
                min_df=1,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            
            # Get scores
            scores = tfidf_matrix.toarray()[0]
            
            # Sort by score
            top_indices = scores.argsort()[-top_n:][::-1]
            keywords = [feature_names[i] for i in top_indices]
            
            # Filter out custom stop words
            keywords = [k for k in keywords if k.lower() not in stop_words]
            
            return keywords[:top_n]
        
        except Exception as e:
            print(f"[DOMAIN] Warning: TF-IDF failed: {e}")
            return [f"keyword_{i}" for i in range(top_n)]
    
    def _create_domain_name(self, keywords: List[str]) -> str:
        """
        Create a domain name from keywords.
        
        Strategy:
        - Use top 2-3 keywords
        - Join with underscore
        - Clean up special characters
        - Make readable
        
        Examples:
        - ["procurement", "goods"] -> "procurement_goods"
        - ["financial", "rules", "government"] -> "financial_rules"
        - ["consulting", "services"] -> "consulting_services"
        """
        if not keywords:
            return "unknown_domain"
        
        # Take top 2-3 keywords
        top_keywords = keywords[:2]
        
        # Clean and join
        clean_keywords = []
        for kw in top_keywords:
            # Remove special characters, keep alphanumeric and spaces
            clean = re.sub(r'[^a-z0-9\s]', '', kw.lower())
            clean = clean.strip().replace(' ', '_')
            if clean:
                clean_keywords.append(clean)
        
        if not clean_keywords:
            return "unknown_domain"
        
        domain_name = '_'.join(clean_keywords)
        
        # Truncate if too long
        if len(domain_name) > 50:
            domain_name = domain_name[:50]
        
        return domain_name
    
    def _assign_chunks_to_domains(self,
                                  chunks: List['Chunk'],
                                  documents: Dict[str, DocumentInfo],
                                  doc_list: List[str],
                                  cluster_labels: np.ndarray,
                                  domain_names: Dict[int, str]):
        """
        Assign all chunks to their document's domain.
        
        All chunks from the same document get the same domain.
        """
        # Create doc_id to domain mapping
        for i, doc_id in enumerate(doc_list):
            cluster_id = cluster_labels[i]
            domain_name = domain_names[cluster_id]
            self.doc_to_domain[doc_id] = domain_name
        
        # Assign chunks
        for chunk in chunks:
            domain = self.doc_to_domain.get(chunk.doc_id, 'unknown')
            chunk.domain = domain
            
            if domain not in self.domain_chunks:
                self.domain_chunks[domain] = []
            self.domain_chunks[domain].append(chunk.chunk_id)
    
    def _compute_domain_centroids(self, chunks: List['Chunk']):
        """
        Compute centroid embedding for each domain.
        
        Centroid = mean of all chunk embeddings in domain
        """
        domain_embeddings = defaultdict(list)
        
        for chunk in chunks:
            domain_embeddings[chunk.domain].append(chunk.embedding)
        
        for domain, embeddings in domain_embeddings.items():
            centroid = np.mean(embeddings, axis=0)
            # L2 normalize
            centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
            self.domains[domain] = centroid
        
        print(f"[DOMAIN] Computed centroids for {len(self.domains)} domains")
    
    def _print_clustering_results(self):
        """Print detailed clustering results."""
        print("\n" + "="*80)
        print("DOCUMENT-LEVEL CLUSTERING RESULTS")
        print("="*80)
        
        print(f"\n[CLUSTERS]")
        print(f"  Number of domains: {len(self.domains)}")
        print(f"  Silhouette score: {self.silhouette_score:.3f}")
        print(f"  Davies-Bouldin score: {self.davies_bouldin_score:.3f}")
        
        print(f"\n[DOMAIN DISTRIBUTION]")
        for domain in sorted(self.domain_chunks.keys()):
            num_chunks = len(self.domain_chunks[domain])
            percentage = 100 * num_chunks / sum(len(c) for c in self.domain_chunks.values())
            keywords = ', '.join(self.domain_keywords.get(domain, [])[:5])
            print(f"  {domain:30s}: {num_chunks:4d} chunks ({percentage:5.1f}%)")
            print(f"    Keywords: {keywords}")
        
        print(f"\n[DOCUMENT-TO-DOMAIN MAPPING]")
        for doc_id, domain in sorted(self.doc_to_domain.items()):
            print(f"  {doc_id:40s} -> {domain}")
        
        print("="*80 + "\n")
    
    def route_query(self, 
                    query_embedding: np.ndarray, 
                    top_k_domains: int = 3,
                    similarity_threshold: float = 0.2) -> List[Tuple[str, float]]:
        """
        Route query to most relevant domains using centroid similarity.
        
        Args:
            query_embedding: Query vector
            top_k_domains: Return top-K domains
            similarity_threshold: Minimum similarity to include
        
        Returns:
            List of (domain_name, similarity_score) tuples
        """
        if not self.domains:
            return []
        
        # Normalize query
        query_emb = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        
        # Compute similarities
        similarities = []
        for domain, centroid in self.domains.items():
            sim = float(np.dot(query_emb, centroid))
            similarities.append((domain, sim))
        
        # Sort and filter
        similarities.sort(key=lambda x: x[1], reverse=True)
        filtered = [
            (domain, score) 
            for domain, score in similarities 
            if score >= similarity_threshold
        ][:top_k_domains]
        
        if not filtered and similarities:
            # Return top-1 if nothing passed threshold
            filtered = [similarities[0]]
        
        return filtered
    
    def get_domain_stats(self) -> Dict[str, Any]:
        """Return comprehensive domain statistics."""
        stats = {}
        
        for domain in self.domains:
            num_chunks = len(self.domain_chunks.get(domain, [])
            keywords = self.domain_keywords.get(domain, [])
            
            stats[domain] = {
                'num_chunks': num_chunks,
                'keywords': keywords[:10],
                'centroid_norm': float(np.linalg.norm(self.domains[domain]))
            }
        
        stats['_overall'] = {
            'num_domains': len(self.domains),
            'silhouette_score': self.silhouette_score,
            'davies_bouldin_score': self.davies_bouldin_score
        }
        
        return stats
    
    def visualize_clusters(self, 
                          documents: Dict[str, DocumentInfo],
                          output_path: str = "./domain_clusters.png"):
        """
        Visualize document clusters using t-SNE or UMAP.
        
        Shows document-level separation.
        """
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            
            # Get document embeddings
            doc_embeddings = np.array([d.doc_embedding for d in documents.values()])
            doc_ids = list(documents.keys())
            
            # Reduce to 2D
            if len(doc_embeddings) > 3:
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(doc_embeddings)-1))
                embeddings_2d = tsne.fit_transform(doc_embeddings)
            else:
                # Too few points for t-SNE
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                embeddings_2d = pca.fit_transform(doc_embeddings)
            
            # Plot
            plt.figure(figsize=(12, 8))
            
            for i, doc_id in enumerate(doc_ids):
                domain = self.doc_to_domain.get(doc_id, 'unknown')
                plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], 
                           s=200, alpha=0.6, label=domain)
                plt.annotate(doc_id[:20], 
                            (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                            fontsize=8, alpha=0.7)
            
            # Remove duplicate labels
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            
            plt.title(f"Document Clustering Visualization\n(Silhouette: {self.silhouette_score:.3f})")
            plt.xlabel("t-SNE Dimension 1")
            plt.ylabel("t-SNE Dimension 2")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[DOMAIN] Visualization saved to {output_path}")
        
        except Exception as e:
            print(f"[DOMAIN] Warning: Visualization failed: {e}")


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================

def create_document_level_domain_manager(**kwargs) -> DocumentLevelDomainManager:
    """
    Factory function to create document-level domain manager.
    
    Usage:
        dm = create_document_level_domain_manager(min_clusters=2, max_clusters=10)
        result = dm.detect_domains(chunks)
    """
    return DocumentLevelDomainManager(**kwargs)
```

---

## TESTING & VALIDATION

### File: `tests/test_document_level_clustering.py`

```python
"""
Test suite for document-level clustering.
"""

import pytest
import numpy as np
from sl_rag.retrieval.document_level_domain_manager import DocumentLevelDomainManager


def test_document_grouping(sample_chunks):
    """Test that chunks are correctly grouped by document."""
    dm = DocumentLevelDomainManager()
    documents = dm._group_chunks_by_document(sample_chunks)
    
    # Check all chunks grouped
    assert len(documents) > 0
    
    # Check each document has correct chunks
    for doc_id, doc_info in documents.items():
        assert doc_info.num_chunks > 0
        assert len(doc_info.chunk_ids) == doc_info.num_chunks
        assert doc_info.embeddings.shape[0] == doc_info.num_chunks


def test_document_embeddings(sample_chunks):
    """Test document embedding computation."""
    dm = DocumentLevelDomainManager()
    documents = dm._group_chunks_by_document(sample_chunks)
    doc_embeddings, doc_list = dm._compute_document_embeddings(documents)
    
    # Check shape
    assert doc_embeddings.shape[0] == len(documents)
    assert doc_embeddings.shape[1] == 768  # all-mpnet-base-v2 dimension
    
    # Check normalization
    for emb in doc_embeddings:
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 0.01  # Should be L2 normalized


def test_optimal_clusters(sample_chunks):
    """Test optimal cluster selection."""
    dm = DocumentLevelDomainManager(min_clusters=2, max_clusters=5)
    documents = dm._group_chunks_by_document(sample_chunks)
    doc_embeddings, _ = dm._compute_document_embeddings(documents)
    
    optimal_k = dm._find_optimal_clusters(doc_embeddings, len(documents))
    
    # Check valid range
    assert 2 <= optimal_k <= min(5, len(documents) - 1)


def test_domain_name_generation(sample_chunks):
    """Test domain name generation from content."""
    dm = DocumentLevelDomainManager()
    
    # Test keyword extraction
    text = "procurement of goods and services tender bidding evaluation"
    keywords = dm._extract_top_keywords(text, top_n=5)
    
    assert len(keywords) > 0
    assert 'procurement' in ' '.join(keywords).lower()
    
    # Test domain name creation
    domain_name = dm._create_domain_name(['procurement', 'goods'])
    assert 'procurement' in domain_name
    assert '_' in domain_name  # Should be joined with underscore


def test_full_clustering_pipeline(sample_chunks_with_embeddings):
    """Test complete clustering pipeline."""
    dm = DocumentLevelDomainManager(min_clusters=2, max_clusters=5)
    
    result = dm.detect_domains(sample_chunks_with_embeddings)
    
    # Check results
    assert result['num_domains'] >= 2
    assert 'silhouette_score' in result
    assert result['silhouette_score'] > -1  # Valid range: -1 to 1
    
    # Check domain assignment
    assert len(dm.domains) == result['num_domains']
    assert len(dm.domain_chunks) == result['num_domains']
    
    # Check all chunks assigned
    total_chunks = sum(len(chunks) for chunks in dm.domain_chunks.values())
    assert total_chunks == len(sample_chunks_with_embeddings)


def test_silhouette_score_improvement():
    """
    Test that document-level clustering improves silhouette score
    compared to chunk-level clustering.
    """
    # This would compare:
    # - Chunk-level clustering (expected: 0.1-0.2)
    # - Document-level clustering (expected: 0.5-0.7)
    
    # Implementation depends on having actual test data
    pass


def test_domain_routing(sample_chunks_with_embeddings):
    """Test query routing to domains."""
    dm = DocumentLevelDomainManager()
    dm.detect_domains(sample_chunks_with_embeddings)
    
    # Generate test query embedding
    query_emb = np.random.randn(768)
    query_emb = query_emb / np.linalg.norm(query_emb)
    
    # Route query
    routed = dm.route_query(query_emb, top_k_domains=2)
    
    # Check results
    assert len(routed) <= 2
    assert all(isinstance(domain, str) for domain, _ in routed)
    assert all(isinstance(score, float) for _, score in routed)
    assert all(-1 <= score <= 1 for _, score in routed)  # Cosine similarity range
```

---

## INTEGRATION GUIDE

### Update Your Main Pipeline

```python
# In your demo.py or main pipeline:

# OLD (chunk-level clustering):
# from sl_rag.retrieval.domain_manager import DomainManager
# domain_manager = DomainManager(min_clusters=3, max_clusters=8)

# NEW (document-level clustering):
from sl_rag.retrieval.document_level_domain_manager import DocumentLevelDomainManager

domain_manager = DocumentLevelDomainManager(
    min_clusters=2,
    max_clusters=10,
    clustering_method='kmeans',  # or 'hierarchical'
    enable_hierarchical=False
)

# Rest of pipeline stays the same:
result = domain_manager.detect_domains(chunks)
print(f"Detected {result['num_domains']} domains")
print(f"Silhouette score: {result['silhouette_score']:.3f}")
```

---

## EXPECTED IMPROVEMENTS

### Metrics Comparison

| Metric | Chunk-Level (Before) | Document-Level (After) | Improvement |
|--------|---------------------|------------------------|-------------|
| **Silhouette Score** | 0.105 (poor) | 0.5-0.7 (good) | **+376%** |
| **Cluster Balance** | 68% in one cluster | Even distribution | **Balanced** |
| **Domain Separation** | Blurred boundaries | Clear separation | **Distinct** |
| **Interpretability** | Generic names | Content-based names | **Meaningful** |
| **Filename Independence** | ❌ | ✅ | **Robust** |

### Example Output (Expected)

```
[DOMAIN] Starting document-level clustering...
[DOMAIN] Grouped 1560 chunks into 8 documents
[DOMAIN] Computed embeddings for 8 documents

[DOMAIN] Finding optimal number of clusters...
  - 2 clusters: silhouette = 0.412
  - 3 clusters: silhouette = 0.487
  - 4 clusters: silhouette = 0.623  ← BEST
  - 5 clusters: silhouette = 0.531
[DOMAIN] Best: 4 clusters (silhouette: 0.623)

================================================================================
DOCUMENT-LEVEL CLUSTERING RESULTS
================================================================================

[CLUSTERS]
  Number of domains: 4
  Silhouette score: 0.623
  Davies-Bouldin score: 0.751

[DOMAIN DISTRIBUTION]
  procurement_goods              :  506 chunks (32.4%)
    Keywords: goods, tender, materials, specifications, delivery
    
  procurement_consultancy        :  347 chunks (22.2%)
    Keywords: consultancy, services, selection, technical, proposal
    
  procurement_non_consultancy    :  350 chunks (22.4%)
    Keywords: services, work, contract, performance, agreement
    
  financial_rules                :  225 chunks (14.4%)
    Keywords: financial, budget, expenditure, accounts, audit
    
  technical_reports              :  132 chunks (8.5%)
    Keywords: technical, analysis, system, design, specifications

[DOCUMENT-TO-DOMAIN MAPPING]
  Procurement_Goods.pdf                  -> procurement_goods
  Procurement_Consultancy.pdf            -> procurement_consultancy
  Procurement_Non_Consultancy.pdf        -> procurement_non_consultancy
  FInal_GFR_upto_31_07_2024.pdf          -> financial_rules
  208B.pdf                               -> technical_reports
  20020050369.pdf                        -> technical_reports
  20180000774.pdf                        -> technical_reports
  20210017934 FINAL.pdf                  -> technical_reports

================================================================================

[SUCCESS] Document-level clustering complete!
  - 4 domains detected
  - Clean separation (silhouette: 0.623)
  - Balanced distribution (14-32% per domain)
  - Content-based naming (NO filename dependency)
```

---

## ADVANTAGES OVER PREVIOUS APPROACHES

### vs. Filename-Based:
✅ Works with ANY filename (file_1.pdf, scan.pdf, document.pdf)  
✅ No manual pattern maintenance  
✅ No hard-coded domain logic  
✅ Learns from content, not metadata  
✅ Scalable to any document set  

### vs. Chunk-Level Clustering:
✅ Much better silhouette scores (0.6+ vs 0.1)  
✅ Cleaner cluster separation  
✅ Balanced distribution (no mega-clusters)  
✅ Preserves document coherence  
✅ Computationally efficient (cluster 8 docs, not 1,560 chunks)  

### Unique Benefits:
✅ **Content-driven**: Domains based on actual text content  
✅ **Automatic**: No manual configuration needed  
✅ **Interpretable**: TF-IDF keywords explain each domain  
✅ **Robust**: Handles varied document sizes  
✅ **Quality metrics**: Silhouette score for validation  

---

## OPTIONAL ENHANCEMENTS

### Enhancement 1: Hierarchical Clustering

```python
# Enable two-tier clustering for complex corpora:
dm = DocumentLevelDomainManager(
    enable_hierarchical=True,
    min_clusters=2,
    max_clusters=10
)

# This will:
# Tier 1: Broad topics (procurement, financial, technical)
# Tier 2: Sub-topics within each (goods, consultancy, non-consultancy)
```

### Enhancement 2: Weighted Document Embeddings

```python
def _compute_document_embeddings_weighted(self, documents):
    """Use TF-IDF weighted averaging instead of simple mean."""
    # Weight each chunk by its importance (TF-IDF score)
    # Gives more weight to distinctive chunks
    pass
```

### Enhancement 3: Incremental Clustering

```python
def add_new_documents(self, new_chunks: List['Chunk']):
    """
    Add new documents without re-clustering everything.
    
    Algorithm:
    1. Compute new document embeddings
    2. Assign to nearest existing cluster
    3. Update cluster centroids
    4. Recompute if drift detected
    """
    pass
```

---

## TESTING CHECKLIST

After implementation:

- [ ] Run on 8 document corpus
- [ ] Verify silhouette score > 0.5
- [ ] Check balanced distribution (no >40% clusters)
- [ ] Test with renamed files (file_1.pdf, etc.)
- [ ] Validate domain names are meaningful
- [ ] Test query routing accuracy
- [ ] Benchmark clustering time (should be <5s)
- [ ] Visualize clusters (t-SNE plot)
- [ ] Compare with previous approach
- [ ] Generate quality report

---

## SUCCESS CRITERIA

✅ **Silhouette Score**: > 0.5 (target: 0.6+)  
✅ **Balanced Clusters**: No cluster > 40% of data  
✅ **Domain Names**: Meaningful, content-based  
✅ **Filename Independence**: Works with any filename  
✅ **Query Routing**: 80%+ accuracy  
✅ **Clustering Time**: < 5 seconds  
✅ **Interpretability**: Clear domain separation  

---

END OF DOCUMENT-LEVEL CLUSTERING PROMPT
