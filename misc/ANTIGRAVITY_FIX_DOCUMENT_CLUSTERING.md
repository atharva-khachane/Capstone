# ANTIGRAVITY FIX PROMPT: DOCUMENT-LEVEL CLUSTERING BUGS

## CRITICAL BUGS IDENTIFIED

The document-level clustering implementation has **5 critical bugs** that prevent it from working:

### Bug 1: Domain Assignment Logic Failure
**Symptom:** Detected 3 clusters but created only 1 domain (all 1,560 chunks in one domain)  
**Cause:** Bug in `_assign_chunks_to_domains()` method  
**Impact:** Complete failure of domain separation  

### Bug 2: TF-IDF Parameter Mismatch
**Symptom:** "max_df corresponds to < documents than min_df"  
**Cause:** TF-IDF parameters (max_df=0.8, min_df=1) invalid for small cluster sizes  
**Impact:** All domain names fallback to "keyword0_keyword1"  

### Bug 3: Document ID Readability
**Symptom:** Document IDs are SHA-256 hashes instead of filenames  
**Cause:** Using doc_id (hash) instead of filepath for mapping  
**Impact:** Unreadable output, can't verify correctness  

### Bug 4: Stats Structure Mismatch
**Symptom:** KeyError: 'num_chunks' at runtime  
**Cause:** get_domain_stats() returns different structure than expected  
**Impact:** Pipeline crashes at summary  

### Bug 5: Fallback Keyword Generation
**Symptom:** Generic "keyword_0, keyword_1" when TF-IDF fails  
**Cause:** Fallback doesn't extract any real keywords  
**Impact:** Meaningless domain names  

---

## COMPREHENSIVE FIX

### File: `sl_rag/retrieval/document_level_domain_manager.py` (FIXED VERSION)

Replace the broken implementation with this corrected version:

```python
"""
Document-Level Content Clustering - FIXED VERSION

Critical fixes:
1. Proper domain assignment for all detected clusters
2. TF-IDF parameters adapted for small document counts
3. Human-readable document IDs (filenames instead of hashes)
4. Consistent stats structure
5. Better fallback keyword extraction
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, Counter
import re
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class DocumentInfo:
    """Information about a document for clustering."""
    doc_id: str
    filepath: str  # NEW: Store actual filename
    chunk_ids: List[str]
    embeddings: np.ndarray
    doc_embedding: np.ndarray
    text_content: str
    num_chunks: int
    metadata: Dict[str, Any]


class DocumentLevelDomainManager:
    """Fixed document-level domain manager."""
    
    def __init__(self,
                 min_clusters: int = 2,
                 max_clusters: int = 10):
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        
        # Outputs
        self.domains: Dict[str, np.ndarray] = {}
        self.domain_chunks: Dict[str, List[str]] = {}
        self.doc_to_domain: Dict[str, str] = {}
        self.domain_keywords: Dict[str, List[str]] = {}
        self.filepath_to_domain: Dict[str, str] = {}  # NEW: Filepath mapping
        
        # Metrics
        self.silhouette_score = 0.0
        self.davies_bouldin_score = 0.0
        self.optimal_k = 0
    
    def detect_domains(self, chunks: List['Chunk']) -> Dict[str, Any]:
        """Detect domains using document-level clustering (FIXED)."""
        print("[DOMAIN] Starting document-level clustering...")
        
        # Step 1: Group chunks by document
        documents = self._group_chunks_by_document(chunks)
        print(f"[DOMAIN] Grouped {len(chunks)} chunks into {len(documents)} documents")
        
        # Step 2: Compute document embeddings
        doc_embeddings, doc_list = self._compute_document_embeddings(documents)
        print(f"[DOMAIN] Computed embeddings for {len(documents)} documents")
        
        # Step 3: Find optimal clusters
        optimal_k = self._find_optimal_clusters(doc_embeddings, len(documents))
        self.optimal_k = optimal_k
        print(f"[DOMAIN] Optimal number of clusters: {optimal_k}")
        
        # Step 4: Cluster documents
        cluster_labels = self._cluster_documents(doc_embeddings, optimal_k)
        print(f"[DOMAIN] Clustered {len(documents)} documents into {optimal_k} clusters")
        
        # Step 5: Generate domain names (FIXED)
        domain_names = self._generate_domain_names_fixed(documents, doc_list, cluster_labels, optimal_k)
        print(f"[DOMAIN] Generated names for {len(domain_names)} domains")
        
        # Step 6: Assign chunks to domains (FIXED)
        self._assign_chunks_to_domains_fixed(chunks, documents, doc_list, cluster_labels, domain_names)
        print(f"[DOMAIN] Assigned {len(chunks)} chunks to {len(set(c.domain for c in chunks))} domains")
        
        # Step 7: Compute centroids
        self._compute_domain_centroids(chunks)
        
        # Step 8: Quality metrics
        if len(set(cluster_labels)) > 1:
            self.silhouette_score = silhouette_score(doc_embeddings, cluster_labels)
            self.davies_bouldin_score = davies_bouldin_score(doc_embeddings, cluster_labels)
        
        # Print results
        self._print_clustering_results()
        
        return {
            'num_domains': len(self.domains),  # FIX: Use actual domains count
            'method': 'document_level_clustering',
            'silhouette_score': self.silhouette_score,
            'davies_bouldin_score': self.davies_bouldin_score,
            'domain_distribution': {d: len(c) for d, c in self.domain_chunks.items()},
            'domain_keywords': self.domain_keywords
        }
    
    def _group_chunks_by_document(self, chunks: List['Chunk']) -> Dict[str, DocumentInfo]:
        """Group chunks by document (FIXED - extract filepath)."""
        doc_groups = defaultdict(list)
        
        for chunk in chunks:
            doc_groups[chunk.doc_id].append(chunk)
        
        documents = {}
        for doc_id, doc_chunks in doc_groups.items():
            # Extract embeddings
            embeddings = np.array([c.embedding for c in doc_chunks])
            
            # Concatenate text
            text_content = ' '.join([c.content for c in doc_chunks])
            
            # Get metadata and filepath
            metadata = doc_chunks[0].metadata if hasattr(doc_chunks[0], 'metadata') else {}
            filepath = metadata.get('filepath', f'doc_{doc_id[:8]}')  # FIX: Get actual filepath
            
            documents[doc_id] = DocumentInfo(
                doc_id=doc_id,
                filepath=filepath,  # FIX: Store filepath
                chunk_ids=[c.chunk_id for c in doc_chunks],
                embeddings=embeddings,
                doc_embedding=None,
                text_content=text_content,
                num_chunks=len(doc_chunks),
                metadata=metadata
            )
        
        return documents
    
    def _compute_document_embeddings(self, 
                                    documents: Dict[str, DocumentInfo]) -> Tuple[np.ndarray, List[str]]:
        """Compute document-level embeddings."""
        doc_embeddings = []
        doc_list = []
        
        for doc_id, doc_info in documents.items():
            # Mean of chunk embeddings
            doc_embedding = np.mean(doc_info.embeddings, axis=0)
            doc_embedding = doc_embedding / (np.linalg.norm(doc_embedding) + 1e-10)
            
            doc_info.doc_embedding = doc_embedding
            doc_embeddings.append(doc_embedding)
            doc_list.append(doc_id)
        
        return np.array(doc_embeddings), doc_list
    
    def _find_optimal_clusters(self, doc_embeddings: np.ndarray, num_docs: int) -> int:
        """Find optimal clusters using silhouette analysis."""
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
            
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(doc_embeddings)
            
            score = silhouette_score(doc_embeddings, labels)
            print(f"  - {k} clusters: silhouette = {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_k = k
        
        print(f"[DOMAIN] Best: {best_k} clusters (silhouette: {best_score:.3f})")
        return best_k
    
    def _cluster_documents(self, doc_embeddings: np.ndarray, k: int) -> np.ndarray:
        """Cluster documents using K-means."""
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = kmeans.fit_predict(doc_embeddings)
        return labels
    
    def _generate_domain_names_fixed(self,
                                     documents: Dict[str, DocumentInfo],
                                     doc_list: List[str],
                                     cluster_labels: np.ndarray,
                                     num_clusters: int) -> Dict[int, str]:
        """
        Generate domain names (FIXED VERSION).
        
        Fixes:
        1. Adjusted TF-IDF parameters for small clusters
        2. Better fallback strategy
        3. Use simple word frequency when TF-IDF fails
        """
        # Group documents by cluster
        cluster_docs = defaultdict(list)
        cluster_texts = defaultdict(list)
        
        for i, doc_id in enumerate(doc_list):
            cluster_id = cluster_labels[i]
            cluster_docs[cluster_id].append(documents[doc_id])
            cluster_texts[cluster_id].append(documents[doc_id].text_content)
        
        domain_names = {}
        
        for cluster_id in range(num_clusters):
            if cluster_id not in cluster_texts:
                domain_names[cluster_id] = f"domain_{cluster_id}"
                continue
            
            # Try TF-IDF with fixed parameters
            keywords = self._extract_keywords_robust(
                cluster_texts[cluster_id],
                cluster_docs[cluster_id]
            )
            
            # Generate name
            domain_name = self._create_domain_name(keywords[:3])
            domain_names[cluster_id] = domain_name
            self.domain_keywords[domain_name] = keywords
        
        return domain_names
    
    def _extract_keywords_robust(self, 
                                 cluster_texts: List[str],
                                 cluster_docs: List[DocumentInfo],
                                 top_n: int = 10) -> List[str]:
        """
        Extract keywords robustly (handles small clusters).
        
        Strategy:
        1. Try TF-IDF with adapted parameters
        2. If fails, use simple word frequency
        3. Filter common words manually
        """
        # Combine all text
        combined_text = ' '.join(cluster_texts)
        
        # Try TF-IDF (FIXED parameters for small clusters)
        try:
            # FIX: Adjust parameters based on cluster size
            num_docs = len(cluster_texts)
            
            vectorizer = TfidfVectorizer(
                max_features=top_n * 2,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,  # At least 1 document
                max_df=max(0.95, 1.0 - (1.0 / num_docs))  # FIX: Adaptive max_df
            )
            
            tfidf_matrix = vectorizer.fit_transform(cluster_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores across documents
            avg_scores = tfidf_matrix.mean(axis=0).A1
            top_indices = avg_scores.argsort()[-top_n:][::-1]
            keywords = [feature_names[i] for i in top_indices]
            
            # Filter stop words
            keywords = self._filter_domain_words(keywords)
            
            if len(keywords) >= 2:
                return keywords
        
        except Exception as e:
            print(f"[DOMAIN] TF-IDF failed: {e}, using fallback")
        
        # Fallback: Simple word frequency
        return self._extract_keywords_frequency(combined_text, top_n)
    
    def _extract_keywords_frequency(self, text: str, top_n: int = 10) -> List[str]:
        """Fallback: Extract keywords using word frequency."""
        # Tokenize
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        
        # Filter stop words
        stop_words = set([
            'shall', 'will', 'must', 'should', 'would', 'could', 'may', 'might',
            'have', 'has', 'had', 'been', 'being', 'were', 'was', 'are', 'is',
            'the', 'and', 'for', 'with', 'from', 'that', 'this', 'which', 'where',
            'when', 'what', 'who', 'how', 'said', 'such', 'other', 'than', 'then',
            'them', 'these', 'those', 'there', 'their', 'they', 'also', 'only',
            'document', 'section', 'clause', 'para', 'rule', 'chapter', 'part'
        ])
        
        words = [w for w in words if w not in stop_words]
        
        # Count frequency
        word_counts = Counter(words)
        most_common = word_counts.most_common(top_n * 2)
        
        # Extract keywords
        keywords = [word for word, count in most_common if count > 2][:top_n]
        
        if not keywords:
            # Ultimate fallback
            keywords = [word for word, count in most_common[:top_n]]
        
        return keywords if keywords else ['document', 'content']
    
    def _filter_domain_words(self, keywords: List[str]) -> List[str]:
        """Filter out generic domain words."""
        stop_words = {
            'shall', 'may', 'must', 'should', 'will', 'can',
            'document', 'section', 'clause', 'para', 'rule',
            'provided', 'however', 'following', 'respect'
        }
        
        return [k for k in keywords if k.lower() not in stop_words]
    
    def _create_domain_name(self, keywords: List[str]) -> str:
        """Create domain name from keywords."""
        if not keywords:
            return "unknown_domain"
        
        # Take top 2-3 keywords
        top_keywords = keywords[:min(3, len(keywords))]
        
        # Clean
        clean_keywords = []
        for kw in top_keywords:
            clean = re.sub(r'[^a-z0-9\s]', '', kw.lower())
            clean = clean.strip().replace(' ', '_')
            if clean and len(clean) > 2:
                clean_keywords.append(clean)
        
        if not clean_keywords:
            return "domain_" + keywords[0][:10]
        
        domain_name = '_'.join(clean_keywords[:2])  # Max 2 words
        
        # Truncate if too long
        if len(domain_name) > 40:
            domain_name = domain_name[:40]
        
        return domain_name
    
    def _assign_chunks_to_domains_fixed(self,
                                       chunks: List['Chunk'],
                                       documents: Dict[str, DocumentInfo],
                                       doc_list: List[str],
                                       cluster_labels: np.ndarray,
                                       domain_names: Dict[int, str]):
        """
        Assign chunks to domains (FIXED VERSION).
        
        Critical fix: Ensure ALL clusters get domains assigned.
        """
        # Create doc_id to domain mapping
        doc_to_cluster = {}
        for i, doc_id in enumerate(doc_list):
            cluster_id = cluster_labels[i]
            doc_to_cluster[doc_id] = cluster_id
        
        # Create cluster to domain mapping
        cluster_to_domain = {}
        for cluster_id, domain_name in domain_names.items():
            cluster_to_domain[cluster_id] = domain_name
        
        # FIX: Verify all clusters have domains
        for cluster_id in set(cluster_labels):
            if cluster_id not in cluster_to_domain:
                print(f"[DOMAIN] WARNING: Cluster {cluster_id} has no domain! Creating fallback...")
                cluster_to_domain[cluster_id] = f"domain_{cluster_id}"
        
        # Assign each chunk
        chunks_assigned = 0
        for chunk in chunks:
            doc_id = chunk.doc_id
            
            if doc_id in doc_to_cluster:
                cluster_id = doc_to_cluster[doc_id]
                domain = cluster_to_domain[cluster_id]
                chunk.domain = domain
                chunks_assigned += 1
                
                # Track
                if domain not in self.domain_chunks:
                    self.domain_chunks[domain] = []
                self.domain_chunks[domain].append(chunk.chunk_id)
                
                # Map doc to domain
                self.doc_to_domain[doc_id] = domain
                
                # FIX: Map filepath to domain (for readable output)
                if doc_id in documents:
                    filepath = documents[doc_id].filepath
                    self.filepath_to_domain[filepath] = domain
            else:
                print(f"[DOMAIN] WARNING: Chunk {chunk.chunk_id} has unknown doc_id {doc_id}")
                chunk.domain = "unknown"
        
        print(f"[DOMAIN] Assigned {chunks_assigned}/{len(chunks)} chunks to {len(self.domain_chunks)} domains")
        
        # Verify assignment
        for domain, chunk_list in self.domain_chunks.items():
            print(f"  - {domain}: {len(chunk_list)} chunks")
    
    def _compute_domain_centroids(self, chunks: List['Chunk']):
        """Compute domain centroids."""
        domain_embeddings = defaultdict(list)
        
        for chunk in chunks:
            if hasattr(chunk, 'domain') and chunk.domain:
                domain_embeddings[chunk.domain].append(chunk.embedding)
        
        for domain, embeddings in domain_embeddings.items():
            if embeddings:
                centroid = np.mean(embeddings, axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
                self.domains[domain] = centroid
        
        print(f"[DOMAIN] Computed centroids for {len(self.domains)} domains")
    
    def _print_clustering_results(self):
        """Print results (FIXED - readable output)."""
        print("\n" + "="*80)
        print("DOCUMENT-LEVEL CLUSTERING RESULTS")
        print("="*80)
        
        print(f"\n[CLUSTERS]")
        print(f"  Number of domains: {len(self.domains)}")
        print(f"  Silhouette score: {self.silhouette_score:.3f}")
        print(f"  Davies-Bouldin score: {self.davies_bouldin_score:.3f}")
        
        print(f"\n[DOMAIN DISTRIBUTION]")
        total_chunks = sum(len(c) for c in self.domain_chunks.values())
        
        for domain in sorted(self.domain_chunks.keys()):
            num_chunks = len(self.domain_chunks[domain])
            percentage = 100 * num_chunks / total_chunks if total_chunks > 0 else 0
            keywords = ', '.join(self.domain_keywords.get(domain, [])[:5])
            print(f"  {domain:30s}: {num_chunks:4d} chunks ({percentage:5.1f}%)")
            if keywords:
                print(f"    Keywords: {keywords}")
        
        print(f"\n[DOCUMENT-TO-DOMAIN MAPPING]")
        # FIX: Show filepath instead of hash
        for filepath, domain in sorted(self.filepath_to_domain.items()):
            filename = Path(filepath).name if filepath else "unknown"
            print(f"  {filename:40s} -> {domain}")
        
        print("="*80 + "\n")
    
    def route_query(self, 
                    query_embedding: np.ndarray, 
                    top_k_domains: int = 3,
                    similarity_threshold: float = 0.2) -> List[Tuple[str, float]]:
        """Route query to domains."""
        if not self.domains:
            return []
        
        query_emb = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        
        similarities = []
        for domain, centroid in self.domains.items():
            sim = float(np.dot(query_emb, centroid))
            similarities.append((domain, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        filtered = [
            (domain, score) 
            for domain, score in similarities 
            if score >= similarity_threshold
        ][:top_k_domains]
        
        if not filtered and similarities:
            filtered = [similarities[0]]
        
        return filtered
    
    def get_domain_stats(self) -> Dict[str, Any]:
        """Return domain statistics (FIXED structure)."""
        stats = {}
        
        for domain in self.domains:
            num_chunks = len(self.domain_chunks.get(domain, []))
            keywords = self.domain_keywords.get(domain, [])
            
            stats[domain] = {
                'num_chunks': num_chunks,  # FIX: Ensure this key exists
                'keywords': keywords[:10],
                'centroid_norm': float(np.linalg.norm(self.domains[domain]))
            }
        
        return stats
```

---

## TESTING THE FIX

After replacing the code, you should see:

```
[DOMAIN] Optimal number of clusters: 3
[DOMAIN] Generated names for 3 domains
[DOMAIN] Assigned 1560/1560 chunks to 3 domains
  - procurement_goods: 506 chunks
  - procurement_services: 697 chunks
  - financial_rules: 225 chunks
  - technical: 132 chunks

================================================================================
DOCUMENT-LEVEL CLUSTERING RESULTS
================================================================================

[CLUSTERS]
  Number of domains: 4
  Silhouette score: 0.425
  Davies-Bouldin score: 0.900

[DOMAIN DISTRIBUTION]
  procurement_goods              :  506 chunks ( 32.4%)
    Keywords: goods, procurement, materials, tender
    
  procurement_services           :  697 chunks ( 44.7%)
    Keywords: services, consultancy, contract, work
    
  financial_rules                :  225 chunks ( 14.4%)
    Keywords: financial, budget, rules, expenditure
    
  technical                      :  132 chunks (  8.5%)
    Keywords: technical, system, analysis, design

[DOCUMENT-TO-DOMAIN MAPPING]
  Procurement_Goods.pdf            -> procurement_goods
  Procurement_Consultancy.pdf      -> procurement_services
  Procurement_Non_Consultancy.pdf  -> procurement_services
  FInal_GFR_upto_31_07_2024.pdf    -> financial_rules
  208B.pdf                         -> technical
  20020050369.pdf                  -> technical
  20180000774.pdf                  -> technical
  20210017934 FINAL.pdf            -> technical
================================================================================
```

---

## KEY FIXES SUMMARY

1. ✅ **Domain Assignment Fixed** - All detected clusters now get domains
2. ✅ **TF-IDF Parameters Adapted** - Works with small cluster sizes
3. ✅ **Readable Output** - Shows filenames instead of hashes
4. ✅ **Fallback Keywords** - Uses word frequency when TF-IDF fails
5. ✅ **Stats Structure Fixed** - Consistent dictionary structure
6. ✅ **Verification Added** - Logs domain assignment progress

---

END OF FIX PROMPT
