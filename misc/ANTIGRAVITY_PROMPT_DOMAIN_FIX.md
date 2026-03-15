# ANTIGRAVITY PROMPT: DOMAIN ROUTING FIX FOR SL-RAG PIPELINE

## CRITICAL PROBLEM IDENTIFIED

**Issue:** Multi-document retrieval is failing to retrieve from correct document domains.

**Symptoms:**
- Query "consultancy services" → Returns results from `procurement_goods` instead of `procurement_consultancy`
- Query "non-consultancy services" → Returns results from `financial_rules` instead of `procurement_non_consultancy`
- 697 chunks (47% of corpus) are being systematically ignored
- Domain coverage: Only 40% (2/5 queries hit expected domains)

**Root Causes:**
1. ❌ Content-based clustering merged multiple documents into single domains
2. ❌ Domain routing may not be actively filtering retrieval
3. ❌ Domain centroids may not be sufficiently distinct
4. ❌ Top-1 domain routing is too restrictive for cross-document queries

---

## MISSION: FIX DOMAIN ROUTING & IMPROVE RETRIEVAL ACCURACY

Generate code fixes to achieve:
- ✅ **100% domain coverage** for all queries
- ✅ **Distinct domains** for each document type
- ✅ **Active domain filtering** during retrieval
- ✅ **Multi-domain routing** for cross-document queries
- ✅ **Validation** that correct documents are being retrieved

---

## APPROACH: THREE-TIER FIX STRATEGY

Implement all three tiers for maximum robustness:

### **TIER 1: Filename-Based Domain Assignment** (Quick Fix - HIGH PRIORITY)
### **TIER 2: Enhanced Content-Based Clustering** (Medium Priority)
### **TIER 3: Hybrid Domain Routing** (Advanced - Best Quality)

---

## TIER 1: FILENAME-BASED DOMAIN ASSIGNMENT

**Purpose:** Guarantee each document gets a unique, meaningful domain based on its filename.

**Implementation:**

### File: `core/domain_manager.py` (Enhanced)

```python
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

class DomainManager:
    """
    Enhanced domain manager with filename-based fallback.
    
    Strategy:
    1. Try content-based clustering first
    2. If clusters are too broad, use filename-based domains
    3. Support manual domain override
    4. Enable multi-domain routing
    """
    
    def __init__(self, 
                 min_clusters: int = 5,
                 max_clusters: int = 15,
                 use_filename_domains: bool = True,
                 enable_multi_domain_routing: bool = True):
        self.domains: Dict[str, np.ndarray] = {}
        self.domain_chunks: Dict[str, List[str]] = {}
        self.cluster_labels: Dict[str, int] = {}
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.use_filename_domains = use_filename_domains
        self.enable_multi_domain_routing = enable_multi_domain_routing
        
        # Filename-to-domain mapping
        self.filename_domain_map = {}
    
    def extract_domain_from_filename(self, filepath: str) -> str:
        """
        Extract domain from filename using intelligent pattern matching.
        
        Examples:
        - "FInal_GFR_upto_31_07_2024.pdf" → "financial_rules"
        - "Procurement_Goods.pdf" → "procurement_goods"
        - "Procurement_Consultancy.pdf" → "procurement_consultancy"
        - "Procurement_Non_Consultancy.pdf" → "procurement_non_consultancy"
        - "208B.pdf" → "technical_memo"
        - "20020050369.pdf" → "technical_report"
        """
        filename = Path(filepath).stem.lower()
        
        # Define domain extraction rules
        domain_patterns = [
            # GFR and financial rules
            (r'gfr|financial.*rule', 'financial_rules'),
            
            # Procurement types (order matters - most specific first!)
            (r'procurement.*non.*consult', 'procurement_non_consultancy'),
            (r'procurement.*consult', 'procurement_consultancy'),
            (r'procurement.*goods', 'procurement_goods'),
            (r'procurement.*works', 'procurement_works'),
            (r'procurement.*services', 'procurement_services'),
            (r'procurement', 'procurement_general'),
            
            # Technical documents
            (r'^\d{3}[a-z]$', 'technical_memo'),  # e.g., 208B
            (r'^\d{11}$', 'technical_report'),    # e.g., 20020050369
            (r'technical.*memo', 'technical_memo'),
            (r'failure.*analysis', 'technical_failure'),
            (r'telemetry', 'technical_telemetry'),
            
            # Quality and audit
            (r'quality.*assurance|qa.*report', 'quality_assurance'),
            (r'audit', 'audit_report'),
            
            # Default
            (r'.*', 'general')
        ]
        
        # Match against patterns
        for pattern, domain in domain_patterns:
            if re.search(pattern, filename):
                return domain
        
        return 'general'
    
    def assign_domains_from_filenames(self, chunks: List['Chunk']) -> None:
        """
        Assign domains based on source document filenames.
        
        This guarantees each document type gets a unique domain.
        """
        print("[DOMAIN] Using filename-based domain assignment...")
        
        # Group chunks by source document
        doc_to_chunks = {}
        for chunk in chunks:
            doc_id = chunk.doc_id
            if doc_id not in doc_to_chunks:
                doc_to_chunks[doc_id] = []
            doc_to_chunks[doc_id].append(chunk)
        
        # Assign domain to each document
        domain_stats = {}
        for doc_id, doc_chunks in doc_to_chunks.items():
            # Get filepath from first chunk's metadata
            filepath = doc_chunks[0].metadata.get('filepath', '')
            
            # Extract domain
            domain = self.extract_domain_from_filename(filepath)
            
            # Assign to all chunks from this document
            for chunk in doc_chunks:
                chunk.domain = domain
                self.cluster_labels[chunk.chunk_id] = domain
            
            # Track stats
            if domain not in domain_stats:
                domain_stats[domain] = 0
            domain_stats[domain] += len(doc_chunks)
            
            print(f"  [DOMAIN] {Path(filepath).name} → {domain} ({len(doc_chunks)} chunks)")
        
        # Compute centroids for each domain
        self.compute_centroids(chunks)
        
        print(f"\n[DOMAIN] ✓ Created {len(domain_stats)} domains:")
        for domain, count in sorted(domain_stats.items()):
            print(f"  - {domain}: {count} chunks")
        
        return {
            'num_domains': len(domain_stats),
            'domain_distribution': domain_stats,
            'method': 'filename_based'
        }
    
    def detect_domains_hybrid(self, chunks: List['Chunk']) -> Dict:
        """
        Hybrid approach: Use filename domains, but verify with content clustering.
        
        Algorithm:
        1. Assign initial domains from filenames
        2. Run K-means clustering on embeddings
        3. Check if content clusters align with filename domains
        4. If misalignment > 20%, flag for manual review
        5. Use filename domains as final assignment
        """
        print("[DOMAIN] Using hybrid domain detection...")
        
        # Step 1: Filename-based assignment
        filename_result = self.assign_domains_from_filenames(chunks)
        
        # Step 2: Content-based clustering for validation
        embeddings = np.array([chunk.embedding for chunk in chunks])
        
        # Determine optimal number of clusters
        n_clusters = min(len(set(c.domain for c in chunks)), self.max_clusters)
        
        if n_clusters < 2:
            print("[DOMAIN] ⚠️ Only 1 domain detected, skipping content validation")
            return filename_result
        
        # Cluster embeddings
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        content_labels = kmeans.fit_predict(embeddings)
        
        # Step 3: Check alignment between filename and content domains
        misalignment_count = 0
        domain_to_cluster = {}
        
        for i, chunk in enumerate(chunks):
            filename_domain = chunk.domain
            content_cluster = content_labels[i]
            
            # Track which cluster each domain maps to
            if filename_domain not in domain_to_cluster:
                domain_to_cluster[filename_domain] = []
            domain_to_cluster[filename_domain].append(content_cluster)
        
        # Calculate domain purity (% of chunks in dominant cluster)
        for domain, clusters in domain_to_cluster.items():
            from collections import Counter
            cluster_counts = Counter(clusters)
            dominant_cluster_count = cluster_counts.most_common(1)[0][1]
            purity = dominant_cluster_count / len(clusters)
            
            if purity < 0.8:
                misalignment_count += 1
                print(f"  [DOMAIN] ⚠️ {domain}: {purity:.1%} purity (content scattered across clusters)")
        
        misalignment_ratio = misalignment_count / len(domain_to_cluster) if domain_to_cluster else 0
        
        if misalignment_ratio > 0.2:
            print(f"[DOMAIN] ⚠️ {misalignment_ratio:.1%} domains show content misalignment")
            print("[DOMAIN] Consider manual review of domain assignments")
        else:
            print(f"[DOMAIN] ✓ {(1-misalignment_ratio):.1%} domain-content alignment")
        
        return {
            **filename_result,
            'content_alignment': 1 - misalignment_ratio,
            'misaligned_domains': misalignment_count
        }
    
    def compute_centroids(self, chunks: List['Chunk']) -> None:
        """
        Compute centroid embedding for each domain.
        """
        domain_embeddings = {}
        
        for chunk in chunks:
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
    
    def validate_domain_routing(self, 
                                test_queries: List[Tuple[str, List[str]]]) -> Dict:
        """
        Validate that domain routing works correctly.
        
        Args:
            test_queries: List of (query, expected_domains) tuples
        
        Returns:
            Validation report with accuracy metrics
        """
        from sentence_transformers import SentenceTransformer
        
        print("\n[VALIDATION] Testing domain routing...")
        
        # Load embedding model
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
        correct = 0
        total = len(test_queries)
        results = []
        
        for query, expected_domains in test_queries:
            # Generate query embedding
            query_emb = model.encode(query)
            
            # Route query
            routed = self.route_query(query_emb, top_k_domains=3)
            routed_domains = [d for d, _ in routed]
            
            # Check if any expected domain is in routed domains
            hit = any(expected in routed_domains for expected in expected_domains)
            
            if hit:
                correct += 1
            
            results.append({
                'query': query,
                'expected': expected_domains,
                'routed': routed_domains,
                'hit': hit
            })
            
            status = "✓" if hit else "✗"
            print(f"  {status} {query[:60]}")
            print(f"    Expected: {expected_domains}")
            print(f"    Routed:   {routed_domains}")
        
        accuracy = correct / total if total > 0 else 0
        
        print(f"\n[VALIDATION] Domain routing accuracy: {accuracy:.1%} ({correct}/{total})")
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'results': results
        }
```

---

## TIER 2: ENHANCED RETRIEVAL WITH DOMAIN FILTERING

**Purpose:** Ensure retrieval actually uses domain routing to filter chunks.

### File: `retrieval/hybrid_retriever.py` (Enhanced)

```python
class HybridRetriever:
    """Enhanced hybrid retriever with strict domain filtering."""
    
    def __init__(self, 
                 faiss_index,
                 chunks: List['Chunk'],
                 alpha: float = 0.7,
                 enable_domain_filtering: bool = True):
        self.faiss_index = faiss_index
        self.chunks = chunks
        self.alpha = alpha
        self.enable_domain_filtering = enable_domain_filtering
        
        # Build BM25 index
        tokenized_corpus = [chunk.content.split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Build domain-specific indices
        self._build_domain_indices()
    
    def _build_domain_indices(self):
        """Create per-domain chunk indices for fast filtering."""
        self.domain_to_indices = {}
        
        for idx, chunk in enumerate(self.chunks):
            domain = chunk.domain
            if domain not in self.domain_to_indices:
                self.domain_to_indices[domain] = []
            self.domain_to_indices[domain].append(idx)
        
        print(f"[HYBRID] Built indices for {len(self.domain_to_indices)} domains")
    
    def retrieve(self, 
                 query: str,
                 query_embedding: np.ndarray,
                 k: int = 20,
                 filter_domains: Optional[List[str]] = None) -> List[Tuple['Chunk', float]]:
        """
        Hybrid retrieval with STRICT domain filtering.
        
        Args:
            query: Query text
            query_embedding: Query vector
            k: Number of results
            filter_domains: List of domains to search (if None, search all)
        
        Returns:
            List of (Chunk, score) tuples
        """
        
        # Step 1: Determine which chunks to search
        if filter_domains and self.enable_domain_filtering:
            # Get chunk indices for specified domains
            valid_indices = []
            for domain in filter_domains:
                valid_indices.extend(self.domain_to_indices.get(domain, []))
            
            if not valid_indices:
                print(f"[HYBRID] ⚠️ No chunks found in domains: {filter_domains}")
                return []
            
            valid_indices = list(set(valid_indices))  # Deduplicate
            print(f"[HYBRID] Searching {len(valid_indices)} chunks in domains: {filter_domains}")
        else:
            # Search all chunks
            valid_indices = list(range(len(self.chunks)))
            print(f"[HYBRID] Searching all {len(valid_indices)} chunks")
        
        # Step 2: Dense retrieval (FAISS)
        dense_scores, dense_indices = self.faiss_index.search(query_embedding, k=len(valid_indices))
        
        # Filter to valid domain indices
        dense_results = {}
        for idx, score in zip(dense_indices, dense_scores):
            if idx in valid_indices:
                dense_results[idx] = float(score)
        
        # Step 3: BM25 retrieval
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        bm25_results = {}
        for idx in valid_indices:
            bm25_results[idx] = float(bm25_scores[idx])
        
        # Step 4: Normalize scores
        if dense_results:
            dense_vals = np.array(list(dense_results.values()))
            dense_min, dense_max = dense_vals.min(), dense_vals.max()
            if dense_max > dense_min:
                for idx in dense_results:
                    dense_results[idx] = (dense_results[idx] - dense_min) / (dense_max - dense_min)
        
        if bm25_results:
            bm25_vals = np.array(list(bm25_results.values()))
            bm25_min, bm25_max = bm25_vals.min(), bm25_vals.max()
            if bm25_max > bm25_min:
                for idx in bm25_results:
                    bm25_results[idx] = (bm25_results[idx] - bm25_min) / (bm25_max - bm25_min)
        
        # Step 5: Fuse scores
        fused_scores = {}
        all_indices = set(dense_results.keys()) | set(bm25_results.keys())
        
        for idx in all_indices:
            dense_score = dense_results.get(idx, 0.0)
            bm25_score = bm25_results.get(idx, 0.0)
            fused_scores[idx] = self.alpha * dense_score + (1 - self.alpha) * bm25_score
        
        # Step 6: Sort and return top-k
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        return [(self.chunks[idx], score) for idx, score in sorted_results]
```

---

## TIER 3: MULTI-DOMAIN RETRIEVAL PIPELINE

**Purpose:** Enable queries to retrieve from multiple relevant domains.

### File: `retrieval/retrieval_pipeline.py` (Enhanced)

```python
class RetrievalPipeline:
    """
    Enhanced retrieval pipeline with multi-domain support.
    """
    
    def __init__(self, 
                 embedding_generator,
                 domain_manager,
                 hybrid_retriever,
                 cross_encoder,
                 similarity_threshold: float = 0.5,
                 multi_domain_retrieval: bool = True,
                 top_k_domains: int = 3):
        self.embedding_generator = embedding_generator
        self.domain_manager = domain_manager
        self.hybrid_retriever = hybrid_retriever
        self.cross_encoder = cross_encoder
        self.similarity_threshold = similarity_threshold
        self.multi_domain_retrieval = multi_domain_retrieval
        self.top_k_domains = top_k_domains
    
    def retrieve(self, 
                 query: str, 
                 top_k: int = 5,
                 enable_reranking: bool = True,
                 debug: bool = False) -> List[Tuple['Chunk', float]]:
        """
        Multi-domain retrieval with reranking.
        
        Algorithm:
        1. Generate query embedding
        2. Route to top-K relevant domains
        3. Retrieve from each domain separately
        4. Merge and deduplicate results
        5. Cross-encoder rerank
        6. Return top-K final results
        """
        
        # Step 1: Generate query embedding
        query_emb = self.embedding_generator.embed_query(query)
        
        # Step 2: Route to domains
        routed_domains = self.domain_manager.route_query(
            query_emb, 
            top_k_domains=self.top_k_domains
        )
        
        if debug:
            print(f"\n[RETRIEVE] Query: {query}")
            print(f"[RETRIEVE] Routed to {len(routed_domains)} domains:")
            for domain, score in routed_domains:
                print(f"  - {domain}: {score:.3f}")
        
        # Step 3: Retrieve from each domain
        domain_names = [d for d, _ in routed_domains]
        
        if self.multi_domain_retrieval:
            # Retrieve ~10 chunks per domain
            chunks_per_domain = max(10, top_k * 2 // len(domain_names)) if domain_names else top_k * 4
            
            all_results = []
            for domain in domain_names:
                domain_results = self.hybrid_retriever.retrieve(
                    query=query,
                    query_embedding=query_emb,
                    k=chunks_per_domain,
                    filter_domains=[domain]
                )
                all_results.extend(domain_results)
                
                if debug:
                    print(f"[RETRIEVE] From {domain}: {len(domain_results)} chunks")
            
            # Deduplicate by chunk_id
            seen = set()
            unique_results = []
            for chunk, score in all_results:
                if chunk.chunk_id not in seen:
                    seen.add(chunk.chunk_id)
                    unique_results.append((chunk, score))
            
            results = unique_results
        else:
            # Single retrieval across all routed domains
            results = self.hybrid_retriever.retrieve(
                query=query,
                query_embedding=query_emb,
                k=top_k * 4,
                filter_domains=domain_names
            )
        
        if debug:
            print(f"[RETRIEVE] Total candidates: {len(results)}")
        
        # Step 4: Filter by similarity threshold
        filtered = [(c, s) for c, s in results if s >= self.similarity_threshold]
        
        if not filtered:
            # If nothing passes threshold, return top-5 anyway
            filtered = results[:5]
        
        # Step 5: Cross-encoder reranking
        if enable_reranking and self.cross_encoder and filtered:
            if debug:
                print(f"[RERANK] Reranking {len(filtered)} candidates...")
            
            reranked = self.cross_encoder.rerank(query, filtered, top_k=top_k)
            
            if debug:
                print(f"[RERANK] Returned top-{len(reranked)} results")
            
            return reranked
        
        # Return top-k without reranking
        return filtered[:top_k]
```

---

## TESTING & VALIDATION

### File: `tests/test_domain_routing.py`

```python
import pytest
from core.domain_manager import DomainManager
from core.embedding_generator import EmbeddingGenerator

def test_filename_domain_extraction():
    """Test domain extraction from filenames."""
    dm = DomainManager()
    
    test_cases = [
        ("FInal_GFR_upto_31_07_2024.pdf", "financial_rules"),
        ("Procurement_Goods.pdf", "procurement_goods"),
        ("Procurement_Consultancy.pdf", "procurement_consultancy"),
        ("Procurement_Non_Consultancy.pdf", "procurement_non_consultancy"),
        ("208B.pdf", "technical_memo"),
        ("20020050369.pdf", "technical_report"),
    ]
    
    for filepath, expected_domain in test_cases:
        result = dm.extract_domain_from_filename(filepath)
        assert result == expected_domain, f"Failed for {filepath}: got {result}, expected {expected_domain}"
    
    print("✓ All filename domain extractions correct")

def test_domain_routing_accuracy():
    """Test that queries route to correct domains."""
    
    # This would use your actual loaded chunks and domain manager
    # For now, pseudocode:
    
    test_queries = [
        ("What are the financial rules for procurement?", ["financial_rules", "procurement_goods"]),
        ("Guidelines for consultancy services", ["procurement_consultancy"]),
        ("How to procure goods?", ["procurement_goods"]),
        ("Disposal of surplus items", ["financial_rules"]),
        ("Non-consultancy services procedures", ["procurement_non_consultancy"]),
    ]
    
    dm = DomainManager()
    # Load your actual chunks here
    # dm.assign_domains_from_filenames(chunks)
    
    results = dm.validate_domain_routing(test_queries)
    
    assert results['accuracy'] >= 0.8, f"Domain routing accuracy too low: {results['accuracy']}"
    print(f"✓ Domain routing accuracy: {results['accuracy']:.1%}")

def test_multi_domain_retrieval():
    """Test that retrieval works across multiple domains."""
    
    # Pseudocode - implement with actual pipeline
    query = "What are the financial rules for procurement?"
    
    # Should retrieve from BOTH financial_rules AND procurement_goods
    results = retrieval_pipeline.retrieve(query, top_k=5, debug=True)
    
    domains_found = set(chunk.domain for chunk, _ in results)
    
    assert "financial_rules" in domains_found, "Missing financial_rules domain"
    assert "procurement_goods" in domains_found, "Missing procurement_goods domain"
    
    print(f"✓ Multi-domain retrieval working: {domains_found}")
```

---

## IMPLEMENTATION CHECKLIST

### Phase 1: Filename-Based Domains (30 minutes)

```bash
# Step 1: Update DomainManager
# - Add extract_domain_from_filename() method
# - Add assign_domains_from_filenames() method

# Step 2: Update ingestion pipeline
# In your ingestion script:
domain_manager = DomainManager(use_filename_domains=True)
domain_result = domain_manager.assign_domains_from_filenames(chunks)

# Step 3: Test
python test_domain_routing.py
```

**Expected Output:**
```
[DOMAIN] Using filename-based domain assignment...
  [DOMAIN] FInal_GFR_upto_31_07_2024.pdf → financial_rules (225 chunks)
  [DOMAIN] Procurement_Goods.pdf → procurement_goods (506 chunks)
  [DOMAIN] Procurement_Consultancy.pdf → procurement_consultancy (347 chunks)
  [DOMAIN] Procurement_Non_Consultancy.pdf → procurement_non_consultancy (350 chunks)
  [DOMAIN] 208B.pdf → technical_memo (25 chunks)
  [DOMAIN] 20020050369.pdf → technical_report (12 chunks)
  [DOMAIN] 20180000774.pdf → technical_report (2 chunks)

[DOMAIN] ✓ Created 6 domains:
  - financial_rules: 225 chunks
  - procurement_goods: 506 chunks
  - procurement_consultancy: 347 chunks
  - procurement_non_consultancy: 350 chunks
  - technical_memo: 25 chunks
  - technical_report: 14 chunks
```

### Phase 2: Enable Domain Filtering (15 minutes)

```bash
# Update HybridRetriever
# - Add _build_domain_indices() method
# - Update retrieve() to filter by domains

# Update RetrievalPipeline
# - Enable multi_domain_retrieval=True
# - Set top_k_domains=3
```

### Phase 3: Validation (15 minutes)

```bash
# Run multi-document demo again
python demo_multi_document.py

# Expected improvements:
# - Query 2 (consultancy) → Should hit procurement_consultancy ✓
# - Query 5 (non-consultancy) → Should hit procurement_non_consultancy ✓
# - Domain coverage: 100% (5/5 queries) ✓
```

---

## EXPECTED RESULTS AFTER FIX

### Before Fix:
```
Query 2: Guidelines for consultancy services
  Retrieved from: procurement_goods, financial_rules ❌
  Expected domain coverage: 0%

Query 5: Non-consultancy services procedures
  Retrieved from: procurement_goods, financial_rules ❌
  Expected domain coverage: 0%
```

### After Fix:
```
Query 2: Guidelines for consultancy services
  Routed to: procurement_consultancy (0.82), procurement_goods (0.45), financial_rules (0.38)
  Retrieved from: procurement_consultancy ✓
  Top result: Procurement_Consultancy.pdf, Score: 5.2
  Expected domain coverage: 100% ✓

Query 5: Non-consultancy services procedures
  Routed to: procurement_non_consultancy (0.79), procurement_goods (0.52)
  Retrieved from: procurement_non_consultancy ✓
  Top result: Procurement_Non_Consultancy.pdf, Score: 4.8
  Expected domain coverage: 100% ✓
```

---

## SUCCESS CRITERIA

After implementing these fixes:

✅ **Domain Uniqueness**: Each document type has distinct domain
✅ **Domain Routing Accuracy**: ≥ 80% queries route to correct domains
✅ **Domain Coverage**: 100% of queries hit expected domains
✅ **Multi-Domain Support**: Queries can retrieve from 2-3 relevant domains
✅ **No Missing Documents**: All 7 documents contribute to retrieval
✅ **Validation Passes**: All test cases pass

---

## TROUBLESHOOTING

### Issue: Domains still merged after filename assignment
**Solution:** Check that `use_filename_domains=True` in DomainManager initialization

### Issue: Domain routing not filtering chunks
**Solution:** Verify `enable_domain_filtering=True` in HybridRetriever and `filter_domains` parameter is passed

### Issue: Poor domain routing accuracy
**Solution:** Increase `top_k_domains` from 1 to 3, or adjust `similarity_threshold`

### Issue: Cross-domain queries fail
**Solution:** Enable `multi_domain_retrieval=True` in RetrievalPipeline

---

## DELIVERABLES

After implementing this fix, you should have:

1. ✅ **Updated DomainManager** with filename-based domain assignment
2. ✅ **Enhanced HybridRetriever** with domain filtering
3. ✅ **Improved RetrievalPipeline** with multi-domain support
4. ✅ **Test suite** validating domain routing
5. ✅ **100% domain coverage** on all test queries
6. ✅ **Distinct domains** for each document type

---

## NEXT STEPS AFTER FIX

Once domain routing is fixed:

1. **Re-run all test queries** - Verify 100% domain coverage
2. **Benchmark retrieval accuracy** - Measure Recall@5, MRR
3. **Proceed to Phase 4** - Validation & Security
4. **Generate compliance report** - Show system is production-ready

---

END OF DOMAIN ROUTING FIX PROMPT
