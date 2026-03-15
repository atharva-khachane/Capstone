"""
Document-Level Domain Manager for intelligent domain detection and routing.

Clusters entire documents (not individual chunks) using K-means on
document-level embeddings, then assigns all chunks from each document
to the document's detected domain. Domain names are auto-generated
from TF-IDF keywords extracted from each cluster's content.

This replaces the chunk-level DomainManager (poor silhouette ~0.1) and
the rule-based DomainClassifier (not in methodology) with a single,
content-driven approach per ANTIGRAVITY_PROMPT_DOCUMENT_CLUSTERING.md.

100% OFFLINE - No external APIs.

References:
    - ANTIGRAVITY_PROMPT.md, Layer 2: Content-Based Clustering
    - ANTIGRAVITY_PROMPT_DOCUMENT_CLUSTERING.md: Document-level strategy
    - papers/10_Reimers2019_Sentence_BERT.pdf: Embedding model foundation
    - papers/Multi_Domain_RAG Paper.pdf: Multi-domain routing methodology
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import re

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.feature_extraction.text import TfidfVectorizer

from ..core.schemas import Chunk


@dataclass
class DocumentInfo:
    """Aggregated information about a single document for clustering."""
    doc_id: str
    chunk_ids: List[str]
    embeddings: np.ndarray
    doc_embedding: Optional[np.ndarray] = None
    text_content: str = ""
    num_chunks: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentLevelDomainManager:
    """
    Domain manager using document-level K-means clustering with TF-IDF naming.

    Strategy (per methodology):
        1. Group chunks by source document
        2. Compute document-level embedding (mean of chunk embeddings, L2-normed)
        3. Auto-tune K via silhouette score in [min_clusters, max_clusters]
        4. Cluster documents with K-means
        5. Generate meaningful domain names from TF-IDF top keywords
        6. Assign every chunk to its parent document's domain
        7. Compute L2-normalized domain centroids for query routing

    Parameters aligned with config.yaml and ANTIGRAVITY_PROMPT.md:
        min_clusters: 2  (methodology + config)
        max_clusters: 10 (methodology + config)
        similarity_threshold: 0.3 for routing (ANTIGRAVITY_PROMPT_DOMAIN_FIX.md)
    """

    def __init__(
        self,
        min_clusters: int = 2,
        max_clusters: int = 10,
        random_state: int = 42,
        n_init: int = 10,
    ):
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.n_init = n_init

        # Populated by detect_domains
        self.domains: Dict[str, np.ndarray] = {}          # domain_name -> centroid
        self.domain_chunks: Dict[str, List[str]] = {}     # domain_name -> [chunk_ids]
        self.doc_to_domain: Dict[str, str] = {}           # doc_id -> domain_name
        self.domain_keywords: Dict[str, List[str]] = {}   # domain_name -> [keywords]
        self.documents: Dict[str, DocumentInfo] = {}      # doc_id -> DocumentInfo

        # Quality metrics
        self.silhouette: float = 0.0
        self.davies_bouldin: float = 0.0
        self.optimal_k: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_domains(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """
        Detect domains using document-level clustering.

        Returns dict with:
            num_domains, method, silhouette_score, davies_bouldin_score,
            domain_distribution, domain_keywords
        """
        print("[DOMAIN] Starting document-level clustering...")

        # Step 1 -- group chunks by document
        self.documents = self._group_chunks_by_document(chunks)
        n_docs = len(self.documents)
        total_chunks = sum(d.num_chunks for d in self.documents.values())
        print(f"[DOMAIN] Grouped {total_chunks} chunks into {n_docs} documents")

        # Step 2 -- document embeddings
        doc_embeddings, doc_list = self._compute_document_embeddings(self.documents)
        print(f"[DOMAIN] Computed document embeddings ({doc_embeddings.shape})")

        # Step 3 -- find optimal K (chunk-weighted balance)
        doc_chunk_counts = np.array(
            [self.documents[did].num_chunks for did in doc_list], dtype=np.float64,
        )
        if n_docs <= 2:
            optimal_k = max(1, n_docs)
            self.silhouette = 0.0
        else:
            optimal_k, self.silhouette = self._find_optimal_k(
                doc_embeddings, n_docs, doc_chunk_counts,
            )
        self.optimal_k = optimal_k
        print(f"[DOMAIN] Optimal K = {optimal_k}  (silhouette = {self.silhouette:.3f})")

        # Step 4 -- cluster documents
        if optimal_k >= n_docs or optimal_k <= 1:
            labels = np.zeros(n_docs, dtype=int)
            optimal_k = 1
            self.optimal_k = 1
        else:
            kmeans = KMeans(
                n_clusters=optimal_k,
                random_state=self.random_state,
                n_init=self.n_init,
            )
            labels = kmeans.fit_predict(doc_embeddings)

        # Step 4b -- refine: split oversized clusters via sub-clustering
        labels, optimal_k = self._refine_large_clusters(
            doc_embeddings, labels, optimal_k, doc_chunk_counts, doc_list,
        )
        self.optimal_k = optimal_k

        # Step 5 -- TF-IDF domain names
        domain_names = self._generate_domain_names(
            self.documents, doc_list, labels, optimal_k,
        )

        # Step 6 -- assign chunks to domains
        self._assign_chunks_to_domains(chunks, doc_list, labels, domain_names)

        # Step 7 -- compute centroids
        self._compute_centroids(chunks)

        # Quality metrics (need >= 2 clusters and >= 2 docs per set)
        n_unique = len(set(labels))
        if n_unique >= 2 and n_docs > n_unique:
            self.silhouette = float(silhouette_score(doc_embeddings, labels))
            self.davies_bouldin = float(davies_bouldin_score(doc_embeddings, labels))

        self._print_results()

        return {
            "num_domains": len(self.domains),
            "method": "document_level_kmeans",
            "silhouette_score": self.silhouette,
            "davies_bouldin_score": self.davies_bouldin,
            "domain_distribution": {
                d: len(cids) for d, cids in self.domain_chunks.items()
            },
            "domain_keywords": self.domain_keywords,
        }

    def route_query(
        self,
        query_embedding: np.ndarray,
        top_k_domains: int = 3,
        similarity_threshold: float = 0.3,
    ) -> List[Tuple[str, float]]:
        """Route a query to the most relevant domains via cosine similarity."""
        if not self.domains:
            return []

        query_emb = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

        similarities = [
            (domain, float(np.dot(query_emb, centroid)))
            for domain, centroid in self.domains.items()
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)

        filtered = [
            (d, s) for d, s in similarities if s >= similarity_threshold
        ][:top_k_domains]

        if not filtered and similarities:
            filtered = [similarities[0]]

        return filtered

    def get_domain_stats(self) -> Dict[str, Any]:
        """Return per-domain and overall statistics."""
        stats: Dict[str, Any] = {}
        for domain, centroid in self.domains.items():
            stats[domain] = {
                "num_chunks": len(self.domain_chunks.get(domain, [])),
                "centroid_norm": float(np.linalg.norm(centroid)),
                "keywords": self.domain_keywords.get(domain, [])[:10],
            }
        stats["_overall"] = {
            "num_domains": len(self.domains),
            "silhouette_score": self.silhouette,
            "davies_bouldin_score": self.davies_bouldin,
        }
        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _group_chunks_by_document(
        self, chunks: List[Chunk],
    ) -> Dict[str, DocumentInfo]:
        groups: Dict[str, List[Chunk]] = defaultdict(list)
        for chunk in chunks:
            groups[chunk.doc_id].append(chunk)

        documents: Dict[str, DocumentInfo] = {}
        for doc_id, doc_chunks in groups.items():
            embeddings = np.array(
                [c.embedding for c in doc_chunks if c.has_embedding()]
            )
            text_content = " ".join(c.content for c in doc_chunks)
            metadata = doc_chunks[0].metadata if doc_chunks else {}

            documents[doc_id] = DocumentInfo(
                doc_id=doc_id,
                chunk_ids=[c.chunk_id for c in doc_chunks],
                embeddings=embeddings,
                text_content=text_content,
                num_chunks=len(doc_chunks),
                metadata=metadata,
            )
        return documents

    def _compute_document_embeddings(
        self, documents: Dict[str, DocumentInfo],
    ) -> Tuple[np.ndarray, List[str]]:
        doc_embeddings = []
        doc_list = []

        for doc_id, info in documents.items():
            if info.embeddings.size == 0:
                continue
            emb = np.mean(info.embeddings, axis=0)
            emb = emb / (np.linalg.norm(emb) + 1e-10)
            info.doc_embedding = emb
            doc_embeddings.append(emb)
            doc_list.append(doc_id)

        return np.array(doc_embeddings, dtype=np.float32), doc_list

    def _find_optimal_k(
        self,
        embeddings: np.ndarray,
        n_docs: int,
        doc_chunk_counts: np.ndarray,
    ) -> Tuple[int, float]:
        """Find optimal K using silhouette + chunk-weighted balance penalty.

        Pure silhouette tends to pick low K, merging semantically close but
        functionally different documents (e.g. GFR + Procurement).  We penalise
        based on the *chunk fraction* of the largest cluster -- not the document
        count -- because documents vary enormously in size.

        Reference: ANTIGRAVITY_PROMPT_DOCUMENT_CLUSTERING.md success criteria
        -- "No cluster > 40% of data" (data = chunks, not documents).
        """
        max_k = min(self.max_clusters, n_docs - 1)
        min_k = min(self.min_clusters, max_k)

        if max_k < 2:
            return 1, 0.0

        max_chunk_frac = 0.45    # ideal max chunk fraction per cluster
        balance_weight = 0.50    # strong penalty for imbalance

        best_k = min_k
        best_adjusted = -1.0
        best_raw_sil = -1.0
        total_chunks = doc_chunk_counts.sum()

        print(f"[DOMAIN] Searching K in [{min_k}, {max_k}]...")
        for k in range(min_k, max_k + 1):
            km = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=self.n_init,
            )
            lbl = km.fit_predict(embeddings)
            sil = float(silhouette_score(embeddings, lbl))

            # chunk-weighted largest cluster fraction
            cluster_chunks = np.zeros(k)
            for ci in range(k):
                cluster_chunks[ci] = doc_chunk_counts[lbl == ci].sum()
            largest_chunk_frac = cluster_chunks.max() / total_chunks

            penalty = max(
                0.0,
                (largest_chunk_frac - max_chunk_frac) / (1.0 - max_chunk_frac),
            )
            adjusted = sil * (1.0 - balance_weight * penalty)

            print(
                f"  K={k}: silhouette={sil:.3f}  largest_chunk_pct={largest_chunk_frac:.1%}"
                f"  adjusted={adjusted:.3f}"
            )
            if adjusted > best_adjusted:
                best_adjusted = adjusted
                best_raw_sil = sil
                best_k = k

        return best_k, best_raw_sil

    # -- Hierarchical refinement ----------------------------------------

    def _refine_large_clusters(
        self,
        doc_embeddings: np.ndarray,
        labels: np.ndarray,
        n_clusters: int,
        doc_chunk_counts: np.ndarray,
        doc_list: List[str],
        max_chunk_frac: float = 0.45,
    ) -> Tuple[np.ndarray, int]:
        """Split any cluster whose chunk fraction exceeds *max_chunk_frac*.

        For each oversized cluster with >= 3 documents, attempt K=2
        sub-clustering.  Accept the split only if the two sub-clusters
        have reasonable separation (silhouette > -0.1 on those docs).

        Returns updated (labels, n_clusters).
        """
        total_chunks = doc_chunk_counts.sum()
        labels = labels.copy()
        next_id = n_clusters

        for cid in range(n_clusters):
            mask = labels == cid
            cluster_chunks = doc_chunk_counts[mask].sum()
            frac = cluster_chunks / total_chunks

            if frac <= max_chunk_frac or mask.sum() < 3:
                continue

            sub_embs = doc_embeddings[mask]
            sub_chunk_counts = doc_chunk_counts[mask]

            km_sub = KMeans(n_clusters=2, random_state=self.random_state, n_init=10)
            sub_labels = km_sub.fit_predict(sub_embs)

            # check that neither sub-cluster is trivially empty
            if len(set(sub_labels)) < 2:
                continue

            # check sub-cluster chunk balance is actually better
            sub0_chunks = sub_chunk_counts[sub_labels == 0].sum()
            sub1_chunks = sub_chunk_counts[sub_labels == 1].sum()
            new_max_frac = max(sub0_chunks, sub1_chunks) / total_chunks

            if new_max_frac >= frac:
                # sub-clustering didn't improve balance
                continue

            sub_sil = float(silhouette_score(sub_embs, sub_labels))
            if sub_sil < -0.1:
                continue

            # Accept the split
            indices = np.where(mask)[0]
            for idx, sl in zip(indices, sub_labels):
                if sl == 0:
                    labels[idx] = cid          # keep original id for sub-0
                else:
                    labels[idx] = next_id      # new id for sub-1

            next_id += 1
            print(
                f"[DOMAIN] Refined cluster {cid}: split into 2 sub-clusters "
                f"({sub0_chunks:.0f} / {sub1_chunks:.0f} chunks, "
                f"sub-silhouette={sub_sil:.3f})"
            )

        # Re-index labels to be contiguous 0..N-1
        unique = sorted(set(labels))
        remap = {old: new for new, old in enumerate(unique)}
        labels = np.array([remap[l] for l in labels])
        return labels, len(unique)

    # -- TF-IDF domain naming (methodology lines 1065-1103) -----------

    def _generate_domain_names(
        self,
        documents: Dict[str, DocumentInfo],
        doc_list: List[str],
        labels: np.ndarray,
        num_clusters: int,
    ) -> Dict[int, str]:
        cluster_texts: Dict[int, List[str]] = defaultdict(list)
        for i, doc_id in enumerate(doc_list):
            cluster_texts[int(labels[i])].append(documents[doc_id].text_content)

        domain_names: Dict[int, str] = {}
        for cid in range(num_clusters):
            texts = cluster_texts.get(cid, [])
            if not texts:
                domain_names[cid] = f"domain_{cid}"
                continue

            combined = " ".join(texts)
            keywords = self._extract_keywords(combined)
            name = self._keywords_to_name(keywords[:3])
            domain_names[cid] = name
            self.domain_keywords[name] = keywords

        # Handle duplicate names by appending suffix
        seen: Dict[str, int] = {}
        for cid in sorted(domain_names):
            name = domain_names[cid]
            if name in seen:
                seen[name] += 1
                domain_names[cid] = f"{name}_{seen[name]}"
            else:
                seen[name] = 0

        return domain_names

    def _extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract top TF-IDF keywords from text, with robust fallback."""
        gov_stop_words = [
            "shall", "may", "must", "should", "will", "can", "document",
            "section", "clause", "para", "rule", "act", "said", "thereof",
            "herein", "hereof", "hereby", "pursuant", "following", "provided",
            "however", "accordance", "subject", "respect", "manner", "period",
            "case", "extent", "purpose", "required", "specified", "applicable",
            "also", "would", "page", "chapter",
        ]

        try:
            n_docs_for_tfidf = max(2, len(text.split(". ")))
            sentences = text.split(". ")
            if len(sentences) < 2:
                sentences = [text[:len(text)//2], text[len(text)//2:]]

            vectorizer = TfidfVectorizer(
                max_features=top_n * 2,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
                max_df=max(0.95, 1.0 - 1.0 / max(len(sentences), 2)),
            )
            tfidf = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()

            scores = np.asarray(tfidf.sum(axis=0)).flatten()
            top_idx = scores.argsort()[-top_n * 2:][::-1]
            raw = [feature_names[i] for i in top_idx]
            filtered = [
                kw for kw in raw if kw.lower() not in gov_stop_words
            ]
            return filtered[:top_n] if filtered else raw[:top_n]

        except Exception:
            return self._extract_keywords_frequency(text, top_n)

    def _extract_keywords_frequency(self, text: str, top_n: int = 10) -> List[str]:
        """Frequency-based fallback when TF-IDF fails."""
        stop = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "shall", "should", "may", "might", "can", "could", "would",
            "of", "in", "to", "for", "with", "on", "at", "by", "from",
            "as", "or", "and", "but", "not", "this", "that", "these",
            "those", "it", "its", "such", "any", "all", "each", "every",
        }
        words = re.findall(r"[a-z]{3,}", text.lower())
        counts = Counter(w for w in words if w not in stop)
        return [w for w, _ in counts.most_common(top_n)]

    @staticmethod
    def _keywords_to_name(keywords: List[str]) -> str:
        if not keywords:
            return "unknown_domain"
        parts = []
        for kw in keywords[:2]:
            clean = re.sub(r"[^a-z0-9\s]", "", kw.lower()).strip().replace(" ", "_")
            if clean:
                parts.append(clean)
        name = "_".join(parts) if parts else "unknown_domain"
        return name[:50]

    # -- Chunk assignment and centroid computation ----------------------

    def _assign_chunks_to_domains(
        self,
        chunks: List[Chunk],
        doc_list: List[str],
        labels: np.ndarray,
        domain_names: Dict[int, str],
    ) -> None:
        self.doc_to_domain.clear()
        self.domain_chunks.clear()

        for i, doc_id in enumerate(doc_list):
            self.doc_to_domain[doc_id] = domain_names[int(labels[i])]

        for chunk in chunks:
            domain = self.doc_to_domain.get(chunk.doc_id, "unknown")
            chunk.domain = domain
            self.domain_chunks.setdefault(domain, []).append(chunk.chunk_id)

    def _compute_centroids(self, chunks: List[Chunk]) -> None:
        domain_embeddings: Dict[str, List[np.ndarray]] = defaultdict(list)
        for chunk in chunks:
            if chunk.has_embedding() and chunk.domain:
                domain_embeddings[chunk.domain].append(chunk.embedding)

        self.domains.clear()
        for domain, embs in domain_embeddings.items():
            centroid = np.mean(embs, axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
            self.domains[domain] = centroid

        print(f"[DOMAIN] Computed centroids for {len(self.domains)} domains")

    # -- Reporting -----------------------------------------------------

    def _print_results(self) -> None:
        total = sum(len(cids) for cids in self.domain_chunks.values())
        print()
        print("=" * 72)
        print("  DOCUMENT-LEVEL CLUSTERING RESULTS")
        print("=" * 72)
        print(f"  Domains detected   : {len(self.domains)}")
        print(f"  Silhouette score   : {self.silhouette:.3f}")
        print(f"  Davies-Bouldin     : {self.davies_bouldin:.3f}")
        print()
        print("  Domain Distribution:")
        print("  " + "-" * 68)
        for domain in sorted(self.domain_chunks):
            n = len(self.domain_chunks[domain])
            pct = 100.0 * n / total if total else 0
            kws = ", ".join(self.domain_keywords.get(domain, [])[:5])
            print(f"    {domain:35s} {n:5d} chunks ({pct:5.1f}%)")
            if kws:
                print(f"      keywords: {kws}")
        print()
        print("  Document -> Domain Mapping:")
        print("  " + "-" * 68)
        for doc_id, domain in sorted(self.doc_to_domain.items()):
            filepath = self.documents.get(doc_id, DocumentInfo(
                doc_id=doc_id, chunk_ids=[], embeddings=np.array([]),
            )).metadata.get("filepath", doc_id)
            label = filepath if isinstance(filepath, str) else doc_id
            if len(label) > 45:
                label = "..." + label[-42:]
            print(f"    {label:45s} -> {domain}")
        print("=" * 72)
        print()
