"""
Retrieval Pipeline with Multi-Domain Support.

Orchestrates the complete retrieval process:
1. Query embedding generation
2. Domain routing
3. Multi-domain retrieval
4. Result fusion and deduplication
5. Cross-encoder reranking
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from ..core.schemas import Chunk
from ..core.embedding_generator import EmbeddingGenerator
from .document_level_domain_manager import DocumentLevelDomainManager
from .hybrid_retriever import HybridRetriever
from .reranker import CrossEncoderReranker
from .query_preprocessor import QueryPreprocessor
from .policy import RetrievalPolicy
from .query_cache import QueryResultCache
from .cross_chunk_resolver import CrossChunkResolver


class RetrievalPipeline:
    """
    Enhanced retrieval pipeline with multi-domain support.
    
    This pipeline implements intelligent query routing across multiple
    document domains and combines results for optimal retrieval quality.
    """
    
    def __init__(self, 
                 embedding_generator: EmbeddingGenerator,
                 domain_manager: DocumentLevelDomainManager,
                 hybrid_retriever: HybridRetriever,
                 cross_encoder: Optional[CrossEncoderReranker] = None,
                 similarity_threshold: float = 0.5,
                 multi_domain_retrieval: bool = True,
                 top_k_domains: int = 3,
                 rerank_candidates: int = 20,
                 final_top_k: int = 5,
                 initial_top_k_candidates: int = 20,
                 cache_ttl_seconds: int = 3600,
                 cache_max_entries: int = 200,
                 cache_enabled: bool = True):
        """
        Initialize retrieval pipeline.
        
        Args:
            embedding_generator: For query embeddings
            domain_manager: For domain routing
            hybrid_retriever: For hybrid BM25+Dense search
            cross_encoder: Optional reranker for final results
            similarity_threshold: Min similarity for domain routing
            multi_domain_retrieval: Whether to retrieve from multiple domains
            top_k_domains: Number of domains to route to
            cache_ttl_seconds: TTL for query result cache (0 = disable expiry)
            cache_max_entries: Max distinct (query, top_k) entries in cache
            cache_enabled: Whether to enable query result caching
        """
        self.embedding_generator = embedding_generator
        self.domain_manager = domain_manager
        self.hybrid_retriever = hybrid_retriever
        self.cross_encoder = cross_encoder
        self.similarity_threshold = similarity_threshold
        self.multi_domain_retrieval = multi_domain_retrieval
        self.top_k_domains = top_k_domains
        self.query_preprocessor = QueryPreprocessor()
        self.policy = RetrievalPolicy(
            similarity_threshold=similarity_threshold,
            rerank_candidates=rerank_candidates,
            final_top_k=final_top_k,
            initial_top_k_candidates=initial_top_k_candidates,
        )
        self._cache: Optional[QueryResultCache] = (
            QueryResultCache(ttl_seconds=cache_ttl_seconds, max_entries=cache_max_entries)
            if cache_enabled else None
        )
        # Fix 3: CrossChunkResolver removes near-duplicate adjacent chunks to
        # improve context precision (deepEval context_precision metric).
        self.cross_chunk_resolver = CrossChunkResolver(bucket_size=8, min_score_gap=0.0)

        print(f"[PIPELINE] Initialized multi-domain retrieval pipeline")
        print(f"[PIPELINE] Multi-domain: {multi_domain_retrieval}, Top-k domains: {top_k_domains}")
        if self._cache:
            print(f"[PIPELINE] Query cache: ENABLED (TTL={cache_ttl_seconds}s, max={cache_max_entries})")
    
    def retrieve(self, 
                 query: str, 
                 top_k: int = 5,
                 enable_reranking: bool = True,
                 debug: bool = False) -> List[Tuple[Chunk, float]]:
        """
        Multi-domain retrieval with reranking.
        
        Algorithm:
        1. Generate query embedding
        2. Route to top-K relevant domains
        3. Retrieve from each domain separately (if multi_domain_retrieval)
        4. Merge and deduplicate results
        5. Cross-encoder rerank (if enabled)
        6. Return top-K final results
        
        Args:
            query: Query string
            top_k: Number of final results to return
            enable_reranking: Whether to use cross-encoder reranking
            debug: Print debug information
            
        Returns:
            List of (chunk, score) tuples
        """
        
        # Step 0: Preprocess query (normalization + acronym expansion)
        query = self.query_preprocessor.preprocess(query)

        # Cache check — return immediately on hit to avoid reranking overhead
        if self._cache:
            cached = self._cache.get(query, top_k)
            if cached is not None:
                print(f"[CACHE] Hit for query ({len(cached)} results) — {self._cache.stats()['hit_rate']:.0%} hit rate")
                return cached

        # Step 1: Generate query embedding
        query_emb = self.embedding_generator.generate_query_embedding(query, normalize=True)
        
        # Step 2: Route to domains
        routed_domains = self.domain_manager.route_query(
            query_emb, 
            top_k_domains=self.top_k_domains,
            similarity_threshold=self.similarity_threshold
        )
        
        if debug:
            print(f"\n[RETRIEVE] Query: {query}")
            print(f"[RETRIEVE] Routed to {len(routed_domains)} domains:")
            for domain, score in routed_domains:
                print(f"  - {domain}: {score:.3f}")
        
        # Step 3: Retrieve from domains
        domain_names = [d for d, _ in routed_domains]
        
        if not domain_names:
            print("[RETRIEVE] WARNING: No domains matched, searching all")
            domain_names = None
        
        if self.multi_domain_retrieval and domain_names:
            # Retrieve from each domain separately, then merge
            all_results = []
            chunks_per_domain = max(
                10,
                self.policy.initial_top_k_candidates // max(len(domain_names), 1),
            )
            
            for domain in domain_names:
                domain_results = self.hybrid_retriever.search(
                    query=query,
                    top_k=chunks_per_domain,
                    top_k_candidates=max(chunks_per_domain * 2, self.policy.initial_top_k_candidates),
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
            results = self.hybrid_retriever.search(
                query=query,
                top_k=max(top_k * 2, self.policy.initial_top_k_candidates // 2),
                top_k_candidates=max(top_k * 4, self.policy.initial_top_k_candidates),
                filter_domains=domain_names
            )
        
        if debug:
            print(f"[RETRIEVE] Total candidates: {len(results)}")

        # Fix 4: Fallback to full corpus when domain routing yields too few results.
        # This handles cases where clustering assigns tech documents to unexpected
        # domain names (e.g. "figure_procurement") and filtered retrieval returns 0.
        if len(results) < top_k:
            if debug or len(results) == 0:
                print(f"[RETRIEVE] Only {len(results)} results from domain routing, "
                      f"falling back to full corpus")
            fallback = self.hybrid_retriever.search(
                query=query,
                top_k=max(top_k * 3, self.policy.initial_top_k_candidates),
                top_k_candidates=max(top_k * 4, self.policy.initial_top_k_candidates),
                filter_domains=None,
            )
            seen_ids = {chunk.chunk_id for chunk, _ in results}
            for chunk, score in fallback:
                if chunk.chunk_id not in seen_ids:
                    seen_ids.add(chunk.chunk_id)
                    results.append((chunk, score))

        # Step 4: Filter by similarity threshold (optional)
        # For hybrid scores, we might skip this as scores are already normalized
        
        # Step 5: Cross-encoder reranking
        if enable_reranking and self.cross_encoder and results:
            if debug:
                print(f"[RERANK] Reranking {len(results)} candidates...")

            target_top_k = self.policy.resolve_top_k(top_k)
            rerank_candidates = results[:self.policy.candidate_window(len(results))]
            reranked = self.cross_encoder.rerank(query, rerank_candidates, top_k=target_top_k)

            if debug:
                print(f"[RERANK] Returned top-{len(reranked)} results")

            # Fix 4b: prevent cross-domain context dilution
            reranked = self._consolidate_sources(reranked)

            # Fix 3: de-duplicate overlapping chunks to improve context precision
            reranked = self.cross_chunk_resolver.resolve(reranked)[:target_top_k]

            if self._cache:
                self._cache.put(query, top_k, reranked)
            return reranked

        # Return top-k without reranking
        final = results[:self.policy.resolve_top_k(top_k)]
        if self._cache:
            self._cache.put(query, top_k, final)
        return final
    
    def _consolidate_sources(
        self,
        results: List[Tuple[Chunk, float]],
    ) -> List[Tuple[Chunk, float]]:
        """Keep only the best domain group when results span multiple domains.

        Prevents context dilution from incompatible document clusters
        (e.g. mixing financial rules with procurement in a single context window).
        Only consolidates when one domain clearly dominates (>60% of aggregate
        score); otherwise keeps all results to allow genuine multi-domain answers.
        """
        if len(results) <= 1:
            return results

        domain_groups: Dict[str, List[Tuple[Chunk, float]]] = defaultdict(list)
        for chunk, score in results:
            domain_groups[chunk.domain or "unknown"].append((chunk, score))

        if len(domain_groups) <= 1:
            return results

        domain_scores = {
            d: sum(s for _, s in grp) for d, grp in domain_groups.items()
        }
        total = sum(domain_scores.values())
        if total <= 0:
            return results

        best = max(domain_scores, key=domain_scores.get)

        if domain_scores[best] / total > 0.6 and len(domain_groups[best]) >= 2:
            return domain_groups[best]

        return results

    def retrieve_with_domain_stats(self,
                                   query: str,
                                   top_k: int = 5,
                                   enable_reranking: bool = True) -> Tuple[List[Tuple[Chunk, float]], dict]:
        """
        Retrieve with domain distribution statistics.
        
        Returns:
            Tuple of (results, stats_dict)
        """
        results = self.retrieve(query, top_k, enable_reranking, debug=False)
        
        # Compute domain distribution
        domain_counts = defaultdict(int)
        for chunk, _ in results:
            domain_counts[chunk.domain or 'unknown'] += 1
        
        stats = {
            'query': query,
            'total_results': len(results),
            'domain_distribution': dict(domain_counts),
            'unique_domains': len(domain_counts),
        }
        
        return results, stats
    
    def batch_retrieve(self,
                      queries: List[str],
                      top_k: int = 5,
                      enable_reranking: bool = True,
                      show_progress: bool = True) -> List[List[Tuple[Chunk, float]]]:
        """
        Retrieve results for multiple queries.
        
        Args:
            queries: List of query strings
            top_k: Results per query
            enable_reranking: Use cross-encoder
            show_progress: Print progress
            
        Returns:
            List of result lists (one per query)
        """
        all_results = []
        
        for i, query in enumerate(queries, 1):
            if show_progress:
                print(f"[BATCH] Processing query {i}/{len(queries)}: {query[:50]}...")
            
            results = self.retrieve(query, top_k, enable_reranking, debug=False)
            all_results.append(results)
        
        return all_results
