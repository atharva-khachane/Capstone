"""
SL-RAG Pipeline Orchestrator (Layer 7 - Response Delivery).

End-to-end pipeline that wires together all 7 layers:
  Layer 1: DocumentLoader, PIIAnonymizer, EncryptionManager
  Layer 2: ChunkGenerator, EmbeddingGenerator, FAISS, DocumentLevelDomainManager
  Layer 3: BM25, HybridRetriever, CrossEncoder, RetrievalPipeline
  Layer 4: PromptBuilder, LLMGenerator (Llama 3.2-3B-Instruct, 4-bit)
  Layer 5: ValidationPipeline
  Layer 6: MonitoringSystem
  Layer 7: This orchestrator + CLI + Python API

Provides:
  - Python API  (``SLRAGPipeline.query()``)
  - JSON output (structured response with citations & confidence)
  - CLI wrapper (``python -m sl_rag.pipeline``)

100% OFFLINE after initial model download.
"""

import json
import time
from pathlib import Path
from typing import Dict, Generator, List, Any, Optional

from .core.document_loader import DocumentLoader
from .core.pii_anonymizer import PIIAnonymizer
from .core.chunk_generator import ChunkGenerator
from .core.embedding_generator import EmbeddingGenerator
from .core.encryption_manager import EncryptionManager
from .core.faiss_index import FAISSIndexManager
from .retrieval.bm25_retriever import BM25Retriever
from .retrieval.hybrid_retriever import HybridRetriever
from .retrieval.reranker import CrossEncoderReranker
from .retrieval.document_level_domain_manager import DocumentLevelDomainManager
from .retrieval.retrieval_pipeline import RetrievalPipeline
from .retrieval.adversarial_detector import ASIDetector
from .retrieval.trust_scorer import TrustScorer
from .generation.prompt_builder import PromptBuilder
from .generation.llm_generator import LLMGenerator
from .generation.entailment_checker import EntailmentChecker
from .validation.validation_pipeline import ValidationPipeline
from .monitoring.monitoring_system import MonitoringSystem
from .security.auth import RBACManager, resolve_auth_context
from .calibrated_confidence import compute_rule_based_confidence
from .guardrail import check_and_gate, SAFE_FALLBACK


class SLRAGPipeline:
    """Full SL-RAG pipeline: ingest, index, retrieve, generate, validate, log."""

    def __init__(
        self,
        data_dir: str = "./data",
        storage_dir: str = "./storage",
        config_path: str = "./config/config.yaml",
        use_gpu: bool = True,
        encryption: bool = True,
        monitor_encrypt_at_rest: bool = False,
        load_llm: bool = True,
        llm_model: str = "google/gemma-4-e4b",
        ocr_enabled: Optional[bool] = None,
    ):
        self.data_dir = Path(data_dir)
        self.storage_dir = Path(storage_dir)
        self.use_gpu = use_gpu
        self.config = self._load_config(config_path)

        # Layer 1 -- Ingestion & Security
        loader_cfg = self.config.get("document_loading", {})
        resolved_ocr = loader_cfg.get("ocr_enabled", False) if ocr_enabled is None else ocr_enabled
        resolved_max_file_mb = int(loader_cfg.get("max_file_size_mb", 300))
        resolved_min_text_chars = int(loader_cfg.get("min_text_chars", 100))
        self.loader = DocumentLoader(
            ocr_enabled=resolved_ocr,
            max_file_size_mb=resolved_max_file_mb,
            min_text_chars=resolved_min_text_chars,
        )
        self.anonymizer = PIIAnonymizer(enable_ner=False)
        self.chunker = ChunkGenerator(chunk_size=512, overlap=50, min_chunk_size=100)
        self.embedder = EmbeddingGenerator(use_gpu=use_gpu, batch_size=32)

        self.encryption_mgr = (
            EncryptionManager(str(self.storage_dir / "keys" / "master.key"))
            if encryption else None
        )

        # Layer 2 -- Embedding & Storage
        self.faiss_index = FAISSIndexManager(
            str(self.storage_dir / "faiss_index" / "index.faiss"),
            encryption_manager=self.encryption_mgr,
        )
        self.domain_manager = DocumentLevelDomainManager(min_clusters=2, max_clusters=10)
        retrieval_cfg = self.config.get("retrieval", {})
        self.retrieval_similarity_threshold = float(retrieval_cfg.get("similarity_threshold", 0.5))
        self.retrieval_top_k_candidates = int(retrieval_cfg.get("top_k_candidates", 20))
        self.retrieval_top_k_final = int(retrieval_cfg.get("top_k_final", 5))

        # Layer 3 -- Retrieval (built after ingestion)
        self.bm25: Optional[BM25Retriever] = None
        self.hybrid: Optional[HybridRetriever] = None
        self.reranker = CrossEncoderReranker(use_gpu=use_gpu)
        self.entailment_checker = EntailmentChecker(
            cross_encoder_model=self.reranker.model,
            threshold=0.20,
            min_words=8,
        )
        self.pipeline: Optional[RetrievalPipeline] = None

        # Layer 4 -- Prompt & Generation
        llm_cfg = self.config.get("llm", {})
        self.prompt_builder = PromptBuilder(max_context_tokens=6000)
        self.llm: Optional[LLMGenerator] = None
        if load_llm:
            try:
                self.llm = LLMGenerator(
                    model_name=llm_model,
                    max_new_tokens=int(llm_cfg.get("max_new_tokens", 512)),
                    temperature=float(llm_cfg.get("temperature", 0.3)),
                )
            except Exception as e:
                print(f"[PIPELINE] WARNING: LLM failed to load ({e}). "
                      "Retrieval-only mode active.")

        # Layer 5 -- Validation
        self.validator = ValidationPipeline(embedding_generator=self.embedder)

        # Layer 6 -- Monitoring
        self.monitor = MonitoringSystem(
            str(self.storage_dir / "audit_logs" / "monitoring.db"),
            encryption_manager=self.encryption_mgr,
            encrypt_at_rest=monitor_encrypt_at_rest,
        )
        self.rbac = RBACManager()

        self.chunks: List = []
        self._ready = False

        # Trust-RAG: ASI detector + Trust Scorer (built/configured after ingest)
        asi_cfg = self.config.get("adversarial", {})
        self.asi_detector = ASIDetector(
            threshold=float(asi_cfg.get("asi_threshold", 2.5)),
        )
        # Bug 1 fix — smoke-test: any future LLM switch that removes the
        # guardrail must break loudly here rather than silently returning 0.
        assert self.asi_detector is not None, (
            "ASI guardrail failed to initialise. "
            "Check ASIDetector.__init__ and the adversarial config block."
        )
        self.trust_scorer = TrustScorer(
            weights=(0.4, 0.2, 0.2, 0.2),
            lambda_decay=0.001,
        )

        # Fix 1: Rule-based confidence (replaces Platt scaling)

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        path = Path(config_path)
        if not path.exists():
            return {}
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, data_dir: Optional[str] = None) -> Dict[str, Any]:
        """Load, anonymize, chunk, embed, cluster, and index all PDFs."""
        d = Path(data_dir) if data_dir else self.data_dir

        # Load
        docs, stats = self.loader.load_directory(str(d), recursive=False)
        print(f"[PIPELINE] Loaded {len(docs)} documents")

        # Anonymize
        for doc in docs:
            doc.content, _ = self.anonymizer.anonymize(doc.content)
            doc.pii_removed = True

        # Chunk
        self.chunks = []
        for doc in docs:
            chunks = self.chunker.chunk_document(doc)
            for c in chunks:
                c.metadata["filepath"] = doc.metadata.get("filepath", "")
                c.metadata["source_document"] = doc.metadata.get("filename", "")
            self.chunks.extend(chunks)
        print(f"[PIPELINE] Created {len(self.chunks)} chunks")

        # Embed
        self.chunks = self.embedder.generate_embeddings(self.chunks, normalize=True)

        # Domain detection
        domain_result = self.domain_manager.detect_domains(self.chunks)
        drift_result = self.monitor.check_domain_drift(self.domain_manager.domains)

        # Build indices
        self.faiss_index.add_chunks(self.chunks)
        self.bm25 = BM25Retriever(self.chunks)

        # Fix 5: read bm25_alpha_param from config; dense_alpha = 1 - bm25_alpha_param
        retrieval_cfg = self.config.get("retrieval", {})
        bm25_alpha_param = float(retrieval_cfg.get("bm25_alpha_param", 0.5))
        dense_alpha = 1.0 - bm25_alpha_param
        tech_domain_alpha = float(retrieval_cfg.get("tech_domain_alpha", 0.95))
        tech_domains = retrieval_cfg.get("tech_domains", ["telemetry", "technical_report"])
        print(
            f"[PIPELINE] Hybrid alpha={dense_alpha:.2f} "
            f"(bm25_alpha_param={bm25_alpha_param}), "
            f"tech_domain_alpha={tech_domain_alpha}"
        )

        self.hybrid = HybridRetriever(
            self.bm25,
            self.faiss_index,
            self.embedder,
            alpha=dense_alpha,
            tech_domain_alpha=tech_domain_alpha,
            tech_domains=tech_domains,
        )
        cache_cfg = self.config.get("cache", {})
        self.pipeline = RetrievalPipeline(
            self.embedder, self.domain_manager, self.hybrid,
            self.reranker,
            similarity_threshold=self.retrieval_similarity_threshold,
            rerank_candidates=self.retrieval_top_k_candidates,
            final_top_k=self.retrieval_top_k_final,
            initial_top_k_candidates=self.retrieval_top_k_candidates,
            cache_enabled=bool(cache_cfg.get("enabled", True)),
            cache_ttl_seconds=int(cache_cfg.get("ttl_seconds", 3600)),
            cache_max_entries=int(cache_cfg.get("max_entries", 200)),
        )
        # Build ASI anchor set from corpus embeddings (Trust-RAG Module 1A)
        self.asi_detector.build_anchors_from_chunks(self.chunks)
        # Bug 1 fix — smoke-test: anchors MUST be populated after ingest.
        # If this fails the corpus has no embeddable chunks — stop immediately.
        assert self.asi_detector.is_ready, (
            "ASI guardrail anchor build failed: no embedded chunks in corpus. "
            "Ensure ingest() embeds documents before querying."
        )

        self._ready = True

        return {
            "documents": len(docs),
            "chunks": len(self.chunks),
            "domains": domain_result,
            "drift": drift_result,
        }

    # ------------------------------------------------------------------
    # Query (Python API)
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: int = 0,
        enable_reranking: bool = True,
        generate_answer: bool = True,
        auth_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Query the pipeline: retrieve, generate, validate, log.

        Args:
            question: User query string
            top_k: Number of chunks to retrieve
            enable_reranking: Use cross-encoder re-ranking
            generate_answer: If True and LLM is loaded, generate a
                natural-language answer from retrieved context

        Returns:
            JSON-serialisable dict with answer, citations, confidence, etc.
        """
        if not self._ready:
            return {"error": "Pipeline not ready. Call ingest() first."}

        identity = resolve_auth_context(auth_context)
        if not self.rbac.can_query(identity.role):
            self.monitor.log_security_event(
                "query_access_denied",
                "warning",
                f"user_id={identity.user_id} role={identity.role} query={question[:100]}",
            )
            return {
                "error": "Access denied for role",
                "user_id": identity.user_id,
                "role": identity.role,
            }

        if not top_k:
            top_k = self.retrieval_top_k_final

        blocked = check_and_gate(question)
        if blocked:
            return {
                "query": question,
                "answer": blocked,
                "confidence": 0.0,
                "raw_confidence": 0.0,
                "hallucination_risk": "high",
                "faithfulness_score": None,
                "citation_quality": "poor",
                "injection_blocked": True,
                "asi_score": 0.0,
                "asi_flagged": True,
                "latency": {
                    "retrieval_ms": 0.0,
                    "generation_ms": 0.0,
                    "total_ms": 0.0,
                },
                "num_results": 0,
                "sources": [],
            }

        t0 = time.time()

        # Trust-RAG Layer 3 — ASI adversarial check (before regex injection)
        # Fix 4: also pass query_text for keyword injection pattern matching
        query_emb_for_asi = self.embedder.generate_query_embedding(question, normalize=True)
        asi_flagged, asi_score = self.asi_detector.is_adversarial(
            query_emb_for_asi, query_text=question
        )
        if asi_flagged:
            self.monitor.log_security_event(
                "asi_adversarial_detected", "warning",
                f"asi_score={asi_score:.4f} query={question[:100]}",
            )

        # Layer 3 -- Retrieve
        results = self.pipeline.retrieve(
            question, top_k=top_k, enable_reranking=enable_reranking,
        )
        generation_results = self.prompt_builder.select_relevant_chunks(question, results)
        retrieval_ms = (time.time() - t0) * 1000

        # Layer 4 -- Prompt & Generate
        answer = ""
        generation_ms = 0.0
        injection_blocked = False
        # Fix 4: block generation when ASI flags the query as adversarial
        _blocked_by_asi = asi_flagged and asi_score >= 9.0  # keyword-matched injections score 9.99
        if generate_answer and self.llm and generation_results:
            if self.prompt_builder.detect_injection(question) or _blocked_by_asi:
                answer = ("I cannot process this query as it contains patterns "
                          "that may compromise the system's integrity.")
                injection_blocked = True
                self.monitor.log_security_event(
                    "prompt_injection_blocked", "warning",
                    f"Query blocked: {question[:100]}",
                )
            else:
                prompt = self.prompt_builder.build_prompt(question, generation_results)
                t1 = time.time()
                # Fix 6: use streaming generation when enabled in config so the
                # LLM starts emitting tokens immediately (time-to-first-token
                # benefit), then collect into a full string for the blocking API.
                llm_cfg = self.config.get("llm", {})
                if llm_cfg.get("streaming", False):
                    tokens: List[str] = []
                    ttft_ms: float = 0.0
                    for tok in self.llm.generate_stream(prompt):
                        if not tokens:
                            ttft_ms = (time.time() - t1) * 1000
                        tokens.append(tok)
                    answer = self.llm._post_process("".join(tokens))
                    generation_ms = (time.time() - t1) * 1000
                    print(f"[LLM] TTFT={ttft_ms:.0f}ms  total={generation_ms:.0f}ms")
                else:
                    answer = self.llm.generate(prompt)
                    generation_ms = (time.time() - t1) * 1000

        total_ms = (time.time() - t0) * 1000

        # Fix 2: Sentence-level entailment filter — strip unsupported sentences
        num_flagged = 0
        if answer and not injection_blocked:
            answer, num_flagged = self.entailment_checker.check_and_filter(
                answer, [c for c, _ in generation_results]
            )
            if num_flagged > 0:
                self.monitor.log_metric(
                    "entailment_flagged_sentences",
                    float(num_flagged),
                    {"query": question[:100]},
                )

        # Layer 5 -- Validate
        # Use generation_results (the chunk subset actually given to the LLM) for
        # both validations so the citation quality and confidence baseline are
        # computed from the same context the answer was generated from.
        retrieval_validation = self.validator.validate_retrieval(question, generation_results)
        answer_validation = {}
        faithfulness_score = None
        if answer and not injection_blocked:
            context_chunks = [c for c, _ in generation_results]
            scores = [s for _, s in generation_results]
            answer_validation = self.validator.validate_answer(
                answer, context_chunks, scores,
            )
            faithfulness_score = answer_validation.get("faithfulness_score")

        # Fix 1: Rule-based confidence from actual pipeline signals
        raw_confidence = answer_validation.get("confidence", retrieval_validation["confidence"])
        top1_retrieval = max((s for _, s in generation_results), default=0.0)
        groundedness = answer_validation.get("faithfulness_score", 0.0) or 0.0
        ctx_precision = answer_validation.get("consistency_score", 0.0) or 0.0
        ctx_recall = (
            answer_validation.get("context_recall")
            or answer_validation.get("contextual_recall")
            or 0.0
        )
        calibrated_confidence = compute_rule_based_confidence(
            top1_retrieval, groundedness, ctx_precision, ctx_recall
        )

        display_answer = answer
        if (
            num_flagged > 1
            and answer
            and answer_validation.get("hallucination_risk") == "high"
        ):
            display_answer = (
                "[Note: parts of this answer could not be fully verified "
                "against the source documents.]\n\n" + answer
            )

        response: Dict[str, Any] = {
            "query": question,
            "answer": display_answer,
            "confidence": calibrated_confidence,
            "raw_confidence": raw_confidence,
            "hallucination_risk": answer_validation.get("hallucination_risk", "n/a"),
            "faithfulness_score": faithfulness_score,
            "citation_quality": retrieval_validation["citation_quality"],
            "injection_blocked": injection_blocked,
            "asi_score": round(asi_score, 4),
            "asi_flagged": asi_flagged,
            "latency": {
                "retrieval_ms": round(retrieval_ms, 1),
                "generation_ms": round(generation_ms, 1),
                "total_ms": round(total_ms, 1),
            },
            "num_results": len(generation_results),
            "sources": [],
        }

        # Trust scoring: re-rank generation_results by composite trust score
        trust_results = self.trust_scorer.score_chunks(query_emb_for_asi, generation_results)
        for chunk, trust_score, trust_breakdown in trust_results:
            if not self.rbac.can_access_document(identity.role, chunk.metadata):
                self.monitor.log_security_event(
                    "document_access_denied",
                    "warning",
                    f"user_id={identity.user_id} role={identity.role} doc_id={chunk.doc_id}",
                )
                continue
            response["sources"].append({
                "content": chunk.content[:500],
                "score": round(float(trust_score), 4),
                "trust_breakdown": trust_breakdown.to_dict(),
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "domain": chunk.domain,
                "source": chunk.metadata.get("source_document", ""),
            })


        # Layer 6 -- Log
        domains = list({s["domain"] for s in response["sources"] if s["domain"]})
        self.monitor.log_query(
            question, domains, len(results), total_ms,
            response["confidence"],
            user_id=identity.user_id,
            role=identity.role,
            session_id=identity.session_id,
        )
        for chunk, _ in results:
            if self.rbac.can_access_document(identity.role, chunk.metadata):
                self.monitor.log_document_access(
                    chunk.doc_id,
                    [chunk.chunk_id],
                    user_id=identity.user_id,
                    role=identity.role,
                    session_id=identity.session_id,
                )

        sensitive_doc_ids = [
            chunk.doc_id for chunk, _ in results
            if str(chunk.metadata.get("sensitivity", "public")).lower() in {"confidential", "restricted"}
        ]
        self.monitor.analyze_query_patterns(
            user_id=identity.user_id,
            domains=domains,
            sensitive_doc_ids=sensitive_doc_ids,
        )

        return response


    # ------------------------------------------------------------------
    # Streaming Query
    # ------------------------------------------------------------------

    def query_stream(
        self,
        question: str,
        top_k: int = 0,
        enable_reranking: bool = True,
        auth_context: Optional[Dict[str, Any]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Like query() but yields SSE-style dicts as generation streams.

        Yields:
            {"type": "token",  "token": "<text>"}       — one per generated piece
            {"type": "done",   **full_response_dict}     — final event with metadata
            {"type": "error",  "message": "<msg>"}       — on failure
        """
        if not self._ready:
            yield {"type": "error", "message": "Pipeline not ready."}
            return

        identity = resolve_auth_context(auth_context)
        if not self.rbac.can_query(identity.role):
            yield {"type": "error", "message": "Access denied for role."}
            return

        if not top_k:
            top_k = self.retrieval_top_k_final

        blocked = check_and_gate(question)
        if blocked:
            yield {"type": "token", "token": SAFE_FALLBACK}
            yield {
                "type": "done",
                "query": question,
                "answer": SAFE_FALLBACK,
                "injection_blocked": True,
                "confidence": 0.0,
                "hallucination_risk": "high",
                "citation_quality": "poor",
                "latency": {"retrieval_ms": 0.0, "generation_ms": 0.0, "total_ms": 0.0},
                "num_results": 0,
                "sources": [],
                "domain": None,
            }
            return

        t0 = time.time()

        # Fix 4: ASI adversarial check with keyword scanning in streaming path too
        query_emb_stream = self.embedder.generate_query_embedding(question, normalize=True)
        asi_flagged_stream, asi_score_stream = self.asi_detector.is_adversarial(
            query_emb_stream, query_text=question
        )
        if asi_flagged_stream:
            self.monitor.log_security_event(
                "asi_adversarial_detected", "warning",
                f"asi_score={asi_score_stream:.4f} query={question[:100]}",
            )

        # Retrieval (fast, synchronous)
        results = self.pipeline.retrieve(question, top_k=top_k, enable_reranking=enable_reranking)
        generation_results = self.prompt_builder.select_relevant_chunks(question, results)
        retrieval_ms = (time.time() - t0) * 1000

        # Injection check (also block keyword-flagged adversarial queries)
        _stream_blocked_by_asi = asi_flagged_stream and asi_score_stream >= 9.0
        if self.prompt_builder.detect_injection(question) or _stream_blocked_by_asi:
            self.monitor.log_security_event(
                "prompt_injection_blocked", "warning",
                f"Query blocked: {question[:100]}",
            )
            yield {"type": "token", "token": "I cannot process this query as it contains patterns that may compromise the system's integrity."}
            yield {"type": "done", "query": question, "answer": "", "injection_blocked": True,
                   "confidence": 0.0, "hallucination_risk": "high", "citation_quality": "",
                   "latency": {"retrieval_ms": round(retrieval_ms, 1), "generation_ms": 0, "total_ms": round((time.time()-t0)*1000, 1)},
                   "num_results": 0, "sources": [], "domain": None}
            return

        if not generation_results or not self.llm:
            msg = "I cannot provide an answer from the provided documents."
            yield {"type": "token", "token": msg}
            yield {"type": "done", "query": question, "answer": msg, "injection_blocked": False,
                   "confidence": 0.0, "hallucination_risk": "high", "citation_quality": "poor",
                   "latency": {"retrieval_ms": round(retrieval_ms, 1), "generation_ms": 0, "total_ms": round((time.time()-t0)*1000, 1)},
                   "num_results": 0, "sources": [], "domain": None}
            return

        # Build prompt
        prompt = self.prompt_builder.build_prompt(question, generation_results)

        # Stream tokens
        t1 = time.time()
        accumulated_tokens: List[str] = []
        for token in self.llm.generate_stream(prompt):
            accumulated_tokens.append(token)
            yield {"type": "token", "token": token}

        generation_ms = (time.time() - t1) * 1000
        total_ms = (time.time() - t0) * 1000

        # Post-process the full accumulated answer
        raw_answer = "".join(accumulated_tokens)
        answer = self.llm._deduplicate_citations(raw_answer).strip()

        # Fix 2: Entailment filter in streaming path
        _stream_flagged = 0
        if answer:
            answer, _stream_flagged = self.entailment_checker.check_and_filter(
                answer, [c for c, _ in generation_results]
            )
            if _stream_flagged > 0:
                self.monitor.log_metric(
                    "entailment_flagged_sentences",
                    float(_stream_flagged),
                    {"query": question[:100]},
                )

        # Validate — use generation_results for both so validation context
        # matches what was actually used to generate the answer.
        retrieval_validation = self.validator.validate_retrieval(question, generation_results)
        context_chunks = [c for c, _ in generation_results]
        scores = [s for _, s in generation_results]
        answer_validation = self.validator.validate_answer(answer, context_chunks, scores)

        display_answer = answer
        if _stream_flagged > 1 and answer_validation.get("hallucination_risk") == "high":
            display_answer = (
                "[Note: parts of this answer could not be fully verified "
                "against the source documents.]\n\n" + answer
            )

        # Build sources
        sources = []
        for chunk, score in generation_results:
            if not self.rbac.can_access_document(identity.role, chunk.metadata):
                continue
            sources.append({
                "content": chunk.content[:500],
                "score": round(float(score), 4),
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "domain": chunk.domain,
                "source": chunk.metadata.get("source_document", ""),
            })

        domain = None
        if sources:
            d_counts: Dict[str, int] = {}
            for s in sources:
                d = s.get("domain", "")
                if d:
                    d_counts[d] = d_counts.get(d, 0) + 1
            if d_counts:
                domain = max(d_counts, key=d_counts.get)

        # Fix 1: Rule-based confidence in streaming path
        _stream_top1 = max((s for _, s in generation_results), default=0.0)
        _stream_ground = answer_validation.get("faithfulness_score", 0.0) or 0.0
        _stream_ctx = answer_validation.get("consistency_score", 0.0) or 0.0
        _stream_recall = (
            answer_validation.get("context_recall")
            or answer_validation.get("contextual_recall")
            or 0.0
        )
        _stream_conf = compute_rule_based_confidence(
            _stream_top1, _stream_ground, _stream_ctx, _stream_recall
        )

        done_payload = {
            "type": "done",
            "query": question,
            "answer": display_answer,
            "confidence": _stream_conf,
            "hallucination_risk": answer_validation.get("hallucination_risk", "n/a"),
            "citation_quality": retrieval_validation["citation_quality"],
            "injection_blocked": False,
            "latency": {
                "retrieval_ms": round(retrieval_ms, 1),
                "generation_ms": round(generation_ms, 1),
                "total_ms": round(total_ms, 1),
            },
            "num_results": len(generation_results),
            "sources": sources,
            "domain": domain,
        }

        yield done_payload

        # Audit log (after yielding)
        domains = list({s["domain"] for s in sources if s["domain"]})
        self.monitor.log_query(
            question, domains, len(results), total_ms,
            done_payload["confidence"],
            user_id=identity.user_id,
            role=identity.role,
            session_id=identity.session_id,
        )
        for chunk, _ in results:
            if self.rbac.can_access_document(identity.role, chunk.metadata):
                self.monitor.log_document_access(
                    chunk.doc_id, [chunk.chunk_id],
                    user_id=identity.user_id,
                    role=identity.role,
                    session_id=identity.session_id,
                )


# ======================================================================
# CLI entry point
# ======================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="SL-RAG Pipeline CLI")
    sub = parser.add_subparsers(dest="command")

    ingest_p = sub.add_parser("ingest", help="Ingest documents from data/")
    ingest_p.add_argument("--data-dir", default="./data")

    query_p = sub.add_parser("query", help="Query the pipeline")
    query_p.add_argument("question", type=str)
    query_p.add_argument("--top-k", type=int, default=5)
    query_p.add_argument("--no-rerank", action="store_true")
    query_p.add_argument("--no-generate", action="store_true",
                         help="Skip LLM answer generation (retrieval only)")
    query_p.add_argument("--no-llm", action="store_true",
                         help="Do not load the LLM at all")

    report_p = sub.add_parser("report", help="Print compliance report")

    args = parser.parse_args()

    load_llm = not getattr(args, "no_llm", False)
    pipe = SLRAGPipeline(load_llm=load_llm)

    if args.command == "ingest":
        result = pipe.ingest(args.data_dir)
        print(json.dumps(result, indent=2, default=str))

    elif args.command == "query":
        pipe.ingest()
        resp = pipe.query(
            args.question,
            top_k=args.top_k,
            enable_reranking=not args.no_rerank,
            generate_answer=not args.no_generate,
        )
        print(json.dumps(resp, indent=2, default=str))

    elif args.command == "report":
        report = pipe.monitor.generate_compliance_report()
        print(json.dumps(report, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
