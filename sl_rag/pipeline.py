"""
SL-RAG Pipeline Orchestrator (Layer 7 - Response Delivery).

End-to-end pipeline that wires together all 7 layers:
  Layer 1: DocumentLoader, PIIAnonymizer, EncryptionManager
  Layer 2: ChunkGenerator, EmbeddingGenerator, FAISS, DocumentLevelDomainManager
  Layer 3: BM25, HybridRetriever, CrossEncoder, RetrievalPipeline
  Layer 4: PromptBuilder, LLMGenerator (Llama 3.2 3B, 4-bit)
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
from typing import Dict, List, Any, Optional

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
from .generation.prompt_builder import PromptBuilder
from .generation.llm_generator import LLMGenerator
from .validation.validation_pipeline import ValidationPipeline
from .monitoring.monitoring_system import MonitoringSystem
from .security.auth import RBACManager, resolve_auth_context


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
        llm_model: str = "models/llama-3.2-3b",
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
        self.pipeline: Optional[RetrievalPipeline] = None

        # Layer 4 -- Prompt & Generation
        self.prompt_builder = PromptBuilder(max_context_tokens=6000)
        self.llm: Optional[LLMGenerator] = None
        if load_llm:
            try:
                self.llm = LLMGenerator(model_name=llm_model)
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
        self.hybrid = HybridRetriever(
            self.bm25, self.faiss_index, self.embedder, alpha=0.7,
        )
        self.pipeline = RetrievalPipeline(
            self.embedder, self.domain_manager, self.hybrid,
            self.reranker,
            similarity_threshold=self.retrieval_similarity_threshold,
            rerank_candidates=self.retrieval_top_k_candidates,
            final_top_k=self.retrieval_top_k_final,
            initial_top_k_candidates=self.retrieval_top_k_candidates,
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
        top_k: int = 5,
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

        t0 = time.time()

        # Layer 3 -- Retrieve
        results = self.pipeline.retrieve(
            question, top_k=top_k, enable_reranking=enable_reranking,
        )
        retrieval_ms = (time.time() - t0) * 1000

        # Layer 4 -- Prompt & Generate
        answer = ""
        generation_ms = 0.0
        injection_blocked = False
        if generate_answer and self.llm and results:
            if self.prompt_builder.detect_injection(question):
                answer = ("I cannot process this query as it contains patterns "
                          "that may compromise the system's integrity.")
                injection_blocked = True
                self.monitor.log_security_event(
                    "prompt_injection_blocked", "warning",
                    f"Query blocked: {question[:100]}",
                )
            else:
                prompt = self.prompt_builder.build_prompt(question, results)
                t1 = time.time()
                answer = self.llm.generate(prompt)
                generation_ms = (time.time() - t1) * 1000

        total_ms = (time.time() - t0) * 1000

        # Layer 5 -- Validate
        retrieval_validation = self.validator.validate_retrieval(question, results)
        answer_validation = {}
        if answer and not injection_blocked:
            context_chunks = [c for c, _ in results]
            scores = [s for _, s in results]
            answer_validation = self.validator.validate_answer(
                answer, context_chunks, scores,
            )

        # Build response
        response: Dict[str, Any] = {
            "query": question,
            "answer": answer,
            "confidence": (
                answer_validation.get("confidence", retrieval_validation["confidence"])
            ),
            "hallucination_risk": answer_validation.get("hallucination_risk", "n/a"),
            "citation_quality": retrieval_validation["citation_quality"],
            "injection_blocked": injection_blocked,
            "latency": {
                "retrieval_ms": round(retrieval_ms, 1),
                "generation_ms": round(generation_ms, 1),
                "total_ms": round(total_ms, 1),
            },
            "num_results": len(results),
            "sources": [],
        }

        for chunk, score in results:
            if not self.rbac.can_access_document(identity.role, chunk.metadata):
                self.monitor.log_security_event(
                    "document_access_denied",
                    "warning",
                    f"user_id={identity.user_id} role={identity.role} doc_id={chunk.doc_id}",
                )
                continue
            response["sources"].append({
                "content": chunk.content[:500],
                "score": round(float(score), 4),
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
