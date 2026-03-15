"""
Full SL-RAG Pipeline Demo: Ingest -> Retrieve -> Generate -> Validate.

Runs 10 demo queries across all document domains, shows LLM-generated
answers alongside expected answers for manual comparison.
"""

import gc
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DEMO_QUERIES = [
    {
        "query": "What are the General Financial Rules regarding government expenditure?",
        "expected": (
            "The GFR governs government expenditure, budgeting, and accounting. "
            "It mandates standards of financial propriety, delegation of financial "
            "powers to ministries/departments, and procedures for the annual "
            "financial statement under Article 112 of the Constitution."
        ),
        "domain": "GFR",
    },
    {
        "query": "What is the delegation of financial powers to government ministries?",
        "expected": (
            "Financial powers are delegated to ministries and departments by the "
            "Ministry of Finance. Each ministry has specific spending limits and "
            "must follow prescribed procedures for sanctions and re-appropriations."
        ),
        "domain": "GFR",
    },
    {
        "query": "What is the process for procurement of goods above 25 lakhs?",
        "expected": (
            "For goods above Rs 25 lakhs, procurement must follow open/advertised "
            "tender enquiry via the Central Public Procurement Portal (CPPP), with "
            "a minimum bid submission period, two-bid system, and evaluation by a "
            "tender committee."
        ),
        "domain": "Procurement_Goods",
    },
    {
        "query": "Explain the GeM portal usage for government procurement",
        "expected": (
            "The Government e-Marketplace (GeM) is the mandatory platform for "
            "procurement of common-use goods and services. It supports direct "
            "purchase up to Rs 25,000 and bidding/reverse auction for higher values."
        ),
        "domain": "Procurement_Goods",
    },
    {
        "query": "What is QCBS method for selecting consultants?",
        "expected": (
            "Quality and Cost Based Selection (QCBS) evaluates both technical "
            "quality (70-80% weight) and cost (20-30% weight). The combined score "
            "determines the winning consultant."
        ),
        "domain": "Procurement_Consultancy",
    },
    {
        "query": "What are the different methods for selection of consultants?",
        "expected": (
            "Methods include: QCBS, Quality Based Selection (QBS), Fixed Budget "
            "Selection (FBS), Least Cost Selection (LCS), Single Source Selection "
            "(SSS), and Consultants' Qualifications (CQS)."
        ),
        "domain": "Procurement_Consultancy",
    },
    {
        "query": "What are the procedures for non-consultancy service procurement?",
        "expected": (
            "Non-consultancy services follow competitive bidding. The process "
            "involves publishing RFP, evaluation criteria, performance security, "
            "and contract management."
        ),
        "domain": "Procurement_NonConsultancy",
    },
    {
        "query": "What are the telemetry data specifications and signal processing requirements?",
        "expected": (
            "The telemetry system uses PCM for data acquisition, covering specs "
            "for data encoding, signal processing, frame synchronization, and "
            "error detection per IRIG standards."
        ),
        "domain": "Technical (208B/telemetry)",
    },
    {
        "query": "What are the thermal protection and insulation design requirements?",
        "expected": (
            "Thermal protection systems include insulation materials, thermal "
            "barriers, heat shields, and temperature management, with material "
            "properties and testing specs."
        ),
        "domain": "Technical (wiring/design docs)",
    },
    {
        "query": "What are the connector and wiring specifications for aerospace systems?",
        "expected": (
            "Wiring specs cover connector types, wire gauges, insulation "
            "requirements, routing procedures, and aerospace-grade quality "
            "standards including environmental sealing and plating."
        ),
        "domain": "Technical (wiring/design docs)",
    },
]


def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def main():
    import torch

    flush_print("")
    flush_print("#" * 80)
    flush_print("#  SL-RAG FULL PIPELINE DEMO")
    flush_print("#  Ingest -> Retrieve -> Generate (Llama 3.2 3B) -> Validate")
    flush_print("#" * 80)

    # Stage 1: Build components manually (so we can manage GPU memory)
    flush_print("\n[1/6] Loading core components...")
    from sl_rag.core.document_loader import DocumentLoader
    from sl_rag.core.pii_anonymizer import PIIAnonymizer
    from sl_rag.core.chunk_generator import ChunkGenerator
    from sl_rag.core.embedding_generator import EmbeddingGenerator
    from sl_rag.core.encryption_manager import EncryptionManager
    from sl_rag.core.faiss_index import FAISSIndexManager
    from sl_rag.retrieval.bm25_retriever import BM25Retriever
    from sl_rag.retrieval.hybrid_retriever import HybridRetriever
    from sl_rag.retrieval.reranker import CrossEncoderReranker
    from sl_rag.retrieval.document_level_domain_manager import DocumentLevelDomainManager
    from sl_rag.retrieval.retrieval_pipeline import RetrievalPipeline
    from sl_rag.generation.prompt_builder import PromptBuilder
    from sl_rag.generation.llm_generator import LLMGenerator
    from sl_rag.validation.validation_pipeline import ValidationPipeline

    t0 = time.time()

    loader = DocumentLoader(ocr_enabled=False, max_file_size_mb=300)
    anonymizer = PIIAnonymizer(enable_ner=False)
    chunker = ChunkGenerator(chunk_size=512, overlap=50, min_chunk_size=100)
    embedder = EmbeddingGenerator(use_gpu=True, batch_size=32)
    prompt_builder = PromptBuilder(max_context_tokens=2048)

    # Stage 2: Ingest
    flush_print("\n[2/6] Loading and processing documents...")
    docs, _ = loader.load_directory("./data", recursive=False)
    flush_print(f"  Loaded {len(docs)} documents")

    for doc in docs:
        doc.content, _ = anonymizer.anonymize(doc.content)
        doc.pii_removed = True

    all_chunks = []
    for doc in docs:
        chunks = chunker.chunk_document(doc)
        for c in chunks:
            c.metadata["filepath"] = doc.metadata.get("filepath", "")
            c.metadata["source_document"] = doc.metadata.get("filename", "")
        all_chunks.extend(chunks)
    flush_print(f"  Created {len(all_chunks)} chunks")

    # Stage 3: Embed
    flush_print("\n[3/6] Generating embeddings (this takes ~5 min)...")
    all_chunks = embedder.generate_embeddings(all_chunks, normalize=True)
    flush_print(f"  Embeddings done. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Stage 4: Domain detection + indexing
    flush_print("\n[4/6] Domain detection and indexing...")
    domain_mgr = DocumentLevelDomainManager(min_clusters=2, max_clusters=10)
    domain_result = domain_mgr.detect_domains(all_chunks)
    flush_print(f"  Domains: {domain_result}")

    faiss_idx = FAISSIndexManager("./storage/faiss_index/index.faiss")
    faiss_idx.add_chunks(all_chunks)
    flush_print(f"  FAISS index: {faiss_idx.index.ntotal} vectors")

    bm25 = BM25Retriever(all_chunks)
    hybrid = HybridRetriever(bm25, faiss_idx, embedder, alpha=0.7)
    reranker = CrossEncoderReranker(use_gpu=True)
    pipeline = RetrievalPipeline(embedder, domain_mgr, hybrid, reranker, similarity_threshold=0.3)
    validator = ValidationPipeline(embedding_generator=embedder)

    ingest_time = time.time() - t0
    flush_print(f"  Ingestion complete in {ingest_time:.1f}s")
    flush_print(f"  VRAM after indexing: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Move embedding model + reranker to CPU to free VRAM for LLM + KV cache
    flush_print("\n  Offloading embedding model + reranker to CPU for LLM headroom...")
    embedder.model.cpu()
    reranker.model.model.cpu()
    torch.cuda.empty_cache()
    gc.collect()
    flush_print(f"  VRAM freed: {torch.cuda.memory_allocated()/1e9:.2f} GB used")

    # Stage 5: Load LLM
    flush_print("\n[5/6] Loading Llama 3.2 3B (4-bit)...")
    try:
        llm = LLMGenerator(model_name="models/llama-3.2-3b", max_new_tokens=256)
        flush_print(f"  VRAM after LLM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    except Exception as e:
        flush_print(f"  ERROR loading LLM: {e}")
        flush_print("  Falling back to retrieval-only mode.")
        llm = None

    # Stage 6: Run queries
    flush_print(f"\n[6/6] Running {len(DEMO_QUERIES)} demo queries...\n")

    all_results = []
    for idx, qinfo in enumerate(DEMO_QUERIES, 1):
        query = qinfo["query"]
        flush_print(f"  Query {idx}/{len(DEMO_QUERIES)}: {query[:60]}...")

        try:
            qt0 = time.time()

            results = pipeline.retrieve(query, top_k=3, enable_reranking=False)
            retrieval_ms = (time.time() - qt0) * 1000

            answer = ""
            gen_ms = 0.0
            if llm and results:
                if prompt_builder.detect_injection(query):
                    answer = "[BLOCKED: Prompt injection detected]"
                else:
                    prompt = prompt_builder.build_prompt(query, results)
                    gt0 = time.time()
                    answer = llm.generate(prompt)
                    gen_ms = (time.time() - gt0) * 1000

            total_ms = (time.time() - qt0) * 1000

            ret_val = validator.validate_retrieval(query, results)
            ans_val = {}
            if answer and not answer.startswith("[BLOCKED"):
                context_chunks = [c for c, _ in results]
                scores = [s for _, s in results]
                ans_val = validator.validate_answer(answer, context_chunks, scores)

            confidence = ans_val.get("confidence", ret_val.get("confidence", "N/A"))
            hallucination = ans_val.get("hallucination_risk", "N/A")

            sources = []
            for chunk, score in results[:3]:
                sources.append({
                    "score": round(float(score), 4),
                    "domain": chunk.domain,
                    "doc": chunk.metadata.get("source_document", ""),
                    "snippet": chunk.content[:120].replace("\n", " "),
                })

            all_results.append({
                "idx": idx,
                "query": query,
                "expected": qinfo["expected"],
                "expected_domain": qinfo["domain"],
                "answer": answer,
                "confidence": confidence,
                "hallucination": hallucination,
                "retrieval_ms": round(retrieval_ms, 1),
                "generation_ms": round(gen_ms, 1),
                "total_ms": round(total_ms, 1),
                "num_sources": len(results),
                "sources": sources,
            })
            flush_print(f"    Done ({total_ms:.0f} ms)")

        except Exception as e:
            flush_print(f"    ERROR: {e}")
            traceback.print_exc()
            all_results.append({
                "idx": idx, "query": query, "answer": f"ERROR: {e}",
                "expected": qinfo["expected"], "expected_domain": qinfo["domain"],
                "confidence": "N/A", "hallucination": "N/A",
                "retrieval_ms": 0, "generation_ms": 0, "total_ms": 0,
                "num_sources": 0, "sources": [],
            })

    # Print detailed results
    flush_print("\n" + "=" * 80)
    flush_print("  DETAILED RESULTS")
    flush_print("=" * 80)

    for r in all_results:
        flush_print(f"\n{'='*80}")
        flush_print(f"  QUERY {r['idx']}: {r['query']}")
        flush_print(f"  Target domain: {r['expected_domain']}")
        flush_print(f"{'-'*80}")

        flush_print(f"\n  >> LLM ANSWER:")
        for line in r["answer"].split("\n"):
            flush_print(f"     {line}")

        flush_print(f"\n  >> EXPECTED ANSWER:")
        for line in r["expected"].split("\n"):
            flush_print(f"     {line}")

        flush_print(f"\n  >> METRICS:")
        flush_print(f"     Confidence:    {r['confidence']}")
        flush_print(f"     Hallucination: {r['hallucination']}")
        flush_print(f"     Retrieval:     {r['retrieval_ms']} ms")
        flush_print(f"     Generation:    {r['generation_ms']} ms")
        flush_print(f"     Total:         {r['total_ms']} ms")

        flush_print(f"\n  >> TOP SOURCES ({r['num_sources']} retrieved):")
        for i, s in enumerate(r["sources"], 1):
            flush_print(f"     [{i}] Score={s['score']:.4f} Domain={s['domain']} "
                        f"Doc={s['doc']}")
            flush_print(f"         \"{s['snippet']}...\"")

    # Summary table
    flush_print(f"\n{'='*80}")
    flush_print("  SUMMARY TABLE")
    flush_print(f"{'='*80}")
    flush_print(f"  {'#':>2} | {'CONF':>6} | {'HALLUC':>8} | "
                f"{'RET ms':>7} | {'GEN ms':>7} | {'SRC':>3} | QUERY")
    flush_print(f"  {'--':>2} | {'------':>6} | {'--------':>8} | "
                f"{'-------':>7} | {'-------':>7} | {'---':>3} | {'--'*20}")

    for r in all_results:
        conf = r["confidence"]
        if isinstance(conf, (int, float)):
            conf = f"{conf:.2f}"
        else:
            conf = str(conf)[:6]
        hall = str(r["hallucination"])[:8]
        flush_print(f"  {r['idx']:2d} | {conf:>6} | {hall:>8} | "
                    f"{r['retrieval_ms']:>7.0f} | {r['generation_ms']:>7.0f} | "
                    f"{r['num_sources']:>3} | {r['query'][:40]}")

    total_time = time.time() - t0
    flush_print(f"\n{'='*80}")
    flush_print(f"  Total execution time: {total_time:.1f}s")
    flush_print(f"{'='*80}")

    # Save to file
    with open("demo_output.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    flush_print(f"\n  Results saved to demo_output.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
