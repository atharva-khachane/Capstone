"""
Comprehensive demo query test across all documents and domains.

Tests 18 queries spanning every document and domain to verify that
the DocumentLevelDomainManager routes correctly.
"""

import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sl_rag.core.document_loader import DocumentLoader
from sl_rag.core.pii_anonymizer import PIIAnonymizer
from sl_rag.core.chunk_generator import ChunkGenerator
from sl_rag.core.embedding_generator import EmbeddingGenerator
from sl_rag.retrieval.document_level_domain_manager import DocumentLevelDomainManager


def ingest():
    print("=" * 72)
    print("  INGESTION")
    print("=" * 72)
    loader = DocumentLoader(ocr_enabled=False, max_file_size_mb=300)
    anonymizer = PIIAnonymizer(enable_ner=False)
    chunker = ChunkGenerator(chunk_size=512, overlap=50, min_chunk_size=100)

    docs, _ = loader.load_directory("./data", recursive=False)
    for doc in docs:
        doc.content, _ = anonymizer.anonymize(doc.content)

    all_chunks = []
    for doc in docs:
        chunks = chunker.chunk_document(doc)
        for c in chunks:
            c.metadata["filepath"] = doc.metadata.get("filepath", "")
            c.metadata["source_document"] = doc.metadata.get("filename", "")
        all_chunks.extend(chunks)

    embedder = EmbeddingGenerator(use_gpu=True, batch_size=32)
    all_chunks = embedder.generate_embeddings(all_chunks, normalize=True)
    print(f"Chunks: {len(all_chunks)}, Docs: {len(docs)}")
    return all_chunks, embedder


def cluster(chunks):
    dm = DocumentLevelDomainManager(min_clusters=2, max_clusters=10)
    dm.detect_domains(chunks)
    return dm


def run_demo_queries(dm, embedder):
    print()
    print("=" * 72)
    print("  DEMO QUERIES -- DOMAIN ROUTING VERIFICATION")
    print("=" * 72)

    # (query, expected_top_domain, target_document_description)
    queries = [
        # --- government_expenditure (GFR document) ---
        ("What are the General Financial Rules for government expenditure?",
         "government_expenditure", "GFR"),
        ("What is the delegation of financial powers to ministries?",
         "government_expenditure", "GFR"),
        ("What are the rules for government budgeting and accounting?",
         "government_expenditure", "GFR"),
        ("Explain the annual financial statement and budget procedures",
         "government_expenditure", "GFR"),

        # --- procurement_contract (Procurement_Goods.pdf) ---
        ("What is the process for procurement of goods above 25 lakhs?",
         "procurement_contract", "Procurement_Goods"),
        ("Explain the two-bid system in goods procurement",
         "procurement_contract", "Procurement_Goods"),
        ("What are the inspection and quality control procedures for goods?",
         "procurement_contract", "Procurement_Goods"),
        ("Describe the GeM portal usage for government goods procurement",
         "procurement_contract", "Procurement_Goods"),

        # --- procurement_contract (Procurement_Consultancy.pdf) ---
        ("What is the QCBS method for selecting consultants?",
         "procurement_contract", "Procurement_Consultancy"),
        ("Explain the Expression of Interest process for consulting services",
         "procurement_contract", "Procurement_Consultancy"),
        ("What is Quality Based Selection for hiring consultants?",
         "procurement_contract", "Procurement_Consultancy"),

        # --- procurement_contract (Procurement_Non_Consultancy.pdf) ---
        ("What are the procedures for non-consultancy service procurement?",
         "procurement_contract", "Procurement_NonConsultancy"),
        ("Explain outsourcing rules for non-consulting services",
         "procurement_contract", "Procurement_NonConsultancy"),

        # --- figure_data (208B.pdf, 20180000774.pdf -- telemetry) ---
        ("Describe the telemetry data acquisition system",
         "figure_data", "208B / 20180000774"),
        ("What are the PCM telemetry signal processing specifications?",
         "figure_data", "208B / 20180000774"),

        # --- figure_socket (20210017934, 20020050369 -- wiring/sockets) ---
        ("What is the connector and wiring specification for the system?",
         "figure_socket", "20210017934 / 20020050369"),
        ("Explain the thermal protection and insulation design",
         "figure_socket", "20210017934 / 20020050369"),
        ("What socket types and zinc plating are specified?",
         "figure_socket", "20210017934 / 20020050369"),
    ]

    pass_count = 0
    fail_count = 0
    results_table = []

    for query, expected_domain, target_doc in queries:
        qe = embedder.generate_query_embedding(query, normalize=True)
        routed = dm.route_query(qe, top_k_domains=4, similarity_threshold=0.1)

        top_domain = routed[0][0] if routed else "NONE"
        top_score = routed[0][1] if routed else 0.0

        match = top_domain == expected_domain
        status = "PASS" if match else "FAIL"
        if match:
            pass_count += 1
        else:
            fail_count += 1

        results_table.append(
            (status, query[:62], expected_domain, top_domain, top_score, target_doc)
        )

    # Print results table
    print()
    hdr = f"  {'STATUS':6s} | {'QUERY':62s} | {'EXPECTED':25s} | {'GOT':25s} | {'SCORE':>6s} | TARGET"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for status, q, exp, got, sc, tdoc in results_table:
        print(f"  {status:6s} | {q:62s} | {exp:25s} | {got:25s} | {sc:6.3f} | {tdoc}")

    print()
    print(f"  TOTAL: {pass_count} PASS / {fail_count} FAIL out of {len(queries)} queries")
    print()

    # Detailed breakdown for failed queries
    if fail_count > 0:
        print("  DETAIL FOR FAILED QUERIES:")
        print("  " + "-" * 72)
        for status, q, exp, got, sc, tdoc in results_table:
            if status != "FAIL":
                continue
            qe = embedder.generate_query_embedding(q, normalize=True)
            routed = dm.route_query(qe, top_k_domains=4, similarity_threshold=0.05)
            print(f"  Q: {q}")
            print(f"    Expected: {exp}  |  Got: {got}")
            for d, s in routed:
                marker = "  <-- expected" if d == exp else ""
                print(f"      {d}: {s:.4f}{marker}")
            print()

    return pass_count, fail_count


def main():
    print()
    print("#" * 72)
    print("#  SL-RAG: COMPREHENSIVE DOMAIN ROUTING DEMO TEST")
    print("#" * 72)
    chunks, embedder = ingest()
    dm = cluster(chunks)
    passed, failed = run_demo_queries(dm, embedder)

    print("#" * 72)
    if failed == 0:
        print("#  ALL QUERIES ROUTED CORRECTLY")
    else:
        print(f"#  {failed} query(ies) routed to unexpected domain -- see details above")
    print("#" * 72)


if __name__ == "__main__":
    main()
