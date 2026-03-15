"""
Test script for Domain Classifier.

Tests the rule-based domain classification system on actual documents.
"""

from pathlib import Path
from sl_rag.core.document_loader import DocumentLoader
from sl_rag.core.pii_anonymizer import PIIAnonymizer
from sl_rag.core.chunk_generator import ChunkGenerator
from sl_rag.core.embedding_generator import EmbeddingGenerator
from sl_rag.retrieval.domain_classifier import DomainClassifier
import tempfile
import time

print("=" * 80)
print(" " * 20 + "DOMAIN CLASSIFIER TEST")
print("=" * 80)

# Setup
DATA_DIR = Path("data")
pdf_files = [
    "FInal_GFR_upto_31_07_2024.pdf",
    "Procurement_Goods.pdf",
    "Procurement_Consultancy.pdf",
    "Procurement_Non_Consultancy.pdf",
    "20020050369.pdf",  # Technical/Telemetry
    "20180000774.pdf",  # Technical report
    "20210017934 FINAL.pdf",  # Technical/DPR
    "208B.pdf",  # Technical memo
]

print("\n[PHASE 1] LOADING AND PROCESSING DOCUMENTS")
print("-" * 80)

loader = DocumentLoader(ocr_enabled=False, sanitize=True)
anonymizer = PIIAnonymizer(enable_ner=False, log_detections=False)
chunker = ChunkGenerator(chunk_size=512, overlap=50, min_chunk_size=100)

all_chunks = []
doc_stats = []

for pdf_file in pdf_files:
    pdf_path = DATA_DIR / pdf_file
    if not pdf_path.exists():
        print(f"[SKIP] {pdf_file} not found")
        continue
    
    print(f"[LOAD] {pdf_file}...")
    
    # Load document
    doc = loader.load_pdf(str(pdf_path))
    doc.content, _ = anonymizer.anonymize(doc.content)
    
    # Chunk
    chunks = chunker.chunk_document(doc)
    
    # Add metadata
    for chunk in chunks:
        if not chunk.metadata:
            chunk.metadata = {}
        chunk.metadata['filepath'] = str(pdf_path)
        chunk.metadata['source_document'] = pdf_file
        chunk.metadata['document_title'] = pdf_file.replace('.pdf', '').replace('_', ' ')
        chunk.metadata['document_id'] = doc.doc_id
    
    all_chunks.extend(chunks)
    
    doc_stats.append({
        'file': pdf_file,
        'chunks': len(chunks),
    })

print(f"\n[OK] Loaded {len(all_chunks)} total chunks from {len(doc_stats)} documents")
for stat in doc_stats:
    print(f"  - {stat['file'][:50]:50s}: {stat['chunks']:4d} chunks")

print("\n" + "-" * 80)
print("[PHASE 2] RULE-BASED CLASSIFICATION (WITHOUT EMBEDDINGS)")
print("-" * 80)

# Initialize classifier
classifier = DomainClassifier(
    confidence_threshold=0.6,
    use_embeddings=False,  # Test rules first
    use_context_propagation=True
)

# Detect document contexts
print("\n[CONTEXT] Detecting document-level contexts...")
docs_by_id = {}
for chunk in all_chunks:
    doc_id = chunk.metadata.get('document_id')
    if doc_id not in docs_by_id:
        docs_by_id[doc_id] = []
    docs_by_id[doc_id].append(chunk)

for doc_id, chunks in docs_by_id.items():
    classifier.detect_document_context(chunks, doc_id)

# Classify
print("\n[CLASSIFY] Running rule-based classification...")
start_time = time.time()
results = classifier.classify_batch(all_chunks, verbose=True)
elapsed = time.time() - start_time

print(f"\n[OK] Classified {len(all_chunks)} chunks in {elapsed:.1f}s ({len(all_chunks)/elapsed:.0f} chunks/sec)")

# Print summary
classifier.print_summary(results)

# Show examples from each domain
print("\n" + "=" * 80)
print("SAMPLE CLASSIFICATIONS")
print("=" * 80)

for domain in ['gfr', 'procurement', 'technical', 'general']:
    domain_chunks = [c for c in all_chunks if c.domain == domain]
    if domain_chunks:
        print(f"\n{domain.upper()} Domain - {len(domain_chunks)} chunks")
        print("-" * 80)
        for chunk in domain_chunks[:3]:  # Show first 3
            conf = chunk.metadata.get('domain_confidence', 0)
            method = chunk.metadata.get('classification_method', 'unknown')
            print(f"  [{conf:.2f}] ({method}) {chunk.content[:100]}...")
            print()

# Test embedding-based classification
if any(pdf_file in ["FInal_GFR_upto_31_07_2024.pdf", "Procurement_Goods.pdf"] for pdf_file in [str(s['file']) for s in doc_stats]):
    print("\n" + "=" * 80)
    print("[PHASE 3] TESTING WITH EMBEDDINGS")
    print("=" * 80)
    
    print("\n[EMBED] Generating embeddings...")
    embed_gen = EmbeddingGenerator(use_gpu=True, batch_size=64, show_progress=True)
    all_chunks = embed_gen.generate_embeddings(all_chunks, normalize=True)
    
    # Build prototypes
    print("\n[PROTO] Building domain prototypes...")
    classifier_emb = DomainClassifier(
        confidence_threshold=0.6,
        use_embeddings=True,
        use_context_propagation=True
    )
    
    # Detect contexts again
    for doc_id, chunks in docs_by_id.items():
        classifier_emb.detect_document_context(chunks, doc_id)
    
    classifier_emb.build_prototypes(all_chunks, min_samples=5, min_confidence=0.7)
    
    # Re-classify with embeddings
    print("\n[CLASSIFY] Re-classifying with embeddings enabled...")
    results_emb = classifier_emb.classify_batch(all_chunks, verbose=False)
    
    # Compare
    print("\n" + "-" * 80)
    print("COMPARISON: Rules Only vs Rules + Embeddings")
    print("-" * 80)
    
    print(f"\n{'Metric':<30s} {'Rules Only':>15s} {'Rules+Embeddings':>20s}")
    print("-" * 70)
    
    # Methods used
    for method in ['rule_based', 'embedding_based', 'combined', 'low_confidence', 'fallback']:
        rules_count = results['stats'].get(method, 0)
        emb_count = results_emb['stats'].get(method, 0)
        print(f"{method:<30s} {rules_count:>15d} {emb_count:>20d}")
    
    print("\n" + "-" * 70)
    
    # Confidence
    for level in ['very_high', 'high', 'medium', 'low']:
        rules_count = results['confidence_distribution'][level]
        emb_count = results_emb['confidence_distribution'][level]
        print(f"{level + ' confidence':<30s} {rules_count:>15d} {emb_count:>20d}")
    
    # Print final summary
    print("\n")
    classifier_emb.print_summary(results_emb)

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
print("\nClassifier successfully tested on real documents!")
print("Next steps:")
print("  1. Review sample classifications above")
print("  2. Adjust domain rules if needed (keywords/patterns)")
print("  3. Integrate with retrieval pipeline")
print()
