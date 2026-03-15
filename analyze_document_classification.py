"""
Analyze domain classification by document.

Shows which documents were classified into which domains.
"""

from pathlib import Path
from sl_rag.core.document_loader import DocumentLoader
from sl_rag.core.pii_anonymizer import PIIAnonymizer
from sl_rag.core.chunk_generator import ChunkGenerator
from sl_rag.retrieval.domain_classifier import DomainClassifier
from collections import defaultdict

print("=" * 80)
print(" " * 15 + "DOCUMENT-LEVEL DOMAIN CLASSIFICATION ANALYSIS")
print("=" * 80)

# Setup
DATA_DIR = Path("data")
pdf_files = [
    ("FInal_GFR_upto_31_07_2024.pdf", "gfr"),  # Expected domain
    ("Procurement_Goods.pdf", "procurement"),
    ("Procurement_Consultancy.pdf", "procurement"),
    ("Procurement_Non_Consultancy.pdf", "procurement"),
    ("20020050369.pdf", "technical"),
    ("20180000774.pdf", "technical"),
    ("20210017934 FINAL.pdf", "technical"),
    ("208B.pdf", "technical"),
]

print("\n[STEP 1] EXPECTED DOMAIN MAPPING (Ground Truth)")
print("-" * 80)
print(f"{'Document':<50s} {'Expected Domain':<20s}")
print("-" * 80)
for filename, expected_domain in pdf_files:
    print(f"{filename:<50s} {expected_domain:<20s}")

print("\n[STEP 2] LOADING AND CLASSIFYING DOCUMENTS")
print("-" * 80)

loader = DocumentLoader(ocr_enabled=False, sanitize=True)
anonymizer = PIIAnonymizer(enable_ner=False, log_detections=False)
chunker = ChunkGenerator(chunk_size=512, overlap=50, min_chunk_size=100)
classifier = DomainClassifier(
    confidence_threshold=0.6,
    use_embeddings=False,
    use_context_propagation=True
)

all_chunks = []
doc_chunks_map = {}  # Maps doc_id to chunks

for filename, expected_domain in pdf_files:
    pdf_path = DATA_DIR / filename
    if not pdf_path.exists():
        print(f"[SKIP] {filename} not found")
        continue
    
    print(f"[LOAD] {filename}...")
    
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
        chunk.metadata['source_document'] = filename
        chunk.metadata['document_title'] = filename.replace('.pdf', '').replace('_', ' ')
        chunk.metadata['document_id'] = doc.doc_id
        chunk.metadata['expected_domain'] = expected_domain
    
    doc_chunks_map[doc.doc_id] = {
        'filename': filename,
        'expected_domain': expected_domain,
        'chunks': chunks
    }
    
    all_chunks.extend(chunks)

print(f"\n[OK] Loaded {len(all_chunks)} total chunks from {len(doc_chunks_map)} documents")

# Detect document contexts
print("\n[STEP 3] DETECTING DOCUMENT-LEVEL CONTEXTS")
print("-" * 80)
for doc_id, doc_info in doc_chunks_map.items():
    classifier.detect_document_context(doc_info['chunks'], doc_id)

# Classify
print("\n[STEP 4] CLASSIFYING CHUNKS")
print("-" * 80)
results = classifier.classify_batch(all_chunks, verbose=False)

# Analyze per-document classification
print("\n[STEP 5] PER-DOCUMENT CLASSIFICATION ANALYSIS")
print("=" * 80)

for doc_id, doc_info in doc_chunks_map.items():
    filename = doc_info['filename']
    expected_domain = doc_info['expected_domain']
    chunks = doc_info['chunks']
    
    # Count domains for this document
    domain_counts = defaultdict(int)
    for chunk in chunks:
        domain_counts[chunk.domain] += 1
    
    # Find dominant domain
    dominant_domain = max(domain_counts, key=domain_counts.get)
    dominant_count = domain_counts[dominant_domain]
    total = len(chunks)
    accuracy = (dominant_count / total * 100) if total > 0 else 0
    
    # Check if correct
    status = "✅ CORRECT" if dominant_domain == expected_domain else "❌ INCORRECT"
    
    print(f"\n{filename}")
    print("-" * 80)
    print(f"  Expected Domain: {expected_domain}")
    print(f"  Dominant Domain: {dominant_domain} ({dominant_count}/{total} = {accuracy:.1f}%)")
    print(f"  Status: {status}")
    print(f"  Breakdown:")
    for domain in sorted(domain_counts.keys()):
        count = domain_counts[domain]
        pct = (count / total * 100) if total > 0 else 0
        bar = "█" * int(pct / 3)
        marker = " ← EXPECTED" if domain == expected_domain else ""
        print(f"    {domain:15s}: {count:4d} chunks ({pct:5.1f}%) {bar}{marker}")

# Overall accuracy summary
print("\n" + "=" * 80)
print("OVERALL ACCURACY SUMMARY")
print("=" * 80)

correct_docs = 0
total_docs = len(doc_chunks_map)

for doc_id, doc_info in doc_chunks_map.items():
    chunks = doc_info['chunks']
    expected_domain = doc_info['expected_domain']
    
    # Count domains
    domain_counts = defaultdict(int)
    for chunk in chunks:
        domain_counts[chunk.domain] += 1
    
    # Check if dominant domain matches expected
    dominant_domain = max(domain_counts, key=domain_counts.get)
    if dominant_domain == expected_domain:
        correct_docs += 1

doc_accuracy = (correct_docs / total_docs * 100) if total_docs > 0 else 0

print(f"\nDocument-Level Accuracy: {correct_docs}/{total_docs} = {doc_accuracy:.1f}%")
print(f"  ✅ Correctly classified: {correct_docs} documents")
print(f"  ❌ Incorrectly classified: {total_docs - correct_docs} documents")

# Chunk-level accuracy
correct_chunks = 0
total_chunks = len(all_chunks)

for chunk in all_chunks:
    expected = chunk.metadata.get('expected_domain')
    actual = chunk.domain
    if expected == actual:
        correct_chunks += 1

chunk_accuracy = (correct_chunks / total_chunks * 100) if total_chunks > 0 else 0

print(f"\nChunk-Level Accuracy: {correct_chunks}/{total_chunks} = {chunk_accuracy:.1f}%")
print(f"  ✅ Correctly classified: {correct_chunks} chunks")
print(f"  ❌ Incorrectly classified: {total_chunks - correct_chunks} chunks")

print("\n" + "=" * 80)
print("ACCURACY BASIS")
print("=" * 80)
print("""
The accuracy is calculated based on:

1. EXPECTED DOMAIN (Ground Truth):
   - Determined by document filename and content type
   - GFR: General Financial Rules document
   - Procurement: Procurement manuals (Goods, Consultancy, Non-Consultancy)
   - Technical: Technical reports, telemetry guides, DPR documents

2. ACTUAL DOMAIN (Classifier Output):
   - Assigned by the rule-based classifier using keywords, patterns, and structure

3. DOCUMENT-LEVEL ACCURACY:
   - A document is "correct" if the MAJORITY of its chunks are classified
     into the expected domain
   - Formula: (Correctly classified documents / Total documents) × 100

4. CHUNK-LEVEL ACCURACY:
   - Each chunk is "correct" if its classified domain matches the expected
     domain for its source document
   - Formula: (Correctly classified chunks / Total chunks) × 100

5. CONFIDENCE METRICS:
   - Very High (≥0.9): Classifier is very certain
   - High (0.7-0.9): Classifier is confident
   - Medium (0.5-0.7): Classifier is moderately confident
   - Low (<0.5): Classifier is uncertain
""")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
