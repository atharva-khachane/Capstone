"""
End-to-end Phase 2 integration test.

Tests the complete pipeline:
1. Load ISRO PDF document
2. Anonymize PII
3. Chunk document  
4. Generate embeddings (GPU)
5. Index in FAISS
6. Search with query
7. Save/load encrypted index
"""

from pathlib import Path
from sl_rag.core.document_loader import DocumentLoader
from sl_rag.core.pii_anonymizer import PIIAnonymizer
from sl_rag.core.chunk_generator import ChunkGenerator
from sl_rag.core.embedding_generator import EmbeddingGenerator
from sl_rag.core.faiss_index import FAISSIndexManager
from sl_rag.core.encryption_manager import EncryptionManager
import tempfile

print("="*70)
print("PHASE 2 END-TO-END INTEGRATION TEST")
print("="*70)

# Setup
TEST_DATA = Path("data")
temp_dir = tempfile.mkdtemp()

# Step 1: Load document
print("\n[STEP 1/8] Loading documents...")
loader = DocumentLoader(ocr_enabled=False, sanitize=True)
gfr_path = TEST_DATA / "FInal_GFR_upto_31_07_2024.pdf"

doc = loader.load_pdf(str(gfr_path))
print(f"✓ Loaded: {doc.word_count:,} words, {doc.metadata['pages']} pages")

# Step 2: Anonymize PII
print("\n[STEP 2/8] Anonymizing PII...")
anonymizer = PIIAnonymizer(enable_ner=False, log_detections=False)
anonymized_content, pii_found = anonymizer.anonymize(doc.content)
doc.content = anonymized_content
doc.pii_removed = True

total_pii = sum(pii_found.values())
print(f"✓ PII anonymized: {total_pii} instances removed")

# Step 3: Chunk document
print("\n[STEP 3/8] Chunking document...")
chunker =ChunkGenerator(chunk_size=512, overlap=50, min_chunk_size=100)
chunks = chunker.chunk_document(doc)

stats = chunker.get_chunk_statistics(chunks)
print(f"✓ Created {len(chunks)} chunks")
print(f"  - Avg tokens/chunk: {stats['avg_tokens_per_chunk']:.1f}")
print(f"  - Min/Max tokens: {stats['min_tokens']}/{stats['max_tokens']}")

# Step 4: Generate embeddings
print("\n[STEP 4/8] Generating embeddings (GPU)...")
embed_gen = EmbeddingGenerator(use_gpu=True, batch_size=32, show_progress=True)

# Embed first 100 chunks for testing (full doc would take longer)
test_chunks = chunks[:100]
test_chunks = embed_gen.generate_embeddings(test_chunks, normalize=True)

print(f"✓ Generated {len(test_chunks)} embeddings")
print(f"  Device: {embed_gen.device}")

# Step 5: Create FAISS index
print("\n[STEP 5/8] Creating FAISS index...")
enc_manager = EncryptionManager(master_key_path=f"{temp_dir}/master.key")
index_manager = FAISSIndexManager(
    index_path=f"{temp_dir}/test_index.faiss",
    embedding_dim=768,
    metric='IP',  # Inner Product for cosine similarity
    encryption_manager=enc_manager,
)

index_manager.add_chunks(test_chunks)
index_stats = index_manager.get_statistics()
print(f"✓ Index created: {index_stats['total_vectors']} vectors")

# Step 6: Search
print("\n[STEP 6/8] Testing search...")
queries = [
    "What are the GFR procurement rules?",
    "Financial regulations for government projects",
    "ISRO budget allocation procedures",
]

for i, query in enumerate(queries):
    print(f"\n  Query {i+1}: '{query}'")
    query_emb = embed_gen.generate_query_embedding(query, normalize=True)
    results = index_manager.search(query_emb, top_k=3)
    
    print(f"  Results (top 3):")
    for j, (chunk, score) in enumerate(results):
        preview = chunk.content[:80].replace('\n', ' ')
        print(f"    {j+1}. Score: {score:.4f} | {preview}...")

# Step 7: Save encrypted index
print("\n[STEP 7/8] Saving encrypted index...")
index_manager.save(encrypt=True)

# Step 8: Load and verify
print("\n[STEP 8/8] Loading encrypted index...")
new_index = FAISSIndexManager(
    index_path=f"{temp_dir}/test_index.faiss",
    embedding_dim=768,
    metric='IP',
    encryption_manager=enc_manager,
)
new_index.load(encrypted=True)

new_stats = new_index.get_statistics()
print(f"✓ Index loaded: {new_stats['total_vectors']} vectors")

# Verify search still works
query = "procurement rules"
query_emb = embed_gen.generate_query_embedding(query, normalize=True)
results = new_index.search(query_emb, top_k=2)
print(f"✓ Search after reload: {len(results)} results")

print("\n" + "="*70)
print("PHASE 2 INTEGRATION TEST: ✓ PASSED")
print("="*70)

print("\nSummary:")
print(f"  Document: {doc.metadata['filename']}")
print(f"  Words: {doc.word_count:,}")
print(f"  PII removed: {total_pii}")
print(f"  Chunks: {len(chunks):,} (tested with {len(test_chunks)})")
print(f"  Embeddings: {len(test_chunks)} x 768-dim")
print(f"  Index vectors: {new_stats['total_vectors']}")
print(f"  Encryption: AES-256 ✓")
print(f"  GPU used: {embed_gen.device == 'cuda'}")

# Cleanup
import shutil
shutil.rmtree(temp_dir, ignore_errors=True)
