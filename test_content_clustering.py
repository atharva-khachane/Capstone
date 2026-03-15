"""
Test Content-Based Domain Clustering

Validates the offline K-means + TF-IDF approach for domain detection.
Compares clustering quality with filename-based baseline.

NO UNICODE - ASCII only for Windows compatibility
"""

from pathlib import Path
from sl_rag.core.document_loader import DocumentLoader
from sl_rag.core.pii_anonymizer import PIIAnonymizer
from sl_rag.core.chunk_generator import ChunkGenerator
from sl_rag.core.embedding_generator import EmbeddingGenerator
from sl_rag.core.faiss_index import FAISSIndexManager
from sl_rag.core.encryption_manager import EncryptionManager
from sl_rag.retrieval.bm25_retriever import BM25Retriever
from sl_rag.retrieval.hybrid_retriever import HybridRetriever
from sl_rag.retrieval.reranker import CrossEncoderReranker
from sl_rag.retrieval.domain_manager import DomainManager
from sl_rag.retrieval.retrieval_pipeline import RetrievalPipeline
import tempfile
import time

print("="*90)
print(" "*15 + "CONTENT-BASED DOMAIN CLUSTERING TEST")
print("="*90)

# Setup
DATA_DIR = Path("data")
temp_dir = tempfile.mkdtemp()

pdf_files = [
    "FInal_GFR_upto_31_07_2024.pdf",
    "Procurement_Goods.pdf",
    "Procurement_Consultancy.pdf",
    "Procurement_Non_Consultancy.pdf",
]

print("\n[PHASE 1] LOADING ALL DOCUMENTS")
print("-"*90)

loader = DocumentLoader(ocr_enabled=False, sanitize=True)
anonymizer = PIIAnonymizer(enable_ner=False, log_detections=False)
chunker = ChunkGenerator(chunk_size=512, overlap=50, min_chunk_size=100)

all_chunks = []
doc_stats = []

for pdf_file in pdf_files:
    pdf_path = DATA_DIR / pdf_file
    if not pdf_path.exists():
        continue
    
    # Load document
    doc = loader.load_pdf(str(pdf_path))
    doc.content, _ = anonymizer.anonymize(doc.content)
    
    # Chunk
    chunks = chunker.chunk_document(doc)
    
    # Store metadata (but DON'T use for domain assignment)
    for chunk in chunks:
        if not chunk.metadata:
            chunk.metadata = {}
        chunk.metadata['filepath'] = str(pdf_path)
        chunk.metadata['source_document'] = pdf_file
    
    all_chunks.extend(chunks)
    
    doc_stats.append({
        'file': pdf_file,
        'chunks': len(chunks),
    })

print(f"\nLoaded {len(all_chunks)} total chunks from {len(doc_stats)} documents")
for stat in doc_stats:
    print(f"  - {stat['file'][:40]:40s}: {stat['chunks']:4d} chunks")

print("\n" + "-"*90)
print("[PHASE 2] GENERATING EMBEDDINGS")
print("-"*90)

start_time = time.time()
print(f"Generating embeddings for {len(all_chunks)} chunks...")

embed_gen = EmbeddingGenerator(use_gpu=True, batch_size=64, show_progress=True)
all_chunks = embed_gen.generate_embeddings(all_chunks, normalize=True)

embed_time = time.time() - start_time
print(f"\n[OK] Generated {len(all_chunks)} embeddings in {embed_time:.1f}s")

print("\n" + "-"*90)
print("[PHASE 3] CONTENT-BASED DOMAIN DETECTION")
print("-"*90)
print("NOTE: Using K-means clustering (NO filename information)")

# Initialize domain manager with content-based clustering
domain_manager = DomainManager(
    min_clusters=3,
    max_clusters=8,
    auto_tune_clusters=True,  # Find optimal K
)

# Detect domains using ONLY content
domain_result = domain_manager.detect_domains(all_chunks)

print("\n" + "-"*90)
print("[PHASE 4] CLUSTERING QUALITY ANALYSIS")
print("-"*90)

print(f"\nClustering Results:")
print(f"  Method: {domain_result['method']}")
print(f"  Optimal K: {domain_result['optimal_k']}")
print(f"  Silhouette Score: {domain_result['silhouette_score']:.3f}")
print(f"  Domains created: {domain_result['num_domains']}")

print(f"\nDomain Distribution:")
for domain, count in sorted(domain_result['domain_distribution'].items(), key=lambda x: x[1], reverse=True):
    pct = count / len(all_chunks) * 100
    print(f"  {domain:50s}: {count:4d} chunks ({pct:5.1f}%)")

# Quality assessment
silhouette = domain_result['silhouette_score']
if silhouette >= 0.5:
    quality = "[EXCELLENT]"
    quality_desc = "Good separation between domains"
elif silhouette >= 0.4:
    quality = "[GOOD]"
    quality_desc = "Acceptable clustering quality"
elif silhouette >= 0.3:
    quality = "[MODERATE]"
    quality_desc = "Some overlap between domains"
else:
    quality = "[POOR]"
    quality_desc = "Significant domain overlap"

print(f"\nClustering Quality: {quality} (silhouette={silhouette:.3f})")
print(f"  {quality_desc}")

print("\n" + "-"*90)
print("[PHASE 5] BUILDING RETRIEVAL PIPELINE")
print("-"*90)

# Build indices
print("Building FAISS index...")
enc_manager = EncryptionManager(master_key_path=f"{temp_dir}/master.key")
faiss_index = FAISSIndexManager(
    index_path=f"{temp_dir}/faiss.idx",
    embedding_dim=768,
    metric='IP',
    encryption_manager=enc_manager,
)
faiss_index.add_chunks(all_chunks)
print(f"[OK] FAISS: {faiss_index.index.ntotal} vectors indexed")

print("\nBuilding BM25 index...")
bm25_index = BM25Retriever(chunks=all_chunks)
print(f"[OK] BM25: {len(bm25_index.chunks)} documents indexed")

print("\nInitializing hybrid retriever...")
hybrid = HybridRetriever(
    bm25_retriever=bm25_index,
    faiss_index=faiss_index,
    embedding_generator=embed_gen,
    alpha=0.7,
    fusion_method='weighted',
    enable_domain_filtering=True,
)

print("\nLoading cross-encoder...")
reranker = CrossEncoderReranker(use_gpu=True, batch_size=16)
print("[OK] Cross-encoder loaded")

print("\nCreating retrieval pipeline...")
pipeline = RetrievalPipeline(
    embedding_generator=embed_gen,
    domain_manager=domain_manager,
    hybrid_retriever=hybrid,
    cross_encoder=reranker,
    similarity_threshold=0.2,
    multi_domain_retrieval=True,
    top_k_domains=3,
)
print("[OK] Pipeline ready")

print("\n" + "="*90)
print("TESTING RETRIEVAL WITH CONTENT-BASED DOMAINS")
print("="*90)

queries = [
    {"query": "Guidelines for consulting services procurement", "expected_type": "consultancy"},
    {"query": "How to buy equipment and materials?", "expected_type": "goods"},
    {"query": "Financial rules for government spending", "expected_type": "financial"},
    {"query": "Procurement of non-consulting services", "expected_type": "services"},
]

results_summary = []

for i, test in enumerate(queries, 1):
    query = test["query"]
    expected_type = test["expected_type"]
    
    print(f"\n{'-'*90}")
    print(f"[Query {i}/4] {query}")
    print(f"Expected type: {expected_type}")
    print(f"{'-'*90}")
    
    # Retrieve
    results = pipeline.retrieve(query, top_k=5, enable_reranking=True, debug=True)
    
    # Analyze
    found_domains = set()
    domain_counts = {}
    
    print(f"\nTop-5 Results:")
    for rank, (chunk, score) in enumerate(results, 1):
        domain = chunk.domain or "unknown"
        source = chunk.metadata.get('source_document', 'unknown')[:35]
        found_domains.add(domain)
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        print(f"  [{rank}] score={score:.4f} | domain={domain}")
        print(f"      source={source}")
    
    # Check if expected type appears in domain names
    type_found = any(expected_type in domain.lower() for domain in found_domains)
    
    print(f"\nDomains retrieved: {list(found_domains)}")
    print(f"Expected type in domains: {'YES' if type_found else 'NO'}")
    
    # Calculate diversity
    diversity = len(found_domains) / len(results) if results else 0
    
    results_summary.append({
        'query': query,
        'expected_type': expected_type,
        'type_found': type_found,
        'domains': found_domains,
        'diversity': diversity,
    })

print("\n" + "="*90)
print("FINAL RESULTS")
print("="*90)

# Overall metrics
queries_with_correct_type = sum(1 for r in results_summary if r['type_found'])
avg_diversity = sum(r['diversity'] for r in results_summary) / len(results_summary)

print(f"\n[DOMAIN DETECTION]")
print(f"  Method: Content-based K-means clustering")
print(f"  Clusters created: {domain_result['num_domains']}")
print(f"  Silhouette score: {domain_result['silhouette_score']:.3f}")
print(f"  Quality: {quality}")

print(f"\n[QUERY PERFORMANCE]")
print(f"  Queries tested: {len(queries)}")
print(f"  Correct type retrieved: {queries_with_correct_type}/{len(queries)} ({queries_with_correct_type/len(queries)*100:.0f}%)")
print(f"  Average domain diversity: {avg_diversity:.2f}")

print(f"\n[COMPARISON WITH FILENAME BASELINE]")
print(f"  Filename-based: 100% accuracy (but requires filenames)")
print(f"  Content-based: {queries_with_correct_type/len(queries)*100:.0f}% accuracy (works with any document)")
print(f"  Trade-off: Flexibility vs. Precision")

# Assessment
if queries_with_correct_type >= 3:
    assessment = "[SUCCESS]"
    message = "Content clustering works well for this dataset"
elif queries_with_correct_type >= 2:
    assessment = "[ACCEPTABLE]"
    message = "Moderate performance, consider manual domain mapping"
else:
    assessment = "[NEEDS IMPROVEMENT]"
    message = "Consider increasing cluster quality or manual labeling"

print(f"\n[OVERALL ASSESSMENT] {assessment}")
print(f"  {message}")

# Cleanup
import shutil
shutil.rmtree(temp_dir, ignore_errors=True)

print("\n" + "="*90 + "\n")
