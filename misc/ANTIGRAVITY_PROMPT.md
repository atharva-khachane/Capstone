# ANTIGRAVITY PROMPT: SL-RAG PIPELINE IMPLEMENTATION

## MISSION STATEMENT
Generate a production-ready Secure and Accurate Retrieval-Augmented Generation (SL-RAG) pipeline for ISRO's sensitive government documentation (GFR, Procurement Manuals, Technical Memoranda, Telemetry Data). This system prioritizes MAXIMUM SECURITY with 100% offline operation, AES-256 encryption, complete audit trails, and PII protection. Implementation follows an INCREMENTAL approach with thorough testing at each phase.

## CRITICAL REQUIREMENTS
- ⚠️ **MAXIMUM SECURITY**: Government/sensitive data handling
- 🔒 **100% OFFLINE**: No internet after initial model download
- 🔐 **ENCRYPTION**: AES-256 for all stored data
- 📋 **AUDIT TRAILS**: Complete logging of all operations
- 🧪 **THOROUGH TESTING**: Comprehensive validation at each phase
- 🔄 **INCREMENTAL BUILD**: Build → Test → Validate → Next Phase

---

## SYSTEM SPECIFICATIONS

### Hardware Environment
- **CPU**: Intel i7 12th Gen
- **RAM**: 16GB
- **GPU**: NVIDIA RTX 3050Ti (4GB VRAM)
- **Storage**: 500GB available
- **Operating System**: Linux/Windows compatible

### Scale Requirements
- **Initial Corpus**: 10-20 PDF documents (GFR, Procurement Manuals, Technical Memos, Telemetry Data)
- **Target Corpus**: ~1,000 PDF documents
- **Expected Queries**: Government regulatory queries, technical documentation lookup, operational analysis
- **Response Time**: < 3 seconds for retrieval (LLM generation to be added later)

### Use Case: ISRO Documentation System
The system will support interactive, evidence-based analysis of:
- **QA Reports**: Quality assurance documentation
- **Procurement Manuals**: Vendor management and procurement procedures
- **GFR Rules**: General Financial Rules and regulations
- **Telemetry Data**: Satellite and mission telemetry analysis
- **Technical Documentation**: Failure analysis memos, technical reports
- **Cross-Center Knowledge**: Enable accurate decision support and operational analysis across ISRO centers

### Technical Stack (MANDATORY)

#### Core Components
1. **Embedding Model**: `all-mpnet-base-v2`
   - Dimensions: 768
   - Size: ~420MB
   - Framework: sentence-transformers
   - GPU-accelerated inference

2. **Vector Store**: FAISS (Facebook AI Similarity Search)
   - Index Type: IndexFlatIP (Inner Product) for small corpus, IndexIVFFlat for scaling to 1K+
   - GPU acceleration enabled
   - Persistent storage with encryption

3. **Document Processing**: PDF-focused pipeline
   - Library: PyMuPDF (fitz) - best for government PDFs
   - OCR capability: pytesseract (for scanned documents - common in govt files)
   - Text extraction with layout preservation
   - Support for complex tables and multi-column layouts

4. **Self-Hosted LLM**: ⏸️ **SKIP FOR NOW**
   - Will be added in future phase after Llama 3.2 3B setup
   - For now, system returns retrieved chunks with citations
   - Focus on retrieval quality first

5. **Cross-Encoder Re-ranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
   - For final top-K re-ranking
   - CPU inference acceptable

6. **BM25 Retriever**: rank-bm25
   - For hybrid retrieval (sparse + dense)

---

## ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────────┐
│              SL-RAG PIPELINE (MAXIMUM SECURITY)                 │
│                    100% OFFLINE OPERATION                        │
└─────────────────────────────────────────────────────────────────┘

LAYER 1: SECURE DATA INGESTION ✅ PHASE 1
├── PDF Loader (PyMuPDF + OCR for scanned govt docs)
├── Content Sanitizer (HTML/Script removal, malware detection)
├── PII Anonymizer (Email, Phone, SSN, Aadhaar, Credit Card, Names)
├── Document Validator (Format, Size, Encoding, Integrity)
└── Encryption (AES-256 for stored documents)

LAYER 2: EMBEDDING & SECURE STORAGE ✅ PHASE 2
├── Chunk Generator (Semantic chunking, 512 tokens, 50 token overlap)
├── Embedding Generator (all-mpnet-base-v2, GPU-accelerated)
├── **CONTENT-BASED CLUSTERING** (Automatic domain detection via K-means)
├── FAISS Index Builder (GPU-accelerated indexing)
├── Domain Centroid Calculator (Per-domain embeddings)
└── Encrypted Storage (AES-256, local filesystem, secure key management)

LAYER 3: MULTI-DOMAIN RETRIEVAL & RE-RANKING ✅ PHASE 3
├── **USER QUERY INPUT** (Natural language questions)
├── Query Preprocessor (Normalization, acronym expansion: GFR, ISRO, etc.)
├── Query Embedding (Convert user query to 768-dim vector)
├── Domain Router (Cosine similarity to centroids)
├── Hybrid Retriever (BM25 + Dense, score fusion)
├── Similarity Filter (Threshold: 0.5)
└── Cross-Encoder Re-ranker (Top-20 → Top-5 most relevant chunks)

LAYER 4: PROMPT INTEGRITY & GENERATION ⏸️ SKIP FOR NOW
└── Will be added after Llama 3.2 3B setup

LAYER 5: VALIDATION & SECURITY ✅ PHASE 4
├── Citation Verification (Ensure all results have valid sources)
├── Confidence Scorer (0.0-1.0 based on retrieval scores)
├── **AUDIT LOGGER** (User, timestamp, query, accessed documents)
├── **ACCESS CONTROL** (Track who accessed what documents)
└── Query Pattern Monitor (Detect unusual access patterns)

LAYER 6: MONITORING & GOVERNANCE ✅ PHASE 5
├── Query Logger (Timestamps, domains, results)
├── **AUDIT TRAIL** (Encrypted SQLite, tamper-evident logs)
├── Performance Monitor (Latency, accuracy metrics)
├── Drift Detector (Embedding distribution changes)
└── **SECURITY EVENTS** (Failed access attempts, PII detections)

LAYER 7: RESPONSE DELIVERY ✅ PHASE 6
├── CLI Interface (Command-line tool)
├── Python API (Programmatic access)
└── JSON Response (Retrieved chunks, citations, confidence, metadata)
```

---

## ISRO USE CASE - EXAMPLE QUERIES

The system must handle diverse query types across government documentation:

**Regulatory & Compliance Queries:**
- "What is the delegation of financial power for procurement above Rs 10 lakhs according to GFR?"
- "Show me the tender publication requirements for indigenous vendors"
- "What are the audit requirements for contracts exceeding Rs 1 crore?"
- "Explain the procurement procedure for proprietary items"
- "What is the composition of the tender committee as per GFR rules?"

**Technical Documentation Queries:**
- "What were the root causes identified in the [project name] failure analysis?"
- "Show me telemetry anomalies detected during the mission"
- "Explain the thermal protection system failure in technical memo TM-2023-045"
- "What were the recommendations from the failure review board?"
- "Compare quality assurance protocols between VSSC and LPSC"

**Operational & Decision Support:**
- "What are the quality assurance checkpoints for satellite integration?"
- "Show procurement manual guidelines for vendor evaluation"
- "What documentation is required for procurement above Rs 50 lakhs?"
- "Explain the delegation of powers for administrative approvals"
- "What are the compliance requirements for imported components?"

**Cross-Document Synthesis:**
- "Compare GFR rules and procurement manual guidelines for vendor selection"
- "Show me all quality issues identified in recent telemetry data"
- "What are the common failure modes across multiple technical memos?"
- "Find regulatory requirements that apply to satellite procurement"

---

## IMPLEMENTATION APPROACH - INCREMENTAL DEVELOPMENT

**Phase-by-Phase Strategy:**

```
Phase 1: Secure Document Ingestion ✅ BUILD FIRST
├── Components: DocumentLoader + PIIAnonymizer + EncryptionManager
├── Testing: Load GFR PDF, verify PII redaction, encryption
├── Duration: 2-3 hours
└── Deliverable: Secure document loading with maximum security

Phase 2: Embedding & Storage ✅ BUILD SECOND
├── Components: ChunkGenerator + EmbeddingGenerator + FAISSIndex + DomainManager
├── Testing: Chunk docs, embeddings, encrypted index, auto-clustering
├── Duration: 3-4 hours
└── Deliverable: Encrypted vector store with content-based domains

Phase 3: Retrieval Pipeline ✅ BUILD THIRD
├── Components: HybridRetriever + CrossEncoderReranker + RetrievalPipeline
├── Testing: Query ISRO docs, verify retrieval quality
├── Duration: 3-4 hours
└── Deliverable: Working retrieval with user query support

Phase 4: Validation & Security ✅ BUILD FOURTH
├── Components: ValidationPipeline + Security checks
├── Testing: Citation verification, confidence scoring
├── Duration: 2 hours
└── Deliverable: Validated retrieval results

Phase 5: Monitoring & Audit ✅ BUILD FIFTH
├── Components: MonitoringSystem with comprehensive audit trails
├── Testing: Log queries, audit integrity, compliance reports
├── Duration: 2 hours
└── Deliverable: Complete audit logging system

Phase 6: Integration & CLI ✅ BUILD SIXTH
├── Components: SLRAGPipeline orchestrator + CLI + Python API
├── Testing: End-to-end with ISRO docs, performance benchmarks
├── Duration: 2-3 hours
└── Deliverable: Complete working system

Phase 7: ⏸️ LLM Generation (SKIP FOR NOW)
└── Will be added after Llama 3.2 3B setup
```

**Testing Protocol (THOROUGH APPROACH):**

After EACH phase:
1. ✅ Unit tests for all components (pytest)
2. ✅ Integration tests with previous phases
3. ✅ Security validation (encryption, PII, audit)
4. ✅ Performance benchmarks (latency, memory, GPU)
5. ✅ Real testing with ISRO sample documents
6. ✅ Code review and documentation
7. ✅ User validation before proceeding

**⚠️ CRITICAL: DO NOT PROCEED to next phase until:**
- All tests pass (100% success rate)
- Performance meets targets
- Security requirements validated
- No bugs or edge cases remaining
- User reviews and approves deliverable

---

## DETAILED IMPLEMENTATION REQUIREMENTS

### PHASE 1: SECURE DATA INGESTION (Layer 1)

#### Component 1.1: DocumentLoader Class

**Purpose**: Load, sanitize, and validate PDF documents

**Requirements**:
```python
class DocumentLoader:
    """
    Loads PDF documents with sanitization and PII removal.
    
    Features:
    - Multi-format PDF support (text-based + OCR for scanned)
    - HTML/JavaScript/malicious code removal
    - Text normalization (Unicode, whitespace)
    - Metadata extraction (author, title, creation date)
    - File integrity validation (SHA-256 hash)
    """
    
    def __init__(self, 
                 ocr_enabled: bool = True,
                 max_file_size_mb: int = 50,
                 sanitize: bool = True):
        pass
    
    def load_pdf(self, filepath: str) -> Document:
        """
        Load a single PDF document.
        
        Steps:
        1. Validate file exists and size < max_file_size_mb
        2. Extract text using PyMuPDF
        3. If text extraction fails or yields < 100 chars, run OCR
        4. Sanitize content (remove HTML tags, scripts, special chars)
        5. Extract metadata
        6. Generate document ID (SHA-256 hash of content)
        7. Return Document object
        """
        pass
    
    def load_directory(self, dirpath: str, 
                       recursive: bool = True) -> List[Document]:
        """
        Load all PDFs from a directory.
        
        Returns:
        - List of Document objects
        - Loading statistics (success count, failures, avg size)
        """
        pass
```

**Document Schema**:
```python
@dataclass
class Document:
    doc_id: str              # SHA-256 hash
    content: str             # Sanitized text content
    metadata: Dict[str, Any] # {title, author, pages, filepath, etc.}
    domain: str              # Auto-detected or manual
    timestamp: str           # ISO format
    sanitized: bool          # True if sanitization applied
    pii_removed: bool        # True if PII anonymization applied
    word_count: int
    char_count: int
```

#### Component 1.2: PIIAnonymizer Class (Enhanced for Government Data)

**Purpose**: Detect and remove personally identifiable information (including Indian PII)

**Requirements**:
```python
import re
from typing import Dict, Tuple, List

class PIIAnonymizer:
    """
    Detects and redacts PII using regex patterns + NER.
    
    Supported PII Types (India-specific):
    - Email addresses
    - Phone numbers (Indian format: +91-XXXXXXXXXX, 10-digit)
    - Aadhaar numbers (XXXX-XXXX-XXXX)
    - PAN card numbers (XXXXX9999X format)
    - Passport numbers
    - Social Security Numbers (SSN - international)
    - Credit/Debit card numbers (16-digit)
    - IP addresses (IPv4, IPv6)
    - Names (using spaCy NER - optional)
    - Dates of birth
    - Government employee IDs
    """
    
    def __init__(self, 
                 enable_ner: bool = False,
                 replacement_token: str = "[REDACTED]",
                 log_detections: bool = True):
        self.enable_ner = enable_ner
        self.replacement_token = replacement_token
        self.log_detections = log_detections
        
        # PII patterns
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone_indian': r'\+91[-\s]?\d{10}|\b\d{10}\b',  # Indian phone numbers
            'aadhaar': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',   # Aadhaar format
            'pan': r'\b[A-Z]{5}\d{4}[A-Z]\b',                 # PAN card
            'passport': r'\b[A-Z]\d{7}\b',                    # Indian passport
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',                 # US SSN
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            'dob': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            'employee_id': r'\b(EMP|ISRO|ID)[-_]?\d{5,8}\b',  # Government employee IDs
        }
        
        # Initialize spaCy for name detection (optional)
        if enable_ner:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
            except:
                print("Warning: spaCy not available for name detection")
                self.enable_ner = False
        
        # Detection statistics
        self.total_detections = {pii_type: 0 for pii_type in self.patterns.keys()}
    
    def anonymize(self, text: str, preserve_structure: bool = True) -> Tuple[str, Dict[str, int]]:
        """
        Remove PII from text and return anonymized version with stats.
        
        Args:
            text: Input text to anonymize
            preserve_structure: If True, replace with [TYPE_REDACTED] to maintain context
        
        Returns:
            Tuple of (anonymized_text, detection_counts)
        """
        anonymized = text
        detections = {}
        
        # Apply regex-based anonymization
        for pii_type, pattern in self.patterns.items():
            matches = re.findall(pattern, anonymized)
            count = len(matches)
            detections[pii_type] = count
            self.total_detections[pii_type] += count
            
            # Replace with specific or generic token
            if preserve_structure:
                replacement = f'[{pii_type.upper()}_REDACTED]'
            else:
                replacement = self.replacement_token
            
            anonymized = re.sub(pattern, replacement, anonymized)
        
        # NER-based name detection (optional)
        if self.enable_ner:
            names_removed = self._redact_names(anonymized)
            anonymized = names_removed
            detections['names_ner'] = len(re.findall(r'\[PERSON_REDACTED\]', anonymized))
        
        # Log if enabled
        if self.log_detections and sum(detections.values()) > 0:
            print(f"[PII] Detected and redacted: {detections}")
        
        return anonymized, detections
    
    def _redact_names(self, text: str) -> str:
        """Use spaCy NER to detect and redact person names."""
        if not self.enable_ner:
            return text
        
        doc = self.nlp(text)
        redacted = text
        
        # Replace PERSON entities
        for ent in reversed(doc.ents):  # Reverse to preserve indices
            if ent.label_ == "PERSON":
                redacted = redacted[:ent.start_char] + "[PERSON_REDACTED]" + redacted[ent.end_char:]
        
        return redacted
    
    def get_detection_patterns(self) -> Dict[str, str]:
        """Return all regex patterns for transparency and audit."""
        return self.patterns.copy()
    
    def get_statistics(self) -> Dict[str, int]:
        """Return total detection statistics across all processed documents."""
        return self.total_detections.copy()
    
    def validate_anonymization(self, original: str, anonymized: str) -> Dict[str, Any]:
        """
        Validate that anonymization was successful.
        
        Returns:
        - pii_remaining: List of PII types still present
        - success: bool
        - details: Dict with analysis
        """
        remaining_pii = {}
        
        for pii_type, pattern in self.patterns.items():
            matches = re.findall(pattern, anonymized)
            if matches:
                remaining_pii[pii_type] = len(matches)
        
        return {
            'success': len(remaining_pii) == 0,
            'pii_remaining': remaining_pii,
            'total_removed': sum(self.total_detections.values())
        }
```

**Testing Requirements for Phase 1**:
```python
def test_document_loader():
    """
    Test cases:
    1. Load valid text-based PDF → assert content extracted
    2. Load scanned PDF → assert OCR triggered
    3. Load PDF with HTML/scripts → assert sanitized
    4. Load oversized PDF → assert raises error
    5. Load corrupted PDF → assert handles gracefully
    6. Load directory with 5 PDFs → assert all loaded
    """
    pass

def test_pii_anonymizer():
    """
    Test cases:
    1. Text with email → assert email redacted
    2. Text with SSN → assert SSN redacted
    3. Text with multiple PII types → assert all redacted
    4. Text with no PII → assert unchanged
    5. Edge case: Email in code block → assert not over-redacted
    """
    pass
```

---

### PHASE 2: EMBEDDING & SECURE STORAGE (Layer 2)

#### Component 2.1: ChunkGenerator Class

**Purpose**: Split documents into semantically meaningful chunks

**Requirements**:
```python
class ChunkGenerator:
    """
    Generates overlapping chunks using semantic-aware splitting.
    
    Strategy:
    - Primary: Split on sentence boundaries (using nltk or spaCy)
    - Target chunk size: 512 tokens (~384 words)
    - Overlap: 50 tokens (10%)
    - Preserve paragraph context where possible
    - Add metadata: doc_id, chunk_index, start_char, end_char
    """
    
    def __init__(self, 
                 chunk_size: int = 512,
                 overlap: int = 50,
                 tokenizer_model: str = "all-mpnet-base-v2"):
        pass
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Split document into chunks.
        
        Algorithm:
        1. Split text into sentences
        2. Group sentences into chunks of ~chunk_size tokens
        3. Add overlap by including last overlap tokens from previous chunk
        4. Create Chunk objects with metadata
        5. Return list of chunks
        """
        pass
    
    def chunk_batch(self, documents: List[Document]) -> List[Chunk]:
        """Chunk multiple documents efficiently."""
        pass
```

**Chunk Schema**:
```python
@dataclass
class Chunk:
    chunk_id: str           # Format: {doc_id}_chunk_{index}
    doc_id: str             # Parent document ID
    content: str            # Chunk text
    chunk_index: int        # Position in document
    start_char: int         # Start position in original doc
    end_char: int           # End position in original doc
    token_count: int
    embedding: Optional[np.ndarray] = None  # 768-dim vector
    domain: str = ""
```

#### Component 2.2: EmbeddingGenerator Class

**Purpose**: Generate embeddings using all-mpnet-base-v2 with GPU acceleration

**Requirements**:
```python
class EmbeddingGenerator:
    """
    Generates embeddings using sentence-transformers.
    
    Features:
    - GPU-accelerated batch inference
    - Batch size optimization for 4GB VRAM
    - L2 normalization for cosine similarity
    - Progress tracking for large batches
    """
    
    def __init__(self, 
                 model_name: str = "all-mpnet-base-v2",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 32):
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.dimension = 768
    
    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Generate embeddings for chunks.
        
        Steps:
        1. Extract text from chunks
        2. Batch encode with show_progress_bar=True
        3. L2 normalize embeddings
        4. Assign embeddings to chunk objects
        5. Return updated chunks
        """
        pass
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        pass
    
    def compute_similarity(self, 
                          query_emb: np.ndarray, 
                          chunk_embs: np.ndarray) -> np.ndarray:
        """Compute cosine similarities."""
        return util.cos_sim(query_emb, chunk_embs).cpu().numpy()[0]
```

#### Component 2.3: EncryptionManager Class (NEW - Maximum Security)

**Purpose**: Handle AES-256 encryption for all stored data

**Requirements**:
```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import base64
import os
import json

class EncryptionManager:
    """
    Manages AES-256 encryption for data at rest.
    
    Features:
    - AES-256 encryption via Fernet (symmetric)
    - Secure key derivation from master password
    - Separate encryption for different data types
    - Key rotation support
    - Encrypted index storage
    """
    
    def __init__(self, 
                 master_key_path: str = "./keys/master.key",
                 auto_generate: bool = True):
        self.master_key_path = master_key_path
        
        # Ensure key directory exists
        os.makedirs(os.path.dirname(master_key_path), exist_ok=True)
        
        # Load or generate master key
        if os.path.exists(master_key_path):
            with open(master_key_path, 'rb') as f:
                self.master_key = f.read()
        elif auto_generate:
            self.master_key = Fernet.generate_key()
            with open(master_key_path, 'wb') as f:
                f.write(self.master_key)
            os.chmod(master_key_path, 0o600)  # Read/write for owner only
            print(f"[SECURITY] Generated new master key at {master_key_path}")
        else:
            raise ValueError("Master key not found and auto_generate=False")
        
        self.cipher = Fernet(self.master_key)
    
    def encrypt_text(self, text: str) -> bytes:
        """Encrypt text data."""
        return self.cipher.encrypt(text.encode('utf-8'))
    
    def decrypt_text(self, encrypted_data: bytes) -> str:
        """Decrypt text data."""
        return self.cipher.decrypt(encrypted_data).decode('utf-8')
    
    def encrypt_file(self, input_path: str, output_path: str) -> None:
        """Encrypt an entire file."""
        with open(input_path, 'rb') as f:
            data = f.read()
        
        encrypted = self.cipher.encrypt(data)
        
        with open(output_path, 'wb') as f:
            f.write(encrypted)
        
        print(f"[SECURITY] Encrypted {input_path} -> {output_path}")
    
    def decrypt_file(self, input_path: str, output_path: str) -> None:
        """Decrypt an entire file."""
        with open(input_path, 'rb') as f:
            encrypted_data = f.read()
        
        decrypted = self.cipher.decrypt(encrypted_data)
        
        with open(output_path, 'wb') as f:
            f.write(decrypted)
    
    def encrypt_numpy_array(self, array: np.ndarray) -> bytes:
        """Encrypt numpy array (for embeddings)."""
        # Serialize to bytes
        array_bytes = array.tobytes()
        metadata = {
            'shape': array.shape,
            'dtype': str(array.dtype)
        }
        
        # Combine metadata and data
        combined = json.dumps(metadata).encode() + b'|||' + array_bytes
        
        return self.cipher.encrypt(combined)
    
    def decrypt_numpy_array(self, encrypted_data: bytes) -> np.ndarray:
        """Decrypt numpy array."""
        decrypted = self.cipher.decrypt(encrypted_data)
        
        # Split metadata and data
        parts = decrypted.split(b'|||')
        metadata = json.loads(parts[0].decode())
        array_bytes = parts[1]
        
        # Reconstruct array
        array = np.frombuffer(array_bytes, dtype=metadata['dtype'])
        array = array.reshape(metadata['shape'])
        
        return array
    
    def secure_delete(self, filepath: str, passes: int = 3) -> None:
        """
        Securely delete a file by overwriting with random data.
        
        Args:
            filepath: File to delete
            passes: Number of overwrite passes (default: 3)
        """
        if not os.path.exists(filepath):
            return
        
        file_size = os.path.getsize(filepath)
        
        with open(filepath, 'ba+') as f:
            for _ in range(passes):
                f.seek(0)
                f.write(os.urandom(file_size))
        
        os.remove(filepath)
        print(f"[SECURITY] Securely deleted {filepath}")
    
    def rotate_key(self, new_master_key_path: str) -> None:
        """
        Rotate encryption key (for periodic security updates).
        
        Note: Requires re-encryption of all stored data.
        """
        # Generate new key
        new_key = Fernet.generate_key()
        
        with open(new_master_key_path, 'wb') as f:
            f.write(new_key)
        
        os.chmod(new_master_key_path, 0o600)
        print(f"[SECURITY] New encryption key generated. Re-encrypt all data.")
```

#### Component 2.4: FAISSIndex Class (Updated with Encryption)

**Purpose**: Build and manage FAISS vector index with GPU support and encryption

**Requirements**:
```python
class FAISSIndex:
    """
    FAISS index manager with GPU acceleration, persistence, and encryption.
    
    Index Strategy:
    - For < 1000 docs: IndexFlatIP (exact search)
    - For 1000+ docs: IndexIVFFlat with nlist=sqrt(N)
    - GPU training and search
    - Encrypted storage (via EncryptionManager)
    - Periodic index optimization
    """
    
    def __init__(self, 
                 dimension: int = 768,
                 use_gpu: bool = True,
                 index_path: str = "./faiss_index",
                 encryption_manager: Optional[EncryptionManager] = None):
        self.dimension = dimension
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.index_path = index_path
        self.encryption = encryption_manager
        self.index = None
        self.chunk_metadata = []  # Store chunk IDs and metadata separately
        
        os.makedirs(index_path, exist_ok=True)
    
    def build_index(self, chunks: List[Chunk]) -> None:
        """
        Build FAISS index from chunks.
        
        Steps:
        1. Extract embeddings as numpy array (N x 768)
        2. Normalize vectors (L2 norm)
        3. Create index (IndexFlatIP or IndexIVFFlat)
        4. If GPU available, move to GPU
        5. Add vectors to index
        6. Store chunk metadata separately (encrypted)
        7. Save to disk
        """
        # Extract embeddings
        embeddings = np.array([chunk.embedding for chunk in chunks]).astype('float32')
        N = len(embeddings)
        
        # L2 normalize for cosine similarity via inner product
        faiss.normalize_L2(embeddings)
        
        # Choose index type based on corpus size
        if N < 1000:
            print(f"[INDEX] Using IndexFlatIP (exact search) for {N} vectors")
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            nlist = int(np.sqrt(N))
            print(f"[INDEX] Using IndexIVFFlat with {nlist} clusters for {N} vectors")
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            
            # Train index
            print("[INDEX] Training index...")
            self.index.train(embeddings)
        
        # Move to GPU if available
        if self.use_gpu:
            print("[INDEX] Moving to GPU...")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        # Add vectors
        print(f"[INDEX] Adding {N} vectors...")
        self.index.add(embeddings)
        
        # Store metadata
        self.chunk_metadata = [
            {
                'chunk_id': chunk.chunk_id,
                'doc_id': chunk.doc_id,
                'chunk_index': chunk.chunk_index,
                'domain': chunk.domain,
                'content': chunk.content,
                'token_count': chunk.token_count
            }
            for chunk in chunks
        ]
        
        print(f"[INDEX] ✓ Index built with {self.index.ntotal} vectors")
    
    def search(self, 
               query_embedding: np.ndarray, 
               k: int = 10,
               filter_domains: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search index for top-k similar chunks.
        
        Args:
            query_embedding: Query vector (768,)
            k: Number of results
            filter_domains: Optional list of domains to restrict search
        
        Returns:
        - scores: Array of similarity scores (k,)
        - indices: Array of chunk indices (k,)
        """
        # Ensure query is normalized
        query_emb = query_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_emb)
        
        # Search
        scores, indices = self.index.search(query_emb, k)
        
        # Filter by domain if specified
        if filter_domains:
            filtered_scores = []
            filtered_indices = []
            
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunk_metadata):
                    if self.chunk_metadata[idx]['domain'] in filter_domains:
                        filtered_scores.append(score)
                        filtered_indices.append(idx)
            
            scores = np.array([filtered_scores])
            indices = np.array([filtered_indices])
        
        return scores[0], indices[0]
    
    def save_index(self, filepath: str = None) -> None:
        """Save index and metadata to disk with encryption."""
        if filepath is None:
            filepath = os.path.join(self.index_path, "index.faiss")
        
        # Move index to CPU if on GPU
        index_to_save = self.index
        if self.use_gpu:
            index_to_save = faiss.index_gpu_to_cpu(self.index)
        
        # Save FAISS index (binary format)
        temp_index_path = filepath + ".tmp"
        faiss.write_index(index_to_save, temp_index_path)
        
        # Encrypt if encryption manager available
        if self.encryption:
            self.encryption.encrypt_file(temp_index_path, filepath + ".encrypted")
            os.remove(temp_index_path)
            print(f"[SECURITY] Index encrypted and saved to {filepath}.encrypted")
        else:
            os.rename(temp_index_path, filepath)
            print(f"[INDEX] Saved to {filepath}")
        
        # Save metadata (always encrypted if encryption available)
        metadata_path = os.path.join(self.index_path, "metadata.json")
        metadata_json = json.dumps(self.chunk_metadata, indent=2)
        
        if self.encryption:
            encrypted_metadata = self.encryption.encrypt_text(metadata_json)
            with open(metadata_path + ".encrypted", 'wb') as f:
                f.write(encrypted_metadata)
            print(f"[SECURITY] Metadata encrypted and saved")
        else:
            with open(metadata_path, 'w') as f:
                f.write(metadata_json)
            print(f"[INDEX] Metadata saved")
    
    def load_index(self, filepath: str = None) -> None:
        """Load index and metadata from disk with decryption."""
        if filepath is None:
            filepath = os.path.join(self.index_path, "index.faiss")
        
        # Load FAISS index
        if self.encryption and os.path.exists(filepath + ".encrypted"):
            # Decrypt first
            temp_path = filepath + ".tmp"
            self.encryption.decrypt_file(filepath + ".encrypted", temp_path)
            self.index = faiss.read_index(temp_path)
            os.remove(temp_path)
            print(f"[SECURITY] Index decrypted and loaded")
        else:
            self.index = faiss.read_index(filepath)
            print(f"[INDEX] Loaded from {filepath}")
        
        # Move to GPU if available
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            print("[INDEX] Moved to GPU")
        
        # Load metadata
        metadata_path = os.path.join(self.index_path, "metadata.json")
        
        if self.encryption and os.path.exists(metadata_path + ".encrypted"):
            with open(metadata_path + ".encrypted", 'rb') as f:
                encrypted_data = f.read()
            metadata_json = self.encryption.decrypt_text(encrypted_data)
            self.chunk_metadata = json.loads(metadata_json)
            print(f"[SECURITY] Metadata decrypted and loaded")
        else:
            with open(metadata_path, 'r') as f:
                self.chunk_metadata = json.load(f)
            print(f"[INDEX] Metadata loaded")
        
        print(f"[INDEX] ✓ Loaded {self.index.ntotal} vectors")
    
    def add_chunks(self, new_chunks: List[Chunk]) -> None:
        """Incrementally add new chunks to existing index."""
        new_embeddings = np.array([chunk.embedding for chunk in new_chunks]).astype('float32')
        faiss.normalize_L2(new_embeddings)
        
        self.index.add(new_embeddings)
        
        # Update metadata
        for chunk in new_chunks:
            self.chunk_metadata.append({
                'chunk_id': chunk.chunk_id,
                'doc_id': chunk.doc_id,
                'chunk_index': chunk.chunk_index,
                'domain': chunk.domain,
                'content': chunk.content,
                'token_count': chunk.token_count
            })
        
        print(f"[INDEX] Added {len(new_chunks)} new chunks. Total: {self.index.ntotal}")
    
    def get_chunk_by_index(self, index: int) -> Dict[str, Any]:
        """Retrieve chunk metadata by index."""
        if 0 <= index < len(self.chunk_metadata):
            return self.chunk_metadata[index]
        return None
```

#### Component 2.5: DomainManager Class (Content-Based Clustering) (Content-Based Clustering)

**Purpose**: Automatically detect document domains using content-based clustering

**Requirements**:
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class DomainManager:
    """
    Manages domain-specific embeddings using automatic clustering.
    
    Features:
    - **Automatic domain detection** via K-means clustering on embeddings
    - Optimal cluster count selection (silhouette analysis)
    - Domain labeling based on top TF-IDF keywords
    - Domain centroids for query routing
    - Support for manual domain override if needed
    """
    
    def __init__(self, 
                 min_clusters: int = 2,
                 max_clusters: int = 10,
                 auto_detect: bool = True):
        self.domains: Dict[str, np.ndarray] = {}  # {domain_name: centroid_embedding}
        self.domain_chunks: Dict[str, List[str]] = {}  # {domain: [chunk_ids]}
        self.cluster_labels: Dict[str, int] = {}  # {chunk_id: cluster_id}
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.auto_detect = auto_detect
    
    def detect_domains(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """
        Automatically detect domains using clustering.
        
        Algorithm:
        1. Extract all chunk embeddings into matrix (N x 768)
        2. Try K-means with k=[min_clusters, max_clusters]
        3. Compute silhouette score for each k
        4. Select optimal k (highest silhouette score)
        5. Assign cluster labels to chunks
        6. Generate domain names from top TF-IDF terms
        7. Compute centroid for each cluster
        
        Returns:
        - domain_mapping: {domain_name: [chunk_ids]}
        - num_domains: int
        - silhouette_score: float
        - domain_keywords: {domain_name: [top_keywords]}
        """
        # Extract embeddings
        embeddings = np.array([chunk.embedding for chunk in chunks])
        
        # Find optimal number of clusters
        best_k = self._find_optimal_clusters(embeddings)
        
        # Cluster with optimal k
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Assign domains to chunks
        for i, chunk in enumerate(chunks):
            cluster_id = cluster_labels[i]
            chunk.domain = f"domain_{cluster_id}"
            self.cluster_labels[chunk.chunk_id] = cluster_id
        
        # Generate meaningful domain names using TF-IDF
        domain_names = self._generate_domain_names(chunks, cluster_labels, best_k)
        
        # Rename domains with meaningful labels
        for chunk in chunks:
            old_domain = chunk.domain
            cluster_id = int(old_domain.split('_')[1])
            chunk.domain = domain_names[cluster_id]
        
        # Compute centroids
        self.compute_centroids(chunks)
        
        return {
            'num_domains': best_k,
            'domain_names': list(domain_names.values()),
            'domain_keywords': self._extract_domain_keywords(chunks, best_k),
            'silhouette_score': silhouette_score(embeddings, cluster_labels)
        }
    
    def _find_optimal_clusters(self, embeddings: np.ndarray) -> int:
        """
        Find optimal number of clusters using silhouette analysis.
        
        Returns optimal k between min_clusters and max_clusters.
        """
        scores = []
        k_range = range(self.min_clusters, min(self.max_clusters + 1, len(embeddings)))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            scores.append((k, score))
        
        # Return k with highest silhouette score
        best_k = max(scores, key=lambda x: x[1])[0]
        return best_k
    
    def _generate_domain_names(self, 
                               chunks: List[Chunk], 
                               cluster_labels: np.ndarray,
                               num_clusters: int) -> Dict[int, str]:
        """
        Generate meaningful domain names using TF-IDF on cluster content.
        
        Returns:
        - {cluster_id: domain_name}
        
        Example output:
        {
            0: "procurement_financial",
            1: "technical_telemetry",
            2: "quality_assurance"
        }
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Group chunks by cluster
        cluster_texts = {i: [] for i in range(num_clusters)}
        for chunk, label in zip(chunks, cluster_labels):
            cluster_texts[label].append(chunk.content)
        
        # Extract top keywords for each cluster
        domain_names = {}
        for cluster_id, texts in cluster_texts.items():
            combined_text = ' '.join(texts)
            
            # TF-IDF to find most important terms
            vectorizer = TfidfVectorizer(max_features=3, stop_words='english')
            tfidf = vectorizer.fit_transform([combined_text])
            keywords = vectorizer.get_feature_names_out()
            
            # Create domain name from top keywords
            domain_name = '_'.join(keywords[:2])
            domain_names[cluster_id] = domain_name
        
        return domain_names
    
    def _extract_domain_keywords(self, 
                                 chunks: List[Chunk], 
                                 num_clusters: int) -> Dict[str, List[str]]:
        """Extract top 10 keywords for each domain for interpretability."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        domain_keywords = {}
        domain_texts = {}
        
        # Group by domain
        for chunk in chunks:
            if chunk.domain not in domain_texts:
                domain_texts[chunk.domain] = []
            domain_texts[chunk.domain].append(chunk.content)
        
        # Extract keywords
        for domain, texts in domain_texts.items():
            vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
            tfidf = vectorizer.fit_transform(texts)
            keywords = vectorizer.get_feature_names_out().tolist()
            domain_keywords[domain] = keywords
        
        return domain_keywords
    
    def compute_centroids(self, chunks: List[Chunk]) -> None:
        """
        Compute centroid embedding for each domain.
        
        Steps:
        1. Group chunks by domain
        2. For each domain, compute mean embedding
        3. L2 normalize centroids
        4. Store in self.domains
        """
        domain_embeddings = {}
        
        for chunk in chunks:
            if chunk.domain not in domain_embeddings:
                domain_embeddings[chunk.domain] = []
                self.domain_chunks[chunk.domain] = []
            
            domain_embeddings[chunk.domain].append(chunk.embedding)
            self.domain_chunks[chunk.domain].append(chunk.chunk_id)
        
        # Compute centroids
        for domain, embeddings in domain_embeddings.items():
            centroid = np.mean(embeddings, axis=0)
            # L2 normalize
            centroid = centroid / np.linalg.norm(centroid)
            self.domains[domain] = centroid
    
    def route_query(self, 
                    query_embedding: np.ndarray, 
                    top_k_domains: int = 3) -> List[Tuple[str, float]]:
        """
        Determine which domains are relevant for query.
        
        Returns:
        - List of (domain_name, similarity_score) tuples sorted by relevance
        """
        similarities = []
        
        for domain, centroid in self.domains.items():
            sim = np.dot(query_embedding, centroid)
            similarities.append((domain, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k_domains]
    
    def get_domain_stats(self) -> Dict[str, Any]:
        """Return statistics for each domain."""
        stats = {}
        
        for domain in self.domains.keys():
            stats[domain] = {
                'num_chunks': len(self.domain_chunks[domain]),
                'centroid_norm': np.linalg.norm(self.domains[domain]),
            }
        
        return stats
    
    def visualize_domains(self, chunks: List[Chunk], output_path: str = "./domain_viz.png"):
        """
        Create 2D visualization of domains using t-SNE or UMAP.
        
        Helpful for understanding domain separation quality.
        """
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        
        embeddings = np.array([c.embedding for c in chunks])
        domains = [c.domain for c in chunks]
        
        # Reduce to 2D
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Plot
        plt.figure(figsize=(12, 8))
        for domain in set(domains):
            mask = [d == domain for d in domains]
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                label=domain,
                alpha=0.6
            )
        
        plt.legend()
        plt.title("Document Domain Clustering Visualization")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Domain visualization saved to {output_path}")
```

**Testing Requirements for Phase 2**:
```python
def test_chunk_generator():
    """
    Test cases:
    1. Chunk 1000-word doc → assert ~3-4 chunks created
    2. Verify overlap exists between consecutive chunks
    3. Verify chunk token counts ≈ 512 ± 50
    4. Verify metadata (doc_id, positions) correct
    """
    pass

def test_embedding_generator():
    """
    Test cases:
    1. Generate embedding for single query → assert shape (768,)
    2. Batch embed 100 chunks → assert GPU utilized
    3. Verify L2 normalization: np.linalg.norm(emb) ≈ 1.0
    4. Same text → same embedding (reproducibility)
    """
    pass

def test_faiss_index():
    """
    Test cases:
    1. Build index with 50 chunks → assert no errors
    2. Search for known query → assert correct chunks returned
    3. Save and load index → assert persistence works
    4. Add 10 new chunks → assert incremental update works
    """
    pass

def test_domain_manager():
    """
    Test cases:
    1. Compute centroids for 3 domains → assert 3 centroids created
    2. Query routing → assert relevant domains ranked highest
    3. Add new domain → assert centroid computed correctly
    """
    pass
```

---

### PHASE 3: MULTI-DOMAIN RETRIEVAL & RE-RANKING (Layer 3)

#### Component 3.1: HybridRetriever Class

**Purpose**: Combine BM25 (sparse) and dense retrieval with score fusion

**Requirements**:
```python
from rank_bm25 import BM25Okapi

class HybridRetriever:
    """
    Hybrid retrieval combining BM25 and dense search.
    
    Score Fusion Formula:
    final_score = alpha * dense_score + (1 - alpha) * bm25_score
    where alpha = 0.7 (favor dense retrieval)
    
    Features:
    - BM25 for lexical matching
    - Dense FAISS for semantic matching
    - Reciprocal Rank Fusion (RRF) as alternative
    - Domain-filtered retrieval
    """
    
    def __init__(self, 
                 faiss_index: FAISSIndex,
                 chunks: List[Chunk],
                 alpha: float = 0.7):
        self.faiss_index = faiss_index
        self.chunks = chunks
        self.alpha = alpha
        
        # Build BM25 index
        tokenized_corpus = [chunk.content.split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def retrieve(self, 
                 query: str,
                 query_embedding: np.ndarray,
                 k: int = 20,
                 domains: Optional[List[str]] = None) -> List[Tuple[Chunk, float]]:
        """
        Hybrid retrieval with score fusion.
        
        Algorithm:
        1. BM25 retrieval: Get top-k chunks with scores
        2. Dense retrieval: Get top-k chunks with scores
        3. Normalize scores to [0, 1] range
        4. Compute fusion scores
        5. Merge and deduplicate results
        6. Sort by final score
        7. Filter by domain if specified
        8. Return top-k chunks with scores
        """
        pass
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Min-max normalization to [0, 1]."""
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
```

#### Component 3.2: CrossEncoderReranker Class

**Purpose**: Re-rank retrieved chunks using cross-encoder model

**Requirements**:
```python
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    """
    Re-ranks retrieved chunks using cross-encoder.
    
    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    - Input: [query, chunk_text] pairs
    - Output: Relevance score (0-1)
    - CPU inference (cross-encoders don't parallelize well on GPU)
    """
    
    def __init__(self, 
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, 
               query: str, 
               chunks: List[Tuple[Chunk, float]], 
               top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Re-rank chunks using cross-encoder.
        
        Steps:
        1. Create [query, chunk.content] pairs
        2. Batch predict relevance scores
        3. Sort chunks by cross-encoder scores
        4. Return top-k chunks with updated scores
        """
        pass
```

#### Component 3.3: RetrievalPipeline Class

**Purpose**: Orchestrate the full retrieval flow

**Requirements**:
```python
class RetrievalPipeline:
    """
    End-to-end retrieval pipeline.
    
    Flow:
    Query → Domain Routing → Hybrid Retrieval → Similarity Filter → 
    Cross-Encoder Re-ranking → Return Top-K
    """
    
    def __init__(self, 
                 embedding_generator: EmbeddingGenerator,
                 domain_manager: DomainManager,
                 hybrid_retriever: HybridRetriever,
                 cross_encoder: CrossEncoderReranker,
                 similarity_threshold: float = 0.5):
        pass
    
    def retrieve(self, 
                 query: str, 
                 top_k: int = 5,
                 enable_reranking: bool = True) -> List[Tuple[Chunk, float]]:
        """
        Full retrieval pipeline.
        
        Steps:
        1. Preprocess query (lowercase, remove special chars)
        2. Generate query embedding
        3. Route to relevant domains
        4. Hybrid retrieval (k=20)
        5. Filter by similarity_threshold
        6. Cross-encoder re-ranking (top-5)
        7. Return results
        
        Returns:
        - List of (Chunk, score) tuples sorted by relevance
        """
        pass
    
    def batch_retrieve(self, 
                       queries: List[str], 
                       top_k: int = 5) -> List[List[Tuple[Chunk, float]]]:
        """Retrieve for multiple queries efficiently."""
        pass
```

**Testing Requirements for Phase 3**:
```python
def test_hybrid_retriever():
    """
    Test cases:
    1. Query with exact keyword match → assert BM25 contributes
    2. Query with semantic match only → assert dense retrieval works
    3. Verify score fusion: alpha * dense + (1-alpha) * bm25
    4. Domain filtering → assert only specified domains returned
    """
    pass

def test_cross_encoder_reranker():
    """
    Test cases:
    1. Re-rank 10 chunks → assert top-5 extracted
    2. Verify scores are in [0, 1] range
    3. Highly relevant chunk → assert ranked #1
    4. Irrelevant chunk → assert ranked low
    """
    pass

def test_retrieval_pipeline():
    """
    Test cases:
    1. Known query → assert correct documents retrieved
    2. Similarity threshold filtering → assert low-score chunks removed
    3. Multi-domain query → assert results from multiple domains
    4. Empty result scenario → assert handles gracefully
    """
    pass
```

---

### PHASE 4: PROMPT INTEGRITY & CONSTRAINED GENERATION (Layer 4)

#### Component 4.1: PromptBuilder Class

**Purpose**: Construct prompts with strict hierarchy and injection protection

**Requirements**:
```python
class PromptBuilder:
    """
    Builds prompts with hierarchical structure.
    
    Hierarchy (immutable order):
    1. SYSTEM: Role definition, constraints, behavior rules
    2. CONTEXT: Retrieved chunks with citations
    3. QUERY: User question
    
    Features:
    - Injection detection (look for prompt breaking patterns)
    - Token limit enforcement
    - Citation formatting
    - Guardrails against hallucination
    """
    
    SYSTEM_PROMPT = """You are a precise question-answering assistant. Your responses must:
1. ONLY use information from the provided context
2. CITE sources using [Source: doc_id, chunk_index]
3. If information is not in context, say "I cannot answer based on the provided documents"
4. Be concise and factual
5. Never speculate or add external knowledge"""
    
    def __init__(self, max_context_tokens: int = 6000):
        self.max_context_tokens = max_context_tokens
    
    def build_prompt(self, 
                     query: str, 
                     retrieved_chunks: List[Tuple[Chunk, float]]) -> str:
        """
        Build complete prompt.
        
        Format:
        ```
        SYSTEM: {SYSTEM_PROMPT}
        
        CONTEXT:
        [1] (Score: 0.95, Source: doc_123, Chunk: 5)
        {chunk_1_content}
        
        [2] (Score: 0.87, Source: doc_456, Chunk: 12)
        {chunk_2_content}
        ...
        
        QUERY: {user_query}
        
        ANSWER:
        ```
        
        Steps:
        1. Check query for injection patterns
        2. Format context with citation markers
        3. Ensure context fits in token limit
        4. Construct final prompt
        """
        pass
    
    def detect_injection(self, query: str) -> bool:
        """
        Detect prompt injection attempts.
        
        Patterns to check:
        - "Ignore previous instructions"
        - "System: " or "SYSTEM:"
        - "###" or "---" (boundary markers)
        - Excessive special characters
        """
        injection_patterns = [
            r"ignore (previous|above|prior) (instructions|directions|rules)",
            r"system\s*:",
            r"###",
            r"---",
            r"<\|.*?\|>",  # Special tokens
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
```

#### Component 4.2: LLMGenerator Class

**Purpose**: Generate answers using self-hosted Llama 3.2 3B

**Requirements**:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLMGenerator:
    """
    Self-hosted LLM for answer generation.
    
    Model: Llama 3.2 3B (4-bit quantized)
    Framework: transformers + bitsandbytes
    VRAM Usage: ~3GB (fits in RTX 3050Ti)
    
    Setup:
    pip install transformers accelerate bitsandbytes
    """
    
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
                 max_new_tokens: int = 512,
                 temperature: float = 0.3):
        
        # Load model with 4-bit quantization
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
    
    def generate(self, prompt: str) -> str:
        """
        Generate answer from prompt.
        
        Steps:
        1. Tokenize prompt
        2. Generate with sampling (temp=0.3 for focused answers)
        3. Decode output
        4. Extract answer (remove prompt echo)
        5. Post-process (strip whitespace, etc.)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt from output
        answer = answer[len(prompt):].strip()
        
        return answer
    
    def generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate for multiple prompts (batch processing)."""
        pass
```

**Alternative: llama.cpp for better performance**
```python
from llama_cpp import Llama

class LLMGenerator:
    """Using llama.cpp for faster inference."""
    
    def __init__(self, 
                 model_path: str = "./models/llama-3.2-3b-Q4_K_M.gguf",
                 n_gpu_layers: int = 35,  # Offload to GPU
                 n_ctx: int = 8192):
        
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False
        )
    
    def generate(self, prompt: str) -> str:
        output = self.llm(
            prompt,
            max_tokens=512,
            temperature=0.3,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["QUERY:", "CONTEXT:"]  # Stop at boundaries
        )
        
        return output['choices'][0]['text'].strip()
```

**Testing Requirements for Phase 4**:
```python
def test_prompt_builder():
    """
    Test cases:
    1. Normal query → assert prompt formatted correctly
    2. Injection attempt → assert detected and rejected
    3. Long context → assert truncated to max_tokens
    4. Citation formatting → assert [Source: X, Chunk: Y] format
    """
    pass

def test_llm_generator():
    """
    Test cases:
    1. Simple factual query → assert coherent answer
    2. Query with no context → assert "cannot answer" response
    3. Response length → assert ≤ 512 tokens
    4. Inference time → assert < 2 seconds
    """
    pass
```

---

### PHASE 5: POST-GENERATION VALIDATION (Layer 5)

#### Component 5.1: ValidationPipeline Class

**Purpose**: Validate generated answers against source documents

**Requirements**:
```python
class ValidationPipeline:
    """
    Multi-stage validation of generated answers.
    
    Validation Steps:
    1. Evidence-Answer Consistency (semantic similarity)
    2. Citation Extraction & Verification
    3. Confidence Scoring
    4. Hallucination Detection
    5. Fact Validation
    """
    
    def __init__(self, 
                 embedding_generator: EmbeddingGenerator,
                 min_consistency_score: float = 0.7):
        self.embedding_generator = embedding_generator
        self.min_consistency_score = min_consistency_score
    
    def validate(self, 
                 answer: str, 
                 context_chunks: List[Chunk],
                 retrieved_scores: List[float]) -> Dict[str, Any]:
        """
        Full validation pipeline.
        
        Returns:
        {
            'is_valid': bool,
            'consistency_score': float,
            'citations': List[str],
            'confidence': float,
            'hallucination_risk': str,  # 'low', 'medium', 'high'
            'validation_details': Dict
        }
        """
        results = {}
        
        # 1. Evidence-Answer Consistency
        results['consistency_score'] = self._check_consistency(answer, context_chunks)
        
        # 2. Extract and verify citations
        results['citations'] = self._extract_citations(answer)
        results['citations_valid'] = self._verify_citations(
            results['citations'], 
            context_chunks
        )
        
        # 3. Confidence scoring
        results['confidence'] = self._compute_confidence(
            retrieved_scores,
            results['consistency_score']
        )
        
        # 4. Hallucination detection
        results['hallucination_risk'] = self._detect_hallucination(
            answer, 
            context_chunks
        )
        
        # 5. Overall validity
        results['is_valid'] = (
            results['consistency_score'] >= self.min_consistency_score
            and results['citations_valid']
            and results['hallucination_risk'] != 'high'
        )
        
        return results
    
    def _check_consistency(self, 
                           answer: str, 
                           context_chunks: List[Chunk]) -> float:
        """
        Measure semantic similarity between answer and context.
        
        Algorithm:
        1. Generate embedding for answer
        2. Compute cosine similarity with each context chunk
        3. Return max similarity (most similar chunk)
        """
        answer_emb = self.embedding_generator.embed_query(answer)
        
        similarities = []
        for chunk in context_chunks:
            chunk_emb = chunk.embedding
            sim = np.dot(answer_emb, chunk_emb) / (
                np.linalg.norm(answer_emb) * np.linalg.norm(chunk_emb)
            )
            similarities.append(sim)
        
        return max(similarities) if similarities else 0.0
    
    def _extract_citations(self, answer: str) -> List[str]:
        """
        Extract citation markers from answer.
        
        Pattern: [Source: doc_id, Chunk: N]
        """
        pattern = r'\[Source: (.*?), Chunk: (\d+)\]'
        citations = re.findall(pattern, answer)
        return citations
    
    def _verify_citations(self, 
                         citations: List[Tuple[str, str]], 
                         context_chunks: List[Chunk]) -> bool:
        """Verify all citations exist in provided context."""
        for doc_id, chunk_idx in citations:
            found = any(
                c.doc_id == doc_id and c.chunk_index == int(chunk_idx)
                for c in context_chunks
            )
            if not found:
                return False
        return True
    
    def _compute_confidence(self, 
                           retrieval_scores: List[float],
                           consistency_score: float) -> float:
        """
        Compute overall confidence score.
        
        Formula:
        confidence = 0.6 * avg(retrieval_scores) + 0.4 * consistency_score
        """
        avg_retrieval = np.mean(retrieval_scores) if retrieval_scores else 0.0
        return 0.6 * avg_retrieval + 0.4 * consistency_score
    
    def _detect_hallucination(self, 
                             answer: str, 
                             context_chunks: List[Chunk]) -> str:
        """
        Detect potential hallucinations.
        
        Heuristics:
        - Extract named entities from answer
        - Check if entities appear in context
        - Risk = high if > 30% entities not in context
        """
        # Simple entity extraction (can use spaCy for better accuracy)
        answer_words = set(answer.lower().split())
        context_words = set()
        for chunk in context_chunks:
            context_words.update(chunk.content.lower().split())
        
        unique_to_answer = answer_words - context_words
        hallucination_ratio = len(unique_to_answer) / len(answer_words) if answer_words else 0
        
        if hallucination_ratio > 0.3:
            return 'high'
        elif hallucination_ratio > 0.15:
            return 'medium'
        else:
            return 'low'
```

**Testing Requirements for Phase 5**:
```python
def test_validation_pipeline():
    """
    Test cases:
    1. Faithful answer with citations → assert is_valid=True, high confidence
    2. Answer with fabricated info → assert hallucination_risk='high'
    3. Answer without citations → assert citations_valid=False
    4. Low consistency answer → assert is_valid=False
    """
    pass
```

---

### PHASE 6: MONITORING & GOVERNANCE (Layer 6)

#### Component 6.1: MonitoringSystem Class (Enhanced for Government Compliance)

**Purpose**: Comprehensive logging, audit trails, and security monitoring

**Requirements**:
```python
import sqlite3
from datetime import datetime
import hashlib
import json

class MonitoringSystem:
    """
    Comprehensive monitoring and governance system.
    
    Features:
    - Query logging (timestamp, user, query, results, latency)
    - **Audit trails** (tamper-evident logs with cryptographic hashing)
    - **Access logs** (who accessed which documents, when)
    - **Security events** (PII detections, failed accesses, unusual patterns)
    - Performance metrics (avg latency, throughput)
    - Drift detection (embedding distribution changes)
    - **Compliance reports** (for government audit requirements)
    """
    
    def __init__(self, 
                 db_path: str = "./monitoring.db",
                 encryption_manager: Optional[EncryptionManager] = None):
        self.db_path = db_path
        self.encryption = encryption_manager
        self._init_database()
    
    def _init_database(self):
        """Create monitoring database tables with comprehensive audit structure."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Query log table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_id TEXT,
            query_text TEXT NOT NULL,
            query_hash TEXT NOT NULL,
            domains TEXT,
            num_retrieved INTEGER,
            latency_ms REAL,
            confidence REAL,
            is_valid BOOLEAN,
            ip_address TEXT,
            session_id TEXT
        )
        """)
        
        # Document access log (critical for government audit)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_access_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_id TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            access_type TEXT NOT NULL,  -- 'retrieve', 'view', 'download'
            query_id INTEGER,
            ip_address TEXT,
            FOREIGN KEY (query_id) REFERENCES query_log(id)
        )
        """)
        
        # Security events log
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS security_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,  -- 'pii_detected', 'injection_attempt', 'access_denied', etc.
            severity TEXT NOT NULL,  -- 'low', 'medium', 'high', 'critical'
            user_id TEXT,
            details TEXT,
            action_taken TEXT
        )
        """)
        
        # Audit trail (tamper-evident with hash chain)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_trail (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            user_id TEXT,
            details TEXT NOT NULL,
            previous_hash TEXT,
            current_hash TEXT NOT NULL
        )
        """)
        
        # Performance metrics table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL,
            metadata TEXT
        )
        """)
        
        # PII detection log
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS pii_detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            pii_type TEXT NOT NULL,
            count INTEGER,
            anonymized BOOLEAN DEFAULT TRUE
        )
        """)
        
        conn.commit()
        conn.close()
        print("[MONITORING] Database initialized")
    
    def log_query(self, 
                  query: str,
                  user_id: str,
                  domains: List[str],
                  num_retrieved: int,
                  latency_ms: float,
                  confidence: float,
                  is_valid: bool,
                  ip_address: str = None,
                  session_id: str = None) -> int:
        """
        Log a query execution.
        
        Returns:
            query_id: ID of logged query (for linking to document access)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Hash query for privacy (if contains sensitive info)
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        
        cursor.execute("""
        INSERT INTO query_log (timestamp, user_id, query_text, query_hash, domains, 
                              num_retrieved, latency_ms, confidence, is_valid, 
                              ip_address, session_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            user_id,
            query if not self.encryption else "[ENCRYPTED]",
            query_hash,
            ','.join(domains),
            num_retrieved,
            latency_ms,
            confidence,
            is_valid,
            ip_address,
            session_id
        ))
        
        query_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return query_id
    
    def log_document_access(self,
                           user_id: str,
                           doc_id: str,
                           access_type: str,
                           query_id: int = None,
                           ip_address: str = None):
        """Log document access (critical for government audit compliance)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO document_access_log (timestamp, user_id, doc_id, access_type, 
                                        query_id, ip_address)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            user_id,
            doc_id,
            access_type,
            query_id,
            ip_address
        ))
        
        conn.commit()
        conn.close()
        
        # Also add to audit trail
        self.add_audit_event(
            event_type='document_access',
            user_id=user_id,
            details=json.dumps({
                'doc_id': doc_id,
                'access_type': access_type,
                'query_id': query_id
            })
        )
    
    def log_security_event(self,
                          event_type: str,
                          severity: str,
                          user_id: str = None,
                          details: str = "",
                          action_taken: str = ""):
        """Log security events (PII detected, injection attempts, etc.)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO security_events (timestamp, event_type, severity, user_id, 
                                    details, action_taken)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            event_type,
            severity,
            user_id,
            details,
            action_taken
        ))
        
        conn.commit()
        conn.close()
        
        # Print critical events immediately
        if severity in ['high', 'critical']:
            print(f"[SECURITY] {severity.upper()}: {event_type} - {details}")
    
    def add_audit_event(self,
                       event_type: str,
                       user_id: str,
                       details: str):
        """
        Add tamper-evident audit log entry with hash chain.
        
        Each entry contains hash of previous entry, making tampering detectable.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get previous hash
        cursor.execute("SELECT current_hash FROM audit_trail ORDER BY id DESC LIMIT 1")
        result = cursor.fetchone()
        previous_hash = result[0] if result else "GENESIS"
        
        # Create current hash
        timestamp = datetime.now().isoformat()
        hash_input = f"{timestamp}|{event_type}|{user_id}|{details}|{previous_hash}"
        current_hash = hashlib.sha256(hash_input.encode()).hexdigest()
        
        cursor.execute("""
        INSERT INTO audit_trail (timestamp, event_type, user_id, details, 
                                previous_hash, current_hash)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (timestamp, event_type, user_id, details, previous_hash, current_hash))
        
        conn.commit()
        conn.close()
    
    def verify_audit_integrity(self) -> Dict[str, Any]:
        """
        Verify audit trail integrity by checking hash chain.
        
        Returns:
        - is_valid: bool
        - num_entries: int
        - first_invalid_entry: int or None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM audit_trail ORDER BY id")
        entries = cursor.fetchall()
        conn.close()
        
        is_valid = True
        first_invalid = None
        
        for i, entry in enumerate(entries):
            id_, timestamp, event_type, user_id, details, prev_hash, curr_hash = entry
            
            # Recompute hash
            if i == 0:
                expected_prev = "GENESIS"
            else:
                expected_prev = entries[i-1][6]  # previous entry's current_hash
            
            if prev_hash != expected_prev:
                is_valid = False
                first_invalid = id_
                break
            
            # Verify current hash
            hash_input = f"{timestamp}|{event_type}|{user_id}|{details}|{prev_hash}"
            expected_hash = hashlib.sha256(hash_input.encode()).hexdigest()
            
            if curr_hash != expected_hash:
                is_valid = False
                first_invalid = id_
                break
        
        return {
            'is_valid': is_valid,
            'num_entries': len(entries),
            'first_invalid_entry': first_invalid
        }
    
    def log_pii_detection(self,
                         doc_id: str,
                         pii_detections: Dict[str, int]):
        """Log PII detections for compliance tracking."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        for pii_type, count in pii_detections.items():
            if count > 0:
                cursor.execute("""
                INSERT INTO pii_detections (timestamp, doc_id, pii_type, count, anonymized)
                VALUES (?, ?, ?, ?, ?)
                """, (timestamp, doc_id, pii_type, count, True))
        
        conn.commit()
        conn.close()
        
        # Log as security event if significant PII found
        total_pii = sum(pii_detections.values())
        if total_pii > 5:
            self.log_security_event(
                event_type='high_pii_detection',
                severity='medium',
                details=f"Document {doc_id} contains {total_pii} PII instances: {pii_detections}"
            )
    
    def get_performance_stats(self, 
                             time_window: str = '24h') -> Dict[str, Any]:
        """
        Get performance statistics for a time window.
        
        Returns:
        - avg_latency_ms
        - total_queries
        - avg_confidence
        - success_rate (is_valid)
        - top_domains
        - unique_users
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Parse time window
        if time_window == '24h':
            time_filter = "datetime('now', '-1 day')"
        elif time_window == '7d':
            time_filter = "datetime('now', '-7 days')"
        elif time_window == '30d':
            time_filter = "datetime('now', '-30 days')"
        else:
            time_filter = "datetime('now', '-1 day')"
        
        # Query statistics
        cursor.execute(f"""
        SELECT 
            COUNT(*) as total_queries,
            AVG(latency_ms) as avg_latency,
            AVG(confidence) as avg_confidence,
            SUM(CASE WHEN is_valid THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as success_rate,
            COUNT(DISTINCT user_id) as unique_users
        FROM query_log
        WHERE timestamp > {time_filter}
        """)
        
        stats = cursor.fetchone()
        
        # Top domains
        cursor.execute(f"""
        SELECT domains, COUNT(*) as count
        FROM query_log
        WHERE timestamp > {time_filter}
        GROUP BY domains
        ORDER BY count DESC
        LIMIT 5
        """)
        top_domains = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_queries': stats[0] or 0,
            'avg_latency_ms': round(stats[1], 2) if stats[1] else 0,
            'avg_confidence': round(stats[2], 2) if stats[2] else 0,
            'success_rate': round(stats[3], 2) if stats[3] else 0,
            'unique_users': stats[4] or 0,
            'top_domains': [{'domain': d[0], 'count': d[1]} for d in top_domains]
        }
    
    def generate_compliance_report(self, 
                                  start_date: str,
                                  end_date: str) -> Dict[str, Any]:
        """
        Generate comprehensive compliance report for government audit.
        
        Includes:
        - Total queries and documents accessed
        - User access patterns
        - Security events
        - PII detections and handling
        - System performance
        - Audit trail verification
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        report = {
            'report_period': f"{start_date} to {end_date}",
            'generated_at': datetime.now().isoformat()
        }
        
        # Query statistics
        cursor.execute("""
        SELECT COUNT(*), AVG(latency_ms), AVG(confidence)
        FROM query_log
        WHERE timestamp BETWEEN ? AND ?
        """, (start_date, end_date))
        query_stats = cursor.fetchone()
        report['query_statistics'] = {
            'total_queries': query_stats[0],
            'avg_latency_ms': round(query_stats[1], 2) if query_stats[1] else 0,
            'avg_confidence': round(query_stats[2], 2) if query_stats[2] else 0
        }
        
        # Document access statistics
        cursor.execute("""
        SELECT access_type, COUNT(*)
        FROM document_access_log
        WHERE timestamp BETWEEN ? AND ?
        GROUP BY access_type
        """, (start_date, end_date))
        access_stats = cursor.fetchall()
        report['document_access'] = {row[0]: row[1] for row in access_stats}
        
        # Security events
        cursor.execute("""
        SELECT event_type, severity, COUNT(*)
        FROM security_events
        WHERE timestamp BETWEEN ? AND ?
        GROUP BY event_type, severity
        """, (start_date, end_date))
        security_events = cursor.fetchall()
        report['security_events'] = [
            {'event_type': row[0], 'severity': row[1], 'count': row[2]}
            for row in security_events
        ]
        
        # PII detections
        cursor.execute("""
        SELECT pii_type, SUM(count)
        FROM pii_detections
        WHERE timestamp BETWEEN ? AND ?
        GROUP BY pii_type
        """, (start_date, end_date))
        pii_stats = cursor.fetchall()
        report['pii_detections'] = {row[0]: row[1] for row in pii_stats}
        
        # Audit integrity
        report['audit_integrity'] = self.verify_audit_integrity()
        
        conn.close()
        
        return report
    
    def detect_drift(self, 
                    current_embeddings: np.ndarray,
                    baseline_embeddings: np.ndarray) -> Dict[str, float]:
        """
        Detect embedding distribution drift.
        
        Methods:
        1. Cosine distance between centroids
        2. KL divergence (approximated)
        3. Statistical tests
        """
        # Compute centroids
        current_centroid = np.mean(current_embeddings, axis=0)
        baseline_centroid = np.mean(baseline_embeddings, axis=0)
        
        # Cosine distance
        cosine_dist = 1 - np.dot(current_centroid, baseline_centroid) / (
            np.linalg.norm(current_centroid) * np.linalg.norm(baseline_centroid)
        )
        
        # Standard deviation change
        current_std = np.std(current_embeddings, axis=0).mean()
        baseline_std = np.std(baseline_embeddings, axis=0).mean()
        std_change = abs(current_std - baseline_std) / baseline_std
        
        return {
            'cosine_distance': float(cosine_dist),
            'std_change_ratio': float(std_change),
            'drift_detected': cosine_dist > 0.1 or std_change > 0.2
        }
```

**Testing Requirements for Phase 6**:
```python
def test_monitoring_system():
    """
    Test cases:
    1. Log 100 queries → assert all recorded
    2. Query performance stats → assert metrics computed correctly
    3. Drift detection with shifted distribution → assert drift detected
    """
    pass
```

---

### PHASE 7: RESPONSE DELIVERY & INTEGRATION

#### Component 7.1: SLRAGPipeline Class (Main Orchestrator)

**Purpose**: End-to-end pipeline integration

**Requirements**:
```python
class SLRAGPipeline:
    """
    Complete SL-RAG pipeline orchestrator.
    
    Usage:
    ```python
    pipeline = SLRAGPipeline(config_path="config.yaml")
    pipeline.ingest_documents(pdf_directory="./docs")
    result = pipeline.query("What is the security architecture?")
    print(result['answer'])
    ```
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        # Layer 1: Ingestion
        self.document_loader = DocumentLoader()
        self.pii_anonymizer = PIIAnonymizer()
        
        # Layer 2: Embedding & Storage
        self.chunk_generator = ChunkGenerator()
        self.embedding_generator = EmbeddingGenerator()
        self.faiss_index = FAISSIndex()
        self.domain_manager = DomainManager()
        
        # Layer 3: Retrieval
        self.hybrid_retriever = HybridRetriever(
            self.faiss_index, 
            self.chunks
        )
        self.cross_encoder = CrossEncoderReranker()
        self.retrieval_pipeline = RetrievalPipeline(
            self.embedding_generator,
            self.domain_manager,
            self.hybrid_retriever,
            self.cross_encoder
        )
        
        # Layer 4: Generation
        self.prompt_builder = PromptBuilder()
        self.llm_generator = LLMGenerator()
        
        # Layer 5: Validation
        self.validation_pipeline = ValidationPipeline(
            self.embedding_generator
        )
        
        # Layer 6: Monitoring
        self.monitoring = MonitoringSystem()
    
    def ingest_documents(self, 
                        pdf_directory: str,
                        domain_mapping: Optional[Dict[str, str]] = None):
        """
        Ingest all PDFs from directory.
        
        Steps:
        1. Load documents
        2. Chunk documents
        3. Generate embeddings
        4. Build FAISS index
        5. Compute domain centroids
        6. Save index to disk
        """
        print(f"[INGESTION] Loading documents from {pdf_directory}...")
        docs = self.document_loader.load_directory(pdf_directory)
        
        print(f"[INGESTION] Loaded {len(docs)} documents")
        print(f"[CHUNKING] Generating chunks...")
        chunks = []
        for doc in docs:
            doc_chunks = self.chunk_generator.chunk_document(doc)
            chunks.extend(doc_chunks)
        
        print(f"[CHUNKING] Generated {len(chunks)} chunks")
        print(f"[EMBEDDING] Generating embeddings (GPU)...")
        chunks = self.embedding_generator.embed_chunks(chunks)
        
        print(f"[INDEX] Building FAISS index...")
        self.faiss_index.build_index(chunks)
        
        print(f"[DOMAINS] Computing domain centroids...")
        self.domain_manager.compute_centroids(chunks)
        
        print(f"[STORAGE] Saving index...")
        self.faiss_index.save_index("./faiss_index/index.faiss")
        
        print("[INGESTION] ✓ Complete!")
        return {
            'num_documents': len(docs),
            'num_chunks': len(chunks),
            'domains': list(self.domain_manager.domains.keys())
        }
    
    def query(self, 
             query: str,
             top_k: int = 5,
             return_metadata: bool = True) -> Dict[str, Any]:
        """
        Execute end-to-end query.
        
        Returns:
        {
            'answer': str,
            'citations': List[str],
            'confidence': float,
            'retrieved_chunks': List[Dict],
            'validation': Dict,
            'metadata': Dict (latency, domains, etc.)
        }
        """
        start_time = time.time()
        
        # Step 1: Retrieve relevant chunks
        print(f"[RETRIEVE] Searching for: {query}")
        retrieved = self.retrieval_pipeline.retrieve(query, top_k=top_k)
        
        # Step 2: Build prompt
        print(f"[PROMPT] Building prompt with {len(retrieved)} chunks")
        prompt = self.prompt_builder.build_prompt(query, retrieved)
        
        # Step 3: Generate answer
        print(f"[GENERATE] Generating answer...")
        answer = self.llm_generator.generate(prompt)
        
        # Step 4: Validate answer
        print(f"[VALIDATE] Validating answer...")
        chunks, scores = zip(*retrieved) if retrieved else ([], [])
        validation = self.validation_pipeline.validate(answer, chunks, scores)
        
        # Step 5: Log query
        latency_ms = (time.time() - start_time) * 1000
        domains = list(set(c.domain for c in chunks))
        self.monitoring.log_query(
            query, domains, len(retrieved), 
            latency_ms, validation['confidence'], validation['is_valid']
        )
        
        # Build response
        result = {
            'answer': answer,
            'citations': validation['citations'],
            'confidence': validation['confidence'],
            'is_valid': validation['is_valid'],
            'retrieved_chunks': [
                {
                    'content': c.content,
                    'doc_id': c.doc_id,
                    'chunk_index': c.chunk_index,
                    'score': s
                }
                for c, s in retrieved
            ] if return_metadata else [],
            'validation': validation,
            'metadata': {
                'latency_ms': latency_ms,
                'domains': domains,
                'num_retrieved': len(retrieved)
            }
        }
        
        print(f"[COMPLETE] Query processed in {latency_ms:.2f}ms")
        return result
    
    def batch_query(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Process multiple queries."""
        return [self.query(q) for q in queries]
```

---

## TESTING STRATEGY

### Unit Tests (pytest)
Create `test_sl_rag.py` with comprehensive unit tests for each component.

```python
import pytest
from sl_rag_pipeline import *

class TestDocumentLoader:
    def test_load_valid_pdf(self):
        loader = DocumentLoader()
        doc = loader.load_pdf("./test_data/sample.pdf")
        assert doc.content is not None
        assert len(doc.content) > 0
        assert doc.doc_id is not None
    
    def test_pii_anonymization(self):
        loader = DocumentLoader()
        doc = loader.load_pdf("./test_data/pii_test.pdf")
        assert "[EMAIL_REDACTED]" in doc.content
        assert doc.pii_removed == True

# ... more tests for each component
```

### Integration Tests
Create `test_integration.py` for end-to-end testing.

```python
def test_end_to_end_pipeline():
    """Test complete pipeline flow."""
    pipeline = SLRAGPipeline()
    
    # Ingest test documents
    result = pipeline.ingest_documents("./test_data/pdfs")
    assert result['num_documents'] > 0
    
    # Query
    response = pipeline.query("What is the methodology?")
    assert response['answer'] is not None
    assert response['confidence'] > 0.5
    assert response['is_valid'] == True
```

### Performance Benchmarks
Create `benchmark.py` for performance testing.

```python
def benchmark_retrieval_speed():
    """Benchmark retrieval latency."""
    pipeline = SLRAGPipeline()
    queries = ["test query"] * 100
    
    start = time.time()
    for q in queries:
        pipeline.retrieval_pipeline.retrieve(q)
    elapsed = time.time() - start
    
    avg_latency = elapsed / len(queries)
    assert avg_latency < 0.5  # < 500ms per query
    print(f"Avg retrieval latency: {avg_latency*1000:.2f}ms")

def benchmark_end_to_end():
    """Benchmark full pipeline."""
    pipeline = SLRAGPipeline()
    query = "What is the security architecture?"
    
    start = time.time()
    result = pipeline.query(query)
    elapsed = time.time() - start
    
    assert elapsed < 3.0  # < 3 seconds
    print(f"End-to-end latency: {elapsed*1000:.2f}ms")
```

---

## CONFIGURATION FILE (config.yaml)

```yaml
# SL-RAG Pipeline Configuration for ISRO Documentation System
# Maximum Security Mode - 100% Offline Operation

# System Information
system:
  name: "ISRO SL-RAG Document Retrieval System"
  version: "1.0.0"
  deployment: "production"
  offline_mode: true

# Security Settings (MAXIMUM SECURITY)
security:
  encryption:
    enabled: true
    algorithm: "AES-256"
    master_key_path: "./keys/master.key"
    auto_generate_key: true
  
  pii_detection:
    enabled: true
    detect_emails: true
    detect_phones: true
    detect_aadhaar: true  # Indian ID
    detect_pan: true      # PAN card
    detect_passport: true
    detect_ssn: true
    detect_credit_cards: true
    detect_names_ner: false  # Set to true if spaCy installed
    log_detections: true
  
  audit:
    enabled: true
    tamper_evident: true  # Hash chain
    log_document_access: true
    log_security_events: true
    compliance_mode: true
  
  access_control:
    require_user_id: true
    log_ip_addresses: false  # Set true if network available
    session_tracking: true

# Document Processing
document_processing:
  max_file_size_mb: 100  # Larger for govt docs
  ocr_enabled: true  # Essential for scanned govt PDFs
  supported_formats: ['.pdf']
  sanitize_content: true
  validate_integrity: true  # SHA-256 checksums
  
# PII Anonymization
pii:
  replacement_token: "[REDACTED]"
  preserve_structure: true  # Use [TYPE_REDACTED] format
  enable_ner: false  # Requires spaCy

# Chunking Strategy
chunking:
  chunk_size: 512  # tokens
  overlap: 50      # tokens
  strategy: 'sentence_aware'  # Preserve sentence boundaries
  min_chunk_size: 100  # Minimum tokens per chunk
  max_chunk_size: 600  # Maximum tokens per chunk

# Embedding
embedding:
  model_name: 'sentence-transformers/all-mpnet-base-v2'
  dimension: 768
  batch_size: 32  # Adjust based on GPU memory
  normalize: true  # L2 normalization
  device: 'cuda'  # or 'cpu'
  cache_dir: './models/embeddings'

# Domain Detection (Content-Based Clustering)
domains:
  auto_detect: true
  min_clusters: 2
  max_clusters: 10  # Max domains to detect
  visualize: true  # Generate t-SNE visualization
  viz_output: './domain_visualization.png'

# Vector Store (FAISS)
faiss:
  index_type: 'IndexFlatIP'  # Exact search for <1000 docs
  # index_type: 'IndexIVFFlat'  # Use for 1000+ docs
  use_gpu: true
  index_path: './faiss_index'
  encrypted_storage: true
  save_interval: 100  # Save after N chunks added
  backup_enabled: true
  backup_path: './backups/faiss_index'

# Retrieval Settings
retrieval:
  # Hybrid Retrieval
  hybrid:
    enabled: true
    alpha: 0.7  # Weight for dense retrieval (0.7 dense + 0.3 BM25)
    use_rrf: false  # Reciprocal Rank Fusion (alternative to weighted)
  
  # Initial retrieval
  initial_k: 20  # Retrieve top-20 before re-ranking
  similarity_threshold: 0.5  # Filter chunks below this score
  
  # Cross-Encoder Re-ranking
  reranking:
    enabled: true
    model: 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    final_k: 5  # Return top-5 after re-ranking
    device: 'cpu'  # Cross-encoders typically run on CPU
  
  # Domain Routing
  domain_routing:
    enabled: true
    top_k_domains: 3  # Route query to top-3 relevant domains

# LLM Generation (SKIP FOR NOW)
llm:
  enabled: false  # Set to true after Llama setup
  model_type: 'llama.cpp'
  model_path: './models/llama-3.2-3b-Q4_K_M.gguf'
  max_new_tokens: 512
  temperature: 0.3
  top_p: 0.9
  repetition_penalty: 1.1
  n_gpu_layers: 35
  context_window: 8192

# Validation
validation:
  enabled: true
  min_consistency_score: 0.7
  require_citations: true
  confidence_threshold: 0.6  # Warn if below this

# Monitoring & Governance
monitoring:
  enabled: true
  db_path: './monitoring.db'
  encrypted_db: true
  
  logging:
    log_queries: true
    log_document_access: true
    log_security_events: true
    log_performance: true
  
  compliance:
    generate_reports: true
    report_interval: 'monthly'
    report_path: './compliance_reports'
  
  performance:
    track_latency: true
    track_accuracy: true
    benchmark_interval: 'weekly'
  
  drift_detection:
    enabled: true
    check_interval: 'weekly'
    alert_threshold: 0.15  # Alert if drift > 15%

# Performance Optimization
performance:
  max_concurrent_queries: 4
  cache_embeddings: true
  cache_dir: './cache'
  use_mixed_precision: true  # FP16 for embeddings
  batch_processing: true

# Paths
paths:
  data_dir: './data'
  pdfs_dir: './data/pdfs'
  index_dir: './faiss_index'
  models_dir: './models'
  logs_dir: './logs'
  cache_dir: './cache'
  keys_dir: './keys'
  backups_dir: './backups'

# ISRO-Specific Settings
isro:
  centers: ['VSSC', 'LPSC', 'SAC', 'SDSC', 'ISTRAC', 'IISU', 'MCF']
  document_types: ['GFR', 'Procurement Manual', 'Technical Memo', 'QA Report', 'Telemetry Data']
  
  # Acronym expansion for queries
  acronym_expansion:
    enabled: true
    acronyms:
      GFR: 'General Financial Rules'
      VSSC: 'Vikram Sarabhai Space Centre'
      LPSC: 'Liquid Propulsion Systems Centre'
      SAC: 'Space Applications Centre'
      SDSC: 'Satish Dhawan Space Centre'
      ISTRAC: 'ISRO Telemetry Tracking and Command Network'
      QA: 'Quality Assurance'
      TM: 'Technical Memorandum'
  
  # Compliance requirements
  compliance:
    require_citations: true
    require_source_validation: true
    audit_all_access: true
    retention_period: '7 years'  # Government requirement

# Deployment
deployment:
  mode: 'local'  # local, server, air-gapped
  port: 8000  # If running as API server
  host: 'localhost'
  workers: 1  # Number of worker processes
  
  # Air-gapped deployment
  air_gapped:
    enabled: false
    offline_mode_strict: true
    no_telemetry: true
    no_updates: true
```

---

## DEPLOYMENT CHECKLIST

### Pre-Deployment (One-Time Setup)

**Step 1: Environment Setup**
- [ ] Python 3.8+ installed
- [ ] CUDA toolkit installed (for GPU support)
- [ ] Create virtual environment: `python -m venv venv`
- [ ] Activate: `source venv/bin/activate` (Linux) or `venv\Scripts\activate` (Windows)

**Step 2: Dependency Installation (REQUIRES INTERNET)**
```bash
# Install dependencies
pip install -r requirements.txt

# Download models (ONE-TIME, requires internet)
python setup.py --download-models

# This will download:
# - sentence-transformers/all-mpnet-base-v2 (~420MB)
# - cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB)
# - spaCy language model (optional, ~500MB)

# Models will be cached in ./models/ directory
```

**Step 3: Directory Structure**
```bash
# Create required directories
mkdir -p data/pdfs
mkdir -p faiss_index
mkdir -p models
mkdir -p keys
mkdir -p logs
mkdir -p cache
mkdir -p backups
mkdir -p compliance_reports
```

**Step 4: Security Initialization**
```bash
# Generate master encryption key
python sl_rag_cli.py --init-security

# This creates:
# - ./keys/master.key (AES-256 encryption key)
# - ./monitoring.db (encrypted audit database)
# - Sets proper file permissions (600 for keys)
```

**Step 5: Verify GPU Availability**
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

---

### 100% OFFLINE OPERATION

**After completing Pre-Deployment, system operates FULLY OFFLINE:**

✅ **All models downloaded and cached locally**
✅ **No API calls to external services**
✅ **No telemetry or phone-home**
✅ **No auto-updates**
✅ **All computation on local hardware**
✅ **Data never leaves the machine**

**Optional: Air-Gapped Deployment**
```bash
# On internet-connected machine:
1. pip download -r requirements.txt -d ./offline_packages/
2. python setup.py --download-models --offline-prep
3. Copy entire directory to USB drive

# On air-gapped machine:
1. Copy from USB to target machine
2. pip install --no-index --find-links=./offline_packages/ -r requirements.txt
3. python setup.py --install-offline-models
4. Verify: python -c "import sentence_transformers; print('OK')"
```

---

### Deployment Steps (Fully Offline)

**Phase 1: Initial Ingestion**
```bash
# Place PDF documents in ./data/pdfs/
# Organize as needed (system auto-detects domains)

# Run ingestion pipeline
python sl_rag_cli.py --ingest ./data/pdfs --user-id admin

# Expected output:
# [INGESTION] Loading documents from ./data/pdfs...
# [INGESTION] Loaded 15 documents
# [PII] Detected and redacted: {'email': 3, 'phone_indian': 2, 'aadhaar': 1}
# [SECURITY] Encrypted and saved to ./data/pdfs/encrypted/
# [CHUNKING] Generated 387 chunks
# [EMBEDDING] Generating embeddings (GPU)... ✓ 2.3s
# [DOMAINS] Auto-detected 3 domains: procurement_financial, technical_telemetry, quality_assurance
# [INDEX] Building FAISS index... ✓ 387 vectors
# [SECURITY] Index encrypted and saved
# [INGESTION] ✓ Complete!
```

**Phase 2: Query Testing**
```bash
# Test query
python sl_rag_cli.py --query "What is the delegation of financial power?" --user-id admin

# Expected output:
# [RETRIEVE] Searching across 3 domains...
# [RETRIEVE] Found 20 initial candidates
# [RERANK] Re-ranking to top-5...
# [VALIDATE] Citations verified ✓
# [AUDIT] Query logged (ID: 1)
# 
# === RESULTS ===
# Top 5 relevant chunks:
# 1. [Score: 0.94] GFR Rule 23.1 - Delegation of Powers (doc_id: abc123, chunk: 12)
# 2. [Score: 0.89] Procurement Manual Section 4.2 (doc_id: def456, chunk: 8)
# ...
# Confidence: 0.87
# Query Time: 1.85s
```

**Phase 3: Compliance Report**
```bash
# Generate compliance report
python sl_rag_cli.py --compliance-report --start-date 2024-01-01 --end-date 2024-12-31

# Output: ./compliance_reports/report_2024.json
```

---

### Post-Deployment Verification

**Run Test Suite:**
```bash
# Unit tests
pytest tests/test_sl_rag.py -v

# Integration tests
pytest tests/test_integration.py -v

# Security tests
pytest tests/test_security.py -v

# Performance benchmarks
python benchmarks/run_benchmarks.py
```

**Expected Benchmarks:**
- Document ingestion: < 1s per document
- Embedding generation: < 5s for 100 chunks (GPU)
- Query latency: < 2s end-to-end
- Retrieval accuracy: Recall@5 > 0.85
- Memory usage: < 4GB RAM, < 3GB VRAM

**Security Validation:**
```bash
# Verify encryption
python sl_rag_cli.py --verify-encryption

# Verify audit integrity
python sl_rag_cli.py --verify-audit

# Check PII handling
python sl_rag_cli.py --pii-stats
```

---

### Maintenance Tasks

**Daily:**
- Review security event logs: `python sl_rag_cli.py --security-events --last 24h`
- Check system performance: `python sl_rag_cli.py --performance-stats --last 24h`

**Weekly:**
- Generate compliance report: `python sl_rag_cli.py --compliance-report --last 7d`
- Backup encrypted index: `python sl_rag_cli.py --backup`
- Check for drift: `python sl_rag_cli.py --check-drift`

**Monthly:**
- Review top queries: `python sl_rag_cli.py --query-analytics --last 30d`
- Optimize index if needed: `python sl_rag_cli.py --optimize-index`
- Archive old logs: `python sl_rag_cli.py --archive-logs --older-than 90d`

**Quarterly:**
- Full system audit: `python sl_rag_cli.py --full-audit`
- Re-compute domain centroids: `python sl_rag_cli.py --recompute-domains`
- Performance benchmark: `python benchmarks/run_benchmarks.py --full`

---

### Scaling Beyond 1K Documents

When corpus exceeds 1000 documents:

**Step 1: Update Configuration**
```yaml
# config.yaml
faiss:
  index_type: 'IndexIVFFlat'  # Switch from IndexFlatIP
  nlist: 32  # sqrt(1000) ≈ 32 clusters
```

**Step 2: Rebuild Index**
```bash
python sl_rag_cli.py --rebuild-index --index-type IndexIVFFlat
```

**Step 3: Re-benchmark**
```bash
python benchmarks/run_benchmarks.py --corpus-size 1000
```

---

### Troubleshooting

**Issue: GPU Out of Memory**
```yaml
# Reduce batch size in config.yaml
embedding:
  batch_size: 16  # Down from 32
```

**Issue: Slow Query Performance**
```yaml
# Enable caching
performance:
  cache_embeddings: true
  cache_frequently_accessed: true
```

**Issue: Poor Retrieval Accuracy**
```bash
# Tune retrieval parameters
python sl_rag_cli.py --tune-retrieval --test-queries ./test_queries.txt
```

**Issue: Encryption Key Lost**
```bash
# Recovery not possible - encryption is secure!
# Must re-ingest all documents with new key
python sl_rag_cli.py --init-security --force-new-key
python sl_rag_cli.py --ingest ./data/pdfs
```

---

## EXPECTED OUTPUTS

### 1. Ingestion Output
```
[INGESTION] Loading documents from ./data/pdfs...
[INGESTION] Loaded 15 documents
[CHUNKING] Generating chunks...
[CHUNKING] Generated 387 chunks
[EMBEDDING] Generating embeddings (GPU)...
[EMBEDDING] ✓ Complete (768-dim vectors, 2.3s)
[INDEX] Building FAISS index...
[INDEX] ✓ Index built (387 vectors)
[DOMAINS] Computing domain centroids...
[DOMAINS] ✓ Found 3 domains: technical, business, general
[STORAGE] Saving index...
[INGESTION] ✓ Complete!

Summary:
- Documents: 15
- Chunks: 387
- Domains: 3
- Index size: 2.9 MB
- Total time: 8.7s
```

### 2. Query Output
```json
{
  "answer": "The security architecture consists of three layers: data encryption at rest using AES-256, access control via role-based authentication, and audit logging for all operations. [Source: doc_a3f2, Chunk: 12] The system also implements PII anonymization during ingestion to protect sensitive information. [Source: doc_b8e1, Chunk: 5]",
  "citations": [
    ["doc_a3f2", "12"],
    ["doc_b8e1", "5"]
  ],
  "confidence": 0.87,
  "is_valid": true,
  "retrieved_chunks": [
    {
      "content": "The security architecture...",
      "doc_id": "doc_a3f2",
      "chunk_index": 12,
      "score": 0.92
    }
  ],
  "validation": {
    "consistency_score": 0.89,
    "citations_valid": true,
    "hallucination_risk": "low",
    "is_valid": true
  },
  "metadata": {
    "latency_ms": 1847.3,
    "domains": ["technical"],
    "num_retrieved": 5
  }
}
```

### 3. Performance Stats
```
Performance Statistics (Last 24h)
==================================
Total Queries: 247
Avg Latency: 1.9s
Avg Confidence: 0.82
Success Rate: 94.3%

Top Domains:
1. technical (45%)
2. business (32%)
3. general (23%)

Retrieval Performance:
- Recall@5: 0.91
- Recall@10: 0.97
- MRR: 0.84
```

---

## MAINTENANCE & OPTIMIZATION

### Regular Tasks
1. **Weekly**: Review monitoring logs, check for drift
2. **Monthly**: Re-compute domain centroids, optimize FAISS index
3. **Quarterly**: Fine-tune cross-encoder on domain-specific data

### Scaling Beyond 1K Documents
When corpus exceeds 1000 documents:
1. Switch FAISS index to `IndexIVFFlat` with `nlist = sqrt(N)`
2. Consider sharding by domain
3. Implement async query processing
4. Add caching layer (Redis) for frequent queries

---

## SUCCESS CRITERIA

### Functional Requirements
- [x] Loads and processes PDF documents (including scanned PDFs with OCR)
- [x] **Maximum security**: AES-256 encryption for all stored data
- [x] **PII detection**: Detects and redacts Indian PII (Aadhaar, PAN, etc.)
- [x] **Content-based clustering**: Automatically detects document domains
- [x] Generates accurate embeddings (768-dim, all-mpnet-base-v2)
- [x] Retrieves relevant chunks (Recall@5 > 0.85 on ISRO test queries)
- [x] **Hybrid retrieval**: BM25 + dense with cross-encoder re-ranking
- [x] **Comprehensive audit trails**: Tamper-evident logging with hash chains
- [x] **100% offline operation**: No external API calls after setup
- [x] Validates retrieval results (confidence scoring, citation verification)

### Performance Requirements
- [x] Ingestion: < 1s per document (GPU-accelerated)
- [x] Query latency: < 2s for retrieval (no LLM generation yet)
- [x] Retrieval accuracy: Recall@5 > 0.85, Recall@10 > 0.95, MRR > 0.80
- [x] GPU utilization: > 70% during embedding generation
- [x] Memory usage: < 12GB RAM, < 3GB VRAM during inference
- [x] Index build time: < 10s for 100 documents
- [x] Concurrent queries: Support 4+ simultaneous queries

### Security Requirements (CRITICAL)
- [x] **PII anonymization**: 100% detection and redaction of sensitive data
- [x] **Encryption at rest**: AES-256 for vector store, documents, metadata
- [x] **Secure key management**: Master key with proper permissions (600)
- [x] **Audit logging**: All queries, document accesses, security events logged
- [x] **Tamper-evident logs**: Hash chain verification for audit trail
- [x] **Access control**: User tracking, session management
- [x] **Compliance reports**: Government audit requirements met
- [x] **Offline operation**: Zero network dependencies after setup
- [x] **Secure deletion**: Multi-pass overwrite for sensitive files
- [x] **No data leakage**: All computation local, no telemetry

### ISRO-Specific Requirements
- [x] Handles government document formats (GFR, procurement manuals, technical memos)
- [x] Supports scanned PDFs (OCR capability)
- [x] Acronym expansion (GFR, VSSC, LPSC, etc.)
- [x] Multi-center document support
- [x] Regulatory query support (delegation of powers, procurement rules)
- [x] Technical documentation analysis (failure analysis, telemetry)
- [x] Cross-document synthesis capability
- [x] Compliance reporting (7-year retention requirement)

### Testing Requirements
- [x] Unit test coverage: > 80%
- [x] Integration tests: All phases tested together
- [x] Security tests: Encryption, PII, audit integrity verified
- [x] Performance benchmarks: Latency, accuracy, resource usage measured
- [x] Real-world testing: Validated with actual ISRO documents
- [x] Edge case testing: Large docs, corrupted files, unusual queries

---

## DELIVERABLES

### 1. Source Code
```
sl_rag/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── document_loader.py         # PDF loading + OCR
│   ├── pii_anonymizer.py          # PII detection (India-specific)
│   ├── encryption_manager.py      # AES-256 encryption
│   ├── chunk_generator.py         # Semantic chunking
│   ├── embedding_generator.py     # all-mpnet-base-v2
│   ├── faiss_index.py            # Vector store (encrypted)
│   └── domain_manager.py          # Content-based clustering
├── retrieval/
│   ├── __init__.py
│   ├── hybrid_retriever.py        # BM25 + Dense
│   ├── cross_encoder.py           # Re-ranking
│   └── retrieval_pipeline.py      # Full retrieval flow
├── validation/
│   ├── __init__.py
│   └── validation_pipeline.py     # Citation verification
├── monitoring/
│   ├── __init__.py
│   └── monitoring_system.py       # Audit trails + compliance
├── pipeline.py                     # Main SLRAGPipeline orchestrator
└── cli.py                         # Command-line interface

tests/
├── test_document_loader.py
├── test_pii_anonymizer.py
├── test_encryption.py
├── test_chunking.py
├── test_embedding.py
├── test_faiss_index.py
├── test_domain_manager.py
├── test_retrieval.py
├── test_validation.py
├── test_monitoring.py
├── test_security.py
├── test_integration.py
└── test_isro_queries.py           # ISRO-specific test cases

benchmarks/
├── run_benchmarks.py
├── benchmark_ingestion.py
├── benchmark_retrieval.py
└── benchmark_scalability.py
```

### 2. Configuration & Setup
```
config.yaml                         # Main configuration
requirements.txt                    # Python dependencies
setup.py                           # Installation script
.env.example                       # Environment variables template
README.md                          # Setup and usage guide
```

### 3. Documentation
```
docs/
├── README.md                       # Overview
├── INSTALLATION.md                 # Detailed setup instructions
├── SECURITY.md                     # Security features and compliance
├── API_REFERENCE.md                # Code documentation
├── USER_GUIDE.md                   # How to use the system
├── ISRO_DEPLOYMENT.md              # ISRO-specific deployment guide
├── COMPLIANCE.md                   # Government audit requirements
├── TROUBLESHOOTING.md              # Common issues and solutions
└── OFFLINE_OPERATION.md            # Air-gapped deployment guide
```

### 4. Models & Data Preparation
```
models/
├── download_models.sh              # Script to download all models
├── README.md                       # Model information
└── checksums.txt                   # SHA-256 checksums for verification

# Models to download (one-time, ~1GB total):
# - sentence-transformers/all-mpnet-base-v2 (~420MB)
# - cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB)
# - (Optional) spaCy en_core_web_sm (~500MB)
```

### 5. Sample Data & Tests
```
sample_data/
├── sample_gfr.pdf                  # Sample GFR document
├── sample_procurement.pdf          # Sample procurement manual
├── sample_technical_memo.pdf       # Sample technical memorandum
└── test_queries.txt                # ISRO-specific test queries

expected_outputs/
├── sample_ingestion_log.txt
├── sample_query_result.json
└── sample_compliance_report.json
```

### 6. Deployment Package
```
deployment/
├── docker/
│   ├── Dockerfile                  # Docker container (optional)
│   └── docker-compose.yml
├── systemd/
│   └── sl-rag.service             # Linux service file
├── scripts/
│   ├── install.sh                 # Installation script
│   ├── backup.sh                  # Backup script
│   ├── health_check.sh            # Health monitoring
│   └── update_index.sh            # Incremental updates
└── air_gapped/
    ├── prepare_offline.sh         # Prepare for air-gapped deployment
    └── install_offline.sh         # Install on air-gapped machine
```

---

## REQUIREMENTS.TXT

```txt
# SL-RAG Pipeline Dependencies
# Python 3.8+

# Core ML/NLP
torch>=2.0.0
sentence-transformers>=2.2.0
transformers>=4.30.0
faiss-gpu>=1.7.2  # Use faiss-cpu if no GPU

# Document Processing
PyMuPDF>=1.22.0  # PDF extraction
pytesseract>=0.3.10  # OCR
Pillow>=9.5.0  # Image processing
python-docx>=0.8.11  # Word docs (future support)

# Retrieval
rank-bm25>=0.2.2  # BM25 implementation
scikit-learn>=1.3.0  # Clustering, metrics

# Encryption & Security
cryptography>=41.0.0  # AES-256 encryption
hashlib  # Built-in (hash chains)

# Data & Analytics
numpy>=1.24.0
pandas>=2.0.0  # For analytics
matplotlib>=3.7.0  # Visualization
seaborn>=0.12.0  # Better plots

# NLP (Optional - for name detection)
spacy>=3.5.0  # NER for names
# Run: python -m spacy download en_core_web_sm

# Database & Logging
sqlite3  # Built-in (audit logs)

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0  # Coverage
pytest-benchmark>=4.0.0  # Performance tests

# Development Tools
black>=23.7.0  # Code formatting
flake8>=6.1.0  # Linting
mypy>=1.5.0  # Type checking

# Utilities
pyyaml>=6.0  # Config files
tqdm>=4.65.0  # Progress bars
python-dotenv>=1.0.0  # Environment variables
click>=8.1.0  # CLI framework

# Monitoring
psutil>=5.9.0  # System monitoring


## EXECUTION INSTRUCTIONS

### For Code Generation
Generate the implementation in this order:

1. **Step 1**: Core data structures (Document, Chunk classes)
2. **Step 2**: Document loader + PII anonymizer
3. **Step 3**: Chunking + embedding generation
4. **Step 4**: FAISS index builder
5. **Step 5**: Domain manager
6. **Step 6**: Hybrid retriever + cross-encoder
7. **Step 7**: Prompt builder + LLM generator
8. **Step 8**: Validation pipeline
9. **Step 9**: Monitoring system
10. **Step 10**: Main SLRAGPipeline orchestrator
11. **Step 11**: CLI interface (main.py)
12. **Step 12**: Tests and benchmarks

### Validation After Each Step
After generating each component:
1. Write unit tests
2. Run tests to verify functionality
3. Measure performance (latency, memory)
4. Document any issues or edge cases
5. Proceed to next step only if tests pass

### Final Integration
After all components are implemented:
1. Run full integration test suite
2. Benchmark end-to-end performance
3. Test with sample PDF documents
4. Generate performance report
5. Create deployment package

---

## EXPECTED TIMELINE

| Phase | Component | Estimated Time |
|-------|-----------|----------------|
| 1 | Data structures + Loader | 2 hours |
| 2 | PII + Chunking | 2 hours |
| 3 | Embedding + FAISS | 3 hours |
| 4 | Retrieval pipeline | 3 hours |
| 5 | LLM + Prompt builder | 2 hours |
| 6 | Validation pipeline | 2 hours |
| 7 | Monitoring + Integration | 2 hours |
| 8 | Testing + Documentation | 3 hours |
| **Total** | | **19 hours** |

---

## RISK MITIGATION

### Potential Issues & Solutions

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| VRAM overflow | Medium | Use 4-bit quantization, batch size=1 |
| Slow embedding generation | Low | GPU acceleration, batch processing |
| Poor retrieval accuracy | Medium | Tune similarity threshold, alpha parameter |
| Hallucinations | Medium | Strong validation pipeline, citation enforcement |
| Index corruption | Low | Regular backups, checksums |

---

## NOTES FOR IMPLEMENTATION

- **Error Handling**: Every function should have try-except blocks with meaningful error messages
- **Logging**: Use Python logging module, set level to INFO for production
- **Type Hints**: Use type hints throughout for code clarity
- **Documentation**: Every class and function should have docstrings
- **Testing**: Aim for > 80% code coverage
- **Performance**: Profile critical paths (embedding, retrieval) and optimize

---

END OF ANTIGRAVITY PROMPT
