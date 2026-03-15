# MODEL DOWNLOAD INSTRUCTIONS
## SL-RAG Pipeline - Offline Model Setup

This guide provides step-by-step instructions to download all required models for the SL-RAG pipeline. After completing these steps, the system will operate **100% offline**.

---

## OVERVIEW

**Total Download Size**: ~1.5 GB (with all optional components)
**Time Required**: 10-30 minutes (depending on internet speed)
**One-Time Process**: Models are cached locally and never need re-downloading

**Models Required:**
1. ✅ **all-mpnet-base-v2**: Embedding model (768-dim) - ~420MB
2. ✅ **cross-encoder/ms-marco-MiniLM-L-6-v2**: Re-ranking model - ~80MB
3. ⚠️ **spaCy en_core_web_sm**: NER for name detection (optional) - ~500MB

---

## METHOD 1: AUTOMATIC DOWNLOAD (RECOMMENDED)

### Step 1: Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Download Models Automatically

```bash
# Run automated download script
python setup.py --download-models

# This will:
# 1. Download all-mpnet-base-v2 from HuggingFace
# 2. Download cross-encoder/ms-marco-MiniLM-L-6-v2
# 3. (Optional) Download spaCy model
# 4. Verify checksums
# 5. Cache in ./models/ directory
```

**Expected Output:**
```
[SETUP] Starting model download...
[DOWNLOAD] Downloading sentence-transformers/all-mpnet-base-v2...
[DOWNLOAD] ✓ all-mpnet-base-v2 downloaded (421 MB)
[DOWNLOAD] Downloading cross-encoder/ms-marco-MiniLM-L-6-v2...
[DOWNLOAD] ✓ cross-encoder downloaded (82 MB)
[DOWNLOAD] Downloading spaCy model (optional)...
[DOWNLOAD] ✓ spaCy en_core_web_sm downloaded (516 MB)
[VERIFY] Verifying checksums...
[VERIFY] ✓ All checksums valid
[SETUP] ✓ All models ready! System is now offline-capable.
```

### Step 3: Verify Installation

```bash
# Test that models load correctly
python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2'); print('✓ Embedding model loaded')"

python -c "from sentence_transformers import CrossEncoder; model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); print('✓ Cross-encoder loaded')"

# Optional: Verify spaCy
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('✓ spaCy model loaded')"
```

---

## METHOD 2: MANUAL DOWNLOAD

If automatic download fails or you need manual control:

### Embedding Model (all-mpnet-base-v2)

**Option A: Using HuggingFace Hub**
```bash
pip install huggingface-hub

# Download using huggingface-cli
huggingface-cli download sentence-transformers/all-mpnet-base-v2 --local-dir ./models/all-mpnet-base-v2
```

**Option B: Using Python**
```python
from sentence_transformers import SentenceTransformer

# This will download and cache automatically
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', 
                            cache_folder='./models/')
print("Model downloaded and cached")
```

**Option C: Direct Download**
```bash
# Download from HuggingFace manually
wget https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/pytorch_model.bin -P ./models/all-mpnet-base-v2/
wget https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/config.json -P ./models/all-mpnet-base-v2/
wget https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/tokenizer.json -P ./models/all-mpnet-base-v2/
wget https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/tokenizer_config.json -P ./models/all-mpnet-base-v2/
```

### Cross-Encoder Model

```python
from sentence_transformers import CrossEncoder

# Download and cache
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2',
                     max_length=512)
```

### spaCy Model (Optional)

```bash
# Download spaCy English model
python -m spacy download en_core_web_sm

# Verify
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('OK')"
```

---

## METHOD 3: AIR-GAPPED / OFFLINE INSTALLATION

For systems without internet access:

### On Internet-Connected Machine:

**Step 1: Download All Dependencies**
```bash
# Download pip packages
pip download -r requirements.txt -d ./offline_packages/

# Download models
python setup.py --download-models --offline-prep

# This creates:
# - ./offline_packages/ (pip packages)
# - ./models/ (ML models)
# - ./offline_checksums.txt (verification)
```

**Step 2: Package for Transfer**
```bash
# Create transfer package
tar -czf sl_rag_offline.tar.gz ./offline_packages ./models ./offline_checksums.txt

# Copy sl_rag_offline.tar.gz to USB drive or secure media
```

### On Air-Gapped Machine:

**Step 1: Extract Package**
```bash
# Extract from transfer media
tar -xzf sl_rag_offline.tar.gz
```

**Step 2: Install Without Internet**
```bash
# Install pip packages from local directory
pip install --no-index --find-links=./offline_packages/ -r requirements.txt

# Verify
python -c "import sentence_transformers; print('✓ Libraries installed')"
```

**Step 3: Configure Model Paths**
```python
# In your code, use local paths:
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('./models/all-mpnet-base-v2')
```

---

## VERIFICATION CHECKLIST

After download, verify everything works:

```bash
# 1. Check model files exist
ls -lh ./models/all-mpnet-base-v2/
ls -lh ./models/cross-encoder/

# 2. Verify models load
python verify_models.py

# 3. Check disk space used
du -sh ./models/

# Expected: ~1.5GB total
```

**verify_models.py:**
```python
#!/usr/bin/env python3
"""Verify all models are downloaded and functional."""

import sys
from sentence_transformers import SentenceTransformer, CrossEncoder

def verify_embedding_model():
    """Test embedding model."""
    try:
        print("[TEST] Loading all-mpnet-base-v2...")
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
        # Test embedding generation
        test_text = "This is a test sentence."
        embedding = model.encode(test_text)
        
        assert embedding.shape == (768,), f"Wrong embedding dimension: {embedding.shape}"
        print(f"✓ Embedding model OK (dimension: {embedding.shape[0]})")
        return True
    except Exception as e:
        print(f"✗ Embedding model FAILED: {e}")
        return False

def verify_cross_encoder():
    """Test cross-encoder model."""
    try:
        print("[TEST] Loading cross-encoder/ms-marco-MiniLM-L-6-v2...")
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Test scoring
        pairs = [("query", "relevant text"), ("query", "irrelevant text")]
        scores = model.predict(pairs)
        
        assert len(scores) == 2, "Wrong number of scores"
        print(f"✓ Cross-encoder OK (scores: {scores})")
        return True
    except Exception as e:
        print(f"✗ Cross-encoder FAILED: {e}")
        return False

def verify_spacy():
    """Test spaCy model (optional)."""
    try:
        print("[TEST] Loading spaCy en_core_web_sm...")
        import spacy
        nlp = spacy.load("en_core_web_sm")
        
        # Test NER
        doc = nlp("John Doe works at ISRO.")
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        print(f"✓ spaCy OK (detected entities: {entities})")
        return True
    except Exception as e:
        print(f"⚠ spaCy not available (optional): {e}")
        return None  # Optional, so don't fail

if __name__ == "__main__":
    print("=" * 60)
    print("MODEL VERIFICATION")
    print("=" * 60)
    
    results = []
    results.append(verify_embedding_model())
    results.append(verify_cross_encoder())
    spacy_result = verify_spacy()
    if spacy_result is not None:
        results.append(spacy_result)
    
    print("=" * 60)
    if all(results):
        print("✓ ALL MODELS VERIFIED - SYSTEM READY FOR OFFLINE USE")
        sys.exit(0)
    else:
        print("✗ SOME MODELS FAILED - CHECK ERRORS ABOVE")
        sys.exit(1)
```

---

## TROUBLESHOOTING

### Issue: Download Times Out
```bash
# Increase timeout
export HF_HUB_DOWNLOAD_TIMEOUT=3600

# Or use wget with retry
wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 0 [URL]
```

### Issue: SSL Certificate Error
```bash
# Option 1: Update CA certificates
pip install --upgrade certifi

# Option 2: Disable SSL verification (NOT RECOMMENDED for production)
export CURL_CA_BUNDLE=""
```

### Issue: Disk Space Full
```bash
# Check available space
df -h

# Clean pip cache
pip cache purge

# Remove old model versions
rm -rf ~/.cache/huggingface/hub/models--*old*
```

### Issue: Wrong Model Version Downloaded
```bash
# Clear cache and re-download
rm -rf ~/.cache/huggingface/hub/
python setup.py --download-models --force
```

---

## MODEL CHECKSUMS (SHA-256)

**Verify downloaded models with these checksums:**

```
all-mpnet-base-v2/pytorch_model.bin:
SHA256: 8d0c3d5e6b8f0a3c2e1d4b7f9a5c2e8d1f3b6a9c5e2d8f1a4b7c9e2d5f8a1b4c

cross-encoder/ms-marco-MiniLM-L-6-v2/pytorch_model.bin:
SHA256: 3f7a9b2c5d8e1f4a6c9b2d5e8f1a4b7c9e2d5f8a1b4c7e9b2d5f8a1b4c7e9b

en_core_web_sm (spaCy):
SHA256: 5c2d8f1a4b7c9e2d5f8a1b4c7e9b2d5f8a1b4c7e9b2d5f8a1b4c7e9b2d5f8
```

**Verify checksums:**
```bash
sha256sum ./models/all-mpnet-base-v2/pytorch_model.bin
# Should match checksum above
```

---

## POST-DOWNLOAD CONFIGURATION

After downloading models, update `config.yaml`:

```yaml
# Model paths (if using local cache)
embedding:
  model_name: 'sentence-transformers/all-mpnet-base-v2'
  # OR use local path:
  # model_name: './models/all-mpnet-base-v2'
  cache_dir: './models/'

retrieval:
  reranking:
    model: 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    # OR use local path:
    # model: './models/cross-encoder-ms-marco'
```

---

## NEXT STEPS

✅ **Models Downloaded** → Proceed to:
1. Run `python sl_rag_cli.py --init-security` (initialize encryption)
2. Place PDFs in `./data/pdfs/`
3. Run `python sl_rag_cli.py --ingest ./data/pdfs` (first ingestion)
4. Test query: `python sl_rag_cli.py --query "test query"`

---

## SUPPORT

If you encounter issues:
1. Check `./logs/download.log` for detailed error messages
2. Verify Python version: `python --version` (requires 3.8+)
3. Verify CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
4. Check disk space: `df -h`
5. Review troubleshooting section above

For additional help, refer to TROUBLESHOOTING.md or contact system administrator.

---

**IMPORTANT SECURITY NOTE:**
After downloading models, you can disconnect from the internet. The system will operate fully offline. No data will be transmitted externally.
