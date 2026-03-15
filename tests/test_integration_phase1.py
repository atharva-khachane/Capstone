"""
Integration test for Phase 1: Full pipeline with real ISRO PDFs.

Tests the complete workflow:
1. Load real ISRO PDF (GFR document)
2. Anonymize PII
3. Encrypt content
4. Verify everything works together
"""

import pytest
from pathlib import Path
from sl_rag.core.document_loader import DocumentLoader
from sl_rag.core.pii_anonymizer import PIIAnonymizer
from sl_rag.core.encryption_manager import EncryptionManager
import tempfile


TEST_DATA_DIR = Path(__file__).parent.parent / "data"


class TestPhase1Integration:
    """Integration tests for Phase 1 components"""
    
    def test_full_secure_ingestion_pipeline(self):
        """
        Test complete secure ingestion workflow:
        Load PDF → Anonymize PII → Encrypt → Decrypt → Verify
        """
        # Setup components
        loader = DocumentLoader(ocr_enabled=False, sanitize=True)
        anonymizer = PIIAnonymizer(enable_ner=False)
        temp_dir = tempfile.mkdtemp()
        enc = EncryptionManager(master_key_path=f"{temp_dir}/master.key")
        
        # 1. Load ISRO document
        gfr_path = TEST_DATA_DIR / "FInal_GFR_upto_31_07_2024.pdf"
        if not gfr_path.exists():
           pytest.skip("GFR PDF not found")
        
        print(f"\n[STEP 1/5] Loading GFR document...")
        doc = loader.load_pdf(str(gfr_path))
        print(f"  ✓ Loaded: {doc.word_count} words, {doc.metadata['pages']} pages")
        assert doc.word_count > 1000
        
        # 2. Anonymize PII (if any)
        print(f"\n[STEP 2/5] Checking for PII...")
        anonymized_content, pii_found = anonymizer.anonymize(doc.content)
        total_pii = sum(pii_found.values())
        print(f"  ✓ PII scan complete: {total_pii} instances found")
        if total_pii > 0:
            print(f"    PII types: {pii_found}")
        
        # 3. Encrypt content
        print(f"\n[STEP 3/5] Encrypting document content...")
        encrypted_content = enc.encrypt_text(anonymized_content)
        print(f"  ✓ Encrypted: {len(encrypted_content)} bytes")
        assert len(encrypted_content) > 0
        
        # 4. Encrypt metadata
        print(f"\n[STEP 4/5] Encrypting metadata...")
        import json
        metadata_json = json.dumps(doc.metadata)
        encrypted_metadata = enc.encrypt_text(metadata_json)
        print(f"  ✓ Metadata encrypted: {len(encrypted_metadata)} bytes")
        
        # 5. Verify decryption works
        print(f"\n[STEP 5/5] Verifying decryption...")
        decrypted_content = enc.decrypt_text(encrypted_content)
        decrypted_metadata = enc.decrypt_text(encrypted_metadata)
        
        assert decrypted_content == anonymized_content
        assert decrypted_metadata == metadata_json
        print(f"  ✓ Decryption successful: content matches")
        
        # Verify security features
        assert doc.sanitized == True
        print(f"\n[SECURITY] ✓ Content sanitized")
        print(f"[SECURITY] ✓ PII anonymized ({total_pii} instances)")
        print(f"[SECURITY] ✓ Content encrypted (AES-256)")
        print(f"[SECURITY] ✓ Metadata encrypted")
        
        print(f"\n{'='*60}")
        print(f"PHASE 1 INTEGRATION TEST: ✓ PASSED")
        print(f"{'='*60}")
        print(f"Document: {doc.metadata['filename']}")
        print(f"Size: {doc.metadata['file_size_mb']}MB")
        print(f"Pages: {doc.metadata['pages']}")
        print(f"Words: {doc.word_count:,}")
        print(f"Document ID: {doc.doc_id[:16]}...")
        print(f"PII Found: {total_pii}")
        print(f"Sanitized: Yes")
        print(f"Encrypted: Yes (AES-256)")
        print(f"{'='*60}")
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_batch_document_processing(self):
        """Test processing multiple documents"""
        loader = DocumentLoader(ocr_enabled=False)
        
        # Load directory (will limit to non-oversized files)
        documents, stats = loader.load_directory(str(TEST_DATA_DIR), recursive=False)
        
        print(f"\n[BATCH PROCESSING] Loaded {stats['successful']}/{stats['total_files']} documents")
        print(f"  Total words: {stats['total_words']:,}")
        print(f"  Avg words/doc: {stats['avg_words_per_doc']:,.1f}")
        
        assert stats['successful'] > 0
        assert stats['total_words'] > 0
        
        # Verify all documents have unique IDs
        doc_ids = [doc.doc_id for doc in documents]
        unique_ids = set(doc_ids)
        assert len(doc_ids) == len(unique_ids), "All document IDs should be unique"
        
        print(f"  ✓ All {len(documents)} documents have unique IDs")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
