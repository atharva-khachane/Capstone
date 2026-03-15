"""
Comprehensive tests for DocumentLoader with real ISRO PDFs.

Tests:
- Loading individual PDFs
- Loading entire directory
- Metadata extraction
- Content sanitization
- Error handling (oversized files, corrupted PDFs)
"""

import pytest
import os
from pathlib import Path
from sl_rag.core.document_loader import DocumentLoader
from sl_rag.core.schemas import Document


# Test data directory
TEST_DATA_DIR = Path(__file__).parent.parent / "data"


class TestDocumentLoader:
    """Comprehensive tests for DocumentLoader"""
    
    def setup_method(self):
        """Setup for each test"""
        self.loader = DocumentLoader(
            ocr_enabled=False,  # Skip OCR for speed (no Tesseract installed yet)
            max_file_size_mb=200,
            sanitize=True,
        )
    
    def test_load_gfr_document(self):
        """Test loading GFR PDF (main reference document)"""
        gfr_path = TEST_DATA_DIR / "FInal_GFR_upto_31_07_2024.pdf"
        
        if not gfr_path.exists():
            pytest.skip(f"GFR PDF not found at {gfr_path}")
        
        doc = self.loader.load_pdf(str(gfr_path))
        
        # Verify document loaded
        assert doc is not None
        assert isinstance(doc, Document)
        
        # Verify content extracted
        assert len(doc.content) > 1000, "GFR should have substantial content"
        assert doc.word_count > 100, "GFR should have many words"
        
        # Verify metadata
        assert doc.metadata['filepath'] == str(gfr_path)
        assert doc.metadata['pages'] > 0
        assert doc.metadata['filename'] == "FInal_GFR_upto_31_07_2024.pdf"
        
        # Verify document ID (SHA-256 hash)
        assert len(doc.doc_id) == 64, "SHA-256 should be 64 hex chars"
        
        # Verify sanitization flag
        assert doc.sanitized == True
        
        print(f"\n✓ GFR loaded: {doc.word_count} words, {doc.metadata['pages']} pages")
    
    def test_load_procurement_goods(self):
        """Test loading Procurement Goods PDF"""
        proc_path = TEST_DATA_DIR / "Procurement_Goods.pdf"
        
        if not proc_path.exists():
            pytest.skip(f"Procurement PDF not found at {proc_path}")
        
        doc = self.loader.load_pdf(str(proc_path))
        
        assert doc is not None
        assert doc.word_count > 0
        assert doc.metadata['pages'] > 0
        assert "Procurement_Goods" in doc.metadata['filename']
        
        print(f"\n✓ Procurement Goods loaded: {doc.word_count} words")
    
    def test_load_technical_document(self):
        """Test loading technical document (smaller PDF)"""
        tech_path = TEST_DATA_DIR / "20180000774.pdf"
        
        if not tech_path.exists():
            pytest.skip(f"Technical PDF not found at {tech_path}")
        
        doc = self.loader.load_pdf(str(tech_path))
        
        assert doc is not None
        assert len(doc.content) > 0
        assert doc.metadata['file_size_mb'] < 1  # Should be small file
        
        print(f"\n✓ Technical doc loaded: {doc.word_count} words, {doc.metadata['file_size_mb']}MB")
    
    def test_load_directory(self):
        """Test loading all PDFs from data directory"""
        if not TEST_DATA_DIR.exists():
            pytest.skip(f"Data directory not found at {TEST_DATA_DIR}")
        
        # Load all PDFs (but limit to avoid the 106MB file for speed)
        documents, stats = self.loader.load_directory(
            str(TEST_DATA_DIR),
            recursive=False,
            pattern="*.pdf"
        )
        
        # Verify statistics
        assert stats['total_files'] == 8, "Should find 8 PDFs"
        assert stats['successful'] > 0, "Should successfully load at least some PDFs"
        assert stats['total_words'] > 0
        
        print(f"\n✓ Directory loaded:")
        print(f"  - Total files: {stats['total_files']}")
        print(f"  - Successful: {stats['successful']}")
        print(f"  - Failed: {stats['failed']}")
        print(f"  - Total words: {stats['total_words']:,}")
        print(f"  - Avg words/doc: {stats['avg_words_per_doc']:,.1f}")
        
        # Check for failures on the oversized file
        if stats['failed'] > 0:
            print(f"  - Failures: {stats['failures']}")
    
    def test_metadata_extraction(self):
        """Test that metadata is properly extracted"""
        gfr_path = TEST_DATA_DIR / "FInal_GFR_upto_31_07_2024.pdf"
        
        if not gfr_path.exists():
            pytest.skip(f"GFR PDF not found")
        
        doc = self.loader.load_pdf(str(gfr_path))
        
        # Check metadata fields
        required_fields = ['filepath', 'filename', 'pages', 'file_size_mb', 'ocr_used']
        for field in required_fields:
            assert field in doc.metadata, f"Missing metadata field: {field}"
        
        # Verify OCR flag
        assert doc.metadata['ocr_used'] == False, "OCR should not be used (disabled)"
        
        print(f"\n✓ Metadata extracted: {list(doc.metadata.keys())}")
    
    def test_content_sanitization(self):
        """Test that content sanitization works"""
        # This test verifies the sanitization flag is set
        gfr_path = TEST_DATA_DIR / "FInal_GFR_upto_31_07_2024.pdf"
        
        if not gfr_path.exists():
            pytest.skip(f"GFR PDF not found")
        
        doc = self.loader.load_pdf(str(gfr_path))
        
        assert doc.sanitized == True
        
        # Verify no HTML tags or excessive whitespace
        assert "<script" not in doc.content.lower()
        assert "</script>" not in doc.content.lower()
        
        print(f"\n✓ Content sanitized")
    
    def test_oversized_file_rejection(self):
        """Test that oversized files are rejected"""
        # Create a loader with very small size limit
        small_loader = DocumentLoader(max_file_size_mb=1)
        
        # Try to load the 106MB file
        large_file = TEST_DATA_DIR / "20210017934 FINAL.pdf"
        
        if not large_file.exists():
            pytest.skip(f"Large file not found")
        
        with pytest.raises(ValueError, match="too large"):
            small_loader.load_pdf(str(large_file))
        
        print(f"\n✓ Oversized file correctly rejected")
    
    def test_nonexistent_file(self):
        """Test error handling for nonexistent files"""
        fake_path = TEST_DATA_DIR / "nonexistent.pdf"
        
        with pytest.raises(FileNotFoundError):
            self.loader.load_pdf(str(fake_path))
        
        print(f"\n✓ Nonexistent file correctly raises error")
    
    def test_document_id_uniqueness(self):
        """Test that different documents get different IDs"""
        gfr_path = TEST_DATA_DIR / "FInal_GFR_upto_31_07_2024.pdf"
        proc_path = TEST_DATA_DIR / "Procurement_Goods.pdf"
        
        if not (gfr_path.exists() and proc_path.exists()):
            pytest.skip("Required PDFs not found")
        
        doc1 = self.loader.load_pdf(str(gfr_path))
        doc2 = self.loader.load_pdf(str(proc_path))
        
        # Document IDs should be different
        assert doc1.doc_id != doc2.doc_id
        
        print(f"\n✓ Document IDs are unique:")
        print(f"  - GFR: {doc1.doc_id[:16]}...")
        print(f"  - Procurement: {doc2.doc_id[:16]}...")
    
    def test_document_id_consistency(self):
        """Test that loading the same document twice gives same ID"""
        gfr_path = TEST_DATA_DIR / "FInal_GFR_upto_31_07_2024.pdf"
        
        if not gfr_path.exists():
            pytest.skip("GFR PDF not found")
        
        doc1 = self.loader.load_pdf(str(gfr_path))
        doc2 = self.loader.load_pdf(str(gfr_path))
        
        # Same document should have same ID
        assert doc1.doc_id == doc2.doc_id
        
        print(f"\n✓ Document ID is consistent: {doc1.doc_id[:16]}...")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
