"""
Comprehensive tests for EncryptionManager.

Tests all encryption capabilities including text, file, and numpy array encryption.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from sl_rag.core.encryption_manager import EncryptionManager


class TestEncryptionManager:
    """Comprehensive tests for Encryption Manager"""
    
    def setup_method(self):
        """Setup for each test - create temp key file"""
        self.temp_dir = tempfile.mkdtemp()
        self.key_path = Path(self.temp_dir) / "test_key.key"
        self.enc = EncryptionManager(master_key_path=str(self.key_path))
    
    def teardown_method(self):
        """Cleanup temp files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_key_generation(self):
        """Test that encryption key is generated"""
        assert self.key_path.exists()
        
        # Key should be 44 bytes (32-byte key base64 encoded)
        key_size = self.key_path.stat().st_size
        assert key_size == 44
        
        print(f"\n✓ Key generated: {key_size} bytes")
    
    def test_text_encryption_decryption(self):
        """Test text encryption and decryption"""
        original = "ISRO confidential satellite procurement data for mission XYZ"
        
        # Encrypt
        encrypted = self.enc.encrypt_text(original)
        assert isinstance(encrypted, bytes)
        assert len(encrypted) > 0
        
        # Decrypt
        decrypted = self.enc.decrypt_text(encrypted)
        assert decrypted == original
        
        print(f"\n✓ Text encryption:")
        print(f"  Original: {len(original)} chars")
        print(f"  Encrypted: {len(encrypted)} bytes")
        print(f"  Decrypted matches: {decrypted == original}")
    
    def test_text_encryption_produces_different_output(self):
        """Test that same text produces different encrypted output (IV randomization)"""
        text = "Test data"
        
        encrypted1 = self.enc.encrypt_text(text)
        encrypted2 = self.enc.encrypt_text(text)
        
        # Encrypted outputs should be different due to random IV
        assert encrypted1 != encrypted2
        
        # But both should decrypt to same text
        assert self.enc.decrypt_text(encrypted1) == text
        assert self.enc.decrypt_text(encrypted2) == text
        
        print(f"\n✓ Encryption randomization working (different IV each time)")
    
    def test_file_encryption_decryption(self):
        """Test file encryption and decryption"""
        # Create temp file
        input_file = Path(self.temp_dir) / "test.txt"
        input_file.write_text("Sensitive ISRO document content")
        
        # Encrypt file
        encrypted_path = self.enc.encrypt_file(str(input_file))
        assert Path(encrypted_path).exists()
        assert encrypted_path.endswith(".encrypted")
        
        # Decrypt file
        decrypted_path = Path(self.temp_dir) / "decrypted.txt"
        self.enc.decrypt_file(encrypted_path, str(decrypted_path))
        
        # Verify content matches
        original_content = input_file.read_text()
        decrypted_content = decrypted_path.read_text()
        assert decrypted_content == original_content
        
        print(f"\n✓ File encryption:")
        print(f"  Original: {input_file.name}")
        print(f"  Encrypted: {Path(encrypted_path).name}")
        print(f"  Content matches: True")
    
    def test_numpy_array_encryption(self):
        """Test numpy array encryption (for embeddings)"""
        # Create test embedding (768-dim like all-mpnet-base-v2)
        original = np.random.rand(768).astype(np.float32)
        
        # Encrypt
        encrypted = self.enc.encrypt_numpy_array(original)
        assert isinstance(encrypted, bytes)
        
        # Decrypt
        decrypted = self.enc.decrypt_numpy_array(encrypted)
        
        # Verify shape and values
        assert decrypted.shape == original.shape
        assert decrypted.dtype == original.dtype
        assert np.allclose(decrypted, original)
        
        print(f"\n✓ NumPy array encryption:")
        print(f"  Shape: {original.shape}")
        print(f"  Dtype: {original.dtype}")
        print(f"  Encrypted size: {len(encrypted)} bytes")
        print(f"  Arrays match: {np.allclose(decrypted, original)}")
    
    def test_numpy_array_2d(self):
        """Test encryption of 2D numpy arrays (batch of embeddings)"""
        # Create batch of 10 embeddings
        original = np.random.rand(10, 768).astype(np.float32)
        
        encrypted = self.enc.encrypt_numpy_array(original)
        decrypted = self.enc.decrypt_numpy_array(encrypted)
        
        assert decrypted.shape == (10, 768)
        assert np.allclose(decrypted, original)
        
        print(f"\n✓ 2D array encryption: shape {original.shape}")
    
    def test_secure_delete(self):
        """Test secure file deletion"""
        # Create temp file
        test_file = Path(self.temp_dir) / "to_delete.txt"
        test_file.write_text("Sensitive data that must be securely deleted")
        
        assert test_file.exists()
        
        # Secure delete
        self.enc.secure_delete(str(test_file))
        
        # File should be gone
        assert not test_file.exists()
        
        print(f"\n✓ Secure delete successful")
    
    def test_invalid_decryption(self):
        """Test that invalid encrypted data raises error"""
        invalid_data = b"this is not encrypted data"
        
        with pytest.raises(Exception):  # Fernet will raise InvalidToken
            self.enc.decrypt_text(invalid_data)
        
        print(f"\n✓ Invalid decryption correctly raises error")
    
    def test_key_info(self):
        """Test getting key information"""
        info = self.enc.get_key_info()
        
        assert 'key_path' in info
        assert 'key_exists' in info
        assert 'algorithm' in info
        
        assert info['key_exists'] == True
        assert info['algorithm'] == 'AES-256 (Fernet)'
        
        print(f"\n✓ Key info: {info}")
    
    def test_encryption_persistence(self):
        """Test that encryption key can be reloaded"""
        # Encrypt with first manager
        text = "Test persistence"
        encrypted = self.enc.encrypt_text(text)
        
        # Create new manager with same key
        enc2 = EncryptionManager(master_key_path=str(self.key_path), auto_generate=False)
        
        # Should be able to decrypt
        decrypted = enc2.decrypt_text(encrypted)
        assert decrypted == text
        
        print(f"\n✓ Encryption key persistence working")
    
    def test_large_text_encryption(self):
        """Test encryption of large text (like full document)"""
        # Create large text (simulate document content)
        large_text = "This is a test document. " * 10000  # ~250KB
        
        encrypted = self.enc.encrypt_text(large_text)
        decrypted = self.enc.decrypt_text(encrypted)
        
        assert decrypted == large_text
        
        print(f"\n✓ Large text encryption:")
        print(f"  Original size: {len(large_text)} chars")
        print(f"  Encrypted size: {len(encrypted)} bytes")
    
    def test_empty_text_encryption(self):
        """Test encryption of empty text"""
        empty = ""
        
        encrypted = self.enc.encrypt_text(empty)
        decrypted = self.enc.decrypt_text(encrypted)
        
        assert decrypted == empty
        
        print(f"\n✓ Empty text encryption working")
    
    def test_unicode_text_encryption(self):
        """Test encryption of Unicode text"""
        unicode_text = "ISRO मिशन चंद्रयान-3 🚀"
        
        encrypted = self.enc.encrypt_text(unicode_text)
        decrypted = self.enc.decrypt_text(encrypted)
        
        assert decrypted == unicode_text
        
        print(f"\n✓ Unicode encryption: {unicode_text}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
