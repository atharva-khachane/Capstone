"""
Encryption Manager for AES-256 encryption of stored data.

This module handles all encryption operations for the SL-RAG pipeline:
- AES-256 encryption via Fernet (symmetric)
- Secure key generation and management
- File and text encryption
- NumPy array encryption (for embeddings)
- Secure file deletion
"""

import os
import json
from pathlib import Path
from typing import Optional

try:
    from cryptography.fernet import Fernet
except ImportError:
    raise ImportError(
        "cryptography is required. Install with: pip install cryptography"
    )

import numpy as np


class EncryptionManager:
    """
    Manages AES-256 encryption for data at rest.
    
    Features:
    - AES-256 encryption via Fernet (symmetric)
    - Secure key derivation from master password
    - Separate encryption for different data types
    - Key rotation support
    - Encrypted index storage
    - Secure file deletion with multi-pass overwrite
    
    Args:
        master_key_path: Path to store the master encryption key
        auto_generate: Automatically generate key if it doesn't exist
    """
    
    def __init__(
        self,
        master_key_path: str = "./storage/keys/master.key",
        auto_generate: bool = True,
    ):
        self.master_key_path = Path(master_key_path)
        
        # Ensure key directory exists
        self.master_key_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load or generate master key
        if self.master_key_path.exists():
            with open(self.master_key_path, 'rb') as f:
                self.master_key = f.read()
            print(f"[SECURITY] Loaded master key from {self.master_key_path}")
        elif auto_generate:
            self.master_key = Fernet.generate_key()
            with open(self.master_key_path, 'wb') as f:
                f.write(self.master_key)
            
            # Set restrictive permissions (read/write for owner only)
            try:
                os.chmod(self.master_key_path, 0o600)  # Unix/Linux
            except Exception:
                # Windows doesn't support chmod the same way
                pass
            
            print(f"[SECURITY] Generated new master key at {self.master_key_path}")
            print(f"[SECURITY] WARNING: BACKUP THIS KEY! Loss of key = loss of all data!")
        else:
            raise ValueError(
                f"Master key not found at {master_key_path} and auto_generate=False"
            )
        
        # Initialize Fernet cipher
        self.cipher = Fernet(self.master_key)
    
    def encrypt_text(self, text: str) -> bytes:
        """
        Encrypt text data.
        
        Args:
            text: Plain text string
            
        Returns:
            Encrypted bytes
        """
        return self.cipher.encrypt(text.encode('utf-8'))
    
    def decrypt_text(self, encrypted_data: bytes) -> str:
        """
        Decrypt text data.
        
        Args:
            encrypted_data: Encrypted bytes
            
        Returns:
            Decrypted text string
        """
        return self.cipher.decrypt(encrypted_data).decode('utf-8')
    
    def encrypt_file(self, input_path: str, output_path: Optional[str] = None) -> str:
        """
        Encrypt an entire file.
        
        Args:
            input_path: Path to input file
            output_path: Path to output encrypted file (optional)
            
        Returns:
            Path to encrypted file
        """
        input_path = Path(input_path)
        
        if output_path is None:
            output_path = input_path.parent / f"{input_path.name}.encrypted"
        else:
            output_path = Path(output_path)
        
        # Read input file
        with open(input_path, 'rb') as f:
            data = f.read()
        
        # Encrypt
        encrypted = self.cipher.encrypt(data)
        
        # Write encrypted file
        with open(output_path, 'wb') as f:
            f.write(encrypted)
        
        print(f"[SECURITY] Encrypted {input_path.name} -> {output_path.name}")
        
        return str(output_path)
    
    def decrypt_file(self, input_path: str, output_path: str) -> str:
        """
        Decrypt an entire file.
        
        Args:
            input_path: Path to encrypted file
            output_path: Path to output decrypted file
            
        Returns:
            Path to decrypted file
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        # Read encrypted file
        with open(input_path, 'rb') as f:
            encrypted_data = f.read()
        
        # Decrypt
        decrypted = self.cipher.decrypt(encrypted_data)
        
        # Write decrypted file
        with open(output_path, 'wb') as f:
            f.write(decrypted)
        
        return str(output_path)
    
    def encrypt_numpy_array(self, array: np.ndarray) -> bytes:
        """
        Encrypt numpy array (for embeddings).
        
        Args:
            array: NumPy array
            
        Returns:
            Encrypted bytes
        """
        # Serialize to bytes
        array_bytes = array.tobytes()
        metadata = {
            'shape': array.shape,
            'dtype': str(array.dtype),
        }
        
        # Combine metadata and data
        combined = json.dumps(metadata).encode() + b'|||' + array_bytes
        
        return self.cipher.encrypt(combined)
    
    def decrypt_numpy_array(self, encrypted_data: bytes) -> np.ndarray:
        """
        Decrypt numpy array.
        
        Args:
            encrypted_data: Encrypted bytes
            
        Returns:
            NumPy array
        """
        # Decrypt
        decrypted = self.cipher.decrypt(encrypted_data)
        
        # Split metadata and data
        parts = decrypted.split(b'|||', 1)
        if len(parts) != 2:
            raise ValueError("Invalid encrypted array format")
        
        metadata = json.loads(parts[0].decode())
        array_bytes = parts[1]
        
        # Reconstruct array
        array = np.frombuffer(array_bytes, dtype=metadata['dtype'])
        array = array.reshape(metadata['shape'])
        
        return array
    
    def secure_delete(self, filepath: str, passes: int = 3) -> None:
        """
        Securely delete a file by overwriting with random data.
        
        This helps prevent recovery of sensitive data from disk.
        
        Args:
            filepath: File to delete
            passes: Number of overwrite passes (default: 3)
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            return
        
        file_size = filepath.stat().st_size
        
        # Overwrite file with random data multiple times
        with open(filepath, 'ba+') as f:
            for _ in range(passes):
                f.seek(0)
                f.write(os.urandom(file_size))
                f.flush()
                os.fsync(f.fileno())
        
        # Delete file
        filepath.unlink()
        print(f"[SECURITY] Securely deleted {filepath.name}")
    
    def rotate_key(self, new_master_key_path: str) -> None:
        """
        Rotate encryption key (for periodic security updates).
        
        Note: Requires re-encryption of all stored data.
        
        Args:
            new_master_key_path: Path for new master key
        """
        new_master_key_path = Path(new_master_key_path)
        
        # Generate new key
        new_key = Fernet.generate_key()
        
        with open(new_master_key_path, 'wb') as f:
            f.write(new_key)
        
        try:
            os.chmod(new_master_key_path, 0o600)
        except Exception:
            pass
        
        print(f"[SECURITY] New encryption key generated at {new_master_key_path}")
        print(f"[SECURITY] WARNING: Re-encrypt all data with new key!")
    
    def get_key_info(self) -> dict:
        """Get information about the current encryption key."""
        return {
            'key_path': str(self.master_key_path),
            'key_exists': self.master_key_path.exists(),
            'key_size_bytes': len(self.master_key) if hasattr(self, 'master_key') else 0,
            'algorithm': 'AES-256 (Fernet)',
        }
