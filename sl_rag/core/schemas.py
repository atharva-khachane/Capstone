"""
Data schemas for the SL-RAG pipeline.

This module defines the core data structures used throughout the system:
- Document: Represents a loaded document with metadata
- Chunk: Represents a semantically meaningful chunk of a document
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class Document:
    """
    Represents a loaded and processed document.
    
    Attributes:
        doc_id: Unique identifier (SHA-256 hash of content)
        content: Sanitized and PII-removed text content
        metadata: Document metadata (title, author, pages, filepath, etc.)
        domain: Auto-detected or manually assigned domain
        timestamp: ISO format timestamp of document ingestion
        sanitized: Whether HTML/JavaScript sanitization was applied
        pii_removed: Whether PII anonymization was applied
        word_count: Total word count
        char_count: Total character count
    """
    
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    domain: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    sanitized: bool = False
    pii_removed: bool = False
    word_count: int = 0
    char_count: int = 0
    
    def __post_init__(self):
        """Calculate word and character counts if not provided."""
        if self.word_count == 0:
            self.word_count = len(self.content.split())
        if self.char_count == 0:
            self.char_count = len(self.content)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary for serialization."""
        return {
            'doc_id': self.doc_id,
            'content': self.content,
            'metadata': self.metadata,
            'domain': self.domain,
            'timestamp': self.timestamp,
            'sanitized': self.sanitized,
            'pii_removed': self.pii_removed,
            'word_count': self.word_count,
            'char_count': self.char_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create document from dictionary."""
        return cls(**data)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        title = self.metadata.get('title', 'Unknown')
        return f"Document(id={self.doc_id[:8]}..., title={title}, words={self.word_count})"


@dataclass
class Chunk:
    """
    Represents a chunk of text from a document with metadata.
    
    Attributes:
        chunk_id: Unique identifier for the chunk
        doc_id: ID of parent document
        content: Text content of the chunk
        chunk_index: Position in document (0-indexed)
        start_char: Starting character position in document
        end_char: Ending character position in document
        token_count: Approximate number of tokens
        embedding: Optional embedding vector (768-dim)
        domain: Optional domain classification
        metadata: Additional metadata (filepath, source_document, etc.)
    """
    
    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    token_count: int
    embedding: Optional[np.ndarray] = None
    domain: Optional[str] = None
    metadata: dict = None
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self, include_embedding: bool = False) -> Dict[str, Any]:
        """
        Convert chunk to dictionary for serialization.
        
        Args:
            include_embedding: Whether to include the embedding array
                              (can be large, so excluded by default)
        """
        data = {
            'chunk_id': self.chunk_id,
            'doc_id': self.doc_id,
            'content': self.content,
            'chunk_index': self.chunk_index,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'token_count': self.token_count,
            'domain': self.domain,
        }
        
        if include_embedding and self.embedding is not None:
            data['embedding'] = self.embedding.tolist()
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chunk':
        """Create chunk from dictionary."""
        # Convert embedding list back to numpy array if present
        if 'embedding' in data and data['embedding'] is not None:
            data['embedding'] = np.array(data['embedding'], dtype=np.float32)
        return cls(**data)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Chunk(id={self.chunk_id}, tokens={self.token_count}, preview='{content_preview}')"
    
    def has_embedding(self) -> bool:
        """Check if chunk has an embedding."""
        return self.embedding is not None and len(self.embedding) > 0
