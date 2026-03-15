"""
Chunk Generator for semantic-aware text splitting.

This module provides intelligent text chunking that:
- Respects sentence boundaries for semantic coherence
- Maintains configurable chunk size and overlap
- Preserves metadata for each chunk
- Supports token-based splitting
"""

import re
from typing import List, Tuple
from dataclasses import dataclass

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("[WARNING] NLTK not available. Using simple sentence splitting.")

from .schemas import Document, Chunk


class ChunkGenerator:
    """
    Generates semantic chunks from documents.
    
    Features:
    - Sentence-boundary aware splitting
    - Configurable chunk size and overlap
    - Token counting (approximate)
    - Metadata preservation
    - Batch processing support
    
    Args:
        chunk_size: Target chunk size in tokens (default: 512)
        overlap: Number of overlapping tokens between chunks (default: 50)
        min_chunk_size: Minimum chunk size in tokens (default: 100)
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        min_chunk_size: int = 100,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        
        # Initialize NLTK sentence tokenizer if available
        if NLTK_AVAILABLE:
            try:
                # Try to use punkt tokenizer
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                # Download if not available
                print("[NLTK] Downloading punkt tokenizer...")
                try:
                    nltk.download('punkt', quiet=True)
                    nltk.download('punkt_tab', quiet=True)  # New punkt data
                except Exception as e:
                    print(f"[WARNING] Failed to download NLTK data: {e}")
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Split document into semantic chunks.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of Chunk objects
        """
        chunks = []
        
        # Split into sentences
        sentences = self._split_sentences(document.content)
        
        # Group sentences into chunks
        current_chunk = []
        current_tokens = 0
        chunk_index = 0
        start_char = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            # If adding this sentence would exceed chunk size
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_content = " ".join(current_chunk)
                end_char = start_char + len(chunk_content)
                
                chunk = Chunk(
                    chunk_id=f"{document.doc_id}_chunk_{chunk_index}",
                    doc_id=document.doc_id,
                    content=chunk_content,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                    token_count=current_tokens,
                    domain=document.domain,
                )
                chunks.append(chunk)
                
                # Handle overlap: keep last few sentences for next chunk
                overlap_sentences, overlap_tokens = self._calculate_overlap(current_chunk)
                current_chunk = overlap_sentences
                current_tokens = overlap_tokens
                
                # Update start_char for next chunk (accounting for overlap)
                if overlap_sentences:
                    overlap_text = " ".join(overlap_sentences)
                    start_char = end_char - len(overlap_text)
                else:
                    start_char = end_char
                
                chunk_index += 1
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        # Add final chunk if any content remains
        if current_chunk and current_tokens >= self.min_chunk_size:
            chunk_content = " ".join(current_chunk)
            end_char = start_char + len(chunk_content)
            
            chunk = Chunk(
                chunk_id=f"{document.doc_id}_chunk_{chunk_index}",
                doc_id=document.doc_id,
                content=chunk_content,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=end_char,
                token_count=current_tokens,
                domain=document.domain,
            )
            chunks.append(chunk)
        
        return chunks
    
    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of documents
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using NLTK or simple regex.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        if NLTK_AVAILABLE:
            try:
                return sent_tokenize(text)
            except Exception as e:
                print(f"[WARNING] NLTK sentence tokenization failed: {e}")
        
        # Fallback: simple regex-based splitting
        # Split on period, question mark, or exclamation followed by space
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _count_tokens(self, text: str) -> int:
        """
        Approximate token count (simple whitespace-based).
        
        For more accurate counting, could use transformers tokenizer,
        but this is faster and sufficient for chunking.
        
        Args:
            text: Text to count tokens in
            
        Returns:
            Approximate token count
        """
        # Simple approximation: split on whitespace + punctuation
        # Real tokens are typically ~75% of word count
        words = text.split()
        return int(len(words) * 1.3)  # Slightly overestimate to be safe
    
    def _calculate_overlap(self, sentences: List[str]) -> Tuple[List[str], int]:
        """
        Calculate overlap sentences to carry to next chunk.
        
        Args:
            sentences: Current chunk sentences
            
        Returns:
            Tuple of (overlap_sentences, overlap_token_count)
        """
        if not sentences or self.overlap == 0:
            return [], 0
        
        # Work backward from end to find sentences that fit in overlap
        overlap_sentences = []
        overlap_tokens = 0
        
        for sentence in reversed(sentences):
            sentence_tokens = self._count_tokens(sentence)
            if overlap_tokens + sentence_tokens <= self.overlap:
                overlap_sentences.insert(0, sentence)
                overlap_tokens += sentence_tokens
            else:
                break
        
        return overlap_sentences, overlap_tokens
    
    def get_chunk_statistics(self, chunks: List[Chunk]) -> dict:
        """
        Calculate statistics for chunks.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_tokens_per_chunk': 0,
                'min_tokens': 0,
                'max_tokens': 0,
            }
        
        token_counts = [chunk.token_count for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_tokens_per_chunk': sum(token_counts) / len(token_counts),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'total_tokens': sum(token_counts),
        }
