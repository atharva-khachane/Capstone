"""
Document Loader for PDF files with OCR support.

This module provides comprehensive PDF loading capabilities including:
- Text extraction from PDF documents
- OCR fallback for scanned documents
- Content sanitization (HTML/JavaScript removal)
- Metadata extraction
- File integrity validation
"""

import os
import hashlib
import re
from typing import List, Dict, Any, Tuple, Optional, Callable, Union
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError(
        "PyMuPDF is required. Install with: pip install PyMuPDF"
    )

try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("[WARNING] OCR not available. Install with: pip install pytesseract Pillow")

from .schemas import Document


class DocumentLoader:
    """
    Loads and processes PDF documents with sanitization and OCR support.
    
    Features:
    - Multi-format PDF support (text-based + OCR for scanned)
    - HTML/JavaScript/malicious code removal
    - Text normalization (Unicode, whitespace)
    - Metadata extraction (author, title, creation date)
    - File integrity validation (SHA-256 hash)
    - Directory loading with recursive support
    
    Args:
        ocr_enabled: Enable OCR for scanned documents (requires pytesseract)
        max_file_size_mb: Maximum file size in MB allowed
        sanitize: Enable HTML/JavaScript sanitization
        min_text_chars: Minimum characters to consider extraction successful
    """
    
    def __init__(
        self,
        ocr_enabled: bool = True,
        max_file_size_mb: int = 300,
        sanitize: bool = True,
        min_text_chars: int = 100,
        malware_scanner: Optional[Callable[[str, bytes], Union[bool, Tuple[bool, str]]]] = None,
    ):
        self.ocr_enabled = ocr_enabled and OCR_AVAILABLE
        self.max_file_size_mb = max_file_size_mb
        self.sanitize = sanitize
        self.min_text_chars = min_text_chars
        self.malware_scanner = malware_scanner
        
        if ocr_enabled and not OCR_AVAILABLE:
            print("[WARNING] OCR requested but not available. Falling back to text-only extraction.")
    
    def load_pdf(self, filepath: str) -> Document:
        """
        Load a single PDF document.
        
        Steps:
        1. Validate file exists and size < max_file_size_mb
        2. Extract text using PyMuPDF
        3. If text extraction fails or yields < min_text_chars, run OCR
        4. Sanitize content (remove HTML tags, scripts, special chars)
        5. Extract metadata
        6. Generate document ID (SHA-256 hash of content)
        7. Return Document object
        
        Args:
            filepath: Path to PDF file
            
        Returns:
            Document object with extracted content and metadata
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is too large or corrupted
        """
        filepath = str(Path(filepath).resolve())
        validation = self.validate_file(filepath)
        if not validation["is_valid"]:
            raise ValueError(f"Document validation failed: {validation['errors']}")
        file_size_mb = validation["file_size_mb"]
        
        # Extract text
        text, metadata = self._extract_text_from_pdf(filepath)
        
        # If extraction yielded insufficient text and OCR is enabled, try OCR
        if len(text) < self.min_text_chars and self.ocr_enabled:
            print(f"[OCR] Text extraction failed ({len(text)} chars), attempting OCR...")
            ocr_text = self._extract_text_with_ocr(filepath)
            if len(ocr_text) > len(text):
                text = ocr_text
                metadata['ocr_used'] = True
        
        # Sanitize content
        if self.sanitize:
            text = self._sanitize_content(text)
        
        metadata['filepath'] = filepath
        metadata['file_size_mb'] = round(file_size_mb, 2)
        metadata['filename'] = os.path.basename(filepath)
        
        # Generate document ID (SHA-256 hash)
        doc_id = self._generate_doc_id(text)
        
        # Create Document object
        document = Document(
            doc_id=doc_id,
            content=text,
            metadata=metadata,
            sanitized=self.sanitize,
        )
        
        return document

    def validate_file(self, filepath: str) -> Dict[str, Any]:
        """
        Validate file format, size, readability, and malware scan status.

        Returns:
            Dict with explicit validation outputs:
            - is_valid: bool
            - errors: list[str]
            - warnings: list[str]
            - file_size_mb: float
            - checks: per-check booleans
        """
        filepath = str(Path(filepath).resolve())
        errors: List[str] = []
        warnings: List[str] = []
        checks = {
            "exists": False,
            "extension_pdf": False,
            "size_within_limit": False,
            "readable": False,
            "malware_scan_passed": True,
        }

        if not os.path.exists(filepath):
            errors.append("file_not_found")
            return {
                "is_valid": False,
                "errors": errors,
                "warnings": warnings,
                "file_size_mb": 0.0,
                "checks": checks,
            }
        checks["exists"] = True

        if not filepath.lower().endswith(".pdf"):
            errors.append("invalid_extension")
        else:
            checks["extension_pdf"] = True

        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            errors.append("file_too_large")
        else:
            checks["size_within_limit"] = True

        file_bytes = b""
        try:
            with open(filepath, "rb") as f:
                file_bytes = f.read(4096)
            checks["readable"] = True
        except Exception:
            errors.append("read_failure")

        if file_bytes and not file_bytes.startswith(b"%PDF"):
            warnings.append("pdf_header_missing")

        if self.malware_scanner is not None:
            try:
                scan_result = self.malware_scanner(filepath, file_bytes)
                if isinstance(scan_result, tuple):
                    clean, reason = scan_result
                else:
                    clean, reason = bool(scan_result), ""
                if not clean:
                    checks["malware_scan_passed"] = False
                    errors.append(f"malware_detected:{reason or 'scanner_blocked'}")
            except Exception as e:
                checks["malware_scan_passed"] = False
                errors.append(f"malware_scan_error:{e}")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "file_size_mb": round(file_size_mb, 2),
            "checks": checks,
        }
    
    def load_directory(
        self, 
        dirpath: str, 
        recursive: bool = True,
        pattern: str = "*.pdf"
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Load all PDFs from a directory.
        
        Args:
            dirpath: Directory path
            recursive: Search subdirectories
            pattern: File pattern (default: *.pdf)
            
        Returns:
            Tuple of (documents list, statistics dict)
        """
        dirpath = Path(dirpath).resolve()
        
        if not dirpath.exists():
            raise FileNotFoundError(f"Directory not found: {dirpath}")
        
        # Find all PDF files
        if recursive:
            pdf_files = list(dirpath.rglob(pattern))
        else:
            pdf_files = list(dirpath.glob(pattern))
        
        documents = []
        failures = []
        
        print(f"[LOADING] Found {len(pdf_files)} PDF files in {dirpath}")
        
        for pdf_file in pdf_files:
            try:
                doc = self.load_pdf(str(pdf_file))
                documents.append(doc)
                print(f"[OK] Loaded: {pdf_file.name} ({doc.word_count} words)")
            except Exception as e:
                failures.append({'file': str(pdf_file), 'error': str(e)})
                print(f"[ERROR] Failed to load {pdf_file.name}: {e}")
        
        # Calculate statistics
        total_words = sum(doc.word_count for doc in documents)
        avg_words = total_words / len(documents) if documents else 0
        
        stats = {
            'total_files': len(pdf_files),
            'successful': len(documents),
            'failed': len(failures),
            'total_words': total_words,
            'avg_words_per_doc': round(avg_words, 1),
            'failures': failures,
        }
        
        return documents, stats
    
    def _extract_text_from_pdf(self, filepath: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text from PDF using PyMuPDF.
        
        Returns:
            Tuple of (text content, metadata dict)
        """
        try:
            doc = fitz.open(filepath)
            
            # Extract text from all pages
            text_parts = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                text_parts.append(page.get_text())
            
            text = "\n".join(text_parts)
            
            # Extract metadata
            metadata = {
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'creator': doc.metadata.get('creator', ''),
                'producer': doc.metadata.get('producer', ''),
                'creation_date': doc.metadata.get('creationDate', ''),
                'pages': len(doc),
                'ocr_used': False,
            }
            
            doc.close()
            
            return text, metadata
            
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {e}")
    
    def _extract_text_with_ocr(self, filepath: str) -> str:
        """
        Extract text using OCR (pytesseract).
        
        This is a fallback for scanned PDFs where text extraction fails.
        """
        if not self.ocr_enabled:
            return ""
        
        try:
            doc = fitz.open(filepath)
            text_parts = []
            
            for page_num in range(min(len(doc), 50)):  # Limit to 50 pages for OCR
                page = doc[page_num]
                
                # Render page to image
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                
                # Convert to PIL Image
                from io import BytesIO
                img = Image.open(BytesIO(img_data))
                
                # Run OCR
                page_text = pytesseract.image_to_string(img)
                text_parts.append(page_text)
                
                print(f"[OCR] Processed page {page_num + 1}/{len(doc)}")
            
            doc.close()
            
            return "\n".join(text_parts)
            
        except Exception as e:
            print(f"[OCR ERROR] {e}")
            return ""
    
    def _sanitize_content(self, text: str) -> str:
        """
        Sanitize text content by removing potentially malicious content.
        
        Removes:
        - HTML tags
        - JavaScript
        - Special control characters
        - Excessive whitespace

        Also applies OCR artifact correction before final whitespace normalization.
        """
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove JavaScript
        text = re.sub(r'<script.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove control characters (except newlines and tabs)
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')

        # Correct common OCR artifacts before whitespace normalization
        text = self._correct_ocr_artifacts(text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s+\n', '\n\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text

    # ------------------------------------------------------------------
    # OCR artifact correction
    # ------------------------------------------------------------------

    # Number words that OCR frequently substitutes for numerals.
    _NUMBER_WORDS: Dict[str, str] = {
        "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
        "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
        "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
        "nineteen": "19", "twenty": "20", "thirty": "30", "fifty": "50",
    }

    # Number word followed by a count/unit noun in procurement / financial text.
    # Captures OCR patterns like "eight tenders", "two years", "fifteen percent".
    _NUM_WORD_BEFORE_NOUN_RE = re.compile(
        r'\b(one|two|three|four|five|six|seven|eight|nine|ten'
        r'|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen'
        r'|eighteen|nineteen|twenty|thirty|fifty)\b'
        r'(?=\s+(?:tenders?|bids?|bidders?|firms?|proposals?|days?|weeks?|months?|years?'
        r'|percent|%|lakh|crore|rupees?|copies|sets?|items?|units?|stages?|phases?))',
        re.IGNORECASE,
    )

    # Ordinal number words preceding a noun (e.g. "eighth tender" → "8th tender")
    _ORDINAL_WORDS: Dict[str, str] = {
        "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th",
        "fifth": "5th", "sixth": "6th", "seventh": "7th", "eighth": "8th",
        "ninth": "9th", "tenth": "10th",
    }
    _ORDINAL_BEFORE_NOUN_RE = re.compile(
        r'\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b'
        r'(?=\s+(?:tenders?|bids?|bidders?|firms?|proposals?|stage|phase|item|copy|set))',
        re.IGNORECASE,
    )

    @classmethod
    def _correct_ocr_artifacts(cls, text: str) -> str:
        """Replace common OCR number-word artifacts with numerals.

        Only replaces number words that appear immediately before a count noun
        or unit, which is the diagnostic context for OCR misreads in legal and
        procurement documents (e.g. 'eight tenders' from a printed '8 tenders').
        Plain number words in prose ('the two parties') are left untouched.
        """
        def _replace_cardinal(m: re.Match) -> str:
            word = m.group(1).lower()
            return cls._NUMBER_WORDS.get(word, m.group(1))

        def _replace_ordinal(m: re.Match) -> str:
            word = m.group(1).lower()
            return cls._ORDINAL_WORDS.get(word, m.group(1))

        text = cls._NUM_WORD_BEFORE_NOUN_RE.sub(_replace_cardinal, text)
        text = cls._ORDINAL_BEFORE_NOUN_RE.sub(_replace_ordinal, text)
        return text
    
    def _generate_doc_id(self, content: str) -> str:
        """
        Generate unique document ID using SHA-256 hash of content.
        
        Args:
            content: Document text content
            
        Returns:
            SHA-256 hash as hex string
        """
        content_bytes = content.encode('utf-8')
        return hashlib.sha256(content_bytes).hexdigest()
