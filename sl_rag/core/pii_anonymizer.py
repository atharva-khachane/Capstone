"""
PII Anonymizer for detecting and redacting personally identifiable information.

This module provides comprehensive PII detection including India-specific patterns:
- Aadhaar numbers, PAN cards, Indian phone numbers
- Email addresses, SSN, credit cards, IP addresses
- Government employee IDs
- Optional NER-based name detection (spaCy)
"""

import re
from typing import Dict, Tuple, List, Any


class PIIAnonymizer:
    """
    Detects and redacts PII using regex patterns and optional NER.
    
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
    
    Args:
        enable_ner: Enable spaCy NER for name detection
        replacement_token: Token to replace PII with
        log_detections: Log detected PII statistics
    """
    
    def __init__(
        self,
        enable_ner: bool = False,
        replacement_token: str = "[REDACTED]",
        log_detections: bool = True,
    ):
        self.enable_ner = enable_ner
        self.replacement_token = replacement_token
        self.log_detections = log_detections
        
        # PII regex patterns (India-specific + international)
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone_indian': r'(\+91[-\s]?)?[6-9]\d{9}\b',  # Indian phone numbers
            'aadhaar': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Aadhaar format
            'pan': r'\b[A-Z]{5}\d{4}[A-Z]\b',  # PAN card
            'passport': r'\b[A-Z]\d{7}\b',  # Indian passport  
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',  # US SSN
            'credit_card': r'\b\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}\b',  # Fixed: require separator
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            'dob': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            'employee_id': r'\b(EMP|ISRO|ID)[-_]?\d{5,8}\b',  # Government employee IDs
        }
        
        # Initialize spaCy for name detection (optional)
        self.nlp = None
        if enable_ner:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                print(f"[WARNING] spaCy not available for name detection: {e}")
                self.enable_ner = False
        
        # Detection statistics
        self.total_detections = {pii_type: 0 for pii_type in self.patterns.keys()}
        if enable_ner:
            self.total_detections['names_ner'] = 0
    
    def anonymize(
        self, 
        text: str, 
        preserve_structure: bool = True
    ) -> Tuple[str, Dict[str, int]]:
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
            matches = list(re.finditer(pattern, anonymized))
            count = len(matches)
            detections[pii_type] = count
            self.total_detections[pii_type] += count
            
            # Replace with specific or generic token
            if preserve_structure:
                replacement = f'[{pii_type.upper()}_REDACTED]'
            else:
                replacement = self.replacement_token
            
            anonymized = re.sub(pattern, replacement, anonymized, flags=re.IGNORECASE)
        
        # NER-based name detection (optional)
        if self.enable_ner and self.nlp:
            anonymized, name_count = self._redact_names(anonymized)
            detections['names_ner'] = name_count
            self.total_detections['names_ner'] += name_count
        
        # Log if enabled
        total_pii = sum(detections.values())
        if self.log_detections and total_pii > 0:
            print(f"[PII] Detected and redacted: {detections}")
        
        return anonymized, detections
    
    def _redact_names(self, text: str) -> Tuple[str, int]:
        """
        Use spaCy NER to detect and redact person names.
        
        Returns:
            Tuple of (redacted_text, name_count)
        """
        if not self.enable_ner or not self.nlp:
            return text, 0
        
        try:
            doc = self.nlp(text)
            redacted = text
            name_count = 0
            
            # Replace PERSON entities (reverse order to preserve indices)
            for ent in reversed(doc.ents):
                if ent.label_ == "PERSON":
                    redacted = (
                        redacted[:ent.start_char] + 
                        "[PERSON_REDACTED]" + 
                        redacted[ent.end_char:]
                    )
                    name_count += 1
            
            return redacted, name_count
            
        except Exception as e:
            print(f"[WARNING] NER failed: {e}")
            return text, 0
    
    def validate_anonymization(
        self, 
        original: str, 
        anonymized: str
    ) -> Dict[str, Any]:
        """
        Validate that anonymization was successful.
        
        Checks if any PII patterns still remain in the anonymized text.
        
        Returns:
            Dict with validation results:
            - success: bool (True if no PII remaining)
            - pii_remaining: Dict[str, int] (PII types still present)
            - total_removed: int (total PII instances removed)
        """
        remaining_pii = {}
        
        for pii_type, pattern in self.patterns.items():
            matches = re.findall(pattern, anonymized, flags=re.IGNORECASE)
            # Filter out our redaction tokens
            actual_matches = [
                m for m in matches 
                if 'REDACTED' not in m.upper()
            ]
            if actual_matches:
                remaining_pii[pii_type] = len(actual_matches)
        
        return {
            'success': len(remaining_pii) == 0,
            'pii_remaining': remaining_pii,
            'total_removed': sum(self.total_detections.values()),
        }
    
    def get_detection_patterns(self) -> Dict[str, str]:
        """Return all regex patterns for transparency and audit."""
        return self.patterns.copy()
    
    def get_statistics(self) -> Dict[str, int]:
        """Return total detection statistics across all processed documents."""
        return self.total_detections.copy()
    
    def reset_statistics(self) -> None:
        """Reset detection statistics."""
        self.total_detections = {pii_type: 0 for pii_type in self.patterns.keys()}
        if self.enable_ner:
            self.total_detections['names_ner'] = 0
