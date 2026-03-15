"""
Query Preprocessor for normalization and acronym expansion.

Per ANTIGRAVITY_PROMPT.md Layer 3, queries must be preprocessed before
embedding to improve retrieval quality:
  - Text normalization (lowercasing, whitespace, special characters)
  - Acronym expansion (GFR, ISRO, QCBS, etc.)
  - Common abbreviation handling

100% OFFLINE - No external APIs.
"""

import re
from typing import Optional


ACRONYM_MAP = {
    "GFR":   "General Financial Rules GFR",
    "ISRO":  "Indian Space Research Organisation ISRO",
    "QCBS":  "Quality and Cost Based Selection QCBS",
    "QBS":   "Quality Based Selection QBS",
    "LCS":   "Least Cost Selection LCS",
    "FBS":   "Fixed Budget Selection FBS",
    "CQS":   "Consultants Qualifications Selection CQS",
    "RFP":   "Request for Proposal RFP",
    "RFQ":   "Request for Quotation RFQ",
    "NIT":   "Notice Inviting Tender NIT",
    "EOI":   "Expression of Interest EOI",
    "EMD":   "Earnest Money Deposit EMD",
    "LOA":   "Letter of Acceptance LOA",
    "SOW":   "Scope of Work SOW",
    "TOR":   "Terms of Reference TOR",
    "BOQ":   "Bill of Quantities BOQ",
    "DPR":   "Detailed Project Report DPR",
    "SCADA": "Supervisory Control and Data Acquisition SCADA",
    "RTU":   "Remote Terminal Unit RTU",
    "MTU":   "Master Terminal Unit MTU",
    "VSSC":  "Vikram Sarabhai Space Centre VSSC",
    "LPSC":  "Liquid Propulsion Systems Centre LPSC",
    "SDSC":  "Satish Dhawan Space Centre SDSC",
    "BM25":  "BM25 best matching",
}


class QueryPreprocessor:
    """Normalize and expand queries before embedding generation."""

    def __init__(
        self,
        expand_acronyms: bool = True,
        normalize: bool = True,
        custom_acronyms: Optional[dict] = None,
    ):
        self.expand_acronyms = expand_acronyms
        self.normalize = normalize
        self.acronyms = dict(ACRONYM_MAP)
        if custom_acronyms:
            self.acronyms.update(custom_acronyms)

    def preprocess(self, query: str) -> str:
        """Full preprocessing pipeline."""
        if self.normalize:
            query = self._normalize(query)
        if self.expand_acronyms:
            query = self._expand_acronyms(query)
        return query.strip()

    @staticmethod
    def _normalize(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s?.,'\"₹/-]", " ", text)
        return text.strip()

    def _expand_acronyms(self, text: str) -> str:
        for acronym, expansion in self.acronyms.items():
            pattern = r"\b" + re.escape(acronym) + r"\b"
            if re.search(pattern, text):
                text = re.sub(pattern, expansion, text, count=1)
        return text
