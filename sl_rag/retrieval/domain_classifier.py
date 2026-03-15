"""
Domain Classifier for rule-based domain assignment.

Implements a multi-stage classification pipeline:
1. Rule-based classification (keywords, patterns, structure)
2. Embedding-based classification (similarity to prototypes)
3. Fallback logic

100% OFFLINE - No external APIs.
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter

from ..core.schemas import Chunk


# Domain Configuration
DOMAIN_RULES = {
    "gfr": {
        "full_name": "General Financial Rules (GFR)",
        "keywords": {
            "high_confidence": [
                "general financial rules", "gfr", "delegation of financial powers",
                "budget allocation", "expenditure sanction", "financial approval",
                "consolidated fund", "contingency fund", "re-appropriation",
                "competent authority", "financial concurrence", "audit objection",
                "financial power", "budget provision", "appropriation"
            ],
            "medium_confidence": [
                "sanction", "budget", "financial", "expenditure", "approval",
                "delegation", "authority", "fund", "audit", "concurrence"
            ]
        },
        "patterns": [
            r"\bGFR\s*(?:Rule|Chapter)?\s*\d+",  # GFR Rule 123
            r"\bRule\s+\d+(?:\.\d+)*\s+of.*(?:GFR|financial rules)",  # Rule 123.4 of GFR
            r"(?:Rs\.|₹)\s*\d+(?:,\d+)*\s*(?:lakh|crore)?",  # Rs. 10 lakh
            r"\bF\.No\.\s*[\w/-]+",  # F.No. 12/34/2024
            r"\bFinancial\s+Year\s+\d{4}-\d{2}",  # FY 2024-25
            r"\b(?:budget|sanction|approval).*(?:Rs\.|₹)\s*\d+",  # budget Rs. 100
        ],
        "section_headers": [
            "financial rules", "delegation of powers", "budget provisions",
            "expenditure control", "audit requirements", "financial discipline",
            "appropriation", "sanction procedure"
        ],
        "document_indicators": [
            "gfr", "general financial rules", "financial manual",
            "budget", "financial delegation"
        ]
    },
    
    "procurement": {
        "full_name": "Procurement Manuals (Consultancy, Non-Consultancy, Goods)",
        "keywords": {
            "high_confidence": [
                "procurement manual", "consultancy", "non-consultancy", "goods procurement",
                "qcbs", "quality-cost based selection", "quality and cost based selection",
                "rfp", "request for proposal", "tender", "bid evaluation",
                "earnest money deposit", "emd", "letter of acceptance", "loa",
                "scope of work", "sow", "terms of reference", "tor",
                "l1 bidder", "lowest evaluated bid", "technical proposal",
                "financial proposal", "two envelope system", "single envelope",
                "prequalification", "expression of interest", "eoi"
            ],
            "medium_confidence": [
                "tender", "bidder", "proposal", "evaluation", "selection",
                "consultant", "vendor", "contractor", "bid", "quotation",
                "procurement", "purchase", "award", "contract"
            ]
        },
        "patterns": [
            r"\bNIT\s*(?:No\.?)?\s*[\w/-]+",  # Notice Inviting Tender
            r"\bRFP\s*(?:No\.?|#)\s*[\w/-]+",  # RFP No. 2024/01
            r"\bRFQ\s*(?:No\.?|#)\s*[\w/-]+",  # RFQ
            r"\bContract\s*(?:No\.?|#)\s*[\w/-]+",  # Contract No. ABC-123
            r"\bEMD\s*(?:of|:)?\s*(?:Rs\.|₹)\s*[\d,]+",  # EMD: Rs. 50,000
            r"\b(?:L1|L-1)\s*(?:bidder|vendor)",  # L1 bidder
            r"\b(?:QCBS|QBS|LCS|FBS|CQS)\b",  # Selection methods
            r"\b(?:90|80|70)/(?:10|20|30)\s*(?:quality|technical)",  # 90/10 QCBS
            r"\b(?:two|2)-envelope\s*system",  # Two envelope
        ],
        "section_headers": [
            "procurement procedure", "bidding process", "consultant selection",
            "evaluation criteria", "scope of work", "terms of reference",
            "contract award", "payment terms", "bill of quantities", "boq",
            "tender document", "bid submission"
        ],
        "document_indicators": [
            "procurement", "tender", "rfp", "consultancy", "goods",
            "bidding", "selection", "contract"
        ]
    },
    
    "technical": {
        "full_name": "Technical Reports, Telemetry Guides & Specifications",
        "keywords": {
            "high_confidence": [
                "technical report", "telemetry", "scada", "detailed project report", "dpr",
                "rtu", "remote terminal unit", "mtu", "master terminal unit",
                "sensor", "calibration", "real-time monitoring", "data acquisition",
                "technical specification", "system architecture", "design basis",
                "feasibility study", "performance test", "commissioning",
                "instrumentation", "measurement", "protocol", "interface"
            ],
            "medium_confidence": [
                "technical", "specification", "design", "system", "test",
                "performance", "monitoring", "data", "sensor", "equipment",
                "installation", "operation", "maintenance", "parameter"
            ]
        },
        "patterns": [
            r"\bDPR\s*(?:No\.?)?\s*[\w/-]*",  # DPR, DPR No. 123
            r"\bIS\s*\d+(?::\d{4})?",  # IS 456:2000
            r"\bIEEE\s*\d+",  # IEEE 802.11
            r"\bISO\s*\d+(?::\d{4})?",  # ISO 9001:2015
            r"\b(?:kW|MW|kVA|MVA|kV|V|A|Hz|m³|kg|ton|°C|bar|psi|Pa)\b",  # Engineering units
            r"\bFigure\s+\d+(?:\.\d+)*",  # Figure 3.2
            r"\bTable\s+\d+(?:\.\d+)*",  # Table 4.1
            r"\b(?:max|min|avg|typical)\s*[:=]\s*\d+",  # max: 100
            r"\b(?:sensor|probe|transducer|actuator)\s*[-:]\s*\w+",  # sensor: PT100
        ],
        "section_headers": [
            "technical specification", "system description", "design criteria",
            "telemetry system", "instrumentation", "testing procedure",
            "operation manual", "maintenance manual", "technical parameters",
            "system architecture", "functional description", "design basis"
        ],
        "document_indicators": [
            "technical", "dpr", "telemetry", "scada", "specification",
            "design", "report", "manual", "guide"
        ]
    }
}


# Scoring Configuration
SCORING_CONFIG = {
    "keyword_high_confidence": 3.0,
    "keyword_medium_confidence": 1.0,
    "pattern_match": 2.0,
    "section_header_match": 8.0,
    "document_title_match": 5.0,
    "context_boost": 1.2,
    "normalization_denominator": 12.0,
}


class DomainClassifier:
    """
    Rule-based domain classifier with multi-stage classification.
    
    Pipeline:
    1. Rule-based classification (keywords + patterns + structure)
    2. Embedding-based classification (similarity to prototypes)
    3. Fallback (best guess or 'general')
    """
    
    def __init__(self,
                 confidence_threshold: float = 0.6,
                 use_embeddings: bool = True,
                 use_context_propagation: bool = True):
        """
        Initialize domain classifier.
        
        Args:
            confidence_threshold: Minimum confidence to accept classification
            use_embeddings: Use embedding-based classification as fallback
            use_context_propagation: Use document-level context for boosting
        """
        self.confidence_threshold = confidence_threshold
        self.use_embeddings = use_embeddings
        self.use_context_propagation = use_context_propagation
        
        # Domain prototypes (built from high-confidence samples)
        self.domain_prototypes: Dict[str, np.ndarray] = {}
        
        # Document-level context cache
        self.document_contexts: Dict[str, str] = {}
        
        # Statistics
        self.classification_stats = {
            'rule_based': 0,
            'embedding_based': 0,
            'combined': 0,
            'low_confidence': 0,
            'fallback': 0
        }
    
    def classify(self, chunk: Chunk) -> Tuple[str, float, str]:
        """
        Classify a single chunk using multi-stage pipeline.
        
        Args:
            chunk: Chunk to classify
            
        Returns:
            Tuple of (domain, confidence, method)
        """
        # Stage 1: Rule-based classification
        rule_domain, rule_conf = self._rule_based_classify(chunk)
        
        if rule_conf >= self.confidence_threshold:
            self.classification_stats['rule_based'] += 1
            return rule_domain, rule_conf, "rule_based"
        
        # Stage 2: Embedding-based classification
        if self.use_embeddings and chunk.has_embedding() and self.domain_prototypes:
            emb_domain, emb_conf = self._embedding_classify(chunk)
            
            if emb_conf >= self.confidence_threshold:
                self.classification_stats['embedding_based'] += 1
                return emb_domain, emb_conf, "embedding_based"
            
            # Both methods agree? Boost confidence
            if emb_domain == rule_domain:
                combined = rule_conf * 0.6 + emb_conf * 0.4
                if combined >= self.confidence_threshold:
                    self.classification_stats['combined'] += 1
                    return rule_domain, combined, "combined"
        
        # Stage 3: Fallback
        if rule_conf > 0.2:
            self.classification_stats['low_confidence'] += 1
            return rule_domain, rule_conf, "low_confidence"
        
        self.classification_stats['fallback'] += 1
        return "general", 0.0, "fallback"
    
    def _rule_based_classify(self, chunk: Chunk) -> Tuple[str, float]:
        """
        Classify using rule-based scoring.
        
        Returns:
            Tuple of (domain, confidence)
        """
        domain_scores = {}
        
        for domain, config in DOMAIN_RULES.items():
            score = 0.0
            
            # Keyword scoring
            score += self._compute_keyword_score(chunk.content, config["keywords"])
            
            # Pattern scoring
            score += self._compute_pattern_score(chunk.content, config["patterns"])
            
            # Structure scoring (headers, metadata)
            score += self._compute_structure_score(chunk, config)
            
            domain_scores[domain] = score
        
        # Apply context boost if enabled
        if self.use_context_propagation:
            domain_scores = self._apply_context_boost(chunk, domain_scores)
        
        # Find best domain
        if not domain_scores or max(domain_scores.values()) == 0:
            return "general", 0.0
        
        best_domain = max(domain_scores, key=domain_scores.get)
        best_score = domain_scores[best_domain]
        
        # Normalize confidence (0-1 range)
        confidence = min(best_score / SCORING_CONFIG["normalization_denominator"], 1.0)
        
        return best_domain, confidence
    
    def _compute_keyword_score(self, text: str, keywords: Dict[str, List[str]]) -> float:
        """
        Compute score based on keyword matches.
        
        Args:
            text: Text content
            keywords: Dict with 'high_confidence' and 'medium_confidence' lists
            
        Returns:
            Keyword score
        """
        score = 0.0
        text_lower = text.lower()
        
        # High-confidence keywords
        for keyword in keywords.get("high_confidence", []):
            count = text_lower.count(keyword.lower())
            if count > 0:
                # Base score + diminishing returns bonus
                score += SCORING_CONFIG["keyword_high_confidence"]
                score += min(count - 1, 3) * 1.0
        
        # Medium-confidence keywords
        for keyword in keywords.get("medium_confidence", []):
            count = text_lower.count(keyword.lower())
            if count > 0:
                score += SCORING_CONFIG["keyword_medium_confidence"]
                score += min(count - 1, 3) * 0.3
        
        return score
    
    def _compute_pattern_score(self, text: str, patterns: List[str]) -> float:
        """
        Compute score based on regex pattern matches.
        
        Args:
            text: Text content
            patterns: List of regex patterns
            
        Returns:
            Pattern score
        """
        score = 0.0
        
        for pattern in patterns:
            try:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    # Cap pattern score at 6 points
                    score += min(len(matches) * SCORING_CONFIG["pattern_match"], 6.0)
            except re.error:
                # Skip invalid patterns
                continue
        
        return score
    
    def _compute_structure_score(self, chunk: Chunk, domain_config: Dict) -> float:
        """
        Compute score based on structural signals (headers, metadata).
        
        Args:
            chunk: Chunk with metadata
            domain_config: Domain configuration
            
        Returns:
            Structure score
        """
        score = 0.0
        
        # Check section headers
        header = chunk.metadata.get('section_header', '').lower()
        for expected in domain_config.get("section_headers", []):
            if expected.lower() in header:
                score += SCORING_CONFIG["section_header_match"]
                break
        
        # Check document title
        title = chunk.metadata.get('document_title', '').lower()
        for indicator in domain_config.get("document_indicators", []):
            if indicator.lower() in title:
                score += SCORING_CONFIG["document_title_match"]
                break
        
        # Check source document filename
        filepath = chunk.metadata.get('filepath', chunk.metadata.get('source_document', '')).lower()
        for indicator in domain_config.get("document_indicators", []):
            if indicator.lower() in filepath:
                score += SCORING_CONFIG["document_title_match"] * 0.5
                break
        
        return score
    
    def _embedding_classify(self, chunk: Chunk) -> Tuple[str, float]:
        """
        Classify using embedding similarity to prototypes.
        
        Args:
            chunk: Chunk with embedding
            
        Returns:
            Tuple of (domain, confidence)
        """
        if not chunk.has_embedding() or not self.domain_prototypes:
            return "general", 0.0
        
        # Normalize chunk embedding
        chunk_emb = chunk.embedding / (np.linalg.norm(chunk.embedding) + 1e-10)
        
        # Compute similarity to each prototype
        similarities = {}
        for domain, prototype in self.domain_prototypes.items():
            similarity = np.dot(chunk_emb, prototype)
            similarities[domain] = float(similarity)
        
        # Find best match
        if not similarities:
            return "general", 0.0
        
        best_domain = max(similarities, key=similarities.get)
        best_similarity = similarities[best_domain]
        
        # Convert cosine similarity to confidence (0-1 range)
        # Cosine similarity ranges from -1 to 1, but typically 0.3-1.0 for related text
        confidence = max(0, (best_similarity - 0.3) / 0.7)
        
        return best_domain, confidence
    
    def _apply_context_boost(self, chunk: Chunk, domain_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Apply document-level context boost to scores.
        
        Args:
            chunk: Chunk with document metadata
            domain_scores: Current domain scores
            
        Returns:
            Boosted domain scores
        """
        doc_id = chunk.metadata.get('document_id')
        if doc_id and doc_id in self.document_contexts:
            context_domain = self.document_contexts[doc_id]
            if context_domain in domain_scores:
                domain_scores[context_domain] *= SCORING_CONFIG["context_boost"]
        
        return domain_scores
    
    def build_prototypes(self, chunks: List[Chunk], min_samples: int = 10, min_confidence: float = 0.75):
        """
        Build domain prototypes from high-confidence classifications.
        
        Args:
            chunks: List of chunks to learn from
            min_samples: Minimum samples needed per domain
            min_confidence: Minimum confidence for training samples
        """
        print("[CLASSIFIER] Building domain prototypes...")
        
        domain_embeddings = defaultdict(list)
        
        for chunk in chunks:
            if not chunk.has_embedding():
                continue
            
            # Classify using rules only
            domain, confidence = self._rule_based_classify(chunk)
            
            # Use only very confident samples
            if confidence >= min_confidence:
                domain_embeddings[domain].append(chunk.embedding)
        
        # Compute prototypes
        for domain, embeddings in domain_embeddings.items():
            if len(embeddings) >= min_samples:
                centroid = np.mean(embeddings, axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
                self.domain_prototypes[domain] = centroid
                print(f"  [CLASSIFIER] {domain:20s}: {len(embeddings):4d} samples → prototype created")
            else:
                print(f"  [CLASSIFIER] {domain:20s}: {len(embeddings):4d} samples (need {min_samples}, skipped)")
        
        print(f"[CLASSIFIER] Built {len(self.domain_prototypes)} domain prototypes")
    
    def detect_document_context(self, chunks: List[Chunk], doc_id: str):
        """
        Detect document-level domain from title + first chunks.
        
        Args:
            chunks: Chunks from the same document
            doc_id: Document ID
        """
        if not chunks:
            return
        
        # Build context text from title + first 3 chunks
        context_text = ""
        for chunk in chunks[:3]:
            title = chunk.metadata.get('document_title', '')
            context_text += title + " " + chunk.content[:500]
        
        # Score each domain based on document indicators
        domain_scores = {}
        for domain, config in DOMAIN_RULES.items():
            score = 0
            for indicator in config["document_indicators"]:
                if indicator.lower() in context_text.lower():
                    score += 10
            domain_scores[domain] = score
        
        # Set context if confident enough
        if domain_scores and max(domain_scores.values()) >= 10:
            context_domain = max(domain_scores, key=domain_scores.get)
            self.document_contexts[doc_id] = context_domain
            print(f"[CLASSIFIER] Document context detected: {doc_id} → {context_domain}")
    
    def classify_batch(self, chunks: List[Chunk], verbose: bool = False) -> Dict:
        """
        Classify multiple chunks and return results.
        
        Args:
            chunks: List of chunks to classify
            verbose: Print detailed progress
            
        Returns:
            Classification results dictionary
        """
        print(f"[CLASSIFIER] Classifying {len(chunks)} chunks...")
        
        results = []
        
        for i, chunk in enumerate(chunks):
            domain, confidence, method = self.classify(chunk)
            
            # Assign to chunk
            chunk.domain = domain
            chunk.metadata['domain_confidence'] = confidence
            chunk.metadata['classification_method'] = method
            
            results.append({
                'chunk_id': chunk.chunk_id,
                'domain': domain,
                'confidence': confidence,
                'method': method
            })
            
            if verbose and (i + 1) % 100 == 0:
                print(f"  [CLASSIFIER] Processed {i + 1}/{len(chunks)} chunks...")
        
        # Compute statistics
        domain_distribution = Counter(r['domain'] for r in results)
        confidence_distribution = {
            'very_high': sum(1 for r in results if r['confidence'] >= 0.9),
            'high': sum(1 for r in results if 0.7 <= r['confidence'] < 0.9),
            'medium': sum(1 for r in results if 0.5 <= r['confidence'] < 0.7),
            'low': sum(1 for r in results if r['confidence'] < 0.5),
        }
        
        return {
            'classifications': results,
            'stats': dict(self.classification_stats),
            'domain_distribution': dict(domain_distribution),
            'confidence_distribution': confidence_distribution
        }
    
    def print_summary(self, results: Dict):
        """
        Print classification summary report.
        
        Args:
            results: Results from classify_batch()
        """
        total = len(results['classifications'])
        
        print("\n" + "=" * 70)
        print(" " * 20 + "CLASSIFICATION SUMMARY")
        print("=" * 70)
        print(f"\nTotal Chunks Classified: {total}")
        
        # Classification methods
        print("\n" + "-" * 70)
        print("Classification Methods:")
        print("-" * 70)
        for method, count in results['stats'].items():
            pct = count / total * 100 if total > 0 else 0
            bar = "█" * int(pct / 3)
            print(f"  {method:25s}: {count:4d} ({pct:5.1f}%) {bar}")
        
        # Domain distribution
        print("\n" + "-" * 70)
        print("Domain Distribution:")
        print("-" * 70)
        for domain, count in results['domain_distribution'].items():
            pct = count / total * 100 if total > 0 else 0
            bar = "█" * int(pct / 3)
            full_name = DOMAIN_RULES.get(domain, {}).get('full_name', domain.upper())
            print(f"  {full_name}")
            print(f"              : {count:4d} chunks ({pct:5.1f}%) {bar}")
        
        # Confidence distribution
        print("\n" + "-" * 70)
        print("Confidence Distribution:")
        print("-" * 70)
        for level, count in results['confidence_distribution'].items():
            pct = count / total * 100 if total > 0 else 0
            bar = "█" * int(pct / 3)
            print(f"  {level:15s}: {count:4d} ({pct:5.1f}%) {bar}")
        
        print("\n" + "=" * 70)
        
        # Quality check
        high_conf = results['confidence_distribution']['very_high'] + results['confidence_distribution']['high']
        high_conf_pct = high_conf / total * 100 if total > 0 else 0
        
        if high_conf_pct >= 80:
            print("\n✓ QUALITY CHECK: Excellent classification quality (≥80% high confidence)")
        elif high_conf_pct >= 60:
            print("\n⚠ QUALITY CHECK: Good classification quality (≥60% high confidence)")
        else:
            print("\n✗ QUALITY CHECK: Poor classification quality (<60% high confidence)")
            print("  Recommendation: Review domain rules and add more keywords/patterns")
        
        print()
