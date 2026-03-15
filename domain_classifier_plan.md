# Rule-Based Domain Classifier - Implementation Plan

## Objective
Implement a **multi-stage domain classifier** for categorizing document chunks into three predefined domains (GFR, Procurement, Technical) using rule-based methods with embedding-based fallback.

## Proposed Changes

### New File: `sl_rag/retrieval/domain_classifier.py`

**Core Components:**

1. **Domain Configuration**
   - Define rules for GFR, Procurement, Technical domains
   - High/medium confidence keywords
   - Regex patterns for domain-specific formats
   - Section headers and document indicators

2. **DomainClassifier Class**
   - Rule-based classification (Stage 1)
   - Embedding-based classification (Stage 2)
   - Fallback logic (Stage 3)
   - Prototype building from high-confidence samples
   - Context propagation for document-level signals

3. **Scoring Functions**
   - Keyword scoring (high vs medium confidence)
   - Pattern matching scoring
   - Structure scoring (headers, metadata)
   - Confidence normalization

4. **Batch Processing**
   - Classify multiple chunks efficiently
   - Generate classification statistics
   - Summary reports with visualization

---

## Implementation Steps

### Step 1: Create Domain Configuration
- [x] Define keyword sets for each domain
- [x] Define regex patterns for domain-specific formats
- [x] Define section headers and document indicators
- [x] Configure scoring weights

### Step 2: Implement Core Classifier
- [x] `DomainClassifier` class structure
- [x] Rule-based classification method
- [x] Scoring functions (keywords, patterns, structure)
- [x] Confidence calculation and normalization

### Step 3: Add Embedding Support
- [x] Prototype building from high-confidence samples
- [x] Embedding-based classification
- [x] Combined scoring (rule + embedding)

### Step 4: Context Propagation
- [x] Document-level context detection
- [x] Context boost for chunk classification

### Step 5: Batch Processing & Reporting
- [x] Batch classification method
- [x] Statistics tracking
- [x] Summary report generation

### Step 6: Integration
- [ ] Update retrieval pipeline to use classifier (deferred for user)
- [ ] Replace/update existing DomainManager usage (optional - both can coexist)
- [ ] Test with existing documents

### Step 7: Testing & Validation
- [x] Create test script
- [x] Validate classification quality (via test script)
- [ ] Manual review of samples (user task)

---

## File Structure

```
sl_rag/retrieval/
├── domain_classifier.py (NEW)
│   ├── DOMAIN_RULES (configuration)
│   └── DomainClassifier (main class)
│
├── domain_manager.py (KEEP for backward compatibility)
│   └── K-means clustering approach
│
└── retrieval_pipeline.py (UPDATE)
    └── Use DomainClassifier instead
```

---

## Testing Plan

### Unit Tests
- Test keyword scoring
- Test pattern matching
- Test confidence calculation
- Test prototype building

### Integration Tests
- Test with sample GFR documents
- Test with procurement documents
- Test with technical reports
- Test mixed document batches

### Validation
- Manual review of 50 random classifications
- Check confidence distribution
- Verify domain balance
- Review low-confidence cases

---

## Success Criteria

- ✅ High confidence rate ≥ 80%
- ✅ Fallback rate < 5%
- ✅ All three domains properly detected
- ✅ Classification summary report generated
- ✅ Integration with existing pipeline works
