---
description: "Use when: optimizing prompt construction for RAG generation, managing context windows, handling multi-domain queries, preventing prompt injection"
applyTo: "sl_rag/generation/prompt_builder.py"
---

# Prompt Builder Quality Standards

## Chunk Organization Rules
1. **Sort by relevance** (reranker score descending)
2. **Group by domain** if multi-domain query
3. **Deduplicate** semantically similar chunks (cosine similarity > 0.85)
4. **Flag conflicts** when chunks contradict (log to audit trail)
5. **Preserve chunk order** in output for citation traceability

## Few-Shot Template Requirements

Every prompt must include 4 canonical examples:

### Example 1: Simple Factual Query
```
Query: "What is GFR Rule 149 about?"
Source chunks: [GFR_2017_Ch3_p45-46]
Expected answer: Direct rule citation with exact text, high confidence (>0.9)
```

### Example 2: Complex Multi-Step Query
```
Query: "Compare budget sanction rules across revenue and capital categories"
Source chunks: [GFR_2017_Ch4, GFR_2017_Ch5]
Expected answer: Structured comparison with multiple citations, medium confidence (0.7-0.8)
```

### Example 3: Edge Case / Non-Answer
```
Query: "Can I use government funds for personal equipment?"
Source chunks: [GFR_2017_Ch2 on unauthorized use]
Expected answer: "This is not permitted under GFR 2017 Rule X..."
```

### Example 4: Explicit Uncertainty
```
Query: "What are the latest amendments to procurement rules in 2026?"
Source chunks: []  (if no recent docs loaded)
Expected answer: "I cannot provide this information as recent amendments are not in my document set."
```

## Injection Detection Checklist
- [ ] Detect prompt escape attempts (e.g., "Ignore previous instructions...")
- [ ] Check for SQL/code injection patterns (e.g., "'; DROP TABLE;--")
- [ ] Validate domain classifier overrides (ensure domain stays within policy)
- [ ] Sanitize user input regex (remove special control characters)
- [ ] Log suspicious queries with severity level to audit trail

## Context Window Management
- **Token budget**: 1536 tokens max for Llama 3.2-3B
- **Context allocation**:
  - System prompt: ~100 tokens
  - Few-shot examples: ~400 tokens
  - Context chunks: ~600 tokens
  - Reserve buffer: ~200 tokens (for safety)
  - Max answer: 256 tokens

## Metadata Injection Standards
```
System: [Role-based context]
User domain classification: {domain}
User role: {user_role}
Session audit ID: {session_id}
Access level: {access_level}
Document classification: {doc_classification}
```

## Citation Format Standard
- Internal format: `[chunk_id:relevance_score]`
- Output format: `[Source: {docname}, Page {page_num}, Section {section}]`
- Never modify original chunk quotes
- Include page coordinates for traceability

## Validation Post-Generation
- Check that every claim has a citation
- Verify no fabricated chunk IDs
- Ensure answer respects max token limit
- Validate confidence score is numeric (0.0-1.0)
