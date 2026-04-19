# SL-RAG Prompt Generation Quality Guidelines

## Prompt Quality Standards

### For Government Documents
- **Legal accuracy**: Never paraphrase policy numbers or regulation names
- **Citation rigor**: All facts must link to source chunk IDs
- **Confidence thresholds**: 
  - GFR queries: Require >0.75 confidence
  - Technical docs: Allow >0.65 confidence
  - Procurement: Require >0.80 confidence

### Context Injection Rules
- Max 3 chunks (≤1536 tokens for Llama 3.2-3B context window)
- Organize by relevance score (descending)
- Include chunk page numbers in citations
- Flag domain mismatches in prompt metadata

### Hallucination Prevention
- Few-shot examples must include "I don't know" cases
- Never generate regulations/amounts not in source
- Use exact quotes when >50% of answer comes from one chunk

## Generation Best Practices
- Temperature: 0.3–0.5 for factual queries
- Max tokens: 256 (leave buffer for safety)
- Stop tokens: Add domain-specific termination patterns
- Post-generation validation: Check for unmatched citations

## Testing & Validation
- Run through `validation_pipeline.py` before returning
- Check hallucination_risk score (should be <0.3 for GFR)
- Log all low-confidence outputs (0.4–0.65) for review

## Domain-Specific Quality Requirements

### GFR (Government Financial Rules)
- Match exact regulation sections (e.g., "GFR 2017, Rule 149")
- Always provide budget category references
- Flag superseded rules with dates
- Confidence floor: 0.75

### Procurement Rules
- Include approval hierarchy levels
- Reference bidding procedure steps exactly
- Cite tender documentation links
- Confidence floor: 0.80

### Technical Documentation
- Allow conceptual explanations (not just factual)
- Can describe processes and their rationale
- Link to equipment specifications when applicable
- Confidence floor: 0.65

## Output Schema
```json
{
  "answer": "string",
  "confidence": 0.0-1.0,
  "sources": ["chunk_id_1", "chunk_id_2"],
  "domain": "GFR|Procurement|Technical",
  "hallucination_risk": 0.0-1.0,
  "citation_quality": 0.0-1.0,
  "latency_ms": integer
}
```
