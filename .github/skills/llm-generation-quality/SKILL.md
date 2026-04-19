---
name: llm-generation-quality
description: "Use when: tuning LLM generation parameters, optimizing token budgets, improving answer quality, reducing hallucinations, calibrating confidence scores"
---

# LLM Generation Quality Optimization

## When to Use This Skill

- Tuning temperature and sampling parameters per domain
- Optimizing token budgets for Llama 3.2-3B
- Implementing hallucination detection and prevention
- Calibrating confidence scores to actual accuracy
- Improving generation latency and output consistency
- Post-generation validation and quality checks

## Core Principles

### 1. Temperature is Domain-Specific
Not all queries need the same "creativity":
- **GFR (0.2)**: Strict legal rules, zero paraphrasing
- **Procurement (0.3)**: Procedural, can explain steps
- **Technical (0.4)**: Conceptual flexibility allowed

### 2. Token Budget is Inviolable
Llama 3.2-3B has 2048 token context; we use 1536:
- System + Few-shot + Context = 1544 tokens fixed
- Generation reserve = 256 tokens
- Safety buffer = 180 tokens (never crossed)
- Total = 1980 tokens (safe margin)

### 3. Confidence Scores Must Be Calibrated
Confidence isn't a guess—it's measured accuracy:
- Train on historical validation results
- 0.8 confidence should mean ~80% accurate historically
- Track confidence vs actual accuracy over time
- Adjust calibration quarterly

## Parameter Tuning Reference

### Conservative (GFR/Legal Documents)
```python
GenerationConfig(
    temperature=0.2,
    top_p=0.85,
    top_k=40,
    repetition_penalty=1.2,
    length_penalty=0.8,
    max_new_tokens=256,
    do_sample=True,
    early_stopping=True
)
```

### Balanced (Procurement/Procedural)
```python
GenerationConfig(
    temperature=0.35,
    top_p=0.90,
    top_k=50,
    repetition_penalty=1.15,
    length_penalty=0.85,
    max_new_tokens=300,
    do_sample=True,
    early_stopping=True
)
```

### Exploratory (Technical/Conceptual)
```python
GenerationConfig(
    temperature=0.45,
    top_p=0.95,
    top_k=60,
    repetition_penalty=1.1,
    length_penalty=0.9,
    max_new_tokens=350,
    do_sample=True,
    early_stopping=True
)
```

## Early Stopping Heuristics

Stop generation if:
1. **Repetition Detected**: Same phrase appears 2+ times
2. **Token Limit**: Count reaches (max_tokens - 50)
3. **Domain Signal**: Sees "[END_OF_DOCUMENT]" or similar
4. **Query Echo**: Model repeats back the query verbatim
5. **Confidence Collapse**: Internal confidence < 0.3 mid-generation

## Hallucination Detection Rules

Red flags that indicate fabrication:

| Pattern | Example | Signal |
|---------|---------|--------|
| Unmatched pronouns | "They said..." (no subject) | Fabricated dialogue |
| Fabricated examples | "Like when ISRO did..." (not in docs) | False specificity |
| Contradictions | Conflicts with earlier outputs | Inconsistency |
| OOV entities | Equipment name not in specs | Hallucinated product |
| Numerical ranges | "Budget is ₹50-100 lakhs" (exact number never stated) | False precision |

**Mitigation**: Train validation pipeline to score hallucination_risk (0.0-1.0) on every output.

## Confidence Calibration Framework

### Method: Historical Accuracy Tracking
```python
# For each generation:
logs = {
    'confidence': 0.82,
    'answer': '...',
    'validation_accurate': True,  # Later verified
}

# Monthly recalibration:
by_confidence_bin = group_by_bin(logs['confidence'], bin_size=0.1)
actual_accuracy = compute_accuracy(by_confidence_bin)
# If confidence=0.8 bin has accuracy=0.75, apply calibration correction
```

### Calibration Targets by Domain
- **GFR**: 0.85 confidence = 0.85 actual accuracy
- **Procurement**: 0.80 confidence = 0.80 actual accuracy
- **Technical**: 0.75 confidence = 0.75 actual accuracy

### Confidence Thresholds
- 0.95+: Publish immediately (high confidence + high accuracy)
- 0.80-0.94: Publish with minor caveat
- 0.65-0.79: Publish with "partially answered" flag
- 0.40-0.64: Mark as "REVIEW_NEEDED", don't publish automatically
- <0.40: Reject, return "I cannot provide a confident answer"

## Citation Quality Scoring

### Citation Density Rules
Minimum citations required:
- **Factual claims**: 1 citation per 15-20 words
- **Procedural steps**: 1 citation per 25 words
- **Explanations**: 1 citation per 40 words

### Citation Format Validation
```python
def validate_citations(answer_text, retrieved_chunks):
    """Check that every in-text [chunk_id] exists in retrieved_chunks"""
    cited_ids = extract_chunk_citations(answer_text)
    available_ids = set(c['id'] for c in retrieved_chunks)
    
    unmatched = cited_ids - available_ids
    if unmatched:
        score = 1.0 - (len(unmatched) / len(cited_ids))
        return score, unmatched
    return 1.0, []
```

## Output Quality Checklist

Before returning response from `llm_generator.generate()`:

```python
quality_checks = {
    'citation_density': validate_citation_density(answer),
    'citation_existence': validate_all_citations_exist(answer, chunks),
    'token_count': len(tokenize(answer)) < 256,
    'confidence_valid': 0.0 <= confidence <= 1.0,
    'hallucination_score': < 0.3,
    'no_prompt_injection': scan_for_injection(answer),
    'domain_consistency': answer_domain == query_domain,
}

if not all(quality_checks.values()):
    log_quality_failure(quality_checks, answer)
    # Option: regenerate or reject
```

## Latency Optimization

### Target Latencies
- GFR queries: < 2.5 seconds (strict model, quick decision)
- Procurement: < 3.0 seconds (moderate reasoning)
- Technical: < 3.5 seconds (more exploration allowed)

### Profiling Breakdown
```
- Embedding query: 200-300ms
- Retrieval (FAISS): 100-200ms
- Reranking: 150-300ms
- LLM generation: 1800-2500ms (Largest component)
- Validation: 100-200ms
─────────────────────────────────
Total: 2.5-3.5 seconds
```

### Optimization Levers
1. Reduce token generation budget (trade-off: answer quality)
2. Use smaller rerank batch (trade-off: retrieval accuracy)
3. Cache embeddings (if repetitive queries)
4. Profile bottleneck component first

## Monitoring & Alerting

### Log Every Generation
Track these metrics:
```
{
  "timestamp": "2026-03-18T10:30:45Z",
  "query_hash": "abc123def456",
  "domain": "GFR",
  "user_role": "analyst",
  "temperature_used": 0.2,
  "max_tokens": 256,
  "tokens_generated": 187,
  "latency_ms": 2340,
  "confidence": 0.87,
  "hallucination_score": 0.12,
  "citation_quality": 0.95,
  "post_validation_passed": true
}
```

### Alert Thresholds (Log & Escalate)
- Hallucination > 0.5: WARNING
- Confidence < 0.4: REVIEW_NEEDED
- Latency > domain_target + 2s: PERFORMANCE_ISSUE
- Unmatched citations: CRITICAL
- Citation density < minimum: QUALITY_ISSUE

## Domain-Specific Generation Tips

### For GFR (Government Financial Rules)
- Use exact rule numbers and sections
- Never paraphrase monetary amounts
- Preserve acronyms exactly (RFD, RE, etc.)
- Flag superseded rules
- Use 0.2 temperature (strict)

### For Procurement
- Explain bidding steps in order
- Include approval hierarchy
- Reference document types (RFQ, Expression of Interest, etc.)
- Use 0.3 temperature (procedural)

### For Technical
- Can use analogies to explain concepts
- Include specifications and parameters
- Reference drawings/diagrams exists
- Use 0.4 temperature (exploratory)
