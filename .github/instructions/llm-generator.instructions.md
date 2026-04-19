---
description: "Use when: tuning LLM generation parameters, optimizing token budgets, improving answer quality, reducing hallucinations"
applyTo: "sl_rag/generation/llm_generator.py"
---

# LLM Generator Optimization

## Temperature Tuning Per Domain

### GFR (Government Financial Rules)
- **Temperature**: 0.2 (strict, regulatory, no creativity)
- **Max new tokens**: 256
- **Top_p**: 0.85 (narrow diversity)
- **Reasoning**: Financial rules require exactness, zero paraphrasing

### Procurement Rules
- **Temperature**: 0.3 (procedural but can explain steps)
- **Max new tokens**: 300
- **Top_p**: 0.90
- **Reasoning**: Process-oriented, structured explanations needed

### Technical Documentation
- **Temperature**: 0.4 (can explore concepts and rationale)
- **Max new tokens**: 350
- **Top_p**: 0.95
- **Reasoning**: Conceptual flexibility while maintaining accuracy

## Token Budget Strategy

**Total context window**: 1536 tokens
**Allocation**:
- System prompt: 100 tokens
- Few-shot examples: 400 tokens
- Retrieved context: 600 tokens
- **Reserve for generation**: 256 tokens
- **Safety buffer**: 180 tokens (never use)

### Early Stopping Rules
- Stop on 2+ repeated phrases (common sign of looping)
- Stop if token count exceeds (max_tokens - 50)
- Stop on domain-specific tokens (e.g., "END_OF_DOCUMENT")
- Stop on repetition of queries (sign of confusion)

## Output Quality Metrics

### Citation Density
- Minimum: 1 citation per 30 words
- For factual claims: 1 citation per 20 words
- For explanations: 1 citation per 40 words
- Unmatched citations: Automatic rejection

### Confidence Score Reporting
- Always output numeric 0.0-1.0
- Calibrated to actual accuracy (0.8 confidence = 80% correct historically)
- Never output placeholder values
- Use calibration dataset from `validation_pipeline.py`

### Hallucination Signal Detection
- Unmatched pronouns (e.g., "they said" without subject)
- Fabricated examples not in sources
- Contradictions with previous outputs in session
- Named entities not in source chunks
- Numerical claims without ranges/citations

## Parameter Optimization Per Query Type

### Factual/Regulatory
```
GenerationConfig:
  temperature: 0.2
  top_p: 0.85
  top_k: 40
  repetition_penalty: 1.2
  length_penalty: 0.8
  max_new_tokens: 256
```

### Explanatory/Procedural
```
GenerationConfig:
  temperature: 0.35
  top_p: 0.90
  top_k: 50
  repetition_penalty: 1.15
  length_penalty: 0.85
  max_new_tokens: 300
```

### Exploratory/Conceptual
```
GenerationConfig:
  temperature: 0.45
  top_p: 0.95
  top_k: 60
  repetition_penalty: 1.1
  length_penalty: 0.9
  max_new_tokens: 350
```

## Monitoring & Logs

### Log on Every Generation
- Query hash (for deduplication)
- Selected parameters (temp, top_p)
- Generated tokens count
- Actual latency (ms)
- Hallucination score (0.0-1.0)
- Confidence score calibration

### Alert Thresholds
- If hallucination_score > 0.5: Log as WARNING
- If confidence < 0.4: Log as REVIEW_NEEDED
- If latency > 5000ms: Log as PERFORMANCE_ISSUE
- If unmatched citations > 0: Log as CRITICAL

## Multi-Turn Refinement

For complex queries requiring follow-up:
1. Store conversation context (last 3 exchanges)
2. Maintain domain consistency across turns
3. Update confidence scores based on follow-up accuracy
4. Limit conversation depth to 3 exchanges (token budget)

## Post-Generation Validation

- [ ] Check all citations exist in retrieved chunks
- [ ] Verify answer fits token limit
- [ ] Confirm confidence is calibrated (0.0-1.0)
- [ ] Check hallucination detection passed
- [ ] Validate no prompt injection artifacts
- [ ] Ensure domain consistency maintained
