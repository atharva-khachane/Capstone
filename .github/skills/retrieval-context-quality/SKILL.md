---
name: retrieval-context-quality
description: "Use when: improving chunk ranking and relevance, optimizing reranker sensitivity, managing similarity thresholds per domain, improving context window organization"
---

# Retrieval Context Quality Optimization

## When to Use This Skill

- Fine-tuning reranker sensitivity for different domains
- Optimizing semantic similarity thresholds
- Improving chunk boundary strategies
- Organizing context windows for maximum relevance
- Deduplicating semantically similar results
- Handling multi-domain query routing

## Core Principle: 70% Dense + 30% Sparse

Hybrid retrieval combines:
- **70% Dense (FAISS)**: Semantic similarity, captures intent
- **30% BM25**: Keyword matching, exact term hits

This weighting helps:
- Dense catches semantic variations ("budget sanction" ≈ "fund allocation")
- Sparse catches regulatory exact matches ("Rule 149", "Section 4")

## Reranker Sensitivity Tuning

### What is Reranking?
Cross-encoder model (`ms-marco-MiniLM`) re-scores top-k results to improve ranking.

### When to Be Strict (High Threshold)
- **GFR queries**: Keep only reranker score > 0.7
- **Procurement procedure**: Keep only > 0.65
- **Sensitive policies**: Keep only > 0.75

### When to Be Permissive (Low Threshold)
- **Technical queries**: Allow > 0.4
- **Exploratory queries**: Allow > 0.35
- **Definitional queries**: Allow > 0.5

### Sensitivity Tuning Template
```python
def get_reranker_threshold(domain, query_type, confidence_level):
    """
    domain: 'GFR', 'Procurement', 'Technical'
    query_type: 'factual', 'procedural', 'conceptual'
    confidence_level: 'strict', 'balanced', 'exploratory'
    """
    thresholds = {
        ('GFR', 'factual', 'strict'): 0.75,
        ('GFR', 'factual', 'balanced'): 0.65,
        ('GFR', 'procedural', 'strict'): 0.70,
        
        ('Procurement', 'procedural', 'strict'): 0.70,
        ('Procurement', 'procedural', 'balanced'): 0.60,
        ('Procurement', 'factual', 'strict'): 0.75,
        
        ('Technical', 'conceptual', 'exploratory'): 0.40,
        ('Technical', 'conceptual', 'balanced'): 0.50,
        ('Technical', 'factual', 'strict'): 0.65,
    }
    return thresholds.get((domain, query_type, confidence_level), 0.50)
```

## Chunk Boundary Strategies

### Strategy 1: Sentence-Aligned (Default)
- Split at sentence boundaries
- Pros: Grammatically complete, no mid-sentence breaks
- Cons: Variable chunk sizes
- **When to use**: Most government documents with clear sentence structure

### Strategy 2: Paragraph-Aligned
- Keep entire paragraphs together
- Pros: Preserves argumentative flow
- Cons: Longer chunks, fewer retrieval precision
- **When to use**: Technical documentation with conceptual paragraphs

### Strategy 3: Section-Aligned
- Keep entire sections (headings + content)
- Pros: Full context for regulation sections
- Cons: Very long chunks, less granular retrieval
- **When to use**: GFR rules (each rule is a unit)

**Current setting**: 512 tokens, 50 token overlap, sentence-aligned. This is good for most use cases.

## Semantic Similarity Threshold Tuning

### For Domain Deduplication
Remove near-duplicate chunks (cosine similarity > threshold):

```python
def remove_semantic_duplicates(chunks, threshold=0.85):
    """
    If two chunks have cosine similarity > threshold, keep only highest-scoring
    """
    embeddings = [c['embedding'] for c in chunks]
    similarity_matrix = cosine_similarity(embeddings, embeddings)
    
    keep_indices = []
    for i, chunk in enumerate(chunks):
        is_duplicate = False
        for j in keep_indices:
            if similarity_matrix[i][j] > threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            keep_indices.append(i)
    
    return [chunks[i] for i in keep_indices]
```

**Recommended threshold by domain**:
- GFR: 0.88 (strict dedup, regulations are specific)
- Procurement: 0.85 (moderate dedup, procedures vary)
- Technical: 0.80 (loose dedup, allow similar concepts)

## Context Window Organization

### Golden Order for LLM Prompt

```
Position 1 (Most relevant): Highest reranker score + highest domain match
Position 2: Second highest score, complements Position 1
Position 3: Third highest OR fills knowledge gap if Pos 1-2 are narrow
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output to prompt in THIS ORDER
```

### Metadata Tags to Include
```json
{
  "chunk_id": "GFR_2017_Ch3_Rule149_p45",
  "reranker_score": 0.92,
  "domain": "GFR",
  "topic": "Budget Sanctions",
  "confidence_signal": "exact_rule_match",
  "page_reference": "45-46",
  "section": "Rule 149"
}
```

## Multi-Domain Query Routing

### Routing Algorithm

When query touches multiple domains (e.g., "What budget can procurement use?"):

```python
def route_query_to_domains(query_text, available_domains):
    """
    1. Classify query into relevant domains
    2. Allocate retrieval budget by domain importance
    3. Merge results, re-rank by cross-domain relevance
    """
    
    domain_scores = domain_classifier(query_text)  # GFR: 0.8, Procurement: 0.6, Tech: 0.1
    
    # Allocate top-k per domain proportionally
    domain_topk = {
        'GFR': int(20 * 0.8 / (0.8+0.6)),  # 12 results
        'Procurement': int(20 * 0.6 / (0.8+0.6)),  # 8 results
    }
    
    # Retrieve from each domain
    results_gfr = retriever.retrieve('GFR', query_text, top_k=12)
    results_procurement = retriever.retrieve('Procurement', query_text, top_k=8)
    
    # Merge and re-rank
    merged = results_gfr + results_procurement
    reranked = reranker.rerank(query_text, merged)
    
    return reranked[:5]  # Return top 5
```

## Conflict Resolution in Retrieved Chunks

If two chunks state contradictory information:

```python
def detect_and_flag_conflicts(chunks):
    """
    1. Extract factual claims from each chunk
    2. Check for contradictions (entity mentions + numeric values)
    3. Flag for human review
    """
    for i, chunk_i in enumerate(chunks):
        for j, chunk_j in enumerate(chunks[i+1:]):
            entities_i = extract_entities(chunk_i['text'])
            entities_j = extract_entities(chunk_j['text'])
            
            conflicts = find_conflicting_claims(entities_i, entities_j)
            if conflicts:
                log_conflict({
                    'chunk_i': chunk_i['id'],
                    'chunk_j': chunk_j['id'],
                    'conflicts': conflicts,
                    'resolution': 'FLAG_FOR_AUDIT'
                })
                # Reduce reranker score for both chunks
                chunk_i['conflict_penalty'] = 0.2
                chunk_j['conflict_penalty'] = 0.2
```

## Context Window Token Budget Allocation

Total 1536 tokens for context:

### Allocation Strategy 1: Equal Distribution (3 chunks)
```
Chunk 1 (relevant): 450 tokens
Chunk 2 (supporting): 400 tokens
Chunk 3 (reference): 300 tokens
Reserve: 86 tokens
━━━━━━━━━━━━━━
Total: 1236 tokens usable, 300 for LLM generation, 180 buffer
```

### Allocation Strategy 2: Skewed (1 Strong Context)
```
Chunk 1 (highly relevant): 800 tokens
Chunk 2 (secondary): 300 tokens
Reserve: 136 tokens
━━━━━━━━━━━━━━
Total: 1100 tokens usable
(Better for very specific queries with one strong source)
```

**When to use**:
- **Equal**: Multi-faceted queries needing multiple perspectives
- **Skewed**: Narrow, specific queries with one clear source

## Quality Metrics to Track

### Per-Query Retrieval Metrics
```
{
  "query_hash": "abc123",
  "domain": "GFR",
  
  "dense_top_k": 20,        # FAISS results
  "sparse_top_k": 20,       # BM25 results
  "hybrid_merged": 20,      # After fusion
  
  "reranker_threshold": 0.65,
  "reranker_top_k": 5,
  
  "dedup_threshold": 0.85,
  "final_chunks": 3,
  "final_avg_reranker_score": 0.78,
  
  "conflict_detected": false,
  "latency_ms": 245
}
```

### Domain-Specific Benchmarks
| Metric | GFR | Procurement | Technical |
|--------|-----|-------------|-----------|
| Reranker threshold | 0.70 | 0.65 | 0.50 |
| Dedup similarity | 0.88 | 0.85 | 0.80 |
| Final chunks (avg) | 3 | 3 | 3 |
| Avg score (final) | >0.80 | >0.75 | >0.65 |

## Optimization Checklist

- [ ] Reranker configured per domain (not one-size-fits-all)
- [ ] Chunk boundaries aligned with document structure
- [ ] Similarity thresholds calibrated on validation set
- [ ] Context ordering prioritizes relevance
- [ ] Conflict detection enabled
- [ ] Token allocation matches query type
- [ ] Metadata tags preserved through pipeline
- [ ] Latency target met (<300ms for retrieval)
