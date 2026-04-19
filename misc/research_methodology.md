# SL-RAG / Trust-RAG: Complete Research Paper Methodology

> **System:** Secure, Layered Retrieval-Augmented Generation with Trust Scoring (SL-RAG / Trust-RAG)  
> **Domain:** ISRO Government Documentation QA  
> **Architecture:** 7-layer, fully offline RAG pipeline

---

## Table of Contents
1. [System Architecture](#1-system-architecture)
2. [All Formulas - Annotated with Sources](#2-all-formulas---annotated-with-sources)
3. [All Thresholds and Hyperparameters](#3-all-thresholds-and-hyperparameters)
4. [Pipeline Methodology - Layer by Layer](#4-pipeline-methodology---layer-by-layer)
5. [Evaluation Methodology (RAGAS)](#5-evaluation-methodology-ragas)
6. [Complete Bibliography Mapping](#6-complete-bibliography-mapping)
7. [Related Work Summary](#7-related-work-summary)

---

## 1. System Architecture

The pipeline is structured as **7 sequential layers**:

```
Query Input
|
|-- Layer 1: Document Ingestion & PII Anonymization
|     DocumentLoader -> PIIAnonymizer -> EncryptionManager
|
|-- Layer 2: Chunking, Embedding & Domain Detection
|     ChunkGenerator -> EmbeddingGenerator -> FAISS Index
|                    -> DocumentLevelDomainManager (K-means)
|
|-- Layer 3: Multi-Domain Hybrid Retrieval
|     BM25Retriever + FAISSIndexManager -> HybridRetriever
|     -> CrossEncoderReranker -> RetrievalPipeline
|     + ASIDetector (adversarial) + TrustScorer
|
|-- Layer 4: Prompt Construction & Answer Generation
|     PromptBuilder (injection detection) -> LLMGenerator
|
|-- Layer 5: Post-Generation Validation
|     ValidationPipeline (faithfulness, hallucination, confidence)
|
|-- Layer 6: Monitoring & Audit
|     MonitoringSystem (SHA-256 hash chain, SQLite)
|
|-- Layer 7: API & Response Delivery
      SLRAGPipeline orchestrator -> JSON / SSE output
```

**Key innovation:** The Trust-RAG module replaces raw retrieval scores with a composite multi-feature trust signal applied before final context selection.

---

## 2. All Formulas - Annotated with Sources

### 2.1 Cosine Similarity (Dense Retrieval & Semantic Features)

```
sim(q, d) = (h_q . h_d) / (||h_q|| * ||h_d||)
```

**Variables:**
- `h_q` = L2-normalized query embedding (768-dim, all-mpnet-base-v2)
- `h_d` = L2-normalized document chunk embedding

**Used in:**  
FAISS inner-product index (IP metric; L2 pre-normalized = cosine), TrustScorer semantic feature S, DomainManager query routing, answer-to-chunk consistency.

**Code:** `sl_rag/core/faiss_index.py` (IndexFlatIP), `sl_rag/retrieval/trust_scorer.py` (_semantic), `sl_rag/retrieval/document_level_domain_manager.py` (route_query)

**Implementation detail — mapped to [0,1]:**
```python
sim_normalized = (cosine + 1.0) / 2.0
```

**Source:**  
Reiners & Gurevych (2019) — *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks* — `papers/10_Reimers2019_Sentence_BERT.pdf`

---

### 2.2 BM25 Okapi Scoring

```
BM25(q, d) = SUM_{t in q} IDF(t) * f(t,d)*(k1+1) / [f(t,d) + k1*(1 - b + b*|d|/avgdl)]

where IDF(t) = log[(N - n_t + 0.5)/(n_t + 0.5) + 1]
```

**Variables:**
- `f(t, d)` = term frequency of term t in document d
- `|d|` = document length in tokens
- `avgdl` = average document length in the corpus
- `k1 = 1.5` — term frequency saturation (tunable)
- `b = 0.75` — length normalization (tunable)
- `N` = total documents; `n_t` = documents containing term t

**Code:** `sl_rag/retrieval/bm25_retriever.py` via library `rank_bm25.BM25Okapi`

**Source:**  
Robertson & Zaragoza (2009) — *The Probabilistic Relevance Framework: BM25 and Beyond* — `papers/06_Robertson2009_BM25.pdf`

---

### 2.3 Hybrid Retrieval — Weighted Score Fusion

```
FusedScore(d) = alpha * S_dense(d) + (1 - alpha) * S_BM25_norm(d)

S_BM25_norm(d) = (S_BM25(d) - S_min) / (S_max - S_min)   [min-max normalization]
```

**Variables:**
- `alpha = 0.7` — dense weight (configured in config.yaml)
- `S_dense` = cosine similarity score (already in [0,1])
- `S_BM25_norm` = min-max normalized BM25 score

**Code:** `sl_rag/retrieval/hybrid_retriever.py` (_weighted_fusion)

**Source:**  
Karpukhin et al. (2020) — *Dense Passage Retrieval for Open-Domain QA* — `papers/09_Karpukhin2020_Dense_Passage_Retrieval.pdf`

---

### 2.4 Hybrid Retrieval — Reciprocal Rank Fusion (RRF)

```
RRF(d) = (1 - alpha) / (k + rank_BM25(d)) + alpha / (k + rank_dense(d))

Default: k = 60
```

**Variables:**
- `k = 60` — smoothing constant (standard from literature)
- `rank_r(d)` = 1-indexed rank of document d in ranker r
- BM25 branch uses weight `(1 - alpha)`, Dense branch uses weight `alpha`

**Code:** `sl_rag/retrieval/hybrid_retriever.py` (_reciprocal_rank_fusion)

**Source:**  
Cormack, Clarke & Buettcher (2009); also surveyed in Gao et al. (2024) — `papers/23_Gao2024_RAG_LLM_Survey.pdf`

---

### 2.5 Cross-Encoder Score Calibration (Sigmoid)

```
score_calibrated(q, d) = 1 / (1 + exp(-z_qd))
```

**Variables:**
- `z_qd` = raw logit from cross-encoder (ms-marco-MiniLM-L-6-v2), range ~[-6, +6]
- Sigmoid maps raw logits to [0, 1] for consistency with cosine scores

**Code:** `sl_rag/retrieval/reranker.py` (line 104)

**Source:**  
Nogueira & Cho (2019) — *Passage Re-ranking with BERT.* HuggingFace model: `cross-encoder/ms-marco-MiniLM-L-6-v2`

---

### 2.6 Trust Score — Composite Multi-Feature Function [CORE FORMULA]

```
T(q, d) = sigmoid( w1*S + w2*C + w3*F + w4*I )

Weights: (w1, w2, w3, w4) = (0.4, 0.2, 0.2, 0.2)  [must sum to 1.0]

Features:
  S = Semantic similarity   = (cosine(q, d) + 1) / 2       in [0, 1]
  C = Source Credibility    = rule-lookup(d.metadata)       in [0.60, 0.95]
  F = Freshness             = exp(-lambda * days_elapsed)   in (0, 1]
  I = Consistency           = mean cosine(d, peers_norm)    in [0, 1]
```

**Code:** `sl_rag/retrieval/trust_scorer.py` (TrustScorer._compute_breakdown)

**Source:**  
Original Trust-RAG contribution. Concept informed by:
- Lewis et al. (2020) — `papers/01_Lewis2020_RAG_Knowledge_Intensive_NLP.pdf`
- Ji et al. (2023) — `papers/04_Ji2023_Hallucination_Survey.pdf`
- Ammann et al. (2024) — `papers/02_Ammann2024_Secure_RAG.pdf`  
- Design specification: `Trust-RAG.md` (Section 5, Module 4)

---

### 2.7 Freshness Decay (Exponential)

```
F(d) = exp(-lambda * delta_t)

lambda = 0.001 per day  =>  half-life ~= 693 days
delta_t = days since document creation date
Default (unknown date): F = 0.75
```

**Code:** `sl_rag/retrieval/trust_scorer.py` (_freshness)

**Source:**  
Exponential decay for content recency — standard in IR recency modeling. Defined in `Trust-RAG.md` (Section 5, Freshness feature).

---

### 2.8 Source Credibility Lookup Table

```
C(d) = lookup(source_keywords_in_metadata)

GFR / General Financial Rules  -->  0.95
Procurement Manuals             -->  0.90
Tender Documents                -->  0.88
Consultancy                     -->  0.87
Manual                          -->  0.85
Technical Report / Memo         -->  0.80
Report                          -->  0.75
Telemetry / SCADA               -->  0.72
Default (unknown)               -->  0.60
```

**Code:** `sl_rag/retrieval/trust_scorer.py` (_CREDIBILITY_RULES, _source_credibility)

**Design rationale:**  
Ordered by normative weight in Indian government: regulatory (GFR) > policy manuals > technical > observational.

---

### 2.9 Inter-Document Consistency Score

```
I(d_i) = (1 / (N-1)) * SUM_{j != i} [(cosine(h_di, h_dj) + 1) / 2]

Special case: if N = 1, I = 1.0 (trivially consistent)
```

**Code:** `sl_rag/retrieval/trust_scorer.py` (_consistency)

**Source:**  
Cross-document consistency as a hallucination signal — Ji et al. (2023) — `papers/04_Ji2023_Hallucination_Survey.pdf`

---

### 2.10 Numerically Stable Sigmoid

```
sigmoid(x) = 1 / (1 + exp(-x))   if x >= 0
           = exp(x) / (1 + exp(x)) if x < 0
```

**Code:** `sl_rag/retrieval/trust_scorer.py` (_sigmoid)

---

### 2.11 Adversarial Detection — Activation Shift Index (ASI)

```
ASI(q) = (1 / |A|) * SUM_{a in A} ||h_q - h_a||^2

Decision:
  if ASI(q) > tau (default: 2.5):  flag adversarial
  else:                             pass to pipeline

Threshold calibration:
  tau = percentile(ASI_scores(normal_queries), 95)
```

**Variables:**
- `h_q` = query embedding (768-dim float32)
- `A` = anchor set of <= 300 randomly subsampled corpus chunk embeddings
- `tau = 2.5` default; auto-calibrated at 95th percentile of normal queries

**Code:** `sl_rag/retrieval/adversarial_detector.py` (ASIDetector)

**Source:**
- Carlini et al. (2021) — *Extracting Training Data from LLMs* — `papers/07_Carlini2021_Extracting_Training_Data.pdf`
- Greshake et al. (2023) — *Indirect Prompt Injection* — `papers/36_Greshake2023_Indirect_Prompt_Injection.pdf`
- Zou et al. (2024) — *PoisonedRAG* — `papers/35_Zou2024_PoisonedRAG.pdf`

---

### 2.12 Prompt Injection Detection — Special Character Ratio

```
R_special(q) = |{c in q : c not in AlphaNum union SAFE_CHARS}| / max(|q|, 1)

SAFE_CHARS = {' ', '.', ',', '?', '!', "'", '-', '/', '(', ')'}

Decision: if R_special > 0.25  -->  flag injection (refusal prompt)
```

**Additionally:** 15 regex patterns checked (covers `[INST]`, `###`, `SYSTEM:`, `ignore previous instructions`, role-switching attempts, ChatML markers, etc.)

**Code:** `sl_rag/generation/prompt_builder.py` (INJECTION_PATTERNS, detect_injection)

**Source:**  
Greshake et al. (2023) — *Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection* — `papers/36_Greshake2023_Indirect_Prompt_Injection.pdf`

---

### 2.13 Confidence Score — Legacy Formula (Retrieval Only)

```
Conf_legacy = 0.75 * mean(top3_scores) + 0.25 * consistency_score

where:
  top3_scores     = top-3 cross-encoder output scores
  consistency     = cosine(embed(answer), embed(best_chunk))
```

**Code:** `sl_rag/validation/validation_pipeline.py` (_compute_confidence, branch: faithfulness < 0.0)

---

### 2.14 Confidence Score — Trust-RAG Formula (Faithfulness-Weighted)

```
Conf_TrustRAG = 0.5 * Faith + 0.3 * mean(top3_scores) + 0.2 * consistency_score

Condition: used when faithfulness_score >= 0.0
```

**Code:** `sl_rag/validation/validation_pipeline.py` (_compute_confidence)

**Source:**  
Es et al. (2023) — *RAGAS* — `papers/19_Es2023_RAGAS_Evaluation.pdf`; Ji et al. (2023) — `papers/04_Ji2023_Hallucination_Survey.pdf`

---

### 2.15 Faithfulness Score — Claim-Level Jaccard Similarity

```
Faith = supported_claims / total_claims

Claim support test:
  J(claim, ctx) = |tokens(claim) ∩ tokens(ctx)| / |tokens(claim) ∪ tokens(ctx)|
  Claim is SUPPORTED if: max_{ctx in context} J(claim, ctx) >= 0.15

Algorithm:
  1. Split answer into sentences (sentence length >= 4 words)
  2. Tokenize: regex [a-zA-Z]{3,}, remove 48-word stoplist
  3. For each claim, compute Jaccard vs each context chunk token set
  4. Mark claim supported if any J >= 0.15
  5. Faith = supported_count / total_claim_count
```

**Code:** `sl_rag/validation/validation_pipeline.py` (_compute_faithfulness)

**Source:**  
Es et al. (2023) — *RAGAS: Automated Evaluation of RAG Pipelines* — `papers/19_Es2023_RAGAS_Evaluation.pdf`

---

### 2.16 Hallucination Detection — Token Novelty + Retrieval Quality

```
novelty_ratio = |tokens(answer) - tokens(all_context)| / |tokens(answer)|

retrieval_quality = mean(top-3 cross-encoder scores)

Risk classification:
  if retrieval_quality < 0.20:                                    RISK = HIGH
  elif novelty_ratio > 0.45:                                      RISK = HIGH
  elif novelty_ratio > 0.28 OR
       (retrieval_quality < 0.35 AND novelty_ratio > 0.20):       RISK = MEDIUM
  else:                                                            RISK = LOW
```

**Token normalization:**
- Lowercase; light stemming: -ing (>5 chars), -ed (>4), -es (>4), -s (>3)
- Minimum 3 chars after stemming; remove from 48-word stoplist

**Code:** `sl_rag/validation/validation_pipeline.py` (_analyze_hallucination, _normalize_tokens, _light_stem)

**Source:**  
Ji et al. (2023) — *A Survey of Hallucination in NLG* — `papers/04_Ji2023_Hallucination_Survey.pdf`  
(Original specification: >30% novel words = high, >15% = medium; refined in implementation.)

---

### 2.17 Document-Level Embedding (Mean Pooling + L2 Normalization)

```
e_chunk_mean = (1/N) * SUM_{i=1}^{N} e_chunk_i
e_doc = e_chunk_mean / ||e_chunk_mean||
```

**Code:** `sl_rag/retrieval/document_level_domain_manager.py` (_compute_document_embeddings)

---

### 2.18 K-Means Clustering Objective

```
argmin_{C} SUM_{k=1}^{K} SUM_{d in C_k} ||e_d - mu_k||^2

K in [2, 10]; n_init=10; random_state=42
```

**Code:** `sl_rag/retrieval/document_level_domain_manager.py` (detect_domains, KMeans)

---

### 2.19 Silhouette Score + Chunk-Balance Adjusted Criterion

```
Raw silhouette:
  s(d) = (b(d) - a(d)) / max(a(d), b(d))
  S_mean = (1/N) * SUM_d s(d)

Chunk-balance penalty:
  f_max(K) = max_cluster_chunk_fraction for clustering K
  penalty(K) = max(0, (f_max(K) - 0.45) / (1 - 0.45))

Adjusted silhouette (optimization criterion):
  S_adjusted(K) = S_mean(K) * (1 - 0.50 * penalty(K))

Optimal K = argmax_{K in [2,10]} S_adjusted(K)
```

**Threshold: no cluster > 45% of chunks; balance_weight = 0.50**

**Code:** `sl_rag/retrieval/document_level_domain_manager.py` (_find_optimal_k)

**Source:**  
Rousseeuw (1987) — *Silhouettes: graphical aid to interpretation of cluster analysis*; `sklearn.metrics.silhouette_score`

---

### 2.20 Davies-Bouldin Score (Cluster Quality Metric)

```
DB = (1/K) * SUM_{i=1}^{K} max_{j != i} [(sigma_i + sigma_j) / d(mu_i, mu_j)]

Lower = better cluster separation
```

**Code:** `sl_rag/retrieval/document_level_domain_manager.py` (`sklearn.metrics.davies_bouldin_score`)

---

### 2.21 Entropy-Based Adaptive Routing Threshold

```
Step 1 - Softmax over domain cosine scores:
  p_d = exp(s_d) / SUM_{d'} exp(s_{d'})

Step 2 - Normalized Shannon entropy:
  H = (-SUM_d p_d * ln(p_d)) / ln(N)    [H in 0..1]

Step 3 - Adaptive threshold:
  tau = tau_base * (0.5 + 0.5 * H)
  Where tau_base = 0.3 (from config.yaml)

Behavior:
  H=1 (max uncertainty): tau = 0.3 * 1.0 = 0.30  [wider net]
  H=0 (max certainty):   tau = 0.3 * 0.5 = 0.15  [focused]
```

**Code:** `sl_rag/retrieval/document_level_domain_manager.py` (_compute_adaptive_threshold, route_query)

**Source:**  
Wang et al. (2024) — *LLM Query Routers* — `papers/20_Wang2024_LLM_Query_Routers.pdf`; Shannon (1948).

---

### 2.22 Domain Centroid (L2-Normalized Mean)

```
mu_domain = e_mean / ||e_mean||
where e_mean = (1/|D|) * SUM_{c in D} e_c
```

**Code:** `sl_rag/retrieval/document_level_domain_manager.py` (_compute_centroids)

---

### 2.23 Cosine Distance (Drift Detection)

```
d_cos(a, b) = 1 - (a . b) / (||a|| * ||b||)

Drift threshold: mean(d_cos_per_domain) > 0.15  -->  emit security event
```

**Code:** `sl_rag/monitoring/monitoring_system.py` (_cosine_distance, check_domain_drift)

---

### 2.24 SHA-256 Audit Hash Chain

```
h_t = SHA256(ts_t | op_t | det_t | h_{t-1})

Genesis: h_0 = "GENESIS"
Separator: '|' (pipe character)

Verification: re-compute from GENESIS; any modification breaks the chain
```

**Code:** `sl_rag/monitoring/monitoring_system.py` (_append_audit, verify_audit_chain)

**Standard:** NIST FIPS 180-4 (SHA-256). EXCLUSIVE SQLite transaction prevents concurrent chain forking.

---

### 2.25 Token Count Approximation

```
n_tokens_approx(text) = floor(word_count(text) * 1.3)
```

Factor 1.3 = deliberate overestimate; sub-word tokenizers produce ~1.3x tokens vs whitespace words.

**Code:** `sl_rag/core/chunk_generator.py` (_count_tokens)

---

### 2.26 Embedding-Based Domain Prototype Classification

```
confidence(domain | chunk) = max(0, cosine(e_chunk, mu_domain) - 0.3) / 0.7
```

Maps cosine similarity [0.3, 1.0] to confidence [0, 1]. Below 0.3 = zero confidence.

**Code:** `sl_rag/retrieval/domain_classifier.py` (_embedding_classify)

---

### 2.27 Combined Classification Confidence (Rule + Embedding Fusion)

```
c_combined = 0.6 * c_rule + 0.4 * c_embedding

Applied only when: both methods agree on same domain AND c_rule < 0.6
```

**Code:** `sl_rag/retrieval/domain_classifier.py` (classify)

---

### 2.28 FAISS Index Auto-Upgrade Rule

```
N < 1000:   Use IndexFlatIP (exact, brute-force)
N >= 1000:  Use IndexIVFFlat with nlist = max(4, floor(sqrt(N)))
```

**Code:** `sl_rag/core/faiss_index.py` (_maybe_upgrade_to_ivf)

**Source:** FAISS — Johnson et al. (2019) — *Billion-scale similarity search with GPUs*

---

### 2.29 L2-Distance to Similarity Conversion (FAISS L2 fallback)

```
score_L2(d) = 1 / (1 + ||q - d||_2)
```

**Code:** `sl_rag/core/faiss_index.py` (search, L2 branch)

---

## 3. All Thresholds and Hyperparameters

| Parameter | Value | Module | Rationale |
|-----------|-------|--------|-----------|
| Embedding dim | 768 | config.yaml | all-mpnet-base-v2 (Reimers 2019) |
| Chunk size | 512 tokens | config.yaml | Context window budget |
| Chunk overlap | 50 tokens | config.yaml | Boundary information preservation |
| Min chunk size | 100 tokens | config.yaml | Filter trivially small chunks |
| alpha (hybrid dense weight) | 0.7 | config.yaml | 70% dense, 30% BM25 |
| BM25 k1 | 1.5 | bm25_retriever.py | Standard Okapi BM25 |
| BM25 b | 0.75 | bm25_retriever.py | Standard Okapi BM25 |
| RRF k constant | 60 | hybrid_retriever.py | Standard RRF value |
| similarity_threshold | 0.5 | config.yaml | Min retrieval candidate score |
| top_k_candidates | 20 | config.yaml | Initial pool per method |
| top_k_final | 7 | config.yaml | After cross-encoder reranking |
| top_k_domains | 3 | config.yaml | Max domains routed per query |
| domain_similarity_threshold | 0.3 | config.yaml | Routing cutoff (base_tau) |
| min_clusters | 2 | config.yaml | K-means lower bound |
| max_clusters | 10 | config.yaml | K-means upper bound |
| K-means n_init | 10 | domain_manager.py | Multiple initializations |
| K-means random_state | 42 | domain_manager.py | Reproducibility |
| max_chunk_frac | 0.45 | domain_manager.py | No cluster > 45% of chunks |
| balance_weight | 0.50 | domain_manager.py | Silhouette balance penalty |
| sub-cluster silhouette cutoff | -0.1 | domain_manager.py | Accept split only if sil > -0.1 |
| prototype min_confidence | 0.75 | domain_classifier.py | High-confidence training samples |
| prototype min_samples | 10 | domain_classifier.py | Min samples per domain |
| classify confidence_threshold | 0.6 | domain_classifier.py | Accept rule-based result |
| embedding class lower bound | 0.3 | domain_classifier.py | Cosine cutoff for confidence map |
| Trust w1 (semantic) | 0.4 | trust_scorer.py | Primary relevance signal |
| Trust w2 (credibility) | 0.2 | trust_scorer.py | Source authority |
| Trust w3 (freshness) | 0.2 | trust_scorer.py | Recency |
| Trust w4 (consistency) | 0.2 | trust_scorer.py | Cross-document agreement |
| lambda_decay | 0.001/day | trust_scorer.py | Half-life ~= 693 days |
| freshness default (unknown) | 0.75 | trust_scorer.py | Neutral freshness proxy |
| credibility GFR | 0.95 | trust_scorer.py | Highest regulatory authority |
| credibility procurement | 0.90 | trust_scorer.py | Official policy docs |
| credibility tender | 0.88 | trust_scorer.py | Formal tender notices |
| credibility consultancy | 0.87 | trust_scorer.py | Consultancy guidelines |
| credibility manual | 0.85 | trust_scorer.py | Official manuals |
| credibility technical/memo | 0.80 | trust_scorer.py | Technical reports |
| credibility report | 0.75 | trust_scorer.py | General reports |
| credibility telemetry/SCADA | 0.72 | trust_scorer.py | Observational data |
| credibility default | 0.60 | trust_scorer.py | Unknown source baseline |
| ASI threshold | 2.5 | pipeline.py | Adversarial query flag |
| ASI max_anchors | 300 | adversarial_detector.py | Subsample for speed |
| ASI calibration percentile | 95 | adversarial_detector.py | 95th percentile of normal ASI |
| min_consistency_score | 0.7 | validation_pipeline.py | Min semantic consistency |
| faithfulness support_threshold | 0.15 | validation_pipeline.py | Min Jaccard for claim support |
| confidence validity threshold | 0.3 | validation_pipeline.py | Min valid confidence |
| novelty_ratio HIGH | > 0.45 | validation_pipeline.py | Ji et al. 2023 (30% -> refined) |
| novelty_ratio MEDIUM | > 0.28 | validation_pipeline.py | Intermediate risk band |
| retrieval_quality HIGH risk | < 0.20 | validation_pipeline.py | Insufficient retrieval quality |
| retrieval_quality MEDIUM | < 0.35 | validation_pipeline.py | Combined medium condition |
| special_char_ratio injection | > 0.25 | prompt_builder.py | Injection via dense punctuation |
| injection regex patterns | 15 | prompt_builder.py | Greshake et al. (2023) |
| max_context_tokens | 6000 | prompt_builder.py | LLM context window budget |
| words_per_token ratio | 0.75 | prompt_builder.py | Word-to-token approximation |
| LLM temperature | 0.3 | config.yaml | Focused / factual generation |
| LLM top_p | 0.9 | llm_generator.py | Nucleus sampling cutoff |
| LLM repetition_penalty | 1.1 | llm_generator.py | Anti-repetition |
| LLM max_new_tokens | 650 | config.yaml | Max generation length |
| LLM max_input_length | 8192 | llm_generator.py | Input truncation limit |
| LLM quantization | 4-bit NF4 double | llm_generator.py | BitsAndBytes; ~3 GB VRAM |
| Audit retention years | 7 | config.yaml | Government compliance |
| Domain drift threshold | 0.15 | monitoring_system.py | Cosine distance flag |
| Rate spike window | 60 seconds | monitoring_system.py | Query rate anomaly window |
| Rate spike threshold | 20 queries | monitoring_system.py | Max queries per window |
| Sensitive repeat threshold | 3 | monitoring_system.py | Repeated sensitive doc access |
| Secure delete passes | 3 | encryption_manager.py | Multi-pass random overwrite |
| BM25 min token length | 2 chars | bm25_retriever.py | Short token filter |
| Keyword high-confidence score | 3.0 pts | domain_classifier.py | Per-keyword match weight |
| Keyword medium-confidence score | 1.0 pts | domain_classifier.py | Per-keyword match weight |
| Pattern match score | 2.0 (cap 6.0) | domain_classifier.py | Per-regex pattern match |
| Section header match | 8.0 pts | domain_classifier.py | Structural signal (strongest) |
| Document title match | 5.0 pts | domain_classifier.py | Title indicator |
| Normalization denominator | 12.0 | domain_classifier.py | Score normalization cap |
| Context boost multiplier | 1.2x | domain_classifier.py | Document-level context propagation |
| Diminishing returns bonus limit | +3 per term | domain_classifier.py | Cap repeat-keyword bonus |

---

## 4. Pipeline Methodology - Layer by Layer

### Layer 1: Document Ingestion & Security

**Components:** DocumentLoader, PIIAnonymizer, EncryptionManager

1. **Document Loading:** Recursive PDF/text loading with pytesseract OCR fallback. Max file 200 MB, min text 100 chars.
2. **PII Anonymization (pre-embedding):** Regex patterns redact:
   - Aadhaar numbers (XXXX-XXXX-XXXX)
   - PAN cards (AAAAA9999A format)
   - Indian phone numbers (+91-XXXXXXXXXX, 10-digit 6-9 prefix)
   - Email addresses
   - Passport numbers (Indian: A1234567)
   - US SSN (XXX-XX-XXXX)
   - Credit cards (16-digit with separators)
   - IP addresses
   - Government employee IDs (EMP/ISRO/ID prefix)
   - Dates of birth
   - Optional: spaCy NER for person name detection
3. **Encryption:** AES-256 via Fernet (symmetric). Master key at `./storage/keys/master.key` (permissions 0o600). Encrypts FAISS index and monitoring SQLite database at rest.
4. **Secure Deletion:** 3-pass os.urandom() overwrite + fsync before unlink.

**Sources:** Ammann et al. (2024) `papers/02`; Zeng et al. (2024) `papers/33`; Morris et al. (2023) `papers/34`

---

### Layer 2: Chunking, Embedding & Domain Detection

**Components:** ChunkGenerator, EmbeddingGenerator, FAISSIndexManager, DocumentLevelDomainManager

1. **Sentence-Aware Chunking:** NLTK `sent_tokenize` splits at sentence boundaries. Chunks assembled until approx 512 tokens (*1.3 word multiplier). 50-token overlap. Min chunk 100 tokens.
2. **Embedding:** `sentence-transformers/all-mpnet-base-v2` (768-dim, float32, L2-normalized, batch=32, GPU CUDA when available).
3. **Document-Level K-Means:**
   - Document embedding = L2-normed mean of chunk embeddings
   - Optimal K in [2,10] via adjusted silhouette + chunk-balance penalty (Formulae 2.18-2.19)
   - Oversized clusters (>45% chunks, >=3 docs) sub-split via K=2
4. **TF-IDF Domain Naming:** Top n-grams (unigrams+bigrams, min_df=1) from cluster text; government stopwords filtered; top 2 keywords form domain name.
5. **FAISS Index:** IndexFlatIP (<1000 vectors); auto-upgrades to IndexIVFFlat (nlist=sqrt(N)) at >=1000 vectors.

**Sources:** Reimers (2019) `papers/10`; Johnson (2019) FAISS; Wu et al. (2024) `papers/15`

---

### Layer 3: Multi-Domain Hybrid Retrieval

**Components:** QueryPreprocessor, DocumentLevelDomainManager, BM25Retriever, HybridRetriever, CrossEncoderReranker, ASIDetector, TrustScorer

1. **ASI Adversarial Check:** Pre-retrieval; L2^2 distance to corpus anchors (Formula 2.11). Flags but does NOT block (logs security event).
2. **Query Preprocessing:** Acronym expansion, lowercase normalization.
3. **Entropy-Adaptive Domain Routing:** Softmax -> Shannon entropy -> adaptive tau (Formula 2.21). Top-3 domains selected.
4. **Per-Domain BM25:** Top-20 candidates from BM25Okapi (k1=1.5, b=0.75).
5. **Per-Domain Dense:** Top-20 candidates from FAISS IP search.
6. **Score Fusion:** Default weighted (alpha=0.7) or RRF (k=60) (Formulae 2.3, 2.4).
7. **Deduplication:** chunk_id-based unique result set.
8. **Cross-Encoder Reranking:** ms-marco-MiniLM-L-6-v2 + sigmoid calibration (Formula 2.5). Returns top-7.
9. **Trust Scoring:** T(q,d) = sigmoid(w*S + w*C + w*F + w*I) (Formula 2.6). Applied over generation_results subset.

**Sources:** Robertson (2009) `papers/06`; Karpukhin (2020) `papers/09`; Reimers (2019) `papers/10`; Gao (2024) `papers/23`

---

### Layer 4: Prompt Construction & Generation

**Components:** PromptBuilder, LLMGenerator

1. **Injection Detection:** 15 regex patterns + R_special > 0.25 (Formula 2.12). Returns refusal prompt if detected.
2. **Context-Aware Chunk Selection:** Budget-intent heuristic for GFR queries — matches rule numbers (25-31) and sanction/appropriation keywords.
3. **Rules Index:** Extracts and deduplicates rule references from chunk content; forms CONSTRAINT hint for LLM.
4. **Prompt Hierarchy (4-level, immutable):**
   ```
   SYSTEM:  [role + 8 behavioral constraints]
   CONTEXT: [N numbered chunks with (Score, Source, Chunk)]
   [RULES INDEX] (conditional)
   QUERY:   [user input]
   [CONSTRAINT: only cite listed rules] (conditional)
   ANSWER:
   ```
5. **LLM:** meta-llama/Llama-3.2-3B-Instruct (4-bit NF4 double quantization, BitsAndBytesConfig).
   - temperature=0.3, top_p=0.9, repetition_penalty=1.1, max_new_tokens=650
   - Stop tokens: QUERY:, CONTEXT:, SYSTEM:, ANSWER:, context headers `[N] (Score:`
   - Citation deduplication removes repeated `[Source: ...]` lines
   - Chat template applied when model supports it

**Sources:** Greshake (2023) `papers/36`; Liu (2024) `papers/03`; OpenAI (2023) `papers/21`

---

### Layer 5: Post-Generation Validation

**Component:** ValidationPipeline

1. **Citation Verification:** All chunks require valid doc_id and chunk_id. Extracted `[Source: doc_id, Chunk: N]` markers verified against context.
2. **Consistency:** cosine(embed(answer), embed(best_chunk)); threshold 0.7.
3. **Faithfulness:** Sentence-level Jaccard claim support (Formula 2.15); threshold 0.15.
4. **Confidence:** Faithfulness-weighted blend (Formula 2.14) or legacy (Formula 2.13).
5. **Hallucination risk:** Token novelty + retrieval quality (Formula 2.16).
6. **Validity gate:** consistency >= 0.7 AND citations_valid AND risk != HIGH.

**Sources:** Es et al. (2023) `papers/19`; Ji et al. (2023) `papers/04`; Huang et al. (2023) `papers/30`

---

### Layer 6: Monitoring & Governance

**Component:** MonitoringSystem, RBACManager

1. **SHA-256 Hash Chain:** EXCLUSIVE SQLite transaction prevents concurrent chain forking (Formula 2.24).
2. **Tables:** query_log, document_access_log, security_events, audit_trail, performance_metrics.
3. **Anomaly Detection:** Rate spikes (20 queries/60s), sensitive document abuse (3 repeats), unusual domain access pattern.
4. **Domain Drift:** Cosine distance of current vs baseline centroids > 0.15 (Formula 2.23).
5. **RBAC:** Roles define query and document-level access.

**Sources:** Ammann (2024) `papers/02`; Shokri (2017) `papers/08`

---

## 5. Evaluation Methodology (RAGAS)

### 5.1 RAGAS Metrics

| Metric | Definition | Computation |
|--------|-----------|-------------|
| **Faithfulness** | Claims in answer supported by retrieved context | LLM-judged claim extraction + NLI |
| **Answer Relevancy** | Answer directly addresses the question | Cosine(embed(generated_questions), embed(original_q)) |
| **Context Precision** | Retrieved chunks useful for ground truth | LLM-judged usefulness fraction |
| **Context Recall** | Ground truth info covered by retrieved chunks | LLM-judged coverage fraction |
| **Answer Correctness** | Factual correctness vs reference answer | Blended F1 + semantic similarity |
| **Answer Similarity** | Semantic closeness to reference | Cosine(embed(answer), embed(reference)) |

**Source:** Es et al. (2023) — `papers/19_Es2023_RAGAS_Evaluation.pdf`

### 5.2 Evaluation Setup

```
Judge LLM:            HuggingFaceH4/zephyr-7b-beta (free-tier fallback chain)
Embedding model:      all-MiniLM-L6-v2 (local, offline)
Dataset:              eval_dataset.json (curated; adversarial excluded)
judge_max_new_tokens: 1400
max_answer_chars:     1800  (clipping prevents judge context overflow)
max_context_chars:    1400  (per chunk, same reason)
judge_timeout:        420s; fallback: 600s with reduced max_new_tokens
max_retries:          8;
max_wait:             5400s
Samples:              20 non-adversarial queries
Domains covered:      GFR, Procurement (Goods/Consultancy), Technical/Telemetry
```

### 5.3 Internal Evaluation Metrics (Beyond RAGAS)

| Metric | Formula | Purpose |
|--------|---------|--------|
| ASI Score | Mean L2^2 distance to anchors | Adversarial detection confidence |
| Trust Score | sigmoid(w*S + w*C + w*F + w*I) | Composite source trustworthiness |
| Internal Faithfulness | Jaccard claim support ratio | Offline faithfulness proxy |
| Confidence | 0.5*Faith + 0.3*S_top3 + 0.2*consistency | Answer reliability score |
| Hallucination Risk | Novelty ratio + retrieval quality | Risk categorization (LOW/MEDIUM/HIGH) |
| Silhouette Score | Mean (b-a)/max(a,b) | Domain clustering quality |
| Davies-Bouldin | Mean max (sigma_i+sigma_j)/d | Domain separation quality |
| Latency (ms) | retrieval_ms + generation_ms | System performance |

---

## 6. Complete Bibliography Mapping

| # | Reference | Component(s) Using This Work |
|---|-----------|-----------------------------|
| 01 | Lewis et al. (2020) *RAG for Knowledge-Intensive NLP* | RAG foundation; retrieval-augmented generation; Trust-RAG extension concept |
| 02 | Ammann et al. (2024) *Secure RAG* | AES-256 encryption design; audit trail; security layer architecture |
| 03 | Liu et al. (2024) *Lost in the Middle* | Prompt chunk ordering; context window design; truncation strategy |
| 04 | Ji et al. (2023) *Hallucination Survey* | Hallucination thresholds (>30% novel = high); novelty_ratio; consistency; validation_pipeline |
| 05 | Zhang et al. (2023) *Understanding Neural Retrievers* | Dense retrieval behavior analysis |
| 06 | Robertson & Zaragoza (2009) *BM25* | BM25Okapi (k1=1.5, b=0.75); bm25_retriever.py |
| 07 | Carlini et al. (2021) *Extracting Training Data* | ASI adversarial detection threat model; privacy motivation |
| 08 | Shokri et al. (2017) *Membership Inference* | Privacy threat model; RBAC design; encryption motivation |
| 09 | Karpukhin et al. (2020) *DPR* | Dense passage retrieval; hybrid fusion motivation; score combination |
| 10 | Reimers & Gurevych (2019) *Sentence-BERT* | all-mpnet-base-v2 model; L2-normalized cosine similarity; 768-dim embeddings |
| 11 | Shojaee et al. (2025) *Federated RAG* | Multi-domain federated retrieval concept; domain isolation |
| 12 | Guu et al. (2020) *REALM* | Retrieval-augmented pre-training; retrieval motivation |
| 13 | Izacard & Grave (2021) *FiD* | Passage fusion for generative models |
| 14 | Khattab et al. (2022) *DSP* | Multi-hop reasoning with retrieval |
| 15 | Wu et al. (2024) *Multi-Source RAG* | Multi-domain retrieval architecture; domain routing |
| 16 | Min et al. (2021) *QA Contextualized KB* | Knowledge-grounded QA |
| 17 | Gao et al. (2022) *Zero-Shot Dense Retrieval* | Dense retrieval without supervision |
| 18 | Zheng et al. (2023) *LLM-as-Judge* | RAGAS judge LLM methodology; evaluation framework |
| 19 | Es et al. (2023) *RAGAS* | All 6 evaluation metrics; RAGAS framework; faithfulness concept; confidence scoring |
| 20 | Wang et al. (2024) *LLM Query Routers* | Entropy-based adaptive routing threshold (Formula 2.21) |
| 21 | OpenAI (2023) *GPT-4 Technical Report* | Prompt hierarchy patterns; instruction following design |
| 22 | Zhao et al. (2024) *RAG AI-Generated Content Survey* | RAG landscape; comparison baselines |
| 23 | Gao et al. (2024) *RAG LLM Survey* | Comprehensive RAG design patterns; RRF reference |
| 25 | Khattab et al. (2023) *ColBERT v2* | Late interaction retrieval reference |
| 27 | Wu et al. (2024) *Multi-Source RAG QA* | Domain-specific QA evaluation strategy |
| 28 | Wang et al. (2024) *LLM Resource Selectors (Federated)* | Federated resource selection |
| 29 | Wang et al. (2024) *FeB4RAG* | Federated search for RAG |
| 30 | Huang et al. (2023) *Hallucination LLM Survey* | Hallucination taxonomy; validation design |
| 31 | Asai et al. (2023) *Self-RAG* | Adaptive retrieval; self-reflection prompting motivation |
| 32 | Xu et al. (2024) *Knowledge Conflicts LLM* | Context vs parametric knowledge conflicts |
| 33 | Zeng et al. (2024) *Privacy RAG* | Privacy-preserving RAG; PII handling design |
| 34 | Morris et al. (2023) *Text Embeddings Privacy* | Embedding inversion attacks; AES-256 encryption motivation |
| 35 | Zou et al. (2024) *PoisonedRAG* | Adversarial corpus injection; ASI detection motivation |
| 36 | Greshake et al. (2023) *Indirect Prompt Injection* | 15 injection patterns; special_char_ratio > 0.25 threshold |

---

## 7. Related Work Summary

### 7.1 RAG Fundamentals
Lewis et al. (2020) introduced the RAG paradigm: a non-parametric retriever indexes a corpus; at query time, retrieved documents are prepended to an LLM prompt. REALM (Guu et al., 2020) and FiD (Izacard, 2021) extend this to pre-training and decoder-side fusion. Our system implements pragmatic RAG as government QA infrastructure, adding trust calibration and adversarial resilience as first-class requirements.

### 7.2 Retrieval Methods
BM25 (Robertson, 2009) provides efficient lexical matching via inverse document frequency. DPR (Karpukhin, 2020) and Sentence-BERT (Reimers, 2019) deliver semantic retrieval via bi-encoder architectures producing dense vectors. ColBERT (Khattab, 2023) adds per-token late interaction scoring. Hybrid BM25+dense retrievers with score fusion outperform either method alone. Our pipeline extends hybridization with trust-aware re-ranking beyond standard cross-encoder re-scoring.

### 7.3 Multi-Domain and Federated RAG
Multi-source RAG (Wu et al., 2024) and federated search (Wang et al., 2024 FeB4RAG; Shojaee et al., 2025) address knowledge distributed across heterogeneous corpora. Our entropy-adaptive routing (Wang et al., 2024 Query Routers; Shannon entropy) automatically balances domain breadth vs. focus without manual threshold tuning.

### 7.4 Hallucination and Validation
Ji et al. (2023) and Huang et al. (2023) survey hallucination in NLG models. Self-RAG (Asai, 2023) introduces retrieval tokens and self-critique. Knowledge conflicts between parametric memory and retrieved context are studied by Xu et al. (2024). Our offline validation pipeline implements lightweight proxies (token novelty + Jaccard faithfulness + consistency scoring) that operate without secondary LLM calls, maintaining the offline constraint.

### 7.5 Security and Privacy
Ammann et al. (2024) formalize security threats in RAG including data poisoning and inference attacks. Greshake et al. (2023) categorize indirect prompt injection. Carlini et al. (2021) demonstrate training data extraction, motivating PII anonymization before embedding. Shokri et al. (2017) define membership inference attacks, motivating RBAC and encrypted storage. Our ASI detector adds embedding-space anomaly detection complementing rule-based injection filtering.

### 7.6 Evaluation
Es et al. (2023) RAGAS defines both reference-based (faithfulness, correctness) and reference-free (relevancy, precision) automated metrics for RAG pipelines. Zheng et al. (2023) establish LLM-as-judge as scalable evaluation. Our evaluation uses both: RAGAS for end-to-end quality assessment and internal metrics (ASI, trust scores, faithfulness, confidence, hallucination risk) for security and robustness profiling.

---

## Appendix: System Configuration Reference (config/config.yaml)

```yaml
models:
  embedding_model:      "sentence-transformers/all-mpnet-base-v2"
  embedding_dimension:  768
  cross_encoder_model:  "cross-encoder/ms-marco-MiniLM-L-6-v2"
  llm_model:            "meta-llama/Llama-3.2-3B-Instruct"
  llm_quantization:     "4bit"  # NF4 double-quantized (BitsAndBytesConfig)

chunking:
  chunk_size:           512   # tokens (approx = words * 1.3)
  overlap:              50    # tokens
  min_chunk_size:       100   # tokens

retrieval:
  alpha:                0.7   # dense weight in weighted hybrid fusion
  similarity_threshold: 0.5   # min retrieval score for candidate acceptance
  top_k_candidates:     20    # initial retrieval pool per method
  top_k_final:          7     # post cross-encoder reranking

domain:
  min_clusters:         2
  max_clusters:         10
  similarity_threshold: 0.3   # tau_base for adaptive routing
  top_k_domains:        3     # max domains routed per query
  adaptive_threshold:   true  # entropy-based adaptive routing

llm:
  temperature:          0.3
  max_new_tokens:       650
  streaming:            true  # SSE token streaming

audit:
  retention_years:      7
  hash_chain_enabled:   true

# Configured in pipeline.py (no config.yaml entry):
# TrustScorer weights:  (0.4, 0.2, 0.2, 0.2)
# TrustScorer lambda:   0.001 per day
# ASI threshold:        2.5
# ASI max_anchors:      300
```
