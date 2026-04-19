# SL-RAG / Trust-RAG (ISRO Document Intelligence): Paper-Ready Project Walkthrough

## 0) What this repository is
This repository implements **SL-RAG / Trust-RAG**, a **fully offline**, security-aware Retrieval-Augmented Generation system for question answering over government and technical PDFs. It includes:

- A Python package implementing a **7-layer RAG pipeline** (ingestion → retrieval → generation → validation → monitoring/audit).
- A **FastAPI backend** exposing the pipeline via REST and SSE streaming.
- A **React + Vite + TypeScript frontend** implementing login, chat with streaming, citations, document management, and audit views.
- Evaluation scripts (local metrics + RAGAS), demo tests, and saved evaluation artifacts.

Core source package: `sl_rag/`

---

## 1) Research framing (problem, goals, contributions)

### Problem setting
Government and mission-critical technical documents require:
- High **faithfulness** (answers must be supported by retrieved evidence)
- Strong **security and governance** (prompt injection resistance, role-based access control, tamper-evident logs)
- **Offline operation** (no dependence on remote APIs for core functionality)

### Goals
1. Build an offline RAG system over PDFs.
2. Improve retrieval quality via **hybrid sparse+dense** retrieval and cross-encoder reranking.
3. Add Trust-RAG controls: **adversarial query detection**, **domain routing**, and **trust scoring** for evidence selection.
4. Provide **auditable** operation: monitoring, integrity checks, and security event logging.

### Claimed contributions (mapped to implementation)
- **Layered offline pipeline**: implemented in `sl_rag/pipeline.py`.
- **Hybrid retrieval + reranking**: implemented in `sl_rag/retrieval/`.
- **Trust-aware evidence selection**: trust scoring and adversarial checks in `sl_rag/retrieval/trust_scorer.py` and `sl_rag/retrieval/adversarial_detector.py`.
- **Prompt injection defenses**: `sl_rag/generation/prompt_builder.py`.
- **Tamper-evident audit trail**: `sl_rag/monitoring/monitoring_system.py`.

---

## 2) System architecture (7-layer pipeline)

### End-to-end flow
At a high level:

1. **Ingestion**: load PDFs, sanitize, optionally OCR, remove PII.
2. **Chunking**: sentence-aware chunking with overlap.
3. **Embedding + indexing**: dense embeddings + FAISS vector index; sparse BM25 index.
4. **Domain routing**: cluster documents/chunks into domains; route queries to top domains.
5. **Retrieval**: hybrid (dense + BM25) fusion; dedup; rerank via cross-encoder.
6. **Trust + validation**: compute trust features, check hallucination risk/faithfulness.
7. **Delivery**: return JSON response and optionally stream answer tokens via SSE.

### Backend + frontend interaction
- Frontend uses **REST** for login, query, and admin functions.
- Frontend uses **SSE-style streaming** for token-by-token answer display.

---

## 3) Implementation mapping: pipeline modules

### 3.1 Orchestrator
- `sl_rag/pipeline.py`
  - `ingest()` wires ingestion, chunking, embedding, indexing, and domain detection.
  - `query()` enforces RBAC, runs adversarial checks, retrieves evidence, optionally generates, validates, computes metrics, and logs.
  - `query_stream()` streams tokens then returns final metadata payload.

### 3.2 Core ingestion + storage
- `sl_rag/core/document_loader.py`: PDF extraction (PyMuPDF) + OCR fallback; doc metadata; SHA-256 IDs.
- `sl_rag/core/pii_anonymizer.py`: India-specific PII patterns (Aadhaar/PAN/phone/email/etc.) with optional NER.
- `sl_rag/core/chunk_generator.py`: sentence-aware chunking + overlap; approximate token counts.
- `sl_rag/core/embedding_generator.py`: sentence-transformers embedding generation; GPU/CPU control.
- `sl_rag/core/faiss_index.py`: FAISS index creation/search; encrypted persistence support.
- `sl_rag/core/encryption_manager.py`: symmetric encryption utilities (Fernet) used for storage artifacts.
- `sl_rag/core/schemas.py`: `Document` and `Chunk` dataclasses.

### 3.3 Retrieval + Trust
- `sl_rag/retrieval/query_preprocessor.py`: query normalization + acronym expansion (GFR/QCBS/EMD/SCADA/RTU/etc.).
- `sl_rag/retrieval/bm25_retriever.py`: sparse retrieval over chunk text.
- `sl_rag/retrieval/hybrid_retriever.py`: fusion of dense+BM25 signals (weighted or RRF).
- `sl_rag/retrieval/reranker.py`: cross-encoder reranking (MiniLM) with sigmoid calibration.
- `sl_rag/retrieval/document_level_domain_manager.py`: domain clustering + adaptive routing (Trust-RAG 1B).
- `sl_rag/retrieval/adversarial_detector.py`: ASI-style anomaly score against anchor embeddings.
- `sl_rag/retrieval/trust_scorer.py`: composite trust score from semantic relevance + credibility + freshness + consistency.
- `sl_rag/retrieval/retrieval_pipeline.py`: coordinates preprocessing, routing, retrieval, rerank, dedup.

Additional/alternative domain modules exist:
- `sl_rag/retrieval/domain_classifier.py`: rule-based domain classification (keywords/patterns/structure).
- `sl_rag/retrieval/domain_manager.py`: embedding-driven KMeans clustering and centroid routing.

### 3.4 Generation
- `sl_rag/generation/prompt_builder.py`: structured prompt formatting + prompt injection detection + context packing.
- `sl_rag/generation/llm_generator.py`: loads local LLM (4-bit quantization) and runs generation (sync + streaming).

### 3.5 Validation
- `sl_rag/validation/validation_pipeline.py`: retrieval quality checks (citation quality, confidence) + answer-level checks (hallucination risk, consistency, faithfulness heuristics).

### 3.6 Monitoring / audit
- `sl_rag/monitoring/monitoring_system.py`: SQLite-based monitoring store; security events; **SHA-256 hash-chain** audit trail integrity.

### 3.7 Security
- `sl_rag/security/auth.py`: `AuthContext` normalization and `RBACManager` (role-based query and document access filtering).

---

## 4) Product surface: backend API (FastAPI)

Backend root: `app/backend/`

### App wiring
- `app/backend/main.py`: FastAPI app, CORS, startup logic; starts background ingestion so API becomes available quickly.
- `app/backend/pipeline_instance.py`: singleton pipeline instance configured with paths and flags.

### Authentication and sessions
- `app/backend/routes/auth.py`: role-based login; returns a session UUID; admin endpoints to list/create users.
- `app/backend/session_store.py`: in-memory session store with TTL (8 hours default).
- `app/backend/auth_dependencies.py`: reads `X-User-Id`, `X-Role`, `X-Session-Id`; validates session; provides `require_roles(...)` dependency.
- `app/backend/credentials.py`: PBKDF2-HMAC-SHA256 password hashing; persists runtime-created users in `storage/auth/users.json`.

### Query + streaming
- `app/backend/routes/query.py`: query endpoint + streaming endpoint (SSE-style token events).

### Document management and ingestion
- `app/backend/routes/docs.py`: list/upload/delete PDFs in `data/`.
- `app/backend/routes/ingest.py`: manual ingestion trigger + status endpoint.

### Audit view
- `app/backend/routes/audit.py`: integrity status + paginated audit logs.

---

## 5) Product surface: frontend (React)

Frontend root: `app/frontend/`

### Entry points
- `app/frontend/src/main.tsx`: React bootstrap.
- `app/frontend/src/App.tsx`: router + route guards (auth required; admin-only areas).
- `app/frontend/src/index.css` + Tailwind config files: styling.

### API client and types
- `app/frontend/src/api/client.ts`: Axios client for REST; streaming via `fetch` parsing `data:` lines.
- `app/frontend/src/api/types.ts`: TypeScript interfaces for sessions, sources, query response, ingest status, audit entries.

### Pages
- `app/frontend/src/pages/LoginPage.tsx`: login form + role selector; polls pipeline status.
- `app/frontend/src/pages/ChatPage.tsx`: streaming chat UI; accumulates tokens and attaches final metadata.
- `app/frontend/src/pages/IngestPage.tsx`: document list + upload/delete + re-ingest trigger.
- `app/frontend/src/pages/AuditPage.tsx`: audit chain integrity + query log table.
- `app/frontend/src/pages/AdminUsersPage.tsx`: admin user creation and listing.

### Components
- `app/frontend/src/components/Navbar.tsx`: navigation and role badge; logout.
- `app/frontend/src/components/ChatMessage.tsx`: renders message markdown + badges + citations.
- `app/frontend/src/components/CitationsPanel.tsx`: expandable list of sources.
- `app/frontend/src/components/DomainBadge.tsx`: maps backend domain labels to user-friendly tags (GFR/Procurement/Technical).
- `app/frontend/src/components/ConfidenceBadge.tsx`: confidence bar + hallucination risk badge.
- `app/frontend/src/components/IngestStatus.tsx`: status card polling backend.

---

## 6) Configuration and reproducibility

### Pipeline configuration
- `config/config.yaml` contains the main hyperparameters:
  - Embedding model + cross-encoder model
  - Chunk sizes and overlap
  - Hybrid fusion weight (`alpha`)
  - Similarity thresholds and top-k settings
  - Domain clustering routing thresholds
  - LLM temperature and max tokens
  - Monitoring DB encryption-at-rest settings

### Data and artifacts
- `data/` contains the PDFs indexed by the pipeline.
- `storage/` contains runtime artifacts:
  - `storage/faiss_index/` (vector indexes)
  - `storage/audit_logs/` (monitoring DB + logs)
  - `storage/auth/` (persisted users)
  - `storage/keys/` (encryption keys)

---

## 7) Evaluation: scripts and outputs

This repo includes both **retrieval-only** tests and **end-to-end generation** evaluations.

### Retrieval-only quality snapshot
- `eval_retrieval.py`: runs queries with `generate_answer=False` and saves to `eval_results.json`.
- Output: `eval_results.json` includes confidence, citation quality, latency, and preview snippets.

### Full LLM generation artifact
- `eval_llm.py`: runs end-to-end generation (`load_llm=True`, `generate_answer=True`) and saves a Markdown artifact.
- Output: `llm_eval_results.md`.

### Comprehensive local evaluation
- `eval_comprehensive.py`: computes BERTScore, ROUGE-L, MRR/Recall@k, calibration (ECE), ASI detection accuracy, hallucination rate, trust score averages, faithfulness averages, and latency.
- Output: `eval_comprehensive_results.json`.

### Fix-up / post-processing
- `eval_fix_metrics.py`: corrects metric computation using robust filename keyword matching + dynamic ASI thresholding.

### RAGAS evaluation (external judge)
- `eval_ragas.py`: runs RAGAS metrics; requires `HF_TOKEN` and Hugging Face Inference API.
- Output: `eval_ragas_results.json`.

### Evaluation dataset
- `eval_dataset.json`: question/ground-truth pairs + domain labels + adversarial samples.

---

## 8) Threat model and governance (what is “Trust-RAG” here?)

The Trust-RAG implementation in this repository operationalizes trust through:
- **Adversarial query detection**: ASI anomaly scoring versus anchor embeddings.
- **Prompt injection detection**: pattern heuristics + structured prompt builder.
- **RBAC**: restricts which roles can access which document sensitivity levels.
- **Audit integrity**: tamper-evident hash chain over query logs.
- **Evidence trust scoring**: composite trust features used for ranking returned sources.

Design reference: `Trust-RAG.md` and `research_methodology.md`.

---

## 9) Known integration note (useful to mention in a paper)

There is a *field naming expectation* worth documenting:
- Some frontend components expect `source.preview` (a text snippet).
- The pipeline sources payload also includes a text snippet, but naming must match end-to-end.

This is a good example to cite when discussing interface contracts between RAG backends and UI citation panels.

---

## Appendix A — File-by-file inventory (by directory)

### Repository root
- `activate.ps1`: activates local `venv` on Windows.
- `start-backend.ps1`: activates venv and runs FastAPI via Uvicorn.
- `start-frontend.ps1`: runs the Vite dev server.
- `requirements.txt`: Python dependencies (torch, sentence-transformers, FAISS, BM25, cryptography, FastAPI tooling).

Evaluation artifacts and scripts:
- `demo_pipeline_test.py`: broad layer-by-layer smoke test (no LLM) covering ingestion, embeddings, FAISS, BM25, hybrid, domain routing, validation.
- `test_components.py`: component-level tests for core pipeline pieces.
- `test_integration.py`: ingest + a few representative queries + RBAC denial + compliance report.
- `test_quality_guidelines.py`: checks that `.github/` guidance files exist and demonstrates a “guidelines validation” harness.
- `test_gemini.py`: probes Google GenAI embedding model access (optional; not required for offline path).
- `eval_retrieval.py`, `eval_llm.py`, `eval_comprehensive.py`, `eval_fix_metrics.py`, `eval_ragas.py`: evaluation drivers.
- `eval_results.json`, `eval_comprehensive_results.json`, `eval_ragas_results.json`, `llm_eval_results.md`: saved results/artifacts.

Documentation:
- `Trust-RAG.md`: Trust-RAG design blueprint (adversarial detection, domain routing, federated retrieval, trust scoring).
- `research_methodology.md`: paper-ready methodology, formulas, thresholds, and bibliographic mapping.
- `research_methodology.md` is the most direct source for “methods” section text.

Misc saved outputs:
- `eval_output.txt`: captured evaluation console output.

### app/backend/
- `main.py`: FastAPI app + lifespan startup behavior.
- `pipeline_instance.py`: singleton pipeline.
- `auth_dependencies.py`: header-based auth dependency.
- `session_store.py`: server-side session TTL.
- `credentials.py`: PBKDF2 user store persisted in `storage/auth/users.json`.

Routes (`app/backend/routes/`):
- `auth.py`: login, list roles, admin user management.
- `query.py`: query and SSE streaming endpoints.
- `ingest.py`: ingestion trigger + status.
- `docs.py`: list/upload/delete documents.
- `audit.py`: audit chain integrity + logs.

### app/frontend/
- `package.json`, `vite.config.ts`, `tailwind.config.js`, `tsconfig.json`: frontend build + style config.
- `src/App.tsx`, `src/main.tsx`, `src/index.css`: SPA entrypoints.

API layer (`app/frontend/src/api/`):
- `client.ts`: REST + streaming implementation.
- `types.ts`: shared types for responses.

Pages (`app/frontend/src/pages/`):
- `LoginPage.tsx`: role-aware login + pipeline warm-up banner.
- `ChatPage.tsx`: streaming chat + citations.
- `IngestPage.tsx`: documents UI.
- `AuditPage.tsx`: audit UI.
- `AdminUsersPage.tsx`: admin UI.

Components (`app/frontend/src/components/`):
- `Navbar.tsx`: app navigation.
- `ChatMessage.tsx`: message rendering.
- `CitationsPanel.tsx`: evidence view.
- `DomainBadge.tsx`: domain tag mapping.
- `ConfidenceBadge.tsx`: confidence + hallucination risk.
- `IngestStatus.tsx`: ingest status card.

Context (`app/frontend/src/context/`):
- `ChatContext.tsx`: state store for message list + streaming.

### sl_rag/
- `pipeline.py`: orchestrator described above.

`sl_rag/core/`:
- `document_loader.py`, `pii_anonymizer.py`, `chunk_generator.py`, `embedding_generator.py`, `faiss_index.py`, `encryption_manager.py`, `schemas.py`.

`sl_rag/retrieval/`:
- `retrieval_pipeline.py`, `hybrid_retriever.py`, `bm25_retriever.py`, `reranker.py`, `query_preprocessor.py`, `trust_scorer.py`, `adversarial_detector.py`.
- Domain modules: `document_level_domain_manager.py`, `domain_manager.py`, `domain_classifier.py`.
- `policy.py`: retrieval policy dataclass.

`sl_rag/generation/`:
- `prompt_builder.py`, `llm_generator.py`.

`sl_rag/validation/`:
- `validation_pipeline.py`.

`sl_rag/monitoring/`:
- `monitoring_system.py`.

`sl_rag/security/`:
- `auth.py`.

### config/
- `config.yaml`: pipeline configuration.
- `domain_mappings.json`: optional domain label mappings.

### data/
- PDF corpus used for indexing (GFR + procurement manuals + technical reports/telemetry).

### papers/
- Literature PDFs supporting the methodology and related work section.

### models/
- Local model artifacts (tokenizers, safetensors, configs) for:
  - `models/llama-3.2-3b/`
  - `models/phi-3.5-mini/`

### storage/
- Runtime persistent data: audit logs, FAISS index, keys, cache, and persisted auth users.

### misc/
- `ANTIGRAVITY_PROMPT.md`, `MODEL_DOWNLOAD_INSTRUCTIONS.md`: internal prompt/design notes and model setup guidance.

### .github/
This folder contains the project’s **prompt generation quality rules** used to keep government-document answers citation-grounded and to reduce hallucinations.

- `.github/copilot-instructions.md`: global constraints (e.g., citation discipline, max chunks, confidence floors).
- `.github/instructions/llm-generator.instructions.md`: generation parameter guidance (temperature, max tokens, stopping).
- `.github/instructions/prompt-builder.instructions.md`: RAG prompt construction constraints (chunk caps, ordering, injection handling).
- `.github/prompts/generate-answer-for-government-doc.prompt.md`: prompt template for government-doc answers.
- `.github/skills/*/SKILL.md`: deeper guides for retrieval context quality and prompt engineering.

---

## Appendix B — Suggested paper outline (drop-in)

1. Abstract
2. Introduction (problem, constraints, offline requirement)
3. Related Work (use `papers/` + mapping in `research_methodology.md`)
4. System Architecture (7-layer pipeline)
5. Methodology
   - Ingestion + PII
   - Chunking
   - Hybrid retrieval
   - Domain routing
   - Trust scoring
   - Prompting + injection defenses
   - Validation + monitoring
6. Implementation (backend + frontend)
7. Evaluation (retrieval-only, comprehensive local, RAGAS)
8. Governance and security analysis
9. Limitations
10. Future work
11. References

