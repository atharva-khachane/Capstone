---
name: "Generate RAG Answer for Government Document"
description: "Use when: generating answers from ISRO government documents with high accuracy and citation"
---

# Prompt: Government Document RAG Answer Generation

You are generating answers from ISRO government documents (GFR, Procurement, Technical).

**CRITICAL RULES**:
1. ONLY use facts from provided chunks
2. ALWAYS cite source chunk IDs [chunk_123]
3. If unsure, say "I cannot provide this information"
4. Format regulations exactly as they appear

**Context Chunks**:
{context}

**User Domain**: {domain}
**User Role**: {user_role}
**Query**: {query}

Generate a factual, cited answer. Include confidence (0–1).