// ── Auth ──────────────────────────────────────────────────────────────────────

export interface Session {
  user_id: string;
  role: string;
  session_id: string;
  can_query: boolean;
  pipeline_ready: boolean;
}

export interface RoleOption {
  role: string;
  access: string;
}

export interface UserAccount {
  user_id: string;
  role: string;
}

// ── Query ─────────────────────────────────────────────────────────────────────

export interface Source {
  rank: number;
  score: number;
  file: string;
  domain: string;
  chunk_index: number;
  preview: string;
  // pipeline may also return source/doc_id keys
  source?: string;
  doc_id?: string;
}

export type HallucinationRisk = "low" | "medium" | "high" | "unknown";

export interface Latency {
  retrieval_ms: number;
  generation_ms: number;
  total_ms: number;
}

export interface QueryResponse {
  query: string;
  answer: string;
  confidence: number;
  hallucination_risk: HallucinationRisk;
  citation_quality: string;
  injection_blocked: boolean;
  latency: Latency;
  num_results: number;
  sources: Source[];
  domain: string | null;
}

// ── Ingest ────────────────────────────────────────────────────────────────────

export interface IngestStatus {
  ready: boolean;
  ingest_running: boolean;
  ingested_at: string | null;
  docs: number;
  chunks: number;
  domains: Record<string, number> | null;
  error: string | null;
}

// ── Docs ──────────────────────────────────────────────────────────────────────

export interface DocEntry {
  filename: string;
  size_bytes: number;
  modified_at: number; // Unix timestamp (seconds)
}

// ── Audit ─────────────────────────────────────────────────────────────────────

export interface AuditEntry {
  id: number;
  timestamp: string;
  query_text: string;
  user_id: string;
  role: string;
  domains: string;        // JSON string from DB
  num_results: number;
  latency_ms: number;
  confidence: number;
}

export interface AuditChain {
  valid: boolean;
  entries: number;
  broken_at: number | null;
}

export interface AuditResponse {
  audit_chain: AuditChain;
  query_stats: {
    total_queries: number;
    avg_latency_ms: number;
    avg_confidence: number;
    avg_results: number;
  };
  entries: AuditEntry[];
  total: number;
  limit: number;
  offset: number;
}

// ── Chat history (client-side only) ──────────────────────────────────────────

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  response?: QueryResponse;
  timestamp: Date;
}
