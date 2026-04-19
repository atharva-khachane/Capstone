import axios from "axios";
import type {
  Session,
  QueryResponse,
  IngestStatus,
  AuditResponse,
  RoleOption,
  DocEntry,
  UserAccount,
} from "./types";

const http = axios.create({ baseURL: "/api" });

// Inject session headers on every request
http.interceptors.request.use((config) => {
  const raw = localStorage.getItem("sl_rag_session");
  if (raw) {
    try {
      const s: Session = JSON.parse(raw);
      config.headers["X-User-Id"] = s.user_id;
      config.headers["X-Role"] = s.role;
      config.headers["X-Session-Id"] = s.session_id;
    } catch {
      // malformed session — ignore
    }
  }
  return config;
});

// ── Auth ──────────────────────────────────────────────────────────────────────

export async function login(user_id: string, password: string): Promise<Session> {
  const { data } = await http.post<Session>("/auth/login", { user_id, password });
  return data;
}

export async function fetchRoles(): Promise<RoleOption[]> {
  const { data } = await http.get<{ roles: RoleOption[] }>("/auth/roles");
  return data.roles;
}

export async function fetchUsers(): Promise<UserAccount[]> {
  const { data } = await http.get<{ users: UserAccount[] }>("/auth/users");
  return data.users;
}

export async function createUser(payload: {
  user_id: string;
  password: string;
  role: string;
}): Promise<{ status: string; user_id: string; role: string }> {
  const { data } = await http.post("/auth/users", payload);
  return data;
}

export async function deleteUser(user_id: string): Promise<{ status: string; user_id: string }> {
  const uid = (user_id || "").trim();
  const { data } = await http.delete(`/auth/users/${encodeURIComponent(uid)}`);
  return data;
}

// ── Query ─────────────────────────────────────────────────────────────────────

export interface QueryRequest {
  question: string;
  user_id: string;
  role: string;
  session_id: string;
  top_k?: number;
  enable_reranking?: boolean;
  generate_answer?: boolean;
}

export async function runQuery(req: QueryRequest): Promise<QueryResponse> {
  const { data } = await http.post<QueryResponse>("/query", req);
  return data;
}

export type StreamEvent =
  | { type: "token"; token: string }
  | ({ type: "done" } & QueryResponse)
  | { type: "error"; message: string };

/**
 * Stream a query via SSE. Calls onEvent for each token/done/error event.
 * Returns a cancel function.
 */
export function streamQuery(
  req: QueryRequest,
  onEvent: (event: StreamEvent) => void
): () => void {
  const session = (() => {
    try { return JSON.parse(localStorage.getItem("sl_rag_session") ?? "{}"); }
    catch { return {}; }
  })();

  const controller = new AbortController();

  fetch("/api/query/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-User-Id": session.user_id ?? "",
      "X-Role": session.role ?? "",
      "X-Session-Id": session.session_id ?? "",
    },
    body: JSON.stringify(req),
    signal: controller.signal,
  })
    .then(async (res) => {
      if (!res.ok) {
        const text = await res.text().catch(() => "Request failed");
        onEvent({ type: "error", message: text });
        return;
      }
      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        // Process complete SSE lines
        const lines = buf.split("\n");
        buf = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6).trim();
          if (data === "[DONE]") return;
          try {
            onEvent(JSON.parse(data) as StreamEvent);
          } catch {
            // skip malformed line
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== "AbortError") {
        onEvent({ type: "error", message: (err as Error).message });
      }
    });

  return () => controller.abort();
}

// ── Ingest ────────────────────────────────────────────────────────────────────

export async function triggerIngest(): Promise<{ status: string; message: string }> {
  const { data } = await http.post("/ingest");
  return data;
}

export async function fetchStatus(): Promise<IngestStatus> {
  const { data } = await http.get<IngestStatus>("/status");
  return data;
}

// ── Docs ──────────────────────────────────────────────────────────────────────

export async function fetchDocs(): Promise<DocEntry[]> {
  const { data } = await http.get<{ docs: DocEntry[] }>("/docs");
  return data.docs;
}

export async function uploadDoc(
  file: File,
  onProgress?: (pct: number) => void
): Promise<{ status: string; filename: string; message: string }> {
  const form = new FormData();
  form.append("file", file);
  const { data } = await http.post("/docs/upload", form, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress: (e) => {
      if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100));
    },
  });
  return data;
}

export async function deleteDoc(
  filename: string
): Promise<{ status: string; filename: string; message: string }> {
  const { data } = await http.delete(`/docs/${encodeURIComponent(filename)}`);
  return data;
}

// ── Audit ─────────────────────────────────────────────────────────────────────

export async function fetchAudit(
  limit = 50,
  offset = 0
): Promise<AuditResponse> {
  const { data } = await http.get<AuditResponse>("/audit", {
    params: { limit, offset },
  });
  return data;
}

// ── Session helpers ───────────────────────────────────────────────────────────

export function saveSession(session: Session): void {
  localStorage.setItem("sl_rag_session", JSON.stringify(session));
}

export function loadSession(): Session | null {
  const raw = localStorage.getItem("sl_rag_session");
  if (!raw) return null;
  try {
    return JSON.parse(raw) as Session;
  } catch {
    return null;
  }
}

export function clearSession(): void {
  localStorage.removeItem("sl_rag_session");
}
