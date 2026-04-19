import React, { useCallback, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { deleteDoc, fetchDocs, triggerIngest, uploadDoc } from "../api/client";
import { loadSession } from "../api/client";
import IngestStatusCard from "../components/IngestStatus";
import type { DocEntry } from "../api/types";

// ── helpers ───────────────────────────────────────────────────────────────────

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function fmtDate(ts: number): string {
  return new Date(ts * 1000).toLocaleDateString(undefined, {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
}

// Domain inference from filename (best-effort)
function inferDomain(name: string): string {
  const lower = name.toLowerCase();
  if (lower.includes("gfr")) return "GFR";
  if (lower.includes("procurement")) return "Procurement";
  return "Technical";
}

const DOMAIN_STYLES: Record<string, string> = {
  GFR:          "bg-blue-500/20 text-blue-300 border border-blue-500/30",
  Procurement:  "bg-violet-500/20 text-violet-300 border border-violet-500/30",
  Technical:    "bg-teal-500/20 text-teal-300 border border-teal-500/30",
};

// ── sub-components ────────────────────────────────────────────────────────────

function UploadZone({
  onFiles,
  uploading,
  uploadPct,
}: {
  onFiles: (files: FileList) => void;
  uploading: boolean;
  uploadPct: number;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      if (e.dataTransfer.files.length) onFiles(e.dataTransfer.files);
    },
    [onFiles]
  );

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => !uploading && inputRef.current?.click()}
      className={`relative border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors select-none ${
        dragging
          ? "border-brand-400 bg-brand-500/10"
          : "border-slate-700 hover:border-slate-500 bg-slate-800/40"
      } ${uploading ? "pointer-events-none opacity-60" : ""}`}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,application/pdf"
        multiple
        className="hidden"
        onChange={(e) => e.target.files && onFiles(e.target.files)}
      />

      {uploading ? (
        <div className="space-y-3">
          <svg className="w-8 h-8 text-brand-400 mx-auto animate-spin" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <p className="text-xs text-slate-400">Uploading… {uploadPct}%</p>
          <div className="w-full bg-slate-700 rounded-full h-1.5 overflow-hidden">
            <div
              className="bg-brand-500 h-1.5 rounded-full transition-all"
              style={{ width: `${uploadPct}%` }}
            />
          </div>
        </div>
      ) : (
        <>
          <svg className="w-8 h-8 text-slate-500 mx-auto mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6h.1a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
          <p className="text-sm text-slate-300 font-medium">
            Drag &amp; drop PDFs here, or <span className="text-brand-400 underline">browse</span>
          </p>
          <p className="text-xs text-slate-600 mt-1">PDF only · max 200 MB per file</p>
        </>
      )}
    </div>
  );
}

function DocRow({
  doc,
  canDelete,
  onDelete,
  deleting,
}: {
  doc: DocEntry;
  canDelete: boolean;
  onDelete: () => void;
  deleting: boolean;
}) {
  const [confirm, setConfirm] = useState(false);
  const domain = inferDomain(doc.filename);

  return (
    <div className="flex items-center gap-3 py-2.5 border-b border-slate-800 last:border-0 group">
      {/* Icon */}
      <svg className="w-4 h-4 text-slate-500 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>

      {/* Name + meta */}
      <div className="flex-1 min-w-0">
        <p className="text-xs text-slate-300 font-mono truncate">{doc.filename}</p>
        <p className="text-[10px] text-slate-600">
          {fmtBytes(doc.size_bytes)} · {fmtDate(doc.modified_at)}
        </p>
      </div>

      {/* Domain badge */}
      <span className={`badge text-[10px] rounded-md shrink-0 ${DOMAIN_STYLES[domain] ?? DOMAIN_STYLES.Technical}`}>
        {domain}
      </span>

      {/* Delete (admin only) */}
      {canDelete && (
        confirm ? (
          <div className="flex items-center gap-1 shrink-0">
            <span className="text-[10px] text-red-400 mr-1">Delete?</span>
            <button
              onClick={onDelete}
              disabled={deleting}
              className="text-[10px] px-2 py-0.5 rounded bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/40 transition-colors"
            >
              {deleting ? "…" : "Yes"}
            </button>
            <button
              onClick={() => setConfirm(false)}
              className="text-[10px] px-2 py-0.5 rounded bg-slate-700 text-slate-400 hover:bg-slate-600 transition-colors"
            >
              No
            </button>
          </div>
        ) : (
          <button
            onClick={() => setConfirm(true)}
            className="shrink-0 opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded hover:bg-red-500/20 text-slate-600 hover:text-red-400"
            title="Delete document"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
          </button>
        )
      )}
    </div>
  );
}

// ── main page ─────────────────────────────────────────────────────────────────

export default function IngestPage() {
  const qc = useQueryClient();
  const session = loadSession();
  const isAdmin = session?.role === "admin";

  const [uploadPct, setUploadPct] = useState(0);
  const [uploadMsg, setUploadMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [deletingFile, setDeletingFile] = useState<string | null>(null);

  // Live doc list
  const { data: docsData, isLoading: docsLoading } = useQuery({
    queryKey: ["docs"],
    queryFn: fetchDocs,
    refetchInterval: 5000,
  });

  // Re-ingest trigger
  const reingestMutation = useMutation({
    mutationFn: triggerIngest,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["status"] }),
  });

  // Upload mutation
  const uploadMutation = useMutation({
    mutationFn: (file: File) => uploadDoc(file, setUploadPct),
    onSuccess: (res) => {
      setUploadMsg({ ok: true, text: res.message });
      setUploadPct(0);
      qc.invalidateQueries({ queryKey: ["docs"] });
      qc.invalidateQueries({ queryKey: ["status"] });
    },
    onError: (err: unknown) => {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
        ?? (err as Error)?.message
        ?? "Upload failed.";
      setUploadMsg({ ok: false, text: msg });
      setUploadPct(0);
    },
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: deleteDoc,
    onSuccess: (res) => {
      setUploadMsg({ ok: true, text: res.message });
      setDeletingFile(null);
      qc.invalidateQueries({ queryKey: ["docs"] });
      qc.invalidateQueries({ queryKey: ["status"] });
    },
    onError: (err: unknown) => {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
        ?? (err as Error)?.message
        ?? "Delete failed.";
      setUploadMsg({ ok: false, text: msg });
      setDeletingFile(null);
    },
  });

  const handleFiles = (files: FileList) => {
    setUploadMsg(null);
    Array.from(files).forEach((f) => {
      if (!f.name.toLowerCase().endsWith(".pdf")) {
        setUploadMsg({ ok: false, text: `'${f.name}' is not a PDF — skipped.` });
        return;
      }
      uploadMutation.mutate(f);
    });
  };

  const handleDelete = (filename: string) => {
    setUploadMsg(null);
    setDeletingFile(filename);
    deleteMutation.mutate(filename);
  };

  const docs = docsData ?? [];

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="shrink-0 px-6 py-4 border-b border-slate-800 bg-slate-900/50">
        <h1 className="text-sm font-semibold text-slate-200">Document Management</h1>
        <p className="text-xs text-slate-500 mt-0.5">
          Manage PDFs in <code className="bg-slate-800 px-1 py-0.5 rounded text-brand-400">./data</code> — changes trigger automatic re-ingestion
        </p>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6 max-w-2xl">
        {/* Pipeline status */}
        <IngestStatusCard />

        {/* Upload zone (admin only) */}
        {isAdmin && (
          <div className="card space-y-4">
            <div>
              <h3 className="text-sm font-semibold text-slate-200">Upload Documents</h3>
              <p className="text-xs text-slate-500 mt-1">
                PDF files are saved to <code className="bg-slate-800 px-1 rounded text-brand-400">./data</code> and
                the pipeline re-ingests automatically.
              </p>
            </div>

            <UploadZone
              onFiles={handleFiles}
              uploading={uploadMutation.isPending}
              uploadPct={uploadPct}
            />

            {uploadMsg && (
              <p className={`text-xs rounded-lg p-3 ${
                uploadMsg.ok
                  ? "text-emerald-400 bg-emerald-500/10"
                  : "text-red-400 bg-red-500/10"
              }`}>
                {uploadMsg.text}
              </p>
            )}
          </div>
        )}

        {/* Document list */}
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-slate-200">
              Indexed Documents
              {docs.length > 0 && (
                <span className="ml-2 text-[10px] font-normal text-slate-500">
                  {docs.length} file{docs.length !== 1 ? "s" : ""}
                </span>
              )}
            </h3>

            {/* Manual re-ingest button */}
            <button
              onClick={() => reingestMutation.mutate()}
              disabled={reingestMutation.isPending}
              className="flex items-center gap-1.5 text-[10px] px-2.5 py-1 rounded-lg bg-slate-700 text-slate-400 hover:bg-slate-600 hover:text-slate-200 transition-colors disabled:opacity-50"
              title="Manually trigger re-ingestion"
            >
              <svg className={`w-3 h-3 ${reingestMutation.isPending ? "animate-spin" : ""}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Re-ingest
            </button>
          </div>

          {docsLoading ? (
            <div className="py-6 text-center">
              <svg className="w-5 h-5 text-slate-600 animate-spin mx-auto" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
            </div>
          ) : docs.length === 0 ? (
            <p className="text-xs text-slate-600 text-center py-6">
              No PDFs found in <code className="bg-slate-800 px-1 rounded text-brand-400">./data</code>
            </p>
          ) : (
            <div className="space-y-0">
              {docs.map((doc) => (
                <DocRow
                  key={doc.filename}
                  doc={doc}
                  canDelete={isAdmin}
                  onDelete={() => handleDelete(doc.filename)}
                  deleting={deletingFile === doc.filename && deleteMutation.isPending}
                />
              ))}
            </div>
          )}

          {!isAdmin && (
            <p className="text-[10px] text-slate-600 mt-3 pt-3 border-t border-slate-800">
              Only admins can upload or delete documents.
            </p>
          )}
        </div>

        {/* Re-ingest warning */}
        <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg p-3">
          <div className="flex items-start gap-2">
            <svg className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <p className="text-xs text-amber-400">
              Re-ingestion runs in the background and may take 2–5 minutes depending on document size.
              You can continue using the chat while it processes.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
