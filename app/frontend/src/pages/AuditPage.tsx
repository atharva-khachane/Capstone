import React, { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchAudit } from "../api/client";
import type { AuditResponse, AuditEntry } from "../api/types";

const PAGE_SIZE = 20;

function ChainBadge({ valid }: { valid: boolean }) {
  return (
    <span
      className={`badge rounded-md text-xs ${
        valid
          ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
          : "bg-red-500/20 text-red-400 border border-red-500/30"
      }`}
    >
      {valid ? "✓ Intact" : "✗ Broken"}
    </span>
  );
}

function confidenceColor(c: number): string {
  if (c >= 0.75) return "text-emerald-400";
  if (c >= 0.40) return "text-amber-400";
  return "text-red-400";
}

function parseDomains(raw: string): string {
  try {
    const arr = JSON.parse(raw);
    if (Array.isArray(arr)) {
      return arr
        .map((d: string) => d.replace("government_expenditure", "GFR").replace("procurement_contract", "Procurement").replace("figure_data", "Tech").replace("figure_socket", "Tech"))
        .join(", ");
    }
  } catch {
    // ignore
  }
  return raw ?? "—";
}

export default function AuditPage() {
  const [page, setPage] = useState(0);
  const offset = page * PAGE_SIZE;

  const { data, isLoading, isError } = useQuery<AuditResponse>({
    queryKey: ["audit", page],
    queryFn: () => fetchAudit(PAGE_SIZE, offset),
    staleTime: 10_000,
  });

  const totalPages = data ? Math.ceil(data.total / PAGE_SIZE) : 0;

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="shrink-0 px-6 py-4 border-b border-slate-800 bg-slate-900/50">
        <h1 className="text-sm font-semibold text-slate-200">Audit Log</h1>
        <p className="text-xs text-slate-500 mt-0.5">
          Tamper-evident query history with hash-chain integrity
        </p>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-5">
        {/* Summary cards */}
        {data && (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <StatCard label="Total Queries" value={data.query_stats?.total_queries ?? 0} />
            <StatCard
              label="Avg Confidence"
              value={`${((data.query_stats?.avg_confidence ?? 0) * 100).toFixed(0)}%`}
            />
            <StatCard
              label="Avg Latency"
              value={`${(data.query_stats?.avg_latency_ms ?? 0).toFixed(0)}ms`}
            />
            <div className="card flex flex-col justify-center gap-1">
              <p className="text-[10px] text-slate-500 uppercase tracking-wider">Chain Integrity</p>
              <ChainBadge valid={data.audit_chain.valid} />
              <p className="text-[10px] text-slate-600 font-mono">
                {data.audit_chain.entries} entries
                {!data.audit_chain.valid && data.audit_chain.broken_at !== null &&
                  ` · broken at #${data.audit_chain.broken_at}`}
              </p>
            </div>
          </div>
        )}

        {/* Table */}
        <div className="card p-0 overflow-hidden">
          <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between">
            <p className="text-xs font-semibold text-slate-300">Query Log</p>
            {data && (
              <p className="text-[10px] text-slate-600 font-mono">
                {data.total} total entries
              </p>
            )}
          </div>

          {isLoading && (
            <div className="p-6 space-y-3">
              {Array.from({ length: 5 }).map((_, i) => (
                <div key={i} className="h-10 bg-slate-800/60 rounded animate-pulse" />
              ))}
            </div>
          )}

          {isError && (
            <p className="p-6 text-sm text-red-400">Failed to load audit log.</p>
          )}

          {data && data.entries.length === 0 && (
            <p className="p-6 text-sm text-slate-500 text-center">No queries logged yet.</p>
          )}

          {data && data.entries.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-500 uppercase tracking-wider">
                    <th className="px-4 py-2.5 text-left font-medium">#</th>
                    <th className="px-4 py-2.5 text-left font-medium">Time</th>
                    <th className="px-4 py-2.5 text-left font-medium">Query</th>
                    <th className="px-4 py-2.5 text-left font-medium">User</th>
                    <th className="px-4 py-2.5 text-left font-medium">Role</th>
                    <th className="px-4 py-2.5 text-left font-medium">Domain</th>
                    <th className="px-4 py-2.5 text-left font-medium">Conf</th>
                    <th className="px-4 py-2.5 text-left font-medium">Latency</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800">
                  {data.entries.map((entry: AuditEntry) => (
                    <tr key={entry.id} className="hover:bg-slate-800/30 transition-colors">
                      <td className="px-4 py-3 text-slate-600 font-mono">{entry.id}</td>
                      <td className="px-4 py-3 text-slate-500 font-mono whitespace-nowrap">
                        {new Date(entry.timestamp).toLocaleTimeString([], {
                          hour: "2-digit", minute: "2-digit", second: "2-digit",
                        })}
                      </td>
                      <td className="px-4 py-3 text-slate-300 max-w-[280px]">
                        <p className="truncate">{entry.query_text}</p>
                      </td>
                      <td className="px-4 py-3 text-slate-400 font-mono">{entry.user_id || "—"}</td>
                      <td className="px-4 py-3">
                        <span className="badge bg-slate-700/60 text-slate-400 rounded text-[10px]">
                          {entry.role || "guest"}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-slate-400">
                        {parseDomains(entry.domains)}
                      </td>
                      <td className={`px-4 py-3 font-mono font-semibold ${confidenceColor(entry.confidence)}`}>
                        {entry.confidence ? `${(entry.confidence * 100).toFixed(0)}%` : "—"}
                      </td>
                      <td className="px-4 py-3 text-slate-500 font-mono">
                        {entry.latency_ms ? `${entry.latency_ms.toFixed(0)}ms` : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="px-4 py-3 border-t border-slate-800 flex items-center justify-between">
              <button
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={page === 0}
                className="btn-ghost text-xs disabled:opacity-30"
              >
                ← Previous
              </button>
              <span className="text-xs text-slate-500 font-mono">
                Page {page + 1} / {totalPages}
              </span>
              <button
                onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                disabled={page >= totalPages - 1}
                className="btn-ghost text-xs disabled:opacity-30"
              >
                Next →
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="card">
      <p className="text-[10px] text-slate-500 uppercase tracking-wider">{label}</p>
      <p className="text-xl font-bold text-slate-100 mt-1 font-mono">{value}</p>
    </div>
  );
}
