import React from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchStatus } from "../api/client";
import type { IngestStatus } from "../api/types";

function Dot({ active }: { active: boolean }) {
  return (
    <span
      className={`inline-block w-2 h-2 rounded-full ${
        active ? "bg-emerald-500 animate-pulse" : "bg-slate-600"
      }`}
    />
  );
}

export default function IngestStatusCard() {
  const { data, isLoading, isError } = useQuery<IngestStatus>({
    queryKey: ["status"],
    queryFn: fetchStatus,
    refetchInterval: 3000,
  });

  if (isLoading) {
    return (
      <div className="card animate-pulse">
        <div className="h-4 bg-slate-800 rounded w-1/3 mb-3" />
        <div className="h-3 bg-slate-800 rounded w-1/2" />
      </div>
    );
  }

  if (isError || !data) {
    return (
      <div className="card border-red-500/30">
        <p className="text-sm text-red-400">Could not reach backend.</p>
      </div>
    );
  }

  return (
    <div className="card space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-200">Pipeline Status</h3>
        <div className="flex items-center gap-2">
          <Dot active={data.ready} />
          <span className={`text-xs font-medium ${data.ready ? "text-emerald-400" : "text-slate-500"}`}>
            {data.ready ? "Ready" : "Not ready"}
          </span>
        </div>
      </div>

      {data.ingest_running && (
        <div className="flex items-center gap-2 text-xs text-amber-400">
          <svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Ingesting documents…
        </div>
      )}

      {data.error && (
        <p className="text-xs text-red-400 bg-red-500/10 rounded p-2">{data.error}</p>
      )}

      {/* Stats grid */}
      <div className="grid grid-cols-2 gap-3">
        <Stat label="Documents" value={data.docs} />
        <Stat label="Chunks" value={data.chunks} />
        {data.ingested_at && (
          <div className="col-span-2">
            <p className="text-[10px] text-slate-600 font-mono">
              Last indexed: {new Date(data.ingested_at).toLocaleString()}
            </p>
          </div>
        )}
      </div>

      {/* Domain distribution */}
      {data.domains && Object.keys(data.domains).length > 0 && (
        <div>
          <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-2">
            Domain Distribution
          </p>
          <div className="space-y-1.5">
            {Object.entries(data.domains)
              .sort(([, a], [, b]) => b - a)
              .map(([domain, count]) => (
                <DomainBar
                  key={domain}
                  domain={domain}
                  count={count as number}
                  total={Object.values(data.domains!).reduce((a, b) => (a as number) + (b as number), 0) as number}
                />
              ))}
          </div>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="bg-slate-800/60 rounded-lg p-3">
      <p className="text-[10px] text-slate-500 uppercase tracking-wider">{label}</p>
      <p className="text-lg font-bold text-slate-100 mt-0.5 font-mono">{value}</p>
    </div>
  );
}

const DOMAIN_COLORS: Record<string, string> = {
  government_expenditure: "bg-blue-500",
  procurement_contract:   "bg-violet-500",
  figure_data:            "bg-teal-500",
  figure_socket:          "bg-teal-600",
};

function DomainBar({ domain, count, total }: { domain: string; count: number; total: number }) {
  const pct = total > 0 ? (count / total) * 100 : 0;
  const color = DOMAIN_COLORS[domain] ?? "bg-slate-500";
  const label = domain.replace(/_/g, " ");

  return (
    <div>
      <div className="flex justify-between items-center mb-0.5">
        <span className="text-[10px] text-slate-400 capitalize truncate max-w-[140px]">{label}</span>
        <span className="text-[10px] text-slate-500 font-mono shrink-0 ml-2">{count}</span>
      </div>
      <div className="h-1 bg-slate-700 rounded-full overflow-hidden">
        <div
          className={`h-full ${color} rounded-full transition-all`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
