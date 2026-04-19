import React, { useState } from "react";
import type { Source } from "../api/types";
import DomainBadge from "./DomainBadge";

interface Props {
  sources: Source[];
}

function scoreColor(score: number): string {
  if (score >= 0.8) return "text-emerald-400";
  if (score >= 0.5) return "text-amber-400";
  return "text-slate-500";
}

function fileName(src: Source): string {
  // try .file first, then basename of .source
  if (src.file) return src.file;
  if (src.source) return src.source.split(/[\\/]/).pop() ?? src.source;
  return "Unknown";
}

export default function CitationsPanel({ sources }: Props) {
  const [open, setOpen] = useState(false);
  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  if (!sources || sources.length === 0) return null;

  const toggleChunk = (idx: number) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      next.has(idx) ? next.delete(idx) : next.add(idx);
      return next;
    });
  };

  return (
    <div className="mt-3 border border-slate-700/60 rounded-lg overflow-hidden">
      {/* Header toggle */}
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-4 py-2.5 bg-slate-800/60 hover:bg-slate-800 transition text-sm font-medium text-slate-300"
      >
        <span className="flex items-center gap-2">
          <svg className="w-4 h-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          {sources.length} source{sources.length !== 1 ? "s" : ""} retrieved
        </span>
        <svg
          className={`w-4 h-4 text-slate-400 transition-transform ${open ? "rotate-180" : ""}`}
          fill="none" viewBox="0 0 24 24" stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Source list */}
      {open && (
        <div className="divide-y divide-slate-700/40">
          {sources.map((src, i) => (
            <div key={i} className="bg-slate-900/50">
              {/* Source header row */}
              <button
                onClick={() => toggleChunk(i)}
                className="w-full flex items-start justify-between px-4 py-3 hover:bg-slate-800/40 transition text-left"
              >
                <div className="flex items-center gap-3 min-w-0">
                  <span className="shrink-0 w-5 h-5 rounded-full bg-slate-700 text-slate-400 text-[10px] font-bold flex items-center justify-center">
                    {src.rank ?? i + 1}
                  </span>
                  <div className="min-w-0">
                    <p className="text-xs font-medium text-slate-200 truncate">
                      {fileName(src)}
                    </p>
                    <p className="text-[10px] text-slate-500 mt-0.5">
                      Chunk #{src.chunk_index}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2 shrink-0 ml-3">
                  <DomainBadge domain={src.domain} size="sm" />
                  <span className={`text-xs font-mono font-semibold ${scoreColor(src.score)}`}>
                    {src.score.toFixed(4)}
                  </span>
                  <svg
                    className={`w-3.5 h-3.5 text-slate-500 transition-transform ${expanded.has(i) ? "rotate-180" : ""}`}
                    fill="none" viewBox="0 0 24 24" stroke="currentColor"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </div>
              </button>

              {/* Expanded preview */}
              {expanded.has(i) && src.preview && (
                <div className="px-4 pb-4">
                  <p className="text-xs text-slate-400 leading-relaxed font-mono bg-slate-800/60 rounded p-3 border border-slate-700/40 whitespace-pre-wrap">
                    {src.preview}
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
