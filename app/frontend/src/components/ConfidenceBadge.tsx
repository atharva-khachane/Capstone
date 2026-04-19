import React from "react";
import type { HallucinationRisk } from "../api/types";

// Per .github quality guidelines:
//   GFR floor 0.75 | Procurement floor 0.80 | Technical floor 0.65
// Using 0.75 as a general "good" threshold in the badge since we
// may not always know the domain at rendering time.
function confidenceLevel(score: number): "high" | "medium" | "low" {
  if (score >= 0.75) return "high";
  if (score >= 0.40) return "medium";
  return "low";
}

const CONF_STYLES = {
  high:   "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30",
  medium: "bg-amber-500/20  text-amber-300  border border-amber-500/30",
  low:    "bg-red-500/20    text-red-300    border border-red-500/30",
};

const RISK_STYLES: Record<HallucinationRisk, string> = {
  low:     "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30",
  medium:  "bg-amber-500/20  text-amber-300  border border-amber-500/30",
  high:    "bg-red-500/20    text-red-300    border border-red-500/30",
  unknown: "bg-slate-500/20  text-slate-300  border border-slate-500/30",
};

interface Props {
  confidence: number;
  hallucinationRisk: HallucinationRisk;
}

export default function ConfidenceBadge({ confidence, hallucinationRisk }: Props) {
  const level = confidenceLevel(confidence);

  return (
    <div className="flex items-center gap-2 flex-wrap">
      {/* Confidence bar + score */}
      <div className={`badge rounded-md font-mono ${CONF_STYLES[level]}`}>
        <span className="mr-1.5 opacity-60 text-[10px]">CONF</span>
        {(confidence * 100).toFixed(0)}%
      </div>

      {/* Confidence bar */}
      <div className="relative h-1.5 w-20 rounded-full bg-slate-700 overflow-hidden">
        <div
          className={`absolute left-0 top-0 h-full rounded-full transition-all ${
            level === "high" ? "bg-emerald-500" :
            level === "medium" ? "bg-amber-500" : "bg-red-500"
          }`}
          style={{ width: `${confidence * 100}%` }}
        />
      </div>

      {/* Hallucination risk chip */}
      <div className={`badge rounded-md font-mono ${RISK_STYLES[hallucinationRisk]}`}>
        <span className="mr-1.5 opacity-60 text-[10px]">HALL</span>
        {hallucinationRisk}
      </div>

      {/* Human-readable label */}
      {level === "low" && (
        <span className="text-xs text-red-400 font-medium">Review needed</span>
      )}
      {level === "medium" && (
        <span className="text-xs text-amber-400 font-medium">Medium confidence</span>
      )}
    </div>
  );
}
