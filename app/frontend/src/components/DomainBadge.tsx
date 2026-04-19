import React from "react";

const DOMAIN_CONFIG: Record<string, { label: string; classes: string }> = {
  government_expenditure: {
    label: "GFR",
    classes: "bg-blue-500/20 text-blue-300 border border-blue-500/30",
  },
  procurement_contract: {
    label: "Procurement",
    classes: "bg-violet-500/20 text-violet-300 border border-violet-500/30",
  },
  figure_data: {
    label: "Technical",
    classes: "bg-teal-500/20 text-teal-300 border border-teal-500/30",
  },
  figure_socket: {
    label: "Technical",
    classes: "bg-teal-500/20 text-teal-300 border border-teal-500/30",
  },
};

const FALLBACK = {
  label: "Mixed",
  classes: "bg-slate-500/20 text-slate-300 border border-slate-500/30",
};

interface Props {
  domain: string | null | undefined;
  size?: "sm" | "md";
}

export default function DomainBadge({ domain, size = "md" }: Props) {
  const cfg = domain ? (DOMAIN_CONFIG[domain] ?? FALLBACK) : FALLBACK;
  const sizeClass = size === "sm" ? "text-[10px] px-2 py-0.5" : "text-xs px-2.5 py-1";

  return (
    <span className={`badge rounded-md font-mono font-semibold tracking-wide ${cfg.classes} ${sizeClass}`}>
      {cfg.label}
    </span>
  );
}
