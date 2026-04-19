import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMessage as ChatMessageType } from "../api/types";
import ConfidenceBadge from "./ConfidenceBadge";
import DomainBadge from "./DomainBadge";
import CitationsPanel from "./CitationsPanel";

interface Props {
  message: ChatMessageType;
  isStreaming?: boolean;
  StreamingCursor?: React.ComponentType;
}

// Tailwind prose-like styles passed as component overrides to react-markdown
const mdComponents: React.ComponentProps<typeof ReactMarkdown>["components"] = {
  p: ({ children }) => (
    <p className="text-slate-200 leading-relaxed mb-2 last:mb-0">{children}</p>
  ),
  strong: ({ children }) => (
    <strong className="text-slate-100 font-semibold">{children}</strong>
  ),
  em: ({ children }) => (
    <em className="text-slate-300 italic">{children}</em>
  ),
  h3: ({ children }) => (
    <h3 className="text-slate-100 font-semibold text-sm mt-3 mb-1.5 first:mt-0">{children}</h3>
  ),
  h4: ({ children }) => (
    <h4 className="text-slate-200 font-medium text-xs mt-2 mb-1">{children}</h4>
  ),
  ul: ({ children }) => (
    <ul className="space-y-1 my-2 pl-4">{children}</ul>
  ),
  ol: ({ children }) => (
    <ol className="space-y-1 my-2 pl-4 list-decimal list-outside">{children}</ol>
  ),
  li: ({ children }) => (
    <li className="text-slate-200 leading-relaxed marker:text-slate-500 list-disc">{children}</li>
  ),
  code: ({ children, className }) => {
    const isBlock = className?.includes("language-");
    return isBlock ? (
      <code className="block bg-slate-900 border border-slate-700 rounded-lg p-3 text-xs font-mono text-brand-300 overflow-x-auto my-2 whitespace-pre">
        {children}
      </code>
    ) : (
      <code className="bg-slate-900 border border-slate-700 rounded px-1 py-0.5 text-[11px] font-mono text-brand-300">
        {children}
      </code>
    );
  },
  blockquote: ({ children }) => (
    <blockquote className="border-l-2 border-brand-500/50 pl-3 my-2 text-slate-400 italic">
      {children}
    </blockquote>
  ),
  hr: () => <hr className="border-slate-700 my-3" />,
};

export default function ChatMessageComponent({ message, isStreaming, StreamingCursor }: Props) {
  const isUser = message.role === "user";
  const resp = message.response;

  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : "flex-row"} items-start mb-6`}>
      {/* Avatar */}
      <div
        className={`shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold ${
          isUser
            ? "bg-brand-600 text-white"
            : "bg-slate-700 text-slate-300"
        }`}
      >
        {isUser ? "U" : "AI"}
      </div>

      {/* Bubble */}
      <div className={`max-w-[80%] ${isUser ? "items-end" : "items-start"} flex flex-col gap-2`}>
        {isUser ? (
          <div className="bg-brand-600/20 border border-brand-500/30 rounded-2xl rounded-tr-sm px-4 py-3">
            <p className="text-sm text-slate-100">{message.content}</p>
          </div>
        ) : (
          <div className="bg-slate-800/80 border border-slate-700/60 rounded-2xl rounded-tl-sm px-4 py-4 w-full">
            {/* Metrics row */}
            {resp && (
              <div className="flex items-center gap-2 flex-wrap mb-3 pb-3 border-b border-slate-700/40">
                <DomainBadge domain={resp.domain} />
                <ConfidenceBadge
                  confidence={resp.confidence}
                  hallucinationRisk={resp.hallucination_risk}
                />
                {resp.injection_blocked && (
                  <span className="badge bg-red-500/20 text-red-400 border border-red-500/30 rounded-md text-[10px]">
                    INJECTION BLOCKED
                  </span>
                )}
                {resp.latency && (
                  <span className="text-[10px] text-slate-600 ml-auto font-mono">
                    {resp.latency.total_ms?.toFixed(0) ?? "—"}ms
                  </span>
                )}
              </div>
            )}

            {/* Answer text — rendered as Markdown while streaming or done */}
            <div className="text-sm">
              {message.content ? (
                <>
                  <ReactMarkdown remarkPlugins={[remarkGfm]} components={mdComponents}>
                    {message.content}
                  </ReactMarkdown>
                  {isStreaming && StreamingCursor && <StreamingCursor />}
                </>
              ) : isStreaming ? (
                /* Empty bubble while waiting for first token */
                <div className="flex gap-1 items-center h-5">
                  {[0, 1, 2].map((i) => (
                    <span
                      key={i}
                      className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce"
                      style={{ animationDelay: `${i * 150}ms` }}
                    />
                  ))}
                </div>
              ) : null}
            </div>

            {/* Citations */}
            {resp?.sources && resp.sources.length > 0 && (
              <CitationsPanel sources={resp.sources} />
            )}
          </div>
        )}

        {/* Timestamp */}
        <span className="text-[10px] text-slate-600 px-1">
          {message.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
        </span>
      </div>
    </div>
  );
}
