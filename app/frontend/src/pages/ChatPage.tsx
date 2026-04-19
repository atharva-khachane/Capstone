import React, { useCallback, useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { streamQuery, fetchStatus, loadSession } from "../api/client";
import type { QueryResponse } from "../api/types";
import { useChatContext } from "../context/ChatContext";
import ChatMessageComponent from "../components/ChatMessage";

const SUGGESTIONS = [
  "What are the rules for budget sanction under GFR?",
  "What is the QCBS method for evaluating procurement proposals?",
  "What is the EMD requirement for tenders?",
  "What are the financial limits for procurement without a tender?",
  "Describe the telemetry data decoding system.",
];

function StreamingCursor() {
  return (
    <span className="inline-block w-[2px] h-[1em] bg-brand-400 align-middle ml-0.5 animate-pulse" />
  );
}

function IngestBanner() {
  const { data } = useQuery({
    queryKey: ["status"],
    queryFn: fetchStatus,
    refetchInterval: 5000,
  });

  if (!data || (data.ready && !data.ingest_running)) return null;

  return (
    <div className="shrink-0 px-6 py-2 bg-amber-500/10 border-b border-amber-500/20 flex items-center gap-2">
      <svg className="w-3 h-3 text-amber-400 animate-spin shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
      </svg>
      <span className="text-xs text-amber-400">
        {data.ingest_running
          ? "Re-indexing documents in background — existing queries still work"
          : "Pipeline warming up, queries will be available shortly…"}
      </span>
    </div>
  );
}

export default function ChatPage() {
  const session = loadSession();
  const { messages, setMessages, isStreaming, setIsStreaming, streamingId, setStreamingId, cancelRef } =
    useChatContext();

  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const inputValueRef = useRef("");  // track without re-render during stream

  // Keep a stable ref to input state for the handler
  const [input, setInputState] = React.useState("");
  const setInput = (v: string) => { inputValueRef.current = v; setInputState(v); };

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = useCallback(
    (question: string) => {
      const q = question.trim();
      if (!q || isStreaming) return;

      const userMsg = {
        id: crypto.randomUUID(),
        role: "user" as const,
        content: q,
        timestamp: new Date(),
      };
      const assistantId = crypto.randomUUID();
      const assistantMsg = {
        id: assistantId,
        role: "assistant" as const,
        content: "",
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setInput("");
      setIsStreaming(true);
      setStreamingId(assistantId);

      let accumulated = "";

      const cancel = streamQuery(
        {
          question: q,
          user_id: session?.user_id ?? "guest",
          role: session?.role ?? "guest",
          session_id: session?.session_id ?? "",
          generate_answer: true,
          enable_reranking: true,
        },
        (event) => {
          if (event.type === "token") {
            accumulated += event.token;
            setMessages((prev) =>
              prev.map((m) => (m.id === assistantId ? { ...m, content: accumulated } : m))
            );
          } else if (event.type === "done") {
            const { type: _t, ...responseData } = event as { type: string } & QueryResponse;
            const finalContent =
              responseData.answer?.trim() ||
              accumulated.trim() ||
              "I cannot provide an answer from the provided documents.";
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, content: finalContent, response: { ...(responseData as QueryResponse), answer: finalContent } }
                  : m
              )
            );
            setIsStreaming(false);
            setStreamingId(null);
          } else if (event.type === "error") {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, content: `Error: ${event.message}` } : m
              )
            );
            setIsStreaming(false);
            setStreamingId(null);
          }
        }
      );

      cancelRef.current = cancel;
    },
    [isStreaming, session, setMessages, setIsStreaming, setStreamingId, cancelRef]
  );

  const handleStop = () => {
    cancelRef.current?.();
    setIsStreaming(false);
    setStreamingId(null);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend(input);
    }
  };

  const handleClear = () => {
    handleStop();
    setMessages([]);
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="shrink-0 flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-900/50 backdrop-blur-sm">
        <div>
          <h1 className="text-sm font-semibold text-slate-200">Document Q&A</h1>
          <p className="text-xs text-slate-500">
            Querying GFR · Procurement · Technical documents
          </p>
        </div>
        {messages.length > 0 && (
          <button onClick={handleClear} className="btn-ghost text-xs text-slate-500">
            Clear chat
          </button>
        )}
      </div>

      {/* Ingest status banner — shown whenever pipeline is busy */}
      <IngestBanner />

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-6">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-6 text-center">
            <div className="w-12 h-12 bg-slate-800 rounded-2xl flex items-center justify-center">
              <svg className="w-6 h-6 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                  d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
            </div>
            <div>
              <h2 className="text-base font-semibold text-slate-300">Ask about ISRO documents</h2>
              <p className="text-xs text-slate-500 mt-1 max-w-xs">
                Query GFR rules, procurement procedures, and technical documentation with cited answers.
              </p>
            </div>
            <div className="flex flex-col gap-2 w-full max-w-lg">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => handleSend(s)}
                  className="text-left text-xs text-slate-400 bg-slate-800/60 border border-slate-700/60 rounded-xl px-4 py-3 hover:bg-slate-800 hover:border-slate-600 hover:text-slate-200 transition-all"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((msg) => (
              <ChatMessageComponent
                key={msg.id}
                message={msg}
                isStreaming={msg.id === streamingId}
                StreamingCursor={StreamingCursor}
              />
            ))}
            <div ref={bottomRef} />
          </>
        )}
      </div>

      {/* Input bar */}
      <div className="shrink-0 border-t border-slate-800 px-6 py-4 bg-slate-900/50 backdrop-blur-sm">
        <div className="flex gap-3 items-end max-w-4xl mx-auto">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about GFR rules, procurement, technical specs…"
            rows={1}
            disabled={isStreaming}
            className="input flex-1 resize-none overflow-hidden leading-relaxed py-3"
            style={{ minHeight: "44px", maxHeight: "120px" }}
            onInput={(e) => {
              const el = e.currentTarget;
              el.style.height = "auto";
              el.style.height = Math.min(el.scrollHeight, 120) + "px";
            }}
          />
          {isStreaming ? (
            <button
              onClick={handleStop}
              className="btn-ghost shrink-0 h-11 w-11 p-0 border border-red-500/40 text-red-400 hover:bg-red-500/10"
              title="Stop generation"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                <rect x="6" y="6" width="12" height="12" rx="1" />
              </svg>
            </button>
          ) : (
            <button
              onClick={() => handleSend(input)}
              disabled={!input.trim()}
              className="btn-primary shrink-0 h-11 w-11 p-0"
              title="Send (Enter)"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            </button>
          )}
        </div>
        <p className="text-[10px] text-slate-600 text-center mt-2">
          Press Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  );
}
