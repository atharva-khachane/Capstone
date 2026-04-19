import React, { createContext, useContext, useRef, useState } from "react";
import type { ChatMessage } from "../api/types";

interface ChatContextValue {
  messages: ChatMessage[];
  setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>;
  isStreaming: boolean;
  setIsStreaming: React.Dispatch<React.SetStateAction<boolean>>;
  streamingId: string | null;
  setStreamingId: React.Dispatch<React.SetStateAction<string | null>>;
  cancelRef: React.MutableRefObject<(() => void) | null>;
  resetChat: () => void;
}

const ChatContext = createContext<ChatContextValue | null>(null);

export function ChatProvider({ children }: { children: React.ReactNode }) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingId, setStreamingId] = useState<string | null>(null);
  const cancelRef = useRef<(() => void) | null>(null);

  const resetChat = () => {
    cancelRef.current?.();
    cancelRef.current = null;
    setIsStreaming(false);
    setStreamingId(null);
    setMessages([]);
  };

  return (
    <ChatContext.Provider
      value={{ messages, setMessages, isStreaming, setIsStreaming, streamingId, setStreamingId, cancelRef, resetChat }}
    >
      {children}
    </ChatContext.Provider>
  );
}

export function useChatContext(): ChatContextValue {
  const ctx = useContext(ChatContext);
  if (!ctx) throw new Error("useChatContext must be used inside <ChatProvider>");
  return ctx;
}
