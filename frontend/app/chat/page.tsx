"use client";

/**
 * app/chat/page.tsx — DeepCoin AI Chat
 * ======================================
 * WHAT: A conversational interface over the 9,541-type Corpus Nummorum
 *       knowledge base. Ask any numismatic question in natural language and
 *       receive a grounded, citation-backed answer.
 *
 * WHY this page exists:
 *   The /api/chat endpoint made the KB conversational, but a bare API call
 *   isn't user-friendly. This page wraps it in a familiar chat UI — message
 *   thread, input form, source chips — so museum curators, researchers, and
 *   students can query the KB without writing code.
 *
 * WHY "use client":
 *   Uses useState (message history, input, loading), useRef (scroll anchor),
 *   and DOM event handlers — all browser-only.
 *
 * NO AUTH REQUIRED:
 *   The chat endpoint is public (same as /api/explore).
 *   Any visitor can ask questions without signing up.
 *   If the LLM is unavailable, the structured-fallback answer still works.
 *
 * ARCHITECTURE:
 *   User types a question →
 *   POST /api/chat { query, n_sources: 5 } →
 *   FastAPI: RAG search → [CONTEXT 1-5] → LLM (Ollama / GitHub / Google) →
 *   { answer, sources, provider } →
 *   Message added to thread with expandable source chips
 */

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence }      from "framer-motion";
import Link                             from "next/link";
import {
  Sparkles, Send, Bot, User, Loader2,
  ExternalLink, BookOpen, ChevronDown, ChevronUp, Coins,
} from "lucide-react";
import { chatQuery }     from "@/lib/api";
import type { ChatSource } from "@/types/api";

/* ── types ────────────────────────────────────────────────────────────────── */

type Role    = "user" | "assistant";
type Message = {
  id:        string;
  role:      Role;
  content:   string;
  sources?:  ChatSource[];
  provider?: string;
  error?:    boolean;
};

/* ── starter questions ────────────────────────────────────────────────────── */

const STARTER_QUESTIONS = [
  "What silver coins were minted in Athens?",
  "Tell me about tetradrachms from the Seleucid empire",
  "What are bronze coins from Maroneia, Thrace?",
  "Describe Roman denarii from the time of Augustus",
  "What coin types show an eagle on the reverse?",
  "Tell me about electrum coins from ancient Lydia",
];

/* ── chunk-type label map ─────────────────────────────────────────────────── */

const CHUNK_LABELS: Record<string, string> = {
  identity:  "Identity",
  obverse:   "Obverse",
  reverse:   "Reverse",
  material:  "Material",
  context:   "Context",
};

/* ── source chip ──────────────────────────────────────────────────────────── */

function SourceChip({ source }: { source: ChatSource }) {
  const pct   = Math.round(source.score * 100);
  const label = CHUNK_LABELS[source.chunk_type] ?? source.chunk_type;
  const cnId  = source.type_id?.match(/\d+/)?.[0];
  const cnUrl = cnId ? `https://www.corpus-nummorum.eu/types/${cnId}` : null;

  return (
    <div
      className="flex flex-col gap-1 rounded-lg p-3"
      style={{ backgroundColor: "var(--surface-2)", border: "1px solid var(--border)" }}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <BookOpen size={10} style={{ color: "#8b5cf6" }} />
          <span className="text-[10px] font-bold uppercase tracking-wide" style={{ color: "#8b5cf6" }}>
            CN {source.type_id} · {label}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] tabular-nums" style={{ color: "var(--text-muted)" }}>{pct}% match</span>
          {cnUrl && (
            <a href={cnUrl} target="_blank" rel="noopener noreferrer"
               className="flex items-center gap-0.5 text-[10px] hover:underline"
               style={{ color: "var(--text-muted)" }}>
              View <ExternalLink size={9} />
            </a>
          )}
        </div>
      </div>
      <p className="text-[11px] leading-relaxed line-clamp-3" style={{ color: "var(--text-secondary)" }}>
        {source.snippet}
      </p>
    </div>
  );
}

/* ── message bubble ───────────────────────────────────────────────────────── */

function MessageBubble({ msg }: { msg: Message }) {
  const [showSources, setShowSources] = useState(false);
  const isUser = msg.role === "user";

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex gap-3 ${isUser ? "flex-row-reverse" : "flex-row"}`}
    >
      {/* Avatar */}
      <div
        className="shrink-0 w-8 h-8 rounded-full flex items-center justify-center"
        style={{
          backgroundColor: isUser ? "var(--brand-gold)" : "rgba(139,92,246,0.15)",
          border:          isUser ? "none" : "1px solid rgba(139,92,246,0.3)",
        }}
      >
        {isUser
          ? <User size={14} style={{ color: "#0d1520" }} />
          : <Bot  size={14} style={{ color: "#8b5cf6" }} />
        }
      </div>

      {/* Content */}
      <div className={`flex flex-col gap-2 max-w-[78%] ${isUser ? "items-end" : "items-start"}`}>
        <div
          className="rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap"
          style={{
            backgroundColor: isUser
              ? "rgba(212,168,83,0.12)"
              : msg.error
                ? "rgba(239,68,68,0.08)"
                : "var(--surface-1)",
            border: `1px solid ${
              isUser ? "rgba(212,168,83,0.25)" : msg.error ? "rgba(239,68,68,0.2)" : "var(--border)"
            }`,
            color: "var(--text-primary)",
          }}
        >
          {msg.content}
        </div>

        {/* Provider badge */}
        {msg.provider && !isUser && (
          <span className="text-[10px] px-2 py-0.5 rounded-full"
                style={{ backgroundColor: "var(--surface-2)", color: "var(--text-muted)", border: "1px solid var(--border)" }}>
            {msg.provider}
          </span>
        )}

        {/* Sources toggle */}
        {msg.sources && msg.sources.length > 0 && (
          <div className="w-full space-y-2">
            <button
              onClick={() => setShowSources(v => !v)}
              className="flex items-center gap-1.5 text-[10px] font-medium hover:opacity-70 transition-opacity"
              style={{ color: "var(--text-muted)" }}
            >
              {showSources ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
              {msg.sources.length} source{msg.sources.length !== 1 ? "s" : ""} from Corpus Nummorum
            </button>
            <AnimatePresence>
              {showSources && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="space-y-2 overflow-hidden"
                >
                  {msg.sources.map((src, i) => (
                    <SourceChip key={i} source={src} />
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )}
      </div>
    </motion.div>
  );
}

/* ── main page ────────────────────────────────────────────────────────────── */

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input,    setInput]    = useState("");
  const [loading,  setLoading]  = useState(false);
  const bottomRef               = useRef<HTMLDivElement>(null);

  /* auto-scroll on new message */
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function handleSubmit(query: string) {
    if (!query.trim() || loading) return;

    const userMsg: Message = {
      id:      crypto.randomUUID(),
      role:    "user",
      content: query.trim(),
    };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await chatQuery(query.trim(), 5);
      const aiMsg: Message = {
        id:       crypto.randomUUID(),
        role:     "assistant",
        content:  res.answer,
        sources:  res.sources,
        provider: res.provider,
      };
      setMessages(prev => [...prev, aiMsg]);
    } catch {
      const errMsg: Message = {
        id:      crypto.randomUUID(),
        role:    "assistant",
        content: "Sorry, I couldn't reach the knowledge base right now. Please try again in a moment.",
        error:   true,
      };
      setMessages(prev => [...prev, errMsg]);
    } finally {
      setLoading(false);
    }
  }

  const isEmpty = messages.length === 0;

  return (
    <div className="flex flex-col h-[calc(100vh-80px)] max-w-3xl pb-2">

      {/* Header */}
      <div className="flex items-center gap-3 py-6 shrink-0">
        <div className="w-10 h-10 rounded-2xl flex items-center justify-center"
             style={{ backgroundColor: "rgba(139,92,246,0.15)", border: "1px solid rgba(139,92,246,0.3)" }}>
          <Sparkles size={18} style={{ color: "#8b5cf6" }} />
        </div>
        <div>
          <h1 className="text-lg font-black" style={{ color: "var(--text-primary)" }}>DeepCoin AI</h1>
          <p className="text-xs" style={{ color: "var(--text-muted)" }}>
            9,541 Corpus Nummorum coin types · answers grounded in real numismatic data
          </p>
        </div>
        <Link
          href="/explore"
          className="ml-auto text-xs flex items-center gap-1 hover:underline shrink-0"
          style={{ color: "var(--text-muted)" }}
        >
          <Coins size={11} /> Explore gallery
        </Link>
      </div>

      {/* Message thread */}
      <div className="flex-1 overflow-y-auto space-y-5 pr-1">

        {/* Empty state: starter questions */}
        {isEmpty && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="py-8 space-y-6"
          >
            <div className="text-center space-y-2">
              <Sparkles size={32} className="mx-auto" style={{ color: "rgba(139,92,246,0.4)" }} />
              <p className="text-sm font-semibold" style={{ color: "var(--text-secondary)" }}>
                Ask anything about ancient coins
              </p>
              <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                Denominations, dynasties, mint cities, iconography, materials — all grounded in the Corpus Nummorum.
              </p>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {STARTER_QUESTIONS.map(q => (
                <button
                  key={q}
                  onClick={() => handleSubmit(q)}
                  className="text-left text-xs px-4 py-3 rounded-xl transition-colors hover:border-[rgba(139,92,246,0.4)]"
                  style={{
                    backgroundColor: "var(--surface-1)",
                    border:          "1px solid var(--border)",
                    color:           "var(--text-secondary)",
                  }}
                >
                  {q}
                </button>
              ))}
            </div>
          </motion.div>
        )}

        {messages.map(msg => (
          <MessageBubble key={msg.id} msg={msg} />
        ))}

        {/* Typing indicator */}
        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex gap-3"
          >
            <div className="w-8 h-8 rounded-full flex items-center justify-center"
                 style={{ backgroundColor: "rgba(139,92,246,0.15)", border: "1px solid rgba(139,92,246,0.3)" }}>
              <Bot size={14} style={{ color: "#8b5cf6" }} />
            </div>
            <div className="rounded-2xl px-4 py-3 flex items-center gap-1.5"
                 style={{ backgroundColor: "var(--surface-1)", border: "1px solid var(--border)" }}>
              {[0, 1, 2].map(i => (
                <motion.div
                  key={i}
                  className="w-1.5 h-1.5 rounded-full"
                  style={{ backgroundColor: "#8b5cf6" }}
                  animate={{ opacity: [0.3, 1, 0.3] }}
                  transition={{ repeat: Infinity, duration: 1.2, delay: i * 0.2 }}
                />
              ))}
            </div>
          </motion.div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <form
        onSubmit={e => { e.preventDefault(); handleSubmit(input); }}
        className="flex gap-2 pt-3 shrink-0"
      >
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          disabled={loading}
          placeholder="Ask about a dynasty, mint, denomination, or iconography…"
          className="flex-1 px-4 py-3 text-sm rounded-xl outline-none transition-colors disabled:opacity-50"
          style={{
            backgroundColor: "var(--surface-1)",
            border:          "1px solid var(--border)",
            color:           "var(--text-primary)",
          }}
        />
        <button
          type="submit"
          disabled={!input.trim() || loading}
          className="px-4 py-3 rounded-xl transition-opacity disabled:opacity-30 hover:opacity-80"
          style={{ backgroundColor: "#8b5cf6", color: "#fff", flexShrink: 0 }}
        >
          {loading
            ? <Loader2 size={16} className="animate-spin" />
            : <Send size={16} />
          }
        </button>
      </form>
    </div>
  );
}
