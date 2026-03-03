"use client";

/**
 * app/chat/page.tsx  DeepCoin AI Numismatic Chat v2
 * ====================================================
 * Enterprise-grade conversational interface over the 9,541-type Corpus
 * Nummorum knowledge base. Grounded answers, cited sources, Google search
 * fallback, animated design. No authentication required.
 *
 * WHY "use client": useState + useRef + useCallback  all browser-only.
 * FIX v2: chatQuery now uses classifyApiClient (direct FastAPI, 180s timeout)
 *         so long LLM inference calls don't hit the Next.js proxy timeout.
 */

import { useState, useRef, useEffect, useCallback, Suspense } from "react";
import { useSearchParams }                            from "next/navigation";
import { motion, AnimatePresence }                   from "framer-motion";
import Link                                          from "next/link";
import {
  Sparkles, Send, Bot, User, Loader2, ExternalLink,
  BookOpen, ChevronDown, ChevronUp, Coins, Copy, Check,
  ArrowRight, Database, Cpu, Zap, Globe,
} from "lucide-react";
import { chatQuery }       from "@/lib/api";
import type { ChatSource }  from "@/types/api";

/*  types  */

type Role    = "user" | "assistant";
type Message = {
  id:              string;
  role:            Role;
  content:         string;
  sources?:        ChatSource[];
  provider?:       string;
  error?:          boolean;
  userQuery?:      string;
};

/*  constants  */

const STARTERS = [
  { icon: "", text: "What silver coins were minted in Athens?" },
  { icon: "", text: "Tell me about tetradrachms from the Seleucid empire" },
  { icon: "", text: "What coin types show an eagle on the reverse?" },
  { icon: "", text: "Describe Roman denarii from Augustus' reign" },
  { icon: "", text: "Bronze coins from Maroneia, Thrace  what do we know?" },
  { icon: "", text: "Tell me about electrum coins from ancient Lydia" },
];

const CHUNK_LABELS: Record<string, string> = {
  identity: "Identity", obverse: "Obverse", reverse: "Reverse",
  material: "Material", context: "Context",
};

const CHUNK_COLORS: Record<string, string> = {
  identity: "#3b82f6", obverse: "#8b5cf6", reverse: "#06b6d4",
  material: "#f59e0b", context: "#10b981",
};

/*  helpers  */

function providerLabel(p?: string): { label: string; color: string } | null {
  if (!p) return null;
  if (p.startsWith("ollama"))      return { label: "Local AI  " + p.replace("ollama:", ""), color: "#10b981" };
  if (p.includes("gemini"))        return { label: "Gemini 2.5 Flash", color: "#8b5cf6" };
  if (p === "structured-fallback") return { label: "KB Direct", color: "#f59e0b" };
  if (p.startsWith("fallback"))    return { label: "Fallback", color: "#6b7280" };
  return { label: p, color: "#6b7280" };
}

/*  CopyButton  */

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      onClick={() => { navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 2000); }}
      className="flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-md transition-all"
      style={{ color: copied ? "#10b981" : "var(--text-muted)", background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.07)" }}
    >
      {copied ? <Check size={9} /> : <Copy size={9} />}
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

/*  SourceChip  */

function SourceChip({ source, idx }: { source: ChatSource; idx: number }) {
  const color = CHUNK_COLORS[source.chunk_type] ?? "#6b7280";
  const label = CHUNK_LABELS[source.chunk_type] ?? source.chunk_type;
  const cnId  = source.type_id?.match(/\d+/)?.[0];
  const cnUrl = cnId ? `https://www.corpus-nummorum.eu/types/${cnId}` : null;
  const pct   = Math.round((source.score ?? 0) * 100);
  return (
    <motion.div
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: idx * 0.06 }}
      className="rounded-xl p-3 flex flex-col gap-2"
      style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 min-w-0">
          <div className="w-1.5 h-1.5 rounded-full shrink-0" style={{ backgroundColor: color }} />
          <span className="text-[10px] font-bold uppercase tracking-wider truncate" style={{ color }}>
            CN {source.type_id}  {label}
          </span>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <div className="flex items-center gap-1">
            <div className="w-12 h-1 rounded-full bg-white/10 overflow-hidden">
              <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: color }} />
            </div>
            <span className="text-[9px] tabular-nums" style={{ color: "var(--text-muted)" }}>{pct}%</span>
          </div>
          {cnUrl && (
            <a href={cnUrl} target="_blank" rel="noopener noreferrer"
               className="flex items-center gap-0.5 text-[9px] hover:text-blue-300 transition-colors"
               style={{ color: "var(--text-muted)" }}>
              View <ExternalLink size={8} />
            </a>
          )}
        </div>
      </div>
      <p className="text-[11px] leading-relaxed line-clamp-3" style={{ color: "var(--text-secondary)" }}>
        {source.snippet}
      </p>
    </motion.div>
  );
}

/*  GoogleSearchCTA  */

function GoogleSearchCTA({ query }: { query: string }) {
  return (
    <div className="flex items-center gap-2 mt-1">
      <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>Continue research:</span>
      <a
        href={`https://www.google.com/search?q=${encodeURIComponent(query + " ancient coin numismatics")}`}
        target="_blank" rel="noopener noreferrer"
        className="flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-md transition-all hover:border-blue-400/50"
        style={{ color: "#60a5fa", background: "rgba(59,130,246,0.07)", border: "1px solid rgba(59,130,246,0.20)" }}
      >
        <Globe size={9} /> Google
      </a>
      <a
        href={`https://scholar.google.com/scholar?q=${encodeURIComponent(query + " ancient coin")}`}
        target="_blank" rel="noopener noreferrer"
        className="flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-md transition-all hover:border-emerald-400/50"
        style={{ color: "#34d399", background: "rgba(16,185,129,0.07)", border: "1px solid rgba(16,185,129,0.20)" }}
      >
        <BookOpen size={9} /> Scholar
      </a>
    </div>
  );
}

/*  MessageBubble  */

function MessageBubble({ msg }: { msg: Message }) {
  const [showSources, setShowSources] = useState(false);
  const isUser = msg.role === "user";
  const pInfo  = providerLabel(msg.provider);
  return (
    <motion.div
      initial={{ opacity: 0, y: 12, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ type: "spring", stiffness: 380, damping: 32 }}
      className={`flex gap-3 ${isUser ? "flex-row-reverse" : "flex-row"}`}
    >
      {/* Avatar */}
      <div
        className="shrink-0 w-8 h-8 rounded-2xl flex items-center justify-center mt-0.5"
        style={{
          background: isUser
            ? "linear-gradient(135deg, #d4a853 0%, #b8860b 100%)"
            : "linear-gradient(135deg, rgba(139,92,246,0.30) 0%, rgba(99,102,241,0.15) 100%)",
          border:     isUser ? "none" : "1px solid rgba(139,92,246,0.35)",
          boxShadow:  isUser ? "0 2px 8px rgba(212,168,83,0.30)" : "0 2px 8px rgba(139,92,246,0.20)",
        }}
      >
        {isUser ? <User size={13} style={{ color: "#0d1520" }} /> : <Bot size={13} style={{ color: "#a78bfa" }} />}
      </div>

      {/* Content */}
      <div className={`flex flex-col gap-1.5 max-w-[82%] ${isUser ? "items-end" : "items-start"}`}>
        <div
          className="rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap"
          style={{
            background: isUser
              ? "linear-gradient(135deg, rgba(212,168,83,0.14) 0%, rgba(180,140,60,0.08) 100%)"
              : msg.error ? "rgba(239,68,68,0.06)" : "var(--surface-1)",
            border: `1px solid ${isUser ? "rgba(212,168,83,0.28)" : msg.error ? "rgba(239,68,68,0.22)" : "var(--border)"}`,
            color:  "var(--text-primary)",
            boxShadow: isUser ? "0 1px 8px rgba(212,168,83,0.10)" : "0 1px 8px rgba(0,0,0,0.15)",
          }}
        >
          {msg.content}
        </div>

        {/* Assistant meta */}
        {!isUser && (
          <div className="flex items-center gap-2 px-1 flex-wrap">
            {pInfo && (
              <span className="text-[9px] font-semibold px-2 py-0.5 rounded-full"
                style={{ color: pInfo.color, background: pInfo.color + "18", border: `1px solid ${pInfo.color}28` }}>
                {pInfo.label}
              </span>
            )}
            <CopyButton text={msg.content} />
            {msg.userQuery && !msg.error && <GoogleSearchCTA query={msg.userQuery} />}
          </div>
        )}

        {/* Sources */}
        {msg.sources && msg.sources.length > 0 && (
          <div className="w-full">
            <button
              onClick={() => setShowSources(v => !v)}
              className="flex items-center gap-1.5 text-[10px] font-medium px-2 py-1 rounded-lg transition-all hover:bg-white/5"
              style={{ color: "var(--text-muted)" }}
            >
              <div className="flex -space-x-0.5 mr-0.5">
                {msg.sources.slice(0, 3).map((s, i) => (
                  <div key={i} className="w-2.5 h-2.5 rounded-full border border-[var(--bg-base)]"
                       style={{ backgroundColor: CHUNK_COLORS[s.chunk_type] ?? "#6b7280" }} />
                ))}
              </div>
              {msg.sources.length} source{msg.sources.length !== 1 ? "s" : ""} cited
              {showSources ? <ChevronUp size={10} className="ml-0.5" /> : <ChevronDown size={10} className="ml-0.5" />}
            </button>
            <AnimatePresence>
              {showSources && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="space-y-2 mt-2 overflow-hidden"
                >
                  {msg.sources.map((src, i) => <SourceChip key={i} source={src} idx={i} />)}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )}
      </div>
    </motion.div>
  );
}

/*  TypingIndicator  */

function TypingIndicator() {
  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="flex gap-3">
      <div className="shrink-0 w-8 h-8 rounded-2xl flex items-center justify-center"
           style={{ background: "linear-gradient(135deg, rgba(139,92,246,0.30) 0%, rgba(99,102,241,0.15) 100%)", border: "1px solid rgba(139,92,246,0.35)" }}>
        <Bot size={13} style={{ color: "#a78bfa" }} />
      </div>
      <div className="rounded-2xl px-4 py-3 flex items-center gap-1"
           style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}>
        {[0, 1, 2].map(i => (
          <motion.div key={i} className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: "#a78bfa" }}
            animate={{ y: [0, -4, 0], opacity: [0.4, 1, 0.4] }}
            transition={{ repeat: Infinity, duration: 1.0, delay: i * 0.18, ease: "easeInOut" }} />
        ))}
      </div>
    </motion.div>
  );
}

/*  EmptyState  */

function EmptyState({ onSelect }: { onSelect: (q: string) => void }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="flex flex-col items-center gap-8 py-6"
    >
      <div className="flex flex-col items-center gap-4">
        <div className="relative">
          <div className="absolute inset-0 rounded-3xl blur-2xl opacity-40"
               style={{ background: "radial-gradient(circle, #8b5cf6, transparent 70%)", transform: "scale(1.5)" }} />
          <div className="relative w-20 h-20 rounded-3xl flex items-center justify-center"
               style={{ background: "linear-gradient(135deg, rgba(139,92,246,0.25) 0%, rgba(99,102,241,0.12) 100%)", border: "1px solid rgba(139,92,246,0.35)", boxShadow: "0 8px 32px rgba(139,92,246,0.20)" }}>
            <Sparkles size={36} style={{ color: "#a78bfa" }} />
          </div>
        </div>
        <div className="text-center">
          <h2 className="text-2xl font-black" style={{ color: "var(--text-primary)" }}>DeepCoin AI</h2>
          <p className="text-sm mt-1" style={{ color: "var(--text-muted)" }}>
            Numismatic Q&A  grounded in 9,541 Corpus Nummorum coin types
          </p>
        </div>
        <div className="flex flex-wrap justify-center gap-2">
          {[
            { icon: <Database size={11} />, label: "47,705 KB chunks" },
            { icon: <Coins     size={11} />, label: "9,541 coin types" },
            { icon: <Cpu       size={11} />, label: "BM25 + vector search" },
            { icon: <Zap       size={11} />, label: "Source-cited answers" },
          ].map(({ icon, label }) => (
            <span key={label} className="flex items-center gap-1.5 text-[10px] font-medium px-3 py-1 rounded-full"
                  style={{ background: "rgba(139,92,246,0.10)", color: "#c4b5fd", border: "1px solid rgba(139,92,246,0.20)" }}>
              {icon} {label}
            </span>
          ))}
        </div>
      </div>

      <div className="w-full max-w-xl">
        <p className="text-xs text-center mb-4 uppercase tracking-wider font-semibold" style={{ color: "var(--text-muted)" }}>
          Try asking
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
          {STARTERS.map(s => (
            <motion.button
              key={s.text}
              whileHover={{ scale: 1.02, borderColor: "rgba(139,92,246,0.45)" }}
              whileTap={{ scale: 0.98 }}
              onClick={() => onSelect(s.text)}
              className="group text-left flex items-start gap-3 px-4 py-3 rounded-xl transition-colors"
              style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}
            >
              <span className="text-base shrink-0 mt-0.5">{s.icon}</span>
              <span className="text-xs leading-relaxed flex-1" style={{ color: "var(--text-secondary)" }}>{s.text}</span>
              <ArrowRight size={11} className="shrink-0 mt-0.5 opacity-0 group-hover:opacity-60 transition-opacity" style={{ color: "#a78bfa" }} />
            </motion.button>
          ))}
        </div>
      </div>
    </motion.div>
  );
}

/*  main page  */

function ChatPageInner() {
  const searchParams            = useSearchParams();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input,    setInput]    = useState("");
  const [loading,  setLoading]  = useState(false);
  const bottomRef               = useRef<HTMLDivElement>(null);
  const inputRef                = useRef<HTMLInputElement>(null);
  const didAutoQuery            = useRef(false);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, loading]);

  const handleSubmit = useCallback(async (query: string) => {
    if (!query.trim() || loading) return;
    const q = query.trim();
    setMessages(prev => [...prev, { id: crypto.randomUUID(), role: "user" as const, content: q }]);
    setInput("");
    setLoading(true);
    try {
      const res = await chatQuery(q, 5);
      setMessages(prev => [...prev, {
        id: crypto.randomUUID(), role: "assistant" as const,
        content: res.answer, sources: res.sources, provider: res.provider, userQuery: q,
      }]);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setMessages(prev => [...prev, {
        id: crypto.randomUUID(), role: "assistant" as const,
        content: `Sorry, I couldn't reach the knowledge base right now.\n\n${msg}\n\nPlease check that the DeepCoin backend is running and try again.`,
        error: true,
      }]);
    } finally {
      setLoading(false);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [loading]);

  // Auto-fire query when navigated here with ?q= (e.g. from AnalysisPanel low-confidence CTA)
  useEffect(() => {
    const q = searchParams.get("q");
    if (q && !didAutoQuery.current) {
      didAutoQuery.current = true;
      handleSubmit(q);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams]);

  const isEmpty = messages.length === 0;

  return (
    <div className="flex flex-col h-[calc(100vh-80px)] max-w-3xl mx-auto w-full">

      {/* Header */}
      <div className="flex items-center justify-between py-4 px-1 shrink-0 border-b" style={{ borderColor: "var(--border)" }}>
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center"
               style={{ background: "linear-gradient(135deg, rgba(139,92,246,0.25) 0%, rgba(99,102,241,0.10) 100%)", border: "1px solid rgba(139,92,246,0.30)", boxShadow: "0 2px 12px rgba(139,92,246,0.15)" }}>
            <Sparkles size={16} style={{ color: "#a78bfa" }} />
          </div>
          <div>
            <h1 className="text-sm font-black" style={{ color: "var(--text-primary)" }}>DeepCoin AI</h1>
            <p className="text-[10px]" style={{ color: "var(--text-muted)" }}>Grounded answers from the Corpus Nummorum</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {!isEmpty && (
            <button onClick={() => setMessages([])}
              className="text-[10px] px-3 py-1 rounded-lg transition-all hover:bg-white/5"
              style={{ color: "var(--text-muted)", border: "1px solid var(--border)" }}>
              New chat
            </button>
          )}
          <Link href="/explore"
            className="hidden sm:flex items-center gap-1 text-[10px] px-3 py-1 rounded-lg transition-all hover:bg-white/5"
            style={{ color: "var(--text-muted)", border: "1px solid var(--border)" }}>
            <Coins size={10} /> Explore
          </Link>
        </div>
      </div>

      {/* Thread */}
      <div className="flex-1 overflow-y-auto py-6 space-y-5">
        {isEmpty
          ? <EmptyState onSelect={handleSubmit} />
          : (<>
              {messages.map(msg => <MessageBubble key={msg.id} msg={msg} />)}
              {loading && <TypingIndicator />}
            </>)}
        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <div className="shrink-0 pb-4 pt-2 border-t" style={{ borderColor: "var(--border)" }}>
        <form onSubmit={e => { e.preventDefault(); handleSubmit(input); }} className="relative flex items-end gap-2">
          <div className="relative flex-1">
            <input
              ref={inputRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              disabled={loading}
              placeholder="Ask about a dynasty, mint, denomination, iconography"
              onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(input); } }}
              className="w-full px-4 py-3 text-sm rounded-xl outline-none transition-all disabled:opacity-50"
              style={{
                background: "var(--surface-1)",
                border:     `1px solid ${input ? "rgba(139,92,246,0.50)" : "var(--border)"}`,
                color:      "var(--text-primary)",
                boxShadow:  input ? "0 0 0 3px rgba(139,92,246,0.10)" : "none",
              }}
            />
            {input.length > 200 && (
              <span className="absolute right-3 top-1/2 -translate-y-1/2 text-[10px] tabular-nums"
                    style={{ color: input.length > 450 ? "#ef4444" : "var(--text-muted)" }}>
                {500 - input.length}
              </span>
            )}
          </div>
          <motion.button type="submit" disabled={!input.trim() || loading} whileTap={{ scale: 0.95 }}
            className="shrink-0 w-11 h-11 rounded-xl flex items-center justify-center transition-all disabled:opacity-30"
            style={{ background: "linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)", boxShadow: input.trim() && !loading ? "0 4px 14px rgba(139,92,246,0.40)" : "none" }}>
            {loading ? <Loader2 size={16} className="animate-spin text-white" /> : <Send size={15} className="text-white ml-0.5" />}
          </motion.button>
        </form>
        <p className="text-center text-[9px] mt-2" style={{ color: "var(--text-muted)" }}>
          Answers grounded in{" "}
          <a href="https://www.corpus-nummorum.eu" target="_blank" rel="noopener noreferrer" className="underline hover:text-blue-300 transition-colors">Corpus Nummorum</a>
          {" "} Cite sources before academic use {" "}
          <Link href="/docs" className="underline hover:text-blue-300 transition-colors">API docs</Link>
        </p>
      </div>
    </div>
  );
}
export default function ChatPage() {
  return (
    <Suspense fallback={<div className="flex items-center justify-center h-[calc(100vh-80px)]" style={{ color: "var(--text-muted)" }}>Loading…</div>}>
      <ChatPageInner />
    </Suspense>
  );
}