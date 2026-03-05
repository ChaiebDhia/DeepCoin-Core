"use client";

/**
 * app/chat/page.tsx  DeepCoin AI Numismatic Chat v3
 * ====================================================
 * Enterprise-grade conversational interface over the 9,541-type Corpus
 * Nummorum knowledge base. Grounded answers, cited sources, Google search
 * fallback, animated design. No authentication required for chatting.
 *
 * v3 additions:
 *  - Left-side history sidebar for authenticated users (collapses on mobile)
 *  - Auto-save on first assistant reply (createChatSession)
 *  - Append on subsequent replies (appendChatSession)
 *  - Load a past session by clicking it in the sidebar
 *  - Delete a past session from the sidebar
 *  - "New chat" clears currentSessionId so the next reply starts a new session
 *
 * WHY "use client": useState + useRef + useCallback + useSession — all browser-only.
 */

import { useState, useRef, useEffect, useCallback, Suspense } from "react";
import { useSearchParams }                            from "next/navigation";
import { useSession }                                 from "next-auth/react";
import { motion, AnimatePresence }                   from "framer-motion";
import Link                                          from "next/link";
import {
  Sparkles, Send, Bot, User, Loader2, ExternalLink,
  BookOpen, ChevronDown, ChevronUp, Coins, Copy, Check,
  ArrowRight, Database, Cpu, Zap, Globe,
  History, Trash2, PanelLeftOpen, PanelLeftClose, MessageSquarePlus,
  AlertCircle,
} from "lucide-react";
import {
  chatQueryStream, listChatSessions, getChatSession,
  createChatSession, appendChatSession, deleteChatSession,
} from "@/lib/api";
import type { ChatSource, ChatSessionSummary, ChatMessageRecord } from "@/types/api";

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
  /** True while SSE token stream is still open for this message. */
  streaming?:      boolean;
};

/*  constants  */

/**
 * Module-level navigation cache.
 *
 * WHY module-level (not useState / Zustand):
 *   React component state is destroyed when the component unmounts (e.g.
 *   when the user navigates to /history and back to /chat).  The module
 *   itself stays loaded for the lifetime of the browser tab — a JS module
 *   is never garbage-collected while the page is open.  Writing conversation
 *   state here means navigating away and back restores the exact same chat
 *   session WITHOUT requiring a new server round-trip or a global Zustand store.
 *
 *   sessionStorage would also work but requires JSON serialisation/deserialisation
 *   and loses ChatSource objects that contain nested arrays.  The module ref
 *   stores the live objects directly.
 *
 * LIFECYCLE:  cleared when the user clicks "New chat" or starts a fresh session.
 */
const _chatCache: {
  messages:         Message[];
  currentSessionId: string | null;
  input:            string;
} = { messages: [], currentSessionId: null, input: "" };

const STARTERS = [
  { icon: "", text: "What silver coins were minted in Athens?" },
  { icon: "", text: "Tell me about tetradrachms from the Seleucid empire" },
  { icon: "", text: "What coin types show an eagle on the reverse?" },
  { icon: "", text: "Describe Roman denarii from Augustus' reign" },
  { icon: "", text: "Bronze coins from Maroneia, Thrace  what do we know?" },
  { icon: "", text: "Tell me about electrum coins from ancient Lydia" },
];

const CHUNK_LABELS: Record<string, string> = {
  identity: "ID", obverse: "Obverse", reverse: "Reverse",
  material: "Material", context: "Hist.",
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

function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60_000);
  if (mins < 1)    return "just now";
  if (mins < 60)   return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs  < 24)   return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 7)    return `${days}d ago`;
  return new Date(iso).toLocaleDateString();
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
               title={`View CN ${source.type_id} on corpus-nummorum.eu (opens in new tab)`}
               className="flex items-center gap-0.5 text-[9px] px-1.5 py-0.5 rounded-md font-medium transition-colors hover:text-blue-300 hover:bg-blue-400/10"
               style={{ color: "#60a5fa", border: "1px solid rgba(96,165,250,0.20)" }}>
              CN ↗
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
          {msg.content || msg.streaming
            ? (
              <>
                {msg.content || (
                  <span className="opacity-40 text-xs">Thinking…</span>
                )}
                {msg.streaming && (
                  /* Blinking cursor — visible while SSE token stream is open */
                  <span
                    className="inline-block w-[2px] h-[1em] align-middle ml-[2px] rounded-full animate-pulse"
                    style={{ background: "rgba(167,139,250,0.85)" }}
                  />
                )}
              </>
            )
            : null
          }
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

/*  HistorySidebar  */

interface HistorySidebarProps {
  sessions:         ChatSessionSummary[];
  activeId:         string | null;
  onSelect:         (id: string) => void;
  onDelete:         (id: string) => void;
  onNewChat:        () => void;
  loadingSession:   boolean;
}

function HistorySidebar({
  sessions, activeId, onSelect, onDelete, onNewChat, loadingSession,
}: HistorySidebarProps) {
  return (
    <div
      className="flex flex-col h-full"
      style={{ background: "var(--surface-1)", borderRight: "1px solid var(--border)" }}
    >
      {/* Sidebar header */}
      <div className="flex items-center justify-between px-3 py-3 shrink-0 border-b" style={{ borderColor: "var(--border)" }}>
        <div className="flex items-center gap-2">
          <History size={13} style={{ color: "#a78bfa" }} />
          <span className="text-[11px] font-bold" style={{ color: "var(--text-secondary)" }}>Chat History</span>
        </div>
        <button
          onClick={onNewChat}
          title="New chat"
          className="flex items-center gap-1 text-[10px] px-2 py-1 rounded-lg transition-all hover:bg-white/5"
          style={{ color: "#a78bfa", border: "1px solid rgba(139,92,246,0.25)" }}
        >
          <MessageSquarePlus size={11} />
        </button>
      </div>

      {/* Session list */}
      <div className="flex-1 overflow-y-auto py-2 space-y-0.5 px-2">
        {sessions.length === 0 && (
          <p className="text-[10px] text-center py-6" style={{ color: "var(--text-muted)" }}>
            No saved chats yet.<br />Start a conversation to save.
          </p>
        )}
        {sessions.map(s => (
          <div
            key={s.id}
            className="group flex items-start gap-2 px-2 py-2 rounded-lg cursor-pointer transition-all"
            style={{
              background: s.id === activeId ? "rgba(139,92,246,0.12)" : "transparent",
              border:     s.id === activeId ? "1px solid rgba(139,92,246,0.25)" : "1px solid transparent",
            }}
            onClick={() => onSelect(s.id)}
          >
            <div className="flex-1 min-w-0">
              <p className="text-[11px] font-medium truncate leading-tight" style={{ color: "var(--text-primary)" }}>
                {s.title}
              </p>
              <p className="text-[9px] mt-0.5 flex items-center gap-1.5" style={{ color: "var(--text-muted)" }}>
                <span>{relativeTime(s.updated_at)}</span>
                <span style={{ color: "var(--border)" }}>·</span>
                <span>{s.msg_count} msg{s.msg_count !== 1 ? "s" : ""}</span>
              </p>
            </div>
            <button
              onClick={e => { e.stopPropagation(); onDelete(s.id); }}
              className="shrink-0 opacity-0 group-hover:opacity-70 hover:!opacity-100 transition-all p-1 rounded-md"
              style={{ color: "#ef4444" }}
              title="Delete this chat"
            >
              <Trash2 size={10} />
            </button>
            {loadingSession && s.id === activeId && (
              <Loader2 size={10} className="shrink-0 animate-spin" style={{ color: "#a78bfa" }} />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

/*  main page  */

function ChatPageInner() {
  const searchParams            = useSearchParams();
  const { data: session }       = useSession();
  const isAuthed                = !!session?.user;

  const [messages, setMessages] = useState<Message[]>(_chatCache.messages);
  const [input,    setInput]    = useState(_chatCache.input);
  const [loading,  setLoading]  = useState(false);

  // History state
  const [sidebarOpen,    setSidebarOpen]    = useState(false);
  const [sessions,       setSessions]       = useState<ChatSessionSummary[]>([]);
  const [sessionsLoaded, setSessionsLoaded] = useState(false);
  const [loadingSession, setLoadingSession] = useState(false);
  const currentSessionId                    = useRef<string | null>(_chatCache.currentSessionId);

  const threadRef    = useRef<HTMLDivElement>(null);
  const inputRef     = useRef<HTMLInputElement>(null);
  const didAutoQuery = useRef(false);
  const prevMsgCount = useRef(0);
  // Mirror messages in a ref so handleSubmit can read them without
  // being added to useCallback dependencies (avoids stale closure).
  const messagesRef  = useRef<Message[]>([]);
  useEffect(() => { messagesRef.current = messages; }, [messages]);

  // Write-through to the navigation cache so state survives unmount ↔ remount
  // (e.g. user navigates to /history then comes back).
  useEffect(() => { _chatCache.messages = messages; }, [messages]);
  useEffect(() => { _chatCache.input    = input;    }, [input]);

  // Scroll-to-bottom: only on new message or while loading (not on source toggle)
  useEffect(() => {
    const isNewMessage = messages.length !== prevMsgCount.current;
    prevMsgCount.current = messages.length;
    if (isNewMessage || loading) {
      const container = threadRef.current;
      if (container) container.scrollTop = container.scrollHeight;
    }
  }, [messages.length, loading]);

  // Load session list when sidebar opens (lazy — fetch once per mount)
  useEffect(() => {
    if (!sidebarOpen || !isAuthed || sessionsLoaded) return;
    listChatSessions(0, 50)
      .then(res => { setSessions(res.items); setSessionsLoaded(true); })
      .catch(() => setSessionsLoaded(true));
  }, [sidebarOpen, isAuthed, sessionsLoaded]);

  // Auto-restore the last chat session on F5 / hard-reload.
  //
  // WHY sessionStorage and not module-level _chatCache:
  //   _chatCache is a JS module constant — it survives React unmount/remount
  //   (e.g. navigating away and back) but is wiped on a full browser refresh
  //   because the JS module is re-executed from scratch.  sessionStorage is
  //   persisted by the browser across F5 refreshes in the same tab, making it
  //   the right primitive to bridge this gap.
  //
  //   Only the session ID (a UUID string) is stored — the full message list is
  //   re-fetched from the DB so we always get the up-to-date content.
  useEffect(() => {
    if (!isAuthed || _chatCache.messages.length > 0) return;
    const savedId = sessionStorage.getItem("dc_chat_sid");
    if (!savedId) return;
    handleSelectSession(savedId).catch(() => {
      // Session was deleted — remove the stale id so we don't retry
      sessionStorage.removeItem("dc_chat_sid");
    });
  // handleSelectSession is created with useCallback([]) and is stable.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAuthed]);

  // Open sidebar when a saved session is loaded
  const handleSelectSession = useCallback(async (id: string) => {
    setLoadingSession(true);
    try {
      const detail = await getChatSession(id);
      currentSessionId.current = id;
      _chatCache.currentSessionId = id;
      // Persist across F5 — see auto-restore effect above
      sessionStorage.setItem("dc_chat_sid", id);
      const restored: Message[] = detail.messages.map(m => ({
        id:       crypto.randomUUID(),
        role:     m.role,
        content:  m.content,
        sources:  m.sources as ChatSource[] | undefined,
        provider: m.provider,
      }));
      setMessages(restored);
    } catch {
      // silent — keep current state
    } finally {
      setLoadingSession(false);
    }
  }, []);

  const handleDeleteSession = useCallback(async (id: string) => {
    if (!window.confirm("Delete this chat session?")) return;
    try {
      await deleteChatSession(id);
      setSessions(prev => prev.filter(s => s.id !== id));
      if (currentSessionId.current === id) {
        currentSessionId.current = null;
        _chatCache.currentSessionId = null;
        setMessages([]);
      }
    } catch {
      // silent
    }
  }, []);

  const handleNewChat = useCallback(() => {
    currentSessionId.current = null;
    _chatCache.currentSessionId = null;
    sessionStorage.removeItem("dc_chat_sid");  // user explicitly started fresh
    _chatCache.messages = [];
    _chatCache.input    = "";
    setMessages([]);
    setInput("");
    inputRef.current?.focus();
  }, []);

  // Convert current messages to ChatMessageRecord[] for persistence
  function toRecord(msg: Message): ChatMessageRecord {
    return { role: msg.role, content: msg.content, sources: msg.sources, provider: msg.provider };
  }

  const handleSubmit = useCallback(async (query: string, top5Labels: string[] = []) => {
    if (!query.trim() || loading) return;
    const q = query.trim();

    // Capture prior conversation turns BEFORE adding the new user message.
    // We use messagesRef (not messages state) to avoid stale closures without
    // adding messages to this callback's dependency array.
    const priorHistory = messagesRef.current
      .slice(-20)  // last 10 exchanges = 20 messages — enough for multi-turn context
      .map(m => ({ role: m.role, content: m.content }));

    const userMsg: Message = { id: crypto.randomUUID(), role: "user" as const, content: q };
    const aiId             = crypto.randomUUID();

    // Add user message + empty AI placeholder in ONE state update.
    // WHY together: guarantees the AI placeholder exists before the first
    // onDelta callback fires — prevents a momentary flash of only the user turn.
    setMessages(prev => [
      ...prev,
      userMsg,
      {
        id: aiId, role: "assistant" as const, content: "",
        sources: [], provider: "", userQuery: q, streaming: true,
      },
    ]);
    setInput("");
    setLoading(true);

    // Local accumulator — updated synchronously inside callbacks so that
    // after `await chatQueryStream` resolves we have the definitive final
    // values for session persistence, without reading potentially-stale state.
    const streamed = {
      content:  "",
      sources:  [] as ChatSource[],
      provider: "",
      hasError: false,
    };

    let assistantMsg: Message;
    try {
      await chatQueryStream(
        q, 5, top5Labels, priorHistory,
        {
          onSources: (sources, provider) => {
            streamed.sources  = sources;
            streamed.provider = provider;
            setMessages(prev => prev.map(m =>
              m.id === aiId ? { ...m, sources, provider } : m
            ));
          },
          onDelta: (delta) => {
            streamed.content += delta;
            setMessages(prev => prev.map(m =>
              m.id === aiId ? { ...m, content: m.content + delta } : m
            ));
          },
          onError: (detail) => {
            streamed.hasError = true;
            setMessages(prev => prev.map(m =>
              m.id === aiId
                ? {
                    ...m,
                    content: `Sorry, I couldn't reach the AI right now.\n\n${detail}\n\nPlease check that the DeepCoin backend is running and try again.`,
                    error:    true,
                    streaming: false,
                  }
                : m
            ));
          },
          onDone: () => {
            // Mark streaming complete so the blinking cursor disappears
            setMessages(prev => prev.map(m =>
              m.id === aiId ? { ...m, streaming: false } : m
            ));
          },
        },
        undefined,   // chat page does not have an AbortController; cancel via navigation
      );

      // Safety-net: ensure streaming flag is cleared even if onDone was missed
      setMessages(prev => prev.map(m =>
        m.id === aiId ? { ...m, streaming: false } : m
      ));

      assistantMsg = {
        id: aiId, role: "assistant" as const,
        content:  streamed.content,
        sources:  streamed.sources,
        provider: streamed.provider,
        userQuery: q,
        error:    streamed.hasError,
      };
    } catch (err: unknown) {
      const isAbort = err instanceof Error && err.name === "AbortError";
      if (isAbort) {
        // User cancelled — remove the empty placeholder cleanly
        setMessages(prev => prev.filter(m => m.id !== aiId));
      } else {
        const msg = err instanceof Error ? err.message : "Unknown error";
        setMessages(prev => prev.map(m =>
          m.id === aiId
            ? {
                ...m,
                content: `Sorry, I couldn't reach the knowledge base right now.\n\n${msg}\n\nPlease check that the DeepCoin backend is running and try again.`,
                error:    true,
                streaming: false,
              }
            : m
        ));
      }
      assistantMsg = { id: aiId, role: "assistant" as const, content: "", error: true };
    }

    setLoading(false);
    setTimeout(() => inputRef.current?.focus(), 50);

    // Persist to DB (auth-only; skip on error messages or aborted/empty streams)
    if (isAuthed && !assistantMsg.error && streamed.content) {
      try {
        if (!currentSessionId.current) {
          // First exchange — create a new session
          const created = await createChatSession({
            title:    q.slice(0, 200),
            messages: [toRecord(userMsg), toRecord(assistantMsg)],
          });
          currentSessionId.current = created.id;
          _chatCache.currentSessionId = created.id;
          sessionStorage.setItem("dc_chat_sid", created.id);
          // Prepend to sidebar list (if already loaded)
          setSessions(prev => [
            { id: created.id, title: created.title, created_at: created.created_at, updated_at: created.updated_at, msg_count: 2 },
            ...prev,
          ]);
        } else {
          // Subsequent exchange — append to existing session
          const updated = await appendChatSession(currentSessionId.current, {
            messages: [toRecord(userMsg), toRecord(assistantMsg)],
          });
          // Update msg_count in sidebar list
          setSessions(prev => prev.map(s =>
            s.id === currentSessionId.current
              ? { ...s, updated_at: updated.updated_at, msg_count: updated.messages.length }
              : s
          ));
        }
      } catch {
        // Persistence failure is non-critical — conversation continues
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loading, isAuthed]);

  // Auto-fire query when navigated here with ?q= (e.g. from AnalysisPanel CTA)
  useEffect(() => {
    const q = searchParams.get("q");
    if (q && !didAutoQuery.current) {
      didAutoQuery.current = true;
      // Parse ?top5= param — e.g. "1015,544,532,220,3987" injected from analysis panel CTA
      const rawTop5 = searchParams.get("top5") ?? "";
      const top5Labels = rawTop5
        ? rawTop5.split(",").map((s) => s.trim()).filter(Boolean)
        : [];
      handleSubmit(q, top5Labels);
      // Remove ?q= and ?top5= from the URL so a page reload does NOT re-fire
      // the same query (which would duplicate the first message every refresh).
      window.history.replaceState({}, "", window.location.pathname);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams]);

  const isEmpty = messages.length === 0;

  return (
    <div className="flex h-[calc(100vh-80px)] w-full">

      {/* History sidebar (authenticated only) */}
      <AnimatePresence initial={false}>
        {sidebarOpen && isAuthed && (
          <motion.div
            key="sidebar"
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 240, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ type: "spring", stiffness: 380, damping: 36 }}
            className="shrink-0 overflow-hidden h-full"
          >
            <HistorySidebar
              sessions={sessions}
              activeId={currentSessionId.current}
              onSelect={handleSelectSession}
              onDelete={handleDeleteSession}
              onNewChat={handleNewChat}
              loadingSession={loadingSession}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main chat area */}
      <div className="flex flex-col flex-1 min-w-0 max-w-3xl mx-auto w-full px-4">

        {/* Header */}
        <div className="flex items-center justify-between py-4 shrink-0 border-b" style={{ borderColor: "var(--border)" }}>
          <div className="flex items-center gap-3">
            {/* History toggle — visible badge button (auth-only) */}
            {isAuthed && (
              <button
                onClick={() => setSidebarOpen(v => !v)}
                title={sidebarOpen ? "Close history panel" : "Open saved conversations"}
                className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl text-[11px] font-semibold transition-all"
                style={{
                  background: sidebarOpen ? "rgba(139,92,246,0.22)" : "rgba(139,92,246,0.10)",
                  color:      "#c4b5fd",
                  border:     "1px solid rgba(139,92,246,0.35)",
                  boxShadow:  sidebarOpen ? "0 0 10px rgba(139,92,246,0.18)" : "none",
                }}
              >
                {sidebarOpen ? <PanelLeftClose size={12} /> : <History size={12} />}
                <span className="hidden sm:inline">{sidebarOpen ? "Close" : "History"}</span>
              </button>
            )}
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
              <button onClick={handleNewChat}
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

        {/* RAG scope notice — persistent amber banner */}
        <div
          className="shrink-0 flex items-start gap-2 px-3 py-2 mx-0 my-0 text-[10px] leading-relaxed"
          style={{
            background:   "rgba(251,191,36,0.05)",
            borderBottom: "1px solid rgba(251,191,36,0.14)",
            color:        "var(--text-muted)",
          }}
        >
          <AlertCircle size={11} className="shrink-0 mt-0.5" style={{ color: "#fbbf24" }} />
          <span>
            <span className="font-semibold" style={{ color: "#fde68a" }}>Domain-specific RAG assistant —</span>
            {" "}answers are grounded in the{" "}
            <span className="font-medium" style={{ color: "var(--text-secondary)" }}>Corpus Nummorum knowledge base</span>
            {" "}(9,541 coin types, 47,705 semantic chunks). This assistant handles{" "}
            <span className="font-medium" style={{ color: "var(--text-secondary)" }}>numismatic questions only</span>{" "}
            — it will not respond to greetings, general topics, or anything outside ancient coin scholarship.
          </span>
        </div>

        {/* Thread */}
        <div ref={threadRef} className="flex-1 overflow-y-auto py-6 space-y-5">
          {isEmpty
            ? <EmptyState onSelect={handleSubmit} />
            : (<>
                {messages.map(msg => <MessageBubble key={msg.id} msg={msg} />)}
                {/* Show the three-dot TypingIndicator ONLY before the first
                    token arrives. Once a streaming Message exists in the array,
                    MessageBubble already renders the bot avatar + blinking
                    cursor — showing TypingIndicator on top creates two bot
                    icons simultaneously. */}
                {loading && !messages.some(m => m.streaming) && <TypingIndicator />}
              </>)}
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
            {!isAuthed && (
              <>{" "} · <Link href="/login?callbackUrl=/chat" className="underline hover:text-purple-300 transition-colors">Sign in to save history</Link></>
            )}
          </p>
        </div>
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

