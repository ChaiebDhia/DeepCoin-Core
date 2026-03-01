"use client";

/**
 * components/coin/AgentPipeline.tsx
 * ===================================
 * Live "Mission Control" visualiser that replaces the boring spinner
 * during the processing phase.
 *
 * WHAT it shows:
 *   4 agent stations (CNN → KB → LLM → PDF) connected by animated data
 *   "beams". As time elapses, each station activates in sequence, glows,
 *   and emits log messages to a live chat log below the cards.
 *
 * WHY time-based simulation (not real server events):
 *   FastAPI does not push SSE/WebSocket events mid-pipeline. The timing
 *   model is built from measured real-world durations:
 *     CNN   0 – 1.2 s   (GPU inference + TTA 5 passes)
 *     KB    1.2 – 2.8 s  (ChromaDB + BM25 hybrid lookup)
 *     LLM   2.8 – 17 s   (Ollama gemma3:4b or remote provider)
 *     PDF   ≥17 s / done  (fpdf2 render, triggered on result)
 *
 * WHY refs for activeStage and msgIdx:
 *   setInterval callbacks close over the value at creation time (stale
 *   closure). Using refs gives the interval access to the CURRENT value
 *   without re-registering the interval on every render.
 */

import { useState, useEffect, useRef }   from "react";
import { motion, AnimatePresence }        from "framer-motion";

// ── Agent definitions ────────────────────────────────────────────────────────

const AGENTS = [
  {
    emoji:    "🔬",
    name:     "CNN",
    subtitle: "EfficientNet-B3",
    color:    "#3b82f6",
    bgActive: "rgba(59,130,246,0.14)",
    border:   "rgba(59,130,246,0.40)",
    messages: [
      "Loading EfficientNet-B3 weights…",
      "Extracting 1,536 visual features…",
      "Running softmax over 438 classes…",
      "TTA — averaging 5 forward passes…",
      "Routing by confidence threshold…",
    ],
  },
  {
    emoji:    "📚",
    name:     "Knowledge Base",
    subtitle: "47,705 vectors",
    color:    "#8b5cf6",
    bgActive: "rgba(139,92,246,0.14)",
    border:   "rgba(139,92,246,0.40)",
    messages: [
      "BM25 keyword index search…",
      "ChromaDB vector similarity…",
      "Reciprocal Rank Fusion merge…",
      "Retrieving 5 semantic chunks…",
      "Assembling [CONTEXT] blocks…",
    ],
  },
  {
    emoji:    "🧠",
    name:     "LLM Agent",
    subtitle: "Historian · Validator · Investigator",
    color:    "#10b981",
    bgActive: "rgba(16,185,129,0.14)",
    border:   "rgba(16,185,129,0.40)",
    messages: [
      "Routing to specialist agent…",
      "Injecting [CONTEXT 1–5] blocks…",
      "Grounding historical narrative…",
      "Verifying forensic evidence…",
      "Composing professional analysis…",
      "Checking numismatic references…",
      "Almost there — finalising…",
    ],
  },
  {
    emoji:    "📄",
    name:     "Synthesis",
    subtitle: "fpdf2 report",
    color:    "#d4a853",
    bgActive: "rgba(212,168,83,0.14)",
    border:   "rgba(212,168,83,0.40)",
    messages: [
      "Compiling analysis results…",
      "Assembling full report…",
      "Generating downloadable PDF…",
    ],
  },
] as const satisfies Array<{
  emoji:    string;
  name:     string;
  subtitle: string;
  color:    string;
  bgActive: string;
  border:   string;
  messages: readonly string[];
}>;

/** ms after processing starts when each stage becomes active */
const STAGE_STARTS = [0, 1200, 2800, 17_000];

type LogEntry = {
  id:       number;
  agentIdx: number;
  message:  string;
};

// ── Component ─────────────────────────────────────────────────────────────────

export function AgentPipeline() {
  const startRef        = useRef(Date.now());
  const logIdRef        = useRef(0);
  const activeStageRef  = useRef(0);        // stale-closure-safe current stage
  const msgIdxRef       = useRef([0, 0, 0, 0]); // next msg index per agent

  const [elapsed, setElapsed]         = useState(0);
  const [activeStage, setActiveStage] = useState(0);
  const [doneStages, setDoneStages]   = useState<Set<number>>(new Set());
  const [log, setLog]                 = useState<LogEntry[]>([]);

  // Helper: emit the next message for the given agent stage
  const addMessageRef = useRef<(stageIdx: number) => void>(null!);
  addMessageRef.current = (stageIdx: number) => {
    const agent   = AGENTS[stageIdx];
    const idx     = msgIdxRef.current[stageIdx];
    // All messages for this stage already shown — emit nothing
    if (idx >= agent.messages.length) return;
    const message = agent.messages[idx];
    msgIdxRef.current[stageIdx] = idx + 1;   // advance past end to silence future ticks
    const id = ++logIdRef.current;
    setLog(l => [...l.slice(-7), { id, agentIdx: stageIdx, message }]);
  };

  // ── Master tick: update elapsed + advance stage ──────────────────────────
  useEffect(() => {
    const tick = setInterval(() => {
      const ms = Date.now() - startRef.current;
      setElapsed(ms);

      // Determine which stage should be active
      let newStage = 0;
      for (let i = STAGE_STARTS.length - 1; i >= 0; i--) {
        if (ms >= STAGE_STARTS[i]) { newStage = i; break; }
      }

      if (newStage !== activeStageRef.current) {
        const prev = activeStageRef.current;
        activeStageRef.current = newStage;
        setActiveStage(newStage);
        setDoneStages(d => {
          const next = new Set(d);
          for (let i = 0; i <= prev; i++) next.add(i);
          return next;
        });
        // Emit an immediate message for the newly activated stage
        addMessageRef.current(newStage);
      }
    }, 100);

    return () => clearInterval(tick);
  }, []);

  // ── Message emitter: emit a new log line every 2.5 s ───────────────────
  useEffect(() => {
    // First message immediately
    addMessageRef.current(0);

    const emit = setInterval(() => {
      addMessageRef.current(activeStageRef.current);
    }, 2500);

    return () => clearInterval(emit);
  }, []);

  const elapsedSec = (elapsed / 1000).toFixed(1);

  return (
    // ── Fixed fullscreen overlay with backdrop blur ──────────────────────────
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.25 }}
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ backdropFilter: "blur(8px)", background: "rgba(4,10,20,0.75)" }}
    >
    <motion.div
      initial={{ opacity: 0, scale: 0.94, y: 24 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.94, y: 16 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
      className="rounded-2xl overflow-hidden w-full max-w-xl"
      style={{
        border: "1px solid rgba(255,255,255,0.07)",
        background: "linear-gradient(145deg, #0d1a2e 0%, #080e1a 100%)",
      }}
    >
      {/* ── Header ───────────────────────────────────────────────────── */}
      <div
        className="flex items-center justify-between px-5 py-3 border-b"
        style={{ borderColor: "rgba(255,255,255,0.07)" }}
      >
        <div className="flex items-center gap-2.5">
          <span className="h-2 w-2 rounded-full bg-green-400 animate-pulse inline-block" />
          <span
            className="text-[10px] font-bold tracking-widest uppercase"
            style={{ color: "var(--text-muted)" }}
          >
            DeepCoin Mission Control
          </span>
        </div>
        <span
          className="text-xs font-mono tabular-nums"
          style={{ color: "var(--brand-gold)" }}
        >
          ⏱ {elapsedSec}s
        </span>
      </div>

      {/* ── Agent stations + connectors ──────────────────────────────── */}
      <div className="flex items-center justify-center px-4 py-7 gap-0 overflow-x-auto">
        {AGENTS.map((agent, i) => {
          const isActive  = activeStage === i;
          const isDone    = doneStages.has(i);
          const isWaiting = !isActive && !isDone;

          return (
            <div key={agent.name} className="flex items-center shrink-0">
              {/* Agent card */}
              <motion.div
                initial={{ opacity: 0, scale: 0.8, y: 10 }}
                animate={{
                  opacity: 1,
                  scale:   isActive ? 1.08 : 1,
                  y:       0,
                }}
                transition={{ delay: i * 0.08, duration: 0.35, ease: "easeOut" }}
                className="flex flex-col items-center gap-2 rounded-xl px-3 py-3.5 min-w-[82px]"
                style={{
                  background:  isActive ? agent.bgActive
                             : isDone   ? "rgba(255,255,255,0.03)"
                             :            "transparent",
                  border:      `1px solid ${isActive ? agent.border : isDone ? "rgba(255,255,255,0.08)" : "rgba(255,255,255,0.03)"}`,
                  boxShadow:   isActive ? `0 0 22px 3px ${agent.color}26` : "none",
                  transition:  "all 0.5s cubic-bezier(0.4,0,0.2,1)",
                }}
              >
                {/* Emoji avatar */}
                <div
                  className="w-12 h-12 rounded-full flex items-center justify-center text-xl"
                  style={{
                    background:  isActive ? agent.bgActive : "rgba(255,255,255,0.04)",
                    border:      `1.5px solid ${isActive ? agent.border : "rgba(255,255,255,0.06)"}`,
                    filter:      isWaiting ? "grayscale(0.7) opacity(0.45)" : "none",
                    transition:  "all 0.5s ease",
                    boxShadow:   isActive ? `0 0 14px 2px ${agent.color}30` : "none",
                  }}
                >
                  {isDone ? "✅" : agent.emoji}
                </div>

                {/* Name + subtitle */}
                <div className="text-center">
                  <p
                    className="text-[11px] font-bold leading-tight"
                    style={{
                      color:      isActive ? agent.color : isDone ? "var(--text-secondary)" : "var(--text-muted)",
                      transition: "color 0.4s ease",
                    }}
                  >
                    {agent.name}
                  </p>
                  <p
                    className="text-[9px] leading-snug mt-0.5"
                    style={{ color: "var(--text-muted)", maxWidth: "76px" }}
                  >
                    {agent.subtitle}
                  </p>
                </div>

                {/* Status indicator dot */}
                <span
                  className={`inline-block h-1.5 w-1.5 rounded-full ${isActive ? "animate-pulse" : ""}`}
                  style={{
                    backgroundColor: isDone    ? "#22c55e"
                                   : isActive  ? agent.color
                                   :             "rgba(255,255,255,0.12)",
                    transition: "background-color 0.4s ease",
                  }}
                />
              </motion.div>

              {/* ── Connector beam ─────────────────────────────────── */}
              {i < AGENTS.length - 1 && (
                <div
                  className="relative mx-1.5 h-0.5 w-10 rounded-full overflow-hidden shrink-0"
                  style={{ backgroundColor: "rgba(255,255,255,0.06)" }}
                >
                  {(isActive || doneStages.has(i)) && (
                    <span
                      className={isDone ? "" : "animate-particle"}
                      style={{
                        position:        "absolute",
                        top:             0,
                        left:            0,
                        height:          "100%",
                        width:           "35%",
                        borderRadius:    "9999px",
                        background:      `linear-gradient(90deg, transparent, ${agent.color}, transparent)`,
                        opacity:         isDone ? 0.45 : 1,
                        transform:       isDone ? "translateX(100%)" : undefined,
                      }}
                    />
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* ── Live chat log ─────────────────────────────────────────────── */}
      <div
        className="border-t px-5 py-4"
        style={{ borderColor: "rgba(255,255,255,0.06)" }}
      >
        <p
          className="text-[9px] uppercase tracking-widest mb-3 font-semibold"
          style={{ color: "var(--text-muted)" }}
        >
          Agent Log
        </p>

        <div className="flex flex-col gap-2 min-h-[80px]">
          <AnimatePresence mode="popLayout" initial={false}>
            {log.slice(-5).map((entry, rank) => {
              const agent     = AGENTS[entry.agentIdx];
              const isNewest  = rank === log.slice(-5).length - 1;
              const entryDone = doneStages.has(entry.agentIdx);

              return (
                <motion.div
                  key={entry.id}
                  initial={{ opacity: 0, x: -16, height: 0 }}
                  animate={{ opacity: 1, x: 0, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.22, ease: "easeOut" }}
                  className="flex items-start gap-2.5 overflow-hidden"
                >
                  {/* Agent emoji */}
                  <span className="text-sm mt-0.5 shrink-0">
                    {entryDone ? "✅" : agent.emoji}
                  </span>

                  {/* Agent label */}
                  <span
                    className="text-[10px] font-bold mt-0.5 shrink-0 w-[72px]"
                    style={{ color: agent.color }}
                  >
                    [{agent.name}]
                  </span>

                  {/* Message */}
                  <span
                    className="text-xs leading-relaxed"
                    style={{
                      color: isNewest && !entryDone
                        ? "var(--text-secondary)"
                        : "var(--text-muted)",
                    }}
                  >
                    {entry.message}
                    {/* Blinking cursor on newest active message */}
                    {isNewest && !entryDone && (
                      <span
                        className="animate-cursor inline-block w-[6px] h-[12px] ml-1 align-middle rounded-[1px]"
                        style={{ backgroundColor: agent.color, opacity: 0.8 }}
                      />
                    )}
                  </span>
                </motion.div>
              );
            })}
          </AnimatePresence>
        </div>
      </div>
    </motion.div>
    </motion.div>
  );
}
