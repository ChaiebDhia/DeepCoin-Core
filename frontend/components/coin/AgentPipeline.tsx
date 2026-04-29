"use client";
import { useTranslations } from "next-intl";

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

import { X }                              from "lucide-react";



// ── Agent definitions ────────────────────────────────────────────────────────



const getAgents = (tCoin: any, tPipe: any) => [

  {

    emoji:    "�️",

    name:     tCoin("preprocessor_name"),

    subtitle: tPipe('sub_crop'),

    color:    "#f59e0b",

    bgActive: "rgba(245,158,11,0.14)",

    border:   "rgba(245,158,11,0.42)",

    messages: [

      tPipe('msg_decoding'),

      tPipe('msg_detecting_boundary'),

      tPipe('msg_cropping'),

      tPipe('msg_clahe'),

      tPipe('msg_resize'),

      tPipe('msg_norm'),

    ],

  },

  {

    emoji:    "🔬",

    name:     tCoin("cnn_name"),

    subtitle: tPipe('sub_eff'),

    color:    "#3b82f6",

    bgActive: "rgba(59,130,246,0.14)",

    border:   "rgba(59,130,246,0.40)",

    messages: [

      tPipe('msg_feat'),

      tPipe('msg_softmax'),

      tPipe('msg_tta'),

      tPipe('msg_top5'),

      tPipe('msg_routing_conf'),

    ],

  },

  {

    emoji:    "📚",

    name:     tCoin("kb_name"),

    subtitle: tCoin("kb_sub"),

    color:    "#8b5cf6",

    bgActive: "rgba(139,92,246,0.14)",

    border:   "rgba(139,92,246,0.40)",

    messages: [

      tPipe('msg_bm25'),

      tPipe('msg_chroma'),

      tPipe('msg_rrf'),

      tPipe('msg_chunks'),

      tPipe('msg_blocks'),

    ],

  },

  {

    emoji:    "🧠",

    name:     tCoin("llm_name"),

    subtitle: tPipe('sub_spec'),

    color:    "#10b981",

    bgActive: "rgba(16,185,129,0.14)",

    border:   "rgba(16,185,129,0.40)",

    messages: [

      tPipe('msg_route_spec'),

      tPipe('msg_inject15'),

      tPipe('msg_grounding'),

      tPipe('msg_verify'),

      tPipe('msg_compose'),

      tPipe('msg_refs'),

      tPipe('msg_almost'),

    ],

  },

  {

    emoji:    "📄",

    name:     tCoin("synth_name"),

    subtitle: tCoin("synth_sub"),

    color:    "#d4a853",

    bgActive: "rgba(212,168,83,0.14)",

    border:   "rgba(212,168,83,0.40)",

    messages: [

      tPipe('msg_comp_results'),

      tPipe('msg_asm_report'),

      tPipe('msg_gen_pdf'),

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



/**

 * ms after processing starts when each stage becomes active.

 * Preprocessor: 0 – 1 200 ms (OpenCV ops — fast)

 * CNN:          1 200 – 2 600 ms (EfficientNet + 8-pass TTA on GPU)

 * KB:           2 600 – 4 200 ms (BM25 + ChromaDB + RRF)

 * LLM:          4 200 – 18 000 ms (Ollama gemma3:4b or remote provider)

 * Synthesis:    ≥ 18 000 ms (fpdf2 render, triggered when result arrives)

 */

const STAGE_STARTS = [0, 1_200, 2_600, 4_200, 18_000];



type LogEntry = {

  id:       number;

  agentIdx: number;

  message:  string;

};



// ── Component ─────────────────────────────────────────────────────────────────



interface AgentPipelineProps {

  /** Called when the user clicks the X button to cancel the analysis. */

  onCancel?: () => void;

}



export function AgentPipeline({ onCancel }: AgentPipelineProps) {
  const tCoin = useTranslations("Coin");
  const tPipe = useTranslations("AgentPipeline");

  const startRef        = useRef(0);

  const logIdRef        = useRef(0);

  const activeStageRef  = useRef(0);           // stale-closure-safe current stage

  const msgIdxRef       = useRef([0, 0, 0, 0, 0]); // next msg index per agent (5 stages)



  useEffect(() => {

    startRef.current = Date.now();

  }, []);



  const [elapsed, setElapsed]         = useState(0);

  const [activeStage, setActiveStage] = useState(0);

  const [doneStages, setDoneStages]   = useState<Set<number>>(new Set());

  const [log, setLog]                 = useState<LogEntry[]>([]);

  const [xHovered, setXHovered]       = useState(false);



  // Helper: emit the next message for the given agent stage

  const addMessageRef = useRef<(stageIdx: number) => void>(null!);

  useEffect(() => {

    addMessageRef.current = (stageIdx: number) => {

      const agent   = getAgents(tCoin, tPipe)[stageIdx];

      const idx     = msgIdxRef.current[stageIdx];

      // All messages for this stage already shown — emit nothing

      if (idx >= agent.messages.length) return;

      const message = agent.messages[idx];

      msgIdxRef.current[stageIdx] = idx + 1;   // advance past end to silence future ticks

      const id = ++logIdRef.current;

      setLog(l => [...l.slice(-7), { id, agentIdx: stageIdx, message }]);

    };

  });



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

      style={{ backdropFilter: "blur(8px)", background: "rgba(8,18,12,0.78)" }}

    >

    <motion.div

      initial={{ opacity: 0, scale: 0.94, y: 24 }}

      animate={{ opacity: 1, scale: 1, y: 0 }}

      exit={{ opacity: 0, scale: 0.94, y: 16 }}

      transition={{ duration: 0.35, ease: "easeOut" }}

      className="rounded-2xl w-full max-w-2xl"

      style={{

        border:     "1px solid var(--border)",

        background: "linear-gradient(145deg, var(--surface-2) 0%, var(--surface-0) 100%)",

        overflow:   "hidden",

      }}

    >

      {/* ── Header ───────────────────────────────────────────────────── */}

      <div

        className="flex items-center justify-between px-5 py-3 border-b"

        style={{ borderColor: "var(--border)" }}

      >

        <div className="flex items-center gap-2.5">

          {/* Coin-flip mascot — scaleX 1→0.1→1 simulates a coin flipping */}

          <motion.span

            animate={{ scaleX: [1, 0.08, 1, 0.08, 1] }}

            transition={{ duration: 1.8, repeat: Infinity, repeatDelay: 3.5, ease: "easeInOut" }}

            className="text-base select-none leading-none"

            style={{ display: "inline-block" }}

          >

            🪙

          </motion.span>

          <span

            className="text-[10px] font-bold tracking-widest uppercase"

            style={{ color: "var(--text-muted)" }}

          >

            {tCoin("missionControl")}

          </span>

        </div>

        <div className="flex items-center gap-3">

          <span

            className="text-xs font-mono tabular-nums"

            style={{ color: "var(--brand-gold)" }}

          >

            ⏱ {elapsedSec}s

          </span>

          {onCancel && (

            <button

              onClick={onCancel}

              title="Cancel analysis"

              onMouseEnter={() => setXHovered(true)}

              onMouseLeave={() => setXHovered(false)}

              className="flex items-center justify-center w-6 h-6 rounded-md transition-colors"

              style={{

                color:       xHovered ? "#f87171" : "var(--text-muted)",

                background:  xHovered ? "rgba(239,68,68,0.15)" : "rgba(8,18,12,0.12)",

                border:      `1px solid ${xHovered ? "rgba(239,68,68,0.35)" : "var(--border)"}`,

                transition:  "color 0.15s, background 0.15s, border-color 0.15s",

              }}

            >

              <X size={13} strokeWidth={2.5} />

            </button>

          )}

        </div>

      </div>



      {/* ── Mascot row: active agent speaks the latest log message ─────

           WHY: the chat log below is useful but small. This persistent

           "speech bubble" always shows what the active agent is doing in a

           prominent, human-readable line. The bouncing dots signal in-progress

           work without a spinner — keeps the UI alive and engaging.          */}

      <AnimatePresence mode="wait">

        {log.length > 0 && (

          <motion.div

            key={log[log.length - 1].id}

            initial={{ opacity: 0, y: -4 }}

            animate={{ opacity: 1, y: 0 }}

            exit={{ opacity: 0 }}

            transition={{ duration: 0.28 }}

            className="flex items-center gap-3 px-5 py-2.5"

            style={{

              borderBottom: "1px solid var(--border)",

              background:   "rgba(10,24,14,0.08)",

            }}

          >

            {/* Active agent emoji — spring-pops on stage change */}

            <motion.span

              key={activeStage}

              initial={{ scale: 0.6 }}

              animate={{ scale: 1 }}

              transition={{ type: "spring", stiffness: 420, damping: 18 }}

              className="text-2xl shrink-0 select-none"

              style={{ filter: `drop-shadow(0 0 9px ${getAgents(tCoin, tPipe)[activeStage].color}55)` }}

            >

              {getAgents(tCoin, tPipe)[activeStage].emoji}

            </motion.span>



            {/* Speech bubble */}

            <div

              className="rounded-lg px-3 py-1.5 text-xs flex-1 min-w-0"

              style={{

                background: getAgents(tCoin, tPipe)[activeStage].bgActive,

                border:     `1px solid ${getAgents(tCoin, tPipe)[activeStage].border}`,

              }}

            >

              <span className="font-bold mr-2" style={{ color: getAgents(tCoin, tPipe)[activeStage].color }}>

                {getAgents(tCoin, tPipe)[activeStage].name}:

              </span>

              <span style={{ color: "var(--text-secondary)" }}>

                {log[log.length - 1].message}

              </span>

            </div>



            {/* Bouncing typing dots */}

            <div className="flex items-center gap-1 shrink-0">

              {[0, 1, 2].map(i => (

                <motion.span

                  key={i}

                  className="inline-block w-1.5 h-1.5 rounded-full"

                  style={{ background: getAgents(tCoin, tPipe)[activeStage].color }}

                  animate={{ opacity: [0.25, 1, 0.25], y: [0, -4, 0] }}

                  transition={{

                    duration: 0.75,

                    repeat:   Infinity,

                    delay:    i * 0.17,

                    ease:     "easeInOut",

                  }}

                />

              ))}

            </div>

          </motion.div>

        )}

      </AnimatePresence>



      {/* ── Agent stations + connectors ──────────────────────────────── */}

      {/* NOTE: overflow-x-auto was removed intentionally. Setting overflow-x:auto

           forces overflow-y:hidden (CSS spec), which clips the vertical glow

           box-shadows.  Sizes are made responsive instead so all 5 stations fit

           at ~343px (375px mobile minus scrollbar and outer padding).  */}

<div className="flex items-center sm:justify-center overflow-x-auto no-scrollbar px-2 sm:px-6 py-6 sm:py-8 gap-0 w-full" style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}>

        {getAgents(tCoin, tPipe).map((agent, i) => {

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

className="flex flex-col items-center text-center gap-1 sm:gap-2 rounded-xl px-1 sm:px-1.5 py-2 sm:py-3 min-w-[44px] sm:min-w-[68px] max-w-[64px] sm:max-w-[80px]"

                style={{

                  background:  isActive ? agent.bgActive

                             : isDone   ? "rgba(10,24,14,0.10)"

                             :            "transparent",

                  border:      `1px solid ${isActive ? agent.border : isDone ? "var(--border)" : "rgba(10,24,14,0.16)"}`,

                  boxShadow:   isActive ? `0 0 22px 3px ${agent.color}26` : "none",

                  transition:  "all 0.5s cubic-bezier(0.4,0,0.2,1)",

                }}

              >

                {/* Emoji avatar */}

                <div

                  className="w-8 h-8 sm:w-12 sm:h-12 rounded-full flex items-center justify-center text-base sm:text-xl"

                  style={{

                    background:  isActive ? agent.bgActive : "rgba(10,24,14,0.12)",

                    border:      `1.5px solid ${isActive ? agent.border : "var(--border)"}`,

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

                    className="text-[9px] sm:text-[11px] font-bold leading-tight"

                    style={{

                      color:      isActive ? agent.color : isDone ? "var(--text-secondary)" : "var(--text-muted)",

                      transition: "color 0.4s ease",

                    }}

                  >

                    {agent.name}

                  </p>

                  <p

                    className="hidden sm:block text-[9px] leading-snug mt-0.5"

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

                                   :             "rgba(10,24,14,0.30)",

                    transition: "background-color 0.4s ease",

                  }}

                />

              </motion.div>



              {/* ── Connector beam ─────────────────────────────────── */}

              {i < getAgents(tCoin, tPipe).length - 1 && (

                <div

                  className="relative mx-0.5 sm:mx-1 h-0.5 w-3 sm:w-8 rounded-full overflow-hidden shrink-0"

                  style={{ backgroundColor: "rgba(10,24,14,0.22)" }}

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

        style={{ borderColor: "var(--border)" }}

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

              const agent     = getAgents(tCoin, tPipe)[entry.agentIdx];

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

className="text-[10px] font-bold mt-0.5 shrink-0 w-[80px] sm:w-[100px] break-words"

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

