"use client";

/**
 * components/home/TechStack.tsx
 * ================================
 * Real technology & research credits â€” bento-grid redesign.
 *
 * WHAT: Four tech pillar cards + accuracy hero tile + dataset credit.
 *       Every fact is verifiable (requirements.txt, training logs, CN website).
 *
 * WHY bento layout:
 *   The previous small-pill design lost visual hierarchy.
 *   Bento gives each pillar room to breathe; the hero tile converts
 *   the most impressive metric (80% accuracy, 438 classes) into
 *   a standalone showpiece.
 */

import { useRef }            from "react";
import type { ElementType }  from "react";
import { motion, useInView } from "framer-motion";
import { Brain, Bot, ServerIcon, Monitor, ExternalLink, Layers } from "lucide-react";


// â”€â”€ Data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

interface Tech {
  name:    string;
  version: string;
  href?:   string;
  note?:   string;
}

interface Pillar {
  label:   string;
  icon:    ElementType;
  color:   string;
  glow:    string;     // semi-transparent glow for hover border
  tech:    Tech[];
}

const PILLARS: Pillar[] = [
  {
    label: "Deep Learning",
    icon:  Brain,
    color: "#60a5fa",
    glow:  "rgba(96,165,250,0.18)",
    tech: [
      { name: "PyTorch",        version: "2.6.0",     href: "https://pytorch.org",               note: "CUDA 12.4" },
      { name: "EfficientNet",   version: "B3",         href: "https://arxiv.org/abs/1905.11946",  note: "12M params" },
      { name: "Albumentations", version: "1.4+",       href: "https://albumentations.ai",         note: "6 augments" },
      { name: "OpenCV",         version: "4.13",       href: "https://opencv.org",                note: "CLAHE/LAB" },
    ],
  },
  {
    label: "Agentic AI",
    icon:  Bot,
    color: "#a78bfa",
    glow:  "rgba(167,139,250,0.18)",
    tech: [
      { name: "LangGraph",    version: "0.3+",  href: "https://langchain-ai.github.io/langgraph/",  note: "5 agents" },
      { name: "ChromaDB",     version: "0.6+",  href: "https://docs.trychroma.com",                  note: "47,705 vecs" },
      { name: "Gemini Flash", version: "2.5",   note: "GitHub Models" },
      { name: "BM25+VEC+RRF", version: "hybrid", note: "9,541 types" },
    ],
  },
  {
    label: "Backend",
    icon:  ServerIcon,
    color: "#34d399",
    glow:  "rgba(52,211,153,0.18)",
    tech: [
      { name: "FastAPI",      version: "0.115+",  href: "https://fastapi.tiangolo.com",  note: "async" },
      { name: "PostgreSQL",   version: "17",      href: "https://postgresql.org",         note: "RBAC, JWT" },
      { name: "SQLAlchemy",   version: "2.x",     note: "Alembic migrations" },
      { name: "fpdf2",        version: "latest",  note: "PDF reports" },
    ],
  },
  {
    label: "Frontend",
    icon:  Monitor,
    color: "#fb923c",
    glow:  "rgba(251,146,60,0.18)",
    tech: [
      { name: "Next.js",       version: "15",   href: "https://nextjs.org",             note: "App Router" },
      { name: "Framer Motion", version: "12",   href: "https://motion.dev",             note: "animations" },
      { name: "TanStack Q",    version: "v5",   href: "https://tanstack.com/query",     note: "caching" },
      { name: "NextAuth",      version: "v5Î²",  href: "https://authjs.dev",             note: "JWT sessions" },
    ],
  },
];

const HERO_STATS = [
  { value: "80.03%",  label: "TTA accuracy",  sub: "EfficientNet-B3 Ã— 438 classes" },
  { value: "47,705",  label: "RAG vectors",    sub: "5 chunks Ã— 9,541 coin types"  },
  { value: "<20 s",   label: "Full pipeline",  sub: "CNN â†’ agents â†’ PDF"           },
];

// â”€â”€ Component â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

export function TechStack() {
  const ref    = useRef<HTMLElement>(null);
  const inView = useInView(ref, { once: true, margin: "-60px" });

  return (
    <section ref={ref} className="py-24">

      {/* â”€â”€ Section header â”€â”€ */}
      <motion.div
        initial={{ opacity: 0, y: 18 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.5 }}
        className="mb-14"
      >
        <div className="flex items-center gap-2 mb-3">
          <Layers size={14} style={{ color: "var(--brand-gold)" }} />
          <span
            className="text-xs font-bold uppercase tracking-widest"
            style={{ color: "var(--brand-gold)" }}
          >
            Open-source stack
          </span>
        </div>
        <h2 className="text-3xl sm:text-4xl font-black leading-tight" style={{ color: "var(--text-primary)" }}>
          The technology behind it
        </h2>
        <p className="mt-2 text-sm max-w-md" style={{ color: "var(--text-secondary)" }}>
          Every package is verified, versioned, and auditable. No black boxes.
        </p>
      </motion.div>

      {/* â”€â”€ Main bento grid â”€â”€ */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-4">

        {/* â”€â”€ Hero metrics tile (left, spans 1 col) â”€â”€ */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={inView ? { opacity: 1, x: 0 } : {}}
          transition={{ duration: 0.5, delay: 0.05 }}
          className="lg:col-span-1 rounded-2xl p-7 flex flex-col justify-between min-h-[280px] relative overflow-hidden"
          style={{
            background:   "linear-gradient(145deg, rgba(212,168,83,0.12) 0%, rgba(212,168,83,0.04) 100%)",
            border:       "1px solid rgba(212,168,83,0.25)",
          }}
        >
          {/* Subtle radial glow */}
          <div
            className="pointer-events-none absolute inset-0"
            style={{ background: "radial-gradient(circle at 20% 80%, rgba(212,168,83,0.1) 0%, transparent 70%)" }}
          />

          <div className="relative">
            <span className="text-[10px] font-bold uppercase tracking-widest" style={{ color: "rgba(212,168,83,0.7)" }}>
              Verified results
            </span>
            <h3 className="mt-3 text-5xl font-black tabular-nums leading-none" style={{ color: "var(--brand-gold)" }}>
              80%
            </h3>
            <p className="mt-1 text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
              Top-1 accuracy
            </p>
            <p className="mt-0.5 text-xs" style={{ color: "var(--text-muted)" }}>
              TTA Ã—8 Â· 438 ancient coin classes
            </p>
          </div>

          <div className="relative space-y-3 mt-6">
            {HERO_STATS.slice(1).map(({ value, label, sub }) => (
              <div key={label} className="flex items-end justify-between">
                <div>
                  <p className="text-xs font-medium" style={{ color: "var(--text-secondary)" }}>{label}</p>
                  <p className="text-[10px]" style={{ color: "var(--text-muted)" }}>{sub}</p>
                </div>
                <span className="text-base font-black tabular-nums" style={{ color: "var(--brand-gold)" }}>
                  {value}
                </span>
              </div>
            ))}
          </div>
        </motion.div>

        {/* â”€â”€ Four pillar cards (right, 2Ã—2 grid) â”€â”€ */}
        <div className="lg:col-span-2 grid grid-cols-1 sm:grid-cols-2 gap-4">
          {PILLARS.map(({ label, icon: Icon, color, glow, tech }, i) => (
            <motion.div
              key={label}
              initial={{ opacity: 0, y: 20 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.45, delay: 0.1 + i * 0.08 }}
              className="group rounded-2xl p-5 flex flex-col gap-4 transition-all duration-300 cursor-default"
              style={{
                border:          "1px solid var(--border)",
                backgroundColor: "var(--surface-1)",
              }}
              onMouseEnter={e => {
                (e.currentTarget as HTMLElement).style.borderColor = glow;
                (e.currentTarget as HTMLElement).style.boxShadow   = `0 0 24px 0 ${glow}`;
              }}
              onMouseLeave={e => {
                (e.currentTarget as HTMLElement).style.borderColor = "var(--border)";
                (e.currentTarget as HTMLElement).style.boxShadow   = "none";
              }}
            >
              {/* Card header */}
              <div className="flex items-center gap-2.5">
                <div
                  className="w-8 h-8 rounded-xl flex items-center justify-center shrink-0"
                  style={{ backgroundColor: `${color}18` }}
                >
                  <Icon size={16} style={{ color }} />
                </div>
                <span className="text-sm font-bold" style={{ color: "var(--text-primary)" }}>
                  {label}
                </span>
              </div>

              {/* Tech rows */}
              <div className="space-y-2">
                {tech.map(({ name, version, href, note }) => {
                  const Inner = (
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2 min-w-0">
                        <span
                          className="w-1.5 h-1.5 rounded-full shrink-0"
                          style={{ backgroundColor: color }}
                        />
                        <span className="text-xs font-semibold truncate" style={{ color: "var(--text-primary)" }}>
                          {name}
                        </span>
                      </div>
                      <div className="flex items-center gap-1.5 shrink-0">
                        <span
                          className="text-[10px] font-mono px-1.5 py-0.5 rounded-md"
                          style={{ backgroundColor: `${color}16`, color: `${color}cc` }}
                        >
                          {version}
                        </span>
                        {note && (
                          <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>
                            {note}
                          </span>
                        )}
                        {href && <ExternalLink size={9} className="opacity-0 group-hover:opacity-40 transition-opacity" style={{ color }} />}
                      </div>
                    </div>
                  );

                  return href ? (
                    <a
                      key={name}
                      href={href}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="block hover:opacity-80 transition-opacity"
                    >
                      {Inner}
                    </a>
                  ) : (
                    <div key={name}>{Inner}</div>
                  );
                })}
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* â”€â”€ Dataset credit banner â”€â”€ */}
      <motion.a
        href="https://www.corpus-nummorum.eu"
        target="_blank"
        rel="noopener noreferrer"
        initial={{ opacity: 0, y: 14 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.45, delay: 0.45 }}
        className="group flex flex-col sm:flex-row items-start sm:items-center justify-between gap-5 rounded-2xl p-5 transition-all duration-300"
        style={{ border: "1px solid var(--border)", backgroundColor: "var(--surface-1)" }}
        onMouseEnter={e => {
          (e.currentTarget as HTMLElement).style.borderColor = "rgba(212,168,83,0.4)";
        }}
        onMouseLeave={e => {
          (e.currentTarget as HTMLElement).style.borderColor = "var(--border)";
        }}
      >
        {/* Left text */}
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-bold uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>
              Training dataset
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-base font-bold" style={{ color: "var(--text-primary)" }}>
              Corpus Nummorum v1
            </span>
            <ExternalLink
              size={13}
              className="opacity-30 group-hover:opacity-70 transition-opacity"
              style={{ color: "var(--brand-gold)" }}
            />
          </div>
          <p className="text-xs mt-0.5" style={{ color: "var(--text-muted)" }}>
            DFG-funded Â· Freie UniversitÃ¤t Berlin Â· CC BY-SA 4.0
          </p>
        </div>

        {/* Right stats */}
        <div className="flex items-center gap-6">
          {[
            { v: "115,160", label: "raw images" },
            { v: "9,716",   label: "coin types" },
            { v: "438",     label: "trained on" },
          ].map(({ v, label }) => (
            <div key={label} className="text-center">
              <div className="text-xl font-black tabular-nums" style={{ color: "var(--brand-gold)" }}>
                {v}
              </div>
              <div className="text-[9px] uppercase tracking-wide mt-0.5" style={{ color: "var(--text-muted)" }}>
                {label}
              </div>
            </div>
          ))}
        </div>
      </motion.a>
    </section>
  );
}
