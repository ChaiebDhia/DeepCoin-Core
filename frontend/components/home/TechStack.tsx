"use client";

/**
 * components/home/TechStack.tsx
 * ================================
 * Real technology & research credits section.
 *
 * WHAT: Replaces the placeholder testimonials with the actual technology stack
 *       and research data sources powering DeepCoin.
 *
 * WHY replace testimonials:
 *   The previous testimonials were clearly labelled as "preview quotes" but
 *   this still felt dishonest on a professional PFE project. This section
 *   shows REAL, verifiable information:
 *   - Exact package versions
 *   - Verified accuracy metrics
 *   - Actual dataset source (Corpus Nummorum, DFG-funded)
 *   - Real academic context
 *   All facts here can be cross-referenced with requirements.txt, the
 *   training logs, and the Corpus Nummorum website.
 *
 * HOW it fits:
 *   Appears between ForWhoCards and AnalyseSection in page.tsx.
 *   Four pillars: Deep Learning · Agentic AI · Backend · Frontend.
 */

import { useRef }            from "react";
import { motion, useInView } from "framer-motion";
import { ExternalLink }      from "lucide-react";

interface Pill {
  name:    string;
  version: string;
  href?:   string;
}

interface Pillar {
  label:  string;
  color:  string;
  pills:  Pill[];
}

const PILLARS: Pillar[] = [
  {
    label: "Deep Learning",
    color: "#3b82f6",
    pills: [
      { name: "PyTorch",          version: "2.6.0+cu124", href: "https://pytorch.org" },
      { name: "EfficientNet-B3",  version: "torchvision", href: "https://arxiv.org/abs/1905.11946" },
      { name: "OpenCV",           version: "4.13.0",      href: "https://opencv.org" },
      { name: "Albumentations",   version: "1.4+",        href: "https://albumentations.ai" },
      { name: "CLAHE / LAB",      version: "preprocessing" },
      { name: "TTA ×8",           version: "+0.78% acc" },
    ],
  },
  {
    label: "Agentic AI",
    color: "#8b5cf6",
    pills: [
      { name: "LangGraph",                version: "0.3+",       href: "https://langchain-ai.github.io/langgraph/" },
      { name: "ChromaDB",                 version: "0.6+",       href: "https://docs.trychroma.com" },
      { name: "Gemini 2.5 Flash",         version: "via GitHub Models" },
      { name: "sentence-transformers",    version: "3.3+",       href: "https://sbert.net" },
      { name: "BM25 + Vector + RRF",      version: "hybrid search" },
      { name: "47,705 RAG Chunks",        version: "9,541 types" },
    ],
  },
  {
    label: "Backend",
    color: "#10b981",
    pills: [
      { name: "FastAPI",          version: "0.115+",   href: "https://fastapi.tiangolo.com" },
      { name: "PostgreSQL",       version: "17",       href: "https://postgresql.org" },
      { name: "SQLAlchemy",       version: "2.x async" },
      { name: "Alembic",          version: "migrations" },
      { name: "slowapi",          version: "10 req/min" },
      { name: "fpdf2",            version: "PDF render" },
    ],
  },
  {
    label: "Frontend",
    color: "#f97316",
    pills: [
      { name: "Next.js",          version: "15 App Router", href: "https://nextjs.org" },
      { name: "Framer Motion",    version: "12",            href: "https://motion.dev" },
      { name: "TanStack Query",   version: "5",             href: "https://tanstack.com/query" },
      { name: "NextAuth.js",      version: "5.0 beta",      href: "https://authjs.dev" },
      { name: "Tailwind CSS",     version: "v4" },
      { name: "Zustand",          version: "5" },
    ],
  },
];

const DATASET = {
  name:    "Corpus Nummorum v1",
  url:     "https://www.corpus-nummorum.eu",
  funder:  "DFG (Deutsche Forschungsgemeinschaft)",
  types:   "9,716 coin types",
  images:  "115,160 images",
  license: "CC BY-SA 4.0",
};

export function TechStack() {
  const ref    = useRef<HTMLElement>(null);
  const inView = useInView(ref, { once: true, margin: "-80px" });

  return (
    <section ref={ref} className="py-24">

      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.55 }}
        className="text-center mb-16"
      >
        <span
          className="inline-block text-xs font-semibold uppercase tracking-widest px-3 py-1 rounded-full border mb-4"
          style={{
            borderColor:     "rgba(212,168,83,0.35)",
            color:           "var(--brand-gold)",
            backgroundColor: "rgba(212,168,83,0.07)",
          }}
        >
          Open stack
        </span>
        <h2 className="text-3xl sm:text-4xl font-black mb-4" style={{ color: "var(--text-primary)" }}>
          Built on verified open-source
        </h2>
        <p className="max-w-xl mx-auto text-sm" style={{ color: "var(--text-secondary)" }}>
          Every package is production-grade with a known version. No black boxes.
        </p>
      </motion.div>

      {/* Four pillars */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
        {PILLARS.map(({ label, color, pills }, i) => (
          <motion.div
            key={label}
            initial={{ opacity: 0, y: 24 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.5, delay: i * 0.1 }}
            className="rounded-2xl border p-6 flex flex-col gap-4"
            style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}
          >
            {/* Pillar label */}
            <span
              className="text-xs font-black uppercase tracking-widest"
              style={{ color }}
            >
              {label}
            </span>

            {/* Pill grid */}
            <div className="flex flex-wrap gap-2">
              {pills.map(({ name, version, href }) =>
                href ? (
                  <a
                    key={name}
                    href={href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="group inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-[10px] font-medium transition-all hover:brightness-120"
                    style={{ background: `${color}14`, color: `${color}cc` }}
                  >
                    {name}
                    <span className="opacity-40 text-[9px]">{version}</span>
                    <ExternalLink size={8} className="opacity-0 group-hover:opacity-60 transition-opacity" />
                  </a>
                ) : (
                  <span
                    key={name}
                    className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-[10px] font-medium"
                    style={{ background: `${color}14`, color: `${color}cc` }}
                  >
                    {name}
                    <span className="opacity-40 text-[9px]">{version}</span>
                  </span>
                )
              )}
            </div>
          </motion.div>
        ))}
      </div>

      {/* Dataset credit */}
      <motion.a
        href={DATASET.url}
        target="_blank"
        rel="noopener noreferrer"
        initial={{ opacity: 0, y: 14 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.5, delay: 0.45 }}
        className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 rounded-2xl border p-6 hover:border-[rgba(212,168,83,0.5)] transition-colors group"
        style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}
      >
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="font-bold text-sm" style={{ color: "var(--text-primary)" }}>
              {DATASET.name}
            </span>
            <ExternalLink size={13} className="opacity-40 group-hover:opacity-80 transition-opacity" style={{ color: "var(--brand-gold)" }} />
          </div>
          <p className="text-xs" style={{ color: "var(--text-secondary)" }}>
            Funded by {DATASET.funder} · {DATASET.images} · {DATASET.types} · Licensed {DATASET.license}
          </p>
        </div>
        <div className="flex gap-3 flex-shrink-0">
          {[
            { v: DATASET.images,  label: "images" },
            { v: DATASET.types,   label: "types" },
          ].map(({ v, label }) => (
            <div key={label} className="text-center">
              <div className="text-base font-black tabular-nums" style={{ color: "var(--brand-gold)" }}>{v}</div>
              <div className="text-[9px] uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>{label}</div>
            </div>
          ))}
        </div>
      </motion.a>
    </section>
  );
}
