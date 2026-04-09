"use client";

/**
 * components/home/PipelineSteps.tsx
 * ====================================
 * Animated 4-step pipeline explainer section.
 *
 * WHAT: Shows the 4 stages (Upload → CNN → Agents → PDF) with icons, tech
 *       stacks, animated connectors, and staggered entrance via useInView.
 *
 * WHY staggered entrance: Users scroll into view here — instant visibility
 *   defeats the purpose. Staggered cards guide attention step-by-step, which
 *   mirrors the actual sequential pipeline.
 *
 * HOW it fits:
 *   Anchored as id="how-it-works" — referenced by the "How it works" hero CTA.
 *   The animated → arrows between the cards hint that data flows left-to-right.
 */

import { useRef }                    from "react";
import { motion, useInView }         from "framer-motion";
import { Camera, Brain, BookOpen, FileText, ArrowRight } from "lucide-react";

const STEPS = [
  {
    step:  "01",
    icon:  Camera,
    title: "Upload a photograph",
    desc:  "Drag-and-drop any coin photo. The engine auto-crops the coin region, applies CLAHE in LAB colour space, and resizes to 299 × 299.",
    tech:  "OpenCV · Auto-crop · CLAHE",
    color: "#3b82f6",
  },
  {
    step:  "02",
    icon:  Brain,
    title: "CNN Classification",
    desc:  "EfficientNet-B3 extracts a 1,536-dim feature vector. 8-pass Test-Time Augmentation averages softmax scores for 80.03 % TTA accuracy.",
    tech:  "EfficientNet-B3 · PyTorch · 80.03 % TTA",
    color: "#8b5cf6",
  },
  {
    step:  "03",
    icon:  BookOpen,
    title: "Agent-based analysis",
    desc:  "A LangGraph state machine routes by confidence. High: Historian  (RAG + LLM). Mid: Validator (OpenCV forensics). Low: Investigator (VLM + BM25).",
    tech:  "LangGraph · ChromaDB · Gemini",
    color: "#d4a853",
  },
  {
    step:  "04",
    icon:  FileText,
    title: "Professional PDF report",
    desc:  "Synthesis agent assembles all outputs into a branded PDF — historical narrative, forensic material check, top-5 candidates, and KB attribution.",
    tech:  "fpdf2 · FastAPI · PostgreSQL",
    color: "#10b981",
  },
];

export function PipelineSteps() {
  const ref     = useRef<HTMLElement>(null);
  const inView  = useInView(ref, { once: true, margin: "-80px" });

  return (
    <section
      id="how-it-works"
      ref={ref}
      className="py-24 scroll-mt-16"
    >
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
            borderColor:     "var(--brand-gold-30)",
            color:           "var(--brand-gold)",
            backgroundColor: "var(--brand-gold-10)",
          }}
        >
          Under the hood
        </span>
        <h2 className="text-3xl sm:text-4xl font-black mb-4" style={{ color: "var(--text-primary)" }}>
          From photograph to report in four steps
        </h2>
        <p className="max-w-xl mx-auto text-sm" style={{ color: "var(--text-secondary)" }}>
          Every inference runs the full production pipeline — the same code that
          powers the live system, not a simplified demo path.
        </p>
      </motion.div>

      {/* Cards + connectors */}
      <div className="relative grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 lg:gap-4">
        {STEPS.map(({ step, icon: Icon, title, desc, tech, color }, i) => (
          <div key={step} className="relative flex items-stretch">
            {/* ── Card ── */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.5, delay: i * 0.13 }}
              className="group flex-1 rounded-2xl border p-6 flex flex-col gap-4 relative overflow-hidden transition-all duration-300 cursor-default"
              style={{
                borderColor:     "var(--border)",
                backgroundColor: "var(--surface-1)",
              }}
              whileHover={{
                boxShadow: `0 0 0 1px ${color}55, 0 8px 32px ${color}18`,
                transition: { duration: 0.15 },
              }}
            >
              {/* step label */}
              <span
                className="text-xs font-black opacity-30 absolute top-5 right-5 tabular-nums"
                style={{ color }}
              >
                {step}
              </span>

              {/* icon */}
              <div
                className="w-11 h-11 rounded-xl flex items-center justify-center animate-glow-pulse"
                style={{ background: `${color}1a`, color }}
              >
                <Icon size={20} />
              </div>

              {/* content */}
              <div>
                <h3 className="font-bold text-base mb-2" style={{ color: "var(--text-primary)" }}>
                  {title}
                </h3>
                <p className="text-xs leading-relaxed" style={{ color: "var(--text-secondary)" }}>
                  {desc}
                </p>
              </div>

              {/* tech badge */}
              <span
                className="mt-auto text-[10px] font-mono font-semibold px-2 py-1 rounded-md self-start"
                style={{ background: `${color}14`, color: `${color}cc` }}
              >
                {tech}
              </span>
            </motion.div>

            {/* ── Connector arrow (between cards, hidden on small screens) ── */}
            {i < STEPS.length - 1 && (
              <motion.div
                className="hidden lg:flex items-center justify-center w-6 flex-shrink-0 self-center"
                animate={{ x: [0, 4, 0] }}
                transition={{ duration: 1.8, repeat: Infinity }}
                aria-hidden
              >
                <ArrowRight size={18} style={{ color: "var(--text-muted)" }} />
              </motion.div>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}
