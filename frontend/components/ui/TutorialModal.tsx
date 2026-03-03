"use client";

/**
 * components/ui/TutorialModal.tsx
 * =================================
 * WHAT: Enterprise guided tour. A floating "Guide" pill (bottom-right) opens a
 *       step-by-step walkthrough of all DeepCoin features for first-time visitors.
 *
 * WHY: Museum curators, students and researchers may not understand the AI pipeline,
 *      confidence tiers, or auxiliary features (Explore, Chat, PDF, Feedback).
 *      A polished onboarding tour prevents drop-off and builds trust.
 *
 * HOW:
 *   - Auto-opens once for first-time visitors (localStorage key "deepcoin_guide_seen")
 *   - "Don't show again" persists the flag so it never auto-opens after dismissal
 *   - Keyboard navigation: ← previous, → / Enter next, Esc close
 *   - Two-column layout: colored left panel (icon + step map) + right content panel
 *   - Framer Motion slide transitions on step change
 *   - "Skip tour" available from every step
 *   - Floating trigger button shows pulse beacon ONLY on first visit
 *
 * INTEGRATION: Rendered once in app/layout.tsx — available on every page.
 */

import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence }           from "framer-motion";
import Link                                  from "next/link";
import {
  X, ChevronLeft, ChevronRight, Check,
  Upload, Cpu, BookOpen, MessageSquare, Download, ThumbsDown,
  Keyboard,
} from "lucide-react";

/* ─── localStorage key ───────────────────────────────────────────────────── */
const STORAGE_KEY = "deepcoin_guide_seen_v2";

/* ─── step definitions ───────────────────────────────────────────────────── */

interface Step {
  icon:  React.ReactNode;
  color: string;
  tag:   string;
  title: string;
  body:  string;
  bullets?: string[];
  cta?:  { label: string; href: string } | null;
}

const STEPS: Step[] = [
  {
    icon:  <Upload   size={32} />,
    color: "#3b82f6",
    tag:   "Step 1 — Upload",
    title: "Upload a coin photo",
    body:  "Navigate to the Analyse page and drag-and-drop any photo of an ancient coin onto the upload zone, or click to open the file picker.",
    bullets: [
      "Accepted formats: JPEG · PNG · WebP",
      "Large images are auto-downsized to 1024 px before upload",
      "Enable TTA (Test-Time Augmentation) for +0.78 % accuracy — takes ~5 s extra",
      "Auto-crop detects the coin boundary and removes background noise",
    ],
    cta:   { label: "Go to Analyse →", href: "/analyse" },
  },
  {
    icon:  <Cpu      size={32} />,
    color: "#8b5cf6",
    tag:   "Step 2 — AI Classification",
    title: "Understand the CNN result",
    body:  "DeepCoin's EfficientNet-B3 model classifies your coin against 438 trained Corpus Nummorum types and routes it through a specialist agent.",
    bullets: [
      "🟢 Identified — confidence ≥ 70 % — Historian agent writes a full narrative",
      "🔵 Consistent Match — 7 of 8 TTA passes agree — high visual consensus",
      "🟣 Deep Search — low confidence — Investigator searches all 9,541 KB types",
      "Top-5 candidates link directly to corpus-nummorum.eu records",
    ],
    cta:   null,
  },
  {
    icon:  <BookOpen size={32} />,
    color: "#10b981",
    tag:   "Step 3 — Explore",
    title: "Browse the public gallery",
    body:  "The Explore page shows all past analyses from every user. Filter by AI route to find specific confidence tiers or coin families.",
    bullets: [
      "Filter by Historian / Validator / Investigator route",
      "Click any card to view the full analysis with PDF download",
      "CN type links open the official Corpus Nummorum record",
      "No sign-in required — the gallery is fully public",
    ],
    cta:   { label: "Open Explore →", href: "/explore" },
  },
  {
    icon:  <MessageSquare size={32} />,
    color: "#a78bfa",
    tag:   "Step 4 — AI Chat",
    title: "Ask the numismatic AI",
    body:  "DeepCoin AI answers ancient coin questions grounded in 47,705 knowledge-base chunks covering 9,541 Corpus Nummorum types — not the open internet.",
    bullets: [
      "Ask about dynasties, mint cities, iconography, denominations, or dating",
      "Every answer cites CN record IDs — all facts are verifiable",
      "Paste a CN type ID from an analysis result for instant context",
      "Domain-specific only — it will not respond to off-topic questions",
    ],
    cta:   { label: "Open AI Chat →", href: "/chat" },
  },
  {
    icon:  <Download size={32} />,
    color: "#f59e0b",
    tag:   "Step 5 — PDF Report",
    title: "Download your PDF report",
    body:  "Every analysis produces a professional PDF with a full historical narrative, forensic material check, visual attributes table, and a top-5 candidate matrix.",
    bullets: [
      "PDF links remain active for 30 days after analysis",
      "Includes Corpus Nummorum references and mint/date attribution",
      "Access past PDFs from the History page at any time",
      "Reports are stored privately — only visible to you",
    ],
    cta:   { label: "View History →", href: "/history" },
  },
  {
    icon:  <ThumbsDown size={32} />,
    color: "#ef4444",
    tag:   "Step 6 — Improve the Model",
    title: "Mark wrong results",
    body:  "If the CNN misidentified a coin, use the 'Mark as wrong' button on the analysis panel to submit the correct CN type ID.",
    bullets: [
      "Your correction is logged in the active-learning feedback store",
      "Admins review corrections in the admin panel",
      "Repeated corrections on the same type trigger a retraining signal",
      "Every expert correction makes the model better for everyone",
    ],
    cta:   null,
  },
];

/* ─── component ──────────────────────────────────────────────────────────── */

export default function TutorialModal() {
  const [open, setOpen]               = useState(false);
  const [step, setStep]               = useState(0);
  const [dontShowAgain, setDontShow]  = useState(false);
  const [hasBeenSeen, setHasBeenSeen] = useState(true); // default true → no pulse until we check

  /* Auto-open for first-time visitors */
  useEffect(() => {
    const seen = typeof window !== "undefined" && !!localStorage.getItem(STORAGE_KEY);
    setHasBeenSeen(seen);
    if (!seen) {
      const t = setTimeout(() => setOpen(true), 1400);
      return () => clearTimeout(t);
    }
  }, []);

  /* ── close helpers ─────────────────────────────────────────────────── */
  const close = useCallback(() => {
    if (dontShowAgain) localStorage.setItem(STORAGE_KEY, "1");
    setOpen(false);
    setTimeout(() => { setStep(0); setDontShow(false); }, 350);
  }, [dontShowAgain]);

  const skipAll = useCallback(() => {
    localStorage.setItem(STORAGE_KEY, "1");
    setHasBeenSeen(true);
    setOpen(false);
    setTimeout(() => { setStep(0); setDontShow(false); }, 350);
  }, []);

  /* ── keyboard navigation ───────────────────────────────────────────── */
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape")    { close(); return; }
      if ((e.key === "ArrowRight" || e.key === "Enter") && step < STEPS.length - 1) {
        setStep(s => s + 1);
      }
      if (e.key === "ArrowLeft" && step > 0) {
        setStep(s => s - 1);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, step, close]);

  const current = STEPS[step];
  const isLast  = step === STEPS.length - 1;
  const isFirst = step === 0;

  /* ── render ────────────────────────────────────────────────────────── */
  return (
    <>
      {/* ── Floating trigger ─────────────────────────────────────────── */}
      <div className="fixed bottom-6 right-6 z-40">
        {/* Pulse beacon — shown only when user has NOT seen the guide yet */}
        {!hasBeenSeen && (
          <span
            className="absolute inset-0 rounded-full pointer-events-none"
            style={{ animation: "tutorial-pulse 2.2s ease-out infinite", background: "rgba(212,168,83,0.30)" }}
          />
        )}
        <button
          onClick={() => { setHasBeenSeen(true); setOpen(true); }}
          aria-label="Open app guide"
          className="relative flex items-center gap-2 rounded-full px-4 py-2 text-sm font-bold transition-all hover:scale-105 active:scale-95"
          style={{
            background: "linear-gradient(135deg, #d4a853 0%, #b8860b 100%)",
            boxShadow:  "0 4px 20px rgba(212,168,83,0.45), 0 0 0 2px rgba(212,168,83,0.20)",
            color:      "#0d1520",
            border:     "none",
          }}
        >
          <span className="text-base leading-none font-black">?</span>
          <span className="hidden sm:inline tracking-tight">Guide</span>
        </button>
      </div>

      <style>{`
        @keyframes tutorial-pulse {
          0%   { transform: scale(1);   opacity: 0.7; }
          70%  { transform: scale(2.0); opacity: 0;   }
          100% { transform: scale(2.0); opacity: 0;   }
        }
      `}</style>

      {/* ── Modal overlay ────────────────────────────────────────────── */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.22 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
            style={{ background: "rgba(5,10,18,0.88)", backdropFilter: "blur(10px)" }}
            onClick={e => { if (e.target === e.currentTarget) close(); }}
          >
            <motion.div
              initial={{ opacity: 0, y: 28, scale: 0.96 }}
              animate={{ opacity: 1, y: 0,  scale: 1    }}
              exit={{    opacity: 0, y: 18, scale: 0.96 }}
              transition={{ type: "spring", stiffness: 340, damping: 28 }}
              className="relative w-full overflow-hidden rounded-2xl"
              style={{
                maxWidth: "780px",
                background: "var(--surface-1, #0d1520)",
                border:     "1px solid var(--border, rgba(255,255,255,0.08))",
                boxShadow:  "0 32px 88px rgba(0,0,0,0.65)",
              }}
            >

              {/* ── TWO-COLUMN GRID ────────────────────────────────── */}
              <div className="grid" style={{ gridTemplateColumns: "200px 1fr" }}>

                {/* LEFT PANEL — color accent, icon, step map */}
                <AnimatePresence mode="wait">
                  <motion.div
                    key={`left-${step}`}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.20 }}
                    className="flex flex-col items-center justify-between gap-5 p-6"
                    style={{
                      background:   `linear-gradient(160deg, ${current.color}12 0%, ${current.color}06 100%)`,
                      borderRight:  `1px solid ${current.color}22`,
                      minHeight:    "420px",
                    }}
                  >
                    {/* Icon ring */}
                    <div className="flex flex-col items-center gap-3 pt-4">
                      <div
                        className="w-20 h-20 rounded-2xl flex items-center justify-center"
                        style={{
                          background: `${current.color}16`,
                          border:     `1.5px solid ${current.color}38`,
                          color:       current.color,
                          boxShadow:  `0 6px 28px ${current.color}28`,
                        }}
                      >
                        {current.icon}
                      </div>
                      <p
                        className="text-[10px] font-black uppercase tracking-widest text-center"
                        style={{ color: current.color }}
                      >
                        {current.tag}
                      </p>
                    </div>

                    {/* Vertical step map */}
                    <div className="flex flex-col gap-2 w-full">
                      {STEPS.map((s, i) => (
                        <button
                          key={i}
                          onClick={() => setStep(i)}
                          className="flex items-center gap-2.5 w-full rounded-xl px-2.5 py-1.5 text-left transition-all"
                          style={{
                            background: i === step ? `${current.color}14` : "transparent",
                            border:     `1px solid ${i === step ? current.color + "35" : "transparent"}`,
                          }}
                        >
                          {/* Small colored dot / check */}
                          <span
                            className="shrink-0 w-4 h-4 rounded-full flex items-center justify-center text-[8px]"
                            style={{
                              background: i < step
                                ? "#10b981"
                                : i === step
                                ? current.color
                                : "rgba(255,255,255,0.08)",
                              color: i <= step ? "#fff" : "rgba(255,255,255,0.25)",
                              boxShadow: i === step ? `0 0 8px ${current.color}60` : "none",
                            }}
                          >
                            {i < step ? <Check size={8} /> : i + 1}
                          </span>
                          <span
                            className="text-[10px] leading-tight font-medium truncate"
                            style={{
                              color: i === step
                                ? current.color
                                : i < step
                                ? "rgba(16,185,129,0.75)"
                                : "rgba(255,255,255,0.28)",
                            }}
                          >
                            {s.title}
                          </span>
                        </button>
                      ))}
                    </div>

                    {/* Keyboard hint */}
                    <div
                      className="flex items-center gap-1.5 text-[9px] pb-1"
                      style={{ color: "rgba(255,255,255,0.18)" }}
                    >
                      <Keyboard size={9} />
                      <span>← → navigate · Esc close</span>
                    </div>
                  </motion.div>
                </AnimatePresence>

                {/* RIGHT PANEL — title, body, bullets, navigation */}
                <div className="flex flex-col" style={{ minHeight: "420px" }}>

                  {/* Header row */}
                  <div className="flex items-start justify-between px-7 pt-6 pb-0 shrink-0">
                    <div>
                      <p className="text-[10px] font-bold uppercase tracking-wider mb-0.5"
                         style={{ color: current.color }}>
                        {step + 1} / {STEPS.length}
                      </p>
                      <h3 className="text-[22px] font-black leading-tight"
                          style={{ color: "var(--text-primary, #e2e8f0)" }}>
                        {current.title}
                      </h3>
                    </div>
                    {/* Close + Skip cluster */}
                    <div className="flex items-center gap-2 ml-4 shrink-0 mt-0.5">
                      <button
                        onClick={skipAll}
                        className="text-[10px] px-3 py-1.5 rounded-lg transition-all hover:bg-white/5"
                        style={{ color: "rgba(255,255,255,0.25)", border: "1px solid rgba(255,255,255,0.07)" }}
                      >
                        Skip tour
                      </button>
                      <button
                        onClick={close}
                        className="w-7 h-7 rounded-lg flex items-center justify-center transition-all hover:bg-white/10"
                        style={{ color: "var(--text-muted, #6b7280)" }}
                      >
                        <X size={13} />
                      </button>
                    </div>
                  </div>

                  {/* Divider */}
                  <div className="mx-7 mt-3 mb-4 h-px" style={{ background: "rgba(255,255,255,0.06)" }} />

                  {/* Content — scrollable if it ever overflows */}
                  <AnimatePresence mode="wait">
                    <motion.div
                      key={step}
                      initial={{ opacity: 0, x: 14 }}
                      animate={{ opacity: 1, x: 0  }}
                      exit={{    opacity: 0, x: -14 }}
                      transition={{ duration: 0.17 }}
                      className="flex-1 px-7 flex flex-col gap-3 overflow-y-auto"
                    >
                      <p className="text-sm leading-relaxed"
                         style={{ color: "var(--text-secondary, #94a3b8)" }}>
                        {current.body}
                      </p>

                      {/* Bullet list */}
                      {current.bullets && (
                        <ul className="flex flex-col gap-1.5">
                          {current.bullets.map((b, i) => (
                            <li key={i} className="flex items-start gap-2 text-[12px] leading-snug"
                                style={{ color: "var(--text-secondary, #94a3b8)" }}>
                              <span
                                className="shrink-0 w-1.5 h-1.5 rounded-full mt-1.5"
                                style={{ background: current.color, opacity: 0.7 }}
                              />
                              {b}
                            </li>
                          ))}
                        </ul>
                      )}

                      {/* CTA link */}
                      {current.cta && (
                        <Link
                          href={current.cta.href}
                          onClick={skipAll}
                          className="self-start inline-flex items-center gap-2 text-xs font-bold px-4 py-2 rounded-xl transition-all hover:opacity-80 mt-1"
                          style={{
                            background: `${current.color}16`,
                            color:       current.color,
                            border:     `1px solid ${current.color}32`,
                          }}
                        >
                          {current.cta.label}
                        </Link>
                      )}
                    </motion.div>
                  </AnimatePresence>

                  {/* Footer — navigation + don't-show-again */}
                  <div
                    className="shrink-0 px-7 pt-4 pb-5 mt-2"
                    style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}
                  >
                    {/* Don't show again (only on last step) */}
                    {isLast && (
                      <label className="flex items-center gap-2 mb-3 cursor-pointer self-start">
                        <input
                          type="checkbox"
                          checked={dontShowAgain}
                          onChange={e => setDontShow(e.target.checked)}
                          className="rounded"
                          style={{ accentColor: current.color }}
                        />
                        <span className="text-[11px]" style={{ color: "rgba(255,255,255,0.35)" }}>
                          Don&apos;t show this guide automatically again
                        </span>
                      </label>
                    )}

                    <div className="flex items-center justify-between gap-3">
                      <button
                        onClick={() => setStep(s => s - 1)}
                        disabled={isFirst}
                        className="flex items-center gap-1.5 text-xs px-4 py-2 rounded-xl transition-all disabled:opacity-25 hover:bg-white/5"
                        style={{ color: "var(--text-muted)", border: "1px solid var(--border, rgba(255,255,255,0.08))" }}
                      >
                        <ChevronLeft size={13} /> Previous
                      </button>

                      {isLast ? (
                        <div className="flex items-center gap-2">
                          <Link
                            href="/analyse"
                            onClick={skipAll}
                            className="flex items-center gap-1.5 text-xs px-5 py-2 rounded-xl font-bold transition-all hover:opacity-90 whitespace-nowrap"
                            style={{
                              background: "linear-gradient(135deg, #d4a853 0%, #b8860b 100%)",
                              color: "#0d1520",
                              boxShadow: "0 2px 12px rgba(212,168,83,0.40)",
                            }}
                          >
                            Start first analysis →
                          </Link>
                          <button
                            onClick={close}
                            className="flex items-center gap-1.5 text-xs px-5 py-2 rounded-xl font-bold transition-all hover:opacity-90"
                            style={{
                              background: "linear-gradient(135deg, #10b981, #059669)",
                              color: "#fff",
                              boxShadow: "0 2px 12px rgba(16,185,129,0.35)",
                            }}
                          >
                            <Check size={13} /> Got it!
                          </button>
                        </div>
                      ) : (
                        <button
                          onClick={() => setStep(s => s + 1)}
                          className="flex items-center gap-1.5 text-xs px-5 py-2 rounded-xl font-bold transition-all hover:opacity-90"
                          style={{
                            background: current.color,
                            color: "#fff",
                            boxShadow: `0 2px 12px ${current.color}40`,
                          }}
                        >
                          Next <ChevronRight size={13} />
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}



