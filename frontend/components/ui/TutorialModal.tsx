"use client";

/**
 * components/ui/TutorialModal.tsx
 * =================================
 * WHAT: A floating "?" help button (bottom-right corner) that opens a full-screen
 *       step-by-step tutorial modal explaining the DeepCoin app to first-time users.
 *
 * WHY: Users -- especially museum curators and students -- may not understand the
 *      AI pipeline, confidence levels, or how to use the Explore/Chat features.
 *      A concise 6-step guided tour eliminates onboarding friction.
 *
 * HOW: Client component with local state (open + step). Framer Motion AnimatePresence
 *      for the overlay fade and slide-up card. Uses the same dark navy design system
 *      as the rest of the app (CSS custom properties via inline styles).
 *
 * INTEGRATION: Rendered once in app/layout.tsx -- available on every page.
 */

import { useState }         from "react";
import { motion, AnimatePresence } from "framer-motion";
import Link                 from "next/link";
import {
  X, ChevronLeft, ChevronRight, Check,
  Upload, Cpu, BookOpen, MessageSquare, Download, ThumbsDown,
} from "lucide-react";

/* ─── step definitions ───────────────────────────────────────────────────── */

const STEPS = [
  {
    icon:  <Upload   size={28} />,
    color: "#3b82f6",
    title: "Upload a coin photo",
    body:  "Go to the Analyse page and drag-and-drop (or click to select) any photo of an ancient coin. The system accepts JPEG, PNG, and WebP. Optionally enable TTA (Test-Time Augmentation) for higher accuracy at the cost of a few extra seconds.",
    cta:   { label: "Go to Analyse", href: "/analyse" },
  },
  {
    icon:  <Cpu      size={28} />,
    color: "#8b5cf6",
    title: "Understand the CNN result",
    body:  "DeepCoin's EfficientNet-B3 model classifies the coin against 438 trained types from the Corpus Nummorum. You'll see a confidence score and route badge:\n\n- Green 'Identified' -- confidence >= 70 % -- high certainty\n- Teal 'Consistent Match' -- TTA consensus (7/8 passes agree)\n- Purple 'Deep Search' -- low confidence, investigator mode",
    cta:   null,
  },
  {
    icon:  <BookOpen size={28} />,
    color: "#10b981",
    title: "Browse the Explore gallery",
    body:  "The /explore page shows all past analyses (yours and public ones). You can filter by route (Historian / Validator / Investigator) to browse only high-confidence identifications or deep-search results. No sign-in required.",
    cta:   { label: "Open Explore", href: "/explore" },
  },
  {
    icon:  <MessageSquare size={28} />,
    color: "#a78bfa",
    title: "Ask the AI Chat",
    body:  "DeepCoin AI answers any numismatic question grounded in 47,705 knowledge-base chunks covering 9,541 coin types. Ask about dynasties, mint cities, iconography, or denominations. Answers cite specific CN record IDs so you can verify every fact.",
    cta:   { label: "Open AI Chat", href: "/chat" },
  },
  {
    icon:  <Download size={28} />,
    color: "#f59e0b",
    title: "Download your PDF report",
    body:  "Every analysis generates a professional PDF report with a historical narrative, forensic material check, visual attributes, and a top-5 candidate table. PDF links are active for 30 days. You can access past reports from your History page.",
    cta:   { label: "View History", href: "/history" },
  },
  {
    icon:  <ThumbsDown size={28} />,
    color: "#ef4444",
    title: "Mark wrong results",
    body:  "If the CNN misidentified a coin, click 'Mark as wrong' on the analysis panel and enter the correct CN type ID. Your correction trains the active-learning loop -- future analyses improve based on expert feedback. Every correction is recorded in the admin panel.",
    cta:   null,
  },
];

/* ─── component ──────────────────────────────────────────────────────────── */

export default function TutorialModal() {
  const [open, setOpen] = useState(false);
  const [step, setStep] = useState(0);

  const current = STEPS[step];
  const isLast  = step === STEPS.length - 1;
  const isFirst = step === 0;

  function close() { setOpen(false); setTimeout(() => setStep(0), 300); }

  return (
    <>
      {/* Floating "?" trigger button */}
      <button
        onClick={() => setOpen(true)}
        aria-label="Open tutorial"
        className="fixed bottom-6 right-6 z-40 w-12 h-12 rounded-full flex items-center justify-center text-base font-black transition-all hover:scale-110 active:scale-95"
        style={{
          background:  "linear-gradient(135deg, #d4a853 0%, #b8860b 100%)",
          boxShadow:   "0 4px 18px rgba(212,168,83,0.45), 0 0 0 3px rgba(212,168,83,0.15)",
          color:       "#0d1520",
          border:      "none",
        }}
      >
        ?
      </button>

      {/* Modal overlay */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
            style={{ background: "rgba(5,10,18,0.85)", backdropFilter: "blur(8px)" }}
            onClick={e => { if (e.target === e.currentTarget) close(); }}
          >
            <motion.div
              initial={{ opacity: 0, y: 24, scale: 0.97 }}
              animate={{ opacity: 1, y: 0,  scale: 1    }}
              exit={{    opacity: 0, y: 16, scale: 0.97 }}
              transition={{ type: "spring", stiffness: 360, damping: 30 }}
              className="relative w-full max-w-lg rounded-2xl overflow-hidden"
              style={{
                background:  "var(--surface-1, #0d1520)",
                border:      "1px solid var(--border, rgba(255,255,255,0.08))",
                boxShadow:   "0 24px 80px rgba(0,0,0,0.60)",
              }}
            >
              {/* Close */}
              <button onClick={close}
                className="absolute top-4 right-4 w-7 h-7 rounded-lg flex items-center justify-center transition-all hover:bg-white/10 z-10"
                style={{ color: "var(--text-muted, #6b7280)" }}>
                <X size={14} />
              </button>

              {/* Progress dots */}
              <div className="flex items-center justify-center gap-1.5 pt-5 pb-2">
                {STEPS.map((_, i) => (
                  <button key={i} onClick={() => setStep(i)}
                    className="transition-all rounded-full"
                    style={{
                      width:      i === step ? "20px" : "6px",
                      height:     "6px",
                      background: i === step ? current.color : "rgba(255,255,255,0.15)",
                    }}
                  />
                ))}
              </div>

              {/* Step content */}
              <AnimatePresence mode="wait">
                <motion.div key={step}
                  initial={{ opacity: 0, x: 12 }}
                  animate={{ opacity: 1, x: 0  }}
                  exit={{    opacity: 0, x: -12 }}
                  transition={{ duration: 0.18 }}
                  className="px-8 py-6 flex flex-col items-center text-center gap-4"
                >
                  {/* Icon circle */}
                  <div className="w-16 h-16 rounded-2xl flex items-center justify-center"
                       style={{ background: current.color + "18", border: `1px solid ${current.color}35`, color: current.color, boxShadow: `0 4px 20px ${current.color}25` }}>
                    {current.icon}
                  </div>

                  {/* Step label */}
                  <div>
                    <p className="text-[10px] font-bold uppercase tracking-wider mb-1"
                       style={{ color: current.color }}>
                      Step {step + 1} of {STEPS.length}
                    </p>
                    <h3 className="text-xl font-black" style={{ color: "var(--text-primary, #e2e8f0)" }}>
                      {current.title}
                    </h3>
                  </div>

                  <p className="text-sm leading-relaxed whitespace-pre-wrap text-left"
                     style={{ color: "var(--text-secondary, #94a3b8)" }}>
                    {current.body}
                  </p>

                  {current.cta && (
                    <Link href={current.cta.href} onClick={close}
                      className="inline-flex items-center gap-2 text-xs font-semibold px-4 py-2 rounded-xl transition-all hover:opacity-80"
                      style={{ background: current.color + "18", color: current.color, border: `1px solid ${current.color}30` }}>
                      {current.cta.label}
                    </Link>
                  )}
                </motion.div>
              </AnimatePresence>

              {/* Navigation */}
              <div className="flex items-center justify-between px-6 pb-6 gap-3">
                <button onClick={() => setStep(s => s - 1)} disabled={isFirst}
                  className="flex items-center gap-1.5 text-xs px-4 py-2 rounded-xl transition-all disabled:opacity-30 hover:bg-white/5"
                  style={{ color: "var(--text-muted)", border: "1px solid var(--border)" }}>
                  <ChevronLeft size={13} /> Previous
                </button>

                {isLast ? (
                  <button onClick={close}
                    className="flex items-center gap-1.5 text-xs px-5 py-2 rounded-xl font-bold transition-all hover:opacity-90"
                    style={{ background: "linear-gradient(135deg, #10b981, #059669)", color: "#fff", boxShadow: "0 2px 12px rgba(16,185,129,0.35)" }}>
                    <Check size={13} /> Got it!
                  </button>
                ) : (
                  <button onClick={() => setStep(s => s + 1)}
                    className="flex items-center gap-1.5 text-xs px-4 py-2 rounded-xl font-bold transition-all hover:opacity-90"
                    style={{ background: current.color, color: "#fff", boxShadow: `0 2px 12px ${current.color}40` }}>
                    Next <ChevronRight size={13} />
                  </button>
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}


