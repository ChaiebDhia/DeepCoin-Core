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

 *   - Keyboard navigation: ? previous, ? / Enter next, Esc close

 *   - Two-column layout: colored left panel (icon + step map) + right content panel

 *   - Framer Motion slide transitions on step change

 *   - "{t('skip')}" available from every step

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
import { useTranslations } from "next-intl";





/* --- localStorage key ----------------------------------------------------- */



const STORAGE_KEY = "deepcoin_guide_seen_v2";



/* --- step definitions ----------------------------------------------------- */



interface Step {

  icon:  React.ReactNode;

  color: string;

  tag:   string;

  title: string;

  body:  string;

  bullets?: string[];

  cta?:  { label: string; href: string } | null;

}






/* --- component ------------------------------------------------------------ */



export default function TutorialModal() {

    const t = useTranslations("Guide");

  const STEPS: Step[] = [
    {
      icon:  <Upload   size={32} />,
      color: "#3b82f6",
      tag:   t("step1_tag"),
      title: t("step1_title"),
      body:  t("step1_body"),
      bullets: [
        t("step1_b1"),
        t("step1_b2"),
        t("step1_b3"),
        t("step1_b4"),
        t("step1_b5"),
      ],
      cta:   { label: t("step1_cta") + " ↗", href: "/analyse" },
    },
    {
      icon:  <Cpu      size={32} />,
      color: "#8b5cf6",
      tag:   t("step2_tag"),
      title: t("step2_title"),
      body:  t("step2_body"),
      bullets: [
        t("step2_b1"),
        t("step2_b2"),
        t("step2_b3"),
        t("step2_b4"),
      ],
      cta:   null,
    },
    {
      icon:  <BookOpen size={32} />,
      color: "#10b981",
      tag:   t("step3_tag"),
      title: t("step3_title"),
      body:  t("step3_body"),
      bullets: [
        t("step3_b1"),
        t("step3_b2"),
        t("step3_b3"),
        t("step3_b4"),
      ],
      cta:   { label: t("step3_cta") + " ↗", href: "/explore" },
    },
    {
      icon:  <MessageSquare size={32} />,
      color: "#a78bfa",
      tag:   t("step4_tag"),
      title: t("step4_title"),
      body:  t("step4_body"),
      bullets: [
        t("step4_b1"),
        t("step4_b2"),
        t("step4_b3"),
        t("step4_b4"),
        t("step4_b5"),
      ],
      cta:   { label: t("step4_cta") + " ↗", href: "/chat" },
    },
    {
      icon:  <Download size={32} />,
      color: "#f59e0b",
      tag:   t("step5_tag"),
      title: t("step5_title"),
      body:  t("step5_body"),
      bullets: [
        t("step5_b1"),
        t("step5_b2"),
        t("step5_b3"),
        t("step5_b4"),
      ],
      cta:   { label: t("step5_cta") + " ↗", href: "/history" },
    },
    {
      icon:  <ThumbsDown size={32} />,
      color: "#ef4444",
      tag:   t("step6_tag"),
      title: t("step6_title"),
      body:  t("step6_body"),
      bullets: [
        t("step6_b1"),
        t("step6_b2"),
        t("step6_b3"),
        t("step6_b4"),
      ],
      cta:   null,
    },
  ];


const [open, setOpen]               = useState(false);

  const [step, setStep]               = useState(0);

  const [dontShowAgain, setDontShow]  = useState(false);

  const [hasBeenSeen, setHasBeenSeen] = useState(true); // default true ? no pulse until we check



  /* Auto-open for first-time visitors */

  useEffect(() => {

    const seen = typeof window !== "undefined" && !!localStorage.getItem(STORAGE_KEY);

    setHasBeenSeen(seen);

    if (!seen) {

      const t = setTimeout(() => setOpen(true), 1400);

      return () => clearTimeout(t);

    }

  }, []);



  /* -- close helpers --------------------------------------------------- */

  const close = useCallback(() => {

    localStorage.setItem(STORAGE_KEY, "1");

    setOpen(false);

    setTimeout(() => { setStep(0); setDontShow(false); }, 350);

  }, [dontShowAgain]);



  const skipAll = useCallback(() => {

    localStorage.setItem(STORAGE_KEY, "1");

    setHasBeenSeen(true);

    setOpen(false);

    setTimeout(() => { setStep(0); setDontShow(false); }, 350);

  }, []);



  /* -- keyboard navigation --------------------------------------------- */

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



  /* -- render ---------------------------------------------------------- */

  return (

    <>

      {/* -- Floating trigger ------------------------------------------- */}

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



      {/* -- Modal overlay ---------------------------------------------- */}

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



              {/* -- TWO-COLUMN GRID: left panel hidden on mobile, visible sm+ -- */}

              <div className="grid grid-cols-1 sm:grid-cols-[200px_1fr]">



                {/* LEFT PANEL — hidden on mobile, shown sm+ */}

                <AnimatePresence mode="wait">

                  <motion.div

                    key={`left-${step}`}

                    initial={{ opacity: 0 }}

                    animate={{ opacity: 1 }}

                    exit={{ opacity: 0 }}

                    transition={{ duration: 0.20 }}

                    className="hidden sm:flex flex-col items-center justify-between gap-5 p-6"

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

                                : "var(--text-muted)",

                              color: i <= step ? "var(--text-primary)" : "var(--text-muted)",

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

                                : "var(--text-muted)",

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

                      style={{ color: "var(--text-muted)" }}

                    >

                      <Keyboard size={9} />

                      <span>? ? navigate · Esc close</span>

                    </div>

                  </motion.div>

                </AnimatePresence>



                {/* RIGHT PANEL — title, body, bullets, navigation */}

                <div className="flex flex-col" style={{ minHeight: "420px" }}>



                  {/* Mobile-only: icon + step dots (left panel hidden on small screens) */}

                  <div className="flex sm:hidden items-center justify-between gap-3 px-5 pt-4 pb-0">

                    <div className="flex items-center gap-2.5">

                      <div

                        className="w-9 h-9 rounded-xl flex items-center justify-center shrink-0"

                        style={{ background: `${current.color}16`, border: `1px solid ${current.color}38`, color: current.color }}

                      >

                        <span style={{ transform: "scale(0.7)" }}>{current.icon}</span>

                      </div>

                      <p className="text-[10px] font-black uppercase tracking-widest" style={{ color: current.color }}>

                        {current.tag}

                      </p>

                    </div>

                    {/* Mobile dot row */}

                    <div className="flex items-center gap-1">

                      {STEPS.map((_, i) => (

                        <button key={i} onClick={() => setStep(i)}

                          className="rounded-full transition-all"

                          style={{

                            width: i === step ? "14px" : "5px",

                            height: "5px",

                            background: i === step ? current.color : i < step ? "#10b981" : "var(--text-muted)",

                          }}

                        />

                      ))}

                    </div>

                  </div>



                  {/* Divider on mobile only */}

                  <div className="block sm:hidden mx-5 mt-3 h-px" style={{ background: "var(--text-muted)" }} />



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

                        style={{ color: "var(--text-muted)", border: "1px solid rgba(255,255,255,0.07)" }}

                      >

                        {t('skip')}

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

                  <div className="mx-7 mt-3 mb-4 h-px" style={{ background: "var(--text-muted)" }} />



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

                        <span className="text-[11px]" style={{ color: "var(--text-muted)" }}>

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

                            Start first analysis ?

                          </Link>

                          <button

                            onClick={close}

                            className="flex items-center gap-1.5 text-xs px-5 py-2 rounded-xl font-bold transition-all hover:opacity-90"

                            style={{

                              background: "linear-gradient(135deg, #10b981, #059669)",

                              color: "var(--text-primary)",

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

                            color: "var(--text-primary)",

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







