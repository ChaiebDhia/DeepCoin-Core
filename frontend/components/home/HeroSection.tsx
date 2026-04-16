"use client";

/**
 * components/home/HeroSection.tsx
 * =================================
 * Full-viewport hero — DUAL THEME (light + dark)
 *
 * FIX 1 – Full bleed: uses the classic 100vw breakout trick
 *   `margin-left: calc(-50vw + 50%)` + `width: 100vw` to escape
 *   any max-width container the parent layout imposes.
 *
 * FIX 2 – Theme-aware: all colors reference CSS variables defined in
 *   globals.css :root (light) and .dark (dark). No hardcoded colors
 *   on backgrounds — only brand/surface tokens.
 */

import { motion }   from "framer-motion";
import Link         from "next/link";
import { ArrowRight, ChevronDown, Cpu, Database, Zap, FileText } from "lucide-react";
import { useTranslations } from "next-intl";

/* ── Floating coin rings ────────────────────────────────────────────────── */
const COINS = [
  { size: 110, top: "7%",  left:  "3%",  delay: 0,   dur: 7  },
  { size:  60, top: "65%", left:  "2%",  delay: 1.5, dur: 9  },
  { size: 148, top: "10%", right: "3%",  delay: 0.7, dur: 8  },
  { size:  70, top: "68%", right: "5%",  delay: 2.1, dur: 6  },
  { size:  46, top: "42%", left:  "12%", delay: 3.0, dur: 11 },
  { size:  76, top: "32%", right: "14%", delay: 1.2, dur: 10 },
  { size:  40, top: "80%", left:  "20%", delay: 2.4, dur: 8  },
  { size:  54, top: "76%", right: "20%", delay: 0.9, dur: 9  },
];

/* ── Neural-net SVG lines ───────────────────────────────────────────────── */
const NEURAL_LINES = [
  { x1: "6%",  y1: "18%", x2: "20%", y2: "44%" },
  { x1: "20%", y1: "44%", x2: "36%", y2: "28%" },
  { x1: "36%", y1: "28%", x2: "54%", y2: "54%" },
  { x1: "54%", y1: "54%", x2: "70%", y2: "33%" },
  { x1: "70%", y1: "33%", x2: "88%", y2: "58%" },
  { x1: "14%", y1: "72%", x2: "34%", y2: "56%" },
  { x1: "58%", y1: "76%", x2: "82%", y2: "64%" },
];
const NEURAL_DOTS = Array.from({ length: 24 }, (_, i) => ({
  cx: `${(i * 41 + 9) % 100}%`,
  cy: `${(i * 57 + 13) % 100}%`,
  r:  i % 4 === 0 ? 3 : 1.6,
}));

/* ── Pipeline badges ────────────────────────────────────────────────────── */
const BADGES = [
  { icon: Cpu,      label: "EfficientNet-B3",  step: 1 },
  { icon: Database, label: "47,705 RAG Chunks", step: 2 },
  { icon: Zap,      label: "Multi-Agent LLM",   step: 3 },
  { icon: FileText, label: "PDF Report",         step: 4 },
];

/* ── Component ─────────────────────────────────────────────────────────── */
export function HeroSection() {
  const t = useTranslations("HeroSection");
  return (
    /**
     * FULL-BLEED BREAKOUT
     * -------------------
     * Most Next.js layouts wrap <main> in a max-w-* container.
     * Setting width:100vw + margin-left:calc(-50vw + 50%) makes this
     * section punch through that container edge-to-edge on every viewport.
     * `overflow-hidden` on this element prevents any horizontal scrollbar.
     */
    <section
      className="relative min-h-[94vh] flex flex-col items-center justify-center overflow-hidden"
      style={{
        width:      "100vw",
        marginLeft: "calc(-50vw + 50%)",
      }}
    >
      {/* ═══════════════════════════════════════════════════════════════
          BACKGROUND LAYER — fully theme-aware via CSS vars
          Light: warm parchment + amber washes
          Dark:  deep navy + blue washes  (globals.css .dark overrides)
         ═══════════════════════════════════════════════════════════════ */}
      <div
        className="absolute inset-0 pointer-events-none select-none"
        aria-hidden
      >
        {/* Base fill — uses surface-0 so it matches the page bg token */}
        <div
          className="absolute inset-0"
          style={{ backgroundColor: "var(--surface-0)" }}
        />

        {/* ── LIGHT-MODE exclusive layers ──────────────────────────── */}
        {/* Warm parchment gradient — hidden in dark via opacity on a
            wrapper we control with a CSS class trick. We use an SVG
            feBlend approach via layered divs: light layers use
            mix-blend-mode multiply so they become invisible on dark
            backgrounds naturally. */}

        {/* Horizontal engraving lines (numismatic texture) */}
        <div
          className="absolute inset-0"
          style={{
            backgroundImage:
              "repeating-linear-gradient(0deg, var(--brand-gold) 0px, var(--brand-gold) 1px, transparent 1px, transparent 7px)",
            opacity: 0.04,
          }}
        />

        {/* Fine dot grid (data / ML grid feel) */}
        <div
          className="absolute inset-0"
          style={{
            backgroundImage:
              "radial-gradient(circle, var(--text-secondary) 1px, transparent 1px)",
            backgroundSize: "32px 32px",
            opacity: 0.05,
          }}
        />

        {/* Diagonal mesh overlay */}
        <div
          className="absolute inset-0"
          style={{
            backgroundImage:
              "linear-gradient(45deg, var(--brand-gold) 1px, transparent 1px), " +
              "linear-gradient(-45deg, var(--brand-gold) 1px, transparent 1px)",
            backgroundSize: "64px 64px",
            opacity: 0.025,
          }}
        />

        {/* Neural network SVG — color adapts via currentColor on parent */}
        <svg
          className="absolute inset-0 w-full h-full"
          xmlns="http://www.w3.org/2000/svg"
          preserveAspectRatio="xMidYMid slice"
          style={{ opacity: 0.18 }}
        >
          {NEURAL_LINES.map((l, i) => (
            <line
              key={i}
              x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2}
              stroke="var(--brand-gold)"
              strokeWidth="0.7"
              strokeDasharray="5 7"
            />
          ))}
          {NEURAL_DOTS.map((d, i) => (
            <circle
              key={i}
              cx={d.cx} cy={d.cy} r={d.r}
              fill="var(--brand-mid)"
              opacity={i % 4 === 0 ? 0.5 : 0.2}
            />
          ))}
        </svg>

        {/* Top-left warm amber radial wash */}
        <div
          className="absolute"
          style={{
            top: "-15%", left: "-10%",
            width: "60%", height: "75%",
            background:
              "radial-gradient(ellipse at 30% 30%, rgba(163,126,44,0.14) 0%, transparent 65%)",
          }}
        />

        {/* Bottom-right cool blue radial wash */}
        <div
          className="absolute"
          style={{
            bottom: "-20%", right: "-12%",
            width: "65%", height: "75%",
            background:
              "radial-gradient(ellipse at 70% 70%, var(--brand-light) 0%, transparent 65%)",
            opacity: 0.07,
          }}
        />

        {/* Centre lift — softens the middle behind the headline */}
        <div
          className="absolute inset-0"
          style={{
            background:
              "radial-gradient(ellipse 68% 52% at 50% 46%, var(--surface-1) 0%, transparent 70%)",
            opacity: 0.6,
          }}
        />

        {/* Top & bottom engraved rules */}
        <div
          className="absolute top-0 left-0 right-0 h-[3px]"
          style={{
            background:
              "linear-gradient(90deg, transparent, rgba(163,126,44,0.35) 20%, rgba(163,126,44,0.6) 50%, rgba(163,126,44,0.35) 80%, transparent)",
          }}
        />
        <div
          className="absolute bottom-0 left-0 right-0 h-[2px]"
          style={{
            background:
              "linear-gradient(90deg, transparent, rgba(163,126,44,0.2) 25%, rgba(163,126,44,0.4) 50%, rgba(163,126,44,0.2) 75%, transparent)",
          }}
        />

        {/* ── Floating coin rings (both themes) ────────────────────── */}
        {COINS.map((c, i) => (
          <motion.div
            key={i}
            className="absolute rounded-full"
            style={{
              width:  c.size,
              height: c.size,
              top:    c.top,
              left:   (c as { left?: string }).left,
              right:  (c as { right?: string }).right,
              border: "1.5px solid var(--brand-gold)",
              opacity: 0.28,
              background:
                "radial-gradient(circle at 38% 38%, var(--brand-gold) 0%, transparent 70%)",
            }}
            animate={{ y: [0, -14, 0], rotate: [0, 3, 0] }}
            transition={{
              duration: c.dur,
              delay:    c.delay,
              repeat:   Infinity,
              ease:     "easeInOut",
            }}
          />
        ))}
      </div>

      {/* ═══════════════════════════════════════════════════════════════
          MAIN CONTENT
         ═══════════════════════════════════════════════════════════════ */}
      <div className="relative z-10 text-center max-w-4xl mx-auto px-6">

        {/* Project badge */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-semibold mb-8"
          style={{
            border:          "1px solid color-mix(in srgb, var(--brand-gold) 35%, transparent)",
            color:           "var(--brand-gold)",
            backgroundColor: "color-mix(in srgb, var(--brand-gold) 10%, transparent)",
            letterSpacing:   "0.04em",
          }}
        >
          <span
            className="w-1.5 h-1.5 rounded-full animate-pulse"
            style={{ backgroundColor: "var(--brand-gold)" }}
          />
          PFE 2026 · Dhia Chaieb · ESPRIT × YEBNI, Tunisia
        </motion.div>

        {/* Headline */}
        <motion.h1
          initial={{ opacity: 0, y: 28 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.65, delay: 0.15 }}
          className="text-5xl sm:text-6xl lg:text-7xl font-black tracking-tight leading-[1.08] mb-6"
          style={{ fontFamily: "'Georgia', 'Times New Roman', serif" }}
        >
          <span style={{ color: "var(--text-primary)" }}>{t("identify")}</span>
          <br />
          <span style={{ color: "var(--brand-primary)" }}>{t("ancient_coin")}</span>
          <br />
          <span
            style={{
              color: "transparent",
              backgroundImage:
                "linear-gradient(135deg, var(--brand-gold) 0%, color-mix(in srgb, var(--brand-gold) 70%, #fff) 50%, var(--brand-gold) 100%)",
              WebkitBackgroundClip: "text",
              backgroundClip: "text",
              display: "inline-block",
            }}
          >
            {t("in_seconds")}
          </span>
        </motion.h1>

        {/* Subtitle */}
        <motion.p
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="text-base sm:text-lg max-w-2xl mx-auto leading-relaxed mb-10"
          style={{ color: "var(--text-secondary)" }}
        >
          {t("subtitle_1")}{" "}
          <strong style={{ color: "var(--text-primary)" }}>{t("subtitle_efficientnet")}</strong>{" "}
          {t("subtitle_2")}{" "}
          <strong style={{ color: "var(--text-primary)" }}>{t("subtitle_rag")}</strong>{" "}
          {t("subtitle_3")}</motion.p>

        {/* CTAs */}
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.45 }}
          className="flex flex-wrap items-center justify-center gap-4 mb-12"
        >
          <Link
            href="/analyse"
            className="inline-flex items-center gap-2 px-8 py-3.5 rounded-xl font-bold text-sm transition-all duration-200 hover:scale-105 active:scale-100"
            style={{
              background:
                "linear-gradient(135deg, var(--brand-gold) 0%, color-mix(in srgb, var(--brand-gold) 75%, #000) 100%)",
              color:     "var(--surface-0)",
              boxShadow: "0 4px 18px color-mix(in srgb, var(--brand-gold) 35%, transparent)",
            }}
          >
            {t("cta_analyse")}
            <ArrowRight size={16} />
          </Link>
          <Link
            href="#how-it-works"
            className="inline-flex items-center gap-2 px-8 py-3.5 rounded-xl font-bold text-sm border transition-all duration-200 hover:scale-105"
            style={{
              borderColor:     "var(--border-strong, var(--border))",
              color:           "var(--text-secondary)",
              backgroundColor: "color-mix(in srgb, var(--surface-1) 80%, transparent)",
              backdropFilter:  "blur(6px)",
            }}
          >
            {t("cta_how")}
            <ChevronDown size={16} />
          </Link>
        </motion.div>

        {/* Technology badges */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.65 }}
          className="flex flex-wrap items-center justify-center gap-3"
        >
          {BADGES.map(({ icon: Icon, label, step }, i) => (
            <motion.span
              key={label}
              initial={{ opacity: 0, scale: 0.82 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.7 + i * 0.1 }}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold border"
              style={{
                borderColor:     `var(--color-step-${step}-bd)`,
                backgroundColor: `var(--color-step-${step}-bg)`,
                color:           `var(--color-step-${step}-tx)`,
                boxShadow:       `0 1px 4px var(--color-step-${step}-sd)`,
              }}
            >
              <Icon size={11} />
              {label}
            </motion.span>
          ))}
        </motion.div>
      </div>
    </section>
  );
}


