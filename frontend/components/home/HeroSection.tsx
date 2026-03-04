"use client";

/**
 * components/home/HeroSection.tsx
 * =================================
 * Full-viewport landing hero.
 *
 * WHAT: Compelling above-the-fold section with animated background, headline,
 *       subtitle, two CTAs, pipeline badge strip, and a floating coin visual.
 *
 * WHY this structure:
 *   - Futuristic deep-tech feel via subtle grid lines + floating coin circles
 *   - Headline uses the `.animate-shimmer-text` gradient sweep (globals.css)
 *   - Two CTAs: primary "Analyse" scrolls to the embedded analyser section;
 *     secondary "How it works" scrolls to the pipeline explainer.
 *   - Pipeline badges anchor the hero to concrete technology — avoids vague marketing.
 *   - All Framer Motion animations use initial/animate (not whileInView) so
 *     they fire immediately on mount for the above-the-fold view.
 *
 * HOW it fits:
 *   Rendered as the first section inside app/page.tsx (server component).
 *   Uses "use client" because of Framer Motion and Lucide.
 *   No API calls — zero loading states here.
 */

import { motion }         from "framer-motion";
import { useSession }     from "next-auth/react";
import Link               from "next/link";
import { ArrowRight, ChevronDown, Cpu, Database, Zap, FileText } from "lucide-react";

/* ── Floating background coins ─────────────────────────────────────────── */

const COINS = [
  { size: 90,  top: "10%", left:  "6%",  delay: 0,   dur: 7  },
  { size: 55,  top: "68%", left:  "4%",  delay: 1.5, dur: 9  },
  { size: 130, top: "15%", right: "5%",  delay: 0.7, dur: 8  },
  { size: 65,  top: "72%", right: "7%",  delay: 2.1, dur: 6  },
  { size: 42,  top: "42%", left:  "14%", delay: 3.0, dur: 11 },
  { size: 70,  top: "35%", right: "16%", delay: 1.2, dur: 10 },
];

/* ── Pipeline badge definitions ─────────────────────────────────────────── */

const BADGES = [
  { icon: Cpu,      label: "EfficientNet-B3",   color: "#3b82f6" },
  { icon: Database, label: "47,705 RAG Chunks",  color: "#8b5cf6" },
  { icon: Zap,      label: "Multi-Agent LLM",    color: "#d4a853" },
  { icon: FileText, label: "PDF Report",          color: "#10b981" },
];

/* ── Component ─────────────────────────────────────────────────────────── */

export function HeroSection() {
  const { data: session } = useSession();

  /**
   * If the user is already authenticated, the CTA goes directly to /analyse.
   * If not (or while the session is loading), it redirects to /login with a
   * callbackUrl so NextAuth brings them back to /analyse after sign-in.
   * WHY: /analyse + the classification pipeline are post-auth features.
   */
  const analyseHref = session ? "/analyse" : "/login?callbackUrl=/analyse";

  return (
    <section
      className="relative min-h-[94vh] flex flex-col items-center justify-center overflow-hidden"
      style={{ marginLeft: "-1.25rem", marginRight: "-1.25rem", padding: "0 1.25rem" }}
    >
      {/* ── Decorative background ── */}
      <div className="absolute inset-0 pointer-events-none select-none" aria-hidden>
        {/* Radial glow centred behind the headline */}
        <div
          className="absolute inset-0"
          style={{
            background:
              "radial-gradient(ellipse 85% 65% at 50% 48%, rgba(30,95,168,0.14) 0%, transparent 70%)",
          }}
        />
        {/* Subtle grid */}
        <div
          className="absolute inset-0 opacity-[0.035]"
          style={{
            backgroundImage:
              "linear-gradient(var(--brand-light) 1px, transparent 1px), " +
              "linear-gradient(90deg, var(--brand-light) 1px, transparent 1px)",
            backgroundSize: "64px 64px",
          }}
        />
        {/* Floating coin circles */}
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
              border: "1px solid rgba(212,168,83,0.14)",
              background:
                "radial-gradient(circle at 38% 38%, rgba(212,168,83,0.08) 0%, transparent 70%)",
            }}
            animate={{ y: [0, -14, 0], rotate: [0, 4, 0] }}
            transition={{
              duration: c.dur,
              delay:    c.delay,
              repeat:   Infinity,
              ease:     "easeInOut",
            }}
          />
        ))}
      </div>

      {/* ── Main content ── */}
      <div className="relative z-10 text-center max-w-4xl mx-auto px-4">

        {/* Project badge */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border text-xs font-semibold mb-8"
          style={{
            borderColor:     "rgba(212,168,83,0.4)",
            color:           "var(--brand-gold)",
            backgroundColor: "rgba(212,168,83,0.08)",
          }}
        >
          <span className="w-1.5 h-1.5 rounded-full bg-[var(--brand-gold)] animate-pulse" />
          PFE 2026 · ESPRIT School of Engineering × YEBNI · Tunisia
        </motion.div>

        {/* Headline ── shimmer on "Identify any" */}
        <motion.h1
          initial={{ opacity: 0, y: 28 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.65, delay: 0.15 }}
          className="text-5xl sm:text-6xl lg:text-7xl font-black tracking-tight leading-[1.08] mb-6"
        >
          <span className="animate-shimmer-text">Identify any</span>
          <br />
          <span style={{ color: "var(--text-primary)" }}>ancient coin</span>
          <br />
          <span style={{ color: "var(--brand-gold)" }}>in seconds.</span>
        </motion.h1>

        {/* Subtitle */}
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="text-base sm:text-lg max-w-2xl mx-auto leading-relaxed mb-10"
          style={{ color: "var(--text-secondary)" }}
        >
          DeepCoin combines an{" "}
          <strong className="text-white">EfficientNet-B3 CNN</strong> with a{" "}
          <strong className="text-white">5-agent RAG system</strong> to classify coins
          against 9,716 Corpus Nummorum types, validate material forensically, and
          generate a professional PDF report — typically in 15–60 seconds.
        </motion.p>

        {/* CTAs */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.45 }}
          className="flex flex-wrap items-center justify-center gap-4 mb-12"
        >
          <Link
            href={analyseHref}
            className="inline-flex items-center gap-2 px-8 py-3.5 rounded-xl font-bold text-sm transition-all duration-200 hover:scale-105 hover:brightness-110 active:scale-100"
            style={{ backgroundColor: "var(--brand-gold)", color: "#0a1628" }}
          >
            Analyse your coin
            <ArrowRight size={16} />
          </Link>
          <Link
            href="#how-it-works"
            className="inline-flex items-center gap-2 px-8 py-3.5 rounded-xl font-bold text-sm border transition-all duration-200 hover:scale-105 hover:bg-[var(--surface-2)]"
            style={{
              borderColor:     "var(--border)",
              color:           "var(--text-secondary)",
              backgroundColor: "var(--surface-1)",
            }}
          >
            How it works
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
          {BADGES.map(({ icon: Icon, label, color }, i) => (
            <motion.span
              key={label}
              initial={{ opacity: 0, scale: 0.82 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.7 + i * 0.1 }}
              className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium border"
              style={{
                borderColor:     `${color}40`,
                backgroundColor: `${color}12`,
                color,
              }}
            >
              <Icon size={11} />
              {label}
            </motion.span>
          ))}
        </motion.div>
      </div>

      {/* Scroll indicator */}
      <motion.div
        className="absolute bottom-8 left-1/2 -translate-x-1/2"
        animate={{ y: [0, 9, 0] }}
        transition={{ duration: 2.2, repeat: Infinity }}
        aria-hidden
      >
        <ChevronDown size={20} style={{ color: "var(--text-muted)" }} />
      </motion.div>
    </section>
  );
}
