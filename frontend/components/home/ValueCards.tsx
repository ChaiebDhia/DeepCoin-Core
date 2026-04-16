"use client";

/**
 * components/home/ValueCards.tsx
 * ================================
 * Feature / value proposition cards.
 *
 * WHAT: Three core feature cards (Forensic Validation, Grounded RAG,
 *       Graceful Degradation) plus three smaller feature chips below.
 *       The central card is highlighted as "Core innovation".
 *
 * WHY three pillars: These answer the three most likely expert objections —
 *   (1) "How do you know the metal is really silver?" → Forensic validator.
 *   (2) "Doesn't the LLM hallucinate historical facts?" → RAG grounding.
 *   (3) "What happens with coins outside your 438 classes?" → Graceful fallback.
 *
 * HOW featured card is implemented:
 *   `featured: true` on the centre card adds a gold border + glow + badge.
 *   We use a slightly wider motion initial offset on features cards (x: ±15)
 *   so they slide in from opposite sides when scrolled into view.
 */

import { useRef }                 from "react";
import { motion, useInView }      from "framer-motion";
import { useTranslations }           from "next-intl";
import { Microscope, BookMarked, ShieldCheck, Users, Lock, TrendingUp } from "lucide-react";

interface FeatureCard {
  icon:     React.ElementType;
  title:    string;
  desc:     string;
  bullets:  string[];
  color:    string;
  featured? :boolean;
}

export function ValueCards() {
  const t = useTranslations("ValueCards");
  const ref    = useRef<HTMLElement>(null);
  const inView = useInView(ref, { once: true, margin: "-80px" });

  const CARDS: FeatureCard[] = [
    {
      icon:     Microscope,
      title:    t("c1_title"),
      desc:     t("c1_desc"),
      bullets:  [
        t("c1_b1"),
        t("c1_b2"),
        t("c1_b3"),
      ],
      color:    "#3b82f6",
    },
    {
      icon:     BookMarked,
      title:    t("c2_title"),
      desc:     t("c2_desc"),
      bullets:  [
        t("c2_b1"),
        t("c2_b2"),
        t("c2_b3"),
      ],
      color:    "#d4a853",
      featured: true,
    },
    {
      icon:     ShieldCheck,
      title:    t("c3_title"),
      desc:     t("c3_desc"),
      bullets:  [
        t("c3_b1"),
        t("c3_b2"),
        t("c3_b3"),
      ],
      color:    "#10b981",
    },
  ];

  const CHIPS = [
    { icon: Users,       label: t("chip1"),             color: "#3b82f6" },
    { icon: Lock,        label: t("chip2"),        color: "#8b5cf6" },
    { icon: TrendingUp,  label: t("chip3"),  color: "#10b981" },
  ];

  return (
    <section id="features" ref={ref} className="py-24 scroll-mt-16">

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
          {t("why")}
        </span>
        <h2 className="text-3xl sm:text-4xl font-black mb-4" style={{ color: "var(--text-primary)" }}>
          {t("title")}
        </h2>
        <p className="max-w-xl mx-auto text-sm" style={{ color: "var(--text-secondary)" }}>
          {t("desc")}
        </p>
      </motion.div>

      {/* Main cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        {CARDS.map(({ icon: Icon, title, desc, bullets, color, featured }, i) => {
          const initial = i === 0 ? { x: -15, opacity: 0 } : i === 2 ? { x: 15, opacity: 0 } : { y: 20, opacity: 0 };
          return (
            <motion.div
              key={title}
              initial={initial}
              animate={inView ? { x: 0, y: 0, opacity: 1 } : {}}
              transition={{ duration: 0.55, delay: i * 0.12 }}
              className="relative rounded-2xl border p-7 flex flex-col gap-5 transition-shadow hover:shadow-lg dark:hover:shadow-none hover:shadow-indigo-500/10"
              style={{
                borderColor:     featured ? `${color}60` : "var(--border)",
                backgroundColor: featured ? `${color}06` : "var(--surface-1)",
                boxShadow:       featured ? `0 0 36px ${color}10` : undefined,
              }}
            >
              {/* Core innovation badge for featured card */}
              {featured && (
                <span
                  className="absolute -top-3 left-1/2 -translate-x-1/2 text-[10px] font-black uppercase tracking-widest px-3 py-1 rounded-full"
                  style={{ backgroundColor: color, color: "var(--brand-navy)" }}
                >
                  {t("core_innovation")}
                </span>
              )}

              {/* Icon */}
              <div
                className="w-12 h-12 rounded-xl flex items-center justify-center animate-glow-pulse"
                style={{ background: `${color}1a`, color }}
              >
                <Icon size={22} />
              </div>

              {/* Content */}
              <div>
                <h3 className="font-bold text-base mb-2" style={{ color: "var(--text-primary)" }}>{title}</h3>
                <p className="text-xs leading-relaxed mb-4" style={{ color: "var(--text-secondary)" }}>{desc}</p>
                <ul className="space-y-1.5">
                  {bullets.map((b) => (
                    <li key={b} className="flex items-start gap-2 text-xs" style={{ color: "var(--text-secondary)" }}>
                      <span className="mt-0.5 w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
                      {b}
                    </li>
                  ))}
                </ul>
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Small chips row */}
      <motion.div
        initial={{ opacity: 0, y: 14 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.5, delay: 0.4 }}
        className="flex flex-wrap justify-center gap-3"
      >
        {CHIPS.map(({ icon: Icon, label, color }) => (
          <span
            key={label}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full border text-xs font-medium"
            style={{
              borderColor:     `${color}35`,
              color:           `${color}cc`,
              backgroundColor: `${color}0d`,
            }}
          >
            <Icon size={12} />
            {label}
          </span>
        ))}
      </motion.div>
    </section>
  );
}

