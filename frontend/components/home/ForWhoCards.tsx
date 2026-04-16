"use client";

/**
 * components/home/ForWhoCards.tsx
 * =================================
 * Audience targeting section ("Who uses DeepCoin?").
 *
 * WHAT: Three audience cards — Museums/Curators, Researchers/Academics,
 *       Collectors/Dealers — each with a description, tag pills, and a
 *       directional entrance animation.
 *
 * WHY directional entrances: Card 0 slides from the left → emphasises
 *   the institutional user first. Card 1 rises from below → academic tone.
 *   Card 2 slides from the right → last impression = the commercial angle.
 *   The sequence mirrors increasing informality, matching the section's
 *   left-to-right reading order.
 *
 * HOW it fits:
 *   Placed between ValueCards and Testimonials so we first explain *what*
 *   the system does, then *who* it is for, then *what users think* of it.
 */

import { useRef }                from "react";
import { motion, useInView }     from "framer-motion";
import { useTranslations }           from "next-intl";
import { Building2, GraduationCap, Search } from "lucide-react";

interface AudienceCard {
  icon:    React.ElementType;
  title:   string;
  desc:    string;
  tags:    string[];
  color:   string;
  initial: { x?: number; y?: number };
}

export function ForWhoCards() {
  const t = useTranslations("ForWhoCards");
  const ref    = useRef<HTMLElement>(null);
  const inView = useInView(ref, { once: true, margin: "-80px" });

  const CARDS: AudienceCard[] = [
    {
      icon:    Building2,
      title:   t("c1_title"),
      desc:    t("c1_desc"),
      tags:    [t("c1_tags.0"), t("c1_tags.1"), t("c1_tags.2"), t("c1_tags.3")],
      color:   "#3b82f6",
      initial: { x: -20 },
    },
    {
      icon:    GraduationCap,
      title:   t("c2_title"),
      desc:    t("c2_desc"),
      tags:    [t("c2_tags.0"), t("c2_tags.1"), t("c2_tags.2"), t("c2_tags.3")],
      color:   "#8b5cf6",
      initial: { y: 20 },
    },
    {
      icon:    Search,
      title:   t("c3_title"),
      desc:    t("c3_desc"),
      tags:    [t("c3_tags.0"), t("c3_tags.1"), t("c3_tags.2"), t("c3_tags.3")],
      color:   "#10b981",
      initial: { x: 20 },
    },
  ];

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
            borderColor:     "var(--brand-gold-30)",
            color:           "var(--brand-gold)",
            backgroundColor: "var(--brand-gold-10)",
          }}
        >
          {t("audience")}
        </span>
        <h2 className="text-3xl sm:text-4xl font-black mb-4" style={{ color: "var(--text-primary)" }}>
          {t("title")}
        </h2>
        <p className="max-w-xl mx-auto text-sm" style={{ color: "var(--text-secondary)" }}>
          {t("desc")}
        </p>
      </motion.div>

      {/* Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {CARDS.map(({ icon: Icon, title, desc, tags, color, initial }, i) => (
          <motion.div
            key={title}
            initial={{ ...initial, opacity: 0 }}
            animate={inView ? { x: 0, y: 0, opacity: 1 } : {}}
            transition={{ duration: 0.55, delay: i * 0.13 }}
            className="rounded-2xl border p-7 flex flex-col gap-5 group cursor-default transition-all duration-300"
            style={{
              borderColor:     "var(--border)",
              backgroundColor: "var(--surface-1)",
            }}
            whileHover={{
              borderColor: `${color}55`,
              boxShadow:   `0 8px 40px ${color}12`,
              transition:  { duration: 0.18 },
            }}
          >
            {/* Icon */}
            <div
              className="w-12 h-12 rounded-xl flex items-center justify-center"
              style={{ background: `${color}18`, color }}
            >
              <Icon size={22} />
            </div>

            {/* Text */}
            <div>
              <h3 className="font-bold text-base mb-2" style={{ color: "var(--text-primary)" }}>
                {title}
              </h3>
              <p className="text-xs leading-relaxed" style={{ color: "var(--text-secondary)" }}>
                {desc}
              </p>
            </div>

            {/* Tags */}
            <div className="flex flex-wrap gap-2 mt-auto">
              {tags.map((tag) => (
                <span
                  key={tag}
                  className="text-[10px] font-medium px-2.5 py-1 rounded-full"
                  style={{ background: `${color}14`, color: `${color}bb` }}
                >
                  {tag}
                </span>
              ))}
            </div>
          </motion.div>
        ))}
      </div>
    </section>
  );
}

