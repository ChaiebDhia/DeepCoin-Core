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
import { Building2, GraduationCap, Search } from "lucide-react";

interface AudienceCard {
  icon:    React.ElementType;
  title:   string;
  desc:    string;
  tags:    string[];
  color:   string;
  initial: { x?: number; y?: number };
}

const CARDS: AudienceCard[] = [
  {
    icon:    Building2,
    title:   "Museums & Curators",
    desc:    "Professional cataloguing in seconds. Attach the full-PDF report directly to collection management systems. Material validation prevents misattribution.",
    tags:    ["Cataloguing", "PDF reports", "Material check", "Batch analysis"],
    color:   "#3b82f6",
    initial: { x: -20 },
  },
  {
    icon:    GraduationCap,
    title:   "Researchers & Academics",
    desc:    "Query 9,716 Corpus Nummorum types through hybrid BM25 + vector search. Full citation trail — every fact is linked to a [CONTEXT N] source block.",
    tags:    ["Corpus Nummorum", "RAG citations", "9,716 types", "Batch export"],
    color:   "#8b5cf6",
    initial: { y: 20 },
  },
  {
    icon:    Search,
    title:   "Collectors & Dealers",
    desc:    "Upload a photo from your phone. Get an instant estimate: denomination, mint, period, and metal — even for degraded specimens no algorithm has seen before.",
    tags:    ["Mobile photo", "Instant estimate", "Unknown coins", "Graceful fallback"],
    color:   "#10b981",
    initial: { x: 20 },
  },
];

export function ForWhoCards() {
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
          Audience
        </span>
        <h2 className="text-3xl sm:text-4xl font-black mb-4" style={{ color: "var(--text-primary)" }}>
          Who uses DeepCoin?
        </h2>
        <p className="max-w-xl mx-auto text-sm" style={{ color: "var(--text-secondary)" }}>
          Designed for everyone who works with ancient coins — from institutional conservators
          to weekend collectors.
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
