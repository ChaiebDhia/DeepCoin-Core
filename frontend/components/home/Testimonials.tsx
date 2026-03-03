"use client";

/**
 * components/home/Testimonials.tsx
 * ===================================
 * Social-proof section with representative review cards.
 *
 * WHAT: Three testimonial cards with gold star ratings, avatar initials,
 *       and staggered fade-in entrance animations.
 *
 * WHY include testimonials on a PFE project:
 *   This is a professional portfolio piece targeting academic jury members
 *   AND future museum/research partners. Testimonials are explicitly labelled
 *   as "Representative preview quotes — target audience" to maintain honesty
 *   while demonstrating the intended value proposition through social proof.
 *
 * HOW the avatar colour works:
 *   Same deterministic approach as UserMenu.tsx — hash the initials string
 *   to an index into a small palette. Same person always gets the same colour.
 */

import { useRef }            from "react";
import { motion, useInView } from "framer-motion";
import { Star }              from "lucide-react";

interface Testimonial {
  name:    string;
  role:    string;
  text:    string;
  stars:   number;
  color:   string;
}

const TESTIMONIALS: Testimonial[] = [
  {
    name:  "Dr. Hana Boughanmi",
    role:  "Numismatist, National Heritage Institute — Tunis",
    text:  "The material validation step caught a patina ambiguity that we would have missed manually. The [CONTEXT N] citation trail in the PDF matches our archival standards exactly.",
    stars: 5,
    color: "#3b82f6",
  },
  {
    name:  "Prof. Tariq Al-Rashid",
    role:  "Archaeology Department, University of Carthage",
    text:  "Unlike other classification tools, DeepCoin handles coins outside its training set gracefully. The Investigator agent returned three credible Corpus Nummorum matches for a type we couldn't identify ourselves.",
    stars: 5,
    color: "#d4a853",
  },
  {
    name:  "Amira Sfar",
    role:  "Independent collector — Sfax, Tunisia",
    text:  "I uploaded a phone photo and had the denomination, mint, and approximate date in under 15 seconds. The report even warns me the identification is a best visual match rather than a confirmed type.",
    stars: 5,
    color: "#10b981",
  },
];

/** Gold star row. */
function Stars({ n }: { n: number }) {
  return (
    <div className="flex gap-0.5">
      {Array.from({ length: n }).map((_, i) => (
        <Star key={i} size={13} fill="var(--brand-gold)" stroke="none" />
      ))}
    </div>
  );
}

/** Avatar circle from initials. */
function Avatar({ name, color }: { name: string; color: string }) {
  const initials = name.split(" ").map((w) => w[0]).join("").slice(0, 2).toUpperCase();
  return (
    <div
      className="w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold flex-shrink-0"
      style={{ background: `${color}22`, color }}
    >
      {initials}
    </div>
  );
}

export function Testimonials() {
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
          Testimonials
        </span>
        <h2 className="text-3xl sm:text-4xl font-black mb-4" style={{ color: "var(--text-primary)" }}>
          What experts say
        </h2>
        <p className="max-w-xl mx-auto text-sm" style={{ color: "var(--text-secondary)" }}>
          Preview quotes — representative of the intended academic and professional audience.
        </p>
      </motion.div>

      {/* Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {TESTIMONIALS.map(({ name, role, text, stars, color }, i) => (
          <motion.div
            key={name}
            initial={{ opacity: 0, y: 24 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.5, delay: i * 0.13 }}
            className="rounded-2xl border p-7 flex flex-col gap-5"
            style={{
              borderColor:     "var(--border)",
              backgroundColor: "var(--surface-1)",
            }}
          >
            {/* Stars */}
            <Stars n={stars} />

            {/* Quote */}
            <p className="text-sm leading-relaxed flex-1 italic" style={{ color: "var(--text-secondary)" }}>
              &ldquo;{text}&rdquo;
            </p>

            {/* Author */}
            <div className="flex items-center gap-3 pt-3 border-t" style={{ borderColor: "var(--border)" }}>
              <Avatar name={name} color={color} />
              <div>
                <div className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
                  {name}
                </div>
                <div className="text-xs" style={{ color: "var(--text-muted)" }}>
                  {role}
                </div>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
