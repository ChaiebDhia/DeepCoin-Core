"use client";

/**
 * components/home/StatsBar.tsx
 * ==============================
 * Animated metric counters bar.
 *
 * WHAT: Five key project metrics that count up when scrolled into view via
 *       Framer Motion's `useMotionValue` + `animate()`. No extra packages needed.
 *
 * WHY animate on scroll: The numbers are impressive (80 % accuracy, 9,716 types)
 *   but static digits feel lifeless. The count-up draws the eye and anchors
 *   the value proposition with concrete evidence rather than adjectives.
 *
 * HOW the counter works:
 *   `useMotionValue(0)` holds the current display value.
 *   When `inView` fires, we call `animate(motionValue, 0, target, { duration })`.
 *   A `useEffect` watching the motion value pipes updates to local state.
 *   This is the idiomatic Framer Motion pattern — no extra counting library needed.
 */

import { useRef, useEffect, useState } from "react";
import { motion, useMotionValue, useInView, animate } from "framer-motion";

interface Stat {
  value:   number;
  label:   string;
  suffix?: string;
  prefix?: string;
  decimal?: number;
  color:   string;
}

const STATS: Stat[] = [
  { value: 80.03, label: "TTA Accuracy",      suffix: "%",  decimal: 2, color: "#3b82f6" },
  { value: 9716,  label: "Coin Types in KB",  suffix: "",   decimal: 0, color: "#8b5cf6" },
  { value: 47705, label: "RAG Chunks",        suffix: "",   decimal: 0, color: "#d4a853" },
  { value: 20,    label: "Max Latency (sec)", suffix: "s",  decimal: 0, color: "#10b981" },
  { value: 46,    label: "Unit Tests",        suffix: "",   decimal: 0, color: "#f97316" },
];

/** Single animated counter. */
function Counter({ stat, active }: { stat: Stat; active: boolean }) {
  const mv    = useMotionValue(0);
  const [display, setDisplay] = useState("0");

  useEffect(() => {
    const unsubscribe = mv.on("change", (latest) => {
      setDisplay(
        latest.toFixed(stat.decimal ?? 0)
          .replace(/\B(?=(\d{3})+(?!\d))/g, ",")
      );
    });
    return unsubscribe;
  }, [mv, stat.decimal]);

  useEffect(() => {
    if (!active) return;
    const controls = animate(mv, stat.value, {
      duration: 1.8,
      ease:     "easeOut",
    });
    return controls.stop;
  }, [active, mv, stat.value]);

  return (
    <div className="text-center py-6 px-4">
      <div
        className="text-3xl lg:text-4xl font-black tabular-nums mb-1"
        style={{ color: stat.color }}
      >
        {stat.prefix ?? ""}{display}{stat.suffix ?? ""}
      </div>
      <div className="text-xs font-medium uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>
        {stat.label}
      </div>
    </div>
  );
}

export function StatsBar() {
  const ref    = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "-60px" });

  return (
    <div
      ref={ref}
      className="rounded-2xl border overflow-hidden"
      style={{
        borderColor:     "var(--border)",
        backgroundColor: "var(--surface-1)",
      }}
    >
      {/* Grid with single-pixel gaps */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={inView ? { opacity: 1 } : {}}
        transition={{ duration: 0.5 }}
        className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5"
        style={{ gap: "1px", backgroundColor: "var(--border)" }}
      >
        {STATS.map((s, i) => (
          <motion.div
            key={s.label}
            initial={{ opacity: 0, y: 16 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ delay: i * 0.1, duration: 0.45 }}
            style={{ backgroundColor: "var(--surface-1)" }}
          >
            <Counter stat={s} active={inView} />
          </motion.div>
        ))}
      </motion.div>
    </div>
  );
}
