"use client";

/**
 * components/home/AnalyseSection.tsx
 * ====================================
 * Client island wrapping the coin analyser on the homepage.
 *
 * WHAT: Applies section framing (id="analyse", header, subtitle) around
 *       the CoinUploader + AnalysisPanel + AgentPipeline components.
 *       Contains all Zustand state-reading so page.tsx can be a pure
 *       Server Component.
 *
 * WHY a dedicated wrapper (instead of keeping it in page.tsx):
 *   All three inner components depend on Zustand (`useDeepCoinStore`).
 *   Zustand requires "use client". If page.tsx were "use client", React
 *   would ship ALL homepage code as a client bundle — wasted for the static
 *   sections above this. Isolating this island keeps scroll-above content as
 *   zero-JS Server Components while the analyser remains fully interactive.
 *
 * HOW it fits:
 *   Placed after Testimonials and before EmailCapture in page.tsx.
 *   The hero's primary CTA points to href="#analyse" which scrolls here.
 *   scroll-mt-16 compensates for the fixed header height.
 */

import { motion, AnimatePresence }   from "framer-motion";
import { Upload }                    from "lucide-react";
import { CoinUploader }              from "@/components/coin/CoinUploader";
import { AnalysisPanel }             from "@/components/coin/AnalysisPanel";
import { AgentPipeline }             from "@/components/coin/AgentPipeline";
import { useDeepCoinStore }          from "@/lib/store";

export function AnalyseSection() {
  const { phase, result, _cancelFn } = useDeepCoinStore();

  const hasResult    = phase === "done" && result != null;
  const isProcessing = phase === "processing";

  return (
    <section
      id="analyse"
      className="py-24 scroll-mt-16"
    >
      {/* Section header */}
      <div className="text-center mb-12">
        <span
          className="inline-block text-xs font-semibold uppercase tracking-widest px-3 py-1 rounded-full border mb-4"
          style={{
            borderColor:     "var(--brand-gold-30)",
            color:           "var(--brand-gold)",
            backgroundColor: "var(--brand-gold-10)",
          }}
        >
          Try it now
        </span>
        <h2
          className="text-3xl sm:text-4xl font-black mb-4 flex items-center justify-center gap-3"
          style={{ color: "var(--text-primary)" }}
        >
          <Upload size={28} style={{ color: "var(--brand-gold)" }} />
          Analyse your coin
        </h2>
        <p className="max-w-lg mx-auto text-sm" style={{ color: "var(--text-secondary)" }}>
          Upload a photograph — the full pipeline runs in real time.
          Drag-and-drop, or click to browse. No account required.
        </p>
      </div>

      {/* AgentPipeline modal — shown during processing */}
      <AnimatePresence>
        {isProcessing && (
          <AgentPipeline key="pipeline" onCancel={_cancelFn ?? undefined} />
        )}
      </AnimatePresence>

      {/* Uploader + result grid */}
      <div
        className={
          hasResult
            ? "grid grid-cols-1 lg:grid-cols-2 gap-8 items-start"
            : "flex justify-center"
        }
      >
        <div className="flex flex-col gap-4">
          <CoinUploader />
        </div>

        <AnimatePresence>
          {hasResult && (
            <motion.div
              key="result"
              initial={{ opacity: 0, x: 30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.45, ease: "easeOut", delay: 0.1 }}
            >
              <AnalysisPanel result={result} showLink />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
}
