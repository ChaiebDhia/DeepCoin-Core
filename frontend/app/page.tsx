"use client";

/**
 * app/page.tsx — Home / Classify page
 * =====================================
 * Layout:
 *   Idle:     hero tagline + CoinUploader centred
 *   Done:     CoinUploader left | AnalysisPanel right (2-col on lg)
 *
 * WHY "use client":
 *   CoinUploader and AnalysisPanel both read from Zustand (client state).
 */

import { motion, AnimatePresence }  from "framer-motion";
import { Coins }                    from "lucide-react";
import { CoinUploader }             from "@/components/coin/CoinUploader";
import { AnalysisPanel }            from "@/components/coin/AnalysisPanel";
import { AgentPipeline }            from "@/components/coin/AgentPipeline";
import { useDeepCoinStore }          from "@/lib/store";

export default function HomePage() {
  const { phase, result } = useDeepCoinStore();

  const hasResult = phase === "done" && result != null;
  const isProcessing = phase === "processing";

  return (
    <div className="flex flex-col gap-8">
      {/* Hero — only when idle */}
      <AnimatePresence>
        {phase === "idle" && (
          <motion.div
            key="hero"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12, scale: 0.98 }}
            transition={{ duration: 0.4, ease: "easeOut" }}
            className="text-center pt-8 pb-2"
          >
            <div className="inline-flex items-center gap-2 mb-4">
              <Coins size={32} style={{ color: "var(--brand-gold)" }} />
              <h1 className="text-3xl font-black tracking-tight" style={{ color: "var(--text-primary)" }}>
                Deep<span style={{ color: "var(--brand-gold)" }}>Coin</span>
              </h1>
            </div>
            <p className="text-sm max-w-lg mx-auto leading-relaxed" style={{ color: "var(--text-secondary)" }}>
              Upload a photograph of an ancient coin. DeepCoin&apos;s EfficientNet-B3 CNN classifies it
              against 9,716 Corpus Nummorum types, then routes it through specialist AI agents
              for historical analysis and forensic validation.
            </p>
            <div className="flex items-center justify-center gap-2 mt-5 text-xs flex-wrap">
              {[
                { label: "CNN Classification", color: "bg-blue-900/50 text-blue-300" },
                { label: "→",                  color: "text-slate-600 bg-transparent px-1" },
                { label: "RAG Knowledge Base", color: "bg-purple-900/50 text-purple-300" },
                { label: "→",                  color: "text-slate-600 bg-transparent px-1" },
                { label: "LLM Narrative",      color: "bg-green-900/50 text-green-300" },
                { label: "→",                  color: "text-slate-600 bg-transparent px-1" },
                { label: "PDF Report",         color: "bg-amber-900/50 text-amber-300" },
              ].map(({ label, color }, i) => (
                <span key={i} className={`px-2.5 py-1 rounded-full font-medium ${color}`}>{label}</span>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main grid */}
      <div className={hasResult
        ? "grid grid-cols-1 lg:grid-cols-2 gap-8 items-start"
        : "flex justify-center"
      }>
        {/* Uploader + AgentPipeline */}
        <div className="flex flex-col gap-4">
          <CoinUploader />
          <AnimatePresence mode="wait">
            {isProcessing && <AgentPipeline key="pipeline" />}
          </AnimatePresence>
        </div>

        {/* Result panel — slides in from right */}
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

      {/* Stats strip */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 rounded-xl border border-[var(--border)] p-5 bg-[var(--surface-1)]">
        {[
          { value: "80.03%", label: "TTA Accuracy",   sub: "EfficientNet-B3" },
          { value: "9,716",  label: "CN Types in KB", sub: "Full Corpus Nummorum" },
          { value: "47,705", label: "Vector Chunks",  sub: "ChromaDB RAG" },
          { value: "3",      label: "Agent Routes",   sub: "Historian / Validator / Investigator" },
        ].map(({ value, label, sub }) => (
          <div key={label} className="text-center">
            <p className="text-xl font-black tabular-nums" style={{ color: "var(--brand-gold)" }}>{value}</p>
            <p className="text-xs font-semibold mt-0.5" style={{ color: "var(--text-primary)" }}>{label}</p>
            <p className="text-xs mt-0.5" style={{ color: "var(--text-muted)" }}>{sub}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
