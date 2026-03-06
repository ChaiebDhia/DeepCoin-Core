"use client";

/**
 * components/coin/AnalysisPanel.tsx
 * ==================================
 * Full analysis result display for a ClassifyResponse.
 * Shows CNN result, route badge, agent output, and PDF download.
 *
 * Component structure:
 *   AnalysisPanel
 *     ├── Header bar (route badge + type ID + confidence pill + time)
 *     ├── CnnSection (top-5 table + TTA indicator)
 *     ├── HistorianSection  (visible when route = historian)
 *     ├── ValidatorSection  (visible when route = validator)
 *     ├── InvestigatorSection (visible when route = investigator)
 *     └── PdfDownloadBar
 */

import Link                                         from "next/link";
import { useState, useEffect, FormEvent }            from "react";
import { AnimatePresence, motion }                   from "framer-motion";
import CountUp                                       from "react-countup";
import { Download, Clock, Cpu, BookOpen, Shield, Search, ExternalLink, ThumbsDown, CheckCircle, X, Sparkles, Eye } from "lucide-react";

import type { ClassifyResponse, Top5Item }           from "@/types/api";
import {
  formatConfidence, formatDate, confidenceBg,
  confidenceText, routeStyle, truncate,
}                                                    from "@/lib/utils";
import { pdfDownloadUrl, submitFeedback, gradcamDisplayUrl } from "@/lib/api";
import { Badge, routeBadgeVariant, confBadgeVariant } from "@/components/ui/badge";
import { Card, CardHeader, CardTitle, CardContent }  from "@/components/ui/card";
import { Button }                                    from "@/components/ui/button";

// ── Section colour palette — each route has its own identity ──────────────────

const SECTION_COLORS = {
  cnn:           { icon: "text-blue-400",    title: "text-blue-300"    },
  historian:     { icon: "text-emerald-400", title: "text-emerald-300" },
  validator:     { icon: "text-amber-400",   title: "text-amber-300"   },
  investigator:  { icon: "text-purple-400",  title: "text-purple-300"  },
} as const;

type SectionVariant = keyof typeof SECTION_COLORS;

// ── Helper: data row ──────────────────────────────────────────────────────────

function DataRow({ label, value }: { label: string; value?: string | null }) {
  if (!value) return null;
  return (
    <div className="flex items-start gap-2 py-1.5 border-b border-[var(--border)] last:border-0">
      <span className="text-xs text-[var(--text-muted)] w-32 shrink-0 pt-0.5">{label}</span>
      <span className="text-sm text-[var(--text-primary)] break-words flex-1">{value}</span>
    </div>
  );
}

// ── Section wrapper ───────────────────────────────────────────────────────────

function Section({
  icon, title, children, variant = "cnn", delay = 0,
}: {
  icon:     React.ReactNode;
  title:    string;
  children: React.ReactNode;
  variant?: SectionVariant;
  delay?:   number;
}) {
  const colors = SECTION_COLORS[variant];
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: "easeOut", delay }}
    >
      <Card>
        <CardHeader>
          <div className={`flex items-center gap-2 ${colors.icon}`}>
            {icon}
            <CardTitle className={colors.title}>{title}</CardTitle>
          </div>
        </CardHeader>
        <CardContent>{children}</CardContent>
      </Card>
    </motion.div>
  );
}

// ── CNN Section ───────────────────────────────────────────────────────────────

// ─── Confidence threshold displayed to end-users ─────────────────────────────
// WHY 0.70: below 70% the top-1 prediction flips too often to be presented as
// a classification. We still run the KB pipeline and show it as "closest match"
// so the user always gets maximum information — just framed honestly.
const DISPLAY_CONF_THRESHOLD = 0.70;

// ─── TTA consensus threshold ─────────────────────────────────────────────────
// When 8 TTA passes are used we track what fraction independently selected the
// same top-1 class. At ≥ 87.5% (7/8+ passes agree) we promote the display to
// "TTA Consensus" even if the raw softmax is below 0.70.
//
// WHY 0.875 (not 0.75): 6/8 agreement is too weak a signal — two passes
// disagreed, and the raw softmax may still be very low (<20%) meaning the
// model genuinely found multiple plausible candidates. At 7/8+, the consensus
// is strong enough to surface as a "Consistent Match" with a review notice.
//
// IMPORTANT: TTA consistency ≠ correctness. A low softmax score alongside
// high TTA agreement means the model found a strong visual pattern, but two
// visually similar types may share that pattern. The user should always review
// the Top-5 predictions in this state.
const TTA_VOTE_THRESHOLD = 0.875;

function CnnSection({ cnn }: { cnn: ClassifyResponse["cnn"] }) {
  /**
   * THREE visual modes:
   * 1. IDENTIFIED    (conf ≥ 0.70)                   → green type + CountUp %
   * 2. TTA CONSENSUS (conf < 0.70, vote ≥ 0.875)    → teal badge + "Consistent Match" + review notice
   * 3. LOW SIGNAL    (conf < 0.70, vote < 0.875)    → purple Deep Search pill
   *
   * State 2 means the model voted consistently across augmented views, but the
   * raw softmax is below the identification threshold. This can happen because
   * two visually similar CN types share features, OR because image conditions
   * reduced the decision margin. Either way, it is NOT a confirmed identification
   * — it is a strong candidate that requires review of the Top-5.
   */
  const [barWidths, setBarWidths] = useState<number[]>(cnn.top5.map(() => 0));
  useEffect(() => {
    const t = setTimeout(() => setBarWidths(cnn.top5.map(i => i.confidence * 100)), 120);
    return () => clearTimeout(t);
  }, [cnn.top5]);

  const identified   = cnn.confidence >= DISPLAY_CONF_THRESHOLD;
  // TTA consensus: low raw confidence but the augmented views all agreed
  const ttaConsensus = !identified
    && cnn.vote_fraction != null
    && cnn.vote_fraction >= TTA_VOTE_THRESHOLD;
  // Derived display helpers
  const numPasses  = cnn.tta_passes ?? (cnn.tta_used ? 8 : 1);
  const agreeCount = cnn.vote_fraction != null
    ? Math.round(cnn.vote_fraction * numPasses)
    : null;
  const passLabel  = agreeCount != null ? `${agreeCount}/${numPasses}` : null;

  return (
    <Section icon={<Cpu size={16} />} title="CNN Classification" variant="cnn" delay={0}>
      <div className="flex flex-col gap-3">

        {identified ? (
          /* ── STATE 1 — IDENTIFIED: conf ≥ 70% ──────────────────────────── */
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-[var(--text-muted)]">Identified Type</p>
              <p className="text-2xl font-bold text-[var(--text-primary)] font-mono">
                CN {cnn.label}
              </p>
            </div>
            <div className="text-right">
              <p className="text-xs text-[var(--text-muted)] mb-1">Confidence</p>
              <span className={`text-3xl font-black tabular-nums ${confidenceText(cnn.confidence)}`}>
                <CountUp end={cnn.confidence * 100} decimals={1} suffix="%" duration={1.1} delay={0.15} />
              </span>
            </div>
          </div>

        ) : ttaConsensus ? (
          /* ── STATE 2 — TTA CONSENSUS ────────────────────────────────────────
           *  7/8+ augmented passes agreed on the same top-1 class, but the raw
           *  softmax is below the 70% identification threshold. This is a strong
           *  candidate, NOT a confirmed identification. The low % can mean two
           *  visually similar types share features. Show clearly and ask the
           *  user to review the Top-5 before treating it as definitive.
           * ─────────────────────────────────────────────────────────────────── */
          <>
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-xs text-[var(--text-muted)] mb-0.5">Consistent Match</p>
                <p className="text-2xl font-bold text-[var(--text-primary)] font-mono">
                  CN {cnn.label}
                </p>
                <p className="text-xs text-[var(--text-muted)] mt-0.5">
                  CNN confidence: {(cnn.confidence * 100).toFixed(1)}%
                </p>
              </div>
              <span
                className="shrink-0 mt-1 text-xs font-semibold px-3 py-1 rounded-full"
                style={{
                  background: "rgba(20,184,166,0.18)",
                  color:      "#5eead4",
                  border:     "1px solid rgba(20,184,166,0.30)",
                }}
              >
                TTA Consensus{passLabel ? ` (${passLabel})` : ""}
              </span>
            </div>
            <div
              className="rounded-lg px-3 py-2.5 text-xs leading-relaxed"
              style={{ background: "rgba(20,184,166,0.07)", border: "1px solid rgba(20,184,166,0.22)" }}
            >
              <p className="font-semibold text-teal-300 mb-1">What does this mean?</p>
              <p className="text-[var(--text-secondary)]">
                <span className="text-teal-200 font-medium">
                  {passLabel ? `${passLabel}` : `${numPasses}/${numPasses}`} augmented views
                </span>
                {" "}independently arrived at the same result — strong internal consistency.
                However, the{" "}
                <span className="text-teal-200 font-medium">
                  low confidence score ({(cnn.confidence * 100).toFixed(1)}%)
                </span>
                {" "}indicates this coin type shares visual features with other CN types.
                Review the{" "}
                <span className="text-teal-200 font-medium">Top-5 predictions</span>
                {" "}below before treating this as a definitive identification.
              </p>
            </div>
          </>

        ) : (
          /* ── STATE 3 — LOW VISUAL SIGNAL ──────────────────────────────────
           *  Low softmax + low TTA agreement. The system has dispatched the
           *  Investigation Agent to run a deep cross-reference across all
           *  9,541 CN types — this is MORE analysis, not less.
           *
           *  DESIGN RULE: never show the raw % to the end-user in this state.
           *  A numismatist who gets 8% with a correct result will feel the
           *  tool failed. The score reflects photo conditions, not quality of
           *  analysis. The KB result below is what matters.
           * ─────────────────────────────────────────────────────────────────── */
          <>
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-[10px] font-semibold text-purple-300/70 mb-0.5 uppercase tracking-[0.08em]">
                  Best Visual Match
                </p>
                <p className="text-2xl font-bold text-[var(--text-primary)] font-mono">
                  CN {cnn.label}
                </p>
                <p className="text-xs text-[var(--text-muted)] mt-0.5">
                  Investigation Agent dispatched · full KB search active
                </p>
              </div>
              <span
                className="shrink-0 mt-1 text-xs font-semibold px-3 py-1 rounded-full"
                style={{ background: "rgba(139,92,246,0.18)", color: "#c4b5fd", border: "1px solid rgba(139,92,246,0.25)" }}
              >
                Deep Search
              </span>
            </div>
            <div className="rounded-lg px-3 py-2.5 text-xs leading-relaxed"
              style={{ background: "rgba(139,92,246,0.08)", border: "1px solid rgba(139,92,246,0.20)" }}>
              <p className="font-semibold text-purple-300 mb-1">🔍 Investigation Pipeline Active</p>
              <p className="text-[var(--text-secondary)]">
                The initial visual scan returned a weak match — the system automatically dispatched
                the Investigation Agent to cross-reference all&nbsp;
                <span className="text-purple-200 font-medium">9,541 Corpus Nummorum types</span>
                &nbsp;and analyse the coin's visual attributes in detail.
                The result below is grounded in the full scholarly knowledge base and is
                <span className="text-purple-200 font-medium"> independent of the visual score</span>.
              </p>
            </div>
          </>
        )}

        {/* Meta row — inference time + TTA summary */}
        <div className="flex items-center gap-3 text-xs text-[var(--text-muted)]">
          <span>Inference: {cnn.inference_time_ms} ms</span>
          <span>·</span>
          <span>
            TTA:{" "}
            <span className={cnn.tta_used
              ? (ttaConsensus ? "text-teal-400" : "text-green-400")
              : "text-[var(--text-muted)]"}
            >
              {cnn.tta_used
                ? `Yes (${numPasses} passes${passLabel ? ` · ${passLabel} agree` : ""})`
                : "No"}
            </span>
          </span>
        </div>

        {/* ── Review all 5 candidates callout (States 2 ─ TTA consensus ─ and 3 ─ Deep Search) ──
         *  WHY: when the model cannot confidently identify one type, the top-5 bar chart
         *  sitting below the header badge is easy to miss.  A small callout band draws
         *  the user’s attention downward before they scroll past to the agent section.
         *  Only shown when !identified so it never appears on clear high-conf results.
         * ───────────────────────────────────────────────────────────────────── */}
        {!identified && (
          <div
            className="rounded-lg px-3 py-2 text-xs"
            style={{ background: "rgba(139,92,246,0.07)", border: "1px solid rgba(139,92,246,0.22)" }}
          >
            <span className="font-bold text-purple-300">↓ Review all 5 candidate types below</span>
            <span className="text-[var(--text-muted)]">
              {" "}— the correct coin may rank 2nd or 3rd. Each label links to the official Corpus Nummorum record.
            </span>
          </div>
        )}

        {/* Top-5 table */}
        <div>
          <p className="text-xs text-[var(--text-muted)] mb-2 uppercase tracking-wide">
            Top-5 Predictions
          </p>
          <div className="rounded-lg overflow-hidden border border-[var(--border)]">
            {cnn.top5.map((item: Top5Item, idx: number) => (
              <div
                key={item.rank}
                className={`flex items-center gap-3 px-3 py-2 text-sm border-b border-[var(--border)] last:border-0
                  ${item.rank === 1 ? "bg-blue-900/20" : "bg-[var(--surface-2)]"}`}
              >
                <span className="text-[var(--text-muted)] w-4 text-xs font-mono">{item.rank}</span>
                <a
                  href={`https://www.corpus-nummorum.eu/types/${item.label}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={(e) => e.stopPropagation()}
                  title={`Opens CN type ${item.label} on corpus-nummorum.eu (external site — opens in new tab)`}
                  className="font-mono text-blue-400 hover:text-blue-300 hover:underline underline-offset-2 transition-colors flex-1"
                >
                  CN {item.label} ↗
                </a>
                <div className="flex items-center gap-2">
                  <div className="w-24 h-1.5 rounded-full bg-[var(--surface-3)] overflow-hidden">
                    <div
                      className={`h-full rounded-full ${confidenceBg(item.confidence)}`}
                      style={{
                        width:      `${barWidths[idx] ?? 0}%`,
                        transition: "width 0.7s cubic-bezier(0.4,0,0.2,1)",
                      }}
                    />
                  </div>
                  <span className={`text-xs font-medium tabular-nums w-12 text-right ${confidenceText(item.confidence)}`}>
                    {formatConfidence(item.confidence)}
                  </span>
                </div>
              </div>
            ))}
          </div>
          {/* External site notice */}
          <p className="mt-1.5 text-[9px] flex items-center gap-1" style={{ color: "var(--text-muted)" }}>
            <ExternalLink size={9} />
            All ↗ links open <span className="font-medium">corpus&#8209;nummorum.eu</span> in a new tab. The site may take a few seconds to respond.
          </p>
        </div>

        {/* ── Grad-CAM Visual Explanation ────────────────────────────────────
         *  WHY here (after top-5, before the CN CTA):
         *    The heatmap answers the question "why did the CNN rank this type
         *    first?" — it belongs right after the confidence numbers and before
         *    we invite the user to open the scholarly record.
         *  WHY conditional: pytorch-grad-cam is optional; if not installed or
         *    the generation failed the field is null and we silently skip it.
         *  HOW: gradcamDisplayUrl() bypasses the Next.js proxy (same reason as
         *    pdfDownloadUrl) to get the PNG directly from FastAPI.
         * ─────────────────────────────────────────────────────────────────── */}
        {cnn.gradcam_url && (
          <div
            className="rounded-xl overflow-hidden"
            style={{ border: "1px solid rgba(99,102,241,0.25)", background: "rgba(99,102,241,0.04)" }}
          >
            {/* Header strip */}
            <div
              className="flex items-center gap-2 px-4 py-2 border-b"
              style={{ borderColor: "rgba(99,102,241,0.20)", background: "rgba(99,102,241,0.08)" }}
            >
              <Eye size={13} className="text-indigo-400" />
              <span className="text-[11px] font-semibold tracking-wide uppercase text-indigo-300">
                Grad-CAM — CNN Visual Explanation
              </span>
            </div>

            {/* Body: image + legend */}
            <div className="flex gap-4 p-4 items-start">
              {/* Heatmap image */}
              <div className="shrink-0 rounded-lg overflow-hidden" style={{ background: "rgba(0,0,0,0.3)" }}>
                <img
                  src={gradcamDisplayUrl(cnn.gradcam_url)}
                  alt="Grad-CAM activation heatmap"
                  width={160}
                  height={160}
                  className="block w-40 h-40 object-cover"
                  onError={(e) => {
                    // Hide the whole card if the PNG is gone (cleaned up after 30 days)
                    const card = (e.currentTarget as HTMLImageElement).closest<HTMLDivElement>(".rounded-xl");
                    if (card) card.style.display = "none";
                  }}
                />
              </div>

              {/* Legend + explanation */}
              <div className="flex flex-col gap-2 min-w-0">
                <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
                  Regions highlighted in{" "}
                  <span className="font-semibold" style={{ color: "#f87171" }}>red/yellow</span>{" "}
                  are the pixels the CNN weighted most when selecting{" "}
                  <span className="font-mono text-blue-300">CN {cnn.label}</span> as the top match.
                  Areas shown in{" "}
                  <span className="font-semibold text-blue-400">dark blue</span>{" "}
                  contributed little to the decision.
                </p>
                {/* Colour scale bar */}
                <div className="flex items-center gap-2 mt-1">
                  <span className="text-[10px] text-[var(--text-muted)]">Low</span>
                  <div
                    className="flex-1 h-2 rounded-full"
                    style={{
                      background: "linear-gradient(to right, #3b82f6, #22c55e, #eab308, #ef4444)",
                    }}
                  />
                  <span className="text-[10px] text-[var(--text-muted)]">High</span>
                </div>
                <p className="text-[10px]" style={{ color: "var(--text-muted)" }}>
                  Generated by Grad-CAM++ on EfficientNet-B3 stage-5 (19×19 feature map, 3.6×
                  finer spatial resolution than the final layer). Overlay is computed on the
                  original (pre-TTA) image.
                </p>
                {/* Confidence-aware note for OOD / low-confidence coins */}
                {cnn.confidence < 0.40 && (
                  <p
                    className="text-[10px] mt-1 px-2 py-1 rounded"
                    style={{ background: "rgba(234,179,8,0.08)", color: "#ca8a04", border: "1px solid rgba(234,179,8,0.20)" }}
                  >
                    ⚠️ Low confidence ({Math.round(cnn.confidence * 100)}%) — this coin type was not in the
                    CNN training set. The heatmap may highlight the coin’s outline and
                    background contrast rather than specific numismatic features.
                  </p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ── Corpus Nummorum CTA ────────────────────────────────────────────
         *  A full-width invitation to open the official scholarly record.
         *  WHY a separate banner: the ↗ table links are fine for power users;
         *  this card is for first-time viewers who need a clear signal that
         *  there's more depth one click away. The hover translate-x animation
         *  on the icon makes the interactivity unmistakable.
         * ─────────────────────────────────────────────────────────────────── */}
        <a
          href={`https://www.corpus-nummorum.eu/types/${cnn.label}`}
          target="_blank"
          rel="noopener noreferrer"
          className="group flex items-center justify-between rounded-xl px-4 py-3 border transition-all hover:border-blue-500/50"
          style={{
            background:  "linear-gradient(135deg, rgba(37,99,235,0.05) 0%, rgba(99,102,241,0.03) 100%)",
            borderColor: "rgba(59,130,246,0.20)",
          }}
        >
          <div className="min-w-0">
            <p className="text-xs font-semibold text-blue-300 group-hover:text-blue-200 transition-colors">
              Explore the official scholarly record
            </p>
            <p className="text-xs mt-0.5" style={{ color: "var(--text-muted)" }}>
              corpus-nummorum.eu · CN{" "}
              <span className="font-mono">{cnn.label}</span>
            </p>
          </div>
          <ExternalLink
            size={15}
            className="shrink-0 ml-3 text-blue-400 opacity-50 group-hover:opacity-100 group-hover:translate-x-0.5 transition-all"
          />
        </a>
      </div>
    </Section>
  );
}

// ── Historian Section ─────────────────────────────────────────────────────────

function HistorianSection({ result }: { result: ClassifyResponse }) {
  // TTA consensus counts as confident — show full "Historical Analysis" title
  // instead of "Best Scholarly Match". Only degrade for genuine uncertainty.
  const isLowConf = result.cnn.confidence < DISPLAY_CONF_THRESHOLD
    && !(result.cnn.vote_fraction != null && result.cnn.vote_fraction >= TTA_VOTE_THRESHOLD);
  const sectionTitle = isLowConf ? "Best Scholarly Match" : "Historical Analysis";
  return (
    <Section icon={<BookOpen size={16} />} title={sectionTitle} variant="historian" delay={0.1}>
      <div className="flex flex-col gap-1">
        {result.cnn.label && (
          <div className="flex items-start gap-2 py-1.5 border-b border-[var(--border)]">
            <span className="text-xs text-[var(--text-muted)] w-32 shrink-0 pt-0.5">CN Type</span>
            <a
              href={`https://www.corpus-nummorum.eu/types/${result.cnn.label}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-blue-400 hover:text-blue-300 hover:underline underline-offset-2 transition-colors flex-1 flex items-center gap-1"
            >
              CN {result.cnn.label}
              <ExternalLink size={11} className="opacity-60" />
            </a>
          </div>
        )}
        <DataRow label="Denomination" value={result.denomination} />
        <DataRow label="Region"       value={result.region} />
        <DataRow label="Mint"         value={result.mint} />
        <DataRow label="Date"         value={result.date_range} />
        <DataRow label="Material"     value={result.material} />
        {result.narrative && (
          <div className="mt-3 pt-3 border-t border-[var(--border)]">
            <p className="text-xs text-[var(--text-muted)] mb-2 uppercase tracking-wide">
              Expert Narrative
            </p>
            <div className="deepcoin-prose text-sm text-[var(--text-secondary)] leading-relaxed">
              {result.narrative.split("\n\n").map((para, i) => (
                <p key={i}>{para}</p>
              ))}
            </div>
          </div>
        )}
      </div>
    </Section>
  );
}

// ── Validator Section ─────────────────────────────────────────────────────────

function ValidatorSection({ result }: { result: ClassifyResponse }) {
  // TTA consensus counts as confident — show full "Forensic Validation" title.
  // Only degrade to "Best Match · Forensic Check" for genuinely uncertain predictions.
  const isLowConf = result.cnn.confidence < DISPLAY_CONF_THRESHOLD
    && !(result.cnn.vote_fraction != null && result.cnn.vote_fraction >= TTA_VOTE_THRESHOLD);
  const statusColor =
    result.material_status === "consistent" ? "text-green-400" :
    result.material_status === "mismatch"   ? "text-red-400"   :
    "text-amber-400";

  return (
    <Section icon={<Shield size={16} />}
      title={isLowConf ? "Best Match · Forensic Check" : "Forensic Validation"}
      variant="validator" delay={0.1}>
      <div className="flex flex-col gap-1">
        {result.cnn.label && (
          <div className="flex items-start gap-2 py-1.5 border-b border-[var(--border)]">
            <span className="text-xs text-[var(--text-muted)] w-32 shrink-0 pt-0.5">CN Type</span>
            <a
              href={`https://www.corpus-nummorum.eu/types/${result.cnn.label}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-blue-400 hover:text-blue-300 hover:underline underline-offset-2 transition-colors flex-1 flex items-center gap-1"
            >
              CN {result.cnn.label}
              <ExternalLink size={11} className="opacity-60" />
            </a>
          </div>
        )}
        <DataRow label="Denomination" value={result.denomination} />
        <DataRow label="Region"       value={result.region} />
        <DataRow label="Material"     value={result.material} />
        <div className="flex items-start gap-2 py-1.5 border-b border-[var(--border)]">
          <span className="text-xs text-[var(--text-muted)] w-32 shrink-0 pt-0.5">Material Check</span>
          <span className={`text-sm font-semibold capitalize ${statusColor}`}>
            {result.material_status ?? "—"}
            {result.material_confidence != null && (
              <span className="ml-2 text-xs font-normal text-[var(--text-muted)]">
                ({(result.material_confidence * 100).toFixed(0)}% confidence)
              </span>
            )}
          </span>
        </div>
        {result.narrative && (
          <div className="mt-3 pt-3 border-t border-[var(--border)]">
            <p className="text-xs text-[var(--text-muted)] mb-2 uppercase tracking-wide">
              Analysis
            </p>
            <div className="deepcoin-prose text-sm text-[var(--text-secondary)] leading-relaxed">
              {result.narrative.split("\n\n").map((para, i) => (
                <p key={i}>{para}</p>
              ))}
            </div>
          </div>
        )}
      </div>
    </Section>
  );
}

// ── Investigator Section ──────────────────────────────────────────────────────

function InvestigatorSection({ result }: { result: ClassifyResponse }) {
  return (
    <Section icon={<Search size={16} />} title="Visual Investigation" variant="investigator" delay={0.1}>
      <div className="flex flex-col gap-3">
        {/* Context banner — positive framing: system is doing MORE work, not less */}
        <div className="rounded-lg px-3 py-2.5 text-xs leading-relaxed"
          style={{ background: "rgba(139,92,246,0.10)", border: "1px solid rgba(139,92,246,0.25)" }}>
          <p className="font-semibold text-purple-300 mb-1">🔍 Deep Investigation Mode</p>
          <p className="text-[var(--text-secondary)]">
            The visual classifier returned a low signal, so the system activated its most powerful
            pipeline: the Investigation Agent has analysed the coin's visual attributes and
            cross-referenced all{" "}
            <span className="text-purple-300 font-medium">9,541 types</span> in the
            Corpus Nummorum knowledge base. The result below comes from the full numismatic
            corpus — not just the 438-type training set.
          </p>
          {/* Obverse tip */}
          <p className="mt-2 pt-2 border-t border-purple-700/30 text-[var(--text-muted)]">
            💡 <span className="text-purple-300 font-medium">Tip:</span> The classification model was
            trained on <strong className="text-purple-200">obverse views</strong> (portrait or main
            inscription side). If you have that side of the coin, re-uploading it may significantly
            improve the confidence score.
          </p>
        </div>
        {result.visual_description && (
          <div>
            <p className="text-xs text-[var(--text-muted)] mb-1 uppercase tracking-wide">
              Visual Description
            </p>
            <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
              {truncate(result.visual_description, 500)}
            </p>
          </div>
        )}
        {result.kb_match_count != null && (
          <p className="text-xs text-[var(--text-muted)]">
            Knowledge Base matches found:{" "}
            <span className="text-blue-400 font-semibold">{result.kb_match_count}</span>
            {" "}(searched {result.kb_match_count > 0 ? "9,541 CN types" : "all CN types — no match"})
          </p>
        )}
        {result.narrative && (
          <div className="pt-3 border-t border-[var(--border)]">
            <p className="text-xs text-[var(--text-muted)] mb-2 uppercase tracking-wide">
              Narrative
            </p>
            <div className="deepcoin-prose text-sm text-[var(--text-secondary)] leading-relaxed">
              {result.narrative.split("\n\n").map((para, i) => (
                <p key={i}>{para}</p>
              ))}
            </div>
          </div>
        )}
      </div>
    </Section>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

interface AnalysisPanelProps {
  result:    ClassifyResponse;
  /** If true, shows a "View full record" link. Used on the home page. */
  showLink?: boolean;
}

export function AnalysisPanel({ result, showLink = false }: AnalysisPanelProps) {
  const { label: routeLabel, color: routeColor } = routeStyle(result.route_taken);

  // ── "Mark as wrong" feedback state ───────────────────────────────────────
  const [feedbackOpen,   setFeedbackOpen]   = useState(false);
  const [feedbackType,   setFeedbackType]   = useState("");
  const [feedbackNote,   setFeedbackNote]   = useState("");
  const [feedbackStatus, setFeedbackStatus] = useState<"idle" | "submitting" | "done" | "error">("idle");

  async function handleFeedbackSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!feedbackType.trim()) return;
    setFeedbackStatus("submitting");
    try {
      await submitFeedback(result.id, feedbackType.trim(), feedbackNote.trim());
      setFeedbackStatus("done");
      setTimeout(() => { setFeedbackOpen(false); setFeedbackStatus("idle"); }, 2500);
    } catch {
      setFeedbackStatus("error");
      setTimeout(() => setFeedbackStatus("idle"), 3000);
    }
  }

  // ── AI chat CTA helpers ─────────────────────────────────────────────────
  // isLowConf — true when CNN signal is weak AND there is no TTA consensus.
  //             Drives the colour/copy variant of the always-visible AI CTA.
  // top5Labels — comma-separated list of all 5 CNN-predicted type IDs, passed
  //             to /chat?top5= so the backend fetches context for ALL candidates.
  // chatHref   — final pre-built URL; computed here once to keep JSX clean.
  const isLowConf = result.cnn.confidence < DISPLAY_CONF_THRESHOLD
    && !(result.cnn.vote_fraction != null && result.cnn.vote_fraction >= TTA_VOTE_THRESHOLD);
  const top5Labels = result.cnn.top5.map((t: Top5Item) => t.label).join(",");
  const chatQ = isLowConf
    ? `Identify ancient coin — top candidates: ${result.cnn.top5.map((t: Top5Item) => `CN ${t.label} (${(t.confidence * 100).toFixed(0)}%)`).join(", ")}`
    : `Tell me about CN ${result.cnn.label} — ${(result.cnn.confidence * 100).toFixed(1)}% confidence identification`;
  const chatHref = `/chat?q=${encodeURIComponent(chatQ)}&top5=${encodeURIComponent(top5Labels)}`;

  return (
    <div className="flex flex-col gap-4 w-full">
      {/* ── Header: route + type + conf + time ── */}
      <div className="flex flex-wrap items-center gap-3 px-1">
        <span className={`text-xs font-bold px-3 py-1 rounded-full ${routeColor}`}>
          {routeLabel}
        </span>
        {/* Wrapping the type badge in <a display:contents> makes the badge itself a CN link
             without altering its layout or visual style at all. */}
        <a
          href={`https://www.corpus-nummorum.eu/types/${result.cnn.label}`}
          target="_blank"
          rel="noopener noreferrer"
          style={{ display: "contents" }}
          title="View this coin type on Corpus Nummorum"
        >
          {result.cnn.confidence >= DISPLAY_CONF_THRESHOLD ? (
            /* High confidence — normal coloured badge */
            <Badge variant={confBadgeVariant(result.cnn.confidence)}>
              CN {result.cnn.label} · {formatConfidence(result.cnn.confidence)}
            </Badge>
          ) : (result.cnn.vote_fraction != null && result.cnn.vote_fraction >= TTA_VOTE_THRESHOLD) ? (
            /* TTA consensus — teal badge, distinguishable from purple "best match" */
            <span
              className="text-xs font-bold px-3 py-1 rounded-full"
              style={{
                background: "rgba(20,184,166,0.16)",
                color:      "#5eead4",
                border:     "1px solid rgba(20,184,166,0.30)",
              }}
            >
              CN {result.cnn.label} · TTA Consensus
            </span>
          ) : (
            /* Genuinely unknown — neutral "Deep Search" badge */
            <span
              className="text-xs font-bold px-3 py-1 rounded-full"
              style={{ background: "rgba(139,92,246,0.15)", color: "#c4b5fd", border: "1px solid rgba(139,92,246,0.25)" }}
            >
              CN {result.cnn.label} · Deep Search
            </span>
          )}
        </a>
        <span className="text-xs text-[var(--text-muted)] flex items-center gap-1 ml-auto">
          <Clock size={11} />
          {result.processing_time_s.toFixed(1)} s · {formatDate(result.timestamp)}
        </span>
      </div>

      {/* ── CNN card ── */}
      <CnnSection cnn={result.cnn} />

      {/* ── Low-confidence explainer ──────────────────────────────────────
       *  Only shown when isLowConf is true (confidence < 70 %, no TTA
       *  consensus).  Goal: calm the user — a low % is NOT a failure.
       *  WHAT it shows:
       *   • The % = rank-1 visual match among 438 CNN-trained types
       *   • A random guess would give 0.23 % → score is still N× better
       *   • The full historical analysis is grounded in 9,541 KB types
       *     independently of this visual score
       * ─────────────────────────────────────────────────────────────────── */}
      {isLowConf && (
        <motion.div
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.20 }}
          className="rounded-xl px-4 py-3.5 text-xs leading-relaxed"
          style={{ background: "rgba(99,102,241,0.07)", border: "1px solid rgba(99,102,241,0.18)" }}
        >
          <p className="font-semibold mb-1.5" style={{ color: "#a5b4fc" }}>
            💡 About this visual score
          </p>
          <p style={{ color: "var(--text-secondary)" }}>
            <strong style={{ color: "#c7d2fe" }}>
              {(result.cnn.confidence * 100).toFixed(1)}%
            </strong>{" "}
            means this is the model&rsquo;s <strong style={{ color: "#c7d2fe" }}>#1 visual match</strong>{" "}
            out of 438 trained coin types — still{" "}
            <strong style={{ color: "#c7d2fe" }}>
              {Math.round(result.cnn.confidence / 0.00228)}×
            </strong>{" "}
            better than a random guess (0.23&nbsp;%). A low margin usually reflects
            photo lighting, coin wear, or an unusual angle — not an incorrect identification.
            The historical analysis below is drawn from all 9,541 CN types and is independent
            of this visual score.
          </p>
        </motion.div>
      )}

      {/* ── Continue Research in AI Chat ─────────────────────────────────────
       *  Shown for ALL analysis results — not just low confidence ones.
       *  HIGH confidence: blue card — invite deeper historical exploration.
       *  LOW confidence:  purple card — invite candidate comparison.
       *  chatHref and isLowConf are computed before the return statement
       *  and pre-load all 5 CNN candidates as ?top5= URL context.
       * ─────────────────────────────────────────────────────────────────── */}
      <motion.div initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.35 }}>
        <Link
          href={chatHref}
          className={`group flex items-center justify-between rounded-xl px-4 py-3.5 border transition-all ${
            isLowConf ? "hover:border-purple-500/50" : "hover:border-blue-500/40"
          }`}
          style={isLowConf
            ? { background: "linear-gradient(135deg, rgba(139,92,246,0.10) 0%, rgba(99,102,241,0.05) 100%)", borderColor: "rgba(139,92,246,0.25)" }
            : { background: "linear-gradient(135deg, rgba(37,99,235,0.08) 0%, rgba(99,102,241,0.04) 100%)", borderColor: "rgba(59,130,246,0.22)" }
          }
        >
          <div>
            <p className={`text-xs font-bold transition-colors ${
              isLowConf ? "text-purple-300 group-hover:text-purple-200" : "text-blue-300 group-hover:text-blue-200"
            }`}>
              {isLowConf
                ? "🔍 Ask DeepCoin AI — explore all 5 candidates"
                : `✨ Ask DeepCoin AI about CN ${result.cnn.label}`
              }
            </p>
            <p className="text-[11px] mt-0.5" style={{ color: "var(--text-muted)" }}>
              {isLowConf
                ? `All ${result.cnn.top5.length} candidate types loaded as context — ask to compare them`
                : "Deep-dive into history, iconography & numismatic significance"
              }
            </p>
          </div>
          <Sparkles size={15} className={`shrink-0 ml-3 opacity-60 group-hover:opacity-100 transition-all ${
            isLowConf ? "text-purple-400" : "text-blue-400"
          }`} />
        </Link>
      </motion.div>

      {/* ── Agent-specific card ── */}
      {result.route_taken === "historian"    && <HistorianSection    result={result} />}
      {result.route_taken === "validator"    && <ValidatorSection    result={result} />}
      {result.route_taken === "investigator" && <InvestigatorSection result={result} />}

      {/* ── Footer: PDF download + history link + mark-as-wrong ───────────── */}
      <div className="flex items-center gap-3 flex-wrap">
        {result.pdf_url && (
          <Button variant="gold" size="md" asChild>
            {/* FIX: target="_blank" conflicts with download attribute — browser opens new tab
                 instead of saving. Remove target to trigger proper file download.
                 rel="noreferrer" kept to suppress referrer header for the file URL. */}
            <a href={pdfDownloadUrl(result.pdf_url)} download rel="noreferrer">
              <Download size={15} />
              Download PDF Report
            </a>
          </Button>
        )}
        {showLink && (
          <Link href={`/history/${result.id}`}>
            <Button variant="secondary" size="md">
              View Full Record
            </Button>
          </Link>
        )}

        {/* Mark-as-wrong button — only shown when a real record id exists */}
        {result.id && (
          <button
            onClick={() => { setFeedbackOpen(v => !v); setFeedbackStatus("idle"); }}
            className="ml-auto flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg transition-all hover:brightness-125 cursor-pointer"
            style={{
              color:      feedbackOpen ? "#f87171" : "#fca5a5",
              background: feedbackOpen ? "rgba(239,68,68,0.14)" : "rgba(239,68,68,0.07)",
              border:     `1px solid ${feedbackOpen ? "rgba(239,68,68,0.45)" : "rgba(239,68,68,0.22)"}`,
            }}
            title="Report a misclassification — help improve the model"
          >
            <ThumbsDown size={12} />
            Mark as wrong
          </button>
        )}
      </div>

      {/* ── Inline feedback form (slide-down) ───────────────────────────── */}
      <AnimatePresence>
        {feedbackOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.22 }}
            style={{ overflow: "hidden" }}
          >
            <div
              className="rounded-xl p-4 flex flex-col gap-3"
              style={{
                background: "rgba(239,68,68,0.05)",
                border:     "1px solid rgba(239,68,68,0.20)",
              }}
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-red-400">Report misclassification</span>
                <button
                  onClick={() => setFeedbackOpen(false)}
                  className="text-[var(--text-muted)] hover:text-red-400 transition-colors"
                >
                  <X size={13} />
                </button>
              </div>

              {feedbackStatus === "done" ? (
                <div className="flex items-center gap-2 text-emerald-400 text-sm py-2">
                  <CheckCircle size={15} />
                  Correction saved — thank you!
                </div>
              ) : (
                <form onSubmit={handleFeedbackSubmit} className="flex flex-col gap-3">
                  <div className="flex flex-col gap-1">
                    <label className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">
                      Correct CN Type ID <span className="text-red-400">*</span>
                    </label>
                    <input
                      type="text"
                      value={feedbackType}
                      onChange={e => setFeedbackType(e.target.value)}
                      placeholder="e.g.&nbsp;1015"
                      required
                      className="rounded-lg px-3 py-1.5 text-sm outline-none"
                      style={{
                        background: "rgba(255,255,255,0.05)",
                        border:     "1px solid rgba(255,255,255,0.12)",
                        color:      "var(--text-primary)",
                      }}
                    />
                  </div>

                  <div className="flex flex-col gap-1">
                    <label className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">
                      Note (optional)
                    </label>
                    <textarea
                      value={feedbackNote}
                      onChange={e => setFeedbackNote(e.target.value)}
                      placeholder="Why do you think it\'s wrong? Any visual details..."
                      rows={2}
                      className="rounded-lg px-3 py-1.5 text-sm outline-none resize-none"
                      style={{
                        background: "rgba(255,255,255,0.05)",
                        border:     "1px solid rgba(255,255,255,0.12)",
                        color:      "var(--text-primary)",
                      }}
                    />
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      type="submit"
                      disabled={feedbackStatus === "submitting" || !feedbackType.trim()}
                      className="flex-1 text-xs py-1.5 rounded-lg font-bold transition-all"
                      style={{
                        background: feedbackStatus === "submitting" ? "rgba(239,68,68,0.20)" : "rgba(239,68,68,0.70)",
                        color:      "#fff",
                        opacity:    feedbackStatus === "submitting" || !feedbackType.trim() ? 0.5 : 1,
                      }}
                    >
                      {feedbackStatus === "submitting" ? "Saving…" : feedbackStatus === "error" ? "Try again" : "Submit correction"}
                    </button>
                    <button
                      type="button"
                      onClick={() => setFeedbackOpen(false)}
                      className="text-xs py-1.5 px-3 rounded-lg transition-all"
                      style={{
                        background: "rgba(255,255,255,0.05)",
                        color:      "var(--text-muted)",
                        border:     "1px solid rgba(255,255,255,0.08)",
                      }}
                    >
                      Cancel
                    </button>
                  </div>
                </form>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
