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
import { useState, useEffect }                       from "react";
import { motion }                                    from "framer-motion";
import CountUp                                       from "react-countup";
import { Download, Clock, Cpu, BookOpen, Shield, Search } from "lucide-react";

import type { ClassifyResponse, Top5Item }           from "@/types/api";
import {
  formatConfidence, formatDate, confidenceBg,
  confidenceText, routeStyle, truncate,
}                                                    from "@/lib/utils";
import { pdfDownloadUrl }                            from "@/lib/api";
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

function CnnSection({ cnn }: { cnn: ClassifyResponse["cnn"] }) {
  /**
   * Two visual modes:
   * - IDENTIFIED  (conf ≥ 0.70):  "CN 1015" + green CountUp % + top-5 bars
   * - UNIDENTIFIED (conf < 0.70): "Nearest Candidate: CN 1015" + purple pill
   *                               + explanation + top-5 still shown
   */
  const [barWidths, setBarWidths] = useState<number[]>(cnn.top5.map(() => 0));
  useEffect(() => {
    const t = setTimeout(() => setBarWidths(cnn.top5.map(i => i.confidence * 100)), 120);
    return () => clearTimeout(t);
  }, [cnn.top5]);

  const identified = cnn.confidence >= DISPLAY_CONF_THRESHOLD;

  return (
    <Section icon={<Cpu size={16} />} title="CNN Classification" variant="cnn" delay={0}>
      <div className="flex flex-col gap-3">

        {identified ? (
          /* ── IDENTIFIED: show type + animated confidence % ─────────────── */
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
        ) : (
          /* ── UNIDENTIFIED: friendly framing, no scary % ─────────────────── */
          <>
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-xs text-[var(--text-muted)] mb-0.5">Nearest Candidate</p>
                <p className="text-2xl font-bold text-[var(--text-primary)] font-mono">
                  CN {cnn.label}
                </p>
                <p className="text-xs text-[var(--text-muted)] mt-0.5">
                  similarity score: {(cnn.confidence * 100).toFixed(1)}%
                </p>
              </div>
              <span className="shrink-0 mt-1 text-xs font-semibold px-3 py-1 rounded-full"
                style={{ background: "rgba(139,92,246,0.18)", color: "#c4b5fd" }}>
                Not identified
              </span>
            </div>
            <div className="rounded-lg px-3 py-2.5 text-xs leading-relaxed"
              style={{ background: "rgba(139,92,246,0.08)", border: "1px solid rgba(139,92,246,0.20)" }}>
              <p className="font-semibold text-purple-300 mb-1">Why no confident match?</p>
              <p className="text-[var(--text-secondary)]">
                The visual classifier was trained on&nbsp;
                <span className="text-purple-200 font-medium">438 of the 9,716 Corpus Nummorum types</span>
                &nbsp;— only those types with 10 or more reference photographs.
                This coin may belong to one of the other 9,278 types for which no training data exists.
                The knowledge-base agents below have cross-referenced all&nbsp;
                <span className="text-purple-200 font-medium">9,541 scholarly records</span>&nbsp;
                to find the closest match.
              </p>
            </div>
          </>
        )}
        {/* Main result */}
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs text-[var(--text-muted)]">Corpus Nummorum Type</p>
            <p className="text-2xl font-bold text-[var(--text-primary)] font-mono">
              CN {cnn.label}
            </p>
          </div>
          <div className="text-right">
            <p className="text-xs text-[var(--text-muted)] mb-1">Confidence</p>
            <span className={`text-3xl font-black tabular-nums ${confidenceText(cnn.confidence)}`}>
              <CountUp
                end={cnn.confidence * 100}
                decimals={1}
                suffix="%"
                duration={1.1}
                delay={0.15}
              />
            </span>
          </div>
        </div>

        {/* Meta row */}
        <div className="flex items-center gap-3 text-xs text-[var(--text-muted)]">
          <span>Inference: {cnn.inference_time_ms} ms</span>
          <span>·</span>
          <span>
            TTA:{" "}
            <span className={cnn.tta_used ? "text-green-400" : "text-[var(--text-muted)]"}>
              {cnn.tta_used ? "Yes (5 passes)" : "No"}
            </span>
          </span>
        </div>

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
                <span className="font-mono text-[var(--text-primary)] flex-1">CN {item.label}</span>
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
        </div>
      </div>
    </Section>
  );
}

// ── Historian Section ─────────────────────────────────────────────────────────

function HistorianSection({ result }: { result: ClassifyResponse }) {
  const isLowConf  = result.cnn.confidence < DISPLAY_CONF_THRESHOLD;
  const sectionTitle = isLowConf ? "Best Scholarly Match" : "Historical Analysis";
  return (
    <Section icon={<BookOpen size={16} />} title={sectionTitle} variant="historian" delay={0.1}>
      <div className="flex flex-col gap-1">
        <DataRow label="CN Type"      value={result.cnn.label ? `CN ${result.cnn.label}` : undefined} />
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
  // Even in validator route, if confidence is below display threshold, frame
  // the section as "Best Match" so the user understands this is a suggestion.
  const isLowConf   = result.cnn.confidence < DISPLAY_CONF_THRESHOLD;
  const statusColor =
    result.material_status === "consistent" ? "text-green-400" :
    result.material_status === "mismatch"   ? "text-red-400"   :
    "text-amber-400";

  return (
    <Section icon={<Shield size={16} />}
      title={isLowConf ? "Best Match · Forensic Check" : "Forensic Validation"}
      variant="validator" delay={0.1}>
      <div className="flex flex-col gap-1">
        <DataRow label="CN Type"      value={result.cnn.label ? `CN ${result.cnn.label}` : undefined} />
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
        {/* Context banner — honest framing of why this route was taken */}
        <div className="rounded-lg px-3 py-2.5 text-xs leading-relaxed"
          style={{ background: "rgba(139,92,246,0.10)", border: "1px solid rgba(139,92,246,0.25)" }}>
          <p className="font-semibold text-purple-300 mb-1">🔍 Visual Investigation Route</p>
          <p className="text-[var(--text-secondary)]">
            CNN confidence is below the classification threshold — this coin may belong to one of the
            9,278 CN types not included in the training set. The investigator agent has analysed the
            visual attributes and cross-referenced all <span className="text-purple-300 font-medium">9,541 types</span> in
            the Corpus Nummorum knowledge base to find the closest scholarly match.
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

  return (
    <div className="flex flex-col gap-4 w-full">
      {/* ── Header: route + type + conf + time ── */}
      <div className="flex flex-wrap items-center gap-3 px-1">
        <span className={`text-xs font-bold px-3 py-1 rounded-full ${routeColor}`}>
          {routeLabel}
        </span>
        {result.cnn.confidence >= DISPLAY_CONF_THRESHOLD ? (
          <Badge variant={confBadgeVariant(result.cnn.confidence)}>
            CN {result.cnn.label} · {formatConfidence(result.cnn.confidence)}
          </Badge>
        ) : (
          <Badge variant="muted">
            Best Match · CN {result.cnn.label}
          </Badge>
        )}
        <span className="text-xs text-[var(--text-muted)] flex items-center gap-1 ml-auto">
          <Clock size={11} />
          {result.processing_time_s.toFixed(1)} s · {formatDate(result.timestamp)}
        </span>
      </div>

      {/* ── CNN card ── */}
      <CnnSection cnn={result.cnn} />

      {/* ── Agent-specific card ── */}
      {result.route_taken === "historian"    && <HistorianSection    result={result} />}
      {result.route_taken === "validator"    && <ValidatorSection    result={result} />}
      {result.route_taken === "investigator" && <InvestigatorSection result={result} />}

      {/* ── Footer: PDF download + history link ── */}
      <div className="flex items-center gap-3 flex-wrap">
        {result.pdf_url && (
          <Button variant="gold" size="md" asChild>
            <a href={pdfDownloadUrl(result.pdf_url)} download target="_blank" rel="noopener noreferrer">
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
      </div>
    </div>
  );
}
