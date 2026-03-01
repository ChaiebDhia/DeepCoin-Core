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
import { Download, Clock, Cpu, BookOpen, Shield, Search } from "lucide-react";

import type { ClassifyResponse, Top5Item }          from "@/types/api";
import {
  formatConfidence, formatDate, confidenceBg,
  confidenceText, routeStyle, truncate,
}                                                   from "@/lib/utils";
import { pdfDownloadUrl }                           from "@/lib/api";
import { Badge, routeBadgeVariant, confBadgeVariant } from "@/components/ui/badge";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button }                                   from "@/components/ui/button";

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
  icon, title, children,
}: { icon: React.ReactNode; title: string; children: React.ReactNode }) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2 text-blue-400">
          {icon}
          <CardTitle className="text-blue-300">{title}</CardTitle>
        </div>
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  );
}

// ── CNN Section ───────────────────────────────────────────────────────────────

function CnnSection({ cnn }: { cnn: ClassifyResponse["cnn"] }) {
  return (
    <Section icon={<Cpu size={16} />} title="CNN Classification">
      <div className="flex flex-col gap-3">
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
            <span
              className={`text-3xl font-black tabular-nums ${confidenceText(cnn.confidence)}`}
            >
              {formatConfidence(cnn.confidence)}
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
            {cnn.top5.map((item: Top5Item) => (
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
                      style={{ width: `${item.confidence * 100}%` }}
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
  return (
    <Section icon={<BookOpen size={16} />} title="Historical Analysis">
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
  const statusColor =
    result.material_status === "consistent" ? "text-green-400" :
    result.material_status === "mismatch"   ? "text-red-400"   :
    "text-amber-400";

  return (
    <Section icon={<Shield size={16} />} title="Forensic Validation">
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
    <Section icon={<Search size={16} />} title="Visual Investigation">
      <div className="flex flex-col gap-3">
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
        <Badge variant={confBadgeVariant(result.cnn.confidence)}>
          CN {result.cnn.label} · {formatConfidence(result.cnn.confidence)}
        </Badge>
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
            <a href={pdfDownloadUrl(result.pdf_url)} download target="_blank" rel="noreferrer">
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
