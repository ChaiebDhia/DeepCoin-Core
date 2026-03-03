"use client";

/**
 * app/history/[id]/page.tsx — History detail page
 * ==================================================
 * Fetches GET /api/history/{id} and renders a rich record view:
 *   1. Page header  — CN type, denomination, filename, date
 *   2. Action bar   — external CN link + PDF download
 *   3. Quick Facts  — denomination / region / mint / period / material in a grid
 *   4. Analysis badge — which agent route handled this coin + confidence tier
 *   5. AnalysisPanel — full deep-dive (same as home page)
 *
 * WHY not SSG: History records are created at runtime; IDs are not known at
 * build time. ISR would add complexity with no benefit for a single-user tool.
 */

import { use, useState }          from "react";
import { useSession }          from "next-auth/react";
import { useQuery }           from "@tanstack/react-query";
import Link                   from "next/link";
import { ArrowLeft, ExternalLink, FileDown, MapPin, Calendar, Coins, FlaskConical, Link2, Check } from "lucide-react";

import { getHistoryItem, pdfDownloadUrl } from "@/lib/api";
import { AnalysisPanel }      from "@/components/coin/AnalysisPanel";
import { Spinner }            from "@/components/ui/spinner";
import { Button }             from "@/components/ui/button";
import { formatDate, routeStyle } from "@/lib/utils";
import type { ClassifyResponse } from "@/types/api";

// ── Quick Fact row ────────────────────────────────────────────────────────────

function QuickFact({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-start gap-2">
      <span className="mt-0.5 shrink-0 text-[var(--text-muted)]">{icon}</span>
      <div className="min-w-0">
        <p className="text-[10px] uppercase tracking-wide text-[var(--text-muted)]">{label}</p>
        <p className="text-sm text-[var(--text-primary)] font-medium truncate">{value}</p>
      </div>
    </div>
  );
}

// ── Confidence tier label ─────────────────────────────────────────────────────

function confidenceTier(conf: number, vote: number | null) {
  if (conf >= 0.70) return { label: "High confidence",   color: "text-green-400" };
  if (vote != null && vote >= 0.875) return { label: "TTA consensus", color: "text-teal-400" };
  if (conf >= 0.40) return { label: "Moderate match",    color: "text-amber-400" };
  return { label: "Low visual signal", color: "text-purple-400" };
}

// ── Record detail view ────────────────────────────────────────────────────────

function RecordDetail({ data }: { data: ClassifyResponse }) {
  const { label: routeLabel, color: routeColor } = routeStyle(data.route_taken);
  const tier = confidenceTier(data.cnn.confidence, data.cnn.vote_fraction ?? null);
  const hasQuickFacts = data.denomination || data.region || data.mint || data.date_range || data.material;
  const cnUrl = `https://www.corpus-nummorum.eu/types/${data.cnn.label}`;

  // Copy-link state — resets to false after 2 seconds
  const [copied, setCopied] = useState(false);
  function handleCopyLink() {
    navigator.clipboard.writeText(window.location.href).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }

  return (
    <>
      {/* ── Page header ─────────────────────────────────────────────── */}
      <div className="flex flex-col gap-1">
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div>
            <h1 className="text-2xl font-bold" style={{ color: "var(--text-primary)" }}>
              <a
                href={cnUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="hover:underline underline-offset-4 hover:text-blue-300 transition-colors"
              >
                CN {data.cnn.label}
              </a>
              {data.denomination && (
                <span className="text-base font-normal ml-2 opacity-60">— {data.denomination}</span>
              )}
            </h1>
            <p className="text-xs mt-0.5" style={{ color: "var(--text-muted)" }}>
              {formatDate(data.timestamp)} · {data.image_filename}
            </p>
          </div>

          {/* ── Action buttons ──────────────────────────────────────── */}
          <div className="flex items-center gap-2 flex-wrap">
            <a
              href={cnUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 rounded-lg transition-colors"
              style={{
                background: "rgba(59,130,246,0.12)",
                color:      "#93c5fd",
                border:     "1px solid rgba(59,130,246,0.25)",
              }}
            >
              <ExternalLink size={12} />
              View on Corpus Nummorum
            </a>
            {data.pdf_url && (
              <a
                href={pdfDownloadUrl(data.pdf_url)}
                download
                className="inline-flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 rounded-lg transition-colors"
                style={{
                  background: "rgba(16,185,129,0.12)",
                  color:      "#6ee7b7",
                  border:     "1px solid rgba(16,185,129,0.25)",
                }}
              >
                <FileDown size={12} />
                Download PDF Report
              </a>
            )}
            {/* Copy link — copies window.location.href to clipboard.
                 Shows a green ✓ for 2 s then resets to the original icon. */}
            <button
              onClick={handleCopyLink}
              className="inline-flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 rounded-lg transition-all"
              style={{
                background: copied ? "rgba(16,185,129,0.12)" : "var(--surface-2)",
                color:      copied ? "#6ee7b7" : "var(--text-muted)",
                border:     copied ? "1px solid rgba(16,185,129,0.25)" : "1px solid var(--border)",
              }}
              title="Copy link to this record"
            >
              {copied ? <Check size={12} /> : <Link2 size={12} />}
              {copied ? "Copied!" : "Copy link"}
            </button>
          </div>
        </div>
      </div>

      {/* ── Analysis metadata strip ──────────────────────────────────── */}
      <div
        className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-lg px-4 py-3 text-xs"
        style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}
      >
        <span className={`font-bold px-2.5 py-1 rounded-full text-xs ${routeColor}`}>
          {routeLabel}
        </span>
        <span className={tier.color}>
          {tier.label}
          {data.cnn.confidence > 0 && (
            <span className="opacity-50 ml-1">({(data.cnn.confidence * 100).toFixed(1)}%)</span>
          )}
        </span>
        {data.cnn.tta_used && (
          <span className="text-[var(--text-muted)]">
            TTA · {data.cnn.tta_passes ?? 8} passes
            {data.cnn.vote_fraction != null && (
              <span className="ml-1">
                · {Math.round(data.cnn.vote_fraction * (data.cnn.tta_passes ?? 8))}/{data.cnn.tta_passes ?? 8} agree
              </span>
            )}
          </span>
        )}
        <span className="text-[var(--text-muted)] ml-auto">
          {data.processing_time_s.toFixed(1)} s · ID&nbsp;
          <span className="font-mono opacity-60">{data.id.slice(0, 8)}…</span>
        </span>
      </div>

      {/* ── Quick Facts card ──────────────────────────────────────────── */}
      {hasQuickFacts && (
        <div
          className="rounded-xl px-5 py-4"
          style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}
        >
          <p className="text-[10px] font-semibold uppercase tracking-widest text-[var(--text-muted)] mb-4">
            Quick Facts
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-8 gap-y-4">
            {data.denomination && (
              <QuickFact icon={<Coins size={13} />}      label="Denomination" value={data.denomination} />
            )}
            {data.region && (
              <QuickFact icon={<MapPin size={13} />}     label="Region"       value={data.region} />
            )}
            {data.mint && (
              <QuickFact icon={<MapPin size={13} />}     label="Mint"         value={data.mint} />
            )}
            {data.date_range && (
              <QuickFact icon={<Calendar size={13} />}   label="Period"       value={data.date_range} />
            )}
            {data.material && (
              <QuickFact icon={<FlaskConical size={13} />} label="Material"   value={data.material} />
            )}
          </div>

          {/* CN reference link at the bottom of the card */}
          <div className="mt-4 pt-3 border-t" style={{ borderColor: "var(--border)" }}>
            <p className="text-xs text-[var(--text-muted)]">
              Scholarly record:{" "}
              <a
                href={cnUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-400 hover:text-blue-300 underline underline-offset-2 transition-colors"
              >
                corpus-nummorum.eu/types/{data.cnn.label} ↗
              </a>
            </p>
          </div>
        </div>
      )}

      {/* ── Full analysis panel (same as home page) ───────────────────── */}
      <AnalysisPanel result={data} />
    </>
  );
}

// ── Page component ────────────────────────────────────────────────────────────

interface PageProps {
  params: Promise<{ id: string }>;
}

export default function HistoryDetailPage({ params }: PageProps) {
  const { id } = use(params);
  const { status } = useSession();

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["history", id],
    queryFn:  () => getHistoryItem(id),
    staleTime: 5 * 60_000,
    // Wait until session is resolved before firing — avoids 401 on direct navigation
    // when the JWT hasn't been written to _authToken yet by SessionSync.
    enabled:  status !== "loading",
  });

  return (
    <div className="flex flex-col gap-5">
      {/* Back navigation */}
      <div>
        <Link href="/history">
          <Button variant="ghost" size="sm" className="gap-1.5 -ml-2">
            <ArrowLeft size={14} />
            Back to History
          </Button>
        </Link>
      </div>

      {isLoading && (
        <div className="flex items-center gap-3 py-12 justify-center text-sm" style={{ color: "var(--text-muted)" }}>
          <Spinner size={20} />
          Loading record…
        </div>
      )}

      {isError && (
        <div className="rounded-xl border border-red-800 bg-red-900/20 px-5 py-4">
          <p className="text-sm text-red-300">
            Record not found or failed to load:{" "}
            {error instanceof Error ? error.message : "Unknown error"}
          </p>
        </div>
      )}

      {data && <RecordDetail data={data} />}
    </div>
  );
}
