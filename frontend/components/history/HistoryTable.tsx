"use client";

/**
 * components/history/HistoryTable.tsx
 * ====================================
 * Sortable, paginated history table component.
 * Receives HistorySummary[] from the parent page (server + client fetch).
 *
 * WHY pagination in the component, not just the API:
 *   The parent page fetches one "window" from the API (skip/limit).
 *   The component shows that window and emits onPageChange callbacks.
 *   This separates concerns: the component only knows about display,
 *   the page decides when to re-fetch.
 */

import Link                                 from "next/link";
import { ChevronLeft, ChevronRight, FileText } from "lucide-react";

import type { HistorySummary }              from "@/types/api";
import {
  formatConfidence, formatDate, routeStyle, confidenceBg,
}                                           from "@/lib/utils";
import { Badge, routeBadgeVariant }         from "@/components/ui/badge";
import { Button }                           from "@/components/ui/button";

interface HistoryTableProps {
  items:          HistorySummary[];
  total:          number;
  skip:           number;
  limit:          number;
  onPageChange:   (newSkip: number) => void;
  isLoading?:     boolean;
}

export function HistoryTable({
  items, total, skip, limit, onPageChange, isLoading = false,
}: HistoryTableProps) {
  const currentPage = Math.floor(skip / limit) + 1;
  const totalPages  = Math.ceil(total / limit);

  return (
    <div className="flex flex-col gap-4">
      {/* Table */}
      <div className="rounded-xl border border-[var(--border)] overflow-hidden">
        {/* Header */}
        <div className="grid grid-cols-[1fr_120px_100px_100px_80px] gap-3 px-4 py-2.5 bg-[var(--surface-2)] text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)]">
          <span>Image / Type</span>
          <span>Route</span>
          <span>Confidence</span>
          <span>Date</span>
          <span className="text-right">Report</span>
        </div>

        {/* Rows */}
        {isLoading ? (
          /* Skeleton rows */
          Array.from({ length: 5 }).map((_, i) => (
            <div
              key={i}
              className="grid grid-cols-[1fr_120px_100px_100px_80px] gap-3 px-4 py-3 border-t border-[var(--border)] animate-pulse"
            >
              <div className="h-4 w-40 rounded bg-[var(--surface-3)]" />
              <div className="h-5 w-20 rounded-full bg-[var(--surface-3)]" />
              <div className="h-4 w-16 rounded bg-[var(--surface-3)]" />
              <div className="h-4 w-28 rounded bg-[var(--surface-3)]" />
              <div className="h-4 w-8 rounded bg-[var(--surface-3)] ml-auto" />
            </div>
          ))
        ) : items.length === 0 ? (
          <div className="px-4 py-12 text-center text-sm text-[var(--text-muted)]">
            No analyses yet. Upload a coin to get started.
          </div>
        ) : (
          items.map((row) => (
            <Link
              key={row.id}
              href={`/history/${row.id}`}
              className="grid grid-cols-[1fr_120px_100px_100px_80px] gap-3 items-center px-4 py-3 border-t border-[var(--border)] bg-[var(--surface-1)] hover:bg-[var(--surface-2)] transition-colors"
            >
              {/* File + type */}
              <div className="flex flex-col min-w-0">
                <span className="text-sm font-medium text-[var(--text-primary)] truncate">
                  CN {row.label}
                </span>
                <span className="text-xs text-[var(--text-muted)] truncate">
                  {row.image_filename}
                </span>
              </div>

              {/* Route badge */}
              <Badge variant={routeBadgeVariant(row.route_taken)} className="w-fit">
                {routeStyle(row.route_taken).label}
              </Badge>

              {/* Confidence pill */}
              <div className="flex items-center gap-2">
                <div className="h-1.5 w-16 rounded-full bg-[var(--surface-3)] overflow-hidden">
                  <div
                    className={`h-full rounded-full ${confidenceBg(row.confidence)}`}
                    style={{ width: `${row.confidence * 100}%` }}
                  />
                </div>
                <span className="text-xs font-mono text-[var(--text-secondary)]">
                  {formatConfidence(row.confidence)}
                </span>
              </div>

              {/* Timestamp */}
              <span className="text-xs text-[var(--text-muted)]">
                {formatDate(row.timestamp)}
              </span>

              {/* PDF link */}
              <div className="text-right">
                {row.pdf_url ? (
                  <button
                    onClick={(e) => {
                      e.preventDefault();
                      window.open(row.pdf_url!, "_blank");
                    }}
                    className="text-[var(--text-muted)] hover:text-blue-400 transition-colors"
                    aria-label="Download PDF"
                  >
                    <FileText size={16} />
                  </button>
                ) : (
                  <span className="text-[var(--surface-3)]">
                    <FileText size={16} />
                  </span>
                )}
              </div>
            </Link>
          ))
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between text-sm text-[var(--text-muted)]">
          <span>
            {skip + 1}–{Math.min(skip + limit, total)} of {total} records
          </span>
          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              size="sm"
              disabled={currentPage === 1}
              onClick={() => onPageChange(Math.max(0, skip - limit))}
            >
              <ChevronLeft size={14} />
              Previous
            </Button>
            <span className="px-2 font-mono">
              {currentPage} / {totalPages}
            </span>
            <Button
              variant="secondary"
              size="sm"
              disabled={currentPage === totalPages}
              onClick={() => onPageChange(skip + limit)}
            >
              Next
              <ChevronRight size={14} />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
