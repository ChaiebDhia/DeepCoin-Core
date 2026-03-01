"use client";

/**
 * components/history/HistoryTable.tsx
 * ====================================
 * Paginated history table with client-side filter bar and per-row delete.
 *
 * FILTER STRATEGY — client-side on the current page window:
 *   The parent fetches a window (skip/limit) from the API. Filtering is
 *   applied on that window in the browser — no extra round-trips for a
 *   single-user local app with ≤20 rows per page. If a future version
 *   needs server-side filtering (large datasets), the filter state can be
 *   lifted to URL params and passed to getHistory(skip, limit, route, q).
 *
 * DELETE BUTTON — outside the <Link> wrapper:
 *   HTML5 forbids interactive content (button) inside <a>. The row uses
 *   a flex container: <Link> takes flex-1 for the data cells, and the
 *   delete <button> sits as a sibling flex child. This is semantically
 *   correct and avoids nested-anchor warnings.
 */

import { useState, useMemo }               from "react";
import Link                                from "next/link";
import {
  ChevronLeft, ChevronRight, FileText, Trash2, Search, X,
}                                          from "lucide-react";

import type { HistorySummary }             from "@/types/api";
import {
  formatConfidence, formatDate, routeStyle, confidenceBg,
}                                          from "@/lib/utils";
import { Badge, routeBadgeVariant }        from "@/components/ui/badge";
import { Button }                          from "@/components/ui/button";

// ── filter constants ──────────────────────────────────────────────────────────

const ROUTE_OPTIONS: { value: string; label: string }[] = [
  { value: "",             label: "All"          },
  { value: "historian",    label: "Historian"    },
  { value: "validator",    label: "Validator"    },
  { value: "investigator", label: "Investigator" },
];

// ── props ─────────────────────────────────────────────────────────────────────

interface HistoryTableProps {
  items:        HistorySummary[];
  total:        number;
  skip:         number;
  limit:        number;
  onPageChange: (newSkip: number) => void;
  onDelete?:    (id: string) => void;
  isLoading?:   boolean;
}

// ── component ─────────────────────────────────────────────────────────────────

export function HistoryTable({
  items, total, skip, limit, onPageChange, onDelete, isLoading = false,
}: HistoryTableProps) {
  const currentPage = Math.floor(skip / limit) + 1;
  const totalPages  = Math.ceil(total / limit);

  // ── filter state ────────────────────────────────────────────────────────────

  const [search,      setSearch]      = useState("");
  const [routeFilter, setRouteFilter] = useState("");

  /**
   * Apply route + search filters on the current page window.
   * Both filters are combined with AND logic.
   */
  const filteredItems = useMemo(() => {
    let result = items;
    if (routeFilter) {
      result = result.filter((r) => r.route_taken === routeFilter);
    }
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      result  = result.filter(
        (r) =>
          r.label.toLowerCase().includes(q) ||
          r.image_filename.toLowerCase().includes(q),
      );
    }
    return result;
  }, [items, search, routeFilter]);

  // ── delete handler ───────────────────────────────────────────────────────────

  function handleDelete(id: string) {
    if (!onDelete) return;
    if (window.confirm("Permanently delete this record from history?")) {
      onDelete(id);
    }
  }

  // ── render ───────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col gap-4">

      {/* ─ Filter bar ─────────────────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center gap-3">

        {/* Search input */}
        <div className="relative flex-1 min-w-[200px]">
          <Search
            size={14}
            className="absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none"
            style={{ color: "var(--text-muted)" }}
          />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Filter by CN type or filename…"
            className="w-full h-9 pl-9 pr-8 rounded-lg text-sm focus:outline-none focus:ring-1"
            style={{
              background:   "var(--surface-2)",
              border:       "1px solid var(--border)",
              color:        "var(--text-primary)",
              // @ts-expect-error — CSS custom property placeholder colour via inline style not typed
              "--tw-ring-color": "var(--brand-light)",
            }}
          />
          {search && (
            <button
              onClick={() => setSearch("")}
              className="absolute right-2.5 top-1/2 -translate-y-1/2 transition-colors"
              style={{ color: "var(--text-muted)" }}
              aria-label="Clear search"
            >
              <X size={12} />
            </button>
          )}
        </div>

        {/* Route filter pills */}
        <div className="flex items-center gap-1.5 flex-wrap">
          {ROUTE_OPTIONS.map((opt) => {
            const active = routeFilter === opt.value;
            return (
              <button
                key={opt.value}
                onClick={() => setRouteFilter(opt.value)}
                className="h-9 px-3 rounded-lg text-xs font-medium border transition-colors"
                style={{
                  background:  active ? "var(--brand-light)"  : "var(--surface-2)",
                  borderColor: active ? "var(--brand-light)"  : "var(--border)",
                  color:       active ? "#ffffff"             : "var(--text-muted)",
                }}
              >
                {opt.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* ─ Table ──────────────────────────────────────────────────────────── */}
      <div className="rounded-xl border overflow-hidden" style={{ borderColor: "var(--border)" }}>

        {/* Header row */}
        <div
          className="flex items-center gap-3 px-4 py-2.5 text-xs font-semibold uppercase tracking-wider"
          style={{ background: "var(--surface-2)", color: "var(--text-muted)" }}
        >
          <span className="flex-1 min-w-0">Image / Type</span>
          <span className="w-[120px] shrink-0">Route</span>
          <span className="w-[110px] shrink-0">Confidence</span>
          <span className="w-[100px] shrink-0">Date</span>
          <span className="w-[44px] shrink-0 text-right">Report</span>
          {/* Delete column header — only shown when handler is wired */}
          {onDelete && <span className="w-10 shrink-0" />}
        </div>

        {/* Skeleton rows */}
        {isLoading && Array.from({ length: 5 }).map((_, i) => (
          <div
            key={i}
            className="flex items-center gap-3 px-4 py-3 border-t animate-pulse"
            style={{ borderColor: "var(--border)" }}
          >
            <div className="h-4 flex-1 rounded" style={{ background: "var(--surface-3)" }} />
            <div className="h-5 w-20 rounded-full shrink-0" style={{ background: "var(--surface-3)" }} />
            <div className="h-4 w-24 rounded shrink-0" style={{ background: "var(--surface-3)" }} />
            <div className="h-4 w-20 rounded shrink-0" style={{ background: "var(--surface-3)" }} />
            <div className="h-4 w-6 rounded ml-auto shrink-0" style={{ background: "var(--surface-3)" }} />
          </div>
        ))}

        {/* Empty state */}
        {!isLoading && filteredItems.length === 0 && (
          <div className="px-4 py-12 text-center text-sm" style={{ color: "var(--text-muted)" }}>
            {items.length === 0
              ? "No analyses yet. Upload a coin to get started."
              : "No records match the current filters."}
          </div>
        )}

        {/* Data rows */}
        {!isLoading && filteredItems.map((row) => (
          <div
            key={row.id}
            className="flex items-center border-t transition-colors"
            style={{
              borderColor: "var(--border)",
              background:  "var(--surface-1)",
            }}
            onMouseEnter={(e) =>
              (e.currentTarget.style.background = "var(--surface-2)")
            }
            onMouseLeave={(e) =>
              (e.currentTarget.style.background = "var(--surface-1)")
            }
          >
            {/*
              Link covers the 5 data columns — flex-1 so it fills all space
              except the delete button. Using flex instead of grid here keeps
              the <button> outside the <a>, satisfying HTML5 interactive-content
              nesting rules.
            */}
            <Link
              href={`/history/${row.id}`}
              className="flex flex-1 items-center gap-3 min-w-0 px-4 py-3"
            >
              {/* Image / Type */}
              <div className="flex-1 flex flex-col min-w-0">
                <span
                  className="text-sm font-medium truncate"
                  style={{ color: "var(--text-primary)" }}
                >
                  CN {row.label}
                </span>
                <span
                  className="text-xs truncate"
                  style={{ color: "var(--text-muted)" }}
                >
                  {row.image_filename}
                </span>
              </div>

              {/* Route badge */}
              <div className="w-[120px] shrink-0">
                <Badge variant={routeBadgeVariant(row.route_taken)} className="w-fit">
                  {routeStyle(row.route_taken).label}
                </Badge>
              </div>

              {/* Confidence bar + value */}
              <div className="w-[110px] shrink-0 flex items-center gap-2">
                <div
                  className="h-1.5 w-14 rounded-full overflow-hidden"
                  style={{ background: "var(--surface-3)" }}
                >
                  <div
                    className={`h-full rounded-full ${confidenceBg(row.confidence)}`}
                    style={{ width: `${row.confidence * 100}%` }}
                  />
                </div>
                <span
                  className="text-xs font-mono"
                  style={{ color: "var(--text-secondary)" }}
                >
                  {formatConfidence(row.confidence)}
                </span>
              </div>

              {/* Timestamp */}
              <div className="w-[100px] shrink-0">
                <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                  {formatDate(row.timestamp)}
                </span>
              </div>

              {/* PDF download */}
              <div className="w-[44px] shrink-0 text-right">
                {row.pdf_url ? (
                  <button
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      window.open(row.pdf_url!, "_blank", "noopener,noreferrer");
                    }}
                    className="transition-colors"
                    style={{ color: "var(--text-muted)" }}
                    onMouseEnter={(e) => (e.currentTarget.style.color = "#60a5fa")}
                    onMouseLeave={(e) => (e.currentTarget.style.color = "var(--text-muted)")}
                    aria-label="Download PDF report"
                  >
                    <FileText size={16} />
                  </button>
                ) : (
                  <span style={{ color: "var(--surface-3)" }}>
                    <FileText size={16} />
                  </span>
                )}
              </div>
            </Link>

            {/* Delete button — sibling of <Link>, NOT inside <a> */}
            {onDelete && (
              <div className="w-10 shrink-0 flex items-center justify-center">
                <button
                  onClick={() => handleDelete(row.id)}
                  className="p-1.5 rounded transition-colors"
                  style={{ color: "var(--text-muted)" }}
                  onMouseEnter={(e) => (e.currentTarget.style.color = "#f87171")}
                  onMouseLeave={(e) => (e.currentTarget.style.color = "var(--text-muted)")}
                  aria-label="Delete record"
                  title="Delete"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* ─ Pagination ─────────────────────────────────────────────────────── */}
      {totalPages > 1 && (
        <div
          className="flex items-center justify-between text-sm"
          style={{ color: "var(--text-muted)" }}
        >
          <span>
            {skip + 1}–{Math.min(skip + limit, total)} of {total}
            {filteredItems.length < items.length && (
              <span style={{ color: "var(--brand-light)" }} className="ml-1">
                · {filteredItems.length} shown after filter
              </span>
            )}
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
