"use client";

/**
 * app/explore/page.tsx — Explore CN Coin Types
 * ===============================================
 * WHAT: A public gallery of all analyses ever run on DeepCoin.
 *       Browsable by anyone, no login required.
 *
 * WHY PUBLIC:
 *   This is the "window display" of the platform. Showing real analyses
 *   (with routes, confidence levels, and coin labels) builds credibility
 *   for new visitors before they sign up.
 *
 * FEATURES:
 *   - Infinite-scroll pagination (page size 12)
 *   - Filter by route (historian / validator / investigator)
 *   - Search by coin label / CN type ID
 *   - Each card links to the full analysis (/history/:id)
 *   - Corpus Nummorum external link for each coin type
 *
 * WHY "use client":
 *   Uses useState for filter state, useQuery for data fetching,
 *   useDebounce for search input debouncing.
 */

import { useState, useMemo }       from "react";
import { useQuery }                from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import Link                        from "next/link";
import { ExternalLink, Search, FlaskConical, BookOpen, ShieldCheck, Loader2 } from "lucide-react";
import { getHistory }              from "@/lib/api";
import type { HistorySummary }     from "@/types/api";

/* ── types ────────────────────────────────────────────────────────────────── */

type RouteFilter = "all" | "historian" | "validator" | "investigator";

/* ── helpers ──────────────────────────────────────────────────────────────── */

const ROUTE_CONFIG: Record<string, { color: string; bg: string; icon: typeof BookOpen; label: string }> = {
  historian: {
    color: "#3b82f6", bg: "#3b82f620",
    icon: BookOpen, label: "Historian",
  },
  validator: {
    color: "#f59e0b", bg: "#f59e0b20",
    icon: ShieldCheck, label: "Validator",
  },
  investigator: {
    color: "#8b5cf6", bg: "#8b5cf620",
    icon: FlaskConical, label: "Investigator",
  },
};

function RoutePill({ route }: { route: string }) {
  const cfg = ROUTE_CONFIG[route];
  if (!cfg) return null;
  const Icon = cfg.icon;
  return (
    <span
      className="inline-flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded-full"
      style={{ backgroundColor: cfg.bg, color: cfg.color }}
    >
      <Icon size={9} /> {cfg.label}
    </span>
  );
}

function ConfidenceBadge({ conf }: { conf: number | null | undefined }) {
  if (conf == null) return null;
  const pct   = Math.round(conf * 100);
  const color = pct >= 70 ? "#22c55e" : pct >= 40 ? "#f59e0b" : "#8b5cf6";
  return (
    <span className="text-xs font-black tabular-nums" style={{ color }}>
      {pct}%
    </span>
  );
}

/* ── main component ───────────────────────────────────────────────────────── */

export default function ExplorePage() {
  const [routeFilter, setRouteFilter] = useState<RouteFilter>("all");
  const [search,      setSearch]      = useState("");
  const [page,        setPage]        = useState(1);
  const PAGE_SIZE = 12;

  const { data, isLoading, isError } = useQuery({
    queryKey: ["explore", page, PAGE_SIZE],
    queryFn:  () => getHistory((page - 1) * PAGE_SIZE, PAGE_SIZE),
    staleTime: 60_000,
  });

  /** Client-side filter on top of the current page */
  const filtered = useMemo(() => {
    let items: HistorySummary[] = data?.items ?? [];
    if (routeFilter !== "all") {
      items = items.filter(i => i.route_taken === routeFilter);
    }
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      items = items.filter(i =>
        (i.label ?? "").toLowerCase().includes(q) ||
        (i.id ?? "").toLowerCase().includes(q),
      );
    }
    return items;
  }, [data, routeFilter, search]);

  const totalPages = data ? Math.ceil(data.total / PAGE_SIZE) : 1;

  return (
    <div className="py-10 max-w-5xl space-y-8">

      {/* Header */}
      <div className="space-y-2">
        <p className="text-xs font-black uppercase tracking-widest" style={{ color: "var(--brand-gold)" }}>
          Community Gallery
        </p>
        <h1 className="text-3xl font-black" style={{ color: "var(--text-primary)" }}>
          Explore Analyses
        </h1>
        <p className="text-sm max-w-xl" style={{ color: "var(--text-secondary)" }}>
          Browse all coin analyses run on DeepCoin. Each card shows the AI route taken, confidence
          level, and links to the full professional report.
        </p>
      </div>

      {/* Filters bar */}
      <div className="flex flex-col sm:flex-row gap-3">
        {/* Search */}
        <div className="relative flex-1 max-w-sm">
          <Search
            size={14}
            className="absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none"
            style={{ color: "var(--text-muted)" }}
          />
          <input
            value={search}
            onChange={e => { setSearch(e.target.value); setPage(1); }}
            placeholder="Search by coin type or label…"
            className="w-full pl-8 pr-4 py-2 text-sm rounded-xl outline-none transition-colors"
            style={{
              backgroundColor: "var(--surface-1)",
              border:          "1px solid var(--border)",
              color:           "var(--text-primary)",
            }}
          />
        </div>

        {/* Route pills */}
        <div className="flex flex-wrap gap-2">
          {(["all", "historian", "validator", "investigator"] as RouteFilter[]).map(r => {
            const cfg   = ROUTE_CONFIG[r];
            const label = r === "all" ? "All routes" : cfg?.label ?? r;
            const active = routeFilter === r;
            return (
              <button
                key={r}
                onClick={() => { setRouteFilter(r); setPage(1); }}
                className="px-3 py-1.5 text-xs font-medium rounded-lg transition-colors"
                style={{
                  backgroundColor: active ? (cfg?.bg ?? "var(--brand-gold)") : "var(--surface-1)",
                  color:           active ? (cfg?.color ?? "#0d1520")         : "var(--text-secondary)",
                  border:          "1px solid var(--border)",
                }}
              >
                {label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Summary line */}
      {data && (
        <p className="text-xs" style={{ color: "var(--text-muted)" }}>
          {data.total} total analyses · showing page {page} of {totalPages}
        </p>
      )}

      {/* Grid */}
      {isLoading && (
        <div className="flex items-center justify-center py-20">
          <Loader2 size={28} className="animate-spin" style={{ color: "var(--brand-gold)" }} />
        </div>
      )}

      {isError && (
        <div className="text-center py-14">
          <p className="text-sm" style={{ color: "var(--text-muted)" }}>Could not load analyses. Try refreshing.</p>
        </div>
      )}

      <AnimatePresence mode="wait">
        {!isLoading && !isError && (
          <motion.div
            key={`${page}-${routeFilter}-${search}`}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{    opacity: 0 }}
            className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5"
          >
            {filtered.map((item, i) => {
              const cnId   = item.label?.match(/\d+/)?.[0];
              const cnHref = cnId ? `https://www.corpus-nummorum.eu/types/${cnId}` : null;

              return (
                <motion.div
                  key={item.id}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.03 }}
                  className="rounded-2xl border overflow-hidden"
                  style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}
                >
                  {/* Card header stripe */}
                  <div
                    className="h-1 w-full"
                    style={{
                      backgroundColor:
                        item.route_taken === "historian"    ? "#3b82f6" :
                        item.route_taken === "validator"    ? "#f59e0b" :
                        item.route_taken === "investigator" ? "#8b5cf6" : "var(--border)",
                    }}
                  />
                  <div className="p-5 space-y-3">
                    {/* Label + CN link */}
                    <div className="flex items-start justify-between gap-2">
                      <p
                        className="text-sm font-bold leading-snug truncate"
                        style={{ color: "var(--text-primary)" }}
                        title={item.label ?? item.id}
                      >
                        {item.label ?? "Unclassified"}
                      </p>
                      {cnHref && (
                        <a
                          href={cnHref}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="shrink-0 text-[10px] flex items-center gap-0.5 hover:underline"
                          style={{ color: "var(--text-muted)" }}
                        >
                          CN <ExternalLink size={9} />
                        </a>
                      )}
                    </div>

                    {/* Route + confidence */}
                    <div className="flex items-center justify-between">
                      {item.route_taken && <RoutePill route={item.route_taken} />}
                      <ConfidenceBadge conf={item.confidence} />
                    </div>

                    {/* View link */}
                    <Link
                      href={`/history/${item.id}`}
                      className="block w-full text-center text-xs font-semibold py-2 rounded-lg transition-colors hover:opacity-80"
                      style={{
                        backgroundColor: "var(--surface-2)",
                        color:           "var(--text-secondary)",
                        border:          "1px solid var(--border)",
                      }}
                    >
                      View full analysis →
                    </Link>
                  </div>
                </motion.div>
              );
            })}

            {filtered.length === 0 && (
              <div className="col-span-full text-center py-14">
                <p className="text-sm" style={{ color: "var(--text-muted)" }}>
                  No analyses match the current filter.
                </p>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex justify-center gap-2">
          <button
            disabled={page === 1}
            onClick={() => setPage(p => p - 1)}
            className="px-4 py-2 text-xs rounded-lg transition-opacity disabled:opacity-30"
            style={{ backgroundColor: "var(--surface-1)", color: "var(--text-secondary)", border: "1px solid var(--border)" }}
          >
            ← Previous
          </button>
          <span className="flex items-center px-4 text-xs" style={{ color: "var(--text-muted)" }}>
            {page} / {totalPages}
          </span>
          <button
            disabled={page === totalPages}
            onClick={() => setPage(p => p + 1)}
            className="px-4 py-2 text-xs rounded-lg transition-opacity disabled:opacity-30"
            style={{ backgroundColor: "var(--surface-1)", color: "var(--text-secondary)", border: "1px solid var(--border)" }}
          >
            Next →
          </button>
        </div>
      )}

      {/* CTA for new visitors */}
      <div
        className="rounded-2xl border p-8 text-center space-y-4"
        style={{ borderColor: "rgba(212,168,83,0.3)", backgroundColor: "rgba(212,168,83,0.04)" }}
      >
        <h2 className="font-bold text-base" style={{ color: "var(--text-primary)" }}>
          Have a coin to identify?
        </h2>
        <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
          Upload a photo and get a full historical report — denomination, mint, period, and provenance.
        </p>
        <Link
          href="/login?callbackUrl=/analyse"
          className="inline-block px-7 py-2.5 rounded-xl font-bold text-sm transition-opacity hover:opacity-80"
          style={{ backgroundColor: "var(--brand-gold)", color: "#0d1520" }}
        >
          Start analysing →
        </Link>
      </div>
    </div>
  );
}
