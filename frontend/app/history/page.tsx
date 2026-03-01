"use client";

/**
 * app/history/page.tsx — History list page
 * ==========================================
 * Paginated history of all coin analyses. Pagination is synced to the URL
 * (?page=N) so the browser back button works correctly.
 *
 * WHY Suspense wraps HistoryContent:
 *   Next.js App Router requires `useSearchParams()` to be enclosed in a
 *   Suspense boundary. Without it the entire page opts out of static
 *   prerendering and throws at build time.
 *
 * WHY URL-synced pagination:
 *   With useState(skip), pressing back after navigating to a detail page
 *   resets to page 1. Storing the page in the URL makes it part of browser
 *   history, so back/forward navigation restores the correct page.
 */

import { Suspense }                                from "react";
import { useSearchParams, useRouter }              from "next/navigation";
import { useQuery, useMutation, useQueryClient }   from "@tanstack/react-query";
import { History }                                 from "lucide-react";

import { getHistory, deleteHistoryItem }           from "@/lib/api";
import { HistoryTable }                            from "@/components/history/HistoryTable";
import { Spinner }                                 from "@/components/ui/spinner";
import { routeStyle }                              from "@/lib/utils";

const PAGE_LIMIT = 20;

// ── Main content (needs Suspense because of useSearchParams) ─────────────────

function HistoryContent() {
  const searchParams = useSearchParams();
  const router       = useRouter();
  const queryClient  = useQueryClient();

  const page = Math.max(1, Number(searchParams.get("page") ?? "1"));
  const skip = (page - 1) * PAGE_LIMIT;

  const { data, isLoading, isError, error } = useQuery({
    queryKey:  ["history", skip],
    queryFn:   () => getHistory(skip, PAGE_LIMIT),
    staleTime: 30_000,
  });

  /**
   * Delete mutation.
   * On success: invalidate ["history"] so every cached page refreshes.
   * WHY invalidateQueries instead of setQueryData:
   *   Deleting one row may shift the entire pagination. A manual cache
   *   update would need to rebuild every page window correctly. Invalidating
   *   triggers a background refetch — simpler and always correct.
   */
  const deleteMutation = useMutation({
    mutationFn: (id: string) => deleteHistoryItem(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["history"] });
    },
  });

  function handlePageChange(newSkip: number) {
    const newPage = Math.floor(newSkip / PAGE_LIMIT) + 1;
    if (newPage <= 1) {
      router.push("/history");
    } else {
      router.push(`/history?page=${newPage}`);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      {/* Page header */}
      <div className="flex items-center gap-3">
        <History size={22} style={{ color: "var(--brand-light)" }} />
        <div>
          <h1 className="text-xl font-bold" style={{ color: "var(--text-primary)" }}>
            Analysis History
          </h1>
          {data && (
            <p className="text-xs mt-0.5" style={{ color: "var(--text-muted)" }}>
              {data.total} records — newest first · page {page}
            </p>
          )}
        </div>
      </div>

      {/* Error */}
      {isError && (
        <div className="rounded-xl border border-red-800 bg-red-900/20 px-5 py-4 text-sm text-red-300">
          Failed to load history:{" "}
          {error instanceof Error ? error.message : "Unknown error"}
        </div>
      )}

      {/* Loading */}
      {isLoading && !data && (
        <div className="flex items-center gap-3 py-8 justify-center text-sm" style={{ color: "var(--text-muted)" }}>
          <Spinner size={18} />
          Loading history…
        </div>
      )}

      {/* Stats strip — quick at-a-glance numbers above the filter bar.
           Uses data.total (global count from SQL) for "total", and computes
           route breakdown + avg confidence from the current page window.
           WHY current page only: we only have the page slice in memory.
           total is still the accurate global count from SELECT COUNT(*). */}
      {data && data.total > 0 && (
        <div
          className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-xl px-4 py-2.5 text-xs border"
          style={{ background: "var(--surface-1)", borderColor: "var(--border)" }}
        >
          <span style={{ color: "var(--text-muted)" }}>
            <span className="font-semibold tabular-nums" style={{ color: "var(--text-primary)" }}>
              {data.total}
            </span>{" "}
            analyses total
          </span>
          <span style={{ color: "var(--border)" }}>·</span>

          {/* Route breakdown — counts from current page window */}
          {Object.entries(
            data.items.reduce<Record<string, number>>(
              (acc, r) => ({ ...acc, [r.route_taken]: (acc[r.route_taken] ?? 0) + 1 }),
              {}
            )
          ).sort().map(([route, count]) => (
            <span
              key={route}
              className={`px-2 py-0.5 rounded-full border text-[11px] font-medium ${routeStyle(route).color}`}
            >
              {count} {routeStyle(route).label}
            </span>
          ))}

          {/* Avg confidence from current page */}
          {data.items.length > 0 && (
            <>
              <span style={{ color: "var(--border)" }}>·</span>
              <span className="ml-auto" style={{ color: "var(--text-muted)" }}>
                page avg{" "}
                <span className="font-semibold" style={{ color: "var(--text-primary)" }}>
                  {(
                    (data.items.reduce((s, r) => s + r.confidence, 0) / data.items.length) * 100
                  ).toFixed(1)}%
                </span>
                {" "}confidence
              </span>
            </>
          )}
        </div>
      )}

      {/* Table */}
      {(data || isLoading) && (
        <HistoryTable
          items={data?.items ?? []}
          total={data?.total ?? 0}
          skip={skip}
          limit={PAGE_LIMIT}
          onPageChange={handlePageChange}
          onDelete={(id) => deleteMutation.mutate(id)}
          isLoading={isLoading}
        />
      )}
    </div>
  );
}

// ── Page entry (Suspense boundary required for useSearchParams) ──────────────

export default function HistoryPage() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center gap-3 py-12 justify-center text-sm" style={{ color: "var(--text-muted)" }}>
          <Spinner size={18} />
          Loading history…
        </div>
      }
    >
      <HistoryContent />
    </Suspense>
  );
}
