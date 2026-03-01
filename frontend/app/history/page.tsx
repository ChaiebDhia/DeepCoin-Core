"use client";

/**
 * app/history/page.tsx — History list page
 * ==========================================
 * Fetches paginated history from GET /api/history and renders it
 * in a sortable HistoryTable.
 *
 * WHY "use client" for a list page:
 *   Pagination state (skip) lives in React useState — purely client-side.
 *   TanStack Query handles the fetch + caching. Server components cannot
 *   use hooks. If we needed SSR for SEO we'd use a RSC wrapper that passes
 *   initial data as props — unnecessary for an internal analysis tool.
 */

import { useState }          from "react";
import { useQuery }          from "@tanstack/react-query";
import { History }           from "lucide-react";

import { getHistory }        from "@/lib/api";
import { HistoryTable }      from "@/components/history/HistoryTable";
import { Spinner }           from "@/components/ui/spinner";

const PAGE_LIMIT = 20;

export default function HistoryPage() {
  const [skip, setSkip] = useState(0);

  const { data, isLoading, isError, error } = useQuery({
    queryKey:  ["history", skip],
    queryFn:   () => getHistory(skip, PAGE_LIMIT),
    staleTime: 30_000,
  });

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
              {data.total} records — newest first
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

      {/* Table */}
      {(data || isLoading) && (
        <HistoryTable
          items={data?.items ?? []}
          total={data?.total ?? 0}
          skip={skip}
          limit={PAGE_LIMIT}
          onPageChange={setSkip}
          isLoading={isLoading}
        />
      )}
    </div>
  );
}
