"use client";

/**
 * app/history/[id]/page.tsx — History detail page
 * ==================================================
 * Fetches GET /api/history/{id} and renders the full AnalysisPanel.
 * Mirrors what the home page shows immediately after analysis, but
 * accessible via a permalink for any past record.
 *
 * WHY not use generateStaticParams for SSG:
 *   History records are created at runtime (user uploads). There is no
 *   known set of IDs at build time. ISR could work but adds complexity
 *   for no benefit on a single-user analysis tool.
 */

import { use }               from "react";
import { useQuery }          from "@tanstack/react-query";
import Link                  from "next/link";
import { ArrowLeft }         from "lucide-react";

import { getHistoryItem }    from "@/lib/api";
import { AnalysisPanel }     from "@/components/coin/AnalysisPanel";
import { Spinner }           from "@/components/ui/spinner";
import { Button }            from "@/components/ui/button";
import { formatDate }        from "@/lib/utils";

interface PageProps {
  params: Promise<{ id: string }>;
}

export default function HistoryDetailPage({ params }: PageProps) {
  // Next.js 15: params is a Promise; React.use() unwraps it synchronously
  const { id } = use(params);

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["history", id],
    queryFn:  () => getHistoryItem(id),
    staleTime: 5 * 60_000,   // detail records don't change — 5 min stale
  });

  return (
    <div className="flex flex-col gap-6">
      {/* Back link */}
      <div>
        <Link href="/history">
          <Button variant="ghost" size="sm" className="gap-1.5 -ml-2">
            <ArrowLeft size={14} />
            Back to History
          </Button>
        </Link>
      </div>

      {/* Loading */}
      {isLoading && (
        <div className="flex items-center gap-3 py-12 justify-center text-sm" style={{ color: "var(--text-muted)" }}>
          <Spinner size={20} />
          Loading record…
        </div>
      )}

      {/* Error */}
      {isError && (
        <div className="rounded-xl border border-red-800 bg-red-900/20 px-5 py-4">
          <p className="text-sm text-red-300">
            Record not found or failed to load:{" "}
            {error instanceof Error ? error.message : "Unknown error"}
          </p>
        </div>
      )}

      {/* Detail */}
      {data && (
        <>
          <div className="flex flex-col gap-1">
            <h1 className="text-xl font-bold" style={{ color: "var(--text-primary)" }}>
              CN {data.cnn.label} — {data.image_filename}
            </h1>
            <p className="text-xs" style={{ color: "var(--text-muted)" }}>
              {formatDate(data.timestamp)} · ID: {data.id}
            </p>
          </div>
          <AnalysisPanel result={data} />
        </>
      )}
    </div>
  );
}
