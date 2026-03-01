"use client";

/**
 * app/history/[id]/error.tsx — History detail error boundary
 * ============================================================
 * Catches 404 (record not found) and other errors on the detail page.
 */

import { useEffect }  from "react";
import Link           from "next/link";
import { ArrowLeft, RefreshCw } from "lucide-react";
import { Button }     from "@/components/ui/button";

export default function HistoryDetailError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("[DeepCoin history detail error]", error);
  }, [error]);

  return (
    <div className="flex flex-col gap-4 max-w-md">
      <Button variant="ghost" size="sm" asChild className="w-fit">
        <Link href="/history">
          <ArrowLeft size={14} />
          Back to history
        </Link>
      </Button>

      <div
        className="rounded-xl border px-6 py-8 text-center"
        style={{ borderColor: "rgba(239,68,68,0.25)", background: "rgba(239,68,68,0.06)" }}
      >
        <p className="text-sm font-semibold text-red-300 mb-1">Record not found</p>
        <p className="text-xs mb-4" style={{ color: "var(--text-muted)" }}>
          {error.message ?? "This analysis record may have been deleted or the ID is invalid."}
        </p>
        <div className="flex items-center justify-center gap-3">
          <Button variant="primary" size="sm" onClick={reset}>
            <RefreshCw size={13} />
            Retry
          </Button>
          <Button variant="secondary" size="sm" asChild>
            <Link href="/history">Browse history</Link>
          </Button>
        </div>
      </div>
    </div>
  );
}
