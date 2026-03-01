"use client";

/**
 * app/history/error.tsx — History list error boundary
 * =====================================================
 * Catches errors in the /history page (e.g. failed API fetch,
 * SQLite read error surfaced through FastAPI).
 */

import { useEffect }  from "react";
import { RefreshCw }  from "lucide-react";
import { Button }     from "@/components/ui/button";

export default function HistoryError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("[DeepCoin history error]", error);
  }, [error]);

  return (
    <div
      className="rounded-xl border px-6 py-8 text-center"
      style={{ borderColor: "rgba(239,68,68,0.25)", background: "rgba(239,68,68,0.06)" }}
    >
      <p className="text-sm font-semibold text-red-300 mb-1">Failed to load history</p>
      <p className="text-xs mb-4" style={{ color: "var(--text-muted)" }}>
        {error.message ?? "Could not reach the backend. Is the API running on port 8000?"}
      </p>
      <Button variant="primary" size="sm" onClick={reset}>
        <RefreshCw size={13} />
        Retry
      </Button>
    </div>
  );
}
