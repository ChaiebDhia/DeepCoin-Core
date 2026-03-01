"use client";

/**
 * app/error.tsx — Root error boundary
 * =====================================
 * Next.js App Router catches unhandled errors in any server or client
 * component and renders this instead of a blank white page.
 *
 * WHY "use client": error.tsx MUST be a client component — Next.js
 * requires it because the `reset` function triggers a re-render cycle.
 *
 * Covers: any page that doesn't have its own error.tsx closer in the tree.
 */

import { useEffect }  from "react";
import { AlertCircle, RefreshCw, Home } from "lucide-react";
import Link           from "next/link";
import { Button }     from "@/components/ui/button";

export default function RootError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // Log to console so it appears in the Next.js server log
    console.error("[DeepCoin root error]", error);
  }, [error]);

  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] gap-6 px-4 text-center">
      <div
        className="rounded-full p-5"
        style={{ background: "rgba(239,68,68,0.12)", border: "1px solid rgba(239,68,68,0.25)" }}
      >
        <AlertCircle size={40} className="text-red-400" />
      </div>

      <div>
        <h1 className="text-xl font-bold mb-2" style={{ color: "var(--text-primary)" }}>
          Something went wrong
        </h1>
        <p className="text-sm max-w-md" style={{ color: "var(--text-secondary)" }}>
          {error.message ?? "An unexpected error occurred."}
        </p>
        {error.digest && (
          <p className="text-xs mt-2 font-mono" style={{ color: "var(--text-muted)" }}>
            Error digest: {error.digest}
          </p>
        )}
      </div>

      <div className="flex items-center gap-3">
        <Button variant="primary" onClick={reset}>
          <RefreshCw size={15} />
          Try again
        </Button>
        <Button variant="secondary" asChild>
          <Link href="/">
            <Home size={15} />
            Back to home
          </Link>
        </Button>
      </div>
    </div>
  );
}
