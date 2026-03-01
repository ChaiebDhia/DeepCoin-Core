"use client";

/**
 * components/ui/health-dot.tsx
 * ============================
 * Live backend health indicator.
 * Polls GET /api/health every 30 seconds and shows a coloured dot:
 *   green  = all 5 components healthy
 *   amber  = status "degraded" (some components unhealthy)
 *   red    = no response (backend down)
 *
 * WHY polling every 30 s:
 *   The backend takes 6–12 s to start (model loading). During development,
 *   the user needs visual feedback that the API is ready without manually
 *   refreshing. 30 s is frequent enough to be useful, slow enough to not
 *   spam the health endpoint with traffic.
 */

import { useQuery }         from "@tanstack/react-query";
import { cn }               from "@/lib/utils";
import { getHealth }        from "@/lib/api";

export function HealthDot() {
  const { data, isError } = useQuery({
    queryKey:        ["health"],
    queryFn:         getHealth,
    refetchInterval: 30_000,
    retry:           false,
  });

  const isHealthy = data?.status === "healthy" || data?.status === "ok";

  const dotColor = isError      ? "bg-red-500" :
                   isHealthy   ? "bg-green-500" :
                   data?.status === "degraded" ? "bg-amber-400" :
                   "bg-slate-600";   // no data yet — neutral

  const label =  isError      ? "Backend unreachable" :
                 isHealthy   ? `API v${data?.version} — all systems OK` :
                 data?.status === "degraded" ? `API degraded — check components` :
                 "Connecting…";

  return (
    <div
      className="flex items-center gap-2 text-xs text-[var(--text-muted)] cursor-default"
      title={label}
      aria-label={label}
    >
      <span
        className={cn(
          "h-2 w-2 rounded-full shrink-0",
          dotColor,
          isHealthy && "animate-pulse",
        )}
      />
      <span className="hidden sm:block">{label}</span>
    </div>
  );
}
