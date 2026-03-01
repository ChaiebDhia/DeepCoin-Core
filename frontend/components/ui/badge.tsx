/**
 * components/ui/badge.tsx
 * =======================
 * Small pill / tag component used for route labels (Historian / Validator /
 * Investigator) and confidence level indicators.
 */

import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold transition-colors whitespace-nowrap",
  {
    variants: {
      variant: {
        historian:    "bg-green-700/80  text-green-100",
        validator:    "bg-amber-600/80  text-amber-100",
        investigator: "bg-purple-700/80 text-purple-100",
        high:         "bg-green-600     text-white",
        medium:       "bg-amber-500     text-white",
        low:          "bg-red-500       text-white",
        outline:      "border border-[var(--border)] text-[var(--text-secondary)]",
        muted:        "bg-[var(--surface-3)] text-[var(--text-muted)]",
        gold:         "bg-[var(--brand-gold)] text-[var(--brand-navy)]",
      },
    },
    defaultVariants: { variant: "outline" },
  },
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

export function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <span className={cn(badgeVariants({ variant }), className)} {...props} />
  );
}

/**
 * Derive the correct badge variant from a route name.
 * Used by HistoryTable rows and AnalysisPanel header.
 */
export function routeBadgeVariant(
  route: string,
): VariantProps<typeof badgeVariants>["variant"] {
  switch (route) {
    case "historian":    return "historian";
    case "validator":    return "validator";
    case "investigator": return "investigator";
    default:             return "muted";
  }
}

/**
 * Derive the correct badge variant from a confidence number (0–1).
 * Mirrors the PDF's coloured confidence pill.
 */
export function confBadgeVariant(
  conf: number,
): VariantProps<typeof badgeVariants>["variant"] {
  if (conf >= 0.85) return "high";
  if (conf >= 0.40) return "medium";
  return "low";
}
