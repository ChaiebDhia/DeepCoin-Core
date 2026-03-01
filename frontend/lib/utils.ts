/**
 * lib/utils.ts
 * ============
 * Shared utility functions used throughout the component tree.
 *
 * cn(): Merges Tailwind classes correctly.
 *   WHY clsx + tailwind-merge:
 *     clsx handles conditional class strings.
 *     tailwind-merge deduplicates conflicting Tailwind utilities
 *     (e.g. "px-2 px-4" → "px-4") so component overrides work predictably.
 */

import { clsx, type ClassValue } from "clsx";
import { twMerge }               from "tailwind-merge";

/** Merge Tailwind class names without conflicts. */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/** Format a 0–1 probability as a percentage string, e.g. "91.2%". */
export function formatConfidence(conf: number): string {
  return `${(conf * 100).toFixed(1)}%`;
}

/**
 * Format an ISO 8601 UTC timestamp for display.
 * Example: "1 Mar 2026, 00:58"
 */
export function formatDate(iso: string): string {
  return new Intl.DateTimeFormat("en-GB", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(iso));
}

/**
 * Return the Tailwind bg-* class for a confidence level.
 * Mirrors the PDF's colored confidence pill:
 *   green  ≥ 85% — high confidence, historian route
 *   amber  40–85% — medium, validator route
 *   red    < 40% — low, investigator route
 */
export function confidenceBg(conf: number): string {
  if (conf >= 0.85) return "bg-green-600";
  if (conf >= 0.40) return "bg-amber-500";
  return "bg-red-500";
}

/** Return the Tailwind text-* class for a confidence level. */
export function confidenceText(conf: number): string {
  if (conf >= 0.85) return "text-green-400";
  if (conf >= 0.40) return "text-amber-400";
  return "text-red-400";
}

/**
 * Map a route name to a display label + Tailwind colour class pair.
 * Used by route badges in the UI.
 */
export function routeStyle(route: string): { label: string; color: string } {
  switch (route) {
    case "historian":    return { label: "Historian",           color: "bg-emerald-800 text-emerald-100" };
    case "validator":    return { label: "Validator",           color: "bg-amber-700  text-amber-100"   };
    case "investigator": return { label: "Visual Investigation", color: "bg-purple-800 text-purple-100"  };
    default:             return { label: route,                  color: "bg-slate-600  text-slate-100"   };
  }
}

/** Clamp a string to a maximum length with ellipsis. */
export function truncate(str: string, maxLen: number): string {
  return str.length <= maxLen ? str : `${str.slice(0, maxLen)}…`;
}
